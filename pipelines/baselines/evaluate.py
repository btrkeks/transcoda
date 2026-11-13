"""One reproducible CPU evaluation command for pinned releases and predictions."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pipelines.baselines.binaries import resolve_binary
from pipelines.baselines.contracts import canonical_json_bytes
from pipelines.baselines.evaluation_inputs import (
    FORMATS,
    MODEL_FAMILIES,
    EvaluationRelease,
    file_provenance,
    load_evaluation_release,
    normalize_notation,
    prediction_route,
    read_jsonl,
    required_text,
    unique_rows,
)
from pipelines.baselines.io import atomic_write_bytes, sha256_file
from pipelines.baselines.renderability import DEFINITION, probe_renderability, verovio_provenance
DEFAULT_ABC2XML_SCRIPT = Path("external/legato/utils/abc2xml.py")
from pipelines.baselines.scoring import ScoreCandidate, VeroSimDriver, build_score_report
from pipelines.baselines.uncertainty import cluster_bootstrap, page_bootstrap

DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "config/evaluation-profiles/transcoda-real-ground-truth-v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--release",
        required=True,
        help="Release name, release directory, or text-only evaluation pin JSON.",
    )
    parser.add_argument("--releases-root", type=Path, default=Path("data/releases"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Override the release's local dataset location; bytes are still verified.",
    )
    parser.add_argument("--model-label")
    parser.add_argument("--model-family", choices=MODEL_FAMILIES)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New output directory outside the prediction inputs.",
    )
    parser.add_argument(
        "--prediction-format",
        choices=sorted(FORMATS),
        help="Otherwise read native_format or prediction_format per row.",
    )
    parser.add_argument(
        "--define-density-cuts",
        action="store_true",
        help="Reference-only command; write frozen cut points without scoring predictions.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="LABEL=EVALUATION_DIR",
        help="Compute paired differences against a prior evaluation; repeat per model.",
    )
    parser.add_argument(
        "--expected-density-source-digest",
        help="Require the frozen cuts to come from this source release digest.",
    )
    parser.add_argument("--verosim-bin", type=Path)
    parser.add_argument("--kern-parser", type=Path)
    parser.add_argument("--evaluation-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--abc2xml-script", type=Path, default=DEFAULT_ABC2XML_SCRIPT)
    parser.add_argument("--abc2xml-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--resamples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--density-cut-points",
        type=Path,
        help="Reuse a previously written density-cut-points.json.",
    )
    parser.add_argument(
        "--min-reference-symbols",
        type=int,
        default=2,
        help="Abort below this n_gt; must be at least 2.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_s <= 0 or not math.isfinite(args.timeout_s):
        raise ValueError("timeout must be positive and finite")
    if args.resamples < 1 or not 0 < args.confidence_level < 1:
        raise ValueError("resamples must be positive and confidence level in (0, 1)")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.min_reference_symbols < 2:
        raise ValueError("min-reference-symbols must be at least 2")
    if args.define_density_cuts:
        if args.predictions or args.model_label or args.compare or args.density_cut_points:
            raise ValueError("--define-density-cuts is a separate reference-only command")
    elif not args.density_cut_points:
        raise ValueError(
            "final evaluation requires --density-cut-points; first run --define-density-cuts"
        )
    elif not args.predictions or not args.model_label or not args.model_label.strip():
        raise ValueError("evaluation requires --predictions and a nonempty --model-label")
    predictions_path = args.predictions.resolve(strict=True) if args.predictions else None
    predictions = unique_rows(read_jsonl(predictions_path)) if predictions_path else {}
    release = load_evaluation_release(
        args.release, releases_root=args.releases_root, dataset_dir=args.dataset_dir
    )
    expected = {page.sample_id for page in release.pages}
    if not args.define_density_cuts and set(predictions) != expected:
        raise ValueError(
            f"prediction/release coverage mismatch: missing={sorted(expected - predictions.keys())[:10]}, "
            f"extra={sorted(predictions.keys() - expected)[:10]}"
        )
    formats = {
        sample_id: prediction_route(row, family=args.model_family, override=args.prediction_format)
        for sample_id, row in predictions.items()
    }
    verosim = resolve_binary(
        "verosim", explicit=args.verosim_bin, env_var="TRANSCODA_VEROSIM_BIN"
    ).path
    needs_parser = "transcoda-v2" in formats.values() or any(
        p.target_format == "transcoda-v2" for p in release.pages
    )
    kern_parser = (
        resolve_binary("kern_parser", explicit=args.kern_parser, env_var="KERN_PARSER_BIN").path
        if needs_parser or args.kern_parser is not None
        else None
    )
    if "abc" in formats.values():
        args.abc2xml_script.resolve(strict=True)
        args.abc2xml_python.resolve(strict=True)
    profile = args.evaluation_profile.resolve(strict=True)
    out = args.out.resolve()
    if predictions_path and (out == predictions_path or predictions_path.is_relative_to(out)):
        raise ValueError("output directory must not contain prediction inputs")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    driver = VeroSimDriver(
        verosim, evaluation_profile=profile, failure_score=1.0, timeout_s=args.timeout_s
    )
    normalization = {
        "kern_parser": kern_parser,
        "abc2xml_script": args.abc2xml_script,
        "abc2xml_python": args.abc2xml_python,
        "timeout_s": args.timeout_s,
    }
    references, counts = validate_references(
        release,
        driver=driver,
        out=out / "references",
        min_symbols=args.min_reference_symbols,
        normalization=normalization,
    )
    provenance = {
        "release": release.provenance,
        "predictions": file_provenance(predictions_path) if predictions_path else None,
        "profile": file_provenance(profile),
        "tools": {
            "python": file_provenance(Path(sys.executable)),
            "numpy_version": np.__version__,
            "verosim": file_provenance(verosim),
            "kern_parser": file_provenance(kern_parser) if kern_parser else None,
            "abc2xml": file_provenance(args.abc2xml_script) if "abc" in formats.values() else None,
            "abc2xml_python": file_provenance(args.abc2xml_python)
            if "abc" in formats.values()
            else None,
            "rendering_verovio": verovio_provenance(),
        },
        "postprocessing": {
            "ordered_options": [],
            "scope": "no repairs for any model",
        },
        "notation_conversion": {
            "prediction_format_override": args.prediction_format,
            "prediction_input_formats": sorted(set(formats.values())),
            "reference_input_formats": sorted({page.target_format for page in release.pages}),
            "model_family": args.model_family,
            "transcoda-kern": "body bytes preserved; exclusive interpretations from first data width; terminators from last record width; no parser gate",
            "transcoda-v2": "same lower_transcoda function, binary and arguments as references; conversion failure scores 1.0",
            "reference_transcoda-v2": "kern_parser serialize-jsonl --preset humdrum-kern --input-dialect transcoda-v2",
            "bekern": "smt_bekern_to_kern; fixed two-spine header and native token expansion",
            "abc": "abc2xml_python abc2xml_script -; output is passed directly as MusicXML",
        },
        "reference_policy": {
            "postprocessing": [],
            "min_projected_symbols": args.min_reference_symbols,
            "near_empty": "2 through 10 symbols retained and flagged; below minimum aborts the evaluation",
        },
        "scoring": {
            "failure_score": 1.0,
            "process_isolation": "one VeroSim process per prediction",
            "timeout_s": args.timeout_s,
            "mode": "evaluation profile (experimental extraction then projection)",
            "note_position": "visual",
            "typed_space_handling": "suppress-straddle-filler",
        },
        "code": {
            p.name: sha256_file(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("evaluation_inputs.py"),
                Path(__file__).with_name("normalization.py"),
                Path(__file__).with_name("scoring.py"),
                Path(__file__).with_name("uncertainty.py"),
                Path(__file__).with_name("renderability.py"),
            ]
        },
    }
    cuts = density_cut_points(counts, args.density_cut_points, provenance)
    if (
        args.expected_density_source_digest
        and cuts["source_release_digest"] != args.expected_density_source_digest
    ):
        raise ValueError("density cut points use a different source release")
    atomic_write_bytes(out / "density-cut-points.json", canonical_json_bytes(cuts))
    provenance["density_cut_points"] = {
        **file_provenance(args.density_cut_points or out / "density-cut-points.json"),
        "policy": cuts,
    }
    if args.define_density_cuts:
        atomic_write_bytes(out / "provenance.json", canonical_json_bytes(provenance))
        print(
            json.dumps(
                {"density_cut_points": str(out / "density-cut-points.json"), "pages": len(counts)}
            )
        )
        return 0
    pages = []
    audits = []
    for index, page in enumerate(release.pages):
        source = predictions[page.sample_id]
        row = {
            "sample_id": page.sample_id,
            "document_id": page.document_id,
            "n_gt": counts[page.sample_id],
            "near_empty_reference": counts[page.sample_id] <= 10,
            "density_tertile": density_label(counts[page.sample_id], cuts),
            "status": "inference_failure",
            "omr_ned": 1.0,
            "renderable": False,
            "renders_svg": False,
            "renders_svg_without_error": False,
            "has_error_diagnostic": False,
            "render_error": "inference failure",
            "error": source.get("error"),
            "termination_reason": source.get("termination_reason"),
        }
        audit = {
            "sample_id": page.sample_id,
            "source_format": formats[page.sample_id],
            "postprocessing": [],
        }
        if not source.get("error"):
            try:
                field = "output_text" if "output_text" in source else "prediction"
                prediction, format, audit["postprocessing"] = normalize_notation(
                    required_text(source, field),
                    formats[page.sample_id],
                    postprocess=(),
                    **normalization,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                row.update(
                    status="normalization_failure",
                    error=str(exc)[:4000],
                    render_error=str(exc)[:4000],
                )
            else:
                audit["normalized_format"] = format
                render = probe_renderability(prediction, format, timeout_s=args.timeout_s)
                row.update(
                    renderable=render["renders_svg"],
                    renders_svg=render["renders_svg"],
                    renders_svg_without_error=render["renders_svg_without_error"],
                    has_error_diagnostic=render["has_error_diagnostic"],
                    render_error=render.get("error"),
                )
                audit["rendering"] = render
                target, target_format = references[page.sample_id]
                candidate = ScoreCandidate(
                    page.sample_id,
                    args.model_label,
                    prediction,
                    target,
                    prediction_format=format,
                    target_format=target_format,
                )
                result, backend = score_isolated(
                    candidate, driver=driver, out=out / "scores" / f"{index:06d}"
                )
                row.update(result)
                audit["backend"] = backend
        pages.append(row)
        audits.append(audit)
    summary = summarize_pages(
        pages,
        label=args.model_label,
        seed=args.seed,
        resamples=args.resamples,
        confidence_level=args.confidence_level,
    )
    summary["provenance"] = provenance
    summary["paired_differences"] = paired_comparisons(
        pages, summary=summary, comparisons=args.compare, cuts=cuts
    )
    write_outputs(out, pages=pages, audits=audits, summary=summary)
    print(
        json.dumps(
            {
                "pages": len(pages),
                "model": args.model_label,
                "omr_ned_x100": summary["omr_ned"] * 100,
                "renderability": summary["renderability"]["fraction"],
                "summary": str(out / "summary.json"),
            },
            sort_keys=True,
        )
    )
    return 0


def validate_references(
    release: EvaluationRelease,
    *,
    driver: VeroSimDriver,
    out: Path,
    min_symbols: int,
    normalization: dict[str, Any],
) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    references = {}
    candidates = []
    for page in release.pages:
        try:
            text, format, _ = normalize_notation(
                page.target, page.target_format, postprocess=(), reference=True, **normalization
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"reference {page.sample_id} could not be lowered: {exc}") from exc
        if format == "kern" and not any("**kern" in line.split("\t") for line in text.splitlines()):
            raise ValueError(
                f"reference {page.sample_id} has no **kern exclusive interpretation (headerless target)"
            )
        references[page.sample_id] = text, format
        candidates.append(
            ScoreCandidate(
                page.sample_id,
                "reference-self-check",
                text,
                text,
                prediction_format=format,
                target_format=format,
            )
        )
    batch = driver.run(candidates, output_dir=out)
    report = build_score_report(candidates, batch=batch)
    if report.summary["scored_count"] != len(candidates) or report.summary["omr_ned"] != 0:
        raise ValueError(
            "reference self-check failed: a reference did not parse and compare identically"
        )
    counts = {}
    for record in batch.records:
        n_gt = record.get("n_gt")
        if not isinstance(n_gt, int) or n_gt < min_symbols:
            raise ValueError(
                f"reference {record['id']} parsed to almost nothing: n_gt={n_gt}, minimum={min_symbols}"
            )
        counts[record["id"]] = n_gt
    return references, counts


def score_isolated(
    candidate: ScoreCandidate, *, driver: VeroSimDriver, out: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        batch = driver.run([candidate], output_dir=out)
    except RuntimeError as exc:
        # Usage errors indicate a broken tool/protocol, not a bad prediction.
        if "exit code 2:" in str(exc) and "usage:" in str(exc):
            raise
        status = "process_timeout" if "timed out" in str(exc) else "process_failure"
        return {"status": status, "omr_ned": 1.0, "error": str(exc)[:4000]}, {}
    report = build_score_report([candidate], batch=batch)
    row = report.samples[0]
    if row["status"] == "scored":
        record = batch.records[0]
        n_pred = record.get("n_pred")
        has_header = candidate.prediction_format != "kern" or any(
            "**kern" in line.split("\t") for line in candidate.prediction.splitlines()
        )
        if not has_header or (isinstance(n_pred, int) and n_pred < 2):
            row = {
                **row,
                "status": "parse_failure",
                "omr_ned": 1.0,
                "error": f"prediction parsed to almost nothing: n_pred={n_pred}; kern_header={has_header}",
            }
    value = row["omr_ned"] if row["status"] == "scored" else 1.0
    if not isinstance(value, int | float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid bounded OMR-NED for {candidate.sample_id}: {value}")
    return {"status": row["status"], "omr_ned": value, "error": row["error"]}, report.backend


def density_cut_points(
    counts: dict[str, int], path: Path | None, provenance: dict[str, Any]
) -> dict[str, Any]:
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "transcoda.density-cut-points.v1":
            raise ValueError("unsupported density cut-point schema")
        lower, upper = payload["lower"], payload["upper"]
        if (
            not all(isinstance(x, int | float) and math.isfinite(x) for x in (lower, upper))
            or lower > upper
        ):
            raise ValueError("density cut points must be finite and ordered")
        if payload["profile_sha256"] != provenance["profile"]["sha256"]:
            raise ValueError("density cut points use a different evaluation profile")
        if payload["verosim_sha256"] != provenance["tools"]["verosim"]["sha256"]:
            raise ValueError("density cut points use a different VeroSim binary")
        if payload.get("kern_parser_sha256") != tool_hash(provenance, "kern_parser"):
            raise ValueError("density cut points use a different kern_parser binary")
        if "kern_parser_sha256" not in payload:
            raise ValueError(
                "density cut points must record kern_parser hash, including null when unused"
            )
        return payload
    lower, upper = np.quantile(list(counts.values()), [1 / 3, 2 / 3])
    return {
        "schema": "transcoda.density-cut-points.v1",
        "lower": float(lower),
        "upper": float(upper),
        "rule": "low: n_gt <= lower; medium: lower < n_gt <= upper; high: n_gt > upper",
        "quantile_method": "numpy linear",
        "source_release_digest": provenance["release"]["release_digest"],
        "source_pages_sha256": provenance["release"]["pages_sha256"],
        "profile_sha256": provenance["profile"]["sha256"],
        "verosim_sha256": provenance["tools"]["verosim"]["sha256"],
        "kern_parser_sha256": tool_hash(provenance, "kern_parser"),
    }


def tool_hash(provenance: dict[str, Any], name: str) -> str | None:
    tool = provenance["tools"].get(name)
    return tool["sha256"] if tool else None


def density_label(count: int, cuts: dict[str, Any]) -> str:
    return "low" if count <= cuts["lower"] else "medium" if count <= cuts["upper"] else "high"


def summarize_pages(
    pages: list[dict[str, Any]], *, label: str, seed: int, resamples: int, confidence_level: float
) -> dict[str, Any]:
    values = np.asarray([row["omr_ned"] for row in pages], dtype=np.float64)
    documents = [row["document_id"] for row in pages]

    def intervals(selected: list[dict[str, Any]], bootstrap_seed: int) -> dict[str, Any]:
        scores = np.asarray([[row["omr_ned"] for row in selected]], dtype=np.float64)
        ids = [row["document_id"] for row in selected]
        result: dict[str, Any] = {}
        for name, enough in [
            ("page", len(selected) >= 2),
            ("document_cluster", len(set(ids)) >= 2),
        ]:
            if not enough:
                result[name] = {
                    "available": False,
                    "reason": f"fewer than two {name} sampling units",
                }
                continue
            kwargs = {
                "seed": bootstrap_seed + (name == "document_cluster"),
                "n_resamples": resamples,
                "level": confidence_level,
            }
            bootstrap = (
                page_bootstrap(scores, **kwargs)
                if name == "page"
                else cluster_bootstrap(scores, ids, **kwargs)
            )
            result[name] = {
                "available": True,
                "point": float(bootstrap.point[0]),
                "ci_low": float(bootstrap.low[0]),
                "ci_high": float(bootstrap.high[0]),
            }
        return result

    thirds = []
    for index, third in enumerate(("low", "medium", "high")):
        selected = [row for row in pages if row["density_tertile"] == third]
        thirds.append(
            {
                "label": third,
                "count": len(selected),
                "min_n_gt": min((row["n_gt"] for row in selected), default=None),
                "max_n_gt": max((row["n_gt"] for row in selected), default=None),
                "omr_ned": float(np.mean([row["omr_ned"] for row in selected]))
                if selected
                else None,
                "intervals": intervals(selected, seed + 100 + 2 * index),
            }
        )
    return {
        "schema_version": 1,
        "metric": "VeroSim OMR-NED",
        "scale": "fraction; multiply by 100 for paper tables",
        "model_label": label,
        "cohort_count": len(pages),
        "document_count": len(set(documents)),
        "omr_ned": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "failure_score": 1.0,
        "status_counts": dict(Counter(row["status"] for row in pages)),
        "failure_count": sum(row["status"] != "scored" for row in pages),
        "near_empty_reference_count": sum(row["near_empty_reference"] for row in pages),
        "renderability": {
            "definition": DEFINITION,
            "count": sum(row["renders_svg"] for row in pages),
            "fraction": sum(row["renders_svg"] for row in pages) / len(pages),
            "without_error_count": sum(row["renders_svg_without_error"] for row in pages),
            "without_error_fraction": sum(row["renders_svg_without_error"] for row in pages)
            / len(pages),
            "error_diagnostic_count": sum(row["has_error_diagnostic"] for row in pages),
        },
        "bootstrap": {
            "method": "percentile",
            "resamples": resamples,
            "confidence_level": confidence_level,
            "page_seed": seed,
            "document_cluster_seed": seed + 1,
            "density_page_seeds": [seed + 100 + 2 * i for i in range(3)],
            "density_document_cluster_seeds": [seed + 101 + 2 * i for i in range(3)],
            "main_interval": "document_cluster",
            **intervals(pages, seed),
        },
        "density_thirds": thirds,
    }


def paired_comparisons(
    pages: list[dict[str, Any]],
    *,
    summary: dict[str, Any],
    comparisons: list[str],
    cuts: dict[str, Any],
) -> dict[str, Any]:
    """Bootstrap aligned page differences with the same seeds and frozen thirds."""
    result = {}
    for declaration in comparisons:
        if "=" not in declaration:
            raise ValueError("--compare must be LABEL=EVALUATION_DIR")
        label, directory = declaration.split("=", 1)
        if not label or label == summary["model_label"] or label in result:
            raise ValueError(
                "comparison labels must be nonempty, unique, and different from the current model"
            )
        directory = Path(directory).resolve(strict=True)
        other = json.loads((directory / "summary.json").read_text())
        if other["model_label"] != label:
            raise ValueError("comparison label differs from its stored evaluation")
        for name in ("report.samples.jsonl", "density-cut-points.json"):
            if sha256_file(directory / name) != other["artifacts"][name]["sha256"]:
                raise ValueError(f"comparison artifact changed: {name}")
        if json.loads((directory / "density-cut-points.json").read_text()) != cuts:
            raise ValueError("comparison must use the same frozen density cut points")
        if any(
            other["provenance"]["release"].get(key) != summary["provenance"]["release"].get(key)
            for key in ("release_digest", "pages_sha256")
        ):
            raise ValueError("comparison release differs")
        if other["provenance"]["postprocessing"] != summary["provenance"]["postprocessing"]:
            raise ValueError("comparison repair policy differs")
        if other["provenance"]["scoring"] != summary["provenance"]["scoring"]:
            raise ValueError("comparison scoring policy differs")
        for key in (
            "resamples",
            "confidence_level",
            "page_seed",
            "document_cluster_seed",
            "density_page_seeds",
            "density_document_cluster_seeds",
        ):
            if other["bootstrap"][key] != summary["bootstrap"][key]:
                raise ValueError(f"comparison bootstrap setting differs: {key}")
        aligned = unique_rows(read_jsonl(directory / "report.samples.jsonl"))
        if set(aligned) != {row["sample_id"] for row in pages}:
            raise ValueError("comparison page coverage differs")
        differences = []
        for row in pages:
            previous = aligned[row["sample_id"]]
            for key in ("document_id", "n_gt", "density_tertile"):
                if previous[key] != row[key]:
                    raise ValueError(f"comparison page metadata differs: {row['sample_id']} {key}")
            for item in (row, previous):
                value = item["omr_ned"]
                if (
                    not isinstance(value, int | float)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError("invalid comparison score")
                if item["status"] != "scored" and value != 1:
                    raise ValueError("comparison failures must score 1.0")
            differences.append({**row, "omr_ned": row["omr_ned"] - previous["omr_ned"]})
        delta = summarize_pages(
            differences,
            label=f"{summary['model_label']} minus {label}",
            seed=summary["bootstrap"]["page_seed"],
            resamples=summary["bootstrap"]["resamples"],
            confidence_level=summary["bootstrap"]["confidence_level"],
        )
        result[label] = {
            "comparison": delta["model_label"],
            "interpretation": "negative favors the current model; lower OMR-NED is better",
            "mean_difference": delta["omr_ned"],
            "bootstrap": delta["bootstrap"],
            "density_thirds": delta["density_thirds"],
            "per_page": [
                {
                    "sample_id": row["sample_id"],
                    "document_id": row["document_id"],
                    "density_tertile": row["density_tertile"],
                    "difference": row["omr_ned"],
                }
                for row in differences
            ],
            "source": file_provenance(directory / "summary.json"),
        }
    return result


def write_outputs(
    out: Path, *, pages: list[dict[str, Any]], audits: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(pages[0]))
    writer.writeheader()
    writer.writerows(pages)
    atomic_write_bytes(out / "pages.csv", buffer.getvalue().encode())
    atomic_write_bytes(
        out / "report.samples.jsonl", b"".join(canonical_json_bytes(row) for row in pages)
    )
    atomic_write_bytes(out / "audit.jsonl", b"".join(canonical_json_bytes(row) for row in audits))
    summary["artifacts"] = {
        name: file_provenance(out / name)
        for name in ("pages.csv", "report.samples.jsonl", "audit.jsonl", "density-cut-points.json")
    }
    atomic_write_bytes(out / "summary.json", canonical_json_bytes(summary))
    lines = [
        f"# {summary['model_label']} evaluation",
        "",
        f"{summary['cohort_count']} pages from {summary['document_count']} documents. OMR-NED × 100; lower is better.",
        "",
        f"Mean: **{100 * summary['omr_ned']:.2f}**. Median: {100 * summary['median']:.2f}. Failures: {summary['failure_count']} (score 100).",
        "",
    ]
    level = 100 * summary["bootstrap"]["confidence_level"]
    for name in ("document_cluster", "page"):
        interval = summary["bootstrap"][name]
        text = (
            f"[{100 * interval['ci_low']:.2f}, {100 * interval['ci_high']:.2f}]"
            if interval["available"]
            else f"unavailable ({interval['reason']})"
        )
        lines.append(f"{name.replace('_', ' ').capitalize()} {level:g}% CI: {text}.")
    lines.extend(
        [
            "",
            f"Renderable: {summary['renderability']['count']}/{summary['cohort_count']} ({100 * summary['renderability']['fraction']:.2f}%).",
            f"Renderable without error diagnostics: {summary['renderability']['without_error_count']}/{summary['cohort_count']}. Error diagnostics: {summary['renderability']['error_diagnostic_count']}.",
            "",
            DEFINITION,
            "",
            "| Symbol-count third | Pages | n_gt range | Mean × 100 |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for third in summary["density_thirds"]:
        mean = f"{100 * third['omr_ned']:.2f}" if third["omr_ned"] is not None else "unavailable"
        lines.append(
            f"| {third['label']} | {third['count']} | {third['min_n_gt']} to {third['max_n_gt']} | {mean} |"
        )
    for pair in summary["paired_differences"].values():
        lines.extend(
            ["", f"Paired {pair['comparison']}: {100 * pair['mean_difference']:.2f} points."]
        )
        for name in ("document_cluster", "page"):
            interval = pair["bootstrap"][name]
            if interval["available"]:
                lines.append(
                    f"{name} CI: [{100 * interval['ci_low']:.2f}, {100 * interval['ci_high']:.2f}]."
                )
    provenance = summary["provenance"]
    lines.extend(
        [
            "",
            f"Release: `{provenance['release']['release_digest']}`.",
            f"Post-processing in order: {', '.join(provenance['postprocessing']['ordered_options']) or 'none'}.",
            f"Prediction format override: {provenance['notation_conversion']['prediction_format_override'] or 'none'}.",
            "",
            "| Provenance | SHA-256 |",
            "| --- | --- |",
            f"| Predictions | `{provenance['predictions']['sha256']}` |",
            f"| Reference pages | `{provenance['release']['pages_sha256']}` |",
            f"| Evaluation profile | `{provenance['profile']['sha256']}` |",
            f"| VeroSim | `{provenance['tools']['verosim']['sha256']}` |",
            f"| kern_parser | `{provenance['tools']['kern_parser']['sha256'] if provenance['tools']['kern_parser'] else 'not used'}` |",
            f"| Density cut points | `{provenance['density_cut_points']['sha256']}` |",
            "",
            f"Rendering Verovio: `{provenance['tools']['rendering_verovio']['version']}`.",
            "Full provenance, commands, conversions, intervals, and per-page diagnostics are in summary.json and audit.jsonl.",
            "",
        ]
    )
    atomic_write_bytes(out / "summary.md", "\n".join(lines).encode())


if __name__ == "__main__":
    raise SystemExit(main())
