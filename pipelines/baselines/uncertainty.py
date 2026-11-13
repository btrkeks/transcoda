"""Reproducible uncertainty and density analysis for aligned OMR-NED reports."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pipelines.baselines.io import atomic_write_bytes, sha256_file


@dataclass(frozen=True)
class PageObservation:
    sample_id: str
    document_id: str
    n_gt: int
    scores: dict[str, float]
    statuses: dict[str, str]


@dataclass(frozen=True)
class BootstrapResult:
    point: np.ndarray
    low: np.ndarray
    high: np.ndarray
    estimates: np.ndarray


def load_score_report(
    path: Path, *, failure_score: float
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    """Load one row-level score report, imputing its bounded failure score."""

    order: list[str] = []
    scores: dict[str, float] = {}
    statuses: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        payload = json.loads(line)
        sample_id = str(payload["sample_id"])
        if sample_id in scores:
            raise ValueError(f"duplicate sample_id {sample_id!r} in {path}:{line_number}")
        status = str(payload["status"])
        value = payload.get("omr_ned")
        if status == "scored":
            if not isinstance(value, int | float):
                raise ValueError(f"scored row {sample_id!r} has no numeric omr_ned")
            score = float(value)
        else:
            score = failure_score
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"bounded OMR-NED for {sample_id!r} is outside [0, 1]: {score}")
        order.append(sample_id)
        scores[sample_id] = score
        statuses[sample_id] = status
    if not order:
        raise ValueError(f"score report is empty: {path}")
    return order, scores, statuses


def load_complexity(path: Path) -> dict[str, int]:
    """Load representation-neutral projected ground-truth symbol counts."""

    result: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        payload = json.loads(line)
        sample_id = str(payload["id"])
        if sample_id in result:
            raise ValueError(f"duplicate complexity id {sample_id!r} in {path}:{line_number}")
        n_gt = payload.get("n_gt")
        if payload.get("ok") is not True or not isinstance(n_gt, int) or n_gt <= 0:
            raise ValueError(f"complexity source lacks a valid n_gt for {sample_id!r}")
        result[sample_id] = n_gt
    return result


def load_document_ids(dataset_dir: Path) -> dict[str, str]:
    """Read only identity columns from a Hugging Face dataset materialization."""

    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_dir)).select_columns(["sample_id", "document_id"])
    result: dict[str, str] = {}
    for row in dataset:
        sample_id = str(row["sample_id"])
        document_id = str(row["document_id"])
        if not document_id:
            raise ValueError(f"dataset row {sample_id!r} has no document_id")
        if sample_id in result:
            raise ValueError(f"dataset contains duplicate sample_id {sample_id!r}")
        result[sample_id] = document_id
    return result


def align_observations(
    reports: dict[str, tuple[list[str], dict[str, float], dict[str, str]]],
    *,
    complexity: dict[str, int],
    document_ids: dict[str, str],
    exclude_prefixes: tuple[str, ...],
) -> list[PageObservation]:
    """Validate exact report alignment and join model-independent page metadata."""

    if not reports:
        raise ValueError("at least one model report is required")
    first_label, (reference_order, _, _) = next(iter(reports.items()))
    reference_set = set(reference_order)
    for label, (order, _, _) in reports.items():
        if order != reference_order:
            missing = sorted(reference_set - set(order))
            extra = sorted(set(order) - reference_set)
            raise ValueError(
                f"model report {label!r} is not in exact {first_label!r} order; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    observations: list[PageObservation] = []
    for sample_id in reference_order:
        if sample_id.startswith(exclude_prefixes):
            continue
        if sample_id not in complexity:
            raise ValueError(f"missing n_gt for retained sample {sample_id!r}")
        if sample_id not in document_ids:
            raise ValueError(f"missing document_id for retained sample {sample_id!r}")
        observations.append(
            PageObservation(
                sample_id=sample_id,
                document_id=document_ids[sample_id],
                n_gt=complexity[sample_id],
                scores={label: values[sample_id] for label, (_, values, _) in reports.items()},
                statuses={label: values[sample_id] for label, (_, _, values) in reports.items()},
            )
        )
    if not observations:
        raise ValueError("cohort filters removed every sample")
    return observations


def page_bootstrap(
    scores: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
    level: float,
    chunk_size: int = 1_000,
) -> BootstrapResult:
    """Paired nonparametric bootstrap over aligned pages."""

    _validate_bootstrap(scores, n_resamples=n_resamples, level=level)
    rng = np.random.default_rng(seed)
    model_count, page_count = scores.shape
    estimates = np.empty((n_resamples, model_count), dtype=np.float64)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, page_count, size=(stop - start, page_count))
        for model_index in range(model_count):
            estimates[start:stop, model_index] = scores[model_index, indices].mean(axis=1)
    return _bootstrap_result(scores.mean(axis=1), estimates, level=level)


def cluster_bootstrap(
    scores: np.ndarray,
    document_ids: list[str],
    *,
    seed: int,
    n_resamples: int,
    level: float,
    chunk_size: int = 1_000,
) -> BootstrapResult:
    """Paired cluster bootstrap over documents, retaining page-weighted means."""

    _validate_bootstrap(scores, n_resamples=n_resamples, level=level)
    if scores.shape[1] != len(document_ids):
        raise ValueError("document id count must equal score column count")
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, document_id in enumerate(document_ids):
        grouped_indices[document_id].append(index)
    groups = list(grouped_indices.values())
    group_count = len(groups)
    if group_count < 2:
        raise ValueError("cluster bootstrap requires at least two documents")

    sizes = np.asarray([len(indices) for indices in groups], dtype=np.float64)
    sums = np.asarray(
        [
            [scores[model_index, indices].sum() for indices in groups]
            for model_index in range(scores.shape[0])
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty((n_resamples, scores.shape[0]), dtype=np.float64)
    probabilities = np.full(group_count, 1.0 / group_count)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        counts = rng.multinomial(group_count, probabilities, size=stop - start)
        denominators = counts @ sizes
        estimates[start:stop] = (counts @ sums.T) / denominators[:, None]
    return _bootstrap_result(scores.mean(axis=1), estimates, level=level)


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
        "fraction_at_most_0_25": float(np.mean(values <= 0.25)),
        "fraction_at_most_0_50": float(np.mean(values <= 0.50)),
        "fraction_at_most_0_75": float(np.mean(values <= 0.75)),
    }


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation with average ranks for ties."""

    if x.size != y.size or x.size < 2:
        raise ValueError("Spearman correlation requires equally sized nontrivial inputs")
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    result = float(np.corrcoef(x_rank, y_rank)[0, 1])
    if not np.isfinite(result):
        raise ValueError("Spearman correlation is undefined for constant input")
    return result


def density_tertiles(n_gt: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Assign fixed cohort-wide tertiles from model-independent n_gt cut points."""

    lower, upper = np.quantile(n_gt, [1 / 3, 2 / 3])
    assignments = np.where(n_gt <= lower, 0, np.where(n_gt <= upper, 1, 2))
    labels = ("low", "medium", "high")
    metadata: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        selected = n_gt[assignments == index]
        metadata.append(
            {
                "label": label,
                "count": int(selected.size),
                "min_n_gt": int(selected.min()),
                "max_n_gt": int(selected.max()),
            }
        )
    return assignments, metadata


def analyze(
    observations: list[PageObservation],
    *,
    model_labels: list[str],
    reference_model: str,
    seed: int,
    n_resamples: int,
    level: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute uncertainty, paired differences, distributions, and density views."""

    if reference_model not in model_labels:
        raise ValueError(f"reference model {reference_model!r} is absent")
    scores = np.asarray(
        [[observation.scores[label] for observation in observations] for label in model_labels],
        dtype=np.float64,
    )
    n_gt = np.asarray([observation.n_gt for observation in observations], dtype=np.float64)
    documents = [observation.document_id for observation in observations]
    page = page_bootstrap(scores, seed=seed, n_resamples=n_resamples, level=level)
    cluster = cluster_bootstrap(
        scores,
        documents,
        seed=seed + 1,
        n_resamples=n_resamples,
        level=level,
    )
    reference_index = model_labels.index(reference_model)

    model_results: dict[str, Any] = {}
    for model_index, label in enumerate(model_labels):
        values = scores[model_index]
        model_results[label] = {
            "bounded_failure_count": sum(
                observation.statuses[label] != "scored" for observation in observations
            ),
            "distribution": distribution_summary(values),
            "page_bootstrap": _interval_payload(page, model_index),
            "document_cluster_bootstrap": _interval_payload(cluster, model_index),
            "spearman_n_gt": spearman_correlation(n_gt, values),
        }

    paired: dict[str, Any] = {}
    for model_index, label in enumerate(model_labels):
        if label == reference_model:
            continue
        key = f"{reference_model} minus {label}"
        paired[key] = {
            "interpretation": "negative favors the reference model because lower OMR-NED is better",
            "page_bootstrap": _difference_payload(page, reference_index, model_index, level=level),
            "document_cluster_bootstrap": _difference_payload(
                cluster,
                reference_index,
                model_index,
                level=level,
            ),
        }

    assignments, tertile_metadata = density_tertiles(n_gt)
    tertiles: list[dict[str, Any]] = []
    for index, metadata in enumerate(tertile_metadata):
        selected = assignments == index
        subset_scores = scores[:, selected]
        subset_bootstrap = page_bootstrap(
            subset_scores,
            seed=seed + 100 + index,
            n_resamples=n_resamples,
            level=level,
        )
        tertiles.append(
            {
                **metadata,
                "models": {
                    label: {
                        "mean": float(subset_scores[model_index].mean()),
                        "page_bootstrap": _interval_payload(subset_bootstrap, model_index),
                    }
                    for model_index, label in enumerate(model_labels)
                },
                "paired_differences": {
                    f"{reference_model} minus {label}": _difference_payload(
                        subset_bootstrap,
                        reference_index,
                        model_index,
                        level=level,
                    )
                    for model_index, label in enumerate(model_labels)
                    if label != reference_model
                },
            }
        )

    payload = {
        "schema_version": 1,
        "metric": "OMR-NED",
        "scale": "fraction; multiply by 100 for paper table",
        "lower_is_better": True,
        "reference_model": reference_model,
        "cohort": {
            "count": len(observations),
            "document_count": len(set(documents)),
            "sample_id_digest": _string_digest([item.sample_id for item in observations]),
        },
        "bootstrap": {
            "confidence_level": level,
            "method": "percentile",
            "page_resamples": n_resamples,
            "page_seed": seed,
            "document_cluster_resamples": n_resamples,
            "document_cluster_seed": seed + 1,
            "paired": True,
        },
        "density": {
            "definition": "VeroSim profile-projected canonical ground-truth symbol count (n_gt)",
            "tertile_rule": "cohort-wide empirical 1/3 and 2/3 quantile cut points",
            "tertiles": tertiles,
        },
        "degradation_severity": {
            "available": False,
            "reason": "The retained dataset has no varying degradation-severity annotation; synthetic degradation fields are constant/not applicable for these real scans.",
        },
        "models": model_results,
        "paired_differences": paired,
    }
    arrays = {
        "scores": scores,
        "n_gt": n_gt,
        "density_assignments": assignments,
        "page_bootstrap": page,
        "cluster_bootstrap": cluster,
    }
    return payload, arrays


def write_analysis(
    output_dir: Path,
    *,
    observations: list[PageObservation],
    model_labels: list[str],
    analysis: dict[str, Any],
    arrays: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Path]:
    """Write machine-readable, tabular, narrative, and visual artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_payload = {**analysis, "provenance": provenance}
    analysis_path = output_dir / "analysis.json"
    atomic_write_bytes(analysis_path, _canonical_json_bytes(analysis_payload))

    pages_path = output_dir / "pages.csv"
    with pages_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "document_id", "n_gt", "density_tertile"]
            + [f"{label} OMR-NED" for label in model_labels]
            + [f"{label} status" for label in model_labels]
        )
        density_labels = ("low", "medium", "high")
        assignments = arrays["density_assignments"]
        for index, observation in enumerate(observations):
            writer.writerow(
                [
                    observation.sample_id,
                    observation.document_id,
                    observation.n_gt,
                    density_labels[int(assignments[index])],
                    *[observation.scores[label] for label in model_labels],
                    *[observation.statuses[label] for label in model_labels],
                ]
            )

    summary_path = output_dir / "summary.md"
    summary_path.write_text(_markdown_summary(analysis, model_labels), encoding="utf-8")
    figure_path = output_dir / "uncertainty-density.pdf"
    png_path = output_dir / "uncertainty-density.png"
    _write_figure(figure_path, png_path, analysis, arrays, model_labels)
    scatter_path = output_dir / "density-continuous.pdf"
    scatter_png_path = output_dir / "density-continuous.png"
    _write_density_figure(scatter_path, scatter_png_path, analysis, arrays, model_labels)

    manifest_path = output_dir / "artifacts.json"
    artifact_paths = [
        analysis_path,
        pages_path,
        summary_path,
        figure_path,
        png_path,
        scatter_path,
        scatter_png_path,
    ]
    manifest = {
        path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
        for path in artifact_paths
    }
    atomic_write_bytes(manifest_path, _canonical_json_bytes(manifest))
    return {
        "analysis": analysis_path,
        "pages": pages_path,
        "summary": summary_path,
        "figure_pdf": figure_path,
        "figure_png": png_path,
        "density_figure_pdf": scatter_path,
        "density_figure_png": scatter_png_path,
        "artifacts": manifest_path,
    }


def _validate_bootstrap(scores: np.ndarray, *, n_resamples: int, level: float) -> None:
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("bootstrap scores must be a nonempty model-by-page matrix")
    if n_resamples <= 0:
        raise ValueError("bootstrap requires at least one resample")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must be between zero and one")


def _bootstrap_result(point: np.ndarray, estimates: np.ndarray, *, level: float) -> BootstrapResult:
    alpha = 1.0 - level
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    return BootstrapResult(point=point, low=low, high=high, estimates=estimates)


def _interval_payload(result: BootstrapResult, model_index: int) -> dict[str, float]:
    return {
        "point": float(result.point[model_index]),
        "ci_low": float(result.low[model_index]),
        "ci_high": float(result.high[model_index]),
    }


def _difference_payload(
    result: BootstrapResult,
    reference_index: int,
    comparison_index: int,
    *,
    level: float,
) -> dict[str, float]:
    values = result.estimates[:, reference_index] - result.estimates[:, comparison_index]
    alpha = 1.0 - level
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "point": float(result.point[reference_index] - result.point[comparison_index]),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _string_digest(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _markdown_summary(analysis: dict[str, Any], model_labels: list[str]) -> str:
    reference_model = str(analysis["reference_model"])
    lines = [
        "# Non-Polish OMR-NED uncertainty and density analysis",
        "",
        f"Cohort: {analysis['cohort']['count']} pages from {analysis['cohort']['document_count']} documents. Values below are OMR-NED × 100; lower is better.",
        "",
        "| Model | Mean | Page-bootstrap 95% CI | Document-cluster 95% CI | Median | IQR | Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in model_labels:
        result = analysis["models"][label]
        page = result["page_bootstrap"]
        cluster = result["document_cluster_bootstrap"]
        distribution = result["distribution"]
        lines.append(
            f"| {label} | {100 * page['point']:.2f} | "
            f"[{100 * page['ci_low']:.2f}, {100 * page['ci_high']:.2f}] | "
            f"[{100 * cluster['ci_low']:.2f}, {100 * cluster['ci_high']:.2f}] | "
            f"{100 * distribution['median']:.2f} | "
            f"[{100 * distribution['q25']:.2f}, {100 * distribution['q75']:.2f}] | "
            f"{result['bounded_failure_count']} |"
        )
    lines.extend(
        [
            "",
            "## Paired differences",
            "",
            f"Negative differences favor {reference_model}.",
            "",
            "| Difference | Point | Page-bootstrap 95% CI | Document-cluster 95% CI |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label, result in analysis["paired_differences"].items():
        page = result["page_bootstrap"]
        cluster = result["document_cluster_bootstrap"]
        lines.append(
            f"| {label} | {100 * page['point']:.2f} | "
            f"[{100 * page['ci_low']:.2f}, {100 * page['ci_high']:.2f}] | "
            f"[{100 * cluster['ci_low']:.2f}, {100 * cluster['ci_high']:.2f}] |"
        )
    lines.extend(["", "## Ground-truth density tertiles", ""])
    header = "| Tertile | n | n_gt range | " + " | ".join(model_labels) + " |"
    separator = "| --- | ---: | ---: | " + " | ".join("---:" for _ in model_labels) + " |"
    lines.extend([header, separator])
    for tertile in analysis["density"]["tertiles"]:
        means = " | ".join(
            f"{100 * tertile['models'][label]['mean']:.2f}" for label in model_labels
        )
        lines.append(
            f"| {tertile['label']} | {tertile['count']} | "
            f"{tertile['min_n_gt']}–{tertile['max_n_gt']} | {means} |"
        )
    lines.extend(
        [
            "",
            f"### Paired {reference_model} differences within density tertiles",
            "",
            f"Negative differences favor {reference_model}. Intervals use paired page bootstrap within each tertile.",
            "",
            "| Tertile | Comparison | Difference | 95% CI |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for tertile in analysis["density"]["tertiles"]:
        for label, result in tertile["paired_differences"].items():
            lines.append(
                f"| {tertile['label']} | {label} | {100 * result['point']:.2f} | "
                f"[{100 * result['ci_low']:.2f}, {100 * result['ci_high']:.2f}] |"
            )
    lines.extend(
        [
            "",
            "### Continuous density association",
            "",
            "| Model | Spearman ρ(n_gt, OMR-NED) |",
            "| --- | ---: |",
        ]
    )
    for label in model_labels:
        lines.append(f"| {label} | {analysis['models'][label]['spearman_n_gt']:.3f} |")
    lines.extend(
        [
            "",
            "Degradation-severity stratification is unavailable because the retained real-scan rows have no varying severity annotation.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_figure(
    pdf_path: Path,
    png_path: Path,
    analysis: dict[str, Any],
    arrays: dict[str, Any],
    model_labels: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#355C7D", "#C06C84", "#2A9D8F", "#E9C46A"]
    scores = arrays["scores"]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    for index, label in enumerate(model_labels):
        ordered = np.sort(scores[index]) * 100.0
        cumulative = np.arange(1, ordered.size + 1) / ordered.size
        axes[0].step(ordered, cumulative, where="post", label=label, color=colors[index])
    axes[0].set(
        xlabel="OMR-NED × 100", ylabel="Cumulative fraction", title="Page-level distribution"
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    tertiles = analysis["density"]["tertiles"]
    x = np.arange(len(tertiles))
    width = 0.8 / len(model_labels)
    for index, label in enumerate(model_labels):
        means = np.asarray([item["models"][label]["mean"] for item in tertiles]) * 100.0
        lows = (
            np.asarray([item["models"][label]["page_bootstrap"]["ci_low"] for item in tertiles])
            * 100.0
        )
        highs = (
            np.asarray([item["models"][label]["page_bootstrap"]["ci_high"] for item in tertiles])
            * 100.0
        )
        position = x - 0.4 + width / 2 + index * width
        axes[1].bar(position, means, width, label=label, color=colors[index], alpha=0.9)
        axes[1].errorbar(
            position,
            means,
            yerr=[means - lows, highs - means],
            fmt="none",
            ecolor="black",
            capsize=2,
            linewidth=0.8,
        )
    axes[1].set_xticks(x, [item["label"] for item in tertiles])
    axes[1].set(
        xlabel="Ground-truth density tertile",
        ylabel="Mean OMR-NED × 100",
        title="Performance by page density",
    )
    axes[1].grid(axis="y", alpha=0.2)

    reference_model = str(analysis["reference_model"])
    differences = list(analysis["paired_differences"].items())
    y = np.arange(len(differences))
    for offset, method in [(-0.12, "page_bootstrap"), (0.12, "document_cluster_bootstrap")]:
        points = np.asarray([item[1][method]["point"] for item in differences]) * 100.0
        lows = np.asarray([item[1][method]["ci_low"] for item in differences]) * 100.0
        highs = np.asarray([item[1][method]["ci_high"] for item in differences]) * 100.0
        axes[2].errorbar(
            points,
            y + offset,
            xerr=[points - lows, highs - points],
            fmt="o",
            capsize=3,
            label="Page" if method == "page_bootstrap" else "Document cluster",
        )
    axes[2].axvline(0.0, color="black", linewidth=0.8)
    axes[2].set_yticks(
        y, [label.replace(f"{reference_model} minus ", "vs. ") for label, _ in differences]
    )
    axes[2].set(
        xlabel="Paired OMR-NED difference × 100",
        title="Reference-model advantage (95% CI)",
    )
    axes[2].grid(axis="x", alpha=0.2)
    axes[2].legend(frameon=False, fontsize=8)

    figure.savefig(pdf_path, metadata={"Creator": "Transcoda baseline uncertainty analysis"})
    figure.savefig(png_path, dpi=180)
    plt.close(figure)


def _write_density_figure(
    pdf_path: Path,
    png_path: Path,
    analysis: dict[str, Any],
    arrays: dict[str, Any],
    model_labels: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#355C7D", "#C06C84", "#2A9D8F", "#E9C46A"]
    scores = arrays["scores"] * 100.0
    n_gt = arrays["n_gt"]
    order = np.argsort(n_gt)
    ordered_n_gt = n_gt[order]
    window = min(31, max(7, (len(order) // 7) | 1))
    kernel = np.full(window, 1.0 / window)

    figure, axes = plt.subplots(
        1,
        len(model_labels),
        figsize=(4.5 * len(model_labels), 4.1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(model_labels) == 1:
        axes = [axes]
    for index, label in enumerate(model_labels):
        axis = axes[index]
        ordered_scores = scores[index, order]
        axis.scatter(n_gt, scores[index], s=14, alpha=0.28, color=colors[index], edgecolors="none")
        rolling = np.convolve(ordered_scores, kernel, mode="valid")
        half = window // 2
        axis.plot(
            ordered_n_gt[half : len(order) - half],
            rolling,
            color=colors[index],
            linewidth=2.0,
            label=f"{window}-page rolling mean",
        )
        axis.set_xscale("log")
        axis.set_title(f"{label}\nSpearman ρ = {analysis['models'][label]['spearman_n_gt']:.3f}")
        axis.set_xlabel("Projected ground-truth symbols (n_gt, log scale)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("OMR-NED × 100")
    figure.savefig(pdf_path, metadata={"Creator": "Transcoda baseline uncertainty analysis"})
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
