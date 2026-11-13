"""Pinned evaluation pages and model-independent notation conversion."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pipelines.baselines.contracts import canonical_json_bytes
from pipelines.baselines.io import sha256_file
from pipelines.baselines.normalization import (
    SubprocessCommandRunner,
    convert_abc_to_musicxml,
    smt_bekern_to_kern,
    transcoda_kern_envelope,
)

FORMATS = frozenset({"kern", "transcoda-kern", "transcoda-v2", "bekern", "abc", "musicxml", "mei"})
MODEL_FAMILIES = ("transcoda", "smt", "legato", "native")


@dataclass(frozen=True)
class EvaluationPage:
    sample_id: str
    document_id: str
    target: str
    target_format: str


@dataclass(frozen=True)
class EvaluationRelease:
    pages: tuple[EvaluationPage, ...]
    provenance: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(row)
    return rows


def unique_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {}
    for row in rows:
        sample_id = required_text(row, "sample_id")
        if sample_id in by_id:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        by_id[sample_id] = row
    if not by_id:
        raise ValueError("evaluation input is empty")
    return by_id


def required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing nonempty string {key}")
    return value


def file_provenance(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def load_evaluation_release(
    ref: str,
    *,
    releases_root: Path = Path("data/releases"),
    dataset_dir: Path | None = None,
) -> EvaluationRelease:
    """Load a repository release or a self-contained, text-only evaluation pin.

    A text pin's entire file is its identity. Repository releases additionally
    verify the dataset bytes against the sealed snapshot, without writing caches.
    """
    path = Path(ref)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == "transcoda.evaluation-pages.v1":
            if payload.get("role") not in {"eval", "test"}:
                raise ValueError("evaluation page pin must have role eval or test")
            rows = list(unique_rows(payload["pages"]).values())
            pages = tuple(
                EvaluationPage(
                    required_text(row, "sample_id"),
                    required_text(row, "document_id"),
                    required_text(row, "target"),
                    required_text(row, "target_format"),
                )
                for row in rows
            )
            provenance = {
                **file_provenance(path),
                "release_digest": "sha256:" + sha256_file(path),
                "name": required_text(payload, "name"),
                "role": payload["role"],
            }
            return _validated_release(pages, provenance)

    from datasets import load_from_disk

    from transcoda.core.dataset_release import build_snapshot_manifest, check_release_dir

    release_dir = Path(ref)
    if not release_dir.is_dir():
        release_dir = releases_root / ref
    release_dir = release_dir.resolve()
    manifest, digest, _snapshot, members = check_release_dir(release_dir)
    if manifest.role not in {"eval", "test"}:
        raise ValueError(f"cannot evaluate training release {manifest.name}")
    dataset_dir = dataset_dir or release_dir.parents[2] / manifest.dataset_path
    actual = build_snapshot_manifest(dataset_dir, strict_metadata=True).manifest_digest
    if actual != manifest.snapshot_digest:
        raise ValueError(f"dataset bytes do not match release snapshot: {actual}")
    dataset = load_from_disk(str(dataset_dir))
    columns = ["sample_id", "document_id", "transcription"]
    if members is not None:
        columns.append("sample_revision_digest")
    rows = list(dataset.select_columns(columns))
    unique_rows(rows)
    if len(rows) != manifest.row_count:
        raise ValueError("release row_count does not match dataset")
    if members is not None:
        expected = {(row.sample_id, str(row.revision_digest)) for row in members}
        found = {(row["sample_id"], row["sample_revision_digest"]) for row in rows}
        if expected != found:
            raise ValueError("dataset does not contain the pinned members and revisions")
    pages = tuple(
        EvaluationPage(
            required_text(row, "sample_id"),
            required_text(row, "document_id"),
            required_text(row, "transcription"),
            "transcoda-v2",
        )
        for row in rows
    )
    return _validated_release(
        pages,
        {
            **file_provenance(release_dir / "release.json"),
            "release_digest": digest,
            "snapshot_digest": manifest.snapshot_digest,
            "dataset_dir": str(dataset_dir.resolve()),
            "name": manifest.name,
            "role": manifest.role,
        },
    )


def _validated_release(
    pages: tuple[EvaluationPage, ...], provenance: dict[str, Any]
) -> EvaluationRelease:
    for page in pages:
        if page.target_format not in {"kern", "transcoda-v2", "musicxml", "mei"}:
            raise ValueError(f"unsupported reference format: {page.target_format}")
    ordered = tuple(sorted(pages, key=lambda page: page.sample_id))
    page_bytes = b"".join(canonical_json_bytes(asdict(page)) for page in ordered)
    return EvaluationRelease(
        ordered,
        {**provenance, "pages_sha256": hashlib.sha256(page_bytes).hexdigest()},
    )


def lower_transcoda(text: str, *, kern_parser: Path, timeout_s: float) -> str:
    result = SubprocessCommandRunner().run(
        [
            str(kern_parser),
            "serialize-jsonl",
            "--preset",
            "humdrum-kern",
            "--input-dialect",
            "transcoda-v2",
            "--workers",
            "1",
            "--text-field",
            "transcription",
            "--id-field",
            "sample_id",
        ],
        input_bytes=canonical_json_bytes({"sample_id": "page", "transcription": text}),
        timeout_s=timeout_s,
    )
    # Rejected rows carry the page-specific cause even when the batch exits nonzero.
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(rows) == 1 and rows[0].get("sample_id") == "page" and rows[0].get("error"):
        raise ValueError(f"kern_parser lowering failed: {rows[0]['error']}")
    if result.returncode != 0:
        raise RuntimeError(
            f"kern_parser exit {result.returncode}: {result.stderr.decode(errors='replace')[-2000:]}"
        )
    if len(rows) != 1 or rows[0].get("sample_id") != "page":
        raise ValueError("kern_parser returned incorrect row coverage")
    if not isinstance(rows[0].get("transcription"), str):
        raise ValueError("kern_parser lowering returned no transcription")
    return rows[0]["transcription"]


def prediction_route(row: dict[str, Any], *, family: str | None, override: str | None) -> str:
    """Use model vocabulary declarations; native rows require an explicit format."""
    model_id = str(row.get("model_id", "")).lower()
    identified = (
        "transcoda"
        if model_id.startswith("transcoda/")
        else "smt"
        if model_id.startswith("praig/smt-")
        else "legato"
        if model_id == "guangyangmusic/legato"
        else None
    )
    if family and identified and family != identified:
        raise ValueError(f"model family {family} contradicts model_id {model_id}")
    family = family or identified
    # These declarations come from the checkpoint/tokenizer evidence in
    # docs/camera-ready-v2-dialects.json, not the archived native_format labels.
    declared = None
    if family == "transcoda":
        if model_id == "transcoda/clean-v2" or model_id.startswith("transcoda/ablation-"):
            declared = "transcoda-v2"
        elif model_id in {
            "transcoda/v1",
            "transcoda/fcmae3",
            "transcoda/transcoda-59m-zeroshot-v1",
        }:
            declared = "transcoda-kern"
        if declared is None and override is None:
            raise ValueError(
                "unknown Transcoda vocabulary; declare --prediction-format from config"
            )
    elif family == "smt":
        declared = "bekern"
    elif family == "legato":
        declared = "abc"
    if declared:
        # bekern/kern are historical aliases for the V1 envelope only.
        requested = (
            "transcoda-kern"
            if override in {"bekern", "kern"} and family == "transcoda"
            else override
        )
        if requested and requested != declared:
            raise ValueError(
                f"prediction format {override} contradicts model configuration {declared}"
            )
        return declared
    format = override or row.get("prediction_format") or row.get("native_format")
    if format not in FORMATS:
        raise ValueError("each prediction needs a supported native_format or --prediction-format")
    if family == "transcoda":
        if format not in {"transcoda-v2", "transcoda-kern", "kern", "bekern"}:
            raise ValueError("Transcoda requires a Kern-family prediction format")
        return "transcoda-v2" if format == "transcoda-v2" else "transcoda-kern"
    if format == "bekern":
        raise ValueError("ambiguous bekern input; declare --model-family smt or transcoda")
    if family and format in {"transcoda-v2", "transcoda-kern"}:
        raise ValueError("Transcoda notation contradicts the declared model family")
    return format


def normalize_notation(
    text: str,
    format: str,
    *,
    postprocess: tuple[str, ...] = (),
    kern_parser: Path | None,
    abc2xml_script: Path,
    abc2xml_python: Path = Path(sys.executable),
    timeout_s: float = 120,
    reference: bool = False,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Convert the declared dialect; compact predictions share the reference converter."""
    if postprocess:
        raise ValueError("semantic repairs and optional post-processing are forbidden")
    if format not in FORMATS:
        raise ValueError(f"unsupported prediction format: {format}")
    audit = []
    original = text
    if format == "transcoda-v2":
        if kern_parser is None:
            raise ValueError("transcoda-v2 conversion requires kern_parser")
        text = lower_transcoda(text, kern_parser=kern_parser, timeout_s=timeout_s)
        audit.append({"name": "transcoda-v2-to-humdrum-kern", "changed": text != original})
        format = "kern"
    elif reference:
        pass
    elif format == "bekern":
        text, format = smt_bekern_to_kern(text), "kern"
    elif format in {"kern", "transcoda-v2", "transcoda-kern"}:
        before = text
        text = transcoda_kern_envelope(text)
        audit.append({"name": "kern-envelope", "changed": before != text})
        format = "kern"
    if format == "abc":
        text = convert_abc_to_musicxml(
            text, python_bin=abc2xml_python, abc2xml_script=abc2xml_script, timeout_s=timeout_s
        )
        format = "musicxml"
    if not text.strip():
        raise ValueError("empty normalized notation")
    return text, format, audit
