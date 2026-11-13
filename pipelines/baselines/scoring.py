"""Exact-cohort VeroSim scoring and deterministic baseline reports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipelines.baselines.contracts import (
    SCHEMA_VERSION,
    GroundTruthManifestRecord,
    PredictionRecord,
    canonical_json_bytes,
)
from pipelines.baselines.io import atomic_write_bytes, sha256_file
from pipelines.baselines.normalization import CommandRunner, SubprocessCommandRunner


@dataclass(frozen=True)
class JoinedPrediction:
    prediction: PredictionRecord
    ground_truth: GroundTruthManifestRecord


@dataclass(frozen=True)
class ScoreCandidate:
    sample_id: str
    route: str
    prediction: str | None
    target: str
    failure_stage: str | None = None
    error: str | None = None
    truncated: bool = False
    prediction_format: str = "kern"
    target_format: str = "kern"

    def __post_init__(self) -> None:
        if (self.failure_stage is None) != (self.prediction is not None):
            raise ValueError("failure_stage must be present exactly when prediction is absent")
        if (self.failure_stage is None) != (self.error is None):
            raise ValueError("error must be present exactly for pre-backend failures")
        supported_formats = {"kern", "mei", "musicxml"}
        if self.prediction_format not in supported_formats:
            raise ValueError(f"unsupported prediction format: {self.prediction_format}")
        if self.target_format not in supported_formats:
            raise ValueError(f"unsupported target format: {self.target_format}")


@dataclass(frozen=True)
class VeroSimBatchResult:
    records: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    command: tuple[str, ...]
    binary_sha256: str
    artifacts: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreReport:
    samples: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    backend: dict[str, Any]


@dataclass(frozen=True)
class ScoreArtifacts:
    report_json: Path
    samples_jsonl: Path


def join_prediction_ground_truth(
    predictions: list[PredictionRecord],
    ground_truth: list[GroundTruthManifestRecord],
) -> list[JoinedPrediction]:
    prediction_by_id = _unique_by_id(predictions, label="prediction")
    ground_truth_by_id = _unique_by_id(ground_truth, label="ground truth")
    missing = sorted(set(prediction_by_id) - set(ground_truth_by_id))
    extra = sorted(set(ground_truth_by_id) - set(prediction_by_id))
    if missing:
        raise ValueError("missing ground truth for sample IDs: " + ", ".join(missing))
    if extra:
        raise ValueError("extra ground truth sample IDs: " + ", ".join(extra))
    return [
        JoinedPrediction(prediction_by_id[sample_id], ground_truth_by_id[sample_id])
        for sample_id in sorted(prediction_by_id)
    ]


class VeroSimDriver:
    def __init__(
        self,
        binary: Path,
        *,
        mode: str = "active",
        layout: str = "system-breaks",
        jobs: int = 1,
        failure_score: float = 100.0,
        evaluation_profile: Path | None = None,
        runner: CommandRunner | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.binary = binary.resolve(strict=True)
        self.binary_sha256 = sha256_file(self.binary)
        if jobs != 1:
            raise ValueError("baseline VeroSim scoring is pinned to jobs=1")
        self.mode = mode
        self.layout = layout
        self.jobs = jobs
        self.failure_score = failure_score
        self.evaluation_profile = (
            evaluation_profile.resolve(strict=True) if evaluation_profile is not None else None
        )
        self.runner = runner or SubprocessCommandRunner()
        self.timeout_s = timeout_s

    def run(self, candidates: list[ScoreCandidate], *, output_dir: Path) -> VeroSimBatchResult:
        scorable = [candidate for candidate in candidates if candidate.prediction is not None]
        if not scorable:
            raise ValueError("VeroSim requires at least one scorable candidate")
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "verosim.input.jsonl"
        records_path = output_dir / "verosim.records.jsonl"
        summary_path = output_dir / "verosim.summary.json"
        input_payload = b"".join(
            canonical_json_bytes(
                {
                    "prediction": candidate.prediction,
                    "prediction_format": candidate.prediction_format,
                    "route": candidate.route,
                    "sample_id": candidate.sample_id,
                    "target": candidate.target,
                    "target_format": candidate.target_format,
                }
            )
            for candidate in sorted(scorable, key=lambda item: (item.route, item.sample_id))
        )
        atomic_write_bytes(input_path, input_payload)
        command_parts = [
            str(self.binary),
            "--batch-jsonl",
            str(input_path),
            "--pred-field",
            "prediction",
            "--gt-field",
            "target",
            "--id-field",
            "sample_id",
            "--group-field",
            "route",
            "--dedupe-key",
            "route,sample_id",
            "--format",
            "kern",
            "--failure-score",
            str(self.failure_score),
            "--summary-json",
            str(summary_path),
            "--jobs",
            str(self.jobs),
        ]
        if self.evaluation_profile is None:
            command_parts.extend(["--mode", self.mode, "--layout", self.layout])
        else:
            command_parts.extend(["--evaluation-profile", str(self.evaluation_profile)])
        command = tuple(command_parts)
        result = self.runner.run(
            list(command),
            input_bytes=b"",
            timeout_s=self.timeout_s,
        )
        if result.returncode != 0:
            diagnostic = result.stderr.decode("utf-8", errors="replace").strip()[:2000]
            raise RuntimeError(
                f"VeroSim failed with exit code {result.returncode}: {diagnostic}"
            )
        atomic_write_bytes(records_path, result.stdout)
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid VeroSim summary: {summary_path}") from exc
        records = tuple(_read_jsonl(records_path, label="VeroSim records"))
        return VeroSimBatchResult(
            records=records,
            summary=summary,
            command=command,
            binary_sha256=self.binary_sha256,
            artifacts={
                "input": {"path": input_path.name, "sha256": sha256_file(input_path)},
                "records": {"path": records_path.name, "sha256": sha256_file(records_path)},
                "summary": {"path": summary_path.name, "sha256": sha256_file(summary_path)},
            },
        )

    def empty_result(self) -> VeroSimBatchResult:
        """Represent a cohort with no scorable rows without invoking the backend."""

        return VeroSimBatchResult(
            records=(),
            summary={
                "overall": {
                    "count": 0,
                    "scored": 0,
                    "failures": 0,
                    "omr_ned": 0.0,
                    "failure_score": self.failure_score,
                },
                "by_group": {},
            },
            command=(str(self.binary), "<not invoked: no scorable rows>"),
            binary_sha256=self.binary_sha256,
        )


def build_score_report(
    candidates: list[ScoreCandidate],
    *,
    batch: VeroSimBatchResult,
) -> ScoreReport:
    if not candidates:
        raise ValueError("score cohort must not be empty")
    candidate_by_key: dict[tuple[str, str], ScoreCandidate] = {}
    for candidate in candidates:
        key = (candidate.route, candidate.sample_id)
        if key in candidate_by_key:
            raise ValueError(f"duplicate score candidate: {key}")
        candidate_by_key[key] = candidate
    scorable_keys = {
        key for key, candidate in candidate_by_key.items() if candidate.prediction is not None
    }
    backend_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in batch.records:
        key = (str(record.get("group", "")), str(record.get("id", "")))
        if key in backend_by_key:
            raise ValueError(f"duplicate backend record: {key}")
        backend_by_key[key] = record
    if set(backend_by_key) != scorable_keys:
        missing = sorted(scorable_keys - set(backend_by_key))
        extra = sorted(set(backend_by_key) - scorable_keys)
        raise ValueError(f"backend records do not match scorable cohort; missing={missing}, extra={extra}")

    overall = batch.summary.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("VeroSim summary is missing overall object")
    failure_score = _required_float(overall, "failure_score")
    if int(overall.get("count", -1)) != len(scorable_keys):
        raise ValueError("VeroSim summary count does not match scorable cohort")

    rows: list[dict[str, Any]] = []
    score_sum = 0.0
    inference_failures = 0
    normalization_failures = 0
    backend_failures = 0
    scored_count = 0
    truncated_count = 0
    backend_score_sum = 0.0
    for key in sorted(candidate_by_key):
        candidate = candidate_by_key[key]
        if candidate.truncated:
            truncated_count += 1
        if candidate.prediction is None:
            status = f"{candidate.failure_stage}_failure"
            if candidate.failure_stage == "inference":
                inference_failures += 1
            else:
                normalization_failures += 1
            rows.append(
                {
                    "error": candidate.error,
                    "omr_ned": None,
                    "route": candidate.route,
                    "sample_id": candidate.sample_id,
                    "status": status,
                    "truncated": candidate.truncated,
                }
            )
            continue
        backend = backend_by_key[key]
        if bool(backend.get("ok", False)) and backend.get("omr_ned") is not None:
            value = float(backend["omr_ned"])
            score_sum += value
            backend_score_sum += value
            scored_count += 1
            status = "scored"
            error = None
        else:
            backend_failures += 1
            status = "backend_failure"
            error = str(backend.get("error") or backend.get("warnings") or "VeroSim failure")
            value = None
        rows.append(
            {
                "error": error,
                "omr_ned": value,
                "route": candidate.route,
                "sample_id": candidate.sample_id,
                "status": status,
                "truncated": candidate.truncated,
            }
        )

    backend_failure_count = len(scorable_keys) - scored_count
    backend_reported = _required_float(overall, "omr_ned")
    if scorable_keys:
        backend_recomputed = (
            backend_score_sum + backend_failure_count * failure_score
        ) / len(scorable_keys)
        if not math.isclose(backend_recomputed, backend_reported, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "VeroSim summary omr_ned does not match records/failure_score: "
                f"reported={backend_reported}, recomputed={backend_recomputed}"
            )
    elif backend_reported != 0.0:
        raise ValueError("empty VeroSim summary must report omr_ned 0")
    if int(overall.get("failures", -1)) != backend_failure_count:
        raise ValueError("VeroSim summary failure count does not match records")

    total = len(candidates)
    total_failures = total - scored_count
    headline = (score_sum + total_failures * failure_score) / total
    summary = {
        "backend_failures": backend_failures,
        "cohort_count": total,
        "coverage": scored_count / total,
        "failure_score": failure_score,
        "inference_failures": inference_failures,
        "normalization_failures": normalization_failures,
        "omr_ned": headline,
        "scored_count": scored_count,
        "truncated_count": truncated_count,
    }
    return ScoreReport(
        samples=tuple(rows),
        summary=summary,
        backend={
            "artifacts": batch.artifacts,
            "binary_sha256": batch.binary_sha256,
            "command": list(batch.command),
            "reported_summary": batch.summary,
        },
    )


def write_score_report(output_dir: Path, report: ScoreReport) -> ScoreArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "report.json"
    samples_jsonl = output_dir / "report.samples.jsonl"
    samples_payload = b"".join(canonical_json_bytes(row) for row in report.samples)
    atomic_write_bytes(
        samples_jsonl,
        samples_payload,
    )
    atomic_write_bytes(
        report_json,
        canonical_json_bytes(
            {
                "backend": report.backend,
                "samples": {
                    "path": samples_jsonl.name,
                    "sha256": sha256_file(samples_jsonl),
                },
                "schema_version": SCHEMA_VERSION,
                "summary": report.summary,
            }
        ),
    )
    return ScoreArtifacts(report_json, samples_jsonl)


def _unique_by_id(records, *, label: str):
    by_id = {}
    for record in records:
        if record.sample_id in by_id:
            raise ValueError(f"duplicate {label} sample_id: {record.sample_id}")
        by_id[record.sample_id] = record
    return by_id


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {label} JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {line_number} is not an object")
        rows.append(row)
    return rows


def _required_float(payload: dict[str, Any], key: str) -> float:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"VeroSim summary missing numeric {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"VeroSim summary {key} must be finite")
    return value
