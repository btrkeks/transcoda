"""VeroSim OMR-NED validation metric backend."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from transcoda.metrics_schema import val_aggregate_metric, val_set_metric

from .base import MetricResult
from .stored_targets import StoredTargetError
from .verosim_options import (
    DEFAULT_VEROSIM_MODE,
    normalize_verosim_mode,
)

try:
    from rapidfuzz.distance import Levenshtein as _RFLev

    _levenshtein = _RFLev.distance
except ImportError:

    def _levenshtein(a: list[Any], b: list[Any]) -> int:
        n, m = len(a), len(b)
        if n > m:
            a, b, n, m = b, a, m, n
        current = list(range(n + 1))
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1] + (a[j - 1] != b[i - 1])
                current[j] = min(add, delete, change)
        return current[n]


KERN_PARSER_ENV_VAR = "TRANSCODA_KERN_PARSER_BIN"
VEROSIM_ENV_VAR = "TRANSCODA_VEROSIM_BIN"
VEROSIM_LAYOUT_SURFACE = "system-breaks"
DIAGNOSTICS_SUMMARY_FILENAME = "validation_diagnostics.summary.json"
DIAGNOSTICS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BinaryResolution:
    path: Path | None
    tried: tuple[Path, ...]


@dataclass(frozen=True)
class PredictionDiagnostics:
    """Cheap diagnostics derived from persisted prediction records."""

    sample_counts: dict[str, int]
    truncated_counts: dict[str, int]
    hit_max_length_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class VeroSimOMRNEDResult:
    skipped: bool
    reason: str | None
    predictions_dir: Path
    merged_jsonl: Path | None = None
    converted_jsonl: Path | None = None
    records_jsonl: Path | None = None
    summary_json: Path | None = None
    summary: dict[str, Any] | None = None
    prediction_diagnostics: PredictionDiagnostics | None = None
    diagnostics_summary_json: Path | None = None
    diagnostics_summary: dict[str, Any] | None = None


class TargetLookup(Protocol):
    """Supplies the OMR-NED target for a prediction record (see ``stored_targets``)."""

    def target(self, record: dict[str, Any]) -> str: ...


class SubprocessRunner(Protocol):
    """Minimal subprocess interface for VeroSim backend tests."""

    def run(
        self,
        cmd: list[str],
        *,
        stdin: Any,
        stdout: Any,
    ) -> subprocess.CompletedProcess[bytes]: ...


class DefaultSubprocessRunner:
    """Subprocess runner using ``subprocess.run``."""

    def run(
        self,
        cmd: list[str],
        *,
        stdin: Any,
        stdout: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            cmd,
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=False,
            check=False,
        )


class VeroSimBinaryResolver:
    """Resolve external binaries needed by the VeroSim metric backend."""

    def __init__(self, *, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root)

    def resolve_kern_parser_binary(self, explicit: str | Path | None = None) -> BinaryResolution:
        return _resolve_binary(
            explicit=explicit,
            env_var=KERN_PARSER_ENV_VAR,
            binary_name="kern_parser",
            repo_candidates=(
                self.repo_root / "target/release/kern_parser",
                self.repo_root / "target/debug/kern_parser",
            ),
        )

    def resolve_verosim_binary(self, explicit: str | Path | None = None) -> BinaryResolution:
        return _resolve_binary(
            explicit=explicit,
            env_var=VEROSIM_ENV_VAR,
            binary_name="verosim",
            repo_candidates=(
                self.repo_root / "tools/verosim/verosim",
                self.repo_root.parent / "verosim/build/verosim",
                self.repo_root.parent / "VeroSim/build/verosim",
            ),
        )


class VeroSimOMRNEDBackend:
    """Compute OMR-NED from validation prediction JSONL files via VeroSim."""

    name = "verosim_omr_ned"

    def __init__(
        self,
        *,
        binary_resolver: VeroSimBinaryResolver,
        runner: SubprocessRunner | None = None,
        mode: str | None = None,
        detail: str | None = None,
        jobs: int = 0,
        kern_parser_binary: str | Path | None = None,
        verosim_binary: str | Path | None = None,
        failure_score: float | None = None,
        target_lookup: TargetLookup | None = None,
        rejected_targets_fatal: bool = True,
        exclude_val_sets: Collection[str] = (),
    ) -> None:
        self._binary_resolver = binary_resolver
        self._target_lookup = target_lookup
        # Validation sets left out of omr_ned.input.jsonl and the diagnostics:
        # no-music sets, whose empty targets VeroSim cannot score.
        self._exclude_val_sets = frozenset(exclude_val_sets)
        # Only the release self-score check, which scores labels against
        # themselves and reports each rejected one, turns this off.
        self._rejected_targets_fatal = rejected_targets_fatal
        self._runner = runner or DefaultSubprocessRunner()
        self._mode = normalize_verosim_mode(mode=mode, detail=detail)
        self._legacy_detail = detail
        self._jobs = int(jobs)
        self._kern_parser_binary = kern_parser_binary
        self._verosim_binary = verosim_binary
        self._failure_score = failure_score

    def compute(self, predictions_dir: str | Path) -> MetricResult:
        result = self.run(predictions_dir=predictions_dir)
        if result.skipped or result.summary is None:
            return MetricResult(
                name=self.name,
                metrics={},
                artifacts=_artifacts_from_result(result),
                raw_summary=result.summary,
            )
        return metric_result_from_summary(result.summary, artifacts=_artifacts_from_result(result))

    def run(self, *, predictions_dir: str | Path) -> VeroSimOMRNEDResult:
        """Convert validation prediction JSONL and score it with VeroSim."""
        predictions_dir = Path(predictions_dir)
        prediction_files = sorted(predictions_dir.glob("predictions.rank*.jsonl"))
        if not prediction_files:
            return VeroSimOMRNEDResult(
                skipped=True,
                reason=f"No validation prediction JSONL files found in {predictions_dir}",
                predictions_dir=predictions_dir,
            )
        prediction_diagnostics = prediction_diagnostics_from_jsonl(prediction_files)

        kern_parser = self._binary_resolver.resolve_kern_parser_binary(self._kern_parser_binary)
        if kern_parser.path is None:
            return VeroSimOMRNEDResult(
                skipped=True,
                reason=missing_binary_reason("kern_parser", kern_parser),
                predictions_dir=predictions_dir,
                prediction_diagnostics=prediction_diagnostics,
            )

        verosim = self._binary_resolver.resolve_verosim_binary(self._verosim_binary)
        if verosim.path is None:
            return VeroSimOMRNEDResult(
                skipped=True,
                reason=missing_binary_reason("verosim", verosim),
                predictions_dir=predictions_dir,
                prediction_diagnostics=prediction_diagnostics,
            )

        merged_jsonl = predictions_dir / "omr_ned.input.jsonl"
        converted_jsonl = predictions_dir / "omr_ned.humdrum.jsonl"
        records_jsonl = predictions_dir / "omr_ned.records.jsonl"
        summary_json = predictions_dir / "omr_ned.summary.json"
        diagnostics_summary_json = predictions_dir / DIAGNOSTICS_SUMMARY_FILENAME

        merge_jsonl_files(prediction_files, merged_jsonl, exclude_val_sets=self._exclude_val_sets)
        if self._exclude_val_sets and not _read_jsonl_records(merged_jsonl):
            return VeroSimOMRNEDResult(
                skipped=True,
                reason=(
                    "every prediction belongs to an excluded validation set: "
                    f"{sorted(self._exclude_val_sets)}"
                ),
                predictions_dir=predictions_dir,
                merged_jsonl=merged_jsonl,
                prediction_diagnostics=prediction_diagnostics,
            )
        if self._target_lookup is not None:
            try:
                replace_targets(merged_jsonl, self._target_lookup)
            except (StoredTargetError, OSError) as exc:
                return VeroSimOMRNEDResult(
                    skipped=True,
                    reason=f"stored OMR-NED targets unavailable: {exc}",
                    predictions_dir=predictions_dir,
                    merged_jsonl=merged_jsonl,
                    prediction_diagnostics=prediction_diagnostics,
                )

        convert_result = self._run_kern_parser(
            kern_parser_path=kern_parser.path,
            merged_jsonl=merged_jsonl,
            converted_jsonl=converted_jsonl,
        )
        conversion_failure = _conversion_failure(
            convert_result,
            merged_jsonl,
            converted_jsonl,
            rejected_targets_fatal=self._rejected_targets_fatal,
        )
        if conversion_failure is not None:
            return VeroSimOMRNEDResult(
                skipped=True,
                reason=conversion_failure,
                predictions_dir=predictions_dir,
                merged_jsonl=merged_jsonl,
                converted_jsonl=converted_jsonl,
                prediction_diagnostics=prediction_diagnostics,
            )

        verosim_result = self._run_verosim(
            verosim_path=verosim.path,
            converted_jsonl=converted_jsonl,
            records_jsonl=records_jsonl,
            summary_json=summary_json,
        )
        if verosim_result.returncode != 0:
            return VeroSimOMRNEDResult(
                skipped=True,
                reason=(
                    f"verosim failed with exit code {verosim_result.returncode}: "
                    f"{_stderr_text(verosim_result)}"
                ),
                predictions_dir=predictions_dir,
                merged_jsonl=merged_jsonl,
                converted_jsonl=converted_jsonl,
                records_jsonl=records_jsonl,
                summary_json=summary_json,
                prediction_diagnostics=prediction_diagnostics,
            )

        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        failure_mismatch = _failure_score_mismatch(summary, self._failure_score)
        if failure_mismatch is not None:
            return VeroSimOMRNEDResult(
                skipped=True,
                reason=failure_mismatch,
                predictions_dir=predictions_dir,
                merged_jsonl=merged_jsonl,
                converted_jsonl=converted_jsonl,
                records_jsonl=records_jsonl,
                summary_json=summary_json,
                prediction_diagnostics=prediction_diagnostics,
            )
        diagnostics_summary = build_validation_diagnostics_summary(
            prediction_files=prediction_files,
            manifest_files=sorted(predictions_dir.glob("manifest.rank*.json")),
            records_jsonl=records_jsonl,
            summary_json=summary_json,
            omr_ned_summary=summary,
            verosim_mode=self._mode,
            legacy_verosim_detail=self._legacy_detail,
            verosim_layout=VEROSIM_LAYOUT_SURFACE,
            exclude_val_sets=self._exclude_val_sets,
        )
        diagnostics_summary_json.write_text(
            json.dumps(diagnostics_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return VeroSimOMRNEDResult(
            skipped=False,
            reason=None,
            predictions_dir=predictions_dir,
            merged_jsonl=merged_jsonl,
            converted_jsonl=converted_jsonl,
            records_jsonl=records_jsonl,
            summary_json=summary_json,
            summary=summary,
            prediction_diagnostics=prediction_diagnostics,
            diagnostics_summary_json=diagnostics_summary_json,
            diagnostics_summary=diagnostics_summary,
        )

    def _run_kern_parser(
        self,
        *,
        kern_parser_path: Path,
        merged_jsonl: Path,
        converted_jsonl: Path,
    ) -> subprocess.CompletedProcess[bytes]:
        conversion_workers = max(1, self._jobs if self._jobs > 0 else 1)
        convert_cmd = [
            str(kern_parser_path),
            "serialize-jsonl",
            "--preset",
            "humdrum-kern",
            "--input-dialect",
            "transcoda-v2",
            "--workers",
            str(conversion_workers),
            "--text-field",
            "prediction",
            "--text-field",
            "target",
            "--id-field",
            "sample_id",
        ]
        with merged_jsonl.open("rb") as stdin, converted_jsonl.open("wb") as stdout:
            return self._runner.run(convert_cmd, stdin=stdin, stdout=stdout)

    def _run_verosim(
        self,
        *,
        verosim_path: Path,
        converted_jsonl: Path,
        records_jsonl: Path,
        summary_json: Path,
    ) -> subprocess.CompletedProcess[bytes]:
        verosim_cmd = [
            str(verosim_path),
            "--batch-jsonl",
            str(converted_jsonl),
            "--pred-field",
            "prediction",
            "--gt-field",
            "target",
            "--id-field",
            "sample_id",
            "--group-field",
            "val_set",
            "--dedupe-key",
            "val_set,sample_id",
            "--mode",
            self._mode,
            "--layout",
            VEROSIM_LAYOUT_SURFACE,
            "--summary-json",
            str(summary_json),
            "--jobs",
            str(resolve_verosim_jobs(self._jobs)),
        ]
        if self._failure_score is not None:
            verosim_cmd.extend(["--failure-score", _format_failure_score(self._failure_score)])
        with records_jsonl.open("wb") as stdout:
            return self._runner.run(verosim_cmd, stdin=None, stdout=stdout)


def build_validation_diagnostics_summary(
    *,
    prediction_files: list[Path],
    manifest_files: list[Path],
    records_jsonl: Path,
    summary_json: Path,
    omr_ned_summary: dict[str, Any],
    verosim_mode: str = DEFAULT_VEROSIM_MODE,
    legacy_verosim_detail: str | None = None,
    verosim_layout: str = VEROSIM_LAYOUT_SURFACE,
    exclude_val_sets: Collection[str] = (),
) -> dict[str, Any]:
    """Build a self-contained validation diagnostics summary from persisted artifacts."""
    predictions = [
        record
        for record in _read_deduped_prediction_records(prediction_files)
        if _prediction_group(record) not in exclude_val_sets
    ]
    prediction_by_key = {_prediction_record_key(record): record for record in predictions}
    verosim_records = _read_jsonl_records(records_jsonl)
    scopes = _diagnostic_scope_names(
        summary=omr_ned_summary,
        prediction_records=predictions,
        verosim_records=verosim_records,
    )

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "provenance": {
            "verosim_mode": verosim_mode,
            **(
                {"legacy_verosim_detail": legacy_verosim_detail}
                if legacy_verosim_detail is not None
                else {}
            ),
            "verosim_layout": verosim_layout,
            "artifact_files": {
                "predictions": [path.name for path in prediction_files],
                "manifests": [path.name for path in manifest_files],
                "omr_ned_records": records_jsonl.name,
                "omr_ned_summary": summary_json.name,
            },
            "prediction_manifests": _read_prediction_manifests(manifest_files),
        },
        "scopes": {
            scope: _build_scope_diagnostics(
                scope=scope,
                omr_ned_summary=omr_ned_summary,
                prediction_records=[
                    record
                    for record in predictions
                    if _record_belongs_to_scope(scope, _prediction_group(record))
                ],
                verosim_records=[
                    record
                    for record in verosim_records
                    if _record_belongs_to_scope(scope, _verosim_group(record))
                ],
                prediction_by_key=prediction_by_key,
            )
            for scope in scopes
        },
    }


def _build_scope_diagnostics(
    *,
    scope: str,
    omr_ned_summary: dict[str, Any],
    prediction_records: list[dict[str, Any]],
    verosim_records: list[dict[str, Any]],
    prediction_by_key: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    summary_stats = _summary_stats_for_scope(omr_ned_summary, scope)
    scored_records = [record for record in verosim_records if _is_scored_verosim_record(record)]
    failed_records = [record for record in verosim_records if not bool(record.get("ok", False))]

    sample_count = _coerce_int(summary_stats.get("count"))
    if sample_count is None:
        sample_count = len(verosim_records) if verosim_records else len(prediction_records)
    failure_count = _coerce_int(summary_stats.get("failures"))
    if failure_count is None:
        failure_count = len(failed_records)
    scored_count = _coerce_int(summary_stats.get("scored"))
    if scored_count is None:
        scored_count = max(0, sample_count - failure_count)

    return {
        "sample_count": sample_count,
        "scored_count": scored_count,
        "failure_count": failure_count,
        "failure_rate": _safe_divide(failure_count, sample_count),
        "omr_ned": _omr_ned_distribution(
            mean=_coerce_float(summary_stats.get("omr_ned")),
            records=verosim_records,
            failure_score=_coerce_float(summary_stats.get("failure_score")),
        ),
        "text_metrics": _weighted_text_metrics(prediction_records),
        "prediction_health": _prediction_health(prediction_records),
        "category_error_burden": _category_error_burden(
            scored_records=scored_records,
            scored_count=scored_count,
        ),
        "worst_samples": _worst_scored_samples(
            scored_records=scored_records,
            prediction_by_key=prediction_by_key,
            limit=10,
        ),
        "failed_samples": _failed_samples(
            failed_records=failed_records,
            prediction_by_key=prediction_by_key,
        ),
    }


def _read_deduped_prediction_records(input_paths: list[Path]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []
    for input_path in input_paths:
        for record in _read_jsonl_records(input_path):
            key = _prediction_record_key(record)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as in_file:
        for line in in_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _read_prediction_manifests(manifest_files: list[Path]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for manifest_file in manifest_files:
        if not manifest_file.is_file():
            continue
        manifests.append(
            {
                "path": manifest_file.name,
                "data": json.loads(manifest_file.read_text(encoding="utf-8")),
            }
        )
    return manifests


def _diagnostic_scope_names(
    *,
    summary: dict[str, Any],
    prediction_records: list[dict[str, Any]],
    verosim_records: list[dict[str, Any]],
) -> list[str]:
    scopes = ["overall"]
    by_group = summary.get("by_group", {})
    if isinstance(by_group, dict):
        scopes.extend(str(group) for group in by_group if str(group))

    extras = {
        group
        for group in (
            [_prediction_group(record) for record in prediction_records]
            + [_verosim_group(record) for record in verosim_records]
        )
        if group
    }
    scopes.extend(group for group in sorted(extras) if group not in scopes)
    return scopes


def _summary_stats_for_scope(summary: dict[str, Any], scope: str) -> dict[str, Any]:
    if scope == "overall":
        stats = summary.get("overall", {})
    else:
        by_group = summary.get("by_group", {})
        stats = by_group.get(scope, {}) if isinstance(by_group, dict) else {}
    return stats if isinstance(stats, dict) else {}


def _record_belongs_to_scope(scope: str, group: str) -> bool:
    return scope == "overall" or group == scope


def _prediction_record_key(record: dict[str, Any]) -> tuple[str, str]:
    return (_prediction_group(record), str(record.get("sample_id", "")))


def _verosim_record_key(record: dict[str, Any]) -> tuple[str, str]:
    return (_verosim_group(record), str(record.get("id", record.get("sample_id", ""))))


def _prediction_group(record: dict[str, Any]) -> str:
    return str(record.get("val_set") or "")


def _verosim_group(record: dict[str, Any]) -> str:
    return str(record.get("group") or record.get("val_set") or "")


def _is_scored_verosim_record(record: dict[str, Any]) -> bool:
    return bool(record.get("ok", False)) and _coerce_float(record.get("omr_ned")) is not None


def _omr_ned_distribution(
    *,
    mean: float | None,
    records: list[dict[str, Any]],
    failure_score: float | None,
) -> dict[str, float | None]:
    values: list[float] = []
    for record in records:
        value = _coerce_float(record.get("omr_ned"))
        if value is not None:
            values.append(value)
        elif not bool(record.get("ok", False)) and failure_score is not None:
            values.append(failure_score)
    ordered = sorted(values)
    return {
        "mean": mean,
        "median": _median(ordered),
        "p90": _nearest_rank_percentile(ordered, 0.90),
        "p95": _nearest_rank_percentile(ordered, 0.95),
        "max": ordered[-1] if ordered else None,
    }


def _weighted_text_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    cer_edit = cer_target = 0
    ser_edit = ser_target = 0
    ler_edit = ler_target = 0

    for record in records:
        prediction = str(record.get("prediction") or "")
        target = str(record.get("target") or "")

        target_chars = list(target)
        if target_chars:
            cer_edit += int(_levenshtein(list(prediction), target_chars))
            cer_target += len(target_chars)

        prediction_tokens = prediction.split()
        target_tokens = target.split()
        if target_tokens:
            ser_edit += int(_levenshtein(prediction_tokens, target_tokens))
            ser_target += len(target_tokens)

        if target:
            prediction_lines = prediction.split("\n")
            target_lines = target.split("\n")
            ler_edit += int(_levenshtein(prediction_lines, target_lines))
            ler_target += len(target_lines)

    return {
        "CER": 100.0 * _safe_divide(cer_edit, cer_target),
        "SER": 100.0 * _safe_divide(ser_edit, ser_target),
        "LER": 100.0 * _safe_divide(ler_edit, ler_target),
    }


def _prediction_health(records: list[dict[str, Any]]) -> dict[str, float | int]:
    sample_count = len(records)
    truncated_count = sum(1 for record in records if bool(record.get("prediction_truncated")))
    hit_max_length_count = sum(
        1 for record in records if bool(record.get("prediction_hit_max_length"))
    )
    return {
        "sample_count": sample_count,
        "truncated_count": truncated_count,
        "truncation_rate": _safe_divide(truncated_count, sample_count),
        "hit_max_length_count": hit_max_length_count,
        "hit_max_length_rate": _safe_divide(hit_max_length_count, sample_count),
    }


def _category_error_burden(
    *,
    scored_records: list[dict[str, Any]],
    scored_count: int,
) -> list[dict[str, float | int | str]]:
    category_costs: dict[str, float] = {}
    category_sample_counts: dict[str, int] = {}
    total_edit_cost = 0.0
    total_gt_symbols = 0.0

    for record in scored_records:
        distance = _coerce_float(record.get("distance"))
        if distance is not None:
            total_edit_cost += distance
        n_gt = _coerce_float(record.get("n_gt"))
        if n_gt is not None:
            total_gt_symbols += n_gt

        categories = record.get("edit_distances_dict")
        if not isinstance(categories, dict):
            continue
        for category, raw_cost in categories.items():
            cost = _coerce_float(raw_cost)
            if cost is None:
                continue
            category_name = str(category)
            category_costs[category_name] = category_costs.get(category_name, 0.0) + cost
            if cost > 0:
                category_sample_counts[category_name] = (
                    category_sample_counts.get(category_name, 0) + 1
                )

    if total_edit_cost <= 0:
        total_edit_cost = sum(category_costs.values())

    rows: list[dict[str, float | int | str]] = []
    for category, cost in sorted(category_costs.items(), key=lambda item: (-item[1], item[0])):
        sample_count = category_sample_counts.get(category, 0)
        rows.append(
            {
                "category": category,
                "cost": cost,
                "share": _safe_divide(cost, total_edit_cost),
                "cost_per_1k_gt_symbols": 1000.0 * _safe_divide(cost, total_gt_symbols),
                "sample_count": sample_count,
                "sample_rate": _safe_divide(sample_count, scored_count),
            }
        )
    return rows


def _worst_scored_samples(
    *,
    scored_records: list[dict[str, Any]],
    prediction_by_key: dict[tuple[str, str], dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(
        scored_records,
        key=lambda item: _coerce_float(item.get("omr_ned")) or 0.0,
        reverse=True,
    )[:limit]:
        prediction = prediction_by_key.get(_verosim_record_key(record), {})
        rows.append(
            {
                "val_set": _verosim_group(record),
                "sample_id": record.get("id", prediction.get("sample_id")),
                "source": record.get("source", prediction.get("source")),
                "omr_ned": _coerce_float(record.get("omr_ned")),
                "cer": _coerce_float(prediction.get("cer")),
                "distance": _coerce_float(record.get("distance")),
                "n_gt": _coerce_float(record.get("n_gt")),
                "prediction_truncated": (
                    bool(prediction.get("prediction_truncated")) if prediction else None
                ),
                "prediction_hit_max_length": (
                    bool(prediction.get("prediction_hit_max_length")) if prediction else None
                ),
                "top_categories": _top_categories(record, limit=5),
            }
        )
    return rows


def _failed_samples(
    *,
    failed_records: list[dict[str, Any]],
    prediction_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in failed_records:
        prediction = prediction_by_key.get(_verosim_record_key(record), {})
        rows.append(
            {
                "val_set": _verosim_group(record),
                "sample_id": record.get("id", prediction.get("sample_id")),
                "source": record.get("source", prediction.get("source")),
                "error": record.get("error"),
                "warnings": record.get("warnings", []),
                "prediction_truncated": (
                    bool(prediction.get("prediction_truncated")) if prediction else None
                ),
                "prediction_hit_max_length": (
                    bool(prediction.get("prediction_hit_max_length")) if prediction else None
                ),
            }
        )
    return rows


def _top_categories(record: dict[str, Any], *, limit: int) -> list[dict[str, float | str]]:
    categories = record.get("edit_distances_dict")
    if not isinstance(categories, dict):
        return []
    rows: list[dict[str, float | str]] = []
    for category, raw_cost in categories.items():
        cost = _coerce_float(raw_cost)
        if cost is not None and cost > 0:
            rows.append({"category": str(category), "cost": cost})
    return sorted(rows, key=lambda row: (-float(row["cost"]), str(row["category"])))[:limit]


def metric_result_from_summary(
    summary: dict[str, Any],
    *,
    artifacts: dict[str, Path] | None = None,
) -> MetricResult:
    metrics: dict[str, float] = {}
    per_group_metrics: dict[str, dict[str, float]] = {}
    failure_counts: dict[str, int] = {}
    sample_counts: dict[str, int] = {}

    overall = summary.get("overall", {})
    if isinstance(overall, dict):
        if "omr_ned" in overall:
            metrics[val_aggregate_metric("OMR_NED")] = float(overall["omr_ned"])
        if "failures" in overall:
            failure_counts["overall"] = int(overall["failures"])
        if "count" in overall:
            sample_counts["overall"] = int(overall["count"])

    by_group = summary.get("by_group", {})
    if isinstance(by_group, dict):
        for group, stats in by_group.items():
            if not isinstance(stats, dict) or not group:
                continue
            group_metrics: dict[str, float] = {}
            if "omr_ned" in stats:
                score = float(stats["omr_ned"])
                metrics[val_set_metric(str(group), "OMR_NED")] = score
                group_metrics["OMR_NED"] = score
            if "failures" in stats:
                failure_counts[str(group)] = int(stats["failures"])
            if "count" in stats:
                sample_counts[str(group)] = int(stats["count"])
            if group_metrics:
                per_group_metrics[str(group)] = group_metrics

    return MetricResult(
        name=VeroSimOMRNEDBackend.name,
        metrics=metrics,
        per_group_metrics=per_group_metrics,
        failure_counts=failure_counts,
        sample_counts=sample_counts,
        artifacts=artifacts or {},
        raw_summary=summary,
    )


def metric_payload_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    return metric_result_from_summary(summary).metrics


def metric_payload_from_result(result: VeroSimOMRNEDResult) -> dict[str, float]:
    if result.summary is None:
        return {}
    metric_result = metric_result_from_summary(result.summary)
    payload = dict(metric_result.metrics)
    payload.update(
        diagnostic_metric_payload(
            metric_result=metric_result,
            prediction_diagnostics=result.prediction_diagnostics,
        )
    )
    payload.update(diagnostics_summary_metric_payload(result.diagnostics_summary))
    return payload


def diagnostics_summary_metric_payload(
    diagnostics_summary: dict[str, Any] | None,
) -> dict[str, float]:
    if not isinstance(diagnostics_summary, dict):
        return {}
    scopes = diagnostics_summary.get("scopes", {})
    if not isinstance(scopes, dict):
        return {}

    payload: dict[str, float] = {}
    for scope, scope_summary in scopes.items():
        if not isinstance(scope_summary, dict):
            continue
        omr_ned = scope_summary.get("omr_ned", {})
        if not isinstance(omr_ned, dict):
            continue
        p90 = _coerce_float(omr_ned.get("p90"))
        if p90 is not None:
            payload[_scoped_metric_name(str(scope), "OMR_NED_p90")] = p90
    return payload


def diagnostic_metric_payload(
    *,
    metric_result: MetricResult,
    prediction_diagnostics: PredictionDiagnostics | None = None,
) -> dict[str, float]:
    payload: dict[str, float] = {}

    for scope, sample_count in metric_result.sample_counts.items():
        failures = metric_result.failure_counts.get(scope, 0)
        payload[_scoped_metric_name(scope, "OMR_NED_samples")] = float(sample_count)
        payload[_scoped_metric_name(scope, "OMR_NED_failures")] = float(failures)
        payload[_scoped_metric_name(scope, "OMR_NED_failure_rate")] = (
            float(failures) / float(sample_count) if sample_count > 0 else 0.0
        )

    if prediction_diagnostics is not None:
        for scope, sample_count in prediction_diagnostics.sample_counts.items():
            truncated = prediction_diagnostics.truncated_counts.get(scope, 0)
            payload[_scoped_metric_name(scope, "prediction_samples")] = float(sample_count)
            payload[_scoped_metric_name(scope, "prediction_truncated")] = float(truncated)
            payload[_scoped_metric_name(scope, "prediction_truncation_rate")] = (
                float(truncated) / float(sample_count) if sample_count > 0 else 0.0
            )

    return payload


def prediction_diagnostics_from_jsonl(input_paths: list[Path]) -> PredictionDiagnostics:
    sample_counts: dict[str, int] = {}
    truncated_counts: dict[str, int] = {}
    hit_max_length_counts: dict[str, int] = {}

    for record in _read_deduped_prediction_records(input_paths):
        val_set = _prediction_group(record)
        _increment_count(sample_counts, "overall")
        if val_set:
            _increment_count(sample_counts, val_set)

        if bool(record.get("prediction_truncated", False)):
            _increment_count(truncated_counts, "overall")
            if val_set:
                _increment_count(truncated_counts, val_set)

        if bool(record.get("prediction_hit_max_length", False)):
            _increment_count(hit_max_length_counts, "overall")
            if val_set:
                _increment_count(hit_max_length_counts, val_set)

    return PredictionDiagnostics(
        sample_counts=sample_counts,
        truncated_counts=truncated_counts,
        hit_max_length_counts=hit_max_length_counts,
    )


# kern_parser serialize-jsonl exits 1 when it rejected some rows (builds since
# ace2438; older pins exit 0). Either way it writes one result row per input,
# and VeroSim scores a row whose prediction was rejected at the failure score.
KERN_PARSER_REJECTED_ROWS_EXIT = 1
_ACCEPTED_CONVERSION_EXITS = (0, KERN_PARSER_REJECTED_ROWS_EXIT)


def _conversion_failure(
    result: subprocess.CompletedProcess[bytes],
    merged_jsonl: Path,
    converted_jsonl: Path,
    *,
    rejected_targets_fatal: bool = True,
) -> str | None:
    """Why the kern_parser conversion cannot be scored, or None when it can.

    Whatever the exit code (0 or 1), the result rows must match the inputs one
    to one by (val_set, sample_id), and every rejected row must be attributable
    to its prediction: target converted (``target_ok`` true), prediction
    rejected (``prediction_ok`` false). VeroSim then scores that page at the
    failure score, inside the mean. A rejected target is a broken ground truth
    no model should be scored against, so it is fatal; so is any other failed
    row, such as one the converter could not read.
    """
    failed = (
        f"kern_parser conversion failed with exit code {result.returncode}: {_stderr_text(result)}"
    )
    if result.returncode not in _ACCEPTED_CONVERSION_EXITS:
        return failed
    try:
        inputs = _read_jsonl_records(merged_jsonl)
        rows = _read_jsonl_records(converted_jsonl)
    except (OSError, ValueError) as exc:
        return f"{failed} (unreadable conversion: {exc})"
    expected, written = Counter(map(_row_identity, inputs)), Counter(map(_row_identity, rows))
    if expected != written:
        missing = sorted((expected - written).elements())[:5]
        extra = sorted((written - expected).elements())[:5]
        return (
            f"kern_parser wrote {len(rows)} rows that do not match its {len(inputs)} inputs "
            f"one to one (missing {missing}, unexpected {extra})"
        )
    rejected_targets: list[str] = []
    unattributed: list[str] = []
    for row in rows:
        if row.get("ok") is True:
            continue
        error = f"{row.get('sample_id')}: {row.get('target_error') or row.get('error')}"
        if row.get("target_ok") is False:
            rejected_targets.append(error)
        elif not (row.get("target_ok") is True and row.get("prediction_ok") is False):
            unattributed.append(f"{row.get('sample_id')}: {row.get('error')}")
    if rejected_targets and rejected_targets_fatal:
        return f"kern_parser rejected {len(rejected_targets)} ground-truth target(s): " + "; ".join(
            rejected_targets[:5]
        )
    if unattributed:
        return (
            f"kern_parser failed {len(unattributed)} row(s) without rejecting their "
            "prediction: " + "; ".join(unattributed[:5])
        )
    return None


def _row_identity(row: dict[str, Any]) -> str:
    return json.dumps([row.get("val_set"), row.get("sample_id")])


def replace_targets(merged_jsonl: Path, lookup: TargetLookup) -> None:
    """Rewrite each merged record's ``target`` with the lookup's stored label."""
    records = _read_jsonl_records(merged_jsonl)
    with merged_jsonl.open("w", encoding="utf-8") as out:
        for record in records:
            record["target"] = lookup.target(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_jsonl_files(
    input_paths: list[Path],
    output_path: Path,
    *,
    exclude_val_sets: Collection[str] = (),
) -> None:
    """Concatenate prediction JSONL files, dropping records of ``exclude_val_sets``."""
    with output_path.open("w", encoding="utf-8") as out:
        for input_path in input_paths:
            with input_path.open("r", encoding="utf-8") as in_file:
                for line in in_file:
                    if not line.strip():
                        continue
                    if exclude_val_sets and _prediction_group(json.loads(line)) in exclude_val_sets:
                        continue
                    out.write(line if line.endswith("\n") else f"{line}\n")


def _format_failure_score(value: float) -> str:
    return repr(float(value))


def _failure_score_mismatch(summary: dict[str, Any], expected: float | None) -> str | None:
    """Reason text when VeroSim's summary did not use the requested failure score."""
    if expected is None:
        return None
    overall = summary.get("overall") if isinstance(summary, dict) else None
    reported = overall.get("failure_score") if isinstance(overall, dict) else None
    if reported is None or float(reported) != float(expected):
        return (
            f"verosim summary failure_score {reported!r} does not match the requested "
            f"--failure-score {expected}"
        )
    return None


def resolve_verosim_jobs(configured_jobs: int) -> int:
    """Resolve VeroSim worker count without exceeding Slurm CPU allocation."""
    if configured_jobs > 0:
        return configured_jobs

    for env_var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        jobs = _positive_int_from_env(env_var)
        if jobs is not None:
            return jobs

    return configured_jobs


def missing_binary_reason(name: str, resolution: BinaryResolution) -> str:
    tried = ", ".join(str(path) for path in resolution.tried) or "no candidates"
    return f"{name} binary not found; tried: {tried}"


def resolve_binary(
    *,
    explicit: str | Path | None,
    env_var: str,
    binary_name: str,
    repo_candidates: tuple[Path, ...],
) -> BinaryResolution:
    return _resolve_binary(
        explicit=explicit,
        env_var=env_var,
        binary_name=binary_name,
        repo_candidates=repo_candidates,
    )


def _resolve_binary(
    *,
    explicit: str | Path | None,
    env_var: str,
    binary_name: str,
    repo_candidates: tuple[Path, ...],
) -> BinaryResolution:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_value = os.getenv(env_var)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    path_binary = shutil.which(binary_name)
    if path_binary:
        candidates.append(Path(path_binary))
    candidates.extend(repo_candidates)

    for candidate in candidates:
        if candidate.is_file():
            return BinaryResolution(path=candidate, tried=tuple(candidates))
    return BinaryResolution(path=None, tried=tuple(candidates))


def _artifacts_from_result(result: VeroSimOMRNEDResult) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    if result.merged_jsonl is not None:
        artifacts["merged_jsonl"] = result.merged_jsonl
    if result.converted_jsonl is not None:
        artifacts["converted_jsonl"] = result.converted_jsonl
    if result.records_jsonl is not None:
        artifacts["records_jsonl"] = result.records_jsonl
    if result.summary_json is not None:
        artifacts["summary_json"] = result.summary_json
    if result.diagnostics_summary_json is not None:
        artifacts["diagnostics_summary_json"] = result.diagnostics_summary_json
    return artifacts


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_divide(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _median(ordered_values: list[float]) -> float | None:
    if not ordered_values:
        return None
    midpoint = len(ordered_values) // 2
    if len(ordered_values) % 2:
        return ordered_values[midpoint]
    return (ordered_values[midpoint - 1] + ordered_values[midpoint]) / 2.0


def _nearest_rank_percentile(ordered_values: list[float], percentile: float) -> float | None:
    if not ordered_values:
        return None
    index = max(0, min(len(ordered_values) - 1, math.ceil(percentile * len(ordered_values)) - 1))
    return ordered_values[index]


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _positive_int_from_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed < 1:
        return None
    return parsed


def _scoped_metric_name(scope: str, metric_name: str) -> str:
    if scope == "overall":
        return val_aggregate_metric(metric_name)
    return val_set_metric(scope, metric_name)


def _stderr_text(result: subprocess.CompletedProcess[bytes]) -> str:
    stderr = result.stderr or b""
    return stderr.decode("utf-8", errors="replace").strip()
