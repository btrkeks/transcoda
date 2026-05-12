#!/usr/bin/env python3
"""Aggregate per-stage dataset drop-off statistics into a single JSON report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageSnapshot:
    name: str
    root: Path
    pattern: str
    count: int
    ids: set[str]


def _count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def _relative_ids(root: Path, pattern: str) -> set[str]:
    if not root.exists():
        return set()
    ids: set[str] = set()
    for path in root.rglob(pattern):
        rel = path.relative_to(root)
        ids.add(str(rel.with_suffix("")))
    return ids


def _basename_collisions(root: Path, pattern: str) -> tuple[int, list[dict[str, Any]]]:
    if not root.exists():
        return 0, []
    counts = Counter(path.stem for path in root.rglob(pattern))
    duplicates = {name: count for name, count in counts.items() if count > 1}
    collision_count = int(sum(count - 1 for count in duplicates.values()))
    examples = [
        {"basename": name, "count": int(count)}
        for name, count in sorted(duplicates.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    return collision_count, examples


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_payload(
    *,
    stage: str,
    input_count: int,
    output_count: int,
    input_ids: set[str],
    output_ids: set[str],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dropped_count = max(0, int(input_count) - int(output_count))
    drop_rate = (dropped_count / input_count) if input_count > 0 else 0.0
    dropped_relative_ids = sorted(input_ids - output_ids)
    payload = {
        "stage": stage,
        "input_count": int(input_count),
        "output_count": int(output_count),
        "dropped_count": int(dropped_count),
        "drop_rate": float(drop_rate),
        "details": {
            "dropped_relative_ids_count": len(dropped_relative_ids),
            "dropped_relative_id_examples": dropped_relative_ids[:20],
        },
    }
    if details:
        payload["details"].update(details)
    return payload


def _resolve_raw_stage(base: Path) -> tuple[Path, str]:
    raw_xml = base / "0_raw_xml"
    raw_ekern = base / "0_raw_ekern"
    if raw_xml.exists():
        return raw_xml, "*.xml"
    if raw_ekern.exists():
        return raw_ekern, "*.ekern"
    # Default to XML semantics for absent/misconfigured directories.
    return raw_xml, "*.xml"


def _load_generation_summary(train_output_dir: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    dataset_runs_dir = train_output_dir.parent / "_runs" / train_output_dir.name
    latest_run_path = dataset_runs_dir / "latest_run.json"
    latest_run_payload = _read_json_if_exists(latest_run_path)
    if latest_run_payload is None:
        warnings.append(f"Missing generation latest run pointer: {latest_run_path}")
        return {
            "input_files_total": 0,
            "variants_per_file": 0,
            "adaptive_variants_enabled": False,
            "variant_policy_summary": None,
            "overflow_truncation_enabled": False,
            "overflow_truncation_max_trials": 0,
            "requested_tasks": 0,
            "successful_samples": 0,
            "failed_samples_total": 0,
            "failure_reason_counts": {},
            "truncation": {"attempted": 0, "rescued": 0, "failed": 0},
        }, warnings

    run_artifacts_dir_raw = latest_run_payload.get("run_artifacts_dir")
    if not isinstance(run_artifacts_dir_raw, str) or not run_artifacts_dir_raw:
        warnings.append("latest_run.json is missing run_artifacts_dir")
        return {
            "input_files_total": 0,
            "variants_per_file": 0,
            "adaptive_variants_enabled": False,
            "variant_policy_summary": None,
            "overflow_truncation_enabled": False,
            "overflow_truncation_max_trials": 0,
            "requested_tasks": 0,
            "successful_samples": 0,
            "failed_samples_total": 0,
            "failure_reason_counts": {},
            "truncation": {"attempted": 0, "rescued": 0, "failed": 0},
        }, warnings

    run_artifacts_dir = Path(run_artifacts_dir_raw)
    failure_summary_path = run_artifacts_dir / "failure_summary.json"
    info_path = run_artifacts_dir / "info.json"

    failure_summary = _read_json_if_exists(failure_summary_path)
    if failure_summary is None:
        warnings.append(f"Missing generation failure summary: {failure_summary_path}")
        failure_summary = {}

    info_summary = _read_json_if_exists(info_path)
    if info_summary is None:
        warnings.append(f"Missing generation info summary: {info_path}")
        info_summary = {}

    generation_config = info_summary.get("generation_config", {})
    variants_per_file = int(generation_config.get("variants_per_file", 0))
    adaptive_variants_enabled = bool(generation_config.get("adaptive_variants_enabled", False))
    variant_policy_summary = generation_config.get("variant_policy_summary")
    overflow_truncation_enabled = bool(generation_config.get("overflow_truncation_enabled", False))
    overflow_truncation_max_trials = int(generation_config.get("overflow_truncation_max_trials", 0))
    if not isinstance(variant_policy_summary, dict):
        variant_policy_summary = None
    truncation = failure_summary.get("truncation", {})
    if not isinstance(truncation, dict):
        truncation = {}

    return {
        "input_files_total": 0,  # populated later from normalized stage totals
        "variants_per_file": variants_per_file,
        "adaptive_variants_enabled": adaptive_variants_enabled,
        "variant_policy_summary": variant_policy_summary,
        "overflow_truncation_enabled": overflow_truncation_enabled,
        "overflow_truncation_max_trials": overflow_truncation_max_trials,
        "requested_tasks": int(failure_summary.get("requested_tasks", 0)),
        "successful_samples": int(failure_summary.get("successful_samples", 0)),
        "failed_samples_total": int(failure_summary.get("failed_samples_total", 0)),
        "failure_reason_counts": failure_summary.get("failure_reason_counts", {}),
        "truncation": {
            "attempted": int(truncation.get("attempted", 0)),
            "rescued": int(truncation.get("rescued", 0)),
            "failed": int(truncation.get("failed", 0)),
        },
        "run_artifacts_dir": str(run_artifacts_dir),
    }, warnings


def build_report(
    *,
    pipeline: str,
    split: str,
    datasets: list[str],
    interim_root: Path,
    stage_stats_dir: Path | None,
    data_spec_path: str,
    workers: int,
    run_id: str,
    train_output_dir: Path,
) -> dict[str, Any]:
    warnings: list[str] = []
    datasets_payload: list[dict[str, Any]] = []
    stage_drop_records: list[tuple[str, str, int]] = []
    total_normalized_files = 0

    raw_source_map: dict[str, tuple[Path, str]] = {
        "pdmx": (Path("data/raw/pdmx/train"), "*.mxl"),
        "musetrainer": (Path("data/raw/musetrainer"), "*.mxl"),
    }

    for dataset in datasets:
        base = interim_root / split / dataset
        raw_root, raw_pattern = _resolve_raw_stage(base)
        convert_root = base / "1_kern_conversions"
        filter_root = base / "2_filtered"
        normalize_root = base / "3_normalized"

        raw_snapshot = StageSnapshot(
            name="extract_output",
            root=raw_root,
            pattern=raw_pattern,
            count=_count_files(raw_root, raw_pattern),
            ids=_relative_ids(raw_root, raw_pattern),
        )
        convert_snapshot = StageSnapshot(
            name="convert",
            root=convert_root,
            pattern="*.krn",
            count=_count_files(convert_root, "*.krn"),
            ids=_relative_ids(convert_root, "*.krn"),
        )
        filter_snapshot = StageSnapshot(
            name="filter",
            root=filter_root,
            pattern="*.krn",
            count=_count_files(filter_root, "*.krn"),
            ids=_relative_ids(filter_root, "*.krn"),
        )
        normalize_snapshot = StageSnapshot(
            name="normalize",
            root=normalize_root,
            pattern="*.krn",
            count=_count_files(normalize_root, "*.krn"),
            ids=_relative_ids(normalize_root, "*.krn"),
        )

        extract_input_count = raw_snapshot.count
        extract_details: dict[str, Any] = {
            "source_count_approximation": "raw_stage_output",
            "source_path": str(raw_root),
        }
        if dataset in raw_source_map:
            source_root, source_pattern = raw_source_map[dataset]
            if source_root.exists():
                extract_input_count = _count_files(source_root, source_pattern)
                extract_details = {
                    "source_count_approximation": "raw_source_scan",
                    "source_path": str(source_root),
                }
            else:
                warnings.append(
                    f"Dataset '{dataset}' source root not found for extract input counting: {source_root}"
                )

        filter_details: dict[str, Any] = {}
        normalize_details: dict[str, Any] = {}

        if stage_stats_dir is not None:
            filter_stats_path = stage_stats_dir / f"{split}_{dataset}_filter.json"
            normalize_stats_path = stage_stats_dir / f"{split}_{dataset}_normalize.json"

            filter_stats = _read_json_if_exists(filter_stats_path)
            if filter_stats is None:
                warnings.append(f"Missing filter stats JSON: {filter_stats_path}")
            else:
                filter_details["rejection_by_filter"] = filter_stats.get("rejection_by_filter", {})
                filter_details["filter_total_failed"] = int(filter_stats.get("total_failed", 0))

            normalize_stats = _read_json_if_exists(normalize_stats_path)
            if normalize_stats is None:
                warnings.append(f"Missing normalization stats JSON: {normalize_stats_path}")
            else:
                normalize_details["error_count"] = int(normalize_stats.get("error_count", 0))
                normalize_details["error_types"] = normalize_stats.get("error_types", {})

        extract_collision_count, extract_collision_examples = _basename_collisions(
            raw_snapshot.root, raw_snapshot.pattern
        )
        convert_collision_count, convert_collision_examples = _basename_collisions(
            convert_snapshot.root, convert_snapshot.pattern
        )
        filter_collision_count, filter_collision_examples = _basename_collisions(
            filter_snapshot.root, filter_snapshot.pattern
        )
        normalize_collision_count, normalize_collision_examples = _basename_collisions(
            normalize_snapshot.root, normalize_snapshot.pattern
        )

        collisions_detected = (
            extract_collision_count
            + convert_collision_count
            + filter_collision_count
            + normalize_collision_count
        )
        collision_examples = [
            *[
                {"stage": "extract", **example}
                for example in extract_collision_examples
            ],
            *[
                {"stage": "convert", **example}
                for example in convert_collision_examples
            ],
            *[
                {"stage": "filter", **example}
                for example in filter_collision_examples
            ],
            *[
                {"stage": "normalize", **example}
                for example in normalize_collision_examples
            ],
        ][:20]
        if collisions_detected > 0:
            warnings.append(
                f"Dataset '{dataset}' has {collisions_detected} basename collision(s) across stage directories"
            )

        stage_entries = [
            _stage_payload(
                stage="extract",
                input_count=extract_input_count,
                output_count=raw_snapshot.count,
                input_ids=raw_snapshot.ids,
                output_ids=raw_snapshot.ids,
                details=extract_details,
            ),
            _stage_payload(
                stage="convert",
                input_count=raw_snapshot.count,
                output_count=convert_snapshot.count,
                input_ids=raw_snapshot.ids,
                output_ids=convert_snapshot.ids,
            ),
            _stage_payload(
                stage="filter",
                input_count=convert_snapshot.count,
                output_count=filter_snapshot.count,
                input_ids=convert_snapshot.ids,
                output_ids=filter_snapshot.ids,
                details=filter_details,
            ),
            _stage_payload(
                stage="normalize",
                input_count=filter_snapshot.count,
                output_count=normalize_snapshot.count,
                input_ids=filter_snapshot.ids,
                output_ids=normalize_snapshot.ids,
                details=normalize_details,
            ),
        ]

        for stage_entry in stage_entries:
            stage_drop_records.append(
                (dataset, stage_entry["stage"], int(stage_entry["dropped_count"]))
            )

        total_normalized_files += normalize_snapshot.count
        datasets_payload.append(
            {
                "split": split,
                "dataset": dataset,
                "stages": stage_entries,
                "identity_checks": {
                    "id_mode": "relative_path",
                    "collisions_detected": int(collisions_detected),
                    "collision_examples": collision_examples,
                },
            }
        )

    generation_payload, generation_warnings = _load_generation_summary(train_output_dir)
    warnings.extend(generation_warnings)
    generation_payload["input_files_total"] = int(total_normalized_files)

    if stage_drop_records:
        largest_drop_dataset, largest_drop_stage, largest_drop_count = max(
            stage_drop_records, key=lambda item: item[2]
        )
    else:
        largest_drop_dataset, largest_drop_stage, largest_drop_count = ("", "", 0)

    return {
        "schema_version": "1.0",
        "pipeline": pipeline,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "script": "scripts/dataset_generation/run_full_pipeline.sh",
            "workers": workers,
            "data_spec_path": data_spec_path,
        },
        "datasets": datasets_payload,
        "generation": generation_payload,
        "summary": {
            "largest_drop_stage": largest_drop_stage,
            "largest_drop_count": int(largest_drop_count),
            "largest_drop_dataset": largest_drop_dataset,
            "total_stage_drops": int(sum(item[2] for item in stage_drop_records)),
        },
        "warnings": sorted(set(warnings)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", default="train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--datasets", default="pdmx,musetrainer,grandstaff")
    parser.add_argument("--interim_root", default="data/interim")
    parser.add_argument("--stage_stats_dir", default=None)
    parser.add_argument("--data_spec_path", default="config/data_spec.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--train_output_dir", default="data/datasets/train_medium")
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--latest_path", default=None)
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    stage_stats_dir = Path(args.stage_stats_dir) if args.stage_stats_dir else None
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(
        pipeline=args.pipeline,
        split=args.split,
        datasets=datasets,
        interim_root=Path(args.interim_root),
        stage_stats_dir=stage_stats_dir,
        data_spec_path=args.data_spec_path,
        workers=args.workers,
        run_id=args.run_id,
        train_output_dir=Path(args.train_output_dir),
    )

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.latest_path:
        latest_path = Path(args.latest_path)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_payload = {
            "run_id": args.run_id,
            "latest_report_path": str(report_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
