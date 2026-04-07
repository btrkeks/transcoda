"""Reporting helpers for dataset generation artifacts."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from scripts.dataset_generation.dataset_generation.base import GenerationStats
from scripts.dataset_generation.dataset_generation.config import (
    FailurePolicySettings,
    GenerationRunConfig,
)
from scripts.dataset_generation.dataset_generation.run_context import RunContext
from scripts.dataset_generation.dataset_generation.worker_models import WorkerInitConfig


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def top_worker_stages(
    worker_stage_stats: dict[str, dict[str, float | int]],
) -> list[dict[str, float | str]]:
    total_worker_ms = float(
        sum(float(stats.get("total_ms", 0.0)) for stats in worker_stage_stats.values())
    )
    ranked = sorted(
        worker_stage_stats.items(),
        key=lambda item: float(item[1].get("total_ms", 0.0)),
        reverse=True,
    )
    top = []
    for stage_name, stats in ranked[:5]:
        stage_total_ms = float(stats.get("total_ms", 0.0))
        share = (stage_total_ms / total_worker_ms) if total_worker_ms > 0.0 else 0.0
        top.append(
            {
                "stage": stage_name,
                "total_ms": stage_total_ms,
                "p95_ms": float(stats.get("p95_ms", 0.0)),
                "time_share": share,
            }
        )
    return top


def write_primary_artifacts(
    *,
    run_context: RunContext,
    quarantine_summary: dict[str, object],
    failure_summary: dict[str, object],
    layout_summary: dict[str, object],
    scheduler_summary: dict[str, object],
    write_latest_quarantine: bool,
) -> None:
    write_json(run_context.quarantine_out_path, quarantine_summary)
    if write_latest_quarantine:
        write_json(run_context.latest_quarantine_path, quarantine_summary)
    write_json(run_context.failure_summary_path, failure_summary)
    write_json(run_context.layout_summary_path, layout_summary)
    write_json(run_context.scheduler_summary_path, scheduler_summary)


def build_info_summary(
    *,
    config: GenerationRunConfig,
    run_context: RunContext,
    worker_config: WorkerInitConfig,
    configured_start_method: str,
    failure_policy: FailurePolicySettings,
    runtime_seconds: dict[str, float],
    total_samples: int,
    total_size_gb: float,
    resolved_quarantine_in: str | None,
    resolved_kern_dirs: list[Path],
    variant_policy_summary: dict[str, object] | None = None,
    prefilter_summary: dict[str, object] | None = None,
    resumable_state: dict[str, object] | None = None,
) -> dict[str, object]:
    effective_variants_per_file = int(
        (variant_policy_summary or {}).get("mean_variants_per_file", config.variants_per_file)
    )
    adaptive_variants_enabled = bool(
        (variant_policy_summary or {}).get("enabled", config.adaptive_variants_enabled)
    )
    return {
        "generation_timestamp": time.time(),
        "run_started_at": run_context.run_started_at,
        "total_samples": total_samples,
        "total_size_gb": round(total_size_gb, 4),
        "runtime_seconds": runtime_seconds,
        "artifacts": {
            "run_id": run_context.run_id,
            "run_artifacts_dir": str(run_context.run_artifacts_dir),
            "info": str(run_context.info_path),
            "quarantine_summary": str(run_context.quarantine_out_path),
            "latest_quarantine_summary": (
                str(run_context.latest_quarantine_path) if config.quarantine_out is None else None
            ),
            "failure_summary": str(run_context.failure_summary_path),
            "layout_summary": str(run_context.layout_summary_path),
            "scheduler_summary": str(run_context.scheduler_summary_path),
            "progress": str(run_context.progress_path),
            "resume_db": str(run_context.resume_db_path),
            "incomplete_marker": str(run_context.incomplete_marker_path),
        },
        "generation_config": {
            "dataset_preset": config.dataset_preset,
            "kern_dirs": [str(path) for path in resolved_kern_dirs],
            "data_spec_path": config.data_spec_path,
            "strict_data_spec": config.strict_data_spec,
            "num_workers": config.num_workers,
            "requested_num_samples": config.requested_num_samples,
            "effective_num_samples": config.effective_num_samples,
            "target_accepted_samples": config.target_accepted_samples,
            "effective_target_accepted_samples": config.effective_target_accepted_samples,
            "max_scheduled_tasks": config.max_scheduled_tasks,
            "effective_max_scheduled_tasks": config.effective_max_scheduled_tasks,
            "variants_per_file": effective_variants_per_file,
            "adaptive_variants_enabled": adaptive_variants_enabled,
            "overflow_truncation_enabled": config.overflow_truncation_enabled,
            "overflow_truncation_max_trials": config.overflow_truncation_max_trials,
            "variant_policy_summary": variant_policy_summary,
            "start_method": configured_start_method,
            "failure_policy": failure_policy.name,
            "resume_mode": config.resume_mode,
            "task_timeout_seconds": failure_policy.task_timeout_seconds,
            "max_task_retries_timeout": failure_policy.max_task_retries_timeout,
            "max_task_retries_expired": failure_policy.max_task_retries_expired,
            "quarantine_in": resolved_quarantine_in,
            "quarantine_out": str(run_context.quarantine_out_path),
            "artifacts_out_dir": str(run_context.artifacts_root_path),
            "profile_enabled": config.profile_enabled,
            "profile_out_dir": str(run_context.profile_path) if config.profile_enabled else None,
            "profile_log_every": config.profile_log_every,
            "profile_sample_limit": config.profile_sample_limit,
            "profile_capture_per_sample": config.profile_capture_per_sample,
            "progress_enabled": config.progress_enabled,
            "progress_update_interval_seconds": config.progress_update_interval_seconds,
            "prefilter_min_non_empty_lines": config.prefilter_min_non_empty_lines,
            "prefilter_max_non_empty_lines": config.prefilter_max_non_empty_lines,
            "prefilter_min_measure_count": config.prefilter_min_measure_count,
            "prefilter_max_measure_count": config.prefilter_max_measure_count,
            "prefilter_summary": prefilter_summary,
            "resumable_state": resumable_state,
            **worker_config.to_dict(),
        },
    }


def write_info_and_latest_pointer(
    *,
    run_context: RunContext,
    info_summary: dict[str, object],
) -> None:
    write_json(run_context.info_path, info_summary)
    write_json(
        run_context.latest_run_path,
        {
            "run_id": run_context.run_id,
            "run_artifacts_dir": str(run_context.run_artifacts_dir),
            "info_path": str(run_context.info_path),
            "updated_at": time.time(),
        },
    )


def write_profile_artifacts(
    *,
    run_context: RunContext,
    profile_report: dict[str, object],
    info_summary: dict[str, object],
    stats: GenerationStats,
    generation_elapsed_s: float,
    total_samples: int,
    argv: list[str],
) -> None:
    worker_stage_stats = profile_report.get("worker_stage_stats", {})
    orchestrator_stats = profile_report.get("orchestrator_stats", {})
    stage_stats_path = run_context.profile_path / "stage_stats.json"
    orchestrator_stats_path = run_context.profile_path / "orchestrator_stats.json"
    manifest_path = run_context.profile_path / "run_manifest.json"
    benchmark_summary_path = run_context.profile_path / "benchmark_summary.json"
    timeout_events_path = run_context.profile_path / "timeout_events.jsonl"
    process_expired_events_path = run_context.profile_path / "process_expired_events.jsonl"
    quarantine_events_path = run_context.profile_path / "quarantine_events.jsonl"
    timeout_events = profile_report.get("timeout_events", [])
    process_expired_events = profile_report.get("process_expired_events", [])
    quarantine_events = profile_report.get("quarantine_events", [])
    quarantine_summary = profile_report.get("quarantine", {})

    write_json(stage_stats_path, worker_stage_stats)
    write_json(orchestrator_stats_path, orchestrator_stats)
    write_jsonl(timeout_events_path, timeout_events)
    write_jsonl(process_expired_events_path, process_expired_events)
    write_jsonl(quarantine_events_path, quarantine_events)
    write_json(run_context.quarantine_out_path, quarantine_summary)

    write_json(
        manifest_path,
        {
            "timestamp": time.time(),
            "argv": argv,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "output_dir": str(run_context.output_path),
            "generation_config": info_summary["generation_config"],
        },
    )
    throughput = (total_samples / generation_elapsed_s) if generation_elapsed_s > 0.0 else 0.0
    benchmark_summary = {
        "total_samples": total_samples,
        "throughput_samples_per_second": throughput,
        "runtime_seconds": info_summary["runtime_seconds"],
        "failure_counts": {
            "overflows": stats.overflows,
            "invalid": stats.invalid,
            "rejected_sparse": stats.rejected_sparse,
            "rejected_render_fit": stats.rejected_render_fit,
            "errors": stats.errors,
            "timeouts": stats.timeouts,
            "expired_workers": stats.expired_workers,
        },
        "outcome_counts": profile_report.get("outcome_counts", {}),
        "failure_code_counts": profile_report.get("failure_code_counts", {}),
        "retry_counts": profile_report.get("retry_counts", {}),
        "quarantine": quarantine_summary,
        "top_worker_stages": top_worker_stages(worker_stage_stats),
        "artifacts": {
            "stage_stats": str(stage_stats_path),
            "orchestrator_stats": str(orchestrator_stats_path),
            "run_manifest": str(manifest_path),
            "timeout_events": str(timeout_events_path),
            "process_expired_events": str(process_expired_events_path),
            "quarantine_events": str(quarantine_events_path),
            "quarantined_files": str(run_context.quarantine_out_path),
            "failure_summary": str(run_context.failure_summary_path),
        },
    }
    write_json(benchmark_summary_path, benchmark_summary)
