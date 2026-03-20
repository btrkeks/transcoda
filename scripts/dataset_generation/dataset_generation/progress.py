"""Live progress helpers for dataset generation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

MAJOR_PROGRESS_FAILURE_KEYS: tuple[str, ...] = (
    "multi_page",
    "system_band_below_min",
    "system_band_above_max",
    "system_band_truncation_exhausted",
    "timeout",
    "process_expired",
    "invalid_kern",
    "render_fit",
    "sparse_render",
)


def build_generation_progress_snapshot(
    *,
    run_id: str,
    status: str,
    started_at: float,
    last_update_at: float,
    target_samples: int,
    accepted_target: int | None,
    max_scheduled_tasks: int | None,
    accepted_samples: int,
    submitted_tasks: int,
    terminal_completed_tasks: int,
    in_flight_tasks: int,
    stop_condition: str | None,
    remaining_schedulable_files: int,
    files_early_stopped: int,
    files_with_at_least_one_success: int,
    truncation_counts: dict[str, int],
    failure_reason_counts: dict[str, int],
    output_dir: str,
    run_artifacts_dir: str,
    progress_path: str,
) -> dict[str, Any]:
    elapsed_seconds = max(0.0, float(last_update_at) - float(started_at))
    accepted_samples_per_second = (
        float(accepted_samples) / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    )
    effective_target = (
        int(accepted_target)
        if accepted_target is not None
        else int(target_samples)
    )
    remaining_samples = max(0, effective_target - int(accepted_samples))
    if remaining_samples == 0:
        eta_seconds: float | None = 0.0
    elif accepted_samples_per_second > 0.0:
        eta_seconds = float(remaining_samples) / accepted_samples_per_second
    else:
        eta_seconds = None

    acceptance_rate = (
        float(accepted_samples) / float(terminal_completed_tasks)
        if terminal_completed_tasks > 0
        else 0.0
    )
    target_progress_rate = (
        float(accepted_samples) / float(effective_target) if effective_target > 0 else 0.0
    )
    major_failure_counts = {
        key: int(failure_reason_counts.get(key, 0))
        for key in MAJOR_PROGRESS_FAILURE_KEYS
    }

    return {
        "version": 1,
        "run_id": run_id,
        "status": status,
        "started_at": float(started_at),
        "last_update_at": float(last_update_at),
        "target_samples": int(target_samples),
        "accepted_target": None if accepted_target is None else int(accepted_target),
        "max_scheduled_tasks": None if max_scheduled_tasks is None else int(max_scheduled_tasks),
        "accepted_samples": int(accepted_samples),
        "target_progress_rate": target_progress_rate,
        "submitted_tasks": int(submitted_tasks),
        "terminal_completed_tasks": int(terminal_completed_tasks),
        "in_flight_tasks": int(in_flight_tasks),
        "stop_condition": stop_condition,
        "remaining_schedulable_files": int(remaining_schedulable_files),
        "files_early_stopped": int(files_early_stopped),
        "files_with_at_least_one_success": int(files_with_at_least_one_success),
        "acceptance_rate": acceptance_rate,
        "elapsed_seconds": elapsed_seconds,
        "accepted_samples_per_second": accepted_samples_per_second,
        "eta_seconds": eta_seconds,
        "truncation": {
            "attempted": int(truncation_counts.get("attempted", 0)),
            "rescued": int(truncation_counts.get("rescued", 0)),
            "failed": int(truncation_counts.get("failed", 0)),
        },
        "failure_reason_counts": major_failure_counts,
        "paths": {
            "output_dir": output_dir,
            "run_artifacts_dir": run_artifacts_dir,
            "progress_path": progress_path,
        },
    }


def format_duration_for_progress(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_generation_progress_line(snapshot: dict[str, Any]) -> str:
    target_samples = int(snapshot.get("target_samples", 0))
    accepted_target = snapshot.get("accepted_target")
    display_target = (
        int(accepted_target)
        if accepted_target is not None
        else target_samples
    )
    accepted_samples = int(snapshot.get("accepted_samples", 0))
    target_progress_rate = float(snapshot.get("target_progress_rate", 0.0))
    max_scheduled_tasks = snapshot.get("max_scheduled_tasks")
    terminal_completed_tasks = int(snapshot.get("terminal_completed_tasks", 0))
    submitted_tasks = int(snapshot.get("submitted_tasks", 0))
    in_flight_tasks = int(snapshot.get("in_flight_tasks", 0))
    stop_condition = snapshot.get("stop_condition") or "running"
    remaining_schedulable_files = int(snapshot.get("remaining_schedulable_files", 0))
    files_early_stopped = int(snapshot.get("files_early_stopped", 0))
    files_with_success = int(snapshot.get("files_with_at_least_one_success", 0))
    acceptance_rate = float(snapshot.get("acceptance_rate", 0.0))
    elapsed_seconds = float(snapshot.get("elapsed_seconds", 0.0))
    throughput = float(snapshot.get("accepted_samples_per_second", 0.0))
    eta_seconds = snapshot.get("eta_seconds")
    truncation = snapshot.get("truncation", {})
    failure_reason_counts = snapshot.get("failure_reason_counts", {})
    failure_summary = " ".join(
        f"{key}={int(failure_reason_counts.get(key, 0))}"
        for key in MAJOR_PROGRESS_FAILURE_KEYS
    )
    return (
        "[progress] "
        f"status={snapshot.get('status', 'unknown')} "
        f"accepted={accepted_samples:,}/{display_target:,} "
        f"({target_progress_rate:.2%}) "
        f"task_cap="
        f"{'unbounded' if max_scheduled_tasks is None else f'{int(max_scheduled_tasks):,}'} "
        f"completed_tasks={terminal_completed_tasks:,} "
        f"submitted={submitted_tasks:,} "
        f"in_flight={in_flight_tasks:,} "
        f"stop={stop_condition} "
        f"schedulable_files={remaining_schedulable_files:,} "
        f"files_with_success={files_with_success:,} "
        f"files_early_stopped={files_early_stopped:,} "
        f"accept_rate={acceptance_rate:.2%} "
        f"elapsed={format_duration_for_progress(elapsed_seconds)} "
        f"throughput={throughput:.2f} accepted/s "
        f"eta={format_duration_for_progress(eta_seconds)} "
        f"truncation={int(truncation.get('attempted', 0))}/"
        f"{int(truncation.get('rescued', 0))}/"
        f"{int(truncation.get('failed', 0))} "
        f"failures={failure_summary}"
    )


def write_progress_snapshot_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        Path(temp_path).replace(path)
    finally:
        temp_file = Path(temp_path)
        if temp_file.exists():
            temp_file.unlink()
