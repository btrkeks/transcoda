"""Run-level path and artifact context resolution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from scripts.dataset_generation.dataset_generation.config import GenerationRunConfig


def build_run_id(run_started_at: float) -> str:
    """Build a stable run identifier from UTC timestamp + millisecond suffix."""
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(run_started_at))
    milliseconds = int((run_started_at - int(run_started_at)) * 1000)
    return f"{timestamp}Z-{milliseconds:03d}"


@dataclass(frozen=True)
class RunContext:
    """Resolved paths and metadata for a single generation run."""

    run_started_at: float
    run_id: str
    output_path: Path
    resume_dir: Path
    resume_db_path: Path
    staged_shards_dir: Path
    incomplete_marker_path: Path
    artifacts_root_path: Path
    dataset_runs_dir: Path
    run_artifacts_dir: Path
    latest_run_path: Path
    latest_quarantine_path: Path
    profile_path: Path
    quarantine_out_path: Path
    resolved_quarantine_in: str | None
    info_path: Path
    failure_summary_path: Path
    layout_summary_path: Path
    scheduler_summary_path: Path
    progress_path: Path


def build_run_context(config: GenerationRunConfig) -> RunContext:
    """Resolve all output and artifact paths for a run."""
    output_path = Path(config.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_started_at = time.time()
    run_id = build_run_id(run_started_at)

    artifacts_root_path = (
        Path(config.artifacts_out_dir) if config.artifacts_out_dir else output_path.parent / "_runs"
    )
    dataset_runs_dir = artifacts_root_path / output_path.name
    run_artifacts_dir = dataset_runs_dir / run_id
    latest_run_path = dataset_runs_dir / "latest_run.json"
    latest_quarantine_path = dataset_runs_dir / "latest_quarantined_files.json"
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)

    profile_path = Path(config.profile_out_dir) if config.profile_out_dir else run_artifacts_dir / "profile"
    if config.profile_enabled:
        profile_path.mkdir(parents=True, exist_ok=True)

    quarantine_out_path = (
        Path(config.quarantine_out) if config.quarantine_out is not None else run_artifacts_dir / "quarantined_files.json"
    )
    quarantine_out_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_quarantine_in = config.quarantine_in
    if resolved_quarantine_in is None:
        if config.quarantine_out is not None and quarantine_out_path.exists():
            resolved_quarantine_in = str(quarantine_out_path)
        elif latest_quarantine_path.exists():
            resolved_quarantine_in = str(latest_quarantine_path)

    return RunContext(
        run_started_at=run_started_at,
        run_id=run_id,
        output_path=output_path,
        resume_dir=output_path / ".resume",
        resume_db_path=output_path / ".resume" / "manifest.sqlite",
        staged_shards_dir=output_path / ".resume" / "shards",
        incomplete_marker_path=output_path / "INCOMPLETE",
        artifacts_root_path=artifacts_root_path,
        dataset_runs_dir=dataset_runs_dir,
        run_artifacts_dir=run_artifacts_dir,
        latest_run_path=latest_run_path,
        latest_quarantine_path=latest_quarantine_path,
        profile_path=profile_path,
        quarantine_out_path=quarantine_out_path,
        resolved_quarantine_in=resolved_quarantine_in,
        info_path=run_artifacts_dir / "info.json",
        failure_summary_path=run_artifacts_dir / "failure_summary.json",
        layout_summary_path=run_artifacts_dir / "layout_summary.json",
        scheduler_summary_path=run_artifacts_dir / "scheduler_summary.json",
        progress_path=run_artifacts_dir / "progress.json",
    )
