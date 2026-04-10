"""Run-level path resolution for the production rewrite."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


def build_run_id(run_started_at: float) -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(run_started_at))
    milliseconds = int((run_started_at - int(run_started_at)) * 1000)
    return f"{timestamp}Z-{milliseconds:03d}"


@dataclass(frozen=True)
class RunContext:
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
    info_path: Path
    progress_path: Path
    quarantined_sources_path: Path
    timeout_events_path: Path
    process_expired_events_path: Path
    verovio_events_path: Path
    augmentation_events_path: Path
    crash_samples_dir: Path


def build_run_context(
    *,
    output_dir: str | Path,
    artifacts_out_dir: str | Path | None = None,
) -> RunContext:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_started_at = time.time()
    run_id = build_run_id(run_started_at)

    artifacts_root_path = (
        Path(artifacts_out_dir).expanduser().resolve()
        if artifacts_out_dir is not None
        else output_path.parent / "_runs"
    )
    dataset_runs_dir = artifacts_root_path / output_path.name
    run_artifacts_dir = dataset_runs_dir / run_id
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)

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
        latest_run_path=dataset_runs_dir / "latest_run.json",
        info_path=run_artifacts_dir / "info.json",
        progress_path=run_artifacts_dir / "progress.json",
        quarantined_sources_path=run_artifacts_dir / "quarantined_sources.json",
        timeout_events_path=run_artifacts_dir / "timeout_events.jsonl",
        process_expired_events_path=run_artifacts_dir / "process_expired_events.jsonl",
        verovio_events_path=run_artifacts_dir / "verovio_events.jsonl",
        augmentation_events_path=run_artifacts_dir / "augmentation_events.jsonl",
        crash_samples_dir=run_artifacts_dir / "crash_samples",
    )
