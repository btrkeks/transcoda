from pathlib import Path

from scripts.dataset_generation.dataset_generation.progress import (
    build_generation_progress_snapshot,
    format_generation_progress_line,
    write_progress_snapshot_atomic,
)


def test_build_generation_progress_snapshot_computes_eta_and_acceptance_rate():
    snapshot = build_generation_progress_snapshot(
        run_id="run-1",
        status="running",
        started_at=100.0,
        last_update_at=140.0,
        target_samples=50,
        accepted_target=None,
        max_scheduled_tasks=80,
        accepted_samples=20,
        submitted_tasks=40,
        terminal_completed_tasks=25,
        in_flight_tasks=3,
        stop_condition=None,
        remaining_schedulable_files=12,
        files_early_stopped=1,
        files_with_at_least_one_success=4,
        truncation_counts={"attempted": 10, "rescued": 4, "failed": 6},
        failure_reason_counts={"multi_page": 7, "render_fit": 2},
        output_dir="data/datasets/train",
        run_artifacts_dir="data/datasets/_runs/train/run-1",
        progress_path="data/datasets/_runs/train/run-1/progress.json",
    )

    assert snapshot["elapsed_seconds"] == 40.0
    assert snapshot["accepted_samples_per_second"] == 0.5
    assert snapshot["eta_seconds"] == 60.0
    assert snapshot["acceptance_rate"] == 0.8
    assert snapshot["failure_reason_counts"]["multi_page"] == 7
    assert snapshot["failure_reason_counts"]["system_band_below_min"] == 0
    assert snapshot["failure_reason_counts"]["system_band_above_max"] == 0
    assert snapshot["failure_reason_counts"]["system_band_truncation_exhausted"] == 0


def test_build_generation_progress_snapshot_returns_null_eta_for_zero_acceptance():
    snapshot = build_generation_progress_snapshot(
        run_id="run-1",
        status="running",
        started_at=100.0,
        last_update_at=140.0,
        target_samples=50,
        accepted_target=None,
        max_scheduled_tasks=None,
        accepted_samples=0,
        submitted_tasks=10,
        terminal_completed_tasks=0,
        in_flight_tasks=5,
        stop_condition=None,
        remaining_schedulable_files=9,
        files_early_stopped=0,
        files_with_at_least_one_success=0,
        truncation_counts={},
        failure_reason_counts={},
        output_dir="out",
        run_artifacts_dir="artifacts",
        progress_path="artifacts/progress.json",
    )

    assert snapshot["accepted_samples_per_second"] == 0.0
    assert snapshot["eta_seconds"] is None
    assert snapshot["acceptance_rate"] == 0.0


def test_write_progress_snapshot_atomic_and_format_line(tmp_path):
    progress_path = tmp_path / "progress.json"
    snapshot = build_generation_progress_snapshot(
        run_id="run-1",
        status="completed",
        started_at=10.0,
        last_update_at=20.0,
        target_samples=10,
        accepted_target=10,
        max_scheduled_tasks=20,
        accepted_samples=10,
        submitted_tasks=12,
        terminal_completed_tasks=12,
        in_flight_tasks=0,
        stop_condition="accepted_target_reached",
        remaining_schedulable_files=0,
        files_early_stopped=2,
        files_with_at_least_one_success=5,
        truncation_counts={"attempted": 2, "rescued": 1, "failed": 1},
        failure_reason_counts={"timeout": 1},
        output_dir="out",
        run_artifacts_dir="artifacts",
        progress_path=str(progress_path),
    )

    write_progress_snapshot_atomic(progress_path, snapshot)

    assert progress_path.exists()
    line = format_generation_progress_line(snapshot)
    assert "[progress]" in line
    assert "accepted=10/10" in line
    assert "task_cap=20" in line
    assert "stop=accepted_target_reached" in line
    assert "throughput=1.00 accepted/s" in line
    assert "system_band_below_min=0" in line
