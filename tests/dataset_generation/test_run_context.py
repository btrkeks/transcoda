from pathlib import Path

from scripts.dataset_generation.dataset_generation.config import GenerationRunConfig
from scripts.dataset_generation.dataset_generation.run_context import build_run_context


def test_build_run_context_uses_run_scoped_artifacts_layout(tmp_path):
    output_dir = tmp_path / "datasets" / "train_medium"
    config = GenerationRunConfig(
        kern_dirs=(),
        output_dir=str(output_dir),
    )

    context = build_run_context(config)

    assert context.output_path == output_dir
    assert context.resume_dir == output_dir / ".resume"
    assert context.resume_db_path == output_dir / ".resume" / "manifest.sqlite"
    assert context.staged_shards_dir == output_dir / ".resume" / "shards"
    assert context.incomplete_marker_path == output_dir / "INCOMPLETE"
    assert context.run_artifacts_dir.parent == context.dataset_runs_dir
    assert context.info_path.parent == context.run_artifacts_dir
    assert context.failure_summary_path.parent == context.run_artifacts_dir
    assert context.layout_summary_path.parent == context.run_artifacts_dir
    assert context.scheduler_summary_path.parent == context.run_artifacts_dir
    assert context.progress_path.parent == context.run_artifacts_dir
    assert context.quarantine_out_path.parent == context.run_artifacts_dir
    assert context.latest_run_path == context.dataset_runs_dir / "latest_run.json"
    assert context.latest_quarantine_path == context.dataset_runs_dir / "latest_quarantined_files.json"


def test_build_run_context_prefers_explicit_quarantine_in(tmp_path):
    output_dir = tmp_path / "datasets" / "train_medium"
    explicit_quarantine = tmp_path / "manual_quarantine.json"
    explicit_quarantine.write_text("[]", encoding="utf-8")
    config = GenerationRunConfig(
        kern_dirs=(),
        output_dir=str(output_dir),
        quarantine_in=str(explicit_quarantine),
    )

    context = build_run_context(config)

    assert context.resolved_quarantine_in == str(explicit_quarantine)


def test_build_run_context_uses_latest_quarantine_fallback(tmp_path):
    output_dir = tmp_path / "datasets" / "train_medium"
    dataset_runs_dir = output_dir.parent / "_runs" / output_dir.name
    dataset_runs_dir.mkdir(parents=True, exist_ok=True)
    latest_quarantine = dataset_runs_dir / "latest_quarantined_files.json"
    latest_quarantine.write_text("[]", encoding="utf-8")
    config = GenerationRunConfig(
        kern_dirs=(),
        output_dir=str(output_dir),
    )

    context = build_run_context(config)

    assert context.resolved_quarantine_in == str(latest_quarantine)


def test_build_run_context_uses_explicit_profile_out_dir(tmp_path):
    output_dir = tmp_path / "datasets" / "train_medium"
    profile_out_dir = tmp_path / "custom_profile_dir"
    config = GenerationRunConfig(
        kern_dirs=(),
        output_dir=str(output_dir),
        profile_enabled=True,
        profile_out_dir=str(profile_out_dir),
    )

    context = build_run_context(config)

    assert context.profile_path == Path(profile_out_dir)
    assert context.profile_path.exists()
