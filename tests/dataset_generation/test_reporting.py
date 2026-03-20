import json

from scripts.dataset_generation.dataset_generation.config import (
    FailurePolicySettings,
    GenerationRunConfig,
)
from scripts.dataset_generation.dataset_generation.reporting import (
    build_info_summary,
    write_info_and_latest_pointer,
    write_primary_artifacts,
)
from scripts.dataset_generation.dataset_generation.run_context import build_run_context
from scripts.dataset_generation.dataset_generation.worker_models import WorkerInitConfig


def test_write_primary_artifacts_writes_quarantine_failure_and_latest(tmp_path):
    config = GenerationRunConfig(kern_dirs=(), output_dir=str(tmp_path / "datasets" / "train"))
    context = build_run_context(config)

    quarantine_summary = {"files": [], "files_count": 0}
    failure_summary = {"failed_samples_total": 1}
    layout_summary = {"accepted_samples_total": 0, "systems_histogram": {}}
    scheduler_summary = {"stop_condition": "task_budget_exhausted"}

    write_primary_artifacts(
        run_context=context,
        quarantine_summary=quarantine_summary,
        failure_summary=failure_summary,
        layout_summary=layout_summary,
        scheduler_summary=scheduler_summary,
        write_latest_quarantine=True,
    )

    assert context.quarantine_out_path.exists()
    assert context.failure_summary_path.exists()
    assert context.layout_summary_path.exists()
    assert context.scheduler_summary_path.exists()
    assert context.latest_quarantine_path.exists()


def test_build_info_summary_and_latest_pointer_include_run_artifacts(tmp_path):
    config = GenerationRunConfig(kern_dirs=(), output_dir=str(tmp_path / "datasets" / "train"))
    context = build_run_context(config)
    worker_config = WorkerInitConfig(image_width=1050, image_height=1485)
    policy = FailurePolicySettings(
        name="throughput",
        task_timeout_seconds=5,
        max_task_retries_timeout=0,
        max_task_retries_expired=0,
    )
    info_summary = build_info_summary(
        config=config,
        run_context=context,
        worker_config=worker_config,
        configured_start_method="spawn",
        failure_policy=policy,
        runtime_seconds={"generation": 1.0, "save_to_disk": 0.5, "total": 1.5},
        total_samples=42,
        total_size_gb=0.1234,
        resolved_quarantine_in=None,
        resolved_kern_dirs=[tmp_path / "a", tmp_path / "b"],
        variant_policy_summary={"enabled": False, "policy": "fixed"},
        prefilter_summary={
            "min_non_empty_lines": 513,
            "max_non_empty_lines": 900,
            "min_measure_count": 12,
            "max_measure_count": 20,
            "retained_file_count": 10,
        },
    )

    assert info_summary["artifacts"]["run_artifacts_dir"] == str(context.run_artifacts_dir)
    assert info_summary["artifacts"]["layout_summary"] == str(context.layout_summary_path)
    assert info_summary["artifacts"]["scheduler_summary"] == str(context.scheduler_summary_path)
    assert info_summary["artifacts"]["progress"] == str(context.progress_path)
    assert info_summary["artifacts"]["resume_db"] == str(context.resume_db_path)
    assert info_summary["artifacts"]["incomplete_marker"] == str(context.incomplete_marker_path)
    assert info_summary["generation_config"]["dataset_preset"] is None
    assert info_summary["generation_config"]["kern_dirs"] == [
        str(tmp_path / "a"),
        str(tmp_path / "b"),
    ]
    assert info_summary["generation_config"]["adaptive_variants_enabled"] is False
    assert info_summary["generation_config"]["target_accepted_samples"] is None
    assert info_summary["generation_config"]["max_scheduled_tasks"] is None
    assert info_summary["generation_config"]["overflow_truncation_enabled"] is True
    assert info_summary["generation_config"]["overflow_truncation_max_trials"] == 24
    assert info_summary["generation_config"]["resume_mode"] == "auto"
    assert info_summary["generation_config"]["render_pedals_probability"] == 0.20
    assert info_summary["generation_config"]["variant_policy_summary"] == {
        "enabled": False,
        "policy": "fixed",
    }
    assert info_summary["generation_config"]["prefilter_summary"] == {
        "min_non_empty_lines": 513,
        "max_non_empty_lines": 900,
        "min_measure_count": 12,
        "max_measure_count": 20,
        "retained_file_count": 10,
    }
    assert info_summary["generation_config"]["prefilter_max_non_empty_lines"] is None
    assert info_summary["generation_config"]["prefilter_min_measure_count"] is None
    assert info_summary["generation_config"]["prefilter_max_measure_count"] is None
    assert info_summary["generation_config"]["progress_enabled"] is True
    assert info_summary["generation_config"]["progress_update_interval_seconds"] == 30
    assert info_summary["generation_config"]["resumable_state"] is None

    write_info_and_latest_pointer(run_context=context, info_summary=info_summary)
    assert context.info_path.exists()
    assert context.latest_run_path.exists()

    latest_payload = json.loads(context.latest_run_path.read_text(encoding="utf-8"))
    assert latest_payload["run_id"] == context.run_id
    assert latest_payload["info_path"] == str(context.info_path)
