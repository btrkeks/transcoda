import json
from pathlib import Path

from scripts.dataset_generation.pipeline_drop_report import build_report


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_report_aggregates_stage_and_generation_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    interim_root = tmp_path / "data" / "interim"
    base = interim_root / "train" / "pdmx"

    # Raw stage with basename collision (sub1/a.xml + sub2/a.xml).
    _write(base / "0_raw_xml" / "sub1" / "a.xml")
    _write(base / "0_raw_xml" / "sub2" / "a.xml")

    # Convert/filter/normalize.
    _write(base / "1_kern_conversions" / "a.krn")
    _write(base / "2_filtered" / "a.krn")
    _write(base / "3_normalized" / "a.krn")

    # Source-root count for extract input.
    _write(tmp_path / "data" / "raw" / "pdmx" / "train" / "x1.mxl")
    _write(tmp_path / "data" / "raw" / "pdmx" / "train" / "x2.mxl")

    stage_stats_dir = tmp_path / "stats"
    stage_stats_dir.mkdir(parents=True, exist_ok=True)
    (stage_stats_dir / "train_pdmx_filter.json").write_text(
        json.dumps({"rejection_by_filter": {"rhythm": 1}, "total_failed": 1}),
        encoding="utf-8",
    )
    (stage_stats_dir / "train_pdmx_normalize.json").write_text(
        json.dumps({"error_count": 0, "error_types": {}}),
        encoding="utf-8",
    )

    train_output_dir = tmp_path / "data" / "datasets" / "train_medium"
    run_artifacts_dir = tmp_path / "data" / "datasets" / "_runs" / "train_medium" / "run123"
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)
    (run_artifacts_dir / "failure_summary.json").write_text(
        json.dumps(
            {
                "requested_tasks": 3,
                "successful_samples": 2,
                "failed_samples_total": 1,
                "failure_reason_counts": {"render_fit": 1},
            }
        ),
        encoding="utf-8",
    )
    (run_artifacts_dir / "info.json").write_text(
        json.dumps(
            {
                "generation_config": {
                    "variants_per_file": 3,
                    "adaptive_variants_enabled": True,
                    "variant_policy_summary": {
                        "enabled": True,
                        "policy": "line_count_v1",
                        "mean_variants_per_file": 2.75,
                    },
                    "overflow_truncation_enabled": True,
                    "overflow_truncation_max_trials": 12,
                }
            }
        ),
        encoding="utf-8",
    )
    latest_run_path = tmp_path / "data" / "datasets" / "_runs" / "train_medium" / "latest_run.json"
    latest_run_path.parent.mkdir(parents=True, exist_ok=True)
    latest_run_path.write_text(
        json.dumps({"run_artifacts_dir": str(run_artifacts_dir)}),
        encoding="utf-8",
    )

    report = build_report(
        pipeline="train",
        split="train",
        datasets=["pdmx"],
        interim_root=interim_root,
        stage_stats_dir=stage_stats_dir,
        data_spec_path="config/data_spec.json",
        workers=4,
        run_id="run-1",
        train_output_dir=train_output_dir,
    )

    assert report["schema_version"] == "1.0"
    assert report["run_id"] == "run-1"

    dataset_payload = report["datasets"][0]
    assert dataset_payload["dataset"] == "pdmx"

    stages = {item["stage"]: item for item in dataset_payload["stages"]}
    assert stages["extract"]["input_count"] == 2
    assert stages["extract"]["output_count"] == 2
    assert stages["convert"]["input_count"] == 2
    assert stages["convert"]["output_count"] == 1
    assert stages["convert"]["dropped_count"] == 1

    assert stages["filter"]["details"]["rejection_by_filter"] == {"rhythm": 1}
    assert stages["normalize"]["details"]["error_count"] == 0

    assert dataset_payload["identity_checks"]["id_mode"] == "relative_path"
    assert dataset_payload["identity_checks"]["collisions_detected"] == 1

    assert report["generation"]["input_files_total"] == 1
    assert report["generation"]["variants_per_file"] == 3
    assert report["generation"]["adaptive_variants_enabled"] is True
    assert report["generation"]["variant_policy_summary"]["policy"] == "line_count_v1"
    assert report["generation"]["overflow_truncation_enabled"] is True
    assert report["generation"]["overflow_truncation_max_trials"] == 12
    assert report["generation"]["requested_tasks"] == 3
    assert report["generation"]["failed_samples_total"] == 1


def test_build_report_warns_when_generation_pointer_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    interim_root = tmp_path / "data" / "interim"
    base = interim_root / "train" / "grandstaff"
    _write(base / "0_raw_ekern" / "000001.ekern")
    _write(base / "1_kern_conversions" / "000001.krn")
    _write(base / "2_filtered" / "000001.krn")
    _write(base / "3_normalized" / "000001.krn")

    report = build_report(
        pipeline="train",
        split="train",
        datasets=["grandstaff"],
        interim_root=interim_root,
        stage_stats_dir=None,
        data_spec_path="config/data_spec.json",
        workers=4,
        run_id="run-2",
        train_output_dir=tmp_path / "data" / "datasets" / "train_medium",
    )

    assert any("Missing generation latest run pointer" in warning for warning in report["warnings"])
