from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.baselines.contracts import GroundTruthManifestRecord, PredictionRecord
from pipelines.baselines.io import sha256_file
from pipelines.baselines.scoring import (
    ScoreCandidate,
    VeroSimBatchResult,
    build_score_report,
    join_prediction_ground_truth,
    write_score_report,
)


def _prediction(sample_id: str, *, error: str | None = None) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample_id,
        image_sha256="a" * 64,
        manifest_digest="b" * 64,
        model_id="model",
        model_revision="c" * 40,
        native_format="bekern",
        output_text="4C<t>4c<b>*-<t>*-" if error is None else "",
        output_tokens=[] if error else ["4C", "<t>", "4c"],
        termination_reason="error" if error else "eos",
        ended_with_eos=False if error else True,
        timing_s=1.0,
        error=error,
    )


def _ground_truth(sample_id: str, path: str) -> GroundTruthManifestRecord:
    return GroundTruthManifestRecord(
        schema_version=1,
        sample_id=sample_id,
        kern_path=path,
        kern_sha256="d" * 64,
    )


def test_ground_truth_join_requires_exact_sample_set() -> None:
    with pytest.raises(ValueError, match="missing ground truth.*b"):
        join_prediction_ground_truth([_prediction("a"), _prediction("b")], [_ground_truth("a", "a.krn")])
    with pytest.raises(ValueError, match="extra ground truth.*b"):
        join_prediction_ground_truth([_prediction("a")], [_ground_truth("a", "a.krn"), _ground_truth("b", "b.krn")])


def test_mixed_failure_report_is_cohort_complete_and_deterministic(tmp_path: Path) -> None:
    candidates = [
        ScoreCandidate("ok", "smt", "**kern\n4c\n*-\n", "**kern\n4c\n*-\n"),
        ScoreCandidate("inference", "smt", None, "**kern\n4c\n*-\n", "inference", "OOM"),
        ScoreCandidate("normalization", "smt", None, "**kern\n4c\n*-\n", "normalization", "bad"),
        ScoreCandidate("backend", "smt", "**kern\nbad\n*-\n", "**kern\n4c\n*-\n"),
    ]
    batch = VeroSimBatchResult(
        records=(
            {"id": "ok", "group": "smt", "ok": True, "omr_ned": 0.25},
            {"id": "backend", "group": "smt", "ok": False, "omr_ned": None},
        ),
        summary={
            "overall": {
                "count": 2,
                "scored": 1,
                "failures": 1,
                "omr_ned": 50.125,
                "failure_score": 100.0,
            },
            "by_group": {},
        },
        command=("/verosim", "--batch-jsonl", "input"),
        binary_sha256="e" * 64,
    )

    report = build_score_report(candidates, batch=batch)
    assert report.summary["cohort_count"] == 4
    assert report.summary["scored_count"] == 1
    assert report.summary["inference_failures"] == 1
    assert report.summary["normalization_failures"] == 1
    assert report.summary["backend_failures"] == 1
    assert report.summary["coverage"] == 0.25
    assert report.summary["omr_ned"] == 75.0625

    first = write_score_report(tmp_path / "one", report)
    second = write_score_report(tmp_path / "two", report)
    assert first.report_json.read_bytes() == second.report_json.read_bytes()
    assert first.samples_jsonl.read_bytes() == second.samples_jsonl.read_bytes()
    assert len(first.samples_jsonl.read_text(encoding="utf-8").splitlines()) == 4
    report_payload = json.loads(first.report_json.read_text(encoding="utf-8"))
    assert report_payload["samples"]["path"] == "report.samples.jsonl"
    assert report_payload["samples"]["sha256"] == sha256_file(first.samples_jsonl)


def test_report_rejects_backend_record_count_or_summary_mismatch() -> None:
    candidate = ScoreCandidate("a", "smt", "**kern\n4c\n*-\n", "**kern\n4c\n*-\n")
    with pytest.raises(ValueError, match="backend records"):
        build_score_report(
            [candidate],
            batch=VeroSimBatchResult(
                records=(),
                summary={"overall": {"count": 1, "failure_score": 100.0}},
                command=("verosim",),
                binary_sha256="e" * 64,
            ),
        )


def test_score_report_json_has_no_operational_timestamp(tmp_path: Path) -> None:
    candidate = ScoreCandidate("a", "smt", "**kern\n4c\n*-\n", "**kern\n4c\n*-\n")
    report = build_score_report(
        [candidate],
        batch=VeroSimBatchResult(
            records=({"id": "a", "group": "smt", "ok": True, "omr_ned": 0.0},),
            summary={
                "overall": {
                    "count": 1,
                    "scored": 1,
                    "failures": 0,
                    "omr_ned": 0.0,
                    "failure_score": 100.0,
                }
            },
            command=("verosim",),
            binary_sha256="e" * 64,
        ),
    )
    artifacts = write_score_report(tmp_path, report)
    assert "time" not in json.loads(artifacts.report_json.read_text(encoding="utf-8"))
