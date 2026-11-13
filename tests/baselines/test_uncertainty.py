from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.baselines.uncertainty import (
    PageObservation,
    _markdown_summary,
    align_observations,
    analyze,
    cluster_bootstrap,
    density_tertiles,
    load_score_report,
    page_bootstrap,
    spearman_correlation,
)


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(payload) + "\n" for payload in payloads), encoding="utf-8")


def test_load_score_report_applies_bounded_failure_policy(tmp_path: Path) -> None:
    report = tmp_path / "report.jsonl"
    _write_jsonl(
        report,
        [
            {"sample_id": "a", "status": "scored", "omr_ned": 0.25},
            {"sample_id": "b", "status": "backend_failure", "omr_ned": None},
        ],
    )

    order, scores, statuses = load_score_report(report, failure_score=1.0)

    assert order == ["a", "b"]
    assert scores == {"a": 0.25, "b": 1.0}
    assert statuses["b"] == "backend_failure"


def test_alignment_filters_prefix_and_requires_exact_order() -> None:
    reports = {
        "A": (
            ["keep", "polish__drop"],
            {"keep": 0.1, "polish__drop": 0.2},
            {"keep": "scored", "polish__drop": "scored"},
        ),
        "B": (
            ["keep", "polish__drop"],
            {"keep": 0.3, "polish__drop": 0.4},
            {"keep": "scored", "polish__drop": "scored"},
        ),
    }

    observations = align_observations(
        reports,
        complexity={"keep": 12},
        document_ids={"keep": "doc"},
        exclude_prefixes=("polish__",),
    )

    assert len(observations) == 1
    assert observations[0].scores == {"A": 0.1, "B": 0.3}

    reversed_reports = {
        **reports,
        "B": (list(reversed(reports["B"][0])), reports["B"][1], reports["B"][2]),
    }
    with pytest.raises(ValueError, match="not in exact"):
        align_observations(
            reversed_reports,
            complexity={"keep": 12},
            document_ids={"keep": "doc"},
            exclude_prefixes=("polish__",),
        )


def test_page_bootstrap_is_paired_and_deterministic() -> None:
    scores = np.asarray([[0.0, 0.2, 0.4], [0.1, 0.3, 0.5]])

    first = page_bootstrap(scores, seed=7, n_resamples=1_000, level=0.95)
    second = page_bootstrap(scores, seed=7, n_resamples=1_000, level=0.95)

    assert np.array_equal(first.estimates, second.estimates)
    assert np.allclose(first.estimates[:, 1] - first.estimates[:, 0], 0.1)


def test_cluster_bootstrap_resamples_whole_documents() -> None:
    scores = np.asarray([[0.0, 0.0, 1.0], [0.2, 0.2, 0.8]])

    result = cluster_bootstrap(
        scores,
        ["doc-a", "doc-a", "doc-b"],
        seed=11,
        n_resamples=1_000,
        level=0.95,
    )

    possible_first_model_means = {0.0, 1 / 3, 1.0}
    assert set(np.unique(result.estimates[:, 0])).issubset(possible_first_model_means)


def test_density_tertiles_and_spearman_are_model_independent() -> None:
    n_gt = np.asarray([10, 20, 30, 40, 50, 60], dtype=float)

    assignments, metadata = density_tertiles(n_gt)

    assert assignments.tolist() == [0, 0, 1, 1, 2, 2]
    assert [item["count"] for item in metadata] == [2, 2, 2]
    assert spearman_correlation(n_gt, n_gt) == pytest.approx(1.0)
    assert spearman_correlation(n_gt, -n_gt) == pytest.approx(-1.0)


def test_analysis_reports_paired_differences_within_density_tertiles() -> None:
    observations = [
        PageObservation(
            sample_id=f"sample-{index}",
            document_id=f"doc-{index // 2}",
            n_gt=10 * (index + 1),
            scores={"Baseline": 0.4 + index / 100, "Ours": 0.2 + index / 100},
            statuses={"Baseline": "scored", "Ours": "scored"},
        )
        for index in range(9)
    ]

    result, _ = analyze(
        observations,
        model_labels=["Baseline", "Ours"],
        reference_model="Ours",
        seed=3,
        n_resamples=200,
        level=0.95,
    )

    for tertile in result["density"]["tertiles"]:
        difference = tertile["paired_differences"]["Ours minus Baseline"]
        assert difference["point"] == pytest.approx(-0.2)
        assert difference["ci_low"] == pytest.approx(-0.2)
        assert difference["ci_high"] == pytest.approx(-0.2)

    summary = _markdown_summary(result, ["Baseline", "Ours"])
    assert "Negative differences favor Ours." in summary
    assert "### Paired Ours differences within density tertiles" in summary
