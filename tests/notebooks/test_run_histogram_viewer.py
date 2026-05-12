from __future__ import annotations

import json

import matplotlib.pyplot as plt

from notebooks.run_histogram_viewer import (
    _build_spine_system_heatmap_payload,
    _collect_plot_panels,
    _collect_rejection_summary,
    collect_histograms,
    plot_histograms,
)


def test_collect_rejection_summary_computes_hard_and_clean_gate_rates() -> None:
    info = {
        "snapshot": {
            "accepted_samples": 200,
            "rejected_samples": 53,
            "failure_reason_counts": {
                "discarded_after_target": 24,
                "timeout": 48,
                "truncation_exhausted": 5,
            },
            "augmentation_outcome_counts": {
                "clean_gate_rejected": 50,
                "fully_augmented": 150,
            },
        }
    }

    summary = _collect_rejection_summary(info)

    assert summary is not None
    assert summary["accepted_samples"] == 200
    assert summary["rejected_samples"] == 53
    assert summary["attempted_samples"] == 277

    rates = {item["label"]: item for item in summary["rates"]}
    assert rates["Hard rejects / attempts"]["numerator"] == 53
    assert rates["Hard rejects / attempts"]["denominator"] == 277
    assert rates["Hard rejects / attempts"]["rate"] == 53 / 277
    assert rates["Hard rejects / resolved"]["rate"] == 53 / 253
    assert rates["Clean-gate rejects / accepted"]["rate"] == 50 / 200


def test_collect_rejection_summary_reads_finalization_snapshot() -> None:
    info = {
        "finalization": {
            "snapshot": {
                "accepted_samples": 10,
                "rejected_samples": 2,
                "failure_reason_counts": {
                    "timeout": 2,
                },
            }
        }
    }

    summary = _collect_rejection_summary(info)

    assert summary is not None
    rates = {item["label"]: item for item in summary["rates"]}
    assert rates["Hard rejects / attempts"]["rate"] == 2 / 12
    assert rates["Hard rejects / resolved"]["rate"] == 2 / 12


def test_build_spine_system_heatmap_payload_uses_spine_classes() -> None:
    info = {
        "snapshot": {
            "accepted_system_histogram": {
                "1": {"2": 3, "4": 1},
                "2": {"3": 2},
                "3_plus": {"6": 5},
            }
        }
    }

    payload = _build_spine_system_heatmap_payload(info)

    assert payload is not None
    assert payload["spine_labels"] == ["1", "2", "3_plus"]
    assert payload["system_values"] == [1, 2, 3, 4, 5, 6, 7]
    assert payload["grid"].tolist() == [
        [0, 0, 0],
        [3, 0, 0],
        [0, 2, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 5],
        [0, 0, 0],
    ]


def test_collect_plot_panels_includes_planner_summary_with_partial_counters() -> None:
    info = {
        "snapshot": {
            "requested_target_bucket_histogram": {"1": 4, "3": 2},
            "accepted_system_histogram": {
                "1": {"1": 1},
                "2": {"3": 2},
            },
        }
    }

    panels = _collect_plot_panels(histograms={}, info=info)

    planner_panel = next(payload for kind, name, payload in panels if name == "planner_vs_realized_systems")
    assert planner_panel["series"]["requested"] == [4, 0, 2, 0, 0, 0, 0]
    assert planner_panel["series"]["accepted"] == [1, 0, 2, 0, 0, 0, 0]
    assert planner_panel["series"]["full_render"] == [0, 0, 0, 0, 0, 0, 0]
    assert planner_panel["series"]["truncated"] == [0, 0, 0, 0, 0, 0, 0]


def test_collect_plot_panels_skips_augmentation_panels_when_counts_missing() -> None:
    info = {
        "snapshot": {
            "accepted_samples": 10,
            "rejected_samples": 1,
            "failure_reason_counts": {"timeout": 1},
        }
    }

    panels = _collect_plot_panels(histograms={}, info=info)
    panel_names = [name for _, name, _ in panels]

    assert "rejection_rate_summary" in panel_names
    assert "augmentation_outcomes" not in panel_names
    assert "augmentation_routing" not in panel_names
    assert "augmentation_gates" not in panel_names


def test_collect_histograms_excludes_new_telemetry_histograms() -> None:
    info = {
        "snapshot": {
            "bottom_whitespace_px_histogram": {"50": 3},
            "augmentation_geom_ms_histogram": {"1": 5},
            "target_full_render_system_histogram": {"3": {"4": 2}},
        }
    }

    histograms = collect_histograms(info)

    assert histograms == {"bottom_whitespace_px_histogram": {50: 3}}


def test_plot_histograms_renders_legacy_snapshot_without_dataset(tmp_path, monkeypatch) -> None:
    info = {
        "snapshot": {
            "accepted_samples": 10,
            "rejected_samples": 2,
            "failure_reason_counts": {"timeout": 2},
            "accepted_system_histogram": {
                "1": {"2": 3},
                "2": {"4": 1},
                "3_plus": {"6": 2},
            },
            "bottom_whitespace_px_histogram": {"100": 4},
        }
    }
    info_path = tmp_path / "info.json"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setattr(plt, "show", lambda: None)

    histograms = collect_histograms(info)
    plot_histograms(histograms, dataset_name="legacy", info_path=info_path, dataset=None)
