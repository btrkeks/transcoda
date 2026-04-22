from __future__ import annotations

from notebooks.run_histogram_viewer import _collect_rejection_summary


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
