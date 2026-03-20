from collections import Counter

from scripts.dataset_generation.dataset_generation.base import GenerationStats
from scripts.dataset_generation.dataset_generation.failures import (
    build_failure_reason_counts,
    legacy_message_to_failure_reason,
    record_worker_failure,
    truncation_exhaustion_subreason_from_detail,
)


def test_legacy_message_to_failure_reason_maps_expected_codes():
    assert legacy_message_to_failure_reason("Reject:multi_page") == "multi_page"
    assert legacy_message_to_failure_reason("Reject:invalid_kern:oops") == "invalid_kern"
    assert legacy_message_to_failure_reason("Invalid kern:foo") == "invalid_kern"
    assert legacy_message_to_failure_reason("Reject:sparse_render:black_ratio=0.1") == "sparse_render"
    assert legacy_message_to_failure_reason("Reject:render_fit:bottom_clearance") == "render_fit"
    assert legacy_message_to_failure_reason("Reject:render_rejected") == "render_rejected"
    assert (
        legacy_message_to_failure_reason("Reject:system_band_below_min:system_count=4")
        == "system_band_below_min"
    )
    assert (
        legacy_message_to_failure_reason("Reject:system_band_above_max:system_count=7")
        == "system_band_above_max"
    )
    assert (
        legacy_message_to_failure_reason(
            "Reject:system_band_truncation_exhausted:target_band=5-6"
        )
        == "system_band_truncation_exhausted"
    )
    assert legacy_message_to_failure_reason("Reject:system_band_rejected:system_count=4") == "system_band_rejected"
    assert legacy_message_to_failure_reason("other failure") == "processing_error"


def test_record_worker_failure_updates_legacy_stats_and_reason_counter():
    stats = GenerationStats()
    reason_counts: Counter[str] = Counter()

    record_worker_failure(stats=stats, reason_counts=reason_counts, reason="multi_page")
    record_worker_failure(stats=stats, reason_counts=reason_counts, reason="invalid_kern")
    record_worker_failure(stats=stats, reason_counts=reason_counts, reason="sparse_render")
    record_worker_failure(stats=stats, reason_counts=reason_counts, reason="render_fit")
    record_worker_failure(stats=stats, reason_counts=reason_counts, reason="render_rejected")
    record_worker_failure(stats=stats, reason_counts=reason_counts, reason="system_band_below_min")
    record_worker_failure(stats=stats, reason_counts=reason_counts, reason="system_band_above_max")
    record_worker_failure(
        stats=stats,
        reason_counts=reason_counts,
        reason="system_band_truncation_exhausted",
        reason_detail="target_band=5-7;diagnostic=below_min",
    )

    assert stats.overflows == 1
    assert stats.invalid == 1
    assert stats.rejected_sparse == 1
    assert stats.rejected_render_fit == 1
    assert stats.errors == 4
    assert reason_counts["render_rejected"] == 1
    assert reason_counts["system_band_below_min"] == 1
    assert reason_counts["system_band_above_max"] == 1
    assert reason_counts["system_band_truncation_exhausted"] == 1
    assert reason_counts["system_band_truncation_exhausted_below_min"] == 1
    assert reason_counts["system_band_rejected"] == 3


def test_build_failure_reason_counts_is_stable_and_zero_filled():
    reason_counts: Counter[str] = Counter({"multi_page": 2, "timeout": 1})
    result = build_failure_reason_counts(reason_counts)

    assert result["multi_page"] == 2
    assert result["timeout"] == 1
    assert result["invalid_kern"] == 0
    assert result["process_expired"] == 0
    assert result["system_band_rejected"] == 0
    assert result["system_band_below_min"] == 0
    assert result["system_band_above_max"] == 0
    assert result["system_band_truncation_exhausted"] == 0
    assert result["system_band_truncation_exhausted_below_min"] == 0
    assert result["system_band_truncation_exhausted_too_large"] == 0
    assert result["system_band_truncation_exhausted_render_failure"] == 0
    assert result["system_band_truncation_exhausted_mixed_gap"] == 0
    assert result["system_band_truncation_exhausted_unknown"] == 0
    assert "unknown_result" in result


def test_truncation_exhaustion_subreason_from_detail_maps_expected_diagnostics():
    assert (
        truncation_exhaustion_subreason_from_detail("target_band=5-7;diagnostic=below_min")
        == "system_band_truncation_exhausted_below_min"
    )
    assert (
        truncation_exhaustion_subreason_from_detail("target_band=5-7;diagnostic=too_large")
        == "system_band_truncation_exhausted_too_large"
    )
    assert (
        truncation_exhaustion_subreason_from_detail("target_band=5-7;diagnostic=render_failure")
        == "system_band_truncation_exhausted_render_failure"
    )
    assert (
        truncation_exhaustion_subreason_from_detail("target_band=5-7;diagnostic=mixed_gap")
        == "system_band_truncation_exhausted_mixed_gap"
    )
    assert (
        truncation_exhaustion_subreason_from_detail("target_band=5-7")
        == "system_band_truncation_exhausted_unknown"
    )
