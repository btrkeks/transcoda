"""Canonical failure taxonomy and mapping helpers for dataset generation."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from scripts.dataset_generation.dataset_generation.base import GenerationStats

WorkerFailureReason = Literal[
    "invalid_kern",
    "multi_page",
    "render_fit",
    "render_rejected",
    "sparse_render",
    "system_band_below_min",
    "system_band_above_max",
    "system_band_truncation_exhausted",
    "system_band_rejected",
    "processing_error",
]

FailureReason = Literal[
    "invalid_kern",
    "multi_page",
    "render_fit",
    "render_rejected",
    "sparse_render",
    "system_band_below_min",
    "system_band_above_max",
    "system_band_truncation_exhausted",
    "system_band_truncation_exhausted_below_min",
    "system_band_truncation_exhausted_too_large",
    "system_band_truncation_exhausted_render_failure",
    "system_band_truncation_exhausted_mixed_gap",
    "system_band_truncation_exhausted_unknown",
    "system_band_rejected",
    "processing_error",
    "unknown_result",
    "timeout",
    "process_expired",
]

SYSTEM_BAND_FAILURE_REASONS: tuple[WorkerFailureReason, ...] = (
    "system_band_below_min",
    "system_band_above_max",
    "system_band_truncation_exhausted",
    "system_band_rejected",
)

WORKER_FAILURE_REASONS: tuple[WorkerFailureReason, ...] = (
    "invalid_kern",
    "multi_page",
    "render_fit",
    "render_rejected",
    "sparse_render",
    "system_band_below_min",
    "system_band_above_max",
    "system_band_truncation_exhausted",
    "system_band_rejected",
    "processing_error",
)

ALL_FAILURE_REASONS: tuple[FailureReason, ...] = (
    "multi_page",
    "invalid_kern",
    "sparse_render",
    "render_fit",
    "render_rejected",
    "system_band_below_min",
    "system_band_above_max",
    "system_band_truncation_exhausted",
    "system_band_truncation_exhausted_below_min",
    "system_band_truncation_exhausted_too_large",
    "system_band_truncation_exhausted_render_failure",
    "system_band_truncation_exhausted_mixed_gap",
    "system_band_truncation_exhausted_unknown",
    "system_band_rejected",
    "processing_error",
    "unknown_result",
    "timeout",
    "process_expired",
)

TRUNCATION_EXHAUSTION_DIAGNOSTIC_REASONS: tuple[FailureReason, ...] = (
    "system_band_truncation_exhausted_below_min",
    "system_band_truncation_exhausted_too_large",
    "system_band_truncation_exhausted_render_failure",
    "system_band_truncation_exhausted_mixed_gap",
    "system_band_truncation_exhausted_unknown",
)


def truncation_exhaustion_subreason_from_detail(detail: str | None) -> FailureReason:
    """Extract a stable truncation-exhaustion subtype from failure detail."""
    if not detail:
        return "system_band_truncation_exhausted_unknown"
    for part in detail.split(";"):
        if not part.startswith("diagnostic="):
            continue
        diagnostic = part.split("=", 1)[1].strip()
        if diagnostic == "below_min":
            return "system_band_truncation_exhausted_below_min"
        if diagnostic == "too_large":
            return "system_band_truncation_exhausted_too_large"
        if diagnostic == "render_failure":
            return "system_band_truncation_exhausted_render_failure"
        if diagnostic == "mixed_gap":
            return "system_band_truncation_exhausted_mixed_gap"
        break
    return "system_band_truncation_exhausted_unknown"


def legacy_message_to_failure_reason(message: str) -> WorkerFailureReason:
    """Map legacy worker tuple messages to canonical failure reasons."""
    if message == "Reject:multi_page":
        return "multi_page"
    if message.startswith("Reject:invalid_kern:") or message.startswith("Invalid kern:"):
        return "invalid_kern"
    if message.startswith("Reject:sparse_render:"):
        return "sparse_render"
    if message.startswith("Reject:render_fit:"):
        return "render_fit"
    if message.startswith("Reject:render_rejected"):
        return "render_rejected"
    if message.startswith("Reject:system_band_below_min:"):
        return "system_band_below_min"
    if message.startswith("Reject:system_band_above_max:"):
        return "system_band_above_max"
    if message.startswith("Reject:system_band_truncation_exhausted:"):
        return "system_band_truncation_exhausted"
    if message.startswith("Reject:system_band_rejected"):
        return "system_band_rejected"
    return "processing_error"


def record_worker_failure(
    *,
    stats: GenerationStats,
    reason_counts: Counter[str],
    reason: WorkerFailureReason,
    reason_detail: str | None = None,
) -> None:
    """Record a worker-originated failure into canonical and legacy counters."""
    reason_counts[reason] += 1
    if reason == "system_band_truncation_exhausted":
        reason_counts[truncation_exhaustion_subreason_from_detail(reason_detail)] += 1
    if reason in SYSTEM_BAND_FAILURE_REASONS and reason != "system_band_rejected":
        reason_counts["system_band_rejected"] += 1
    if reason == "multi_page":
        stats.overflows += 1
    elif reason == "invalid_kern":
        stats.invalid += 1
    elif reason == "sparse_render":
        stats.rejected_sparse += 1
    elif reason == "render_fit":
        stats.rejected_render_fit += 1
    else:
        stats.errors += 1


def build_failure_reason_counts(reason_counts: Counter[str]) -> dict[str, int]:
    """Return a stable, fully populated failure-reason dict."""
    return {reason: int(reason_counts.get(reason, 0)) for reason in ALL_FAILURE_REASONS}
