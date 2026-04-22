"""Persistence snapshots for resumable runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResumeSnapshot:
    next_sample_idx: int
    accepted_samples: int
    rejected_samples: int
    failure_reason_counts: dict[str, int]
    truncation_counts: dict[str, int]
    full_render_system_histogram: dict[str, int]
    accepted_system_histogram: dict[str, dict[str, int]]
    truncated_output_system_histogram: dict[str, int]
    preferred_5_6_counts: dict[str, int]
    bottom_whitespace_px_histogram: dict[str, int]
    top_whitespace_px_histogram: dict[str, int]
    content_height_px_histogram: dict[str, int]
    terminal_timeout_crash_artifacts: int
    terminal_process_expired_crash_artifacts: int
    requested_target_bucket_histogram: dict[str, int]
    candidate_hit_counts: dict[str, int]
    retry_counts: dict[str, int]
    quarantined_sources: tuple[str, ...]
    augmentation_outcome_counts: dict[str, int]
    augmentation_band_counts: dict[str, int]
    augmentation_branch_counts: dict[str, int]
    final_geometry_counts: dict[str, int]
    oob_failure_reason_counts: dict[str, int]
    outer_gate_failure_reason_counts: dict[str, int]
