"""Trace and event types: Verovio diagnostics, geometry traces, success/failure events."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.dataset_generation.dataset_generation.types_domain import (
    AttemptStageName,
    PreferredFiveSixStatus,
    TruncationMode,
)


@dataclass(frozen=True)
class VerovioDiagnostic:
    diagnostic_kind: str
    raw_message: str
    render_attempt_idx: int | None = None
    near_line: int | None = None
    expected_duration_from_start: str | None = None
    found_duration_from_start: str | None = None
    line_text: str | None = None


@dataclass(frozen=True)
class VerovioDiagnosticEvent:
    event: str
    sample_id: str
    sample_idx: int
    source_paths: tuple[str, ...]
    stage: AttemptStageName
    seed: int
    render_attempt_idx: int | None
    diagnostic_kind: str
    raw_message: str
    near_line: int | None = None
    expected_duration_from_start: str | None = None
    found_duration_from_start: str | None = None
    line_text: str | None = None
    truncation_chunk_count: int | None = None
    truncation_total_chunks: int | None = None
    truncation_ratio: float | None = None


@dataclass(frozen=True)
class MarginTrace:
    top_px: int | None
    bottom_px: int | None
    left_px: int | None
    right_px: int | None


@dataclass(frozen=True)
class GeometryTrace:
    sampled: bool
    conservative: bool
    angle_deg: float | None
    scale: float | None
    tx_px: float | None
    ty_px: float | None
    x_scale: float | None
    y_scale: float | None
    perspective_applied: bool


@dataclass(frozen=True)
class BoundsGateTrace:
    passed: bool
    failure_reason: str | None
    margins_px: MarginTrace | None
    border_touch_count: int | None
    dx_frac: float | None
    dy_frac: float | None
    area_retention: float | None


@dataclass(frozen=True)
class QualityGateTrace:
    passed: bool
    failure_reason: str | None
    mean_luma: float | None
    content_ratio: float | None
    margins_px: MarginTrace | None
    border_touch_count: int | None


@dataclass(frozen=True)
class OuterGateTrace:
    passed: bool
    failure_reason: str | None
    quality_gate: QualityGateTrace
    transform_consistency: BoundsGateTrace


@dataclass(frozen=True)
class OfflineAugmentTrace:
    """Structured trace of every decision made inside ``offline_augment``."""

    branch: str  # "geometric" | "none"
    initial_geometry: GeometryTrace
    retry_geometry: GeometryTrace | None
    selected_geometry: GeometryTrace
    final_geometry_applied: bool
    initial_oob_gate: BoundsGateTrace
    retry_oob_gate: BoundsGateTrace | None
    outer_gate: OuterGateTrace
    augraphy_outcome: str  # "applied" | "noop" | "error" | "invalid_input"
    augraphy_normalize_accepted: bool
    augraphy_fallback_attempted: bool
    augraphy_fallback_outcome: str | None
    augraphy_fallback_normalize_accepted: bool | None
    offline_geom_ms: float
    offline_gates_ms: float
    offline_augraphy_ms: float
    offline_texture_ms: float


@dataclass(frozen=True)
class AugmentationTraceEvent:
    event: str  # always "augmentation_trace"
    sample_id: str
    sample_idx: int
    seed: int
    render_height_px: int | None
    bottom_padding_px: int | None
    top_whitespace_px: int | None
    bottom_whitespace_px: int | None
    content_height_px: int | None
    band: str  # "roomy" | "balanced" | "tight"
    branch: str  # "geometric" | "none"
    initial_geometry: GeometryTrace
    retry_geometry: GeometryTrace | None
    selected_geometry: GeometryTrace
    final_geometry_applied: bool
    initial_oob_gate: BoundsGateTrace
    retry_oob_gate: BoundsGateTrace | None
    augraphy_outcome: str
    augraphy_normalize_accepted: bool
    augraphy_fallback_attempted: bool
    augraphy_fallback_outcome: str | None
    augraphy_fallback_normalize_accepted: bool | None
    outer_gate: OuterGateTrace
    final_outcome: str
    offline_geom_ms: float
    offline_gates_ms: float
    offline_augraphy_ms: float
    offline_texture_ms: float


@dataclass(frozen=True)
class AugmentationPreviewArtifacts:
    base_image_jpeg: bytes
    pre_augraphy_image_jpeg: bytes
    final_image_jpeg: bytes


@dataclass(frozen=True)
class FailureRenderAttempt:
    stage: AttemptStageName
    seed: int
    chunk_count: int | None = None
    total_chunks: int | None = None
    ratio: float | None = None
    system_count: int | None = None
    page_count: int | None = None
    content_height_px: int | None = None
    vertical_fill_ratio: float | None = None
    render_rejection_reason: str | None = None
    decision_reason: str | None = None
    accepted: bool = False
    verovio_diagnostic_count: int = 0


@dataclass(frozen=True)
class FailureTraceEvent:
    event: str
    sample_id: str
    sample_idx: int
    source_paths: tuple[str, ...]
    target_bucket: int | None
    planned_line_count: int | None
    candidate_in_target_range: bool | None
    failure_reason: str
    truncation_mode: TruncationMode | None
    truncation_attempted: bool
    preferred_5_6_rescue_attempted: bool = False
    preferred_5_6_rescue_succeeded: bool = False
    preferred_5_6_status: PreferredFiveSixStatus | None = None
    attempts: tuple[FailureRenderAttempt, ...] = ()


@dataclass(frozen=True)
class SuccessTraceEvent:
    event: str
    sample_id: str
    sample_idx: int
    source_paths: tuple[str, ...]
    target_bucket: int | None
    planned_line_count: int | None
    candidate_in_target_range: bool | None
    committed_to_dataset: bool
    full_render_system_count: int | None
    full_render_content_height_px: int | None
    full_render_vertical_fill_ratio: float | None
    full_render_rejection_reason: str | None
    accepted_render_system_count: int | None
    truncation_attempted: bool
    truncation_rescued: bool
    preferred_5_6_rescue_attempted: bool = False
    preferred_5_6_rescue_succeeded: bool = False
    preferred_5_6_status: PreferredFiveSixStatus | None = None
    initial_kern_spine_count: int | None = None
    segment_count: int | None = None
    source_non_empty_line_count: int | None = None
    truncation_applied: bool = False
    truncation_reason: str | None = None
    truncation_ratio: float | None = None
