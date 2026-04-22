"""Core data types for the dataset-generation rewrite."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np

from scripts.dataset_generation.dataset_generation.image_generation.types import RenderedPage

AcceptanceAction = Literal[
    "accept_without_truncation",
    "accept_with_truncation",
    "reject",
]
TruncationMode = Literal["forbidden", "preferred", "required"]
AugmentationBand = Literal["roomy", "balanced", "tight"]
PreferredFiveSixStatus = Literal[
    "preferred_5_6_accepted_full",
    "preferred_5_6_rescued",
    "preferred_5_6_truncated",
    "preferred_5_6_failed",
]


class AttemptStageName(StrEnum):
    FULL = "full"
    FULL_LAYOUT_RESCUE = "full_layout_rescue"
    TRUNCATION_CANDIDATE = "truncation_candidate"
    TRUNCATION_CANDIDATE_LAYOUT_RESCUE = "truncation_candidate_layout_rescue"


@dataclass(frozen=True)
class SourceEntry:
    path: Path
    source_id: str
    root_dir: Path
    root_label: str
    measure_count: int
    non_empty_line_count: int
    has_header: bool
    initial_spine_count: int
    terminal_spine_count: int
    restored_terminal_spine_count: int


@dataclass(frozen=True)
class SourceSegment:
    source_id: str
    path: Path
    order: int


@dataclass(frozen=True)
class SamplePlan:
    sample_id: str
    seed: int
    segments: tuple[SourceSegment, ...]
    label_transcription: str
    source_measure_count: int
    source_non_empty_line_count: int
    source_max_initial_spine_count: int
    segment_count: int


@dataclass(frozen=True)
class SvgLayoutDiagnostics:
    system_count: int
    page_count: int
    system_bbox_stats: dict[str, float] | None = None


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
    black_ratio: float | None
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
    augraphy_outcome: str  # "applied" | "noop" | "error" | "invalid_input"
    augraphy_normalize_accepted: bool
    augraphy_fallback_attempted: bool
    augraphy_fallback_outcome: str | None
    augraphy_fallback_normalize_accepted: bool | None
    outer_gate: OuterGateTrace
    final_outcome: str
    # "fully_augmented" | "clean_gate_rejected" | "augraphy_on_base" |
    # "clean_augraphy_failed" | "clean_early_exit"
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
class AugmentedRenderResult:
    final_image: np.ndarray | bytes
    trace: AugmentationTraceEvent | None
    base_image: np.ndarray | None = None
    pre_augraphy_image: np.ndarray | None = None


@dataclass(frozen=True)
class RenderResult:
    image: np.ndarray | None
    render_layers: RenderedPage | None
    svg_diagnostics: SvgLayoutDiagnostics
    bottom_whitespace_ratio: float | None
    vertical_fill_ratio: float | None
    render_height_px: int | None = None
    bottom_padding_px: int | None = None
    bottom_whitespace_px: int | None = None
    top_whitespace_px: int | None = None
    content_height_px: int | None = None
    rejection_reason: str | None = None
    metadata_prefix: str = ""
    verovio_diagnostics: tuple[VerovioDiagnostic, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.image is not None and self.rejection_reason is None


@dataclass(frozen=True)
class AcceptanceDecision:
    action: AcceptanceAction
    reason: str | None = None


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
class AcceptedSample:
    sample_id: str
    label_transcription: str
    image_bytes: bytes
    initial_kern_spine_count: int
    segment_count: int
    source_ids: tuple[str, ...]
    source_measure_count: int
    source_non_empty_line_count: int
    system_count: int
    truncation_applied: bool
    truncation_reason: str | None
    truncation_ratio: float | None
    bottom_whitespace_ratio: float | None
    vertical_fill_ratio: float | None
    bottom_whitespace_px: int | None = None
    top_whitespace_px: int | None = None
    content_height_px: int | None = None


@dataclass(frozen=True)
class WorkerSuccess:
    sample: AcceptedSample
    truncation_attempted: bool
    truncation_rescued: bool
    full_render_system_count: int | None = None
    full_render_content_height_px: int | None = None
    full_render_vertical_fill_ratio: float | None = None
    full_render_rejection_reason: str | None = None
    accepted_render_system_count: int | None = None
    preferred_5_6_rescue_attempted: bool = False
    preferred_5_6_rescue_succeeded: bool = False
    preferred_5_6_status: PreferredFiveSixStatus | None = None
    verovio_diagnostics: tuple[VerovioDiagnosticEvent, ...] = ()
    augmentation_trace: AugmentationTraceEvent | None = None
    augmentation_preview: AugmentationPreviewArtifacts | None = None


@dataclass(frozen=True)
class WorkerFailure:
    sample_id: str
    failure_reason: str
    truncation_attempted: bool
    truncation_rescued: bool = False
    truncation_mode: TruncationMode | None = None
    full_render_system_count: int | None = None
    full_render_content_height_px: int | None = None
    full_render_vertical_fill_ratio: float | None = None
    full_render_rejection_reason: str | None = None
    accepted_render_system_count: int | None = None
    preferred_5_6_rescue_attempted: bool = False
    preferred_5_6_rescue_succeeded: bool = False
    preferred_5_6_status: PreferredFiveSixStatus | None = None
    failure_attempts: tuple[FailureRenderAttempt, ...] = ()
    verovio_diagnostics: tuple[VerovioDiagnosticEvent, ...] = ()


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


WorkerOutcome = WorkerSuccess | WorkerFailure


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
