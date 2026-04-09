"""Core data types for the dataset-generation rewrite."""

from __future__ import annotations

from dataclasses import dataclass
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
class RenderResult:
    image: np.ndarray | None
    render_layers: RenderedPage | None
    svg_diagnostics: SvgLayoutDiagnostics
    bottom_whitespace_ratio: float | None
    vertical_fill_ratio: float | None
    bottom_whitespace_px: int | None = None
    top_whitespace_px: int | None = None
    content_height_px: int | None = None
    rejection_reason: str | None = None
    metadata_prefix: str = ""

    @property
    def succeeded(self) -> bool:
        return self.image is not None and self.rejection_reason is None


@dataclass(frozen=True)
class AcceptanceDecision:
    action: AcceptanceAction
    reason: str | None = None


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


@dataclass(frozen=True)
class WorkerFailure:
    sample_id: str
    failure_reason: str
    truncation_attempted: bool
    truncation_rescued: bool = False
    full_render_system_count: int | None = None
    full_render_content_height_px: int | None = None
    full_render_vertical_fill_ratio: float | None = None
    full_render_rejection_reason: str | None = None
    accepted_render_system_count: int | None = None
    preferred_5_6_rescue_attempted: bool = False
    preferred_5_6_rescue_succeeded: bool = False
    preferred_5_6_status: PreferredFiveSixStatus | None = None


WorkerOutcome = WorkerSuccess | WorkerFailure


@dataclass(frozen=True)
class ResumeSnapshot:
    next_sample_idx: int
    accepted_samples: int
    rejected_samples: int
    failure_reason_counts: dict[str, int]
    truncation_counts: dict[str, int]
    full_render_system_histogram: dict[str, int]
    accepted_system_histogram: dict[str, int]
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
