"""Worker outcomes — what the worker pool emits per sample."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.dataset_generation.dataset_generation.types_domain import (
    PreferredFiveSixStatus,
    TruncationMode,
)
from scripts.dataset_generation.dataset_generation.types_events import (
    AugmentationPreviewArtifacts,
    AugmentationTraceEvent,
    FailureRenderAttempt,
    VerovioDiagnosticEvent,
)
from scripts.dataset_generation.dataset_generation.types_render import AcceptedSample


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


WorkerOutcome = WorkerSuccess | WorkerFailure
