"""Shared builders for dataset-generation test fixtures.

Each ``make_*`` helper returns a fully-populated dataclass instance with
sensible defaults. Tests pass only the fields they care about; everything
else is filled in from the default recipe-like values used across the
existing suite.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from scripts.dataset_generation.dataset_generation.io import encode_jpeg_image
from scripts.dataset_generation.dataset_generation.types_domain import (
    AttemptStageName,
    SamplePlan,
    SourceSegment,
)
from scripts.dataset_generation.dataset_generation.types_events import (
    AugmentationPreviewArtifacts,
    AugmentationTraceEvent,
    BoundsGateTrace,
    FailureRenderAttempt,
    GeometryTrace,
    MarginTrace,
    OuterGateTrace,
    QualityGateTrace,
    VerovioDiagnosticEvent,
)
from scripts.dataset_generation.dataset_generation.types_outcomes import (
    WorkerFailure,
    WorkerSuccess,
)
from scripts.dataset_generation.dataset_generation.types_render import (
    AcceptedSample,
    RenderResult,
    SvgLayoutDiagnostics,
)

_UNSET: Any = object()


def _default_image() -> np.ndarray:
    image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    image[10:70, 10:600] = 0
    return image


def make_sample_plan(
    *,
    sample_idx: int = 0,
    seed: int | None = None,
    segments: tuple[SourceSegment, ...] | None = None,
    label_transcription: str = "**kern\n=1\n4c\n*-\n",
    source_measure_count: int = 1,
    source_non_empty_line_count: int = 4,
    source_max_initial_spine_count: int = 1,
    segment_count: int | None = None,
    **overrides: Any,
) -> SamplePlan:
    if segments is None:
        segments = (
            SourceSegment(source_id="input/piece", path=Path("/tmp/piece.krn"), order=0),
        )
    plan = SamplePlan(
        sample_id=f"sample_{sample_idx:08d}",
        seed=sample_idx if seed is None else seed,
        segments=segments,
        label_transcription=label_transcription,
        source_measure_count=source_measure_count,
        source_non_empty_line_count=source_non_empty_line_count,
        source_max_initial_spine_count=source_max_initial_spine_count,
        segment_count=len(segments) if segment_count is None else segment_count,
    )
    if overrides:
        plan = replace(plan, **overrides)
    return plan


def make_render_result(
    *,
    system_count: int = 4,
    page_count: int = 1,
    image: Any = _UNSET,
    rejection_reason: str | None = None,
    bottom_whitespace_ratio: float | None = 0.10,
    vertical_fill_ratio: float | None = 0.72,
    bottom_whitespace_px: int | None = 149,
    top_whitespace_px: int | None = 33,
    content_height_px: int | None = 1069,
    verovio_diagnostics: tuple = (),
    **overrides: Any,
) -> RenderResult:
    if image is _UNSET:
        image = None if rejection_reason is not None else _default_image()
    result = RenderResult(
        image=image,
        render_layers=None,
        svg_diagnostics=SvgLayoutDiagnostics(system_count=system_count, page_count=page_count),
        bottom_whitespace_ratio=bottom_whitespace_ratio,
        vertical_fill_ratio=vertical_fill_ratio,
        bottom_whitespace_px=bottom_whitespace_px,
        top_whitespace_px=top_whitespace_px,
        content_height_px=content_height_px,
        rejection_reason=rejection_reason,
        verovio_diagnostics=verovio_diagnostics,
    )
    if overrides:
        result = replace(result, **overrides)
    return result


def make_accepted_sample(
    *,
    plan: SamplePlan | None = None,
    sample_id: str | None = None,
    label_transcription: str | None = None,
    image_bytes: bytes | None = None,
    system_count: int = 4,
    truncation_applied: bool = False,
    truncation_reason: str | None = None,
    truncation_ratio: float | None = None,
    bottom_whitespace_ratio: float | None = 0.10,
    vertical_fill_ratio: float | None = 0.72,
    bottom_whitespace_px: int | None = 149,
    top_whitespace_px: int | None = 33,
    content_height_px: int | None = 1069,
    **overrides: Any,
) -> AcceptedSample:
    if plan is None:
        plan = make_sample_plan()
    if image_bytes is None:
        image_bytes = encode_jpeg_image(_default_image())
    resolved_transcription = (
        label_transcription if label_transcription is not None else plan.label_transcription
    )
    sample = AcceptedSample(
        sample_id=sample_id if sample_id is not None else plan.sample_id,
        label_transcription=resolved_transcription,
        image_bytes=image_bytes,
        initial_kern_spine_count=resolved_transcription.splitlines()[0].count("\t") + 1,
        segment_count=plan.segment_count,
        source_ids=tuple(segment.source_id for segment in plan.segments),
        source_measure_count=plan.source_measure_count,
        source_non_empty_line_count=plan.source_non_empty_line_count,
        system_count=system_count,
        truncation_applied=truncation_applied,
        truncation_reason=truncation_reason,
        truncation_ratio=truncation_ratio,
        bottom_whitespace_ratio=bottom_whitespace_ratio,
        vertical_fill_ratio=vertical_fill_ratio,
        bottom_whitespace_px=bottom_whitespace_px,
        top_whitespace_px=top_whitespace_px,
        content_height_px=content_height_px,
    )
    if overrides:
        sample = replace(sample, **overrides)
    return sample


def make_worker_success(
    *,
    plan: SamplePlan | None = None,
    sample: AcceptedSample | None = None,
    system_count: int = 4,
    truncation_applied: bool = False,
    truncation_attempted: bool = False,
    truncation_rescued: bool = False,
    full_render_system_count: int | None = None,
    full_render_content_height_px: int | None = 1069,
    full_render_vertical_fill_ratio: float | None = 0.72,
    full_render_rejection_reason: str | None = None,
    accepted_render_system_count: int | None = None,
    verovio_diagnostics: tuple = (),
    augmentation_trace: AugmentationTraceEvent | None = None,
    augmentation_preview: AugmentationPreviewArtifacts | None = None,
    **overrides: Any,
) -> WorkerSuccess:
    if plan is None:
        plan = make_sample_plan()
    if sample is None:
        sample = make_accepted_sample(
            plan=plan,
            system_count=system_count,
            truncation_applied=truncation_applied,
        )
    if full_render_system_count is None:
        full_render_system_count = system_count
    if accepted_render_system_count is None:
        accepted_render_system_count = system_count
    success = WorkerSuccess(
        sample=sample,
        truncation_attempted=truncation_attempted,
        truncation_rescued=truncation_rescued,
        full_render_system_count=full_render_system_count,
        full_render_content_height_px=full_render_content_height_px,
        full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
        full_render_rejection_reason=full_render_rejection_reason,
        accepted_render_system_count=accepted_render_system_count,
        verovio_diagnostics=verovio_diagnostics,
        augmentation_trace=augmentation_trace,
        augmentation_preview=augmentation_preview,
    )
    if overrides:
        success = replace(success, **overrides)
    return success


def make_worker_failure(
    *,
    plan: SamplePlan | None = None,
    sample_id: str | None = None,
    failure_reason: str = "render_failed",
    truncation_attempted: bool = False,
    truncation_rescued: bool = False,
    truncation_mode: str | None = None,
    failure_attempts: tuple[FailureRenderAttempt, ...] = (),
    verovio_diagnostics: tuple = (),
    **overrides: Any,
) -> WorkerFailure:
    if sample_id is None:
        sample_id = plan.sample_id if plan is not None else make_sample_plan().sample_id
    failure = WorkerFailure(
        sample_id=sample_id,
        failure_reason=failure_reason,
        truncation_attempted=truncation_attempted,
        truncation_rescued=truncation_rescued,
        truncation_mode=truncation_mode,
        failure_attempts=failure_attempts,
        verovio_diagnostics=verovio_diagnostics,
    )
    if overrides:
        failure = replace(failure, **overrides)
    return failure


def make_verovio_diagnostic(
    *,
    plan: SamplePlan | None = None,
    stage: AttemptStageName | str = AttemptStageName.FULL,
    seed: int | None = None,
    render_attempt_idx: int | None = 1,
    diagnostic_kind: str = "inconsistent_rhythm_analysis",
    raw_message: str = "Error: Inconsistent rhythm analysis occurring near line 12",
    near_line: int | None = 12,
    expected_duration_from_start: str | None = "64",
    found_duration_from_start: str | None = "62",
    line_text: str | None = "4G\t.\t.\t4c 4e",
    truncation_chunk_count: int | None = None,
    truncation_total_chunks: int | None = None,
    truncation_ratio: float | None = None,
    **overrides: Any,
) -> VerovioDiagnosticEvent:
    if plan is None:
        plan = make_sample_plan()
    event = VerovioDiagnosticEvent(
        event="verovio_diagnostic",
        sample_id=plan.sample_id,
        sample_idx=int(plan.sample_id.split("_")[-1]),
        source_paths=tuple(str(segment.path.resolve()) for segment in plan.segments),
        stage=stage,
        seed=plan.seed if seed is None else seed,
        render_attempt_idx=render_attempt_idx,
        diagnostic_kind=diagnostic_kind,
        raw_message=raw_message,
        near_line=near_line,
        expected_duration_from_start=expected_duration_from_start,
        found_duration_from_start=found_duration_from_start,
        line_text=line_text,
        truncation_chunk_count=truncation_chunk_count,
        truncation_total_chunks=truncation_total_chunks,
        truncation_ratio=truncation_ratio,
    )
    if overrides:
        event = replace(event, **overrides)
    return event


def make_geometry_trace(
    *,
    sampled: bool = True,
    conservative: bool = False,
    angle_deg: float | None = _UNSET,
    scale: float | None = _UNSET,
    tx_px: float | None = _UNSET,
    ty_px: float | None = _UNSET,
    x_scale: float | None = _UNSET,
    y_scale: float | None = _UNSET,
    perspective_applied: bool = False,
    **overrides: Any,
) -> GeometryTrace:
    def _pick(override: Any, sampled_value: float | None) -> float | None:
        return sampled_value if override is _UNSET else override

    trace = GeometryTrace(
        sampled=sampled,
        conservative=conservative,
        angle_deg=_pick(angle_deg, 0.8 if sampled else None),
        scale=_pick(scale, 1.02 if sampled else None),
        tx_px=_pick(tx_px, 2.0 if sampled else None),
        ty_px=_pick(ty_px, 3.0 if sampled else None),
        x_scale=_pick(x_scale, 0.95 if sampled else None),
        y_scale=_pick(y_scale, 1.0 if sampled else None),
        perspective_applied=perspective_applied,
    )
    if overrides:
        trace = replace(trace, **overrides)
    return trace


def make_margin_trace(
    *,
    top_px: int | None = 20,
    bottom_px: int | None = 1200,
    left_px: int | None = 20,
    right_px: int | None = 500,
) -> MarginTrace:
    return MarginTrace(top_px=top_px, bottom_px=bottom_px, left_px=left_px, right_px=right_px)


def make_bounds_gate_trace(
    *,
    passed: bool = True,
    failure_reason: str | None = None,
    margins_px: MarginTrace | None = None,
    border_touch_count: int | None = 0,
    dx_frac: float | None = 0.01,
    dy_frac: float | None = 0.02,
    area_retention: float | None = 0.95,
    **overrides: Any,
) -> BoundsGateTrace:
    if margins_px is None:
        margins_px = make_margin_trace()
    trace = BoundsGateTrace(
        passed=passed,
        failure_reason=failure_reason,
        margins_px=margins_px,
        border_touch_count=border_touch_count,
        dx_frac=dx_frac,
        dy_frac=dy_frac,
        area_retention=area_retention,
    )
    if overrides:
        trace = replace(trace, **overrides)
    return trace


def make_quality_gate_trace(
    *,
    passed: bool = True,
    failure_reason: str | None = None,
    mean_luma: float | None = 200.0,
    black_ratio: float | None = 0.1,
    margins_px: MarginTrace | None = None,
    border_touch_count: int | None = 0,
) -> QualityGateTrace:
    if margins_px is None:
        margins_px = make_margin_trace()
    return QualityGateTrace(
        passed=passed,
        failure_reason=failure_reason,
        mean_luma=mean_luma,
        black_ratio=black_ratio,
        margins_px=margins_px,
        border_touch_count=border_touch_count,
    )


def make_outer_gate_trace(
    *,
    passed: bool = True,
    failure_reason: str | None = None,
    quality_gate: QualityGateTrace | None = None,
    transform_consistency: BoundsGateTrace | None = None,
) -> OuterGateTrace:
    if quality_gate is None:
        quality_gate = make_quality_gate_trace(
            passed=passed,
            failure_reason=None if passed else "min_margin",
        )
    if transform_consistency is None:
        transform_consistency = make_bounds_gate_trace(
            passed=passed,
            failure_reason=None if passed else "min_margin",
        )
    return OuterGateTrace(
        passed=passed,
        failure_reason=failure_reason,
        quality_gate=quality_gate,
        transform_consistency=transform_consistency,
    )
