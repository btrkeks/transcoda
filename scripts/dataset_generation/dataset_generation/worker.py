"""Worker-side sample evaluation for production dataset generation."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from scripts.dataset_generation.dataset_generation.acceptance import decide_acceptance
from scripts.dataset_generation.dataset_generation.augmentation import augment_accepted_render
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
)
from scripts.dataset_generation.dataset_generation.io import encode_jpeg_image
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.records import build_dataset_row
from scripts.dataset_generation.dataset_generation.render_transcription import (
    build_render_transcription,
    ensure_render_header,
)
from scripts.dataset_generation.dataset_generation.renderer import (
    render_sample,
    render_sample_with_layout_rescue,
)
from scripts.dataset_generation.dataset_generation.truncation import (
    build_prefix_candidates,
    classify_truncation_mode,
)
from scripts.dataset_generation.dataset_generation.types import (
    AcceptedSample,
    AugmentedRenderResult,
    AugmentationPreviewArtifacts,
    AugmentationTraceEvent,
    RenderResult,
    SamplePlan,
    VerovioDiagnosticEvent,
    WorkerFailure,
    WorkerOutcome,
    WorkerSuccess,
)

_WORKER_RECIPE: ProductionRecipe | None = None
_WORKER_RENDERER: VerovioRenderer | None = None
_WORKER_CAPTURE_VEROVIO_DIAGNOSTICS = True
_RATIONAL_DURATION_PATTERN = re.compile(r"\d%-?\d")


def compute_initial_kern_spine_count(transcription: str) -> int:
    for raw_line in transcription.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!!"):
            continue
        return raw_line.count("\t") + 1
    raise ValueError("Cannot infer initial kern spine count from empty transcription")


def is_valid_kern(content: str) -> tuple[bool, str | None]:
    if _RATIONAL_DURATION_PATTERN.search(content):
        return False, "rational duration (corrupted source)"
    return True, None


def ensure_kern_header(content: str) -> str:
    return ensure_render_header(content)


def init_generation_worker(
    recipe: ProductionRecipe,
    capture_verovio_diagnostics: bool = True,
) -> None:
    global _WORKER_RECIPE, _WORKER_RENDERER, _WORKER_CAPTURE_VEROVIO_DIAGNOSTICS
    _WORKER_RECIPE = recipe
    _WORKER_RENDERER = VerovioRenderer()
    _WORKER_CAPTURE_VEROVIO_DIAGNOSTICS = capture_verovio_diagnostics


def process_sample_plan(plan: SamplePlan) -> WorkerOutcome:
    assert _WORKER_RECIPE is not None, "Worker recipe not initialized"
    assert _WORKER_RENDERER is not None, "Worker renderer not initialized"
    return evaluate_sample_plan(
        plan,
        recipe=_WORKER_RECIPE,
        renderer=_WORKER_RENDERER,
        capture_verovio_diagnostics=_WORKER_CAPTURE_VEROVIO_DIAGNOSTICS,
    )


def process_calibration_sample_plan(plan: SamplePlan) -> WorkerOutcome:
    assert _WORKER_RECIPE is not None, "Worker recipe not initialized"
    assert _WORKER_RENDERER is not None, "Worker renderer not initialized"
    return evaluate_sample_plan(
        plan,
        recipe=_WORKER_RECIPE,
        renderer=_WORKER_RENDERER,
        augment_fn=_skip_calibration_image_bytes,
        capture_verovio_diagnostics=_WORKER_CAPTURE_VEROVIO_DIAGNOSTICS,
    )


def _skip_calibration_image_bytes(
    plan: SamplePlan,
    render_result: RenderResult,
    recipe: ProductionRecipe,
) -> bytes:
    del plan, render_result, recipe
    return b""


def evaluate_sample_plan(
    plan: SamplePlan,
    *,
    recipe: ProductionRecipe,
    renderer: VerovioRenderer,
    render_fn: Callable[..., RenderResult] = render_sample,
    rescue_render_fn: Callable[..., RenderResult] | None = None,
    augment_fn: Callable[..., object] = augment_accepted_render,
    capture_verovio_diagnostics: bool = True,
) -> WorkerOutcome:
    verovio_events: list[VerovioDiagnosticEvent] = []
    render_transcription = build_render_transcription(plan.label_transcription, recipe, seed=plan.seed)
    full_render = _call_render(
        render_fn,
        render_transcription,
        recipe,
        seed=plan.seed,
        renderer=renderer,
        capture_verovio_diagnostics=capture_verovio_diagnostics,
    )
    verovio_events.extend(_build_verovio_events(plan=plan, stage="full", seed=plan.seed, result=full_render))
    full_decision = decide_acceptance(full_render, recipe, truncation_applied=False)
    truncation_mode = classify_truncation_mode(full_render.svg_diagnostics, recipe)
    full_render_system_count = full_render.svg_diagnostics.system_count
    full_render_content_height_px = full_render.content_height_px
    full_render_vertical_fill_ratio = full_render.vertical_fill_ratio
    full_render_rejection_reason = full_render.rejection_reason
    preferred_5_6_rescue_attempted = False
    preferred_5_6_rescue_succeeded = False
    truncation_attempted = False

    if full_decision.action == "accept_without_truncation":
        preferred_status = (
            "preferred_5_6_accepted_full"
            if truncation_mode == "preferred"
            and recipe.truncation.preferred_min_systems
            <= full_render_system_count
            <= recipe.truncation.preferred_max_systems
            else None
        )
        sample, aug_trace, aug_preview = _finalize_sample(
            plan=plan,
            render_result=full_render,
            transcription=plan.label_transcription,
            truncation_applied=False,
            truncation_ratio=None,
            truncation_reason=None,
            recipe=recipe,
            augment_fn=augment_fn,
        )
        return WorkerSuccess(
            sample=sample,
            truncation_attempted=False,
            truncation_rescued=False,
            full_render_system_count=full_render_system_count,
            full_render_content_height_px=full_render_content_height_px,
            full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
            full_render_rejection_reason=full_render_rejection_reason,
            accepted_render_system_count=full_render_system_count,
            preferred_5_6_rescue_attempted=False,
            preferred_5_6_rescue_succeeded=False,
            preferred_5_6_status=preferred_status,
            verovio_diagnostics=tuple(verovio_events),
            augmentation_trace=aug_trace,
            augmentation_preview=aug_preview,
        )

    if (
        truncation_mode == "preferred"
        and recipe.truncation.preferred_min_systems
        <= full_render_system_count
        <= recipe.truncation.preferred_max_systems
    ):
        preferred_5_6_rescue_attempted = True
        rescue_seed = ((plan.seed & 0xFFFFFFFF) ^ 0x5F3759DF) & 0xFFFFFFFF
        rescue_callable = rescue_render_fn
        if rescue_callable is None and render_fn is render_sample:
            rescue_callable = render_sample_with_layout_rescue
        rescue_render = _call_render(
            rescue_callable or render_fn,
            render_transcription,
            recipe,
            seed=rescue_seed,
            renderer=renderer,
            capture_verovio_diagnostics=capture_verovio_diagnostics,
        )
        verovio_events.extend(
            _build_verovio_events(
                plan=plan,
                stage="preferred_5_6_rescue",
                seed=rescue_seed,
                result=rescue_render,
            )
        )
        rescue_decision = decide_acceptance(
            rescue_render,
            recipe,
            truncation_applied=False,
        )
        if rescue_decision.action == "accept_without_truncation":
            preferred_5_6_rescue_succeeded = True
            sample, aug_trace, aug_preview = _finalize_sample(
                plan=plan,
                render_result=rescue_render,
                transcription=plan.label_transcription,
                truncation_applied=False,
                truncation_ratio=None,
                truncation_reason=None,
                recipe=recipe,
                augment_fn=augment_fn,
            )
            return WorkerSuccess(
                sample=sample,
                truncation_attempted=False,
                truncation_rescued=False,
                full_render_system_count=full_render_system_count,
                full_render_content_height_px=full_render_content_height_px,
                full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
                full_render_rejection_reason=full_render_rejection_reason,
                accepted_render_system_count=rescue_render.svg_diagnostics.system_count,
                preferred_5_6_rescue_attempted=True,
                preferred_5_6_rescue_succeeded=True,
                preferred_5_6_status="preferred_5_6_rescued",
                verovio_diagnostics=tuple(verovio_events),
                augmentation_trace=aug_trace,
                augmentation_preview=aug_preview,
            )

    if truncation_mode in {"preferred", "required"}:
        for candidate in build_prefix_candidates(plan.label_transcription, recipe):
            truncation_attempted = True
            candidate_seed = (plan.seed + candidate.chunk_count * 17) & 0xFFFFFFFF
            candidate_render = _call_render(
                render_fn,
                build_render_transcription(candidate.transcription, recipe, seed=candidate_seed),
                recipe,
                seed=candidate_seed,
                renderer=renderer,
                capture_verovio_diagnostics=capture_verovio_diagnostics,
            )
            verovio_events.extend(
                _build_verovio_events(
                    plan=plan,
                    stage="truncation_candidate",
                    seed=candidate_seed,
                    result=candidate_render,
                    truncation_chunk_count=candidate.chunk_count,
                    truncation_total_chunks=candidate.total_chunks,
                    truncation_ratio=candidate.ratio,
                )
            )
            candidate_decision = decide_acceptance(
                candidate_render,
                recipe,
                truncation_applied=True,
            )
            if candidate_decision.action == "accept_with_truncation":
                sample, aug_trace, aug_preview = _finalize_sample(
                    plan=plan,
                    render_result=candidate_render,
                    transcription=candidate.transcription,
                    truncation_applied=True,
                    truncation_ratio=candidate.ratio,
                    truncation_reason="system_count_policy",
                    recipe=recipe,
                    augment_fn=augment_fn,
                )
                return WorkerSuccess(
                    sample=sample,
                    truncation_attempted=True,
                    truncation_rescued=True,
                    full_render_system_count=full_render_system_count,
                    full_render_content_height_px=full_render_content_height_px,
                    full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
                    full_render_rejection_reason=full_render_rejection_reason,
                    accepted_render_system_count=candidate_render.svg_diagnostics.system_count,
                    preferred_5_6_rescue_attempted=preferred_5_6_rescue_attempted,
                    preferred_5_6_rescue_succeeded=preferred_5_6_rescue_succeeded,
                    preferred_5_6_status=(
                        "preferred_5_6_truncated"
                        if truncation_mode == "preferred"
                        and recipe.truncation.preferred_min_systems
                        <= full_render_system_count
                        <= recipe.truncation.preferred_max_systems
                        else None
                    ),
                    verovio_diagnostics=tuple(verovio_events),
                    augmentation_trace=aug_trace,
                    augmentation_preview=aug_preview,
                )

    failure_reason = full_decision.reason or full_render.rejection_reason or "rejected"
    if truncation_mode in {"preferred", "required"} and truncation_attempted:
        failure_reason = "truncation_exhausted"
    return WorkerFailure(
        sample_id=plan.sample_id,
        failure_reason=failure_reason,
        truncation_attempted=truncation_attempted,
        truncation_rescued=False,
        full_render_system_count=full_render_system_count,
        full_render_content_height_px=full_render_content_height_px,
        full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
        full_render_rejection_reason=full_render_rejection_reason,
        accepted_render_system_count=None,
        preferred_5_6_rescue_attempted=preferred_5_6_rescue_attempted,
        preferred_5_6_rescue_succeeded=False,
        preferred_5_6_status=(
            "preferred_5_6_failed"
            if truncation_mode == "preferred"
            and recipe.truncation.preferred_min_systems
            <= full_render_system_count
            <= recipe.truncation.preferred_max_systems
            else None
        ),
        verovio_diagnostics=tuple(verovio_events),
    )


def _call_render(
    render_callable: Callable[..., RenderResult],
    render_transcription: str,
    recipe: ProductionRecipe,
    *,
    seed: int,
    renderer: VerovioRenderer,
    capture_verovio_diagnostics: bool,
) -> RenderResult:
    kwargs = {
        "seed": seed,
        "renderer": renderer,
    }
    if _callable_accepts_capture_flag(render_callable):
        kwargs["capture_verovio_diagnostics"] = capture_verovio_diagnostics
    return render_callable(render_transcription, recipe, **kwargs)


def _callable_accepts_capture_flag(render_callable: Callable[..., object]) -> bool:
    try:
        signature = inspect.signature(render_callable)
    except (TypeError, ValueError):
        return False
    if "capture_verovio_diagnostics" in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _build_verovio_events(
    *,
    plan: SamplePlan,
    stage: str,
    seed: int,
    result: RenderResult,
    truncation_chunk_count: int | None = None,
    truncation_total_chunks: int | None = None,
    truncation_ratio: float | None = None,
) -> tuple[VerovioDiagnosticEvent, ...]:
    sample_idx = int(plan.sample_id.split("_")[-1])
    source_paths = tuple(str(Path(segment.path).resolve()) for segment in plan.segments)
    return tuple(
        VerovioDiagnosticEvent(
            event="verovio_diagnostic",
            sample_id=plan.sample_id,
            sample_idx=sample_idx,
            source_paths=source_paths,
            stage=stage,
            seed=seed,
            render_attempt_idx=diagnostic.render_attempt_idx,
            diagnostic_kind=diagnostic.diagnostic_kind,
            raw_message=diagnostic.raw_message,
            near_line=diagnostic.near_line,
            expected_duration_from_start=diagnostic.expected_duration_from_start,
            found_duration_from_start=diagnostic.found_duration_from_start,
            line_text=diagnostic.line_text,
            truncation_chunk_count=truncation_chunk_count,
            truncation_total_chunks=truncation_total_chunks,
            truncation_ratio=truncation_ratio,
        )
        for diagnostic in result.verovio_diagnostics
    )


def _finalize_sample(
    *,
    plan: SamplePlan,
    render_result: RenderResult,
    transcription: str,
    truncation_applied: bool,
    truncation_ratio: float | None,
    truncation_reason: str | None,
    recipe: ProductionRecipe,
    augment_fn: Callable[..., object],
) -> tuple[AcceptedSample, AugmentationTraceEvent | None, AugmentationPreviewArtifacts | None]:
    augmented_result = augment_fn(plan, render_result, recipe)
    augmentation_preview: AugmentationPreviewArtifacts | None = None
    if isinstance(augmented_result, AugmentedRenderResult):
        augmented_image = augmented_result.final_image
        aug_trace = augmented_result.trace
        if (
            isinstance(augmented_result.final_image, np.ndarray)
            and isinstance(augmented_result.base_image, np.ndarray)
            and isinstance(augmented_result.pre_augraphy_image, np.ndarray)
        ):
            augmentation_preview = AugmentationPreviewArtifacts(
                base_image_jpeg=encode_jpeg_image(augmented_result.base_image),
                pre_augraphy_image_jpeg=encode_jpeg_image(augmented_result.pre_augraphy_image),
                final_image_jpeg=encode_jpeg_image(augmented_result.final_image),
            )
    elif isinstance(augmented_result, tuple):
        augmented_image, aug_trace = augmented_result
    else:
        augmented_image = augmented_result
        aug_trace = None
    if not isinstance(augmented_image, bytes):
        image_bytes = encode_jpeg_image(augmented_image)
    else:
        image_bytes = augmented_image
    sample = AcceptedSample(
        sample_id=plan.sample_id,
        label_transcription=transcription,
        image_bytes=image_bytes,
        initial_kern_spine_count=compute_initial_kern_spine_count(transcription),
        segment_count=plan.segment_count,
        source_ids=tuple(segment.source_id for segment in plan.segments),
        source_measure_count=plan.source_measure_count,
        source_non_empty_line_count=plan.source_non_empty_line_count,
        system_count=render_result.svg_diagnostics.system_count,
        truncation_applied=truncation_applied,
        truncation_reason=truncation_reason,
        truncation_ratio=truncation_ratio,
        bottom_whitespace_ratio=render_result.bottom_whitespace_ratio,
        vertical_fill_ratio=render_result.vertical_fill_ratio,
        bottom_whitespace_px=render_result.bottom_whitespace_px,
        top_whitespace_px=render_result.top_whitespace_px,
        content_height_px=render_result.content_height_px,
    )
    return sample, aug_trace, augmentation_preview


def outcome_to_dataset_row(
    outcome: WorkerSuccess,
    *,
    recipe: ProductionRecipe,
) -> dict[str, object]:
    return build_dataset_row(outcome.sample, recipe=recipe)
