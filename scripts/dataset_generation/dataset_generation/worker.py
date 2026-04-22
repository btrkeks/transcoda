"""Worker-side sample evaluation for production dataset generation."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from scripts.dataset_generation.dataset_generation.acceptance import decide_acceptance
from scripts.dataset_generation.dataset_generation.attempts import (
    AttemptLedger,
    ExecutedRenderAttempt,
    RenderAttemptPlan,
    execute_render_attempt,
)
from scripts.dataset_generation.dataset_generation.augmentation import augment_accepted_render
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
)
from scripts.dataset_generation.dataset_generation.io import encode_jpeg_image
from scripts.dataset_generation.dataset_generation.policy import (
    RenderMode,
    finalize_failure_reason,
    is_in_preferred_band,
    layout_rescue_seed,
    should_attempt_layout_rescue,
)
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
    AttemptStageName,
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
    attempt_ledger = AttemptLedger()
    render_transcription = build_render_transcription(plan.label_transcription, recipe, seed=plan.seed)
    full_attempt = _execute_attempt(
        sample_plan=plan,
        attempt_ledger=attempt_ledger,
        attempt_plan=RenderAttemptPlan(
            stage=AttemptStageName.FULL,
            seed=plan.seed,
            render_transcription=render_transcription,
            truncation_applied=False,
        ),
        recipe=recipe,
        renderer=renderer,
        render_callable=render_fn,
        capture_verovio_diagnostics=capture_verovio_diagnostics,
    )
    full_render = full_attempt.render_result
    full_decision = full_attempt.decision
    truncation_mode = classify_truncation_mode(full_render.svg_diagnostics, recipe)
    full_render_system_count = full_render.svg_diagnostics.system_count
    full_render_content_height_px = full_render.content_height_px
    full_render_vertical_fill_ratio = full_render.vertical_fill_ratio
    full_render_rejection_reason = full_render.rejection_reason
    full_render_in_preferred_band = truncation_mode == "preferred" and is_in_preferred_band(
        full_render_system_count,
        recipe,
    )
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
            verovio_diagnostics=attempt_ledger.verovio_tuple(),
            augmentation_trace=aug_trace,
            augmentation_preview=aug_preview,
        )

    if should_attempt_layout_rescue(full_render, recipe):
        preferred_5_6_rescue_attempted = full_render_in_preferred_band
        rescue_attempt = _execute_layout_rescue_attempt(
            sample_plan=plan,
            attempt_ledger=attempt_ledger,
            base_seed=plan.seed,
            stage=AttemptStageName.FULL_LAYOUT_RESCUE,
            render_transcription=render_transcription,
            recipe=recipe,
            renderer=renderer,
            render_fn=render_fn,
            rescue_render_fn=rescue_render_fn,
            truncation_applied=False,
            capture_verovio_diagnostics=capture_verovio_diagnostics,
        )
        rescue_render = rescue_attempt.render_result
        rescue_decision = rescue_attempt.decision
        if rescue_decision.action == "accept_without_truncation":
            preferred_5_6_rescue_succeeded = full_render_in_preferred_band
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
                preferred_5_6_rescue_attempted=full_render_in_preferred_band,
                preferred_5_6_rescue_succeeded=full_render_in_preferred_band,
                preferred_5_6_status=(
                    "preferred_5_6_rescued" if full_render_in_preferred_band else None
                ),
                verovio_diagnostics=attempt_ledger.verovio_tuple(),
                augmentation_trace=aug_trace,
                augmentation_preview=aug_preview,
            )

    if truncation_mode in {"preferred", "required"}:
        for candidate in build_prefix_candidates(plan.label_transcription, recipe):
            truncation_attempted = True
            candidate_seed = (plan.seed + candidate.chunk_count * 17) & 0xFFFFFFFF
            candidate_render_transcription = build_render_transcription(
                candidate.transcription,
                recipe,
                seed=candidate_seed,
            )
            candidate_attempt = _execute_attempt(
                sample_plan=plan,
                attempt_ledger=attempt_ledger,
                attempt_plan=RenderAttemptPlan(
                    stage=AttemptStageName.TRUNCATION_CANDIDATE,
                    seed=candidate_seed,
                    render_transcription=candidate_render_transcription,
                    truncation_applied=True,
                    chunk_count=candidate.chunk_count,
                    total_chunks=candidate.total_chunks,
                    ratio=candidate.ratio,
                ),
                recipe=recipe,
                renderer=renderer,
                render_callable=render_fn,
                capture_verovio_diagnostics=capture_verovio_diagnostics,
            )
            candidate_render = candidate_attempt.render_result
            candidate_decision = candidate_attempt.decision
            if candidate_decision.action == "accept_with_truncation":
                return _build_truncated_success(
                    plan=plan,
                    recipe=recipe,
                    augment_fn=augment_fn,
                    render_result=candidate_render,
                    transcription=candidate.transcription,
                    truncation_ratio=candidate.ratio,
                    truncation_attempted=truncation_attempted,
                    full_render_system_count=full_render_system_count,
                    full_render_content_height_px=full_render_content_height_px,
                    full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
                    full_render_rejection_reason=full_render_rejection_reason,
                    preferred_5_6_rescue_attempted=preferred_5_6_rescue_attempted,
                    preferred_5_6_rescue_succeeded=preferred_5_6_rescue_succeeded,
                    preferred_5_6_status=(
                        "preferred_5_6_truncated" if full_render_in_preferred_band else None
                    ),
                    verovio_events=attempt_ledger.verovio_tuple(),
                )

            if should_attempt_layout_rescue(candidate_render, recipe):
                rescued_candidate_attempt = _execute_layout_rescue_attempt(
                    sample_plan=plan,
                    attempt_ledger=attempt_ledger,
                    base_seed=candidate_seed,
                    stage=AttemptStageName.TRUNCATION_CANDIDATE_LAYOUT_RESCUE,
                    render_transcription=candidate_render_transcription,
                    recipe=recipe,
                    renderer=renderer,
                    render_fn=render_fn,
                    rescue_render_fn=rescue_render_fn,
                    truncation_applied=True,
                    chunk_count=candidate.chunk_count,
                    total_chunks=candidate.total_chunks,
                    ratio=candidate.ratio,
                    capture_verovio_diagnostics=capture_verovio_diagnostics,
                )
                rescued_candidate_render = rescued_candidate_attempt.render_result
                rescued_candidate_decision = rescued_candidate_attempt.decision
                if rescued_candidate_decision.action == "accept_with_truncation":
                    return _build_truncated_success(
                        plan=plan,
                        recipe=recipe,
                        augment_fn=augment_fn,
                        render_result=rescued_candidate_render,
                        transcription=candidate.transcription,
                        truncation_ratio=candidate.ratio,
                        truncation_attempted=truncation_attempted,
                        full_render_system_count=full_render_system_count,
                        full_render_content_height_px=full_render_content_height_px,
                        full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
                        full_render_rejection_reason=full_render_rejection_reason,
                        preferred_5_6_rescue_attempted=preferred_5_6_rescue_attempted,
                        preferred_5_6_rescue_succeeded=preferred_5_6_rescue_succeeded,
                        preferred_5_6_status=(
                            "preferred_5_6_truncated" if full_render_in_preferred_band else None
                        ),
                        verovio_events=attempt_ledger.verovio_tuple(),
                    )

    failure_reason = finalize_failure_reason(
        full_decision_reason=full_decision.reason,
        full_render_rejection_reason=full_render.rejection_reason,
        truncation_attempted=truncation_attempted,
        truncation_mode=truncation_mode,
    )
    return WorkerFailure(
        sample_id=plan.sample_id,
        failure_reason=failure_reason,
        truncation_attempted=truncation_attempted,
        truncation_rescued=False,
        truncation_mode=truncation_mode,
        full_render_system_count=full_render_system_count,
        full_render_content_height_px=full_render_content_height_px,
        full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
        full_render_rejection_reason=full_render_rejection_reason,
        accepted_render_system_count=None,
        preferred_5_6_rescue_attempted=preferred_5_6_rescue_attempted,
        preferred_5_6_rescue_succeeded=False,
        preferred_5_6_status=(
            "preferred_5_6_failed" if full_render_in_preferred_band else None
        ),
        failure_attempts=attempt_ledger.failure_tuple(),
        verovio_diagnostics=attempt_ledger.verovio_tuple(),
    )


def _build_truncated_success(
    *,
    plan: SamplePlan,
    recipe: ProductionRecipe,
    augment_fn: Callable[..., object],
    render_result: RenderResult,
    transcription: str,
    truncation_ratio: float,
    truncation_attempted: bool,
    full_render_system_count: int | None,
    full_render_content_height_px: int | None,
    full_render_vertical_fill_ratio: float | None,
    full_render_rejection_reason: str | None,
    preferred_5_6_rescue_attempted: bool,
    preferred_5_6_rescue_succeeded: bool,
    preferred_5_6_status: str | None,
    verovio_events: tuple[VerovioDiagnosticEvent, ...],
) -> WorkerSuccess:
    sample, aug_trace, aug_preview = _finalize_sample(
        plan=plan,
        render_result=render_result,
        transcription=transcription,
        truncation_applied=True,
        truncation_ratio=truncation_ratio,
        truncation_reason="system_count_policy",
        recipe=recipe,
        augment_fn=augment_fn,
    )
    return WorkerSuccess(
        sample=sample,
        truncation_attempted=truncation_attempted,
        truncation_rescued=True,
        full_render_system_count=full_render_system_count,
        full_render_content_height_px=full_render_content_height_px,
        full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
        full_render_rejection_reason=full_render_rejection_reason,
        accepted_render_system_count=render_result.svg_diagnostics.system_count,
        preferred_5_6_rescue_attempted=preferred_5_6_rescue_attempted,
        preferred_5_6_rescue_succeeded=preferred_5_6_rescue_succeeded,
        preferred_5_6_status=preferred_5_6_status,
        verovio_diagnostics=verovio_events,
        augmentation_trace=aug_trace,
        augmentation_preview=aug_preview,
    )


def _execute_attempt(
    *,
    sample_plan: SamplePlan,
    attempt_ledger: AttemptLedger,
    attempt_plan: RenderAttemptPlan,
    recipe: ProductionRecipe,
    renderer: VerovioRenderer,
    render_callable: Callable[..., RenderResult],
    capture_verovio_diagnostics: bool,
) -> ExecutedRenderAttempt:
    attempt = execute_render_attempt(
        sample_plan=sample_plan,
        attempt_plan=attempt_plan,
        recipe=recipe,
        renderer=renderer,
        render_callable=render_callable,
        capture_verovio_diagnostics=capture_verovio_diagnostics,
    )
    attempt_ledger.record(attempt)
    return attempt


def _execute_layout_rescue_attempt(
    *,
    sample_plan: SamplePlan,
    attempt_ledger: AttemptLedger,
    base_seed: int,
    stage: AttemptStageName,
    render_transcription: str,
    recipe: ProductionRecipe,
    renderer: VerovioRenderer,
    render_fn: Callable[..., RenderResult],
    rescue_render_fn: Callable[..., RenderResult] | None,
    truncation_applied: bool,
    chunk_count: int | None = None,
    total_chunks: int | None = None,
    ratio: float | None = None,
    capture_verovio_diagnostics: bool,
) -> ExecutedRenderAttempt:
    rescue_callable = rescue_render_fn
    if rescue_callable is None and render_fn is render_sample:
        rescue_callable = render_sample_with_layout_rescue
    return _execute_attempt(
        sample_plan=sample_plan,
        attempt_ledger=attempt_ledger,
        attempt_plan=RenderAttemptPlan(
            stage=stage,
            seed=layout_rescue_seed(base_seed),
            render_transcription=render_transcription,
            truncation_applied=truncation_applied,
            chunk_count=chunk_count,
            total_chunks=total_chunks,
            ratio=ratio,
        ),
        recipe=recipe,
        renderer=renderer,
        render_callable=rescue_callable or render_fn,
        capture_verovio_diagnostics=capture_verovio_diagnostics,
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
