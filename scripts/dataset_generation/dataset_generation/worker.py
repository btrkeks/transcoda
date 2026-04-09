"""Worker-side sample evaluation for production dataset generation."""

from __future__ import annotations

from typing import Callable

from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
)
from scripts.dataset_generation.dataset_generation_new.acceptance import decide_acceptance
from scripts.dataset_generation.dataset_generation_new.augmentation import augment_accepted_render
from scripts.dataset_generation.dataset_generation_new.io import encode_jpeg_image
from scripts.dataset_generation.dataset_generation_new.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation_new.records import build_dataset_row
from scripts.dataset_generation.dataset_generation_new.render_transcription import (
    build_render_transcription,
)
from scripts.dataset_generation.dataset_generation_new.renderer import render_sample
from scripts.dataset_generation.dataset_generation_new.renderer import (
    render_sample_with_layout_rescue,
)
from scripts.dataset_generation.dataset_generation_new.truncation import (
    build_prefix_candidates,
    classify_truncation_mode,
)
from scripts.dataset_generation.dataset_generation_new.types import (
    AcceptedSample,
    RenderResult,
    SamplePlan,
    WorkerFailure,
    WorkerOutcome,
    WorkerSuccess,
)

_WORKER_RECIPE: ProductionRecipe | None = None
_WORKER_RENDERER: VerovioRenderer | None = None


def init_generation_worker(recipe: ProductionRecipe) -> None:
    global _WORKER_RECIPE, _WORKER_RENDERER
    _WORKER_RECIPE = recipe
    _WORKER_RENDERER = VerovioRenderer()


def process_sample_plan(plan: SamplePlan) -> WorkerOutcome:
    assert _WORKER_RECIPE is not None, "Worker recipe not initialized"
    assert _WORKER_RENDERER is not None, "Worker renderer not initialized"
    return evaluate_sample_plan(
        plan,
        recipe=_WORKER_RECIPE,
        renderer=_WORKER_RENDERER,
    )


def evaluate_sample_plan(
    plan: SamplePlan,
    *,
    recipe: ProductionRecipe,
    renderer: VerovioRenderer,
    render_fn: Callable[..., RenderResult] = render_sample,
    rescue_render_fn: Callable[..., RenderResult] | None = None,
    augment_fn: Callable[..., object] = augment_accepted_render,
) -> WorkerOutcome:
    render_transcription = build_render_transcription(plan.label_transcription, recipe, seed=plan.seed)
    full_render = render_fn(
        render_transcription,
        recipe,
        seed=plan.seed,
        renderer=renderer,
    )
    full_decision = decide_acceptance(full_render, recipe, truncation_applied=False)
    truncation_mode = classify_truncation_mode(full_render.svg_diagnostics, recipe)
    full_render_system_count = full_render.svg_diagnostics.system_count
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
        return WorkerSuccess(
            sample=_finalize_sample(
                plan=plan,
                render_result=full_render,
                transcription=plan.label_transcription,
                truncation_applied=False,
                truncation_ratio=None,
                truncation_reason=None,
                recipe=recipe,
                augment_fn=augment_fn,
            ),
            truncation_attempted=False,
            truncation_rescued=False,
            full_render_system_count=full_render_system_count,
            accepted_render_system_count=full_render_system_count,
            preferred_5_6_rescue_attempted=False,
            preferred_5_6_rescue_succeeded=False,
            preferred_5_6_status=preferred_status,
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
        rescue_render = (rescue_callable or render_fn)(
            render_transcription,
            recipe,
            seed=rescue_seed,
            renderer=renderer,
        )
        rescue_decision = decide_acceptance(
            rescue_render,
            recipe,
            truncation_applied=False,
        )
        if rescue_decision.action == "accept_without_truncation":
            preferred_5_6_rescue_succeeded = True
            return WorkerSuccess(
                sample=_finalize_sample(
                    plan=plan,
                    render_result=rescue_render,
                    transcription=plan.label_transcription,
                    truncation_applied=False,
                    truncation_ratio=None,
                    truncation_reason=None,
                    recipe=recipe,
                    augment_fn=augment_fn,
                ),
                truncation_attempted=False,
                truncation_rescued=False,
                full_render_system_count=full_render_system_count,
                accepted_render_system_count=rescue_render.svg_diagnostics.system_count,
                preferred_5_6_rescue_attempted=True,
                preferred_5_6_rescue_succeeded=True,
                preferred_5_6_status="preferred_5_6_rescued",
            )

    if truncation_mode in {"preferred", "required"}:
        for candidate in build_prefix_candidates(plan.label_transcription, recipe):
            truncation_attempted = True
            candidate_seed = (plan.seed + candidate.chunk_count * 17) & 0xFFFFFFFF
            candidate_render = render_fn(
                build_render_transcription(candidate.transcription, recipe, seed=candidate_seed),
                recipe,
                seed=candidate_seed,
                renderer=renderer,
            )
            candidate_decision = decide_acceptance(
                candidate_render,
                recipe,
                truncation_applied=True,
            )
            if candidate_decision.action == "accept_with_truncation":
                return WorkerSuccess(
                    sample=_finalize_sample(
                        plan=plan,
                        render_result=candidate_render,
                        transcription=candidate.transcription,
                        truncation_applied=True,
                        truncation_ratio=candidate.ratio,
                        truncation_reason="system_count_policy",
                        recipe=recipe,
                        augment_fn=augment_fn,
                    ),
                    truncation_attempted=True,
                    truncation_rescued=True,
                    full_render_system_count=full_render_system_count,
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
) -> AcceptedSample:
    augmented_image = augment_fn(plan, render_result, recipe)
    if not isinstance(augmented_image, bytes):
        image_bytes = encode_jpeg_image(augmented_image)
    else:
        image_bytes = augmented_image
    return AcceptedSample(
        sample_id=plan.sample_id,
        label_transcription=transcription,
        image_bytes=image_bytes,
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


def outcome_to_dataset_row(
    outcome: WorkerSuccess,
    *,
    recipe: ProductionRecipe,
) -> dict[str, object]:
    return build_dataset_row(outcome.sample, recipe=recipe)
