"""Layout-aware offline augmentation."""

from __future__ import annotations

import numpy as np

from scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment import (
    OfflineAugmentTrace,
    evaluate_outer_gate,
    offline_augment,
)
from scripts.dataset_generation.dataset_generation.recipe import (
    ProductionRecipe,
)
from scripts.dataset_generation.dataset_generation.types_domain import AugmentationBand, SamplePlan
from scripts.dataset_generation.dataset_generation.types_events import (
    AugmentationTraceEvent,
    OuterGateTrace,
)
from scripts.dataset_generation.dataset_generation.types_render import (
    AugmentedRenderResult,
    RenderResult,
)


def select_augmentation_band(
    plan: SamplePlan,
    render_result: RenderResult,
    recipe: ProductionRecipe,
) -> AugmentationBand:
    system_count = render_result.svg_diagnostics.system_count
    fill = render_result.vertical_fill_ratio
    bottom_whitespace = render_result.bottom_whitespace_ratio

    if (
        system_count >= 6
        or plan.segment_count >= 3
        or (fill is not None and fill >= 0.82)
        or (bottom_whitespace is not None and bottom_whitespace <= 0.06)
    ):
        return "tight"
    if (
        system_count <= 3
        and plan.segment_count == 1
        and (fill is None or fill <= 0.60)
        and (bottom_whitespace is None or bottom_whitespace >= 0.14)
    ):
        return "roomy"
    return "balanced"


def augment_accepted_render(
    plan: SamplePlan,
    render_result: RenderResult,
    recipe: ProductionRecipe,
) -> AugmentedRenderResult:
    """Augment an accepted render and return both the image and a structured trace."""
    if render_result.image is None:
        raise ValueError("Cannot augment a render result without image data")

    band_name = select_augmentation_band(plan, render_result, recipe)
    band_policy = getattr(recipe.offline_aug, band_name)
    base_image = np.ascontiguousarray(render_result.image)

    augmented_image, pre_augraphy_candidate, offline_trace = offline_augment(
        base_image,
        render_layers=render_result.render_layers,
        texturize_image=True,
        filename=plan.sample_id,
        variant_idx=0,
        augment_seed=plan.seed,
        geom_x_squeeze_prob=band_policy.geom_x_squeeze_prob,
        geom_x_squeeze_min_scale=band_policy.geom_x_squeeze_min_scale,
        geom_x_squeeze_max_scale=band_policy.geom_x_squeeze_max_scale,
        geom_x_squeeze_apply_in_conservative=band_policy.geom_x_squeeze_apply_in_conservative,
    )

    outer_gate = evaluate_outer_gate(base_image, pre_augraphy_candidate)
    if not outer_gate.passed:
        result_image = base_image
    else:
        result_image = np.ascontiguousarray(augmented_image)

    final_outcome = _classify_final_outcome(offline_trace, outer_gate)

    sample_idx = int(plan.sample_id.split("_")[-1])
    trace_event = AugmentationTraceEvent(
        event="augmentation_trace",
        sample_id=plan.sample_id,
        sample_idx=sample_idx,
        seed=plan.seed,
        render_height_px=render_result.render_height_px,
        bottom_padding_px=render_result.bottom_padding_px,
        top_whitespace_px=render_result.top_whitespace_px,
        bottom_whitespace_px=render_result.bottom_whitespace_px,
        content_height_px=render_result.content_height_px,
        band=band_name,
        branch=offline_trace.branch,
        initial_geometry=offline_trace.initial_geometry,
        retry_geometry=offline_trace.retry_geometry,
        selected_geometry=offline_trace.selected_geometry,
        final_geometry_applied=offline_trace.final_geometry_applied,
        initial_oob_gate=offline_trace.initial_oob_gate,
        retry_oob_gate=offline_trace.retry_oob_gate,
        augraphy_outcome=offline_trace.augraphy_outcome,
        augraphy_normalize_accepted=offline_trace.augraphy_normalize_accepted,
        augraphy_fallback_attempted=offline_trace.augraphy_fallback_attempted,
        augraphy_fallback_outcome=offline_trace.augraphy_fallback_outcome,
        augraphy_fallback_normalize_accepted=offline_trace.augraphy_fallback_normalize_accepted,
        outer_gate=outer_gate,
        final_outcome=final_outcome,
        offline_geom_ms=offline_trace.offline_geom_ms,
        offline_gates_ms=offline_trace.offline_gates_ms,
        offline_augraphy_ms=offline_trace.offline_augraphy_ms,
        offline_texture_ms=offline_trace.offline_texture_ms,
    )

    return AugmentedRenderResult(
        final_image=result_image,
        trace=trace_event,
        base_image=base_image,
        pre_augraphy_image=pre_augraphy_candidate,
    )


def _classify_final_outcome(trace: OfflineAugmentTrace, outer_gate: OuterGateTrace) -> str:
    """Derive a single outcome label from the trace and outer gate result."""
    if trace.branch == "none":
        return "clean_early_exit"
    if not outer_gate.passed:
        return "clean_gate_rejected"
    if trace.augraphy_normalize_accepted:
        return "fully_augmented"
    if trace.augraphy_fallback_attempted and trace.augraphy_fallback_normalize_accepted:
        return "augraphy_on_base"
    return "clean_augraphy_failed"
