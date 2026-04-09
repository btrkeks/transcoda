"""Layout-aware offline augmentation."""

from __future__ import annotations

import numpy as np

from scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment import (
    offline_augment,
    passes_quality_gate,
    passes_transform_consistency,
)
from scripts.dataset_generation.dataset_generation_new.recipe import (
    OfflineAugmentationBandPolicy,
    ProductionRecipe,
)
from scripts.dataset_generation.dataset_generation_new.types import (
    AugmentationBand,
    RenderResult,
    SamplePlan,
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
) -> np.ndarray:
    if render_result.image is None:
        raise ValueError("Cannot augment a render result without image data")

    band_name = select_augmentation_band(plan, render_result, recipe)
    band_policy = getattr(recipe.offline_aug, band_name)
    base_image = np.ascontiguousarray(render_result.image)
    augmented = offline_augment(
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
    augmented_image = augmented[0] if isinstance(augmented, tuple) else augmented
    if not _passes_augmented_image_gates(base_image, augmented_image):
        return base_image
    return np.ascontiguousarray(augmented_image)


def _passes_augmented_image_gates(base_image: np.ndarray, candidate: np.ndarray) -> bool:
    return passes_quality_gate(candidate) and passes_transform_consistency(base_image, candidate)
