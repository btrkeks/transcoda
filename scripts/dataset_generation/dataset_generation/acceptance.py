"""Acceptance decisions for rendered samples."""

from __future__ import annotations

import numpy as np

from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.truncation import classify_truncation_mode
from scripts.dataset_generation.dataset_generation.types import AcceptanceDecision, RenderResult


def decide_acceptance(
    render_result: RenderResult,
    recipe: ProductionRecipe,
    *,
    truncation_applied: bool,
) -> AcceptanceDecision:
    quality_reason = evaluate_render_quality(render_result, recipe)
    if quality_reason is not None:
        return AcceptanceDecision(action="reject", reason=quality_reason)

    truncation_mode = classify_truncation_mode(render_result.svg_diagnostics, recipe)
    if truncation_applied:
        if truncation_mode != "forbidden":
            return AcceptanceDecision(
                action="reject",
                reason=f"post_truncation_{truncation_mode}",
            )
        return AcceptanceDecision(action="accept_with_truncation")

    if truncation_mode == "required":
        return AcceptanceDecision(action="reject", reason="truncation_required")
    return AcceptanceDecision(action="accept_without_truncation")


def evaluate_render_quality(
    render_result: RenderResult,
    recipe: ProductionRecipe,
) -> str | None:
    if render_result.rejection_reason is not None:
        return render_result.rejection_reason
    if render_result.image is None:
        return "missing_image"
    if render_result.svg_diagnostics.page_count != 1:
        return "multi_page"
    sparse, _ = _is_sparse_render(
        render_result.image,
        min_black_ratio=recipe.acceptance.sparse_render_black_ratio_min,
    )
    if sparse:
        return "sparse_render"
    return None


def _is_sparse_render(
    image: np.ndarray,
    *,
    min_black_ratio: float,
) -> tuple[bool, float]:
    gray = image[:, :, :3].mean(axis=2)
    black_ratio = float((gray <= 120.0).mean())
    return black_ratio < min_black_ratio, black_ratio
