"""Acceptance decisions for rendered samples."""

from __future__ import annotations

from scripts.dataset_generation.dataset_generation.policy import DecisionReason
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.truncation import classify_truncation_mode
from scripts.dataset_generation.dataset_generation.types_render import (
    AcceptanceDecision,
    RenderResult,
)


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
        if truncation_mode == "required":
            return AcceptanceDecision(
                action="reject",
                reason=DecisionReason.POST_TRUNCATION_REQUIRED,
            )
        return AcceptanceDecision(action="accept_with_truncation")

    if truncation_mode == "required":
        return AcceptanceDecision(action="reject", reason=DecisionReason.TRUNCATION_REQUIRED)
    return AcceptanceDecision(action="accept_without_truncation")


def evaluate_render_quality(
    render_result: RenderResult,
    _recipe: ProductionRecipe,
) -> str | None:
    if render_result.rejection_reason is not None:
        return render_result.rejection_reason
    if render_result.image is None:
        return "missing_image"
    if render_result.svg_diagnostics.page_count != 1:
        return "multi_page"
    return None
