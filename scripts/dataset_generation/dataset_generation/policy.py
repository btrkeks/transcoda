"""Typed policy helpers for the dataset-generation rewrite."""

from __future__ import annotations

from enum import StrEnum

from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.types_render import RenderResult


class RenderMode(StrEnum):
    DEFAULT = "default"
    LAYOUT_RESCUE = "layout_rescue"


class RejectionReason(StrEnum):
    MULTI_PAGE = "multi_page"
    TOP_CLEARANCE = "top_clearance"
    BOTTOM_CLEARANCE = "bottom_clearance"
    LEFT_CLEARANCE = "left_clearance"
    RIGHT_CLEARANCE = "right_clearance"
    CROP_RISK = "crop_risk"
    NO_CONTENT_DETECTED = "no_content_detected"


class DecisionReason(StrEnum):
    TRUNCATION_REQUIRED = "truncation_required"
    POST_TRUNCATION_REQUIRED = "post_truncation_required"


DEFAULT_RETRYABLE_REJECTION_REASONS = frozenset(
    {
        RejectionReason.TOP_CLEARANCE,
        RejectionReason.BOTTOM_CLEARANCE,
        RejectionReason.LEFT_CLEARANCE,
        RejectionReason.RIGHT_CLEARANCE,
        RejectionReason.CROP_RISK,
        RejectionReason.NO_CONTENT_DETECTED,
    }
)
LAYOUT_RESCUE_RETRYABLE_REJECTION_REASONS = (
    DEFAULT_RETRYABLE_REJECTION_REASONS | {RejectionReason.MULTI_PAGE}
)
LAYOUT_RESCUE_SEED_XOR = 0x5F3759DF


def is_in_preferred_band(system_count: int, recipe: ProductionRecipe) -> bool:
    return (
        recipe.truncation.preferred_min_systems
        <= system_count
        <= recipe.truncation.preferred_max_systems
    )


def is_within_layout_rescue_band(system_count: int, recipe: ProductionRecipe) -> bool:
    return system_count <= recipe.truncation.preferred_max_systems


def is_layout_rejection_reason(reason: str | None) -> bool:
    if reason is None:
        return False
    return reason in {value.value for value in LAYOUT_RESCUE_RETRYABLE_REJECTION_REASONS}


def should_attempt_layout_rescue(
    render_result: RenderResult,
    recipe: ProductionRecipe,
) -> bool:
    return is_layout_rejection_reason(render_result.rejection_reason) and is_within_layout_rescue_band(
        render_result.svg_diagnostics.system_count,
        recipe,
    )


def layout_rescue_seed(seed: int) -> int:
    return ((seed & 0xFFFFFFFF) ^ LAYOUT_RESCUE_SEED_XOR) & 0xFFFFFFFF


def finalize_failure_reason(
    *,
    full_decision_reason: str | None,
    full_render_rejection_reason: str | None,
    truncation_attempted: bool,
    truncation_mode: str | None,
) -> str:
    if truncation_mode in {"preferred", "required"} and truncation_attempted:
        return "truncation_exhausted"
    return full_decision_reason or full_render_rejection_reason or "rejected"
