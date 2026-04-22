import numpy as np

from scripts.dataset_generation.dataset_generation.policy import (
    RejectionReason,
    finalize_failure_reason,
    is_in_preferred_band,
    layout_rescue_seed,
    should_attempt_layout_rescue,
)
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.types import RenderResult, SvgLayoutDiagnostics


def _render_result(*, system_count: int, rejection_reason: str | None) -> RenderResult:
    image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    return RenderResult(
        image=image if rejection_reason is None else None,
        render_layers=None,
        svg_diagnostics=SvgLayoutDiagnostics(system_count=system_count, page_count=1),
        bottom_whitespace_ratio=0.10,
        vertical_fill_ratio=0.72,
        rejection_reason=rejection_reason,
    )


def test_is_in_preferred_band_uses_recipe_bounds():
    recipe = ProductionRecipe()

    assert is_in_preferred_band(5, recipe) is True
    assert is_in_preferred_band(7, recipe) is True
    assert is_in_preferred_band(4, recipe) is False
    assert is_in_preferred_band(8, recipe) is False


def test_should_attempt_layout_rescue_requires_layout_reason_and_rescue_band():
    recipe = ProductionRecipe()

    assert should_attempt_layout_rescue(
        _render_result(system_count=6, rejection_reason=RejectionReason.MULTI_PAGE),
        recipe,
    )
    assert should_attempt_layout_rescue(
        _render_result(system_count=7, rejection_reason=RejectionReason.RIGHT_CLEARANCE),
        recipe,
    )
    assert not should_attempt_layout_rescue(
        _render_result(system_count=8, rejection_reason=RejectionReason.MULTI_PAGE),
        recipe,
    )
    assert not should_attempt_layout_rescue(
        _render_result(system_count=6, rejection_reason="render_error:boom"),
        recipe,
    )


def test_finalize_failure_reason_prefers_truncation_exhausted_when_attempted():
    assert finalize_failure_reason(
        full_decision_reason="multi_page",
        full_render_rejection_reason="multi_page",
        truncation_attempted=True,
        truncation_mode="required",
    ) == "truncation_exhausted"


def test_layout_rescue_seed_is_stable_and_changes_seed():
    seed = 12345

    assert layout_rescue_seed(seed) == layout_rescue_seed(seed)
    assert layout_rescue_seed(seed) != seed
