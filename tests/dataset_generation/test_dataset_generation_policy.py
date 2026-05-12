from scripts.dataset_generation.dataset_generation.policy import (
    RejectionReason,
    finalize_failure_reason,
    is_in_preferred_band,
    layout_rescue_seed,
    should_attempt_layout_rescue,
)
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from tests.dataset_generation.factories import make_render_result


def test_is_in_preferred_band_uses_recipe_bounds():
    recipe = ProductionRecipe()

    assert is_in_preferred_band(5, recipe) is True
    assert is_in_preferred_band(7, recipe) is True
    assert is_in_preferred_band(4, recipe) is False
    assert is_in_preferred_band(8, recipe) is False


def test_should_attempt_layout_rescue_requires_layout_reason_and_rescue_band():
    recipe = ProductionRecipe()

    assert should_attempt_layout_rescue(
        make_render_result(system_count=6, rejection_reason=RejectionReason.MULTI_PAGE),
        recipe,
    )
    assert should_attempt_layout_rescue(
        make_render_result(system_count=7, rejection_reason=RejectionReason.RIGHT_CLEARANCE),
        recipe,
    )
    assert not should_attempt_layout_rescue(
        make_render_result(system_count=8, rejection_reason=RejectionReason.MULTI_PAGE),
        recipe,
    )
    assert not should_attempt_layout_rescue(
        make_render_result(system_count=6, rejection_reason="render_error:boom"),
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
