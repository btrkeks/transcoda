import numpy as np

from scripts.dataset_generation.dataset_generation.acceptance import (
    decide_acceptance,
    evaluate_render_quality,
)
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from tests.dataset_generation.factories import make_render_result


def test_evaluate_render_quality_accepts_sparse_images():
    image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    image[10:15, 10:15] = 0

    assert evaluate_render_quality(make_render_result(image=image), ProductionRecipe()) is None


def test_evaluate_render_quality_preserves_renderer_rejection_reason():
    result = make_render_result(image=None, rejection_reason="left_clearance")

    assert evaluate_render_quality(result, ProductionRecipe()) == "left_clearance"


def test_evaluate_render_quality_rejects_missing_image():
    assert (
        evaluate_render_quality(make_render_result(image=None), ProductionRecipe())
        == "missing_image"
    )


def test_evaluate_render_quality_rejects_multi_page_output():
    image = np.full((1485, 1050, 3), 255, dtype=np.uint8)

    assert (
        evaluate_render_quality(
            make_render_result(image=image, page_count=2), ProductionRecipe()
        )
        == "multi_page"
    )


def test_decide_acceptance_accepts_truncated_single_page_preferred_candidate():
    decision = decide_acceptance(
        make_render_result(image=np.full((1485, 1050, 3), 255, dtype=np.uint8)),
        ProductionRecipe(),
        truncation_applied=True,
    )

    assert decision.action == "accept_with_truncation"
    assert decision.reason is None


def test_decide_acceptance_rejects_truncated_required_candidate():
    decision = decide_acceptance(
        make_render_result(
            image=np.full((1485, 1050, 3), 255, dtype=np.uint8),
            system_count=8,
        ),
        ProductionRecipe(),
        truncation_applied=True,
    )

    assert decision.action == "reject"
    assert decision.reason == "post_truncation_required"
