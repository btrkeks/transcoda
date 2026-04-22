import numpy as np

from scripts.dataset_generation.dataset_generation.acceptance import evaluate_render_quality
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.types import RenderResult, SvgLayoutDiagnostics


def _render_result(
    *,
    image: np.ndarray | None,
    page_count: int = 1,
    rejection_reason: str | None = None,
) -> RenderResult:
    return RenderResult(
        image=image,
        render_layers=None,
        svg_diagnostics=SvgLayoutDiagnostics(system_count=4, page_count=page_count),
        bottom_whitespace_ratio=0.10,
        vertical_fill_ratio=0.72,
        rejection_reason=rejection_reason,
    )


def test_evaluate_render_quality_accepts_sparse_images():
    image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    image[10:15, 10:15] = 0

    assert evaluate_render_quality(_render_result(image=image), ProductionRecipe()) is None


def test_evaluate_render_quality_preserves_renderer_rejection_reason():
    result = _render_result(image=None, rejection_reason="left_clearance")

    assert evaluate_render_quality(result, ProductionRecipe()) == "left_clearance"


def test_evaluate_render_quality_rejects_missing_image():
    assert evaluate_render_quality(_render_result(image=None), ProductionRecipe()) == "missing_image"


def test_evaluate_render_quality_rejects_multi_page_output():
    image = np.full((1485, 1050, 3), 255, dtype=np.uint8)

    assert evaluate_render_quality(_render_result(image=image, page_count=2), ProductionRecipe()) == (
        "multi_page"
    )
