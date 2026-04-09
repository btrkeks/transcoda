from scripts.dataset_generation.dataset_generation_new.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation_new.truncation import classify_truncation_mode
from scripts.dataset_generation.dataset_generation_new.types import SvgLayoutDiagnostics


def test_truncation_mode_is_forbidden_for_four_or_fewer_systems():
    recipe = ProductionRecipe()

    mode = classify_truncation_mode(
        SvgLayoutDiagnostics(system_count=4, page_count=1),
        recipe,
    )

    assert mode == "forbidden"


def test_truncation_mode_is_preferred_for_five_or_six_systems():
    recipe = ProductionRecipe()

    mode = classify_truncation_mode(
        SvgLayoutDiagnostics(system_count=6, page_count=1),
        recipe,
    )

    assert mode == "preferred"


def test_truncation_mode_is_required_for_overfull_or_multi_page_renders():
    recipe = ProductionRecipe()

    assert (
        classify_truncation_mode(
            SvgLayoutDiagnostics(system_count=7, page_count=1),
            recipe,
        )
        == "required"
    )
    assert (
        classify_truncation_mode(
            SvgLayoutDiagnostics(system_count=3, page_count=2),
            recipe,
        )
        == "required"
    )
