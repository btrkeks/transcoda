from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.truncation import classify_truncation_mode
from scripts.dataset_generation.dataset_generation.truncation import (
    validate_truncation_candidate_contains_music,
)
from scripts.dataset_generation.dataset_generation.types_render import SvgLayoutDiagnostics


def test_truncation_mode_is_forbidden_for_four_or_fewer_systems():
    recipe = ProductionRecipe()

    mode = classify_truncation_mode(
        SvgLayoutDiagnostics(system_count=4, page_count=1),
        recipe,
    )

    assert mode == "forbidden"


def test_truncation_mode_is_preferred_for_five_to_seven_systems():
    recipe = ProductionRecipe()

    mode = classify_truncation_mode(
        SvgLayoutDiagnostics(system_count=7, page_count=1),
        recipe,
    )

    assert mode == "preferred"


def test_truncation_mode_is_required_for_overfull_or_multi_page_renders():
    recipe = ProductionRecipe()

    assert (
        classify_truncation_mode(
            SvgLayoutDiagnostics(system_count=8, page_count=1),
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


def test_truncation_candidate_without_duration_bearing_data_is_rejected():
    transcription = (
        "*clefF4\t*clefG2\n"
        "*k[f#]\t*k[f#]\n"
        "*M3/4\t*M3/4\n"
        "*^\t*\n"
        ".\t.\t.\n"
        "=\t=\t=\n"
    )

    assert (
        validate_truncation_candidate_contains_music(transcription)
        == "non_musical_truncation_candidate"
    )


def test_truncation_candidate_with_duration_bearing_note_or_rest_is_allowed():
    assert validate_truncation_candidate_contains_music("*clefG2\n*M3/4\n4c\n") is None
    assert validate_truncation_candidate_contains_music("*clefG2\n*M3/4\n2.r\n") is None
