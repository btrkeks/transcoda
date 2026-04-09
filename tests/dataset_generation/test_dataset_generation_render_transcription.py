from scripts.dataset_generation.dataset_generation.recipe import (
    ProductionRecipe,
    RenderOnlyAugmentationPolicy,
)
from scripts.dataset_generation.dataset_generation.render_transcription import (
    build_render_transcription,
)


def _make_recipe() -> ProductionRecipe:
    return ProductionRecipe(
        render_only_aug=RenderOnlyAugmentationPolicy(
            include_title_probability=0.0,
            include_author_probability=0.0,
            render_pedals_probability=0.0,
            render_pedals_measures_probability=0.0,
            render_instrument_piano_probability=0.0,
            render_sforzando_probability=0.0,
            render_sforzando_per_note_probability=0.0,
            render_accent_probability=0.0,
            render_accent_per_note_probability=0.0,
            render_tempo_probability=0.0,
            render_tempo_include_mm_probability=0.0,
            render_hairpins_probability=0.0,
            render_hairpins_max_spans=1,
            render_dynamic_marks_probability=0.0,
            render_dynamic_marks_min_count=1,
            render_dynamic_marks_max_count=1,
            max_render_attempts=1,
        )
    )


def test_build_render_transcription_appends_missing_terminator_for_normalized_input():
    render_text = build_render_transcription(
        "**kern\t**kern\n*clefG2\t*clefF4\n=1\t=1\n4c\t4C\n=\t=",
        _make_recipe(),
        seed=7,
    )

    assert render_text.endswith("*-\t*-")


def test_build_render_transcription_does_not_duplicate_existing_terminator():
    render_text = build_render_transcription(
        "**kern\t**kern\n*clefG2\t*clefF4\n=1\t=1\n4c\t4C\n*-\t*-",
        _make_recipe(),
        seed=7,
    )

    assert render_text.count("*-\t*-") == 1
