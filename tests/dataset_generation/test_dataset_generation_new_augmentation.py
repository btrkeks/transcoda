import numpy as np
from pathlib import Path

from scripts.dataset_generation.dataset_generation_new.augmentation import augment_accepted_render
from scripts.dataset_generation.dataset_generation_new.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation_new.types import (
    RenderResult,
    SamplePlan,
    SourceSegment,
    SvgLayoutDiagnostics,
)


def test_augment_accepted_render_falls_back_to_base_image_when_candidate_is_invalid(monkeypatch):
    base = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    base[20:80, 20:500] = 0
    invalid = np.zeros_like(base)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation_new.augmentation.offline_augment",
        lambda *args, **kwargs: invalid,
    )

    plan = SamplePlan(
        sample_id="sample_00000000",
        seed=7,
        segments=(SourceSegment(source_id="a", path=Path("a.krn"), order=0),),
        label_transcription="=1\n4c\n*-\n",
        source_measure_count=8,
        source_non_empty_line_count=16,
        segment_count=1,
    )
    render_result = RenderResult(
        image=base,
        render_layers=None,
        svg_diagnostics=SvgLayoutDiagnostics(system_count=4, page_count=1),
        bottom_whitespace_ratio=0.15,
        vertical_fill_ratio=0.62,
    )

    augmented = augment_accepted_render(plan, render_result, ProductionRecipe())

    assert np.array_equal(augmented, base)
