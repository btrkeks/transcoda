"""Row construction for accepted samples."""

from __future__ import annotations

from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.types import AcceptedSample


def build_dataset_row(
    sample: AcceptedSample,
    recipe: ProductionRecipe,
) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "image": sample.image_bytes,
        "transcription": sample.label_transcription,
        "initial_kern_spine_count": sample.initial_kern_spine_count,
        "source_ids": list(sample.source_ids),
        "segment_count": sample.segment_count,
        "source_measure_count": sample.source_measure_count,
        "source_non_empty_line_count": sample.source_non_empty_line_count,
        "svg_system_count": sample.system_count,
        "truncation_applied": sample.truncation_applied,
        "truncation_reason": sample.truncation_reason,
        "truncation_ratio": sample.truncation_ratio,
        "vertical_fill_ratio": sample.vertical_fill_ratio,
        "bottom_whitespace_ratio": sample.bottom_whitespace_ratio,
        "bottom_whitespace_px": sample.bottom_whitespace_px,
        "top_whitespace_px": sample.top_whitespace_px,
        "content_height_px": sample.content_height_px,
        "recipe_version": recipe.version,
    }


build_dataset_record = build_dataset_row
