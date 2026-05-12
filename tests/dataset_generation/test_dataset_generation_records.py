from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.records import build_dataset_row
from scripts.dataset_generation.dataset_generation.types_render import AcceptedSample


def test_build_dataset_row_matches_rewrite_schema():
    sample = AcceptedSample(
        sample_id="sample_00000001",
        label_transcription="*clefG2\n=1\n4c\n*-\n",
        image_bytes=b"jpeg",
        initial_kern_spine_count=1,
        segment_count=2,
        source_ids=("a", "b"),
        source_measure_count=12,
        source_non_empty_line_count=30,
        system_count=4,
        truncation_applied=True,
        truncation_reason="system_count_policy",
        truncation_ratio=0.75,
        bottom_whitespace_ratio=0.12,
        vertical_fill_ratio=0.68,
        bottom_whitespace_px=178,
        top_whitespace_px=44,
        content_height_px=1012,
    )

    record = build_dataset_row(
        sample,
        recipe=ProductionRecipe(),
    )

    assert set(record) == {
        "sample_id",
        "image",
        "transcription",
        "initial_kern_spine_count",
        "source_ids",
        "segment_count",
        "source_measure_count",
        "source_non_empty_line_count",
        "svg_system_count",
        "truncation_applied",
        "truncation_reason",
        "truncation_ratio",
        "vertical_fill_ratio",
        "bottom_whitespace_ratio",
        "bottom_whitespace_px",
        "top_whitespace_px",
        "content_height_px",
        "recipe_version",
    }
    assert record["image"] == b"jpeg"
    assert record["initial_kern_spine_count"] == 1
    assert record["source_ids"] == ["a", "b"]
    assert record["svg_system_count"] == 4
    assert record["bottom_whitespace_px"] == 178
    assert record["top_whitespace_px"] == 44
    assert record["content_height_px"] == 1012
