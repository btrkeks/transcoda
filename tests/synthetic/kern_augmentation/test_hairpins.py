"""Tests for render-only hairpin augmentation."""

from __future__ import annotations

from scripts.dataset_generation.augmentation.hairpins import apply_render_hairpins


def test_apply_render_hairpins_adds_trailing_dynam_and_preserves_structure() -> None:
    krn = "\n".join(
        [
            "**kern\t**kern",
            "*clefF4\t*clefG2",
            "*M4/4\t*M4/4",
            "*^\t*",
            "4c\t4e\t4g",
            "4d\t.\t.",
            "*v\t*v\t*",
            "=\t=",
            "4e\t4f",
            "*-\t*-",
        ]
    )

    result = apply_render_hairpins(krn, sample_probability=1.0, max_spans=1)
    result_lines = result.splitlines()
    source_lines = krn.splitlines()

    assert result_lines[0] == "**kern\t**kern\t**dynam"
    assert "*^\t*\t*" in result
    assert "*v\t*v\t*" in result
    assert any(token in result for token in ("<", ">", "[", "]", "(", ")"))

    for source_line, result_line in zip(source_lines, result_lines, strict=True):
        assert result_line.count("\t") == source_line.count("\t") + 1


def test_apply_render_hairpins_noop_with_insufficient_candidates() -> None:
    krn = "\n".join(
        [
            "**kern",
            "*M4/4",
            "4c",
            ".",
            "=",
            "*-",
        ]
    )

    result = apply_render_hairpins(krn, sample_probability=1.0, max_spans=1)
    assert result == krn


def test_apply_render_hairpins_keeps_null_data_records_as_dots() -> None:
    krn = "\n".join(
        [
            "**kern",
            "4c",
            ".",
            "4d",
            "*-",
        ]
    )

    result = apply_render_hairpins(krn, sample_probability=1.0, max_spans=1)
    assert ".\t." in result


def test_apply_render_hairpins_noop_when_probability_is_zero(simple_kern: str) -> None:
    result = apply_render_hairpins(simple_kern, sample_probability=0.0, max_spans=1)
    assert result == simple_kern
