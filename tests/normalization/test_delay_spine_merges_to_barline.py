"""Tests for DelaySpineMergesToBarline pass."""

from __future__ import annotations

import pytest

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import DelaySpineMergesToBarline


def _normalize(text: str) -> str:
    pass_obj = DelaySpineMergesToBarline()
    ctx = NormalizationContext()
    pass_obj.prepare(text, ctx)
    result = pass_obj.transform(text, ctx)
    pass_obj.validate(result, ctx)
    return result


class TestDelaySpineMergesToBarline:
    """Tests for safe spine-merge delay."""

    def test_pass_exists(self) -> None:
        pass_obj = DelaySpineMergesToBarline()
        assert pass_obj.name == "delay_spine_merges_to_barline"

    def test_delays_merge_from_prompt_example(self) -> None:
        input_text = "*v\t*v\t*\t*\n8D 8A\t8a)J\t.\n=\t=\t="
        expected = "8D 8A\t8ryy\t8a)J\t.\n*v\t*v\t*\t*\n=\t=\t="

        assert _normalize(input_text) == expected

    def test_delays_multiple_independent_merge_groups(self) -> None:
        input_text = "*v\t*v\t*\t*v\t*v\n4c\t4e\t4g\n=\t=\t="
        expected = "4c\t4ryy\t4e\t4g\t4ryy\n*v\t*v\t*\t*v\t*v\n=\t=\t="

        assert _normalize(input_text) == expected

    def test_uses_chord_duration_when_all_notes_match(self) -> None:
        input_text = "*v\t*v\n8D 8A\n="
        expected = "8D 8A\t8ryy\n*v\t*v\n="

        assert _normalize(input_text) == expected

    def test_preserves_dotted_and_rational_duration_text(self) -> None:
        input_text = "*v\t*v\n4.c\n1%6d\n="
        expected = "4.c\t4.ryy\n1%6d\t1%6ryy\n*v\t*v\n="

        assert _normalize(input_text) == expected

    @pytest.mark.parametrize(
        "input_text",
        [
            "*v\t*v\n*^\n4c\t4e\n=",
            "*v\t*v\n*clefG2\n4c\n=",
            "*v\t*v\n!local\n4c\n=",
            "*v\t*v\n4c\t4e\n=",
            "*v\t*v\n4c 8e\n=",
        ],
    )
    def test_leaves_ambiguous_regions_unchanged(self, input_text: str) -> None:
        assert _normalize(input_text) == input_text

    def test_eof_without_barline_leaves_input_unchanged(self) -> None:
        input_text = "*v\t*v\n4c\n4d"

        assert _normalize(input_text) == input_text
