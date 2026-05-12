"""Tests for HoistSpineSplitsToBarline pass."""

from __future__ import annotations

import pytest

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import (
    HoistSpineSplitsToBarline,
    ValidateSpineOperations,
)


def _normalize(text: str) -> str:
    pass_obj = HoistSpineSplitsToBarline()
    ctx = NormalizationContext()
    pass_obj.prepare(text, ctx)
    result = pass_obj.transform(text, ctx)
    pass_obj.validate(result, ctx)
    return result


def _assert_valid_spines(text: str) -> None:
    ValidateSpineOperations().validate(text, NormalizationContext())


class TestHoistSpineSplitsToBarline:
    """Tests for safe spine-split hoisting."""

    def test_pass_exists(self) -> None:
        pass_obj = HoistSpineSplitsToBarline()
        assert pass_obj.name == "hoist_spine_splits_to_barline"

    def test_hoists_basic_split_after_previous_barline(self) -> None:
        input_text = "=\t=\n4c\t4e\n*^\t*\n8d\t8f\t8a\n=\t=\t="
        expected = "=\t=\n*^\t*\n4c\t4ryy\t4e\n8d\t8f\t8a\n=\t=\t="

        result = _normalize(input_text)

        assert result == expected
        _assert_valid_spines(result)

    def test_hoists_multiple_independent_split_fields(self) -> None:
        input_text = "=\t=\t=\n4c\t4e\t4g\n*^\t*\t*^\n8a\t8b\t8c\t8d\t8e\n=\t=\t=\t=\t="
        expected = "=\t=\t=\n*^\t*\t*^\n4c\t4ryy\t4e\t4g\t4ryy\n8a\t8b\t8c\t8d\t8e\n=\t=\t=\t=\t="

        result = _normalize(input_text)

        assert result == expected
        _assert_valid_spines(result)

    def test_hoists_multiple_split_lines_in_relative_order(self) -> None:
        input_text = (
            "=\t=\n"
            "4c\t4e\n"
            "*^\t*\n"
            "4d\t4f\t4a\n"
            "*\t*^\t*\n"
            "8g\t8b\t8cc\t8dd\n"
            "=\t=\t=\t="
        )
        expected = (
            "=\t=\n"
            "*^\t*\n"
            "*\t*^\t*\n"
            "4c\t4ryy\t4ryy\t4e\n"
            "4d\t4f\t4ryy\t4a\n"
            "8g\t8b\t8cc\t8dd\n"
            "=\t=\t=\t="
        )

        result = _normalize(input_text)

        assert result == expected
        _assert_valid_spines(result)

    def test_uses_chord_duration_when_all_notes_match(self) -> None:
        input_text = "=\n8D 8A\n*^\n16d\t16f\n=\t="
        expected = "=\n*^\n8D 8A\t8ryy\n16d\t16f\n=\t="

        result = _normalize(input_text)

        assert result == expected
        _assert_valid_spines(result)

    def test_preserves_dotted_and_rational_duration_text(self) -> None:
        input_text = "=\n4.c\n1%6d\n*^\n8e\t8g\n=\t="
        expected = "=\n*^\n4.c\t4.ryy\n1%6d\t1%6ryy\n8e\t8g\n=\t="

        result = _normalize(input_text)

        assert result == expected
        _assert_valid_spines(result)

    def test_split_followed_by_later_merge_stays_valid(self) -> None:
        input_text = "=\t=\n4c\t4e\n*^\t*\n8d\t8f\t8a\n*v\t*v\t*\n4g\t4b\n=\t="
        expected = "=\t=\n*^\t*\n4c\t4ryy\t4e\n8d\t8f\t8a\n*v\t*v\t*\n4g\t4b\n=\t="

        result = _normalize(input_text)

        assert result == expected
        _assert_valid_spines(result)

    @pytest.mark.parametrize(
        "input_text",
        [
            "4c\t4e\n*^\t*\n8d\t8f\t8a",
            "=\t=\n!local\t!local\n4c\t4e\n*^\t*\n8d\t8f\t8a",
            "=\t=\n*M4/4\t*M4/4\n4c\t4e\n*^\t*\n8d\t8f\t8a",
            "=\t=\n4c\n*^\t*\n8d\t8f\t8a",
            "=\t=\n.\t4e\n*^\t*\n8d\t8f\t8a",
            "=\t=\n4c 8e\t4g\n*^\t*\n8d\t8f\t8a",
        ],
    )
    def test_leaves_ambiguous_chunks_unchanged(self, input_text: str) -> None:
        assert _normalize(input_text) == input_text
