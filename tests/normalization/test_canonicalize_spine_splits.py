"""Tests for CanonicalizeSpineSplits pass."""

import pytest

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import (
    CanonicalizeSpineSplits,
    ValidateSpineOperations,
)
from src.core.spine_state import InvalidSpineOperationError


class TestCanonicalizeSpineSplits:
    """Tests for CanonicalizeSpineSplits pass."""

    def test_pass_exists(self):
        pass_obj = CanonicalizeSpineSplits()
        assert pass_obj.name == "canonicalize_spine_splits"

    def test_coalesces_staggered_two_spine_split(self):
        pass_obj = CanonicalizeSpineSplits()
        ctx = NormalizationContext()

        input_text = "*^\t*\n*\t*\t*^"
        expected = "*^\t*^"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_coalesces_multiple_safe_adjacent_split_lines(self):
        pass_obj = CanonicalizeSpineSplits()
        ctx = NormalizationContext()

        input_text = "*^\t*\t*\n*\t*\t*\t*^\n*\t*\t*^\t*\t*"
        expected = "*^\t*^\t*^"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    @pytest.mark.parametrize(
        "input_text",
        [
            "*^\t*\n4c\t4e\t4g",
            "*^\t*\n=\t=\t=",
            "*^\t*\n*clefG2\t*clefG2\t*clefF4",
        ],
    )
    def test_preserves_split_followed_by_non_split_line(self, input_text):
        pass_obj = CanonicalizeSpineSplits()
        ctx = NormalizationContext()

        result = pass_obj.transform(input_text, ctx)

        assert result == input_text

    def test_preserves_split_of_newly_created_subspine(self):
        pass_obj = CanonicalizeSpineSplits()
        ctx = NormalizationContext()

        input_text = "*^\t*\n*\t*^\t*"

        result = pass_obj.transform(input_text, ctx)

        assert result == input_text

    def test_preserves_malformed_width_for_final_validator(self):
        pass_obj = CanonicalizeSpineSplits()
        validator = ValidateSpineOperations()
        ctx = NormalizationContext()

        input_text = "*^\t*\n*\t*^"

        result = pass_obj.transform(input_text, ctx)

        assert result == input_text
        with pytest.raises(InvalidSpineOperationError, match="expected 3 spine fields"):
            validator.validate(result, ctx)
