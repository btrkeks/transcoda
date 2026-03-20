"""Tests for CanonicalizeBarlines pass."""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import CanonicalizeBarlines


class TestCanonicalizeBarlines:
    """Tests for CanonicalizeBarlines pass."""

    def test_pass_exists(self):
        """CanonicalizeBarlines pass should be instantiable."""
        pass_obj = CanonicalizeBarlines()
        assert pass_obj.name == "canonicalize_barlines"

    def test_normalizes_visual_modifier_barline_two_spines(self):
        """Should normalize =|! to = across two spines."""
        pass_obj = CanonicalizeBarlines()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=|!\t=|!"
        expected = "4G\t4e\n=\t="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_visual_modifier_barline_single_spine(self):
        """Should normalize =|! to = with a single spine."""
        pass_obj = CanonicalizeBarlines()
        ctx = NormalizationContext()

        input_text = "4G\n=|!"
        expected = "4G\n="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_double_repeat_barline_two_spines(self):
        """Should normalize ==:|! to =:|! across two spines."""
        pass_obj = CanonicalizeBarlines()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n==:|!\t==:|!"
        expected = "4G\t4e\n=:|!\t=:|!"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_double_repeat_barline_single_spine(self):
        """Should normalize ==:|! to =:|! with a single spine."""
        pass_obj = CanonicalizeBarlines()
        ctx = NormalizationContext()

        input_text = "4G\n==:|!"
        expected = "4G\n=:|!"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_interior_barlines(self):
        """Should normalize non-standard barlines in the middle of the piece, not just final."""
        pass_obj = CanonicalizeBarlines()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n==:|!\t==:|!\n4A\t4f\n==\t=="
        expected = "4G\t4e\n=:|!\t=:|!\n4A\t4f\n==\t=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_interior_visual_modifier_barline(self):
        """Should normalize =|! in interior barline lines."""
        pass_obj = CanonicalizeBarlines()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=|!\t=|!\n4A\t4f\n==\t=="
        expected = "4G\t4e\n=\t=\n4A\t4f\n==\t=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_standard_barlines(self):
        """Should not modify standard barlines."""
        pass_obj = CanonicalizeBarlines()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=\t=\n4A\t4f\n=:|!\t=:|!\n4B\t4g\n=!|:\t=!|:\n4C\t4a\n=||\t=||\n4D\t4b\n==\t=="
        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_empty_input(self):
        """Should handle empty string input."""
        pass_obj = CanonicalizeBarlines()
        ctx = NormalizationContext()

        input_text = ""
        expected = ""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_mixed_non_standard_in_same_line(self):
        """Should handle fields with different non-standard tokens on the same line."""
        pass_obj = CanonicalizeBarlines()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=|!\t==:|!"
        expected = "4G\t4e\n=\t=:|!"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected
