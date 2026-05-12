"""Tests for NormalizeFinalBarline pass."""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import NormalizeFinalBarline


class TestNormalizeFinalBarline:
    """Tests for NormalizeFinalBarline pass."""

    def test_pass_exists(self):
        """NormalizeFinalBarline pass should be instantiable."""
        pass_obj = NormalizeFinalBarline()
        assert pass_obj.name == "normalize_final_barline"

    def test_normalizes_final_barline(self):
        """Should normalize =|| to == on the final barline line."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=||\t=||"
        expected = "4G\t4e\n==\t=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_interior_double_barline(self):
        """Should not modify =|| barlines in the middle of the piece."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=||\t=||\n4A\t4f\n==\t=="
        expected = "4G\t4e\n=||\t=||\n4A\t4f\n==\t=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_final_double_barline_already_correct(self):
        """Should not modify == final barlines."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n==\t=="
        expected = "4G\t4e\n==\t=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_repeat_final_barline(self):
        """Should not modify =:|! final barlines."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=:|!\t=:|!"
        expected = "4G\t4e\n=:|!\t=:|!"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_single_spine(self):
        """Should normalize =|| to == with a single spine."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\n=||"
        expected = "4G\n=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_bare_barline(self):
        """Should normalize bare = to == on the final barline line."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=\t="
        expected = "4G\t4e\n==\t=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_bare_barline_single_spine(self):
        """Should normalize bare = to == with a single spine."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\n="
        expected = "4G\n=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_mixed_bare_and_double_barline(self):
        """Should normalize when spines have a mix of = and =||."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=\t=||"
        expected = "4G\t4e\n==\t=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_interior_bare_barline(self):
        """Should not modify bare = barlines in the middle of the piece."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=\t=\n4A\t4f\n==\t=="
        expected = "4G\t4e\n=\t=\n4A\t4f\n==\t=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_numbered_barline(self):
        """Should not modify numbered barlines like =2 on the final line."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=2\t=2"
        expected = "4G\t4e\n=2\t=2"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_start_repeat_to_end_repeat(self):
        """Should normalize =!|: (start repeat) to =:|! (end repeat) on the final barline."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=!|:\t=!|:"
        expected = "4G\t4e\n=:|!\t=:|!"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_start_repeat_single_spine(self):
        """Should normalize =!|: to =:|! with a single spine."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\n=!|:"
        expected = "4G\n=:|!"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_interior_start_repeat(self):
        """Should not modify =!|: barlines in the middle of the piece."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=!|:\t=!|:\n4A\t4f\n==\t=="
        expected = "4G\t4e\n=!|:\t=!|:\n4A\t4f\n==\t=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_end_start_repeat_variant(self):
        """Should normalize =:|!|: (end-and-start repeat) to =:|! on the final barline."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=:|!|:\t=:|!|:"
        expected = "4G\t4e\n=:|!\t=:|!"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_end_start_repeat(self):
        """Should normalize =:!|: (end-and-start repeat) to =:|! on the final barline."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=:!|:\t=:!|:"
        expected = "4G\t4e\n=:|!\t=:|!"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_interior_end_start_repeat(self):
        """Should not modify =:|!|: barlines in the middle of the piece."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = "4G\t4e\n=:|!|:\t=:|!|:\n4A\t4f\n==\t=="
        expected = "4G\t4e\n=:|!|:\t=:|!|:\n4A\t4f\n==\t=="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_empty_input(self):
        """Should handle empty string input."""
        pass_obj = NormalizeFinalBarline()
        ctx = NormalizationContext()

        input_text = ""
        expected = ""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected
