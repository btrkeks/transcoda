"""Tests for RemoveLeadingBarlines pass."""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import RemoveLeadingBarlines


class TestRemoveLeadingBarlines:
    """Tests for RemoveLeadingBarlines pass."""

    def test_pass_exists(self):
        """RemoveLeadingBarlines pass should be instantiable."""
        pass_obj = RemoveLeadingBarlines()
        assert pass_obj.name == "remove_leading_barlines"

    def test_removes_single_leading_barline(self):
        """Should remove a single barline before first data."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
=\t=
4c\t4e"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4c\t4e"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_removes_multiple_leading_barlines(self):
        """Should remove multiple consecutive barlines before first data."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
=\t=
=\t=
4c\t4e"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4c\t4e"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_barlines_after_data(self):
        """Should preserve barlines that appear after musical data."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
=\t=
4c\t4e
=\t=
4d\t4f"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4c\t4e
=\t=
4d\t4f"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_all_interpretations(self):
        """Should preserve all interpretation lines (clefs, keys, meters, spine ops)."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
*k[b-e-a-]\t*k[b-e-a-]
*M3/4\t*M3/4
*^\t*
=\t=\t=
4c\t4e\t4g"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
*k[b-e-a-]\t*k[b-e-a-]
*M3/4\t*M3/4
*^\t*
4c\t4e\t4g"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_empty_input(self):
        """Should handle empty string input."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = ""
        expected = ""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_no_barlines(self):
        """Should return unchanged if no leading barlines exist."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
4c\t4e
=\t=
4d\t4f"""

        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_no_data_lines(self):
        """Should return unchanged if no data lines exist."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
=\t="""

        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        # Note: validate would fail here since there's a barline but no data
        # But since there's no data, the pass doesn't remove anything

        assert result == expected

    def test_real_world_example(self):
        """Should handle the user's exact example."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
*k[b-e-a-d-g-c-]\t*k[b-e-a-d-g-c-]
*M3/4\t*M3/4
*^\t*
=\t=\t=
4F/\t4FF/\t4ff[\\ 4ddd-[\\
4d-(/\t2B-\\ 2d-\\\t8.ff\\] 8.ddd-\\]L
.\t.\t16ff\\ 16ddd-\\Jk
4f)/\t.\t4dd-\\ 4bb-\\"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
*k[b-e-a-d-g-c-]\t*k[b-e-a-d-g-c-]
*M3/4\t*M3/4
*^\t*
4F/\t4FF/\t4ff[\\ 4ddd-[\\
4d-(/\t2B-\\ 2d-\\\t8.ff\\] 8.ddd-\\]L
.\t.\t16ff\\ 16ddd-\\Jk
4f)/\t.\t4dd-\\ 4bb-\\"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_numbered_barlines_after_data(self):
        """Should preserve numbered barlines (=42) after data."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
=1\t=1
4c\t4e
=2\t=2
4d\t4f"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4c\t4e
=2\t=2
4d\t4f"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_double_barlines_after_data(self):
        """Should preserve double barlines (==) after data."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
=\t=
4c\t4e
==\t==
*-\t*-"""

        expected = """**kern\t**kern
4c\t4e
==\t==
*-\t*-"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_three_spines(self):
        """Should handle three-spine notation correctly."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
=\t=\t=
4c\t4e\t4g"""

        expected = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
4c\t4e\t4g"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_rest_as_data(self):
        """Should recognize rests as data lines."""
        pass_obj = RemoveLeadingBarlines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
=\t=
4r\t4r
4c\t4e"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4r\t4r
4c\t4e"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected
