"""Tests for RemoveRedundantTimeSignatures pass."""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import RemoveRedundantTimeSignatures


class TestRemoveRedundantTimeSignatures:
    """Tests for RemoveRedundantTimeSignatures pass."""

    def test_pass_exists(self):
        """RemoveRedundantTimeSignatures pass should be instantiable."""
        pass_obj = RemoveRedundantTimeSignatures()
        assert pass_obj.name == "remove_redundant_time_signatures"

    def test_basic_met_c_with_m44(self):
        """Should remove *M4/4 when *met(c) is present."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c)
*M4/4\t*M4/4
*clefF4\t*clefG2"""

        expected = """*met(c)\t*met(c)
*clefF4\t*clefG2"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_cut_time_met_c_bar_with_m22(self):
        """Should remove *M2/2 when *met(c|) is present."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c|)\t*met(c|)
*M2/2\t*M2/2
*clefF4\t*clefG2"""

        expected = """*met(c|)\t*met(c|)
*clefF4\t*clefG2"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_uppercase_met_C(self):
        """Should handle uppercase *met(C) the same as lowercase."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(C)\t*met(C)
*M4/4\t*M4/4
*clefF4\t*clefG2"""

        expected = """*met(C)\t*met(C)
*clefF4\t*clefG2"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_uppercase_met_C_bar(self):
        """Should handle uppercase *met(C|) the same as lowercase."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(C|)\t*met(C|)
*M2/2\t*M2/2
*clefF4\t*clefG2"""

        expected = """*met(C|)\t*met(C|)
*clefF4\t*clefG2"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_non_matching_meter_kept(self):
        """Should keep meter when it doesn't match mensuration equivalent."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c)
*M3/4\t*M3/4
*clefF4\t*clefG2"""

        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_meter_without_mensuration_kept(self):
        """Should keep meter when no mensuration sign is present."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*M4/4\t*M4/4
*clefF4\t*clefG2
4c\t4e"""

        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_partial_null_spines(self):
        """Should remove meter when both mensuration and meter have matching null spines."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*
*M4/4\t*
*clefF4\t*clefG2"""

        expected = """*met(c)\t*
*clefF4\t*clefG2"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_spine_split_between_resets_tracking(self):
        """Should keep meter when spine manipulation occurs between mensuration and meter."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c)
*^\t*
*M4/4\t*M4/4\t*M4/4
*clefF4\t*clefG2\t*clefG2"""

        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_data_line_between_resets_tracking(self):
        """Should keep meter when data line occurs between mensuration and meter."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c)
4c\t4e
*M4/4\t*M4/4"""

        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_barline_between_resets_tracking(self):
        """Should keep meter when barline occurs between mensuration and meter."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c)
=\t=
*M4/4\t*M4/4"""

        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_mixed_spines_some_match_some_dont(self):
        """Should keep meter when only some spines match their expected equivalent."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c|)
*M4/4\t*M4/4
*clefF4\t*clefG2"""

        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_multiple_occurrences(self):
        """Should handle multiple mensuration/meter pairs in a piece."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c)
*M4/4\t*M4/4
*clefF4\t*clefG2
4c\t4e
=\t=
*met(c|)\t*met(c|)
*M2/2\t*M2/2
4d\t4f"""

        expected = """*met(c)\t*met(c)
*clefF4\t*clefG2
4c\t4e
=\t=
*met(c|)\t*met(c|)
4d\t4f"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_empty_input(self):
        """Should handle empty string input."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = ""
        expected = ""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_single_spine(self):
        """Should work with single spine notation."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)
*M4/4
*clefG2
4c"""

        expected = """*met(c)
*clefG2
4c"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_three_spines(self):
        """Should work with three spine notation."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c)\t*met(c)
*M4/4\t*M4/4\t*M4/4
*clefF4\t*clefG2\t*clefG2
4c\t4e\t4g"""

        expected = """*met(c)\t*met(c)\t*met(c)
*clefF4\t*clefG2\t*clefG2
4c\t4e\t4g"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_real_world_example(self):
        """Should handle the exact example from the problem statement."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c)
*M4/4\t*M4/4
*clefF4\t*clefG2"""

        expected = """*met(c)\t*met(c)
*clefF4\t*clefG2"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_intervening_interpretation_allowed(self):
        """Should remove meter even with other interpretations between mensuration and meter."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c)
*k[b-]\t*k[b-]
*M4/4\t*M4/4
*clefF4\t*clefG2"""

        expected = """*met(c)\t*met(c)
*k[b-]\t*k[b-]
*clefF4\t*clefG2"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_spine_merge_resets_tracking(self):
        """Should keep meter when spine merge (*v) occurs between mensuration and meter."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c)\t*met(c)
*v\t*v\t*
*M4/4\t*M4/4"""

        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_spine_exchange_resets_tracking(self):
        """Should keep meter when spine exchange (*x) occurs between mensuration and meter."""
        pass_obj = RemoveRedundantTimeSignatures()
        ctx = NormalizationContext()

        input_text = """*met(c)\t*met(c)
*x\t*x
*M4/4\t*M4/4"""

        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected
