"""Tests for SymbolsBeforeSplit pass."""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import SymbolsBeforeSplit


class TestSymbolsBeforeSplit:
    """Tests for SymbolsBeforeSplit pass."""

    def test_pass_exists(self):
        """SymbolsBeforeSplit pass should be instantiable."""
        pass_obj = SymbolsBeforeSplit()
        assert pass_obj.name == "symbols_before_split"

    def test_doesnt_reorder_no_splits(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        expected = """*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_doesnt_reorder_correct_order(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*clefF4\t*clefG2
*k[b-]\t*k[b-]
*M3/4\t*M3/4
*\t*^
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        expected = """*clefF4\t*clefG2
*k[b-]\t*k[b-]
*M3/4\t*M3/4
*\t*^
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_reorders_main_example(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        expected = """*clefF4\t*clefG2
*k[b-]\t*k[b-]
*M3/4\t*M3/4
*\t*^
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_reorders_alto_clef(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
*clefF4\t*clefC3\t*clefC3
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        expected = """*clefF4\t*clefC3
*k[b-]\t*k[b-]
*M3/4\t*M3/4
*\t*^
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_reorders_bottom_clefs(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*^\t*
*clefF4\t*clefF4\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""
        expected = """*clefF4\t*clefG2
*k[b-]\t*k[b-]
*M3/4\t*M3/4
*^\t*
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_doesnt_reorder_non_mergable_clefs(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
*clefF4\t*clefF4\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        expected = """*\t*^
*clefF4\t*clefF4\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_doesnt_reorder_different_time_signatures(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M2/2
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        expected = """*clefF4\t*clefG2
*k[b-]\t*k[b-]
*\t*^
*M3/4\t*M3/4\t*M2/2
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_reorders_different_key_signatures_in_different_spines(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[]\t*k[]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        expected = """*clefF4\t*clefG2
*k[b-]\t*k[]
*M3/4\t*M3/4
*\t*^
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_doesnt_reorder_different_key_signatures(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        expected = """*clefF4\t*clefG2
*\t*^
*k[b-]\t*k[b-]\t*k[]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_doesnt_reorder_for_merge_down(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*v
*clefF4\t*clefG2
*k[b-]\t*k[b-]
*M3/4\t*M3/4"""

        expected = """*\t*v
*clefF4\t*clefG2
*k[b-]\t*k[b-]
*M3/4\t*M3/4"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_split_in_the_middle(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^\t*
*clefF4\t*clefG2\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r\t4a
"""

        expected = """*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
*\t*^\t*
4r\t8ddL\t4r\t4a
"""
        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_reorders_mid_piece(self):
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """4r\t8ddL
.\t8b-J
4r\t4dd
=\t=
*\t*^
*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        expected = """4r\t8ddL
.\t8b-J
4r\t4dd
=\t=
*clefF4\t*clefG2
*k[b-]\t*k[b-]
*M3/4\t*M3/4
*\t*^
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_multiple_splits_in_file(self):
        """Test multiple split events in the same file."""
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4c\t4e\t4g
=\t=\t=
*v\t*v\t*
4c\t4e
=\t=
*\t*^
*clefG2\t*clefG2\t*clefG2
*k[]\t*k[]\t*k[]
*M4/4\t*M4/4\t*M4/4
4d\t4f\t4a"""

        expected = """*clefF4\t*clefG2
*k[b-]\t*k[b-]
*M3/4\t*M3/4
*\t*^
4c\t4e\t4g
=\t=\t=
*v\t*v\t*
4c\t4e
=\t=
*clefG2\t*clefG2
*k[]\t*k[]
*M4/4\t*M4/4
*\t*^
4d\t4f\t4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_reorders_with_only_clef(self):
        """Test reordering when only clef is present after split."""
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
*clefF4\t*clefG2\t*clefG2
4c\t4e\t4g"""

        expected = """*clefF4\t*clefG2
*\t*^
4c\t4e\t4g"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_reorders_with_only_key_signature(self):
        """Test reordering when only key signature is present after split."""
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
*k[b-]\t*k[b-]\t*k[b-]
4c\t4e\t4g"""

        expected = """*k[b-]\t*k[b-]
*\t*^
4c\t4e\t4g"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_reorders_with_clef_and_key_no_time(self):
        """Test reordering with clef and key signature but no time signature."""
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
4c\t4e\t4g"""

        expected = """*clefF4\t*clefG2
*k[b-]\t*k[b-]
*\t*^
4c\t4e\t4g"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_split_with_no_interpretations_after(self):
        """Test split followed immediately by data (no interpretations)."""
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
4c\t4e\t4g
4d\t4f\t4a"""

        expected = """*\t*^
4c\t4e\t4g
4d\t4f\t4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_doesnt_reorder_across_barlines(self):
        """Test that reordering doesn't happen across barlines."""
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
=1\t=1\t=1
*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4c\t4e\t4g"""

        expected = """*\t*^
=1\t=1\t=1
*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4c\t4e\t4g"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_reorders_with_other_tandem_interpretations(self):
        """Test reordering with instrument, tempo, and other tandem interpretations."""
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*\t*^
*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
*I"Piano\t*I"Piano\t*I"Piano
*MM120\t*MM120\t*MM120
4c\t4e\t4g"""

        expected = """*clefF4\t*clefG2
*k[b-]\t*k[b-]
*M3/4\t*M3/4
*I"Piano\t*I"Piano
*MM120\t*MM120
*\t*^
4c\t4e\t4g"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_reorders_double_split_with_multiple_interpretations(self):
        """Test reordering with double split (*^ *^) and multiple tandem interpretations."""
        pass_obj = SymbolsBeforeSplit()
        ctx = NormalizationContext()

        input_text = """*^\t*^
*clefF4\t*clefF4\t*clefG2\t*clefG2
*k[f#c#g#d#a#]\t*k[f#c#g#d#a#]\t*k[f#c#g#d#a#]\t*k[f#c#g#d#a#]
*M2/2\t*M2/2\t*M2/2\t*M2/2
*met(c|)\t*met(c|)\t*met(c|)\t*met(c|)"""

        expected = """*clefF4\t*clefG2
*k[f#c#g#d#a#]\t*k[f#c#g#d#a#]
*M2/2\t*M2/2
*met(c|)\t*met(c|)
*^\t*^"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected
