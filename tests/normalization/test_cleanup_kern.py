"""Tests for CleanupKern pass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import CleanupKern


class TestRemovePartTokens:
    """Tests for *part token removal in CleanupKern."""

    def test_removes_part_tokens_two_spines(self):
        """Should replace *part tokens with '*' in a two-spine piece."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        input_text = "*part2\t*part1\n*clefF4\t*clefG2\n4C\t4c"
        # *part line becomes *\t* which gets removed by _remove_all_star_lines
        expected = "*clefF4\t*clefG2\n4C\t4c"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected

    def test_removes_part_tokens_four_spines(self):
        """Should replace *part tokens with '*' across multiple spines."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        input_text = "*part2\t*part2\t*part1\t*part1\n*clefF4\t*clefG2\t*clefF4\t*clefG2\n4C\t4c\t4C\t4c"
        expected = "*clefF4\t*clefG2\t*clefF4\t*clefG2\n4C\t4c\t4C\t4c"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected

    def test_removes_high_part_numbers(self):
        """Should handle high part numbers like *part10, *part42."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        input_text = "*part10\t*part1\n*clefF4\t*clefG2\n4C\t4c"
        expected = "*clefF4\t*clefG2\n4C\t4c"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected

    def test_removes_part_and_staff_together(self):
        """Should remove both *part and *staff lines."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        input_text = "*part2\t*part1\n*staff2\t*staff1\n*clefF4\t*clefG2\n4C\t4c"
        expected = "*clefF4\t*clefG2\n4C\t4c"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected

    def test_no_part_tokens_unchanged(self):
        """Should not modify input that has no *part tokens."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        input_text = "*clefF4\t*clefG2\n*k[]\t*k[]\n4C\t4c"
        expected = "*clefF4\t*clefG2\n*k[]\t*k[]\n4C\t4c"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected

    def test_part_with_null_interpretation(self):
        """Should handle lines where some spines have *part and others have *."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        # Some spines have * instead of *partN (common in real data)
        input_text = "*part2\t*\t*part1\n*clefF4\t*clefG2\t*clefG2\n4C\t4c\t4e"
        expected = "*clefF4\t*clefG2\t*clefG2\n4C\t4c\t4e"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected


class TestRemoveUserDefinedNMarks:
    """Tests for removal of uppercase N user-defined marks in data fields."""

    def test_removes_n_marks_in_note_fields(self):
        """Should strip uppercase N from note/chord data fields."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        input_text = "*clefF4\t*clefG2\n12aa(JN\t.\n.\t12ggg)JN"
        expected = "*clefF4\t*clefG2\n12aa(J\t.\n.\t12ggg)J"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected


class TestTupletAndUnsupportedSymbolCleanup:
    """Tests for *tuplet removal and unsupported note symbol stripping."""

    def test_removes_tuplet_markers_and_malformed_star_lines(self):
        """Should remove *tuplet/*Xtuplet markers and collapse resulting star-only lines."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        input_text = "*\t*tuplet\n*\t*Xtuplet    *\n*clefF4\t*clefG2\n8G\t8ffSLs"
        expected = "*clefF4\t*clefG2\n8G\t8ffL"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected

    def test_strips_unsupported_symbols_from_note_fields(self):
        """Should strip S/$/&/p and beam-suffix s in note/chord fields."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        input_text = "*clefF4\t*clefG2\n8ffS$&pLs\t8ddSLs"
        expected = "*clefF4\t*clefG2\n8ffL\t8ddL"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected

    def test_beam_suffix_rule_is_conservative(self):
        """Should remove trailing s only when directly after beam markers."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        input_text = "*clefF4\t*clefG2\n8as\t8ffLs"
        expected = "*clefF4\t*clefG2\n8as\t8ffL"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected

    def test_keeps_lowercase_n_accidentals(self):
        """Should not remove lowercase n natural accidentals."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        input_text = "*clefF4\t*clefG2\n8BBn 8D\t8ffnL"
        expected = "*clefF4\t*clefG2\n8BBn 8D\t8ffnL"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected

    def test_preserves_interpretation_and_barline_fields(self):
        """Should not alter interpretation or barline fields when removing N."""
        pass_obj = CleanupKern()
        ctx = NormalizationContext()

        input_text = "*Ncustom\t*clefG2\n=\t=\n12ggg)JN\t4r"
        expected = "*Ncustom\t*clefG2\n=\t=\n12ggg)J\t4r"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected
