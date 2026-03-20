"""Tests for RemoveNullTies pass."""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import RemoveNullTies


class TestRemoveNullTies:
    """Tests for RemoveNullTies pass."""

    def test_pass_exists(self):
        """RemoveNullTies pass should be instantiable."""
        pass_obj = RemoveNullTies()
        assert pass_obj.name == "remove_null_ties"

    def test_basic_null_tie_removal(self):
        """Should remove single null ties from notes."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G[]"
        expected = "4G"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_double_null_tie_removal(self):
        """Should remove double null ties from notes."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G[[]]"
        expected = "4G"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_valid_opening_tie(self):
        """Should preserve notes with only opening tie marker."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G["
        expected = "4G["

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_valid_closing_tie(self):
        """Should preserve notes with only closing tie marker."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G]"
        expected = "4G]"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_works_in_chords(self):
        """Should remove null ties from notes in chords."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G[] 4e"
        expected = "4G 4e"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_works_across_multiple_spines(self):
        """Should remove null ties across multiple spines."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G[]\t4e[]"
        expected = "4G\t4e"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_notes_with_other_modifiers(self):
        """Should preserve other modifiers when removing null ties."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G#[]L"
        expected = "4G#L"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_accidentals_and_beams(self):
        """Should handle notes with accidentals and beam markers."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "8ee-[]J"
        expected = "8ee-J"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_multiline_document(self):
        """Should handle multiline documents."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
4G[]\t4cc[]
4A\t4dd
*-\t*-"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4G\t4cc
4A\t4dd
*-\t*-"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_non_note_tokens(self):
        """Should not modify non-note tokens."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "*k[b-]\t=\t."

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == input_text

    def test_handles_empty_input(self):
        """Should handle empty string input."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = ""
        expected = ""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_multiple_null_ties_in_chord(self):
        """Should remove null ties from multiple notes in same chord."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G[] 4B[] 4d[]"
        expected = "4G 4B 4d"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_mixed_valid_and_null_ties(self):
        """Should only remove null ties, preserving valid ties."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G[ 4B[] 4d]"
        expected = "4G[ 4B 4d]"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_basic_null_slur_removal(self):
        """Should remove single null slurs from notes."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G()"
        expected = "4G"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_double_null_slur_removal(self):
        """Should remove double null slurs from notes."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G(())"
        expected = "4G"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_valid_opening_slur(self):
        """Should preserve notes with only opening slur marker."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G("
        expected = "4G("

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_valid_closing_slur(self):
        """Should preserve notes with only closing slur marker."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G)"
        expected = "4G)"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_null_slurs_in_chords(self):
        """Should remove null slurs from notes in chords."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G() 4e"
        expected = "4G 4e"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_mixed_null_ties_and_slurs(self):
        """Should remove both null ties and null slurs from the same note."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G()[]"
        expected = "4G"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_mixed_valid_and_null_slurs(self):
        """Should only remove null slurs, preserving valid slurs."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G( 4B() 4d)"
        expected = "4G( 4B 4d)"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_collapses_double_open_tie(self):
        """Should collapse repeated opening ties to a single marker."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4B[["
        expected = "4B["

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_collapses_double_close_tie(self):
        """Should collapse repeated closing ties to a single marker."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4B]]"
        expected = "4B]"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_collapses_double_continue_tie(self):
        """Should collapse repeated continuation ties to a single marker."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "2a__"
        expected = "2a_"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_collapses_repeated_ties_but_preserves_double_slur(self):
        """Should canonicalize ties while preserving slur multiplicity."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "16FF#([[L"
        expected = "16FF#([L"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_double_slurs(self):
        """Should not collapse double slurs, which are grammar-valid."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "4G(( 4A))"
        expected = "4G(( 4A))"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_null_key_signature(self):
        """Should not touch key-signature tokens that include [] by design."""
        pass_obj = RemoveNullTies()
        ctx = NormalizationContext()

        input_text = "*k[]"
        expected = "*k[]"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected
