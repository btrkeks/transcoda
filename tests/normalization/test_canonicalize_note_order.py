"""Tests for CanonicalizeNoteOrder pass."""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import CanonicalizeNoteOrder
from scripts.dataset_generation.normalization.passes.canonicalize_note_order import (
    _parse_note_components,
    _reconstruct_canonical,
    _canonicalize_token,
)


class TestCanonicalizeNoteOrder:
    """Tests for CanonicalizeNoteOrder pass."""

    def test_pass_exists(self):
        """CanonicalizeNoteOrder pass should be instantiable."""
        pass_obj = CanonicalizeNoteOrder()
        assert pass_obj.name == "canonicalize_note_order"

    # Regular note tests

    def test_tie_at_start(self):
        """Tie marker at start should be moved after pitch."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "[8fJ"
        expected = "8f[J"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_slur_at_start(self):
        """Slur marker at start should be moved after pitch/accidental."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "(4cL"
        expected = "4c(L"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_already_canonical(self):
        """Already canonical notes should remain unchanged."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "8f#[J"
        expected = "8f#[J"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_complex_reordering(self):
        """Complex note with multiple misplaced markers."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        # [(8c#'L should become 8c#'([L
        input_text = "[(8c#'L"
        expected = "8c#'([L"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_rest_unchanged(self):
        """Rests should remain unchanged."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "4r"
        expected = "4r"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_rest_with_editorial(self):
        """Rests with editorial markers should be preserved."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "4ryy"
        expected = "4ryy"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_articulation_before_slur(self):
        """Articulation should come before slur in canonical order."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        # Already canonical
        input_text = "4c#'(L"
        expected = "4c#'(L"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    # Grace note tests

    def test_grace_q_after_duration(self):
        """Grace note: q after duration should move after pitch."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "8qc"
        expected = "8cq"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_grace_tie_before_q(self):
        """Grace note: tie before q should move after q."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "[cqL"
        expected = "cq[L"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_grace_already_canonical(self):
        """Grace note: already canonical should remain unchanged."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "8cqL"
        expected = "8cqL"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_groupetto_unchanged(self):
        """Groupetto (qq) should keep qq before pitch."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "8qqc"
        expected = "8qqc"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_groupetto_with_beam(self):
        """Groupetto with beam marker."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "8qqcL"
        expected = "8qqcL"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    # Structure tests

    def test_multi_spine(self):
        """Tab-separated spines should all be processed."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "[8fJ\t(4cL\t8d"
        expected = "8f[J\t4c(L\t8d"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_chord_space_separated(self):
        """Space-separated notes (chords) should all be processed."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "[8fJ (4cL 8d"
        expected = "8f[J 4c(L 8d"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_skip_interpretations(self):
        """Interpretation tokens should be unchanged."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "*clefG2\t*k[b-]"
        expected = "*clefG2\t*k[b-]"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_skip_barlines(self):
        """Barlines should be unchanged."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "=\t=12\t=="
        expected = "=\t=12\t=="

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_skip_null_tokens(self):
        """Null tokens (.) should be unchanged."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = ".\t.\t."
        expected = ".\t.\t."

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_multiline(self):
        """Multi-line kern string should be processed line by line."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "[8fJ\t(4cL\n8d\t4e\n=\t="
        expected = "8f[J\t4c(L\n8d\t4e\n=\t="

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    # Edge case tests

    def test_double_ties(self):
        """Double ties [[ should be preserved."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "[[8fJ"
        expected = "8f[[J"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_double_slurs(self):
        """Double slurs (( should be preserved."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "((8fL"
        expected = "8f((L"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_double_accidentals(self):
        """Double accidentals ## and -- should be preserved."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "4c##"
        expected = "4c##"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_stacked_beams(self):
        """Stacked beam markers should be preserved."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "16cLLK"
        expected = "16cLLK"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_ornament_placement(self):
        """Ornaments should come after accidentals."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        # T is ornament, should come after pitch/accidental
        input_text = "4c#T"
        expected = "4c#T"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_arpeggio_placement(self):
        """Arpeggio should come after articulation."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "4c:"
        expected = "4c:"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    # Harmonic tests

    def test_harmonic_after_accidental(self):
        """Harmonic 'o' after accidental should be canonical."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "4cc#o"
        expected = "4cc#o"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_harmonic_after_slur_reordered(self):
        """Harmonic 'o' after slur should be moved before slur."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "8d)o"
        expected = "8do)"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_harmonic_after_articulation(self):
        """Harmonic 'o' after articulation should be moved before it."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "16aa~oLL"
        expected = "16aao~LL"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected

    def test_harmonic_with_slur_and_beam(self):
        """Harmonic with slur open and beam."""
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()

        input_text = "8d(oL"
        expected = "8do(L"

        result = pass_obj.transform(input_text, ctx)
        assert result == expected


class TestParseNoteComponents:
    """Tests for the _parse_note_components helper function."""

    def test_simple_note(self):
        """Parse simple note."""
        result = _parse_note_components("4c")
        assert result["duration"] == "4"
        assert result["pitch"] == "c"
        assert result["accidental"] == ""

    def test_note_with_accidental(self):
        """Parse note with accidental."""
        result = _parse_note_components("4c#")
        assert result["pitch"] == "c"
        assert result["accidental"] == "#"

    def test_note_with_double_accidental(self):
        """Parse note with double accidental."""
        result = _parse_note_components("4c##")
        assert result["accidental"] == "##"

    def test_note_with_beam(self):
        """Parse note with beam marker."""
        result = _parse_note_components("8cL")
        assert result["beam"] == "L"

    def test_note_with_tie(self):
        """Parse note with tie marker."""
        result = _parse_note_components("4c[")
        assert result["tie"] == "["

    def test_note_with_slur(self):
        """Parse note with slur marker."""
        result = _parse_note_components("4c(")
        assert result["slur"] == "("

    def test_grace_note_q_after_pitch(self):
        """Parse grace note with q after pitch."""
        result = _parse_note_components("cq")
        assert result["pitch"] == "c"
        assert result["grace_q"] == "q"

    def test_grace_note_q_before_pitch(self):
        """Parse grace note with q before pitch."""
        result = _parse_note_components("8qc")
        assert result["duration"] == "8"
        assert result["pitch"] == "c"
        assert result["grace_q"] == "q"

    def test_groupetto(self):
        """Parse groupetto."""
        result = _parse_note_components("8qqc")
        assert result["is_groupetto"] is True
        assert result["duration"] == "8"
        assert result["pitch"] == "c"

    def test_rest(self):
        """Parse rest."""
        result = _parse_note_components("4r")
        assert result["is_rest"] is True
        assert result["duration"] == "4"

    def test_rest_with_editorial(self):
        """Parse rest with editorial marker."""
        result = _parse_note_components("4ryy")
        assert result["is_rest"] is True
        assert result["editorial"] == "yy"

    def test_complex_note(self):
        """Parse complex note with many components."""
        result = _parse_note_components("8f#'([J")
        assert result["duration"] == "8"
        assert result["pitch"] == "f"
        assert result["accidental"] == "#"
        assert result["articulation"] == "'"
        assert result["slur"] == "("
        assert result["tie"] == "["
        assert result["beam"] == "J"


class TestReconstructCanonical:
    """Tests for the _reconstruct_canonical helper function."""

    def test_simple_note(self):
        """Reconstruct simple note."""
        components = {
            "duration": "4",
            "pitch": "c",
            "accidental": "",
            "pause": "",
            "ornament": "",
            "harmonic": "",
            "articulation": "",
            "arpeggio": "",
            "slur": "",
            "breath": "",
            "tie": "",
            "beam": "",
            "editorial": "",
            "grace_q": "",
            "is_groupetto": False,
            "is_rest": False,
        }
        assert _reconstruct_canonical(components) == "4c"

    def test_note_with_all_components(self):
        """Reconstruct note with multiple components."""
        components = {
            "duration": "8",
            "pitch": "f",
            "accidental": "#",
            "pause": "",
            "ornament": "",
            "harmonic": "",
            "articulation": "'",
            "arpeggio": "",
            "slur": "(",
            "breath": "",
            "tie": "[",
            "beam": "J",
            "editorial": "",
            "grace_q": "",
            "is_groupetto": False,
            "is_rest": False,
        }
        assert _reconstruct_canonical(components) == "8f#'([J"

    def test_grace_note(self):
        """Reconstruct grace note."""
        components = {
            "duration": "8",
            "pitch": "c",
            "accidental": "",
            "pause": "",
            "ornament": "",
            "harmonic": "",
            "articulation": "",
            "arpeggio": "",
            "slur": "",
            "breath": "",
            "tie": "",
            "beam": "L",
            "editorial": "",
            "grace_q": "q",
            "is_groupetto": False,
            "is_rest": False,
        }
        assert _reconstruct_canonical(components) == "8cqL"

    def test_groupetto(self):
        """Reconstruct groupetto."""
        components = {
            "duration": "8",
            "pitch": "c",
            "accidental": "#",
            "pause": "",
            "ornament": "",
            "harmonic": "",
            "articulation": "",
            "arpeggio": "",
            "slur": "",
            "breath": "",
            "tie": "",
            "beam": "L",
            "editorial": "",
            "grace_q": "",
            "is_groupetto": True,
            "is_rest": False,
        }
        assert _reconstruct_canonical(components) == "8qqc#L"

    def test_rest(self):
        """Reconstruct rest."""
        components = {
            "duration": "4",
            "pitch": "r",
            "accidental": "",
            "pause": "",
            "ornament": "",
            "harmonic": "",
            "articulation": "",
            "arpeggio": "",
            "slur": "",
            "breath": "",
            "tie": "",
            "beam": "",
            "editorial": "yy",
            "grace_q": "",
            "is_groupetto": False,
            "is_rest": True,
        }
        assert _reconstruct_canonical(components) == "4ryy"


class TestCanonicalizeToken:
    """Tests for the _canonicalize_token helper function."""

    def test_non_note_tokens_unchanged(self):
        """Non-note tokens should pass through unchanged."""
        assert _canonicalize_token(".") == "."
        assert _canonicalize_token("=") == "="
        assert _canonicalize_token("*clefG2") == "*clefG2"
        assert _canonicalize_token("!comment") == "!comment"

    def test_canonical_note_unchanged(self):
        """Already canonical notes should be unchanged."""
        assert _canonicalize_token("8f#[J") == "8f#[J"

    def test_reorders_misplaced_tie(self):
        """Misplaced tie should be moved."""
        assert _canonicalize_token("[8fJ") == "8f[J"

    def test_reorders_misplaced_slur(self):
        """Misplaced slur should be moved."""
        assert _canonicalize_token("(4cL") == "4c(L"

    def test_drops_unknown_chars_instead_of_preserving(self):
        """Unknown chars should be dropped (not preserved in beam)."""
        assert _canonicalize_token("8ffLs") == "8ffL"

    def test_counts_unknown_chars_when_counter_provided(self):
        """Unknown-char drops should be counted when a counter is provided."""
        unknown_counter = {}
        assert _canonicalize_token("8ffLs", unknown_counter=unknown_counter) == "8ffL"
        assert unknown_counter == {"s": 1}


class TestUnknownDropStats:
    """Tests for per-pass unknown-character drop stats."""

    def test_transform_populates_context_stats(self):
        pass_obj = CanonicalizeNoteOrder()
        ctx = NormalizationContext()
        pass_obj.prepare("8czL\t8dQ", ctx)

        result = pass_obj.transform("8czL\t8dQ", ctx)

        assert result == "8cL\t8d"
        assert ctx["canonicalize_note_order"]["unknown_char_drops_total"] == 2
        assert ctx["canonicalize_note_order"]["unknown_char_drop_counts"] == {"Q": 1, "z": 1}
