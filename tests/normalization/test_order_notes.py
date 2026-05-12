"""Tests for OrderNotes pass and helper functions."""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import OrderNotes


class TestOrderNotes:
    """Tests for OrderNotes pass."""

    def test_pass_exists(self):
        """OrderNotes pass should be instantiable."""
        pass_obj = OrderNotes()
        assert pass_obj.name == "order_notes"

    def test_accepts_ascending_parameter(self):
        """OrderNotes should accept ascending parameter."""
        pass_obj = OrderNotes(ascending=False)
        assert pass_obj.ascending is False

    def test_orders_two_note_chord_ascending(self):
        """OrderNotes should order two-note chord from low to high pitch."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # A (MIDI 69) is higher than F# (MIDI 66)
        input_text = "4A 4F#"
        expected = "4F# 4A"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_orders_three_note_chord_ascending(self):
        """OrderNotes should order three-note chord from low to high pitch."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # G E C should become C E G
        input_text = "4g 4e 4c"
        expected = "4c 4e 4g"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_orders_chord_with_sharps(self):
        """OrderNotes should handle notes with sharp accidentals."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # C# is higher than C, so order should be C C# E
        input_text = "4c# 4c 4e"
        expected = "4c 4c# 4e"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_orders_chord_with_flats(self):
        """OrderNotes should handle notes with flat accidentals."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # E- (Eb) is lower than E, so order should be E- E G
        input_text = "4e 4g 4e-"
        expected = "4e- 4e 4g"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_orders_chord_across_octaves(self):
        """OrderNotes should handle notes across different octaves."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # Uppercase C is lower octave than lowercase c
        # Order should be: C (MIDI ~48) < c (MIDI ~60) < e (MIDI ~64)
        input_text = "4e 4C 4c"
        expected = "4C 4c 4e"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_orders_chord_with_multiple_octave_markers(self):
        """OrderNotes should handle notes with multiple octave markers."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # ee is higher octave than e
        input_text = "4ee 4c 4e"
        expected = "4c 4e 4ee"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_orders_complex_chord(self):
        """OrderNotes should handle complex chords with mixed characteristics."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # Mixed octaves, accidentals, and pitches
        input_text = "4a 4c# 4E 4g"
        expected = "4E 4c# 4g 4a"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_already_ordered_chord(self):
        """OrderNotes should preserve already correctly ordered chords."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        input_text = "4c 4e 4g"
        expected = "4c 4e 4g"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_single_note(self):
        """OrderNotes should leave single notes unchanged."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        input_text = "4c"
        expected = "4c"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_rest(self):
        """OrderNotes should leave rests unchanged."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        input_text = "4r"
        expected = "4r"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_orders_notes_with_articulation_marks(self):
        """OrderNotes should preserve articulation marks while ordering."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # Notes with L, J, k markers should be ordered and beams consolidated on last note
        input_text = "4gL 4eL 4cL"
        expected = "4c 4e 4gL"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_orders_notes_with_ties_and_slurs(self):
        """OrderNotes should preserve ties and slurs while ordering."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # Notes with [ ] markers (ties/slurs)
        input_text = "4g[ 4c["
        expected = "4c[ 4g["

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_line_with_multiple_tokens(self):
        """OrderNotes should handle lines with multiple tab-separated tokens."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # Line with three tokens (three spines)
        input_text = "4r\t2.D 2.DD\t4d\t4A 4F#"
        expected = "4r\t2.DD 2.D\t4d\t4F# 4A"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_multiline_kern_string(self):
        """OrderNotes should handle multi-line kern strings."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
4g 4e 4c\t4ee 4cc
4F# 4A\t4a 4f#"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4c 4e 4g\t4cc 4ee
4F# 4A\t4f# 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_barlines(self):
        """OrderNotes should leave barlines unchanged."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        input_text = "=\t="
        expected = "=\t="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_interpretations(self):
        """OrderNotes should leave interpretation tokens unchanged."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        input_text = "*clefF4\t*clefG2\n*k[b-]\t*k[b-]"
        expected = "*clefF4\t*clefG2\n*k[b-]\t*k[b-]"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_null_tokens(self):
        """OrderNotes should leave null tokens (.) unchanged."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        input_text = ".\t.\t."
        expected = ".\t.\t."

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_orders_notes_descending(self):
        """OrderNotes should order notes from high to low when ascending=False."""
        pass_obj = OrderNotes(ascending=False)
        ctx = NormalizationContext()

        input_text = "4c 4e 4g"
        expected = "4g 4e 4c"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    @pytest.mark.skip(reason="Not yet implemented")
    def test_handles_same_pitch_different_duration(self):
        """OrderNotes should handle same pitch with different durations."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # Two C notes with different durations - order should be stable
        # (or based on duration if that's desired behavior)
        input_text = "4c 2c"
        expected = "2c 4c"  # Or could be "2c 4c" depending on desired behavior

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        # For now, accept either ordering for same pitch
        assert result == expected

    def test_real_world_example_from_docs(self):
        """Test the exact example from normalization_candidates.txt."""
        pass_obj = OrderNotes(ascending=True)
        ctx = NormalizationContext()

        # Exact example from the docs
        input_text = "4r\t2.D 2.DD\t4d\t4A 4F#"
        expected = "4r\t2.DD 2.D\t4d\t4F# 4A"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected


class TestKernNoteParser:
    """Tests for parse_kern_note() helper function."""

    def test_parse_simple_note(self):
        """Should parse a simple note like '4c'."""
        from scripts.dataset_generation.normalization.passes.order_notes import parse_kern_note

        result = parse_kern_note("4c")

        assert result["duration"] == "4"
        assert result["pitch"] == "c"
        assert result["octave"] == 4  # Assuming middle C is octave 4
        assert result["accidental"] == ""

    def test_parse_note_with_sharp(self):
        """Should parse a note with sharp accidental."""
        from scripts.dataset_generation.normalization.passes.order_notes import parse_kern_note

        result = parse_kern_note("4c#")

        assert result["pitch"] == "c"
        assert result["accidental"] == "#"

    def test_parse_note_with_flat(self):
        """Should parse a note with flat accidental."""
        from scripts.dataset_generation.normalization.passes.order_notes import parse_kern_note

        result = parse_kern_note("4e-")

        assert result["pitch"] == "e"
        assert result["accidental"] == "-"

    def test_parse_uppercase_note(self):
        """Should recognize uppercase as lower octave."""
        from scripts.dataset_generation.normalization.passes.order_notes import parse_kern_note

        result = parse_kern_note("4C")

        assert result["pitch"] == "C"
        assert result["octave"] == 3  # Assuming uppercase is one octave lower

    def test_parse_multiple_octave_markers(self):
        """Should parse notes with multiple octave markers like 'ee'."""
        from scripts.dataset_generation.normalization.passes.order_notes import parse_kern_note

        result = parse_kern_note("4ee")

        assert result["pitch"] == "e"
        assert result["octave"] == 5  # Two e's means higher octave

    def test_parse_note_with_articulation(self):
        """Should parse and preserve articulation marks."""
        from scripts.dataset_generation.normalization.passes.order_notes import parse_kern_note

        result = parse_kern_note("4cL")

        assert result["pitch"] == "c"
        assert result["duration"] == "4"
        assert "L" in result.get("articulations", []) or result.get("articulation") == "L"

    def test_parse_note_with_tie(self):
        """Should parse and preserve tie markers."""
        from scripts.dataset_generation.normalization.passes.order_notes import parse_kern_note

        result = parse_kern_note("4c[")

        assert result["pitch"] == "c"
        assert "[" in result.get("ties", []) or result.get("tie") == "["

    def test_parse_rest(self):
        """Should recognize and parse rests."""
        from scripts.dataset_generation.normalization.passes.order_notes import parse_kern_note

        result = parse_kern_note("4r")

        assert result["is_rest"] is True or result["pitch"] == "r"
        assert result["duration"] == "4"

    def test_parse_complex_note(self):
        """Should parse complex notes with multiple modifiers."""
        from scripts.dataset_generation.normalization.passes.order_notes import parse_kern_note

        result = parse_kern_note("8dd#LJ")

        assert result["pitch"] == "d"
        assert result["accidental"] == "#"
        assert result["octave"] == 5
        assert result["duration"] == "8"

    def test_parse_dotted_duration(self):
        """Should parse dotted durations."""
        from scripts.dataset_generation.normalization.passes.order_notes import parse_kern_note

        result = parse_kern_note("2.c")

        assert result["duration"] == "2." or (result["duration"] == "2" and result.get("dotted"))
        assert result["pitch"] == "c"


class TestKernPitchToMidi:
    """Tests for kern_pitch_to_midi() helper function."""

    def test_middle_c(self):
        """Middle C should be MIDI 60."""
        from scripts.dataset_generation.normalization.passes.order_notes import kern_pitch_to_midi

        result = kern_pitch_to_midi("c", 4, "")
        assert result == 60

    def test_pitch_with_sharp(self):
        """C# should be MIDI 61."""
        from scripts.dataset_generation.normalization.passes.order_notes import kern_pitch_to_midi

        result = kern_pitch_to_midi("c", 4, "#")
        assert result == 61

    def test_pitch_with_flat(self):
        """Db should be MIDI 61 (same as C#)."""
        from scripts.dataset_generation.normalization.passes.order_notes import kern_pitch_to_midi

        result = kern_pitch_to_midi("d", 4, "-")
        assert result == 61

    def test_different_octaves(self):
        """Same pitch in different octaves should differ by 12."""
        from scripts.dataset_generation.normalization.passes.order_notes import kern_pitch_to_midi

        c3 = kern_pitch_to_midi("c", 3, "")
        c4 = kern_pitch_to_midi("c", 4, "")
        c5 = kern_pitch_to_midi("c", 5, "")

        assert c4 - c3 == 12
        assert c5 - c4 == 12

    def test_chromatic_scale(self):
        """Chromatic scale should increment by 1."""
        from scripts.dataset_generation.normalization.passes.order_notes import kern_pitch_to_midi

        c = kern_pitch_to_midi("c", 4, "")
        d = kern_pitch_to_midi("d", 4, "")
        e = kern_pitch_to_midi("e", 4, "")
        f = kern_pitch_to_midi("f", 4, "")
        g = kern_pitch_to_midi("g", 4, "")
        a = kern_pitch_to_midi("a", 4, "")
        b = kern_pitch_to_midi("b", 4, "")

        assert d - c == 2  # C to D is whole step
        assert e - d == 2  # D to E is whole step
        assert f - e == 1  # E to F is half step
        assert g - f == 2  # F to G is whole step
        assert a - g == 2  # G to A is whole step
        assert b - a == 2  # A to B is whole step

    def test_a440(self):
        """A4 should be MIDI 69 (A440)."""
        from scripts.dataset_generation.normalization.passes.order_notes import kern_pitch_to_midi

        result = kern_pitch_to_midi("a", 4, "")
        assert result == 69

    def test_uppercase_lower_octave(self):
        """Uppercase C (C3) should be lower than lowercase c (C4)."""
        from scripts.dataset_generation.normalization.passes.order_notes import kern_pitch_to_midi

        # Assuming uppercase is C3 and lowercase is C4
        C = kern_pitch_to_midi("C", 3, "")
        c = kern_pitch_to_midi("c", 4, "")

        assert C < c
        assert c - C == 12


class TestSortNotesByPitch:
    """Tests for sort_notes_by_pitch() helper function."""

    def test_sort_ascending(self):
        """Should sort notes from low to high pitch."""
        from scripts.dataset_generation.normalization.passes.order_notes import sort_notes_by_pitch

        notes = [
            {"pitch": "e", "octave": 4, "accidental": "", "midi": 64},
            {"pitch": "c", "octave": 4, "accidental": "", "midi": 60},
            {"pitch": "g", "octave": 4, "accidental": "", "midi": 67},
        ]

        result = sort_notes_by_pitch(notes, ascending=True)

        assert result[0]["midi"] == 60  # C
        assert result[1]["midi"] == 64  # E
        assert result[2]["midi"] == 67  # G

    def test_sort_descending(self):
        """Should sort notes from high to low pitch."""
        from scripts.dataset_generation.normalization.passes.order_notes import sort_notes_by_pitch

        notes = [
            {"pitch": "c", "octave": 4, "accidental": "", "midi": 60},
            {"pitch": "e", "octave": 4, "accidental": "", "midi": 64},
            {"pitch": "g", "octave": 4, "accidental": "", "midi": 67},
        ]

        result = sort_notes_by_pitch(notes, ascending=False)

        assert result[0]["midi"] == 67  # G
        assert result[1]["midi"] == 64  # E
        assert result[2]["midi"] == 60  # C

    def test_sort_with_accidentals(self):
        """Should sort correctly with accidentals."""
        from scripts.dataset_generation.normalization.passes.order_notes import sort_notes_by_pitch

        notes = [
            {"pitch": "c", "octave": 4, "accidental": "#", "midi": 61},  # C#
            {"pitch": "c", "octave": 4, "accidental": "", "midi": 60},  # C
            {"pitch": "d", "octave": 4, "accidental": "", "midi": 62},  # D
        ]

        result = sort_notes_by_pitch(notes, ascending=True)

        assert result[0]["midi"] == 60  # C
        assert result[1]["midi"] == 61  # C#
        assert result[2]["midi"] == 62  # D

    def test_sort_across_octaves(self):
        """Should sort correctly across octaves."""
        from scripts.dataset_generation.normalization.passes.order_notes import sort_notes_by_pitch

        notes = [
            {"pitch": "c", "octave": 5, "accidental": "", "midi": 72},  # c (higher octave)
            {"pitch": "C", "octave": 3, "accidental": "", "midi": 48},  # C (lower octave)
            {"pitch": "c", "octave": 4, "accidental": "", "midi": 60},  # c (middle octave)
        ]

        result = sort_notes_by_pitch(notes, ascending=True)

        assert result[0]["midi"] == 48  # C
        assert result[1]["midi"] == 60  # c
        assert result[2]["midi"] == 72  # cc

    def test_sort_empty_list(self):
        """Should handle empty list."""
        from scripts.dataset_generation.normalization.passes.order_notes import sort_notes_by_pitch

        result = sort_notes_by_pitch([], ascending=True)
        assert result == []

    def test_sort_single_note(self):
        """Should handle single note."""
        from scripts.dataset_generation.normalization.passes.order_notes import sort_notes_by_pitch

        notes = [{"pitch": "c", "octave": 4, "accidental": "", "midi": 60}]
        result = sort_notes_by_pitch(notes, ascending=True)

        assert len(result) == 1
        assert result[0]["midi"] == 60
