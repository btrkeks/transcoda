"""Tests for kern_utils module."""

from __future__ import annotations

from scripts.dataset_generation.augmentation.kern_utils import (
    append_to_token,
    apply_suffix_to_notes,
    find_barline_indices,
    find_note_tokens,
    get_spine_count,
    sample_positions,
)


class TestFindNoteTokens:
    def test_finds_notes_in_simple_kern(self, simple_kern: str) -> None:
        positions = find_note_tokens(simple_kern)

        # Should find 8 notes (4 lines x 2 spines)
        assert len(positions) == 8

        # Check first note position
        first = positions[0]
        assert first.line_idx == 4  # After header lines
        assert first.col_idx == 0
        assert first.token == "4c"

    def test_skips_interpretation_lines(self, simple_kern: str) -> None:
        positions = find_note_tokens(simple_kern)

        # No positions should be on interpretation lines (starting with *)
        for pos in positions:
            assert not pos.token.startswith("*")

    def test_skips_barlines(self, simple_kern: str) -> None:
        positions = find_note_tokens(simple_kern)

        # No positions should be on barlines
        for pos in positions:
            assert not pos.token.startswith("=")

    def test_empty_string_returns_empty_list(self, empty_kern: str) -> None:
        positions = find_note_tokens(empty_kern)
        assert positions == []

    def test_skips_rests(self) -> None:
        kern = "**kern\n4r\n4c\n*-"
        positions = find_note_tokens(kern)

        # Should only find one note (not the rest)
        assert len(positions) == 1
        assert positions[0].token == "4c"

    def test_skips_grace_notes(self) -> None:
        kern = "**kern\n8qc\n4c\n*-"
        positions = find_note_tokens(kern)

        # Should only find the regular note
        assert len(positions) == 1
        assert positions[0].token == "4c"


class TestFindBarlineIndices:
    def test_finds_barlines(self, simple_kern: str) -> None:
        indices = find_barline_indices(simple_kern)

        # Should find 2 barlines
        assert len(indices) == 2

    def test_empty_string_returns_empty_list(self, empty_kern: str) -> None:
        indices = find_barline_indices(empty_kern)
        assert indices == []


class TestAppendToToken:
    def test_appends_to_simple_note(self) -> None:
        result = append_to_token("4c", "'")
        assert result == "4c'"

    def test_appends_before_beam_markers(self) -> None:
        # L and J should stay at the end
        result = append_to_token("8cL", "'")
        assert result == "8c'L"

        result = append_to_token("8cJ", "'")
        assert result == "8c'J"

        result = append_to_token("8cLJ", "'")
        assert result == "8c'LJ"

    def test_appends_before_tie_markers(self) -> None:
        result = append_to_token("[4c", "'")
        assert result == "[4c'"

        result = append_to_token("4c]", "'")
        assert result == "4c']"

    def test_appends_with_accidentals(self) -> None:
        result = append_to_token("4c#", "'")
        assert result == "4c#'"

        result = append_to_token("4c-", "'")
        assert result == "4c-'"

    def test_complex_token(self) -> None:
        # Note with accidental and beam markers
        result = append_to_token("8ee-LJ", ";")
        assert result == "8ee-;LJ"


class TestApplySuffixToNotes:
    def test_applies_suffix_to_selected_positions(self, simple_kern: str) -> None:
        positions = find_note_tokens(simple_kern)

        # Select first two positions
        selected = positions[:2]
        result = apply_suffix_to_notes(simple_kern, selected, "'")

        # Check that staccatos were added
        lines = result.splitlines()
        assert "4c'" in lines[4]
        assert "4e" in lines[4]  # Second note not selected

    def test_empty_positions_returns_unchanged(self, simple_kern: str) -> None:
        result = apply_suffix_to_notes(simple_kern, [], "'")
        assert result == simple_kern


class TestSamplePositions:
    def test_probability_zero_returns_empty(self, simple_kern: str) -> None:
        positions = find_note_tokens(simple_kern)
        selected = sample_positions(positions, 0.0)
        assert selected == []

    def test_probability_one_returns_all(self, simple_kern: str) -> None:
        positions = find_note_tokens(simple_kern)
        selected = sample_positions(positions, 1.0)
        assert len(selected) == len(positions)


class TestGetSpineCount:
    def test_simple_kern(self, simple_kern: str) -> None:
        assert get_spine_count(simple_kern) == 2

    def test_empty_string(self, empty_kern: str) -> None:
        assert get_spine_count(empty_kern) == 0

    def test_single_spine(self) -> None:
        kern = "**kern\n4c\n*-"
        assert get_spine_count(kern) == 1
