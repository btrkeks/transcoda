"""Tests for courtesy-natural augmentation."""

from __future__ import annotations

from scripts.dataset_generation.augmentation.courtesy_naturals import (
    apply_courtesy_naturals,
    parse_key_signature_token,
)


class TestParseKeySignatureToken:
    def test_parses_empty_key_signature(self) -> None:
        assert parse_key_signature_token("*k[]") == {}

    def test_parses_sharp_key_signature(self) -> None:
        assert parse_key_signature_token("*k[f#c#g#]") == {
            "f": "#",
            "c": "#",
            "g": "#",
        }

    def test_parses_flat_key_signature(self) -> None:
        assert parse_key_signature_token("*k[b-e-a-]") == {
            "b": "-",
            "e": "-",
            "a": "-",
        }

    def test_rejects_invalid_key_signature(self) -> None:
        assert parse_key_signature_token("*k[f]") is None


class TestApplyCourtesyNaturals:
    def test_adds_natural_when_implied_by_key_signature(self) -> None:
        krn = """**kern
*clefG2
*k[f#]
*M4/4
4g
4f#
4g
*-"""
        result = apply_courtesy_naturals(krn, per_note_probability=1.0)

        assert "4gn" in result
        assert "4f#" in result

    def test_skips_when_natural_is_required_after_measure_accidental(self) -> None:
        krn = """**kern
*clefG2
*k[]
*M4/4
4f#
4f
*-"""
        result = apply_courtesy_naturals(krn, per_note_probability=1.0)

        assert result == krn

    def test_barline_resets_measure_accidental_state(self) -> None:
        krn = """**kern
*clefG2
*k[]
*M4/4
4f#
=1
4f
*-"""
        result = apply_courtesy_naturals(krn, per_note_probability=1.0)

        assert "4fn" in result

    def test_respects_flat_key_signature(self) -> None:
        krn = """**kern
*clefG2
*k[b-e-]
*M4/4
4a
4b-
4e-
*-"""
        result = apply_courtesy_naturals(krn, per_note_probability=1.0)

        assert "4an" in result
        assert "4b-" in result
        assert "4e-" in result

    def test_existing_explicit_naturals_are_preserved(self) -> None:
        krn = """**kern
*clefG2
*k[]
*M4/4
4cn
4d
*-"""
        result = apply_courtesy_naturals(krn, per_note_probability=1.0)

        assert result.count("4cn") == 1
        assert "4dn" in result

    def test_chords_are_left_unchanged_and_lock_spine_until_barline(self) -> None:
        krn = """**kern
*clefG2
*k[]
*M4/4
4c 4e
4g
=1
4a
*-"""
        result = apply_courtesy_naturals(krn, per_note_probability=1.0)

        lines = result.splitlines()
        assert lines[4] == "4c 4e"
        assert lines[5] == "4g"
        assert lines[7] == "4an"

    def test_multispine_input_preserves_columns(self) -> None:
        krn = """**kern\t**kern
*clefF4\t*clefG2
*k[b-]\t*k[]
*M4/4\t*M4/4
4B-\t4c
4A\t4d
*-\t*-"""
        result = apply_courtesy_naturals(krn, per_note_probability=1.0)

        for line in result.splitlines():
            assert line.count("\t") == 1
        assert "4An\t4dn" in result

    def test_spine_manipulators_cause_safe_noop(self) -> None:
        krn = """**kern\t**kern
*clefG2\t*clefG2
*k[]\t*k[]
*\t*^
4c\t4e\t4g
*\t*v\t*v
*-\t*-"""
        result = apply_courtesy_naturals(krn, per_note_probability=1.0)

        assert result == krn
