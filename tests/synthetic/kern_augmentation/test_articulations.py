"""Tests for articulations module."""

from __future__ import annotations

from scripts.dataset_generation.augmentation.articulations import (
    ACCENT,
    FERMATA,
    SFORZANDO,
    STACCATO,
    apply_accents,
    apply_fermatas,
    apply_sforzandos,
    apply_staccatos,
)


class TestApplyStaccatos:
    def test_adds_staccatos_with_probability_one(self, simple_kern: str) -> None:
        result = apply_staccatos(simple_kern, per_note_probability=1.0)

        # All notes should have staccato
        assert result.count(STACCATO) == 8  # 8 notes in simple_kern

    def test_no_staccatos_with_probability_zero(self, simple_kern: str) -> None:
        result = apply_staccatos(simple_kern, per_note_probability=0.0)

        # No staccatos should be added
        assert STACCATO not in result

    def test_empty_string_returns_empty(self, empty_kern: str) -> None:
        result = apply_staccatos(empty_kern)
        assert result == ""

    def test_preserves_beam_markers(self, kern_with_beams: str) -> None:
        result = apply_staccatos(kern_with_beams, per_note_probability=1.0)

        # Staccatos should be before beam markers
        assert "8c'L" in result or "8c'" in result.replace("L", "").replace("J", "")
        # Beam markers should still be present
        assert "L" in result
        assert "J" in result


class TestApplySforzandos:
    def test_adds_sforzandos_with_probability_one(self, simple_kern: str) -> None:
        result = apply_sforzandos(simple_kern, per_note_probability=1.0)

        # All notes should have sforzando
        assert result.count(SFORZANDO) == 8

    def test_no_sforzandos_with_probability_zero(self, simple_kern: str) -> None:
        result = apply_sforzandos(simple_kern, per_note_probability=0.0)

        assert SFORZANDO not in result


class TestApplyAccents:
    def test_adds_accents_with_probability_one(self, simple_kern: str) -> None:
        result = apply_accents(simple_kern, per_note_probability=1.0)

        # All notes should have accent
        assert result.count(ACCENT) == 8

    def test_no_accents_with_probability_zero(self, simple_kern: str) -> None:
        result = apply_accents(simple_kern, per_note_probability=0.0)

        assert ACCENT not in result


class TestApplyFermatas:
    def test_adds_fermatas_with_probability_one(self, simple_kern: str) -> None:
        result = apply_fermatas(simple_kern, per_note_probability=1.0)

        # All notes should have fermata
        assert result.count(FERMATA) == 8

    def test_no_fermatas_with_probability_zero(self, simple_kern: str) -> None:
        result = apply_fermatas(simple_kern, per_note_probability=0.0)

        assert FERMATA not in result
