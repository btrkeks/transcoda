"""Tests for xtuplet augmentation."""

from __future__ import annotations

import pytest

from scripts.dataset_generation.augmentation.xtuplet import (
    XTUPLET,
    _has_tuplet_rhythm,
    _is_power_of_two,
    apply_xtuplet,
)


class TestIsPowerOfTwo:
    """Tests for _is_power_of_two helper."""

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128])
    def test_powers_of_two(self, n: int) -> None:
        assert _is_power_of_two(n) is True

    @pytest.mark.parametrize("n", [0, 3, 5, 6, 7, 9, 10, 12, 14, 15, 24, 48])
    def test_not_powers_of_two(self, n: int) -> None:
        assert _is_power_of_two(n) is False


class TestHasTupletRhythm:
    """Tests for _has_tuplet_rhythm helper."""

    @pytest.mark.parametrize(
        "token",
        [
            "3c",  # triplet half
            "6c",  # triplet quarter
            "12c",  # triplet eighth
            "24c",  # triplet sixteenth
            "5c",  # quintuplet
            "7c",  # septuplet
            "12g-L",  # triplet eighth with accidental and beam
            "6d#J",  # triplet quarter with sharp and beam
        ],
    )
    def test_detects_tuplet_rhythms(self, token: str) -> None:
        assert _has_tuplet_rhythm(token) is True

    @pytest.mark.parametrize(
        "token",
        [
            "1c",  # whole
            "2c",  # half
            "4c",  # quarter
            "8c",  # eighth
            "16c",  # sixteenth
            "32c",  # thirty-second
            "4c'",  # quarter with staccato
            "8eL",  # eighth with beam
            "4d-",  # quarter with flat
        ],
    )
    def test_ignores_non_tuplet_rhythms(self, token: str) -> None:
        assert _has_tuplet_rhythm(token) is False

    @pytest.mark.parametrize("token", ["r", "4r", "."])
    def test_ignores_rests_and_non_notes(self, token: str) -> None:
        assert _has_tuplet_rhythm(token) is False


class TestApplyXtuplet:
    """Tests for apply_xtuplet function."""

    def test_adds_xtuplet_with_probability_one(self) -> None:
        """All tuplet-containing spines should be marked with probability 1.0."""
        krn = """**kern\t**kern
*clefF4\t*clefG2
*k[]\t*k[]
4c\t12e
4d\t12f
=\t=
*-\t*-"""
        result = apply_xtuplet(krn, per_spine_probability=1.0)

        # Should have *Xtuplet in spine 1 (contains 12e, 12f)
        assert XTUPLET in result
        lines = result.splitlines()
        # Find the xtuplet line
        xtuplet_lines = [line for line in lines if XTUPLET in line]
        assert len(xtuplet_lines) == 1
        # Spine 0 should have *, spine 1 should have *Xtuplet
        assert xtuplet_lines[0] == "*\t*Xtuplet"

    def test_no_xtuplet_with_probability_zero(self) -> None:
        """No *Xtuplet should be added with probability 0.0."""
        krn = """**kern\t**kern
*clefF4\t*clefG2
4c\t12e
*-\t*-"""
        result = apply_xtuplet(krn, per_spine_probability=0.0)

        assert XTUPLET not in result

    def test_only_marks_spines_with_tuplets(self) -> None:
        """Spines without tuplets should get * even with probability 1.0."""
        krn = """**kern\t**kern
*clefF4\t*clefG2
4c\t4e
8d\t8f
=\t=
*-\t*-"""
        result = apply_xtuplet(krn, per_spine_probability=1.0)

        # No tuplets in either spine, so no *Xtuplet should be added
        assert XTUPLET not in result
        # Should be unchanged
        assert result == krn

    def test_marks_multiple_spines_with_tuplets(self) -> None:
        """Multiple spines with tuplets should all be marked."""
        krn = """**kern\t**kern
*clefF4\t*clefG2
12c\t12e
12d\t12f
=\t=
*-\t*-"""
        result = apply_xtuplet(krn, per_spine_probability=1.0)

        assert XTUPLET in result
        lines = result.splitlines()
        xtuplet_lines = [line for line in lines if XTUPLET in line]
        assert len(xtuplet_lines) == 1
        # Both spines should have *Xtuplet
        assert xtuplet_lines[0] == "*Xtuplet\t*Xtuplet"

    def test_insertion_before_spine_split(self) -> None:
        """*Xtuplet should be inserted before spine manipulators like *^."""
        krn = """**kern\t**kern
*clefF4\t*clefG2
*k[b-]\t*k[b-]
*\t*^
4c\t12e\t12r
*-\t*-\t*-"""
        result = apply_xtuplet(krn, per_spine_probability=1.0)

        lines = result.splitlines()
        # Find indices
        xtuplet_idx = None
        split_idx = None
        for i, line in enumerate(lines):
            if XTUPLET in line:
                xtuplet_idx = i
            if "*^" in line:
                split_idx = i

        # *Xtuplet should come before *^
        assert xtuplet_idx is not None
        assert split_idx is not None
        assert xtuplet_idx < split_idx

    def test_insertion_after_header(self) -> None:
        """*Xtuplet should be inserted after clef/key/time signature."""
        krn = """**kern\t**kern
*clefF4\t*clefG2
*k[b-]\t*k[b-]
*M4/4\t*M4/4
12c\t4e
*-\t*-"""
        result = apply_xtuplet(krn, per_spine_probability=1.0)

        lines = result.splitlines()
        # Find indices
        time_sig_idx = None
        xtuplet_idx = None
        for i, line in enumerate(lines):
            if "*M4/4" in line:
                time_sig_idx = i
            if XTUPLET in line:
                xtuplet_idx = i

        # *Xtuplet should come after time signature
        assert time_sig_idx is not None
        assert xtuplet_idx is not None
        assert xtuplet_idx > time_sig_idx

    def test_empty_kern(self, empty_kern: str) -> None:
        """Empty kern should be returned unchanged."""
        result = apply_xtuplet(empty_kern, per_spine_probability=1.0)
        assert result == empty_kern

    def test_no_tuplets_returns_unchanged(self, simple_kern: str) -> None:
        """Kern without tuplets should be returned unchanged."""
        result = apply_xtuplet(simple_kern, per_spine_probability=1.0)
        assert result == simple_kern

    def test_various_tuplet_rhythms_detected(self) -> None:
        """Various tuplet rhythms (3, 5, 6, 7, 12, 24) should all be detected."""
        for rhythm in [3, 5, 6, 7, 12, 24]:
            krn = f"""**kern
*clefG2
{rhythm}c
*-"""
            result = apply_xtuplet(krn, per_spine_probability=1.0)
            assert XTUPLET in result, f"Failed to detect rhythm {rhythm}"

    def test_power_of_two_rhythms_not_marked(self) -> None:
        """Power-of-two rhythms (1, 2, 4, 8, 16, 32) should not be marked."""
        for rhythm in [1, 2, 4, 8, 16, 32]:
            krn = f"""**kern
*clefG2
{rhythm}c
*-"""
            result = apply_xtuplet(krn, per_spine_probability=1.0)
            assert XTUPLET not in result, f"Incorrectly marked rhythm {rhythm}"
