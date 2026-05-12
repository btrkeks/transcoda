"""Tests for invisible_rests augmentation."""

from __future__ import annotations

import pytest

from scripts.dataset_generation.augmentation.invisible_rests import (
    INVISIBLE_REST_SUFFIX,
    _all_tokens_are_rests,
    _is_rest_token,
    _is_structural_line,
    apply_invisible_rests,
)


class TestIsRestToken:
    """Tests for _is_rest_token helper."""

    @pytest.mark.parametrize("token", ["4r", "8r", "2r", "16r", "4rr", "8ryy"])
    def test_detects_rests(self, token: str) -> None:
        assert _is_rest_token(token) is True

    @pytest.mark.parametrize("token", ["4c", "8eL", "4d#", "16ff-J", "[4c"])
    def test_ignores_notes(self, token: str) -> None:
        assert _is_rest_token(token) is False


class TestIsStructuralLine:
    """Tests for _is_structural_line helper."""

    @pytest.mark.parametrize(
        "line",
        ["", "*clefG2", "*^", "*v", "**kern", "*-", "=1", "=||"],
    )
    def test_structural_lines(self, line: str) -> None:
        assert _is_structural_line(line) is True

    @pytest.mark.parametrize("line", ["4c\t4e", "8rL\t8eL", "2r\t2r"])
    def test_data_lines_not_structural(self, line: str) -> None:
        assert _is_structural_line(line) is False


class TestAllTokensAreRests:
    """Tests for _all_tokens_are_rests helper."""

    def test_all_rests(self) -> None:
        assert _all_tokens_are_rests(["4r", "8r"]) is True
        assert _all_tokens_are_rests(["4r"]) is True

    def test_mixed(self) -> None:
        assert _all_tokens_are_rests(["4r", "4c"]) is False
        assert _all_tokens_are_rests(["4e", "4r"]) is False

    def test_no_rests(self) -> None:
        assert _all_tokens_are_rests(["4c", "4e"]) is False


class TestApplyInvisibleRests:
    """Tests for apply_invisible_rests function."""

    def test_adds_yy_in_multivoice_region_prob_one(self) -> None:
        """Rests in multi-voice regions should get yy with probability 1.0."""
        krn = """**kern\t**kern
*clefG2\t*clefG2
*\t*^
4c\t4e\t4r
4d\t4f\t8r
*\t*v\t*v
4e\t4g
*-\t*-"""
        result = apply_invisible_rests(krn, per_rest_probability=1.0)

        # The rests in the split region should have yy
        assert "4ryy" in result
        assert "8ryy" in result

    def test_no_yy_with_probability_zero(self) -> None:
        """No yy should be added with probability 0.0."""
        krn = """**kern\t**kern
*\t*^
4c\t4e\t4r
*\t*v\t*v
*-\t*-"""
        result = apply_invisible_rests(krn, per_rest_probability=0.0)

        assert INVISIBLE_REST_SUFFIX not in result

    def test_no_yy_outside_multivoice_region(self) -> None:
        """Rests outside multi-voice regions should NOT get yy."""
        krn = """**kern\t**kern
*clefG2\t*clefG2
4r\t4r
4c\t4e
*-\t*-"""
        result = apply_invisible_rests(krn, per_rest_probability=1.0)

        # No yy should be added - not in a multi-voice region
        assert INVISIBLE_REST_SUFFIX not in result
        assert result == krn

    def test_skips_all_rests_line(self) -> None:
        """Lines where ALL tokens are rests should NOT get yy."""
        krn = """**kern\t**kern
*\t*^
4r\t4r\t4r
4c\t4e\t4r
*\t*v\t*v
*-\t*-"""
        result = apply_invisible_rests(krn, per_rest_probability=1.0)

        lines = result.splitlines()
        # Line with all rests (4r\t4r\t4r) should be unchanged
        all_rests_line = lines[2]
        assert all_rests_line == "4r\t4r\t4r"
        # Line with mixed (4c\t4e\t4r) should have yy on the rest
        mixed_line = lines[3]
        assert "4ryy" in mixed_line

    def test_empty_string(self) -> None:
        """Empty kern should be returned unchanged."""
        assert apply_invisible_rests("") == ""

    def test_no_multivoice_regions(self, simple_kern: str) -> None:
        """Kern without multi-voice regions should be unchanged."""
        result = apply_invisible_rests(simple_kern, per_rest_probability=1.0)
        assert result == simple_kern

    def test_multivoice_exit_resets_state(self) -> None:
        """After *v merge, rests should NOT get yy until next *^."""
        krn = """**kern\t**kern
*\t*^
4c\t4e\t4r
*\t*v\t*v
4r\t4r
*-\t*-"""
        result = apply_invisible_rests(krn, per_rest_probability=1.0)

        lines = result.splitlines()
        # Inside region: should have yy
        assert "4ryy" in lines[2]
        # Outside region after merge: should NOT have yy
        assert lines[4] == "4r\t4r"

    def test_preserves_trailing_markers(self) -> None:
        """yy should be inserted before trailing markers (L, J, etc.)."""
        krn = """**kern\t**kern
*\t*^
4c\t4e\t8rL
4d\t4f\t8rJ
*\t*v\t*v
*-\t*-"""
        result = apply_invisible_rests(krn, per_rest_probability=1.0)

        # yy should come before L and J
        assert "8ryyL" in result
        assert "8ryyJ" in result

    def test_multiple_regions(self) -> None:
        """Multiple multi-voice regions should all be processed."""
        krn = """**kern\t**kern
*\t*^
4c\t4e\t4r
*\t*v\t*v
4c\t4e
*\t*^
4d\t4f\t8r
*\t*v\t*v
*-\t*-"""
        result = apply_invisible_rests(krn, per_rest_probability=1.0)

        # Both regions should have yy on rests
        assert result.count(INVISIBLE_REST_SUFFIX) == 2
