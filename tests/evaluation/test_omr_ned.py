"""Tests for OMR-NED metric computation.

These tests verify the OMR-NED (Normalized Edit Distance) metric implementation.
The musicdiff library is an optional dependency, so tests are skipped if not installed.
"""

import tempfile
from pathlib import Path

import pytest

from src.evaluation.omr_ned import (
    OMRNEDResult,
    _ensure_kern_header,
    compute_omr_ned,
    is_musicdiff_available,
)


class TestMusicdiffAvailability:
    """Test musicdiff availability detection."""

    def test_is_musicdiff_available_returns_bool(self):
        """is_musicdiff_available() should return a boolean."""
        result = is_musicdiff_available()
        assert isinstance(result, bool)


class TestEnsureKernHeader:
    """Test the header addition utility function."""

    def test_adds_header_when_missing_single_spine(self):
        """Should add **kern header for single-spine data."""
        kern = "4c\n4d\n4e"
        result = _ensure_kern_header(kern)
        assert result.startswith("**kern\n")
        assert "4c" in result

    def test_adds_header_when_missing_multiple_spines(self):
        """Should add correct number of **kern headers for multi-spine data."""
        kern = "4c\t4e\n4d\t4f"
        result = _ensure_kern_header(kern)
        assert result.startswith("**kern\t**kern\n")

    def test_preserves_existing_header(self):
        """Should not modify data that already has a header."""
        kern = "**kern\t**kern\n4c\t4e\n*-\t*-"
        result = _ensure_kern_header(kern)
        assert result == kern

    def test_adds_terminator_when_missing(self):
        """Should add spine terminator if missing."""
        kern = "4c\n4d"
        result = _ensure_kern_header(kern)
        assert result.endswith("*-")

    def test_does_not_add_duplicate_terminator(self):
        """Should not add terminator if already present."""
        kern = "4c\n4d\n*-"
        result = _ensure_kern_header(kern)
        # Should only have one *- at the end
        assert result.count("*-") == 1

    def test_handles_empty_string(self):
        """Should handle empty string gracefully."""
        result = _ensure_kern_header("")
        assert result == ""


class TestComputeOMRNEDWithoutMusicdiff:
    """Tests that work regardless of musicdiff installation."""

    def test_returns_omrned_result_type(self):
        """Should always return an OMRNEDResult dataclass."""
        result = compute_omr_ned("**kern\n4c\n*-", "**kern\n4c\n*-")
        assert isinstance(result, OMRNEDResult)

    def test_empty_prediction_returns_error(self):
        """Empty prediction should return error (either Empty or not installed)."""
        result = compute_omr_ned("", "**kern\n4c\n*-")
        assert result.omr_ned is None
        assert result.parse_error is not None
        # Either musicdiff not installed or Empty error
        assert "Empty" in result.parse_error or "not installed" in result.parse_error

    def test_empty_target_returns_error(self):
        """Empty target should return parse error."""
        result = compute_omr_ned("**kern\n4c\n*-", "")
        assert result.omr_ned is None
        assert result.parse_error is not None

    def test_both_empty_returns_error(self):
        """Both empty should return parse error."""
        result = compute_omr_ned("", "")
        assert result.omr_ned is None
        assert result.parse_error is not None


# Helper to skip tests requiring musicdiff
requires_musicdiff = pytest.mark.skipif(
    not is_musicdiff_available(),
    reason="musicdiff not installed. Install with: uv sync --group omr-ned",
)


@requires_musicdiff
class TestComputeOMRNEDWithMusicdiff:
    """Tests that require musicdiff to be installed."""

    def test_identical_scores_return_zero(self):
        """Identical scores should have 0% NED."""
        kern = "**kern\n4c\n4d\n4e\n*-"
        result = compute_omr_ned(kern, kern)

        assert result.parse_error is None
        assert result.omr_ned is not None
        assert result.omr_ned == pytest.approx(0.0, abs=0.01)
        assert result.edit_distance == 0

    def test_different_pitches_return_nonzero(self):
        """Different pitches should have >0% NED."""
        pred = "**kern\n4c\n4d\n4e\n*-"
        target = "**kern\n4c\n4d\n4f\n*-"  # e -> f
        result = compute_omr_ned(pred, target)

        assert result.parse_error is None
        assert result.omr_ned is not None
        assert result.omr_ned > 0

    def test_different_durations_return_nonzero(self):
        """Different durations should have >0% NED."""
        pred = "**kern\n4c\n4d\n*-"
        target = "**kern\n8c\n8d\n*-"  # quarter -> eighth
        result = compute_omr_ned(pred, target)

        assert result.parse_error is None
        assert result.omr_ned is not None
        assert result.omr_ned > 0

    def test_returns_notation_sizes(self):
        """Should return notation sizes for both scores."""
        kern = "**kern\n4c\n4d\n*-"
        result = compute_omr_ned(kern, kern)

        assert result.pred_notation_size is not None
        assert result.gt_notation_size is not None
        assert result.pred_notation_size > 0
        assert result.gt_notation_size > 0

    def test_without_header_still_works(self):
        """Should handle kern without headers (auto-adds them)."""
        pred = "4c\n4d"
        target = "4c\n4d"
        result = compute_omr_ned(pred, target)

        # Should either work or fail gracefully
        assert isinstance(result, OMRNEDResult)

    def test_syntax_errors_fixed_tracked(self):
        """Should track syntax errors that were fixed."""
        kern = "**kern\n4c\n4d\n*-"
        result = compute_omr_ned(kern, kern)

        assert result.syntax_errors_fixed >= 0

    def test_omr_ned_is_percentage(self):
        """OMR-NED should be expressed as percentage (0-100)."""
        pred = "**kern\n4c\n*-"
        target = "**kern\n4c\n4d\n4e\n4f\n*-"  # Much longer target
        result = compute_omr_ned(pred, target)

        if result.omr_ned is not None:
            # Should be a percentage, not a decimal
            assert 0 <= result.omr_ned <= 100

    def test_ned_clamped_to_100(self):
        """NED should be clamped to 100% maximum."""
        # Very different scores
        pred = "**kern\n1c\n*-"
        target = "**kern\n4c\n4d\n4e\n4f\n4g\n4a\n4b\n4cc\n*-"
        result = compute_omr_ned(pred, target)

        if result.omr_ned is not None:
            assert result.omr_ned <= 100.0

    def test_repeated_identical_diffs_return_same_ops_and_cost(self):
        """Repeated identical comparisons should stay stable with patched memo copying."""
        from musicdiff.annotation import AnnScore
        from musicdiff.comparison import Comparison
        import music21 as m21

        pred = "**kern\n4c\n4d\n4e\n*-"
        target = "**kern\n4c\n4d\n4f\n*-"

        def parse_ann_score(kern: str) -> AnnScore:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".krn", delete=False, encoding="utf-8"
            ) as handle:
                handle.write(kern)
                path = Path(handle.name)
            try:
                score = m21.converter.parse(str(path), forceSource=True, acceptSyntaxErrors=False)
            finally:
                path.unlink(missing_ok=True)
            return AnnScore(score)

        first_ops, first_cost = Comparison.annotated_scores_diff(
            parse_ann_score(pred),
            parse_ann_score(target),
        )
        second_ops, second_cost = Comparison.annotated_scores_diff(
            parse_ann_score(pred),
            parse_ann_score(target),
        )

        assert second_cost == first_cost
        assert second_ops == first_ops

    def test_mutating_cached_helper_result_does_not_leak_to_next_lookup(self):
        """Memoized helper results must still be isolated across cache hits."""
        from musicdiff.comparison import Comparison

        original = ["start", "continue", "stop"]
        compare_to = ["start", "stop"]

        Comparison._clear_memoizer_caches()
        first_ops, first_cost = Comparison._beamtuplet_levenshtein_diff(
            original,
            compare_to,
            None,
            None,
            "beam",
        )
        baseline_ops = list(first_ops)
        first_ops.append(("sentinel", None, None, 999))

        second_ops, second_cost = Comparison._beamtuplet_levenshtein_diff(
            original,
            compare_to,
            None,
            None,
            "beam",
        )

        assert second_cost == first_cost
        assert second_ops == baseline_ops
        assert ("sentinel", None, None, 999) not in second_ops


@requires_musicdiff
class TestComputeOMRNEDMultiSpine:
    """Tests for multi-spine (grand staff) kern data."""

    def test_two_spine_identical(self):
        """Two-spine identical scores should have 0% NED."""
        kern = "**kern\t**kern\n4c\t4e\n4d\t4f\n*-\t*-"
        result = compute_omr_ned(kern, kern)

        assert result.parse_error is None
        assert result.omr_ned is not None
        assert result.omr_ned == pytest.approx(0.0, abs=0.01)

    def test_two_spine_different(self):
        """Two-spine different scores should have >0% NED."""
        pred = "**kern\t**kern\n4c\t4e\n4d\t4f\n*-\t*-"
        target = "**kern\t**kern\n4c\t4e\n4d\t4g\n*-\t*-"  # f -> g
        result = compute_omr_ned(pred, target)

        assert result.parse_error is None
        assert result.omr_ned is not None
        assert result.omr_ned > 0
