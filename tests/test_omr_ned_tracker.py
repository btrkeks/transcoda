"""Tests for OMRNEDTracker class."""

import pytest

from src.core.metrics import OMRNEDTracker
from src.evaluation.omr_ned import is_musicdiff_available

# Sample **kern strings for testing
SIMPLE_KERN_1 = "**kern\t**kern\n4c\t4e\n4d\t4f\n*-\t*-"
SIMPLE_KERN_2 = "**kern\t**kern\n4c\t4e\n4e\t4g\n*-\t*-"  # Different second note
SIMPLE_KERN_IDENTICAL = SIMPLE_KERN_1


# Decorator to skip tests when musicdiff is not available
requires_musicdiff = pytest.mark.skipif(
    not is_musicdiff_available(),
    reason="musicdiff not installed (install with: uv sync --group omr-ned)",
)


class TestOMRNEDTrackerBasics:
    """Test basic tracker functionality that works without musicdiff."""

    def test_tracker_initialization(self):
        """Test tracker can be instantiated."""
        tracker = OMRNEDTracker()
        assert tracker is not None

    def test_enabled_property_reflects_musicdiff_availability(self):
        """Test enabled property matches musicdiff availability."""
        tracker = OMRNEDTracker()
        assert tracker.enabled == is_musicdiff_available()

    def test_empty_tracker_returns_zero_metrics(self):
        """Test compute returns zeros when no samples accumulated."""
        tracker = OMRNEDTracker()
        metrics = tracker.compute()

        assert metrics.overall_score == 0.0
        assert metrics.overall_failures == 0
        assert metrics.by_source_score == {}
        assert metrics.by_source_failures == {}

    def test_reset_clears_samples(self):
        """Test reset clears accumulated samples."""
        tracker = OMRNEDTracker()

        # Accumulate some samples (won't actually store if musicdiff not available)
        tracker.update("pred", "target", "source")
        tracker.reset()

        assert len(tracker) == 0

    def test_len_returns_sample_count(self):
        """Test __len__ returns number of accumulated samples."""
        tracker = OMRNEDTracker()
        initial_len = len(tracker)

        if tracker.enabled:
            tracker.update("pred", "target", "source")
            assert len(tracker) == initial_len + 1
        else:
            # When disabled, update doesn't store samples
            tracker.update("pred", "target", "source")
            assert len(tracker) == 0


@requires_musicdiff
class TestOMRNEDTrackerWithMusicdiff:
    """Tests that require musicdiff to be installed."""

    def test_update_accumulates_samples(self):
        """Test update accumulates samples for later computation."""
        tracker = OMRNEDTracker()

        tracker.update(SIMPLE_KERN_1, SIMPLE_KERN_1, "synth")
        assert len(tracker) == 1

        tracker.update(SIMPLE_KERN_2, SIMPLE_KERN_1, "polish")
        assert len(tracker) == 2

    def test_identical_scores_returns_zero_ned(self):
        """Test identical prediction and target yields 0% OMR-NED."""
        tracker = OMRNEDTracker()

        tracker.update(SIMPLE_KERN_IDENTICAL, SIMPLE_KERN_IDENTICAL, "synth")
        metrics = tracker.compute()

        assert metrics.overall_score == pytest.approx(0.0, abs=0.1)
        assert metrics.overall_failures == 0

    def test_different_scores_returns_nonzero_ned(self):
        """Test different scores yield non-zero OMR-NED."""
        tracker = OMRNEDTracker()

        tracker.update(SIMPLE_KERN_2, SIMPLE_KERN_1, "synth")
        metrics = tracker.compute()

        assert metrics.overall_score > 0.0
        assert metrics.overall_failures == 0

    def test_per_source_metrics(self):
        """Test per-source metrics are computed correctly."""
        tracker = OMRNEDTracker()

        # Add samples from different sources
        tracker.update(SIMPLE_KERN_IDENTICAL, SIMPLE_KERN_IDENTICAL, "synth")
        tracker.update(SIMPLE_KERN_2, SIMPLE_KERN_1, "polish")

        metrics = tracker.compute()

        assert metrics.by_source_score["synth"] == pytest.approx(0.0, abs=0.1)
        assert metrics.by_source_score["polish"] > 0.0
        assert metrics.by_source_failures["synth"] == 0
        assert metrics.by_source_failures["polish"] == 0

    def test_overall_ned_averages_all_sources(self):
        """Test overall OMR-NED is average across all valid samples."""
        tracker = OMRNEDTracker()

        # Two identical pairs (0% each)
        tracker.update(SIMPLE_KERN_IDENTICAL, SIMPLE_KERN_IDENTICAL, "synth")
        tracker.update(SIMPLE_KERN_IDENTICAL, SIMPLE_KERN_IDENTICAL, "polish")

        metrics = tracker.compute()

        assert metrics.overall_score == pytest.approx(0.0, abs=0.1)

    def test_invalid_kern_yields_high_ned(self):
        """Test invalid **kern strings yield high (100%) OMR-NED.

        Note: music21 is very forgiving and parses most invalid strings.
        Invalid content results in high OMR-NED rather than parse failures.
        """
        tracker = OMRNEDTracker()

        # This is clearly invalid kern - but music21 will parse it
        invalid_kern = "not valid kern at all!!!"

        tracker.update(invalid_kern, SIMPLE_KERN_1, "synth")
        metrics = tracker.compute()

        # music21 parses it, resulting in 100% OMR-NED (max error)
        assert metrics.overall_failures == 0
        assert metrics.overall_score == pytest.approx(100.0, abs=1.0)

    def test_mixed_valid_and_garbage(self):
        """Test handling of mix of valid and garbage samples."""
        tracker = OMRNEDTracker()

        # One valid (identical), one garbage (will parse but high error)
        tracker.update(SIMPLE_KERN_IDENTICAL, SIMPLE_KERN_IDENTICAL, "synth")  # 0% NED
        tracker.update("garbage", SIMPLE_KERN_1, "polish")  # ~100% NED

        metrics = tracker.compute()

        # Both parse successfully, average should be ~50%
        assert metrics.overall_failures == 0
        assert metrics.overall_score > 40.0  # Average of ~0% and ~100%
        assert metrics.by_source_score["synth"] == pytest.approx(0.0, abs=0.1)
        assert metrics.by_source_score["polish"] == pytest.approx(100.0, abs=1.0)

    def test_reset_allows_fresh_computation(self):
        """Test reset allows computing fresh metrics."""
        tracker = OMRNEDTracker()

        # First epoch
        tracker.update(SIMPLE_KERN_2, SIMPLE_KERN_1, "synth")
        metrics1 = tracker.compute()
        assert metrics1.overall_score > 0.0

        tracker.reset()

        # Second epoch - identical samples
        tracker.update(SIMPLE_KERN_IDENTICAL, SIMPLE_KERN_IDENTICAL, "synth")
        metrics2 = tracker.compute()
        assert metrics2.overall_score == pytest.approx(0.0, abs=0.1)

    def test_empty_strings_count_as_failures(self):
        """Test empty strings are handled gracefully as failures."""
        tracker = OMRNEDTracker()

        tracker.update("", SIMPLE_KERN_1, "synth")
        tracker.update(SIMPLE_KERN_1, "", "polish")

        metrics = tracker.compute()

        assert metrics.overall_failures == 2
        assert metrics.overall_score == 100.0
        assert metrics.by_source_score["synth"] == 100.0
        assert metrics.by_source_score["polish"] == 100.0
