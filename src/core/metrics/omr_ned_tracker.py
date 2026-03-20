"""OMR-NED tracker for validation loop integration."""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.omr_ned import compute_omr_ned, is_musicdiff_available
from src.evaluation.omr_ned_aggregation import OMRNEDAggregator, OMRNEDAggregateSummary


@dataclass(frozen=True)
class OMRNEDTrackerResult:
    """Structured OMR-NED stats for validation logging."""

    overall_score: float
    overall_failures: int
    by_source_score: dict[str, float]
    by_source_failures: dict[str, int]


class OMRNEDTracker:
    """Accumulates (pred_str, target_str, source) tuples and computes OMR-NED at epoch end.

    This tracker is designed for integration with PyTorch Lightning's validation loop.
    It accumulates prediction/target string pairs during validation_step() and computes
    OMR-NED metrics at the end of the epoch.

    Unlike TorchMetrics, this class:
    - Works with string inputs (not tensors)
    - Handles the slow musicdiff computation (~100-200ms per sample)
    - Tracks parse failures separately from valid samples
    - Provides per-source breakdown (synth, polish, custom)

    Example:
        tracker = OMRNEDTracker()

        # In validation_step:
        tracker.update(pred_str, target_str, source="polish")

        # In on_validation_epoch_end:
        metrics = tracker.compute()
        # metrics.overall_score == 15.3
        # metrics.by_source_score["polish"] == 18.1
        tracker.reset()
    """

    def __init__(self) -> None:
        """Initialize the tracker."""
        self._samples: list[tuple[str, str, str]] = []  # (pred, target, source)
        self._enabled = is_musicdiff_available()

    @property
    def enabled(self) -> bool:
        """Return True if musicdiff is available and tracker is functional."""
        return self._enabled

    def update(self, pred_str: str, target_str: str, source: str = "unknown") -> None:
        """Accumulate a sample for later computation.

        Args:
            pred_str: Predicted **kern string
            target_str: Ground truth **kern string
            source: Data source identifier (e.g., "synth", "polish", "custom")
        """
        if self._enabled:
            self._samples.append((pred_str, target_str, source))

    def compute(self) -> OMRNEDTrackerResult:
        """Compute OMR-NED statistics over accumulated samples.

        Returns:
            Structured OMR-NED metrics for overall and per-source logging.

        Note:
            Parse failures count as 100.0 in the aggregate score and are
            also counted separately.
        """
        if not self._enabled or not self._samples:
            return OMRNEDTrackerResult(
                overall_score=0.0,
                overall_failures=0,
                by_source_score={},
                by_source_failures={},
            )

        aggregator = OMRNEDAggregator()

        for pred_str, target_str, source in self._samples:
            result = compute_omr_ned(pred_str, target_str)
            aggregator.add_result(result, source)

        summary = aggregator.compute()
        return _tracker_result_from_summary(summary)

    def reset(self) -> None:
        """Clear accumulated samples for next epoch."""
        self._samples.clear()

    def __len__(self) -> int:
        """Return number of accumulated samples."""
        return len(self._samples)


def _tracker_result_from_summary(summary: OMRNEDAggregateSummary) -> OMRNEDTrackerResult:
    return OMRNEDTrackerResult(
        overall_score=summary.overall.score,
        overall_failures=summary.overall.failures,
        by_source_score={source: stats.score for source, stats in summary.by_source.items()},
        by_source_failures={source: stats.failures for source, stats in summary.by_source.items()},
    )
