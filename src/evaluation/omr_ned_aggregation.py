"""Shared OMR-NED aggregation helpers.

This module centralizes the repo-wide policy for turning low-level OMR-NED
results into reportable scores. Parse or comparison failures count as the
worst possible score (100.0) and are tracked separately.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .omr_ned import OMRNEDResult

OMR_NED_FAILURE_SCORE = 100.0


@dataclass(frozen=True)
class OMRNEDAggregateStats:
    """Aggregate OMR-NED stats for one cohort of samples."""

    score: float
    failures: int
    samples: int


@dataclass(frozen=True)
class OMRNEDAggregateSummary:
    """Aggregate OMR-NED stats for all samples and per-source subsets."""

    overall: OMRNEDAggregateStats
    by_source: dict[str, OMRNEDAggregateStats]


def resolve_omr_ned_score(result: OMRNEDResult) -> tuple[float, bool]:
    """Convert a low-level OMR-NED result to a reportable score."""
    if result.omr_ned is None:
        return OMR_NED_FAILURE_SCORE, True
    return result.omr_ned, False


class OMRNEDAggregator:
    """Accumulate OMR-NED scores under the repo's shared failure policy."""

    def __init__(self) -> None:
        self._scores_by_source: dict[str, list[float]] = defaultdict(list)
        self._failures_by_source: dict[str, int] = defaultdict(int)

    def add_result(self, result: OMRNEDResult, source: str) -> None:
        score, failed = resolve_omr_ned_score(result)
        self.add_score(score, source=source, failed=failed)

    def add_score(self, score: float, *, source: str, failed: bool = False) -> None:
        self._scores_by_source[source].append(score)
        if failed:
            self._failures_by_source[source] += 1

    def compute(self) -> OMRNEDAggregateSummary:
        by_source: dict[str, OMRNEDAggregateStats] = {}
        all_scores: list[float] = []
        all_failures = 0

        for source, scores in self._scores_by_source.items():
            failures = self._failures_by_source.get(source, 0)
            all_scores.extend(scores)
            all_failures += failures
            by_source[source] = OMRNEDAggregateStats(
                score=sum(scores) / len(scores) if scores else 0.0,
                failures=failures,
                samples=len(scores),
            )

        overall = OMRNEDAggregateStats(
            score=sum(all_scores) / len(all_scores) if all_scores else 0.0,
            failures=all_failures,
            samples=len(all_scores),
        )
        return OMRNEDAggregateSummary(overall=overall, by_source=by_source)

    def reset(self) -> None:
        self._scores_by_source.clear()
        self._failures_by_source.clear()
