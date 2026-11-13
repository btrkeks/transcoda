"""Shared validation metric backend interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class MetricResult:
    """Backend-neutral validation metric output."""

    name: str
    metrics: dict[str, float]
    per_group_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)
    sample_counts: dict[str, int] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    raw_summary: dict[str, Any] | None = None


class ValidationMetricBackend(Protocol):
    """A backend that computes validation metrics from recorded predictions."""

    name: str

    def compute(self, predictions_dir: str | Path) -> MetricResult:
        ...
