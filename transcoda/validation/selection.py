"""Checkpoint selection policies for expensive validation metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_OMR_NED_DECISION_METRIC = "val/selection/OMR_NED_checkpoint_eligible"


@dataclass(frozen=True)
class SelectionDecision:
    """A checkpoint eligibility decision plus human-readable blocking reasons."""

    eligible: bool
    reasons: tuple[str, ...] = ()
    observed_metrics: dict[str, float] = field(default_factory=dict)


class CheckpointSelectionPolicy(Protocol):
    """Decide whether a validation pass may update a checkpoint stream."""

    primary_metric: str

    def decide(
        self,
        metrics: Mapping[str, Any],
        *,
        global_step: int,
    ) -> SelectionDecision:
        ...

    def is_eligible(
        self,
        metrics: Mapping[str, Any],
        *,
        global_step: int,
    ) -> bool:
        ...


@dataclass(frozen=True)
class OMRNEDCheckpointSelectionPolicy:
    """Gate OMR-NED checkpointing until visual metrics are reliable enough."""

    primary_metric: str = "val/real/OMR_NED"
    min_step: int = 20_000
    min_samples: int = 1
    max_verosim_failure_rate: float | None = 0.10
    max_prediction_truncation_rate: float | None = 0.02
    max_runaway_rate: float | None = 0.02
    step_metric: str | None = None
    failure_rate_metric: str | None = None
    sample_count_metric: str | None = None
    truncation_rate_metric: str | None = None
    runaway_rate_metric: str | None = None
    runaway_rate_metric_is_percent: bool = True

    def decide(
        self,
        metrics: Mapping[str, Any],
        *,
        global_step: int,
    ) -> SelectionDecision:
        reasons: list[str] = []
        observed: dict[str, float] = {}

        if global_step < self.min_step:
            reasons.append(f"global_step {global_step} < min_step {self.min_step}")

        primary = _metric_value(metrics, self.primary_metric)
        if primary is None:
            reasons.append(f"missing {self.primary_metric}")
        else:
            observed[self.primary_metric] = primary

        metric_step = _metric_value(metrics, self._step_metric)
        if metric_step is None:
            reasons.append(f"missing {self._step_metric}")
        else:
            observed[self._step_metric] = metric_step
            if int(metric_step) != global_step:
                reasons.append(f"{self._step_metric} {int(metric_step)} != global_step {global_step}")

        sample_count = _metric_value(metrics, self._sample_count_metric)
        if sample_count is None:
            reasons.append(f"missing {self._sample_count_metric}")
        else:
            observed[self._sample_count_metric] = sample_count
            if sample_count < self.min_samples:
                reasons.append(
                    f"{self._sample_count_metric} {sample_count:g} < min_samples {self.min_samples}"
                )

        self._check_fraction_gate(
            metrics=metrics,
            metric_name=self._failure_rate_metric,
            max_value=self.max_verosim_failure_rate,
            reasons=reasons,
            observed=observed,
        )
        self._check_fraction_gate(
            metrics=metrics,
            metric_name=self._truncation_rate_metric,
            max_value=self.max_prediction_truncation_rate,
            reasons=reasons,
            observed=observed,
        )
        runaway = _metric_value(metrics, self._runaway_rate_metric)
        if self.max_runaway_rate is not None:
            if runaway is None:
                reasons.append(f"missing {self._runaway_rate_metric}")
            else:
                observed[self._runaway_rate_metric] = runaway
                comparable = runaway / 100.0 if self.runaway_rate_metric_is_percent else runaway
                if comparable > self.max_runaway_rate:
                    reasons.append(
                        f"{self._runaway_rate_metric} {comparable:g} > max {self.max_runaway_rate:g}"
                    )
        elif runaway is not None:
            observed[self._runaway_rate_metric] = runaway

        return SelectionDecision(eligible=not reasons, reasons=tuple(reasons), observed_metrics=observed)

    def is_eligible(
        self,
        metrics: Mapping[str, Any],
        *,
        global_step: int,
    ) -> bool:
        return self.decide(metrics, global_step=global_step).eligible

    @property
    def _failure_rate_metric(self) -> str:
        return self.failure_rate_metric or sibling_metric_name(
            self.primary_metric, "OMR_NED_failure_rate"
        )

    @property
    def _step_metric(self) -> str:
        return self.step_metric or sibling_metric_name(self.primary_metric, "OMR_NED_step")

    @property
    def _sample_count_metric(self) -> str:
        return self.sample_count_metric or sibling_metric_name(self.primary_metric, "OMR_NED_samples")

    @property
    def _truncation_rate_metric(self) -> str:
        return self.truncation_rate_metric or sibling_metric_name(
            self.primary_metric, "prediction_truncation_rate"
        )

    @property
    def _runaway_rate_metric(self) -> str:
        return self.runaway_rate_metric or sibling_metric_name(self.primary_metric, "runaway_rate")

    @staticmethod
    def _check_fraction_gate(
        *,
        metrics: Mapping[str, Any],
        metric_name: str,
        max_value: float | None,
        reasons: list[str],
        observed: dict[str, float],
    ) -> None:
        value = _metric_value(metrics, metric_name)
        if max_value is None:
            if value is not None:
                observed[metric_name] = value
            return
        if value is None:
            reasons.append(f"missing {metric_name}")
            return
        observed[metric_name] = value
        if value > max_value:
            reasons.append(f"{metric_name} {value:g} > max {max_value:g}")


def sibling_metric_name(metric_name: str, sibling: str) -> str:
    """Return ``sibling`` in the same metric namespace as ``metric_name``."""
    prefix, separator, _leaf = metric_name.rpartition("/")
    if not separator:
        return sibling
    return f"{prefix}/{sibling}"


def selection_decision_payload(
    decision: SelectionDecision,
    *,
    metric_name: str = DEFAULT_OMR_NED_DECISION_METRIC,
) -> dict[str, float]:
    """Convert a selection decision to scalar metrics suitable for logging."""
    return {metric_name: 1.0 if decision.eligible else 0.0}


def _metric_value(metrics: Mapping[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None

    value = metrics[key]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()

    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None
