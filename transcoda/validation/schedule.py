"""Schedules for expensive validation metric backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ValidationMetricSchedule(Protocol):
    """Decide whether an expensive metric should run for a validation pass."""

    def should_run(self, *, global_step: int, validation_kind: str) -> bool:
        ...


@dataclass(frozen=True)
class NeverValidationMetricSchedule:
    """Schedule that never runs."""

    def should_run(self, *, global_step: int, validation_kind: str) -> bool:
        return False


@dataclass(frozen=True)
class FitValidationWarmupSchedule:
    """Skip fit-time validation before an optimizer-step threshold."""

    min_step: int = 0

    def should_skip(self, *, global_step: int, validation_kind: str) -> bool:
        if validation_kind != "fit":
            return False
        min_step = max(0, int(self.min_step))
        if min_step <= 0:
            return False
        return int(global_step) < min_step


@dataclass(frozen=True)
class FullValidationEveryNStepsSchedule:
    """Run on the first fit-time validation at or after each step boundary."""

    every_n_steps: int

    def boundary_step(self, *, global_step: int, validation_kind: str) -> int | None:
        if validation_kind != "fit":
            return None
        step = int(global_step)
        if step <= 0:
            return None
        every_n_steps = max(1, int(self.every_n_steps))
        boundary = (step // every_n_steps) * every_n_steps
        return boundary if boundary > 0 else None

    def should_run(self, *, global_step: int, validation_kind: str) -> bool:
        return self.boundary_step(
            global_step=global_step,
            validation_kind=validation_kind,
        ) is not None
