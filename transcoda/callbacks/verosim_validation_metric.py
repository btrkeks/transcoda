"""Lightning callback for scheduled VeroSim OMR-NED validation scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer

from transcoda.callbacks.validation_prediction_writer import predictions_output_dir
from transcoda.metrics_schema import VAL_PREFIX
from transcoda.validation.metrics.verosim import (
    VeroSimOMRNEDBackend,
    VeroSimOMRNEDResult,
    metric_payload_from_result,
)
from transcoda.validation.schedule import ValidationMetricSchedule
from transcoda.validation.selection import sibling_metric_name


class VeroSimValidationMetricCallback(Callback):
    """Record predictions and run VeroSim on scheduled validation passes."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        backend: VeroSimOMRNEDBackend,
        schedule: ValidationMetricSchedule,
    ) -> None:
        super().__init__()
        self.output_root = Path(output_root)
        self.backend = backend
        self.schedule = schedule
        self._run_this_validation = False
        self._previous_recording_state: bool | None = None
        self._pending_schedule_boundary: int | None = None
        self._last_completed_schedule_boundary = 0
        self.last_result: VeroSimOMRNEDResult | None = None

    def state_dict(self) -> dict[str, int]:
        return {
            "last_completed_schedule_boundary": int(self._last_completed_schedule_boundary),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        value = state_dict.get("last_completed_schedule_boundary", 0)
        self._last_completed_schedule_boundary = int(value or 0)
        self._pending_schedule_boundary = None

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._run_this_validation = self._should_run(trainer, pl_module)
        self._previous_recording_state = None
        if not self._run_this_validation:
            return

        self._previous_recording_state = _module_is_recording(pl_module)
        setter = getattr(pl_module, "set_validation_prediction_recording", None)
        if callable(setter):
            setter(True)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._run_this_validation:
            return
        pending_boundary = self._pending_schedule_boundary
        completed_boundary: int | None = None
        _strategy_barrier(trainer, "verosim_validation_metric_before_score")
        payload: dict[str, float] | None = None
        try:
            is_rank_zero = int(getattr(trainer, "global_rank", 0) or 0) == 0
            if is_rank_zero:
                metric_prefix = str(getattr(pl_module, "validation_metric_prefix", VAL_PREFIX))
                predictions_dir = predictions_output_dir(self.output_root, metric_prefix)
                result = self.backend.run(predictions_dir=predictions_dir)
                self.last_result = result
                if not result.skipped and result.summary is not None:
                    payload = metric_payload_from_result(result)
                    payload.update(
                        _omr_ned_step_payload(
                            payload,
                            global_step=int(getattr(trainer, "global_step", 0) or 0),
                        )
                    )
                elif result.skipped:
                    payload = _omr_ned_skip_payload(
                        global_step=int(getattr(trainer, "global_step", 0) or 0),
                    )
                    print(f"[verosim] skipped OMR_NED: {result.reason}", flush=True)

            payload = _strategy_broadcast(trainer, payload, src=0)
            if payload:
                _publish_callback_metrics(pl_module, payload)
                if not _payload_marks_omr_ned_skip(payload):
                    completed_boundary = pending_boundary
        finally:
            _strategy_barrier(trainer, "verosim_validation_metric_after_score")
        if completed_boundary is not None:
            self._last_completed_schedule_boundary = max(
                self._last_completed_schedule_boundary,
                completed_boundary,
            )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._run_this_validation and self._previous_recording_state is not None:
            setter = getattr(pl_module, "set_validation_prediction_recording", None)
            if callable(setter):
                setter(self._previous_recording_state)
        self._run_this_validation = False
        self._previous_recording_state = None
        self._pending_schedule_boundary = None

    def _should_run(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        if bool(getattr(trainer, "sanity_checking", False)):
            return False
        if not _trainer_is_fitting(trainer):
            return False
        metric_prefix = str(getattr(pl_module, "validation_metric_prefix", VAL_PREFIX))
        validation_kind = "fit" if metric_prefix == VAL_PREFIX else metric_prefix
        boundary = _schedule_boundary_step(
            self.schedule,
            global_step=int(getattr(trainer, "global_step", 0) or 0),
            validation_kind=validation_kind,
        )
        if boundary is not None:
            self._pending_schedule_boundary = boundary
            return boundary > self._last_completed_schedule_boundary
        self._pending_schedule_boundary = None
        return self.schedule.should_run(
            global_step=int(getattr(trainer, "global_step", 0) or 0),
            validation_kind=validation_kind,
        )


def _module_is_recording(pl_module: LightningModule) -> bool:
    checker = getattr(pl_module, "is_recording_validation_predictions", None)
    return bool(checker()) if callable(checker) else False


def _strategy_barrier(trainer: Trainer, name: str) -> None:
    barrier = getattr(getattr(trainer, "strategy", None), "barrier", None)
    if not callable(barrier):
        return
    try:
        barrier(name)
    except TypeError:
        barrier()


def _strategy_broadcast(
    trainer: Trainer,
    value: dict[str, float] | None,
    *,
    src: int,
) -> dict[str, float] | None:
    broadcast = getattr(getattr(trainer, "strategy", None), "broadcast", None)
    if not callable(broadcast):
        return value
    try:
        return broadcast(value, src=src)
    except TypeError:
        return broadcast(value)


def _trainer_is_fitting(trainer: Trainer) -> bool:
    state_fn = getattr(getattr(trainer, "state", None), "fn", None)
    if state_fn is not None:
        state_value = getattr(state_fn, "value", state_fn)
        state_text = str(state_value).lower()
        return state_text in {"fit", "fitting"} or state_text.endswith((".fit", ".fitting"))
    training = getattr(trainer, "training", None)
    if training is not None:
        return bool(training)
    return True


def _schedule_boundary_step(
    schedule: ValidationMetricSchedule,
    *,
    global_step: int,
    validation_kind: str,
) -> int | None:
    boundary_fn = getattr(schedule, "boundary_step", None)
    if not callable(boundary_fn):
        return None
    boundary = boundary_fn(global_step=global_step, validation_kind=validation_kind)
    return None if boundary is None else int(boundary)


def _publish_callback_metrics(
    pl_module: LightningModule,
    payload: dict[str, float],
) -> None:
    for key, value in payload.items():
        pl_module.log(
            key,
            float(value),
            on_epoch=True,
            prog_bar=key.endswith("/OMR_NED"),
            logger=True,
            sync_dist=False,
        )


def _omr_ned_step_payload(payload: dict[str, float], *, global_step: int) -> dict[str, float]:
    return {
        sibling_metric_name(key, "OMR_NED_step"): float(global_step)
        for key in payload
        if key.endswith("/OMR_NED")
    }


def _omr_ned_skip_payload(*, global_step: int) -> dict[str, float]:
    return {
        "val/aggregate/OMR_NED_backend_skipped": 1.0,
        "val/aggregate/OMR_NED_backend_step": float(global_step),
    }


def _payload_marks_omr_ned_skip(payload: dict[str, float]) -> bool:
    return float(payload.get("val/aggregate/OMR_NED_backend_skipped", 0.0)) > 0.0
