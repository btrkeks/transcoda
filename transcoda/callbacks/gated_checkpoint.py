"""Model checkpoint callback guarded by an explicit selection policy."""

from __future__ import annotations

from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from transcoda.validation.selection import (
    DEFAULT_OMR_NED_DECISION_METRIC,
    CheckpointSelectionPolicy,
    SelectionDecision,
    selection_decision_payload,
)


class GatedModelCheckpoint(ModelCheckpoint):
    """Save a checkpoint only when a selection policy marks validation eligible."""

    def __init__(
        self,
        *,
        selection_policy: CheckpointSelectionPolicy,
        decision_metric_name: str = DEFAULT_OMR_NED_DECISION_METRIC,
        **checkpoint_kwargs: Any,
    ) -> None:
        super().__init__(**checkpoint_kwargs)
        self.selection_policy = selection_policy
        self.decision_metric_name = decision_metric_name
        self.last_decision: SelectionDecision | None = None
        self.current_validation_decision: SelectionDecision | None = None

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.current_validation_decision = None

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = getattr(trainer, "callback_metrics", {})
        decision = self.selection_policy.decide(
            metrics,
            global_step=int(getattr(trainer, "global_step", 0) or 0),
        )
        self.last_decision = decision
        self.current_validation_decision = decision
        if self.verbose:
            _log_selection_decision(decision, global_step=int(getattr(trainer, "global_step", 0) or 0))
        _publish_decision_metrics(
            pl_module,
            selection_decision_payload(decision, metric_name=self.decision_metric_name),
        )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        decision = self.current_validation_decision
        try:
            if decision is None or not decision.eligible:
                return
            super().on_validation_end(trainer, pl_module)
        finally:
            self.current_validation_decision = None

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return


def _publish_decision_metrics(
    pl_module: LightningModule,
    payload: dict[str, float],
) -> None:
    for key, value in payload.items():
        pl_module.log(
            key,
            float(value),
            on_epoch=True,
            prog_bar=key == DEFAULT_OMR_NED_DECISION_METRIC,
            logger=True,
            sync_dist=False,
        )


def _log_selection_decision(decision: SelectionDecision, *, global_step: int) -> None:
    if decision.eligible:
        rank_zero_info(f"OMR-NED checkpoint eligible at step {global_step}.")
        return

    reason_text = "; ".join(decision.reasons) if decision.reasons else "unknown reason"
    rank_zero_info(f"OMR-NED checkpoint not eligible at step {global_step}: {reason_text}.")
