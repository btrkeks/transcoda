from __future__ import annotations

import torch
from lightning.pytorch import Callback


class CompactLearningRateMonitor(Callback):
    """Log one representative optimizer learning rate per optimizer step."""

    def __init__(self, metric_name: str = "train/lr") -> None:
        super().__init__()
        self.metric_name = metric_name
        self._last_logged_step: int | None = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del pl_module, outputs, batch, batch_idx
        if not getattr(trainer, "is_global_zero", True):
            return
        if trainer.logger is None:
            return

        step = int(getattr(trainer, "global_step", 0) or 0)
        if step <= 0 or step == self._last_logged_step:
            return

        optimizer = self._first_optimizer(trainer)
        if optimizer is None:
            return

        lrs = [float(group["lr"]) for group in optimizer.param_groups if "lr" in group]
        if not lrs:
            return

        value = max(lrs)
        trainer.logger.log_metrics({self.metric_name: value}, step=step)
        trainer.callback_metrics[self.metric_name] = torch.tensor(
            value,
            device=trainer.strategy.root_device,
        )
        self._last_logged_step = step

    @staticmethod
    def _first_optimizer(trainer):
        optimizers = getattr(trainer, "optimizers", None) or []
        if not optimizers:
            return None
        optimizer = optimizers[0]
        return optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
