from __future__ import annotations

from lightning.pytorch import Callback

from transcoda.validation.schedule import FitValidationWarmupSchedule


class FitValidationWarmupCallback(Callback):
    """Suppress scheduled fit-time validation before a global-step threshold."""

    def __init__(self, min_validation_step: int) -> None:
        super().__init__()
        self.schedule = FitValidationWarmupSchedule(min_step=min_validation_step)
        self.min_validation_step = max(0, int(min_validation_step))

    def on_train_start(self, trainer, pl_module) -> None:
        if self.min_validation_step <= 0:
            return
        if bool(getattr(trainer, "is_global_zero", True)):
            print(
                "[val] fit-time validation warmup: skipping scheduled validation "
                f"until global_step >= {self.min_validation_step}",
                flush=True,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del pl_module, outputs, batch, batch_idx
        global_step = int(getattr(trainer, "global_step", 0) or 0)
        if not self.schedule.should_skip(global_step=global_step, validation_kind="fit"):
            return

        epoch_loop = getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None)
        if epoch_loop is None or not hasattr(epoch_loop, "_skip_next_val"):
            return

        # Lightning has no public "start validating at global_step=N" switch.
        epoch_loop._skip_next_val = True
