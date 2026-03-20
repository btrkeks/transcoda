import time

from lightning.pytorch import Callback


class EpochTimingCallback(Callback):
    """
    Callback to track detailed timing metrics for training and validation epochs.

    This callback logs:
    - time/train_epoch_s: Total wall-clock time for the training epoch
    - time/train_batch_compute_s: Aggregate time spent in forward/backward/optimizer steps
    - time/train_batch_wait_s: Aggregate time between batches (dataloader + callbacks + logging)
    - time/val_epoch_s: Total wall-clock time for the validation epoch
    - time/val_postproc_s: Time spent in CPU post-processing during validation
    """

    def __init__(self):
        super().__init__()
        self._t0 = None
        self._v0 = None
        self._last_batch_end = None
        self._batch_compute_sum = 0.0
        self._batch_wait_sum = 0.0
        self._val_postproc_sum = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()
        # for batch-level breakdown
        self._last_batch_end = self._t0
        self._batch_compute_sum = 0.0
        self._batch_wait_sum = 0.0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        # Skip wait time calculation if we haven't started tracking yet (e.g., checkpoint resumption)
        if self._last_batch_end is not None:
            # time since last batch end ≈ dataloader wait + callbacks/logging outside the step
            self._batch_wait_sum += max(0.0, now - self._last_batch_end)
        self._tb0 = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        # duration of the step (forward+backward+opt+hooks within the step)
        self._batch_compute_sum += max(0.0, now - self._tb0)
        self._last_batch_end = now

    def on_train_epoch_end(self, trainer, pl_module):
        # Skip if epoch start was never called (e.g., when resuming from checkpoint)
        if self._t0 is None:
            return

        train_epoch_s = time.perf_counter() - self._t0
        trainer.logger.log_metrics(
            {
                "time/train_epoch_s": train_epoch_s,
                "time/train_batch_compute_s": self._batch_compute_sum,
                "time/train_batch_wait_s": self._batch_wait_sum,
            },
            step=trainer.global_step,
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        self._v0 = time.perf_counter()
        self._val_postproc_sum = 0.0  # pl_module can add to this

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if validation epoch start was never called
        if self._v0 is None:
            return

        val_epoch_s = time.perf_counter() - self._v0
        trainer.logger.log_metrics(
            {
                "time/val_epoch_s": val_epoch_s,
                "time/val_postproc_s": self._val_postproc_sum,
            },
            step=trainer.global_step,
        )
