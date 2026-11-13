import time

import torch
from lightning.pytorch import Callback


class EpochTimingCallback(Callback):
    """
    Callback to track detailed timing metrics for training and validation epochs.

    This callback logs:
    - time/train_epoch_s: Total wall-clock time for the training epoch
    - time/train_batch_compute_s: Aggregate time spent in forward/backward/optimizer steps
    - time/train_batch_wait_s: Aggregate time between batches (dataloader + callbacks + logging)
    - time/train_interrupt_s: Mid-epoch validation/checkpoint interruption time
    - time/train_wait_excl_interrupt_s: Wait time excluding mid-epoch interruptions
    - time/train_fetch_s: Train DataLoader iterator fetch time, when available
    - time/train_step_gpu_s: CUDA stream elapsed step time, when CUDA is available
    - time/val_epoch_s: Total wall-clock time for the validation epoch
    """

    def __init__(self):
        super().__init__()
        self._t0 = None
        self._v0 = None
        self._tb0 = None
        self._last_batch_end = None
        self._legacy_last_batch_end = None
        self._batch_compute_sum = 0.0
        self._batch_wait_sum = 0.0
        self._legacy_batch_wait_sum = 0.0
        self._train_epoch_active = False
        self._interrupt_start = None
        self._train_interrupt_sum = 0.0
        self._gpu_timing_enabled = False
        self._gpu_batch_start_event = None
        self._pending_gpu_event_pairs = []
        self._gpu_step_sum = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()
        # for batch-level breakdown
        self._last_batch_end = self._t0
        self._legacy_last_batch_end = self._t0
        self._tb0 = None
        self._batch_compute_sum = 0.0
        self._batch_wait_sum = 0.0
        self._legacy_batch_wait_sum = 0.0
        self._train_epoch_active = True
        self._interrupt_start = None
        self._train_interrupt_sum = 0.0
        self._reset_train_fetch_sink(trainer)
        self._reset_gpu_timing(pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        # Skip wait time calculation if we haven't started tracking yet (e.g., checkpoint resumption)
        if self._last_batch_end is not None:
            # time since last batch end ≈ dataloader wait + callbacks/logging outside the step
            self._batch_wait_sum += max(0.0, now - self._last_batch_end)
        if self._legacy_last_batch_end is not None:
            self._legacy_batch_wait_sum += max(0.0, now - self._legacy_last_batch_end)
        self._tb0 = now
        self._record_gpu_batch_start()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        # duration of the step (forward+backward+opt+hooks within the step)
        if self._tb0 is not None:
            self._batch_compute_sum += max(0.0, now - self._tb0)
        self._last_batch_end = now
        self._legacy_last_batch_end = now
        self._record_gpu_batch_end()
        self._drain_completed_gpu_events()

    def on_train_epoch_end(self, trainer, pl_module):
        # Skip if epoch start was never called (e.g., when resuming from checkpoint)
        if self._t0 is None:
            return

        self._finish_gpu_timing()
        train_epoch_s = time.perf_counter() - self._t0
        metrics = {
            "time/train_epoch_s": train_epoch_s,
            "time/train_batch_compute_s": self._batch_compute_sum,
            "time/train_batch_wait_s": self._legacy_batch_wait_sum,
            "time/train_interrupt_s": self._train_interrupt_sum,
            "time/train_wait_excl_interrupt_s": self._batch_wait_sum,
        }
        fetch_s = self._train_fetch_s(trainer)
        if fetch_s is not None:
            metrics["time/train_fetch_s"] = fetch_s
        if self._gpu_timing_enabled:
            metrics["time/train_step_gpu_s"] = self._gpu_step_sum

        trainer.logger.log_metrics(metrics, step=trainer.global_step)
        self._train_epoch_active = False
        self._interrupt_start = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self._v0 = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if validation epoch start was never called
        if self._v0 is None:
            return

        val_epoch_s = time.perf_counter() - self._v0
        trainer.logger.log_metrics(
            {"time/val_epoch_s": val_epoch_s},
            step=trainer.global_step,
        )

    def on_validation_start(self, trainer, pl_module):
        if not self._should_track_train_interrupt(trainer):
            return
        if self._interrupt_start is not None:
            return

        now = time.perf_counter()
        if self._last_batch_end is not None:
            self._batch_wait_sum += max(0.0, now - self._last_batch_end)
            self._last_batch_end = None
        self._interrupt_start = now

    def on_validation_end(self, trainer, pl_module):
        if self._interrupt_start is None:
            return

        now = time.perf_counter()
        self._train_interrupt_sum += max(0.0, now - self._interrupt_start)
        self._interrupt_start = None
        if self._train_epoch_active:
            self._last_batch_end = now

    @staticmethod
    def _normalized_state_value(value) -> str | None:
        if value is None:
            return None
        raw = getattr(value, "value", value)
        normalized = str(raw).strip().lower()
        if "." in normalized:
            normalized = normalized.split(".")[-1]
        return normalized

    def _should_track_train_interrupt(self, trainer) -> bool:
        if not self._train_epoch_active:
            return False
        if getattr(trainer, "sanity_checking", False):
            return False

        state = getattr(trainer, "state", None)
        fn = self._normalized_state_value(getattr(state, "fn", None))
        if fn in {"fit", "fitting"}:
            return True

        return bool(getattr(trainer, "training", False) or getattr(trainer, "fitting", False))

    @staticmethod
    def _train_fetch_sink(trainer):
        datamodule = getattr(trainer, "datamodule", None)
        return getattr(datamodule, "train_fetch_sink", None)

    def _reset_train_fetch_sink(self, trainer) -> None:
        sink = self._train_fetch_sink(trainer)
        reset = getattr(sink, "reset", None)
        if callable(reset):
            reset()

    def _train_fetch_s(self, trainer) -> float | None:
        sink = self._train_fetch_sink(trainer)
        total_s = getattr(sink, "total_s", None)
        if isinstance(total_s, (int, float)):
            return float(total_s)
        return None

    @staticmethod
    def _module_is_on_cuda(pl_module) -> bool:
        device = getattr(pl_module, "device", None)
        if getattr(device, "type", None) == "cuda":
            return True

        parameters = getattr(pl_module, "parameters", None)
        if not callable(parameters):
            return False
        try:
            first_parameter = next(parameters())
        except StopIteration:
            return False
        return bool(getattr(first_parameter, "is_cuda", False))

    def _reset_gpu_timing(self, pl_module) -> None:
        self._gpu_timing_enabled = bool(
            torch.cuda.is_available() and self._module_is_on_cuda(pl_module)
        )
        self._gpu_batch_start_event = None
        self._pending_gpu_event_pairs = []
        self._gpu_step_sum = 0.0

    def _record_gpu_batch_start(self) -> None:
        if not self._gpu_timing_enabled:
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._gpu_batch_start_event = start

    def _record_gpu_batch_end(self) -> None:
        if not self._gpu_timing_enabled or self._gpu_batch_start_event is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending_gpu_event_pairs.append((self._gpu_batch_start_event, end))
        self._gpu_batch_start_event = None

    def _drain_completed_gpu_events(self) -> None:
        if not self._gpu_timing_enabled:
            return
        while self._pending_gpu_event_pairs and self._pending_gpu_event_pairs[0][1].query():
            start, end = self._pending_gpu_event_pairs.pop(0)
            self._gpu_step_sum += max(0.0, start.elapsed_time(end) / 1000.0)

    def _finish_gpu_timing(self) -> None:
        if not self._gpu_timing_enabled or not self._pending_gpu_event_pairs:
            return
        torch.cuda.synchronize()
        self._drain_completed_gpu_events()
