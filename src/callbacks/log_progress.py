import time
from typing import Any

import torch
from lightning.pytorch import Callback


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_bar(progress: float, width: int = 20) -> str:
    clamped = min(1.0, max(0.0, progress))
    filled = int(clamped * width)
    return "#" * filled + "-" * (width - filled)


class LogProgressCallback(Callback):
    """
    Emit plain-text progress updates suitable for non-interactive Slurm logs.

    Lightning's interactive progress bars are typically hidden or hard to read in batch logs.
    This callback prints periodic, newline-delimited progress lines for train/val loops.
    """

    def __init__(
        self,
        every_n_steps: int | None = None,
        train_every_n_steps: int | None = None,
        train_interval_seconds: float = 30.0,
        val_percent_interval: int = 10,
        enable_ascii_bar: bool = False,
    ):
        super().__init__()
        # Backward-compatible alias: previous API used every_n_steps for both train/val.
        if train_every_n_steps is None:
            train_every_n_steps = every_n_steps if every_n_steps is not None else 10

        self.train_every_n_steps = max(1, int(train_every_n_steps))
        self.train_interval_seconds = max(0.1, float(train_interval_seconds))
        self.val_percent_interval = max(1, min(50, int(val_percent_interval)))
        self.enable_ascii_bar = bool(enable_ascii_bar)

        self._train_t0: float | None = None
        self._train_epoch_t0: float | None = None
        self._train_last_emit_t: float | None = None
        self._train_epoch_seen_batches: int = 0
        self._train_epoch_seen_steps: int = 0
        self._train_last_seen_global_step: int = 0
        self._train_total_batches: int | None = None
        self._val_t0: float | None = None
        self._val_total_batches: int | None = None
        self._val_seen_batches: int = 0
        self._val_last_logged_pct: int = 0

    def _should_train_log(self, step: int, total_steps: int | None, step_advanced: bool, now: float) -> bool:
        if step <= 0:
            return False
        if step_advanced and (step % self.train_every_n_steps == 0):
            return True
        if total_steps is not None and step_advanced and step >= total_steps:
            return True
        if self._train_last_emit_t is None:
            return True
        return (now - self._train_last_emit_t) >= self.train_interval_seconds

    def _extract_loss(self, outputs: Any) -> float | None:
        if outputs is None:
            return None
        if torch.is_tensor(outputs):
            return float(outputs.detach().item())
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
            if torch.is_tensor(loss):
                return float(loss.detach().item())
        return None

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        now = time.perf_counter()
        self._train_t0 = now
        self._train_last_emit_t = now
        total_steps = getattr(trainer, "estimated_stepping_batches", None)
        max_epochs = getattr(trainer, "max_epochs", None)
        if isinstance(total_steps, int) and total_steps > 0:
            if isinstance(max_epochs, int) and max_epochs > 0:
                print(f"[train] started total_steps={total_steps} max_epochs={max_epochs}", flush=True)
            else:
                print(f"[train] started total_steps={total_steps}", flush=True)
        else:
            print("[train] started", flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._train_epoch_t0 = time.perf_counter()
        self._train_epoch_seen_batches = 0
        self._train_epoch_seen_steps = 0
        self._train_last_seen_global_step = int(trainer.global_step)
        self._train_total_batches = self._resolve_train_total_batches(trainer)

    def _resolve_train_total_batches(self, trainer) -> int | None:
        total_batches = getattr(trainer, "num_training_batches", None)
        if isinstance(total_batches, int) and total_batches > 0:
            return total_batches
        return None

    def _epoch_info(self, trainer) -> tuple[str, str]:
        current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        max_epochs = getattr(trainer, "max_epochs", None)
        if isinstance(max_epochs, int) and max_epochs > 0:
            return f"{current_epoch}/{max_epochs}", str(max_epochs)
        return f"{current_epoch}/?", "?"

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        now = time.perf_counter()
        self._train_epoch_seen_batches += 1

        step = int(trainer.global_step)
        prev_step = self._train_last_seen_global_step
        step_advanced = step > prev_step
        if step_advanced:
            self._train_epoch_seen_steps += step - prev_step
            self._train_last_seen_global_step = step

        total_steps = getattr(trainer, "estimated_stepping_batches", None)
        if not isinstance(total_steps, int) or total_steps <= 0:
            total_steps = None

        if not self._should_train_log(step, total_steps, step_advanced, now):
            return

        self._train_last_emit_t = now
        elapsed = 0.0 if self._train_t0 is None else now - self._train_t0
        epoch_elapsed = 0.0 if self._train_epoch_t0 is None else now - self._train_epoch_t0
        loss = self._extract_loss(outputs)
        loss_str = f" loss={loss:.4f}" if loss is not None else ""
        epoch_str, _ = self._epoch_info(trainer)
        steps_per_sec = (
            (self._train_epoch_seen_steps / epoch_elapsed)
            if epoch_elapsed > 0 and self._train_epoch_seen_steps > 0
            else 0.0
        )

        batch_total = self._train_total_batches
        if batch_total is not None:
            batch_seen = self._train_epoch_seen_batches
            epoch_progress = batch_seen / batch_total
            epoch_eta = (epoch_elapsed / batch_seen) * max(0, batch_total - batch_seen) if batch_seen > 0 else 0.0
            if self.enable_ascii_bar:
                bar = _progress_bar(epoch_progress)
                bar_str = f" [{bar}]"
            else:
                bar_str = ""
            step_part = f" step={step}/{total_steps}" if total_steps is not None else f" step={step}"
            print(
                f"[train]{bar_str} epoch={epoch_str} "
                f"epoch_pct={epoch_progress * 100:5.1f}% "
                f"batch={batch_seen}/{batch_total}{step_part}{loss_str} "
                f"steps/s={steps_per_sec:.3f} elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(epoch_eta)}",
                flush=True,
            )
            return

        if total_steps is not None:
            progress = step / total_steps
            eta = (elapsed / step) * max(0, total_steps - step) if step > 0 else 0.0
            if self.enable_ascii_bar:
                bar = _progress_bar(progress)
                bar_str = f" [{bar}]"
            else:
                bar_str = ""
            print(
                f"[train]{bar_str} epoch={epoch_str} "
                f"step_pct={progress * 100:5.1f}% "
                f"step={step}/{total_steps}{loss_str} "
                f"steps/s={steps_per_sec:.3f} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                flush=True,
            )
            return

        print(
            f"[train] epoch={epoch_str} step={step}{loss_str} "
            f"steps/s={steps_per_sec:.3f} elapsed={_format_duration(elapsed)}",
            flush=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        now = time.perf_counter()
        elapsed = 0.0 if self._train_t0 is None else now - self._train_t0
        epoch_elapsed = 0.0 if self._train_epoch_t0 is None else now - self._train_epoch_t0
        epoch_str, _ = self._epoch_info(trainer)
        steps_per_sec = (
            (self._train_epoch_seen_steps / epoch_elapsed)
            if epoch_elapsed > 0 and self._train_epoch_seen_steps > 0
            else 0.0
        )
        batch_total = self._train_total_batches
        if batch_total is not None and batch_total > 0:
            print(
                f"[train] epoch_end epoch={epoch_str} "
                f"batch={self._train_epoch_seen_batches}/{batch_total} "
                f"step={int(trainer.global_step)} steps/s={steps_per_sec:.3f} "
                f"elapsed={_format_duration(elapsed)}",
                flush=True,
            )
        else:
            print(
                f"[train] epoch_end epoch={epoch_str} "
                f"step={int(trainer.global_step)} steps/s={steps_per_sec:.3f} "
                f"elapsed={_format_duration(elapsed)}",
                flush=True,
            )

    def on_validation_epoch_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._val_t0 = time.perf_counter()
        self._val_seen_batches = 0
        self._val_last_logged_pct = 0

        num_val_batches = getattr(trainer, "num_val_batches", None)
        total = 0
        if isinstance(num_val_batches, (list, tuple)):
            for v in num_val_batches:
                if isinstance(v, int) and v > 0:
                    total += v
        elif isinstance(num_val_batches, int) and num_val_batches > 0:
            total = num_val_batches

        self._val_total_batches = total if total > 0 else None
        if self._val_total_batches is not None:
            print(f"[val] started total_batches={self._val_total_batches}", flush=True)
        else:
            print("[val] started", flush=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.is_global_zero:
            return

        self._val_seen_batches += 1
        seen = self._val_seen_batches
        total = self._val_total_batches

        elapsed = 0.0 if self._val_t0 is None else time.perf_counter() - self._val_t0

        if total is not None:
            progress = seen / total
            eta = (elapsed / seen) * max(0, total - seen) if seen > 0 else 0.0
            progress_pct = int(progress * 100)
            milestone = (progress_pct // self.val_percent_interval) * self.val_percent_interval
            should_emit = milestone >= self.val_percent_interval and milestone > self._val_last_logged_pct
            if not should_emit and seen < total:
                return
            self._val_last_logged_pct = max(self._val_last_logged_pct, min(100, milestone))
            if self.enable_ascii_bar:
                bar = _progress_bar(progress)
                bar_str = f" [{bar}]"
            else:
                bar_str = ""
            print(
                f"[val]{bar_str} "
                f"pct={progress * 100:5.1f}% "
                f"batch={seen}/{total} elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                flush=True,
            )
            return

        if seen % max(1, self.train_every_n_steps) != 0:
            return
        print(f"[val] batch={seen} elapsed={_format_duration(elapsed)}", flush=True)
