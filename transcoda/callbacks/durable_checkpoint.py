"""Periodic checkpoints that survive a killed training job."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import Callback

from transcoda.core.checkpoint_record import (
    CommittedCheckpoint,
    fsync_dir,
    fsync_file,
    read_committed_records,
    record_path,
    sha256_file,
    write_committed_record,
)

FILENAME_PREFIX = "durable-step="


class DurableCheckpoint(Callback):
    """Every `every_steps` optimizer steps and at fit end: save to a temp file, fsync,
    rename, hash, then write `<dirpath>/committed/<step:08d>.json`. Keep the newest
    `keep` committed checkpoints. A checkpoint without a committed record does not exist.

    SIGUSR1 requests one extra commit at the next optimizer step."""

    def __init__(
        self,
        dirpath: str | Path,
        every_steps: int,
        *,
        target_steps: int | None = None,
        keep: int = 2,
    ) -> None:
        if every_steps < 1:
            raise ValueError("every_steps must be >= 1")
        if keep < 1:
            raise ValueError("keep must be >= 1")
        self.dirpath = Path(dirpath)
        self.every_steps = every_steps
        self.target_steps = target_steps
        self.keep = keep
        self._last_seen_step = -1
        self._last_committed_step = -1
        self._save_requested = False
        self._previous_usr1: Any = None
        self._usr1_installed = False

    def setup(self, trainer, pl_module, stage: str) -> None:
        if stage != "fit" or not hasattr(signal, "SIGUSR1"):
            return
        try:
            self._previous_usr1 = signal.signal(signal.SIGUSR1, self._request_save)
        except ValueError:
            return
        self._usr1_installed = True

    def teardown(self, trainer, pl_module, stage: str) -> None:
        if not self._usr1_installed:
            return
        try:
            signal.signal(signal.SIGUSR1, self._previous_usr1)
        except (ValueError, TypeError):
            pass
        self._usr1_installed = False

    def _request_save(self, signum: int, frame: Any) -> None:
        self._save_requested = True
        if callable(self._previous_usr1):
            self._previous_usr1(signum, frame)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if step <= 0 or step == self._last_seen_step:
            return
        self._last_seen_step = step
        requested = self._save_requested
        if trainer.world_size > 1:
            requested = bool(trainer.strategy.broadcast(requested, src=0))
        if step % self.every_steps == 0 or requested:
            self.commit(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        if int(trainer.global_step) > 0:
            self.commit(trainer)

    def commit(self, trainer) -> CommittedCheckpoint | None:
        step = int(trainer.global_step)
        self._save_requested = False
        if step == self._last_committed_step:
            return None
        filename = f"{FILENAME_PREFIX}{step:08d}.ckpt"
        tmp = self.dirpath / f".{filename}.tmp"
        if trainer.is_global_zero:
            self.dirpath.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(tmp)
        record = None
        if trainer.is_global_zero:
            record = publish_checkpoint(
                self.dirpath, tmp, filename, step=step, target_steps=self.target_steps
            )
            prune_committed(self.dirpath, keep=self.keep)
        self._last_committed_step = step
        return record


def publish_checkpoint(
    dirpath: Path,
    tmp: Path,
    filename: str,
    *,
    step: int,
    target_steps: int | None,
) -> CommittedCheckpoint:
    fsync_file(tmp)
    record = CommittedCheckpoint(
        step=step,
        filename=filename,
        sha256=sha256_file(tmp),
        size=tmp.stat().st_size,
        target_steps=target_steps,
    )
    os.replace(tmp, dirpath / filename)
    fsync_dir(dirpath)
    write_committed_record(dirpath, record)
    return record


def prune_committed(dirpath: Path, *, keep: int) -> None:
    records = read_committed_records(dirpath)
    kept = records[-keep:]
    for record in records[:-keep]:
        record_path(dirpath, record.step).unlink(missing_ok=True)
        if all(item.filename != record.filename for item in kept):
            (dirpath / record.filename).unlink(missing_ok=True)
    kept_files = {record.filename for record in kept}
    for path in dirpath.glob(f"{FILENAME_PREFIX}*.ckpt"):
        if path.name not in kept_files:
            path.unlink(missing_ok=True)
    for path in dirpath.glob(f".{FILENAME_PREFIX}*.ckpt.tmp"):
        path.unlink(missing_ok=True)
    fsync_dir(record_path(dirpath, 0).parent)
