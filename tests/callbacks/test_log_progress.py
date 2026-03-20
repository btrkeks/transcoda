import torch

from src.callbacks.log_progress import LogProgressCallback


class _FakeClock:
    def __init__(self):
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


class _DummyTrainer:
    def __init__(self):
        self.is_global_zero = True
        self.global_step = 0
        self.estimated_stepping_batches = 100
        self.current_epoch = 0
        self.max_epochs = 10
        self.num_training_batches = 60
        self.num_val_batches = 100


def test_train_logs_deduplicate_same_global_step(monkeypatch, capsys):
    clock = _FakeClock()
    monkeypatch.setattr("src.callbacks.log_progress.time.perf_counter", clock.now)
    callback = LogProgressCallback(
        train_every_n_steps=10,
        train_interval_seconds=9999.0,
        val_percent_interval=10,
        enable_ascii_bar=False,
    )
    trainer = _DummyTrainer()

    callback.on_train_start(trainer, pl_module=None)
    callback.on_train_epoch_start(trainer, pl_module=None)

    trainer.global_step = 10
    for _ in range(6):
        clock.advance(1.0)
        callback.on_train_batch_end(
            trainer, pl_module=None, outputs=torch.tensor(0.2), batch=None, batch_idx=0
        )

    lines = capsys.readouterr().out.splitlines()
    step_lines = [line for line in lines if "step=10/100" in line]
    assert len(step_lines) == 1
    assert "steps/s=" in step_lines[0]


def test_train_logs_heartbeat_without_step_advance(monkeypatch, capsys):
    clock = _FakeClock()
    monkeypatch.setattr("src.callbacks.log_progress.time.perf_counter", clock.now)
    callback = LogProgressCallback(
        train_every_n_steps=1000,
        train_interval_seconds=30.0,
        val_percent_interval=10,
        enable_ascii_bar=False,
    )
    trainer = _DummyTrainer()

    callback.on_train_start(trainer, pl_module=None)
    callback.on_train_epoch_start(trainer, pl_module=None)

    trainer.global_step = 1
    clock.advance(5.0)
    callback.on_train_batch_end(trainer, pl_module=None, outputs=torch.tensor(0.2), batch=None, batch_idx=0)

    clock.advance(15.0)
    callback.on_train_batch_end(trainer, pl_module=None, outputs=torch.tensor(0.2), batch=None, batch_idx=1)

    clock.advance(11.0)
    callback.on_train_batch_end(trainer, pl_module=None, outputs=torch.tensor(0.2), batch=None, batch_idx=2)

    lines = capsys.readouterr().out.splitlines()
    step_lines = [line for line in lines if "step=1/100" in line]
    assert len(step_lines) == 1


def test_validation_logs_sparse_percent_milestones(monkeypatch, capsys):
    clock = _FakeClock()
    monkeypatch.setattr("src.callbacks.log_progress.time.perf_counter", clock.now)
    callback = LogProgressCallback(
        train_every_n_steps=50,
        train_interval_seconds=30.0,
        val_percent_interval=10,
        enable_ascii_bar=False,
    )
    trainer = _DummyTrainer()

    callback.on_validation_epoch_start(trainer, pl_module=None)
    for batch_idx in range(100):
        clock.advance(1.0)
        callback.on_validation_batch_end(
            trainer,
            pl_module=None,
            outputs=None,
            batch=None,
            batch_idx=batch_idx,
            dataloader_idx=0,
        )

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[val]")]
    assert len(lines) == 11
    assert any("pct= 10.0%" in line for line in lines)
    assert any("batch=100/100" in line for line in lines)
