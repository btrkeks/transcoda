from types import SimpleNamespace

import pytest
import torch

from src.core.metrics import RunawayMonitorConfig, RunawayMonitorTracker
from src.training.lightning_module import SMTTrainer


class _ValMetricRecorder:
    def __init__(self):
        self.calls = []
        self.reset_called = False

    def update(self, preds, target):
        self.calls.append((preds, target))

    def compute(self):
        return {
            "SER": torch.tensor(10.0),
            "CER": torch.tensor(20.0),
            "LER": torch.tensor(30.0),
        }

    def reset(self):
        self.reset_called = True


class _ModelStub:
    def __call__(self, **kwargs):
        return SimpleNamespace(loss=torch.tensor(0.5))


def _make_tracker(i2w: dict[int, str]) -> RunawayMonitorTracker:
    return RunawayMonitorTracker(
        pad_id=0,
        bos_id=1,
        eos_id=2,
        i2w=i2w,
        config=RunawayMonitorConfig(
            max_len_ratio=1.8,
            repeat_ngram_size=4,
            repeat_ngram_max_occurrences=5,
            max_identical_line_run=6,
            flag_no_eos_at_max_length=True,
        ),
    )


class _ValidationTrainerStub:
    def __init__(self):
        self.config = SimpleNamespace(
            maxlen=16,
            pad_token_id=0,
            i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "A\n", 4: "B\n", 5: "C\n"},
        )
        self.hparams = SimpleNamespace(training=SimpleNamespace(log_example_images=True))
        self.model = _ModelStub()
        self._val_set_names = ["synth", "polish"]
        self.val_metrics_by_set = {"synth": _ValMetricRecorder(), "polish": _ValMetricRecorder()}
        self._val_batches_seen_by_set = {"synth": 0, "polish": 0}
        self._compute_omr_ned = False
        self._omr_ned_tracker = None
        self._val_runaway_tracker_by_set = {
            "synth": _make_tracker(self.config.i2w),
            "polish": _make_tracker(self.config.i2w),
        }
        self.logged = []

    def log(self, name, value, **kwargs):
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        self.logged.append((name, float(value), kwargs))

    def _forward_model(self):
        return self.model

    def _mark_compiled_step_begin(self):
        return None

    def _generate_with_grammar(self, pixel_values, image_sizes, max_length):
        return torch.tensor(
            [
                [1, 3, 3, 3, 3, 0],  # no eos, long
                [1, 3, 4, 2, 0, 0],  # normal
            ]
        )


def _log_map(logged):
    return {name: value for name, value, _ in logged}


def test_validation_step_updates_runaway_tracker():
    stub = _ValidationTrainerStub()
    stub._val_set_names = ["synth"]
    stub.val_metrics_by_set = {"synth": _ValMetricRecorder()}
    stub._val_batches_seen_by_set = {"synth": 0}
    stub._val_runaway_tracker_by_set = {"synth": _make_tracker(stub.config.i2w)}

    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.tensor([[1, 3, 2, -100, -100, -100], [1, 3, 4, 2, -100, -100]]),
        "sample_ids": torch.tensor([11, 12]),
    }

    outputs = SMTTrainer.validation_step(stub, batch, batch_idx=0, dataloader_idx=0)

    computed = stub._val_runaway_tracker_by_set["synth"].compute()
    assert computed["runaway_samples"] == 2
    assert computed["runaway_count"] >= 1
    assert outputs is not None
    assert outputs["val_set_name"] == "synth"
    assert len(stub.val_metrics_by_set["synth"].calls) == 1


def test_on_validation_epoch_end_logs_per_set_and_overall_runaway_metrics():
    stub = _ValidationTrainerStub()
    stub._val_batches_seen_by_set = {"synth": 1, "polish": 1}

    synth_preds = torch.tensor([[1, 3, 3, 3, 3, 0], [1, 3, 4, 2, 0, 0]])
    synth_targets = torch.tensor([[1, 3, 2, 0, 0, 0], [1, 3, 4, 2, 0, 0]])
    stub._val_runaway_tracker_by_set["synth"].update_batch(
        synth_preds, synth_targets, max_length_cap=5
    )

    polish_preds = torch.tensor([[1, 3, 3, 3, 3, 0]])
    polish_targets = torch.tensor([[1, 3, 2, 0, 0, 0]])
    stub._val_runaway_tracker_by_set["polish"].update_batch(
        polish_preds, polish_targets, max_length_cap=5
    )

    SMTTrainer.on_validation_epoch_end(stub)
    logs = _log_map(stub.logged)

    assert "val/synth/runaway_rate" in logs
    assert "val/polish/runaway_rate" in logs
    assert "val/aggregate/runaway_rate" in logs
    assert logs["val/aggregate/runaway_samples"] == 3.0
    assert logs["val/aggregate/runaway_rate"] == pytest.approx(66.666, rel=1e-2)

    assert stub._val_runaway_tracker_by_set["synth"].compute()["runaway_samples"] == 0
    assert stub._val_runaway_tracker_by_set["polish"].compute()["runaway_samples"] == 0


def test_on_validation_epoch_end_skips_unseen_set_from_overall_runaway_metrics():
    stub = _ValidationTrainerStub()
    stub._val_batches_seen_by_set = {"synth": 1, "polish": 0}

    synth_preds = torch.tensor([[1, 3, 3, 3, 3, 0], [1, 3, 4, 2, 0, 0]])
    synth_targets = torch.tensor([[1, 3, 2, 0, 0, 0], [1, 3, 4, 2, 0, 0]])
    stub._val_runaway_tracker_by_set["synth"].update_batch(
        synth_preds, synth_targets, max_length_cap=5
    )
    stub._val_runaway_tracker_by_set["polish"].update_batch(
        torch.tensor([[1, 3, 3, 3, 3, 0]]),
        torch.tensor([[1, 3, 2, 0, 0, 0]]),
        max_length_cap=5,
    )

    SMTTrainer.on_validation_epoch_end(stub)
    logs = _log_map(stub.logged)

    assert "val/synth/runaway_rate" in logs
    assert "val/polish/runaway_rate" not in logs
    assert logs["val/aggregate/runaway_rate"] == pytest.approx(50.0, rel=1e-5)
