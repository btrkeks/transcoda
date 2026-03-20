from types import SimpleNamespace

import torch

from src.core.metrics import RunawayMonitorConfig, RunawayMonitorTracker
from src.training.lightning_module import SMTTrainer


class _TestMetricRecorder:
    def __init__(self):
        self.calls = []

    def update(self, preds, target):
        self.calls.append((preds, target))

    def compute(self):
        return {"test_SER": torch.tensor(12.0)}


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


class _TestTrainerStub:
    def __init__(self):
        self.config = SimpleNamespace(pad_token_id=0, maxlen=16, i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "A\n"})
        self.test_metrics = _TestMetricRecorder()
        self._test_runaway_tracker = _make_tracker(self.config.i2w)
        self.logged = []
        self.logged_dicts = []

    def _generate_with_grammar(self, pixel_values, image_sizes, max_length):
        return torch.tensor(
            [
                [1, 3, 3, 3, 3, 0],  # runaway candidate
                [1, 3, 2, 0, 0, 0],  # non-runaway
            ]
        )

    def log(self, name, value, **kwargs):
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        self.logged.append((name, float(value), kwargs))

    def log_dict(self, values, **kwargs):
        out = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu().item())
            out[key] = float(value)
        self.logged_dicts.append((out, kwargs))


def _log_map(logged):
    return {name: value for name, value, _ in logged}


def test_test_step_updates_runaway_tracker():
    stub = _TestTrainerStub()
    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.tensor([[1, 3, 2, -100, -100, -100], [1, 3, 2, -100, -100, -100]]),
    }

    SMTTrainer.test_step(stub, batch, _batch_idx=0)
    computed = stub._test_runaway_tracker.compute()

    assert len(stub.test_metrics.calls) == 1
    assert computed["runaway_samples"] == 2
    assert computed["runaway_count"] >= 1


def test_on_test_epoch_end_logs_and_resets_runaway_metrics():
    stub = _TestTrainerStub()
    stub._test_runaway_tracker.update_batch(
        torch.tensor([[1, 3, 3, 3, 3, 0], [1, 3, 2, 0, 0, 0]]),
        torch.tensor([[1, 3, 2, 0, 0, 0], [1, 3, 2, 0, 0, 0]]),
        max_length_cap=5,
    )

    SMTTrainer.on_test_epoch_end(stub)
    logs = _log_map(stub.logged)

    assert len(stub.logged_dicts) == 0
    assert "test/SER" in logs
    assert "test/runaway_rate" in logs
    assert "test/runaway_samples" in logs
    assert stub._test_runaway_tracker.compute()["runaway_samples"] == 0
