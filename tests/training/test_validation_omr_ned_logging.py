from types import SimpleNamespace

import torch

from src.core.metrics import OMRNEDTracker
from src.evaluation.omr_ned import OMRNEDResult
from src.training.lightning_module import SMTTrainer


class _MetricCollectionStub:
    def __init__(self):
        self.updates = []
        self.compute_calls = 0
        self.reset_calls = 0

    def update(self, preds, targets):
        self.updates.append((preds, targets))

    def compute(self):
        self.compute_calls += 1
        return {"SER": torch.tensor(0.25)}

    def reset(self):
        self.reset_calls += 1


class _ModelStub:
    def __call__(self, **_kwargs):
        return SimpleNamespace(loss=torch.tensor(0.5))


class _TrainerStub:
    def __init__(self):
        self.config = SimpleNamespace(
            maxlen=16,
            pad_token_id=0,
            i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "a", 4: "b"},
        )
        self.hparams = SimpleNamespace(training=SimpleNamespace(log_example_images=False))
        self.model = _ModelStub()
        self._val_set_names = ["synth"]
        self.val_metrics_by_set = {"synth": _MetricCollectionStub()}
        self._val_batches_seen_by_set = {"synth": 0}
        self._val_runaway_tracker_by_set = None
        self._validation_metric_prefix = "val"
        self._validation_example_logging_override = None
        self._compute_omr_ned = True
        self._omr_ned_tracker = OMRNEDTracker()
        self._omr_ned_tracker._enabled = True
        self.logged = []

    def log(self, name, value, **kwargs):
        self.logged.append((name, value, kwargs))

    def _forward_model(self):
        return self.model

    def _mark_compiled_step_begin(self):
        return None

    def _generate_with_grammar(self, pixel_values, image_sizes, max_length):
        del pixel_values, image_sizes, max_length
        return torch.tensor([[1, 3, 2, 0], [1, 4, 2, 0]])

    def _validation_set_metric_name(self, set_name, metric_name):
        return SMTTrainer._validation_set_metric_name(self, set_name, metric_name)

    def _validation_aggregate_metric_name(self, metric_name):
        return SMTTrainer._validation_aggregate_metric_name(self, metric_name)

    def should_log_validation_examples(self):
        return SMTTrainer.should_log_validation_examples(self)


def test_validation_logs_canonical_omr_ned_metrics(monkeypatch):
    def fake_compute(pred_str, target_str):
        del target_str
        if pred_str.endswith("\nb"):
            return OMRNEDResult(
                omr_ned=None,
                edit_distance=None,
                pred_notation_size=None,
                gt_notation_size=None,
                parse_error="parse failed",
                syntax_errors_fixed=0,
            )
        return OMRNEDResult(
            omr_ned=12.5,
            edit_distance=1,
            pred_notation_size=2,
            gt_notation_size=2,
            parse_error=None,
            syntax_errors_fixed=0,
        )

    monkeypatch.setattr("src.core.metrics.omr_ned_tracker.compute_omr_ned", fake_compute)

    stub = _TrainerStub()
    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.tensor([[1, 3, 2, -100], [1, 4, 2, -100]]),
        "sample_ids": torch.tensor([10, 11]),
    }

    SMTTrainer.validation_step(stub, batch, batch_idx=0, dataloader_idx=0)
    SMTTrainer.on_validation_epoch_end(stub)

    logged = {name: value for name, value, _kwargs in stub.logged}
    assert logged["val/synth/OMR_NED"] == 56.25
    assert logged["val/synth/OMR_NED_failures"] == 1.0
    assert logged["val/aggregate/OMR_NED"] == 56.25
    assert logged["val/aggregate/OMR_NED_failures"] == 1.0

    metric_collection = stub.val_metrics_by_set["synth"]
    assert len(metric_collection.updates) == 1
    assert metric_collection.compute_calls == 1
    assert metric_collection.reset_calls == 1
