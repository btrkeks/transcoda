from types import SimpleNamespace

import pytest
import torch

from src.training.lightning_module import SMTTrainer


class _SubsetMetricRecorder:
    def __init__(self, values: dict[str, torch.Tensor] | None = None):
        self.calls = []
        self.values = values or {"CER": torch.tensor(12.5)}
        self.reset_calls = 0

    def update(self, preds, target):
        self.calls.append((preds, target))

    def compute(self):
        return self.values

    def reset(self):
        self.reset_calls += 1


class _TrackerStub:
    def __init__(self):
        self.calls = 0

    def update_batch(self, preds, targets, max_length_cap):
        del preds, targets, max_length_cap
        self.calls += 1

    def compute(self):
        return {
            "runaway_rate": 0.0,
            "runaway_length_blowup_rate": 0.0,
            "runaway_repeat_loop_rate": 0.0,
            "runaway_no_eos_at_max_length_rate": 0.0,
            "runaway_max_length_hit_rate": 0.0,
            "runaway_samples": 0,
        }

    def reset(self):
        return None


class _ModelStub:
    def __call__(self, **kwargs):
        del kwargs
        return SimpleNamespace(loss=torch.tensor(0.5))


class _ValidationSubsetTrainerStub:
    def __init__(self):
        self.config = SimpleNamespace(
            maxlen=16,
            pad_token_id=0,
            i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "a", 4: "b"},
        )
        self.hparams = SimpleNamespace(training=SimpleNamespace(log_example_images=True))
        self._validation_metric_prefix = "val"
        self.model = _ModelStub()
        self._val_set_names = ["synth", "synth_subset"]
        self._base_val_set_names = ["synth"]
        self.val_metrics_by_set = {
            "synth": _SubsetMetricRecorder({"SER": torch.tensor(0.9), "CER": torch.tensor(1.2)}),
            "synth_subset": _SubsetMetricRecorder({"CER": torch.tensor(7.5)}),
        }
        self._val_batches_seen_by_set = {"synth": 0, "synth_subset": 0}
        self._compute_omr_ned = False
        self._omr_ned_tracker = None
        self._val_runaway_tracker_by_set = {"synth": _TrackerStub()}
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
        del pixel_values, image_sizes, max_length
        return torch.tensor([[1, 3, 2, 0], [1, 4, 2, 0]])

    def should_log_validation_examples(self):
        return bool(self.hparams.training.log_example_images)

    def _validation_set_metric_name(self, set_name, metric_name):
        return SMTTrainer._validation_set_metric_name(self, set_name, metric_name)

    def _validation_aggregate_metric_name(self, metric_name):
        return SMTTrainer._validation_aggregate_metric_name(self, metric_name)


def _log_names(logged):
    return [name for name, _, _ in logged]


def test_validation_step_subset_loader_logs_no_loss_and_skips_auxiliary_metrics():
    stub = _ValidationSubsetTrainerStub()
    stub._val_set_names = ["synth_subset"]
    stub._base_val_set_names = ["synth"]
    stub.val_metrics_by_set = {"synth_subset": _SubsetMetricRecorder({"CER": torch.tensor(7.5)})}
    stub._val_batches_seen_by_set = {"synth_subset": 0}
    synth_tracker = _TrackerStub()
    stub._val_runaway_tracker_by_set = {"synth": synth_tracker}

    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.tensor([[1, 3, 2, -100], [1, 4, 2, -100]]),
        "sample_ids": torch.tensor([10, 11]),
    }

    outputs = SMTTrainer.validation_step(stub, batch, batch_idx=0, dataloader_idx=0)

    assert outputs is None
    assert len(stub.val_metrics_by_set["synth_subset"].calls) == 1
    assert synth_tracker.calls == 0
    assert _log_names(stub.logged) == []


def test_on_validation_epoch_end_logs_subset_cer_without_full_set_metrics_on_frequent_pass():
    stub = _ValidationSubsetTrainerStub()
    stub._val_batches_seen_by_set = {"synth": 0, "synth_subset": 1}

    SMTTrainer.on_validation_epoch_end(stub)

    logged_names = _log_names(stub.logged)
    assert "val/synth_subset/CER" in logged_names
    assert "val/synth/CER" not in logged_names
    assert "val/synth_subset/SER" not in logged_names
    assert "val/synth_subset/LER" not in logged_names
    assert "val/aggregate/active_set_count" in logged_names
    assert "val/aggregate/is_full_pass" in logged_names


def test_on_validation_epoch_end_logs_full_and_subset_metrics_on_full_pass():
    stub = _ValidationSubsetTrainerStub()
    stub._val_batches_seen_by_set = {"synth": 2, "synth_subset": 1}

    SMTTrainer.on_validation_epoch_end(stub)

    logs = {name: value for name, value, _ in stub.logged}
    assert logs["val/synth/CER"] == pytest.approx(1.2)
    assert logs["val/synth_subset/CER"] == pytest.approx(7.5)
    assert logs["val/aggregate/active_set_count"] == 1.0
    assert logs["val/aggregate/is_full_pass"] == 1.0


def test_on_validation_epoch_end_final_validation_namespace_has_no_subset_metrics():
    stub = _ValidationSubsetTrainerStub()
    stub._validation_metric_prefix = "final_val"
    stub._val_batches_seen_by_set = {"synth": 2, "synth_subset": 0}

    SMTTrainer.on_validation_epoch_end(stub)

    logged_names = _log_names(stub.logged)
    assert "final_val/synth/CER" in logged_names
    assert "final_val/aggregate/active_set_count" in logged_names
    assert "final_val/synth_subset/CER" not in logged_names
