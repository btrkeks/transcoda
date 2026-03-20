from types import SimpleNamespace

import torch

from src.training.lightning_module import SMTTrainer


class _MetricRecorder:
    def __init__(self):
        self.calls = []

    def update(self, preds, target):
        self.calls.append((preds, target))


class _ModelStub:
    def __call__(self, **kwargs):
        return SimpleNamespace(loss=torch.tensor(0.5))


class _TrainerStub:
    def __init__(self):
        self.config = SimpleNamespace(
            maxlen=16,
            pad_token_id=0,
            i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "a", 4: "b"},
        )
        self.hparams = SimpleNamespace(training=SimpleNamespace(log_example_images=True))
        self._validation_example_logging_override = None
        self._validation_metric_prefix = "val"
        self.model = _ModelStub()
        self._val_set_names = ["synth"]
        self.val_metrics_by_set = {"synth": _MetricRecorder()}
        self._compute_omr_ned = False
        self._omr_ned_tracker = None

    def log(self, *args, **kwargs):
        return None

    def _forward_model(self):
        return self.model

    def _mark_compiled_step_begin(self):
        return None

    def _generate_with_grammar(self, pixel_values, image_sizes, max_length):
        return torch.tensor([[1, 3, 2, 0], [1, 4, 3, 2]])

    def should_log_validation_examples(self):
        return SMTTrainer.should_log_validation_examples(self)

    def _validation_set_metric_name(self, set_name, metric_name):
        return SMTTrainer._validation_set_metric_name(self, set_name, metric_name)


def test_validation_step_outputs_exclude_images():
    stub = _TrainerStub()
    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.tensor([[1, 3, 2, -100], [1, 4, 3, 2]]),
        "sample_ids": torch.tensor([10, 11]),
    }

    outputs = SMTTrainer.validation_step(stub, batch, batch_idx=0, dataloader_idx=0)

    assert outputs is not None
    assert "images" not in outputs
    assert outputs["val_set_name"] == "synth"
    assert outputs["sample_ids"].tolist() == [10, 11]
    assert outputs["sources"] == [None, None]
    assert len(outputs["cers"]) == 2
    assert outputs["cers_no_ties_beams"] is None
    assert len(outputs["pred_ids"]) == 2
    assert len(outputs["gt_ids"]) == 2
    assert len(stub.val_metrics_by_set["synth"].calls) == 1


def test_validation_step_outputs_include_polish_stripped_cers():
    stub = _TrainerStub()
    stub._val_set_names = ["polish"]
    stub.val_metrics_by_set = {"polish": _MetricRecorder()}
    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.tensor([[1, 3, 2, -100], [1, 4, 3, 2]]),
        "sample_ids": torch.tensor([20, 21]),
    }

    outputs = SMTTrainer.validation_step(stub, batch, batch_idx=0, dataloader_idx=0)

    assert outputs is not None
    assert outputs["val_set_name"] == "polish"
    assert outputs["sources"] == [None, None]
    assert outputs["cers_no_ties_beams"] is not None
    assert len(outputs["cers_no_ties_beams"]) == 2
    assert len(stub.val_metrics_by_set["polish"].calls) == 1


def test_validation_step_outputs_include_polish_family_stripped_cers():
    stub = _TrainerStub()
    stub._val_set_names = ["polish_dev"]
    stub.val_metrics_by_set = {"polish_dev": _MetricRecorder()}
    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.tensor([[1, 3, 2, -100], [1, 4, 3, 2]]),
        "sample_ids": torch.tensor([20, 21]),
    }

    outputs = SMTTrainer.validation_step(stub, batch, batch_idx=0, dataloader_idx=0)

    assert outputs is not None
    assert outputs["val_set_name"] == "polish_dev"
    assert outputs["cers_no_ties_beams"] is not None
    assert len(outputs["cers_no_ties_beams"]) == 2
    assert len(stub.val_metrics_by_set["polish_dev"].calls) == 1


def test_validation_step_outputs_include_sources_when_present():
    stub = _TrainerStub()
    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.tensor([[1, 3, 2, -100], [1, 4, 3, 2]]),
        "sample_ids": torch.tensor([30, 31]),
        "source": ["train_000030.krn", "train_000031.krn"],
    }

    outputs = SMTTrainer.validation_step(stub, batch, batch_idx=0, dataloader_idx=0)

    assert outputs is not None
    assert outputs["sources"] == ["train_000030.krn", "train_000031.krn"]


def test_validation_step_outputs_respect_runtime_example_logging_override():
    stub = _TrainerStub()
    stub._validation_example_logging_override = False
    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.tensor([[1, 3, 2, -100], [1, 4, 3, 2]]),
        "sample_ids": torch.tensor([40, 41]),
    }

    outputs = SMTTrainer.validation_step(stub, batch, batch_idx=0, dataloader_idx=0)

    assert outputs is None
    assert len(stub.val_metrics_by_set["synth"].calls) == 1
