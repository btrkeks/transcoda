from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.evaluation import harness as harness_module
from src.evaluation.omr_ned import OMRNEDResult


class _FakeDataset:
    def __init__(self, samples: list[dict]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]

    def select(self, indices) -> "_FakeDataset":
        return _FakeDataset([self._samples[i] for i in indices])


class _SequentialModel:
    name = "seq-model"

    def __init__(self, predictions: dict[int, str]):
        self._predictions = predictions

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        sample_id = int(np.array(image)[0, 0, 0])
        return torch.full((1, 1, 1), float(sample_id), dtype=torch.float32)

    def predict(self, image_tensor: torch.Tensor) -> str:
        sample_id = int(image_tensor[0, 0, 0].item())
        return self._predictions[sample_id]


class _BatchedModel(_SequentialModel):
    name = "batch-model"

    def batch_predict(self, pixel_values: torch.Tensor, image_sizes: torch.Tensor) -> list[str]:
        del image_sizes
        predictions: list[str] = []
        for i in range(pixel_values.shape[0]):
            sample_id = int(pixel_values[i, 0, 0, 0].item())
            predictions.append(self._predictions[sample_id])
        return predictions


def _sample(sample_id: int, transcription: str, source: str) -> dict:
    img = Image.fromarray(np.full((2, 2, 3), sample_id, dtype=np.uint8))
    return {"image": img, "transcription": transcription, "source": source}


@pytest.fixture
def fake_dataset():
    return _FakeDataset(
        [
            _sample(1, "abc", "synth"),
            _sample(2, "abc", "synth"),
            _sample(3, "xy", "polish"),
        ]
    )


def test_evaluation_harness_batched_matches_sequential(monkeypatch, fake_dataset):
    monkeypatch.setattr(harness_module, "load_from_disk", lambda _path: fake_dataset)

    predictions = {1: "abc", 2: "axc", 3: "xz"}
    sequential_harness = harness_module.EvaluationHarness(
        dataset_path="unused",
        models=[_SequentialModel(predictions)],
        device=torch.device("cpu"),
        batch_size=2,
    )
    batched_harness = harness_module.EvaluationHarness(
        dataset_path="unused",
        models=[_BatchedModel(predictions)],
        device=torch.device("cpu"),
        batch_size=2,
    )

    seq_result = sequential_harness.run().models[0]
    batch_result = batched_harness.run().models[0]

    assert seq_result.num_samples == batch_result.num_samples == 3
    assert seq_result.cer == pytest.approx(batch_result.cer)
    assert seq_result.ser == pytest.approx(batch_result.ser)
    assert seq_result.ler == pytest.approx(batch_result.ler)
    assert set(seq_result.per_source.keys()) == set(batch_result.per_source.keys()) == {
        "synth",
        "polish",
    }
    assert seq_result.per_source["synth"].cer == pytest.approx(batch_result.per_source["synth"].cer)
    assert seq_result.per_source["polish"].ser == pytest.approx(
        batch_result.per_source["polish"].ser
    )


def test_evaluation_harness_limit_selects_subset(monkeypatch, fake_dataset):
    monkeypatch.setattr(harness_module, "load_from_disk", lambda _path: fake_dataset)

    harness = harness_module.EvaluationHarness(
        dataset_path="unused",
        models=[_SequentialModel({1: "abc", 2: "abc", 3: "xy"})],
        device=torch.device("cpu"),
        limit=2,
    )

    results = harness.run()
    assert results.total_samples == 2
    assert results.models[0].num_samples == 2


def test_evaluation_harness_counts_omr_failures_as_100(monkeypatch, fake_dataset):
    monkeypatch.setattr(harness_module, "load_from_disk", lambda _path: fake_dataset)
    monkeypatch.setattr(
        harness_module.EvaluationHarness,
        "_resolve_omr_ned_compute_fn",
        lambda self: (
            lambda prediction, _ground_truth: OMRNEDResult(
                omr_ned=None if prediction == "bad" else 25.0,
                edit_distance=None,
                pred_notation_size=None,
                gt_notation_size=None,
                parse_error="parse failed" if prediction == "bad" else None,
                syntax_errors_fixed=0,
            )
        ),
    )

    harness = harness_module.EvaluationHarness(
        dataset_path="unused",
        models=[_SequentialModel({1: "ok", 2: "bad", 3: "ok"})],
        device=torch.device("cpu"),
        compute_omr_ned=True,
    )

    result = harness.run().models[0]

    assert result.omr_ned == pytest.approx((25.0 + 100.0 + 25.0) / 3)
    assert result.omr_ned_parse_failures == 1
    assert result.omr_ned_valid_count == 2
    assert result.per_source["synth"].omr_ned == pytest.approx((25.0 + 100.0) / 2)
    assert result.per_source["synth"].omr_ned_parse_failures == 1
    assert result.per_source["synth"].omr_ned_valid_count == 1
