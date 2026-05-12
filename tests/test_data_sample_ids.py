"""Tests for sample_id propagation in training data pipeline."""

import torch

from src.data.collators import Img2SeqCollator
from src.data.datasets import HFDatasetWrapper


def test_hf_dataset_wrapper_adds_sample_id():
    """Wrapped samples should include stable sample_id equal to dataset index."""
    base_dataset = [
        {"pixel_values": torch.randn(3, 8, 8), "labels": [1, 2, 3]},
        {"pixel_values": torch.randn(3, 8, 8), "labels": [4, 5, 6]},
    ]
    wrapped = HFDatasetWrapper(base_dataset)

    sample0 = wrapped[0]
    sample1 = wrapped[1]

    assert sample0["sample_id"] == 0
    assert sample1["sample_id"] == 1


def test_img2seq_collator_passes_sample_ids():
    """Collator should include sample_ids when sample_id is present in samples."""
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    collator = Img2SeqCollator(tokenizer=tokenizer, enforce_fixed_size=False)

    batch = [
        {
            "pixel_values": torch.randn(3, 8, 10),
            "labels": [1, 3, 2],
            "sample_id": 7,
        },
        {
            "pixel_values": torch.randn(3, 10, 12),
            "labels": [1, 4, 5, 2],
            "sample_id": 11,
        },
    ]
    result = collator(batch)

    assert "sample_ids" in result
    assert result["sample_ids"].tolist() == [7, 11]
