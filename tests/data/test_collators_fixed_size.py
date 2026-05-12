import pytest
import torch

from src.data.collators import Img2SeqCollator


def _tokenizer_stub():
    return type("Tokenizer", (), {"pad_token_id": 0})()


def test_fixed_size_collator_stacks_without_padding():
    collator = Img2SeqCollator(
        tokenizer=_tokenizer_stub(),
        fixed_height=1485,
        fixed_width=1050,
        enforce_fixed_size=True,
        fixed_decoder_length=2000,
    )
    batch = [
        {"pixel_values": torch.zeros(3, 1485, 1050), "labels": [1, 2, 3]},
        {"pixel_values": torch.zeros(3, 1485, 1050), "labels": [1, 4, 5, 2]},
    ]

    result = collator(batch)

    assert result["pixel_values"].shape == (2, 3, 1485, 1050)
    assert result["image_sizes"].shape == (2, 2)
    assert result["image_sizes"].tolist() == [[1485, 1050], [1485, 1050]]
    assert result["labels"].shape == (2, 2000)
    assert result["decoder_attention_mask"].shape == (2, 2000)
    assert result["labels"][0, 3].item() == -100


def test_fixed_size_collator_raises_on_mismatch():
    collator = Img2SeqCollator(
        tokenizer=_tokenizer_stub(),
        fixed_height=1485,
        fixed_width=1050,
        enforce_fixed_size=True,
    )
    batch = [
        {"pixel_values": torch.zeros(3, 1485, 1050), "labels": [1, 2, 3], "sample_id": 7},
        {"pixel_values": torch.zeros(3, 1480, 1050), "labels": [1, 4, 2], "sample_id": 11},
    ]

    with pytest.raises(ValueError, match=r"expected fixed image size \(H, W\)=\(1485, 1050\)"):
        collator(batch)


def test_fixed_decoder_length_raises_on_overlength():
    collator = Img2SeqCollator(
        tokenizer=_tokenizer_stub(),
        fixed_height=1485,
        fixed_width=1050,
        enforce_fixed_size=True,
        fixed_decoder_length=2000,
    )
    batch = [
        {"pixel_values": torch.zeros(3, 1485, 1050), "labels": [1] * 2001, "sample_id": 99},
    ]

    with pytest.raises(ValueError, match=r"expected decoder length <= 2000 but got 2001"):
        collator(batch)
