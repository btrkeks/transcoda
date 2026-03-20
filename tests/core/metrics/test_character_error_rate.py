import pytest
import torch

from src.core.kern_utils import strip_tie_beam_markers_from_kern_text
from src.core.metrics import CharacterErrorRate


def test_character_error_rate_compute_single_without_normalizer():
    i2w = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4a]L", 4: "4a"}
    cer = CharacterErrorRate.compute_single(
        pred=[1, 3, 2, 0],
        target=[1, 4, 2, 0],
        pad_id=0,
        i2w=i2w,
    )
    assert cer == pytest.approx(100.0)


def test_character_error_rate_compute_single_with_normalizer():
    i2w = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4a]L", 4: "4a"}
    cer = CharacterErrorRate.compute_single(
        pred=[1, 3, 2, 0],
        target=[1, 4, 2, 0],
        pad_id=0,
        i2w=i2w,
        text_normalizer=strip_tie_beam_markers_from_kern_text,
    )
    assert cer == pytest.approx(0.0)


def test_character_error_rate_update_with_normalizer():
    i2w = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4a]L", 4: "4a"}
    metric = CharacterErrorRate(
        pad_id=0,
        bos_id=1,
        eos_id=2,
        i2w=i2w,
        text_normalizer=strip_tie_beam_markers_from_kern_text,
    )

    preds = torch.tensor([[1, 3, 2, 0]], dtype=torch.long)
    targets = torch.tensor([[1, 4, 2, 0]], dtype=torch.long)
    metric.update(preds, targets)
    assert metric.compute().item() == pytest.approx(0.0)
