import pytest
import torch

from src.core.metrics import SymbolErrorRate

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


@pytest.fixture
def ser_metric():
    return SymbolErrorRate(pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID)


def test_symbol_error_rate_perfect_match(ser_metric):
    preds = torch.tensor([[BOS_ID, 10, 11, 12, EOS_ID, PAD_ID]], dtype=torch.long)
    targets = torch.tensor([[BOS_ID, 10, 11, 12, EOS_ID, PAD_ID]], dtype=torch.long)

    ser_metric.update(preds, targets)

    assert ser_metric.compute().item() == pytest.approx(0.0)


def test_symbol_error_rate_single_substitution(ser_metric):
    preds = torch.tensor([[BOS_ID, 10, 11, 99, 13, EOS_ID, PAD_ID]], dtype=torch.long)
    targets = torch.tensor([[BOS_ID, 10, 11, 12, 13, EOS_ID, PAD_ID]], dtype=torch.long)

    ser_metric.update(preds, targets)

    assert ser_metric.compute().item() == pytest.approx(25.0)


def test_symbol_error_rate_insertion_exceeds_hundred(ser_metric):
    preds = torch.tensor([[BOS_ID, 5, 6, 7, 8, 9, 10, 11, EOS_ID, PAD_ID]], dtype=torch.long)
    targets = torch.tensor(
        [[BOS_ID, 5, 6, EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]], dtype=torch.long
    )

    ser_metric.update(preds, targets)

    assert ser_metric.compute().item() == pytest.approx(250.0)


def test_symbol_error_rate_handles_special_tokens_mid_sequence(ser_metric):
    preds = torch.tensor([[BOS_ID, 4, BOS_ID, 5, 6, EOS_ID, PAD_ID]], dtype=torch.long)
    targets = torch.tensor([[BOS_ID, 4, 5, 6, EOS_ID, PAD_ID, PAD_ID]], dtype=torch.long)

    ser_metric.update(preds, targets)

    assert ser_metric.compute().item() == pytest.approx(0.0)


def test_symbol_error_rate_ignores_empty_targets(ser_metric):
    preds = torch.tensor([[BOS_ID, 3, 4, EOS_ID, PAD_ID]], dtype=torch.long)
    targets = torch.tensor([[BOS_ID, EOS_ID, PAD_ID, PAD_ID]], dtype=torch.long)

    ser_metric.update(preds, targets)

    assert ser_metric.compute().item() == pytest.approx(0.0)


def test_symbol_error_rate_accumulates_across_batches(ser_metric):
    preds_first = torch.tensor(
        [
            [BOS_ID, 10, 11, 12, EOS_ID, PAD_ID],
            [BOS_ID, 20, 21, EOS_ID, PAD_ID, PAD_ID],
        ],
        dtype=torch.long,
    )
    targets_first = torch.tensor(
        [
            [BOS_ID, 10, 11, 12, EOS_ID, PAD_ID],
            [BOS_ID, 20, 21, 22, EOS_ID, PAD_ID],
        ],
        dtype=torch.long,
    )

    ser_metric.update(preds_first, targets_first)

    preds_second = torch.tensor(
        [[BOS_ID, 30, 31, 99, 33, EOS_ID, PAD_ID]],
        dtype=torch.long,
    )
    targets_second = torch.tensor(
        [[BOS_ID, 30, 31, 32, 33, EOS_ID, PAD_ID]],
        dtype=torch.long,
    )

    ser_metric.update(preds_second, targets_second)

    expected = 100.0 * (1 + 1) / (3 + 3 + 4)

    assert ser_metric.compute().item() == pytest.approx(expected, rel=1e-4)
