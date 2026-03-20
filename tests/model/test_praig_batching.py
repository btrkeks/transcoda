from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

PRAIG_PATH = Path("models/external/praig-smt").resolve()
if str(PRAIG_PATH) not in sys.path:
    sys.path.insert(0, str(PRAIG_PATH))

import smt_trainer as praig_trainer_module
from smt_model import SMTConfig, SMTModelForCausalLM


@pytest.fixture
def praig_model() -> SMTModelForCausalLM:
    config = SMTConfig(
        maxh=64,
        maxw=80,
        maxlen=8,
        out_categories=6,
        padding_token=0,
        in_channels=1,
        w2i={"<pad>": 0, "<bos>": 1, "<eos>": 2, "A": 3, "B": 4, "C": 5},
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "A", 4: "B", 5: "C"},
        d_model=256,
        dim_ff=256,
        num_dec_layers=2,
        attn_heads=4,
    )
    model = SMTModelForCausalLM(config)
    model.eval()
    return model


def test_praig_forward_decoder_supports_batching(praig_model: SMTModelForCausalLM) -> None:
    encoder_output = praig_model.forward_encoder(torch.randn(2, 1, 64, 80))
    image_sizes = torch.tensor([[64, 80], [33, 48]], dtype=torch.long)
    decoder_input = torch.tensor([[1, 3, 4], [1, 4, 0]], dtype=torch.long)

    output = praig_model.forward_decoder(
        encoder_output,
        decoder_input,
        memory_key_padding_mask=praig_model._generate_memory_key_padding_mask(
            encoder_output,
            image_sizes=image_sizes,
        ),
    )

    assert output.logits.shape == (2, 3, 6)


def test_praig_memory_mask_tracks_valid_encoder_regions(praig_model: SMTModelForCausalLM) -> None:
    encoder_output = torch.zeros((2, 256, 4, 5), dtype=torch.float32)
    image_sizes = torch.tensor([[64, 80], [17, 33]], dtype=torch.long)

    mask = praig_model._generate_memory_key_padding_mask(
        encoder_output,
        image_sizes=image_sizes,
    )

    assert mask.shape == (2, 20)
    assert torch.all(mask[0])

    expected = torch.tensor(
        [
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask[1].cpu(), expected)


def test_praig_predict_preserves_single_shape_and_batches_without_scalar_crash(
    monkeypatch: pytest.MonkeyPatch,
    praig_model: SMTModelForCausalLM,
) -> None:
    def fake_forward_encoder(input_tensor, image_sizes=None):
        del image_sizes
        return torch.zeros((input_tensor.shape[0], 256, 2, 2), dtype=torch.float32)

    def fake_forward_decoder(
        encoder_output,
        last_predictions,
        memory_key_padding_mask=None,
        return_weights=False,
    ):
        del encoder_output, memory_key_padding_mask, return_weights
        batch_size = last_predictions.shape[0]
        seq_len = last_predictions.shape[1]
        logits = torch.full((batch_size, seq_len, 6), -1e9, dtype=torch.float32)

        if batch_size == 1:
            next_tokens = [3 if seq_len == 1 else 2]
        elif seq_len == 1:
            next_tokens = [3, 4]
        elif seq_len == 2:
            next_tokens = [2, 5]
        else:
            next_tokens = [0, 2]

        for row, token_id in enumerate(next_tokens[:batch_size]):
            logits[row, -1, token_id] = 0.0

        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(praig_model, "forward_encoder", fake_forward_encoder)
    monkeypatch.setattr(praig_model, "forward_decoder", fake_forward_decoder)

    single_predictions, _ = praig_model.predict(
        torch.zeros((1, 1, 8, 8)),
        image_sizes=torch.tensor([[8, 8]], dtype=torch.long),
    )
    batched_predictions, _ = praig_model.predict(
        torch.zeros((2, 1, 8, 8)),
        image_sizes=torch.tensor([[8, 8], [6, 7]], dtype=torch.long),
    )

    assert single_predictions == ["A"]
    assert batched_predictions == [["A"], ["B", "C"]]


def test_praig_trainer_training_step_passes_image_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(praig_trainer_module, "summary", lambda *args, **kwargs: None)

    trainer = praig_trainer_module.SMT_Trainer(
        maxh=64,
        maxw=80,
        maxlen=8,
        out_categories=6,
        padding_token=0,
        in_channels=1,
        w2i={"<pad>": 0, "<bos>": 1, "<eos>": 2, "A": 3, "B": 4, "C": 5},
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "A", 4: "B", 5: "C"},
        d_model=256,
        dim_ff=256,
        num_dec_layers=2,
    )

    captured: dict[str, torch.Tensor] = {}

    class FakeTrainModel(nn.Module):
        def forward(self, *, encoder_input, decoder_input, labels, image_sizes):
            del encoder_input, decoder_input, labels
            captured["image_sizes"] = image_sizes
            return SimpleNamespace(loss=torch.tensor(1.0))

    trainer.model = FakeTrainModel()
    batch = (
        torch.zeros((2, 1, 8, 8)),
        torch.zeros((2, 3), dtype=torch.long),
        torch.zeros((2, 3), dtype=torch.long),
        torch.tensor([[8, 8], [6, 7]], dtype=torch.long),
    )

    loss = trainer.training_step(batch)

    assert torch.equal(captured["image_sizes"], batch[3])
    assert loss.item() == pytest.approx(1.0)


def test_praig_trainer_validation_step_records_every_batch_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(praig_trainer_module, "summary", lambda *args, **kwargs: None)

    trainer = praig_trainer_module.SMT_Trainer(
        maxh=64,
        maxw=80,
        maxlen=8,
        out_categories=6,
        padding_token=0,
        in_channels=1,
        w2i={"<pad>": 0, "<bos>": 1, "<eos>": 2, "A": 3, "B": 4, "C": 5},
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "A", 4: "B", 5: "C"},
        d_model=256,
        dim_ff=256,
        num_dec_layers=2,
    )

    class FakeEvalModel(nn.Module):
        i2w = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "A", 4: "B", 5: "C"}

        def predict(self, input, image_sizes=None):
            del input, image_sizes
            return [["A", "<b>"], ["B", "C"]], None

    trainer.model = FakeEvalModel()
    trainer.validation_step(
        (
            torch.zeros((2, 1, 8, 8)),
            torch.zeros((2, 3), dtype=torch.long),
            torch.tensor([[3, 2, 0], [4, 5, 2]], dtype=torch.long),
            torch.tensor([[8, 8], [6, 7]], dtype=torch.long),
        )
    )

    assert trainer.preds == ["A\n", "BC"]
    assert trainer.grtrs == ["A", "BC"]
