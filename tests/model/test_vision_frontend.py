from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.configuration_smt import SMTConfig
from src.model.frontends.conv_frontend import ConvVisionFrontend
from src.model.modeling_smt import SMTModelForCausalLM
from src.model.vision_frontend import ProjectorMLP


class _DummyEncoder(nn.Module):
    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        self.out_channels = out_channels

    def gradient_checkpointing_enable(self) -> None:
        return None

    def forward(self, pixel_values: torch.Tensor):
        # Downsample by 4x to mimic a vision backbone feature map.
        x = pixel_values.mean(dim=1, keepdim=True)
        features = F.avg_pool2d(x, kernel_size=4, stride=4)
        features = features.repeat(1, self.out_channels, 1, 1)
        return SimpleNamespace(last_hidden_state=features)


@pytest.fixture
def patch_encoder_loader(monkeypatch):
    def _load(_config):
        return _DummyEncoder(out_channels=16), 16

    monkeypatch.setattr("src.model.frontends.conv_frontend.EncoderLoader.load", _load)


def test_projector_mlp_shape():
    projector = ProjectorMLP(in_dim=16, out_dim=32, hidden_mult=3.0)
    tokens = torch.randn(2, 10, 16)
    projected = projector(tokens)

    assert projector.hidden_dim == 96
    assert projected.shape == (2, 10, 32)


def test_conv_frontend_output_contract(patch_encoder_loader):
    config = SMTConfig(
        d_model=32,
        num_hidden_layers=2,
        dim_ff=64,
        num_attn_heads=4,
        out_categories=128,
        projector_hidden_mult=2.0,
    )
    frontend = ConvVisionFrontend(config)

    images = torch.randn(2, 3, 64, 80)
    image_sizes = torch.tensor([[64, 80], [60, 72]])

    outputs = frontend(images=images, image_sizes=image_sizes)

    assert outputs.encoder_tokens_raw.shape == (2, 16 * 20, 32)
    assert outputs.encoder_tokens_pos.shape == (2, 16 * 20, 32)
    assert outputs.encoder_attention_mask is not None
    assert outputs.encoder_attention_mask.dtype == torch.bool
    assert outputs.encoder_attention_mask.shape == (2, 16 * 20)
    assert outputs.encoder_attention_mask[0].sum() >= outputs.encoder_attention_mask[1].sum()


def test_model_rejects_legacy_projector_checkpoint(patch_encoder_loader):
    config = SMTConfig(
        d_model=32,
        num_hidden_layers=2,
        dim_ff=64,
        num_attn_heads=4,
        out_categories=128,
    )
    model = SMTModelForCausalLM(config)
    state_dict = model.state_dict()

    # Simulate an old checkpoint where only legacy bridge keys exist.
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("frontend.projector.")}
    filtered["encoder_to_decoder_projection.weight"] = torch.randn(32, 16, 1, 1)
    filtered["encoder_to_decoder_projection.bias"] = torch.randn(32)

    with pytest.raises(RuntimeError, match="legacy single-layer projector"):
        model.load_state_dict(filtered, strict=False)
