from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from src.model.modeling_smt import SMTModelForCausalLM


def _legacy_past_key_values():
    k = torch.zeros(2, 1, 1, 1)
    v = torch.zeros(2, 1, 1, 1)
    return (((k, v), (k, v)),)


def test_prepare_inputs_for_generation_with_past_requires_encoder_outputs():
    stub_model = SimpleNamespace(config=SimpleNamespace(bos_token_id=1))

    with pytest.raises(ValueError, match="encoder_outputs must be preserved"):
        SMTModelForCausalLM.prepare_inputs_for_generation(
            stub_model,
            input_ids=torch.tensor([[1, 2], [3, 4]]),
            past_key_values=_legacy_past_key_values(),
            encoder_outputs=None,
            attention_mask=torch.ones(2, 2, dtype=torch.long),
        )


def test_prepare_inputs_for_generation_with_past_keeps_encoder_outputs():
    stub_model = SimpleNamespace(config=SimpleNamespace(bos_token_id=1))
    encoder_outputs = object()

    prepared = SMTModelForCausalLM.prepare_inputs_for_generation(
        stub_model,
        input_ids=torch.tensor([[1, 2], [3, 4]]),
        past_key_values=_legacy_past_key_values(),
        encoder_outputs=encoder_outputs,
        encoder_attention_mask=torch.ones(2, 4, dtype=torch.bool),
        attention_mask=torch.ones(2, 2, dtype=torch.long),
    )

    assert prepared["encoder_outputs"] is encoder_outputs
    assert prepared["input_ids"].tolist() == [[2], [4]]
    assert prepared["attention_mask"] is None


def test_update_model_kwargs_persists_encoder_state_from_outputs():
    model = SMTModelForCausalLM.__new__(SMTModelForCausalLM)
    encoder_outputs = object()
    encoder_attention_mask = torch.ones(2, 4, dtype=torch.bool)

    k = torch.zeros(2, 1, 1, 1)
    v = torch.zeros(2, 1, 1, 1)
    outputs = CausalLMOutputWithCrossAttentions(
        logits=torch.zeros(2, 1, 8),
        past_key_values=(((k, v), (k, v)),),
    )
    outputs["encoder_outputs"] = encoder_outputs
    outputs["encoder_attention_mask"] = encoder_attention_mask

    updated = SMTModelForCausalLM._update_model_kwargs_for_generation(
        model,
        outputs=outputs,
        model_kwargs={
            "cache_position": torch.tensor([0]),
            "use_cache": True,
            "pixel_values": torch.randn(2, 3, 8, 8),
        },
        is_encoder_decoder=False,
    )

    assert updated["encoder_outputs"] is encoder_outputs
    assert updated["encoder_attention_mask"] is encoder_attention_mask
    assert "pixel_values" not in updated
