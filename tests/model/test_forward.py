"""Test forward pass with new HuggingFace-compatible signature."""

import pytest
import torch

from src.model.configuration_smt import SMTConfig
from src.model.modeling_smt import SMTModelForCausalLM


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    config = SMTConfig(
        d_model=64,
        num_hidden_layers=2,
        dim_ff=128,
        num_attn_heads=2,
        maxlen=50,
        encoder_model_name_or_path="facebook/convnextv2-tiny-1k-224",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SMTModelForCausalLM(config)
    model.eval()
    return model


@pytest.fixture
def sample_batch():
    """Create a sample batch with minimal size."""
    batch_size = 2
    seq_len = 10
    return {
        "pixel_values": torch.randn(batch_size, 3, 64, 64),
        "labels": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "image_sizes": torch.tensor([[64, 64], [60, 60]]),
    }


def test_forward_with_labels(small_model, sample_batch):
    """Test forward pass with labels (training mode)."""
    with torch.no_grad():
        output = small_model(
            pixel_values=sample_batch["pixel_values"],
            labels=sample_batch["labels"],
            attention_mask=sample_batch["attention_mask"],
            image_sizes=sample_batch["image_sizes"],
        )

    assert output.logits is not None
    assert output.loss is not None
    assert output.logits.shape[0] == sample_batch["pixel_values"].shape[0]
    assert output.logits.shape[2] == small_model.config.out_categories


def test_forward_with_input_ids(small_model, sample_batch):
    """Test forward pass with explicit input_ids (no labels)."""
    with torch.no_grad():
        output = small_model(
            pixel_values=sample_batch["pixel_values"],
            input_ids=sample_batch["labels"],
            attention_mask=sample_batch["attention_mask"],
            image_sizes=sample_batch["image_sizes"],
        )

    assert output.logits is not None
    assert output.loss is None  # No loss computed without labels
    assert output.logits.shape[0] == sample_batch["pixel_values"].shape[0]


def test_forward_with_use_cache(small_model, sample_batch):
    """Test forward pass with caching enabled."""
    with torch.no_grad():
        output = small_model(
            pixel_values=sample_batch["pixel_values"],
            input_ids=sample_batch["labels"],
            attention_mask=sample_batch["attention_mask"],
            image_sizes=sample_batch["image_sizes"],
            use_cache=True,
        )

    assert output.logits is not None
    assert output.past_key_values is not None
    assert len(output.past_key_values) == small_model.config.num_hidden_layers


def test_forward_with_encoder_outputs(small_model, sample_batch):
    """Test forward pass with pre-computed encoder outputs."""
    with torch.no_grad():
        # First pass: compute encoder outputs
        first_output = small_model(
            pixel_values=sample_batch["pixel_values"],
            input_ids=sample_batch["labels"][:, :1],  # First token only
            image_sizes=sample_batch["image_sizes"],
            use_cache=True,
        )

        # Get encoder outputs from first forward pass
        encoder_outputs = small_model.forward_encoder(
            sample_batch["pixel_values"], image_sizes=sample_batch["image_sizes"]
        )

        # Second pass: reuse encoder outputs
        # Note: image_sizes not needed when using cached encoder_outputs
        second_output = small_model(
            encoder_outputs=encoder_outputs,
            input_ids=sample_batch["labels"][:, :2],  # First two tokens
            past_key_values=first_output.past_key_values,
            use_cache=True,
        )

    assert second_output.logits is not None
    assert second_output.past_key_values is not None


def test_generate_greedy(small_model, sample_batch):
    """Test that model.generate() works with greedy decoding."""
    with torch.no_grad():
        generated_ids = small_model.generate(
            pixel_values=sample_batch["pixel_values"],
            image_sizes=sample_batch["image_sizes"],
            max_length=20,
            do_sample=False,  # Greedy decoding
        )

    assert generated_ids is not None
    assert generated_ids.shape[0] == sample_batch["pixel_values"].shape[0]
    assert generated_ids.shape[1] <= 20
    # Check that all sequences start with BOS token
    assert (generated_ids[:, 0] == small_model.config.bos_token_id).all()


def test_generate_beam_search(small_model, sample_batch):
    """Test that model.generate() works with beam search."""
    with torch.no_grad():
        generated_ids = small_model.generate(
            pixel_values=sample_batch["pixel_values"],
            image_sizes=sample_batch["image_sizes"],
            max_length=15,
            num_beams=2,  # Beam search with 2 beams
            do_sample=False,
        )

    assert generated_ids is not None
    assert generated_ids.shape[0] == sample_batch["pixel_values"].shape[0]
    assert generated_ids.shape[1] <= 15
    # Check that all sequences start with BOS token
    assert (generated_ids[:, 0] == small_model.config.bos_token_id).all()


def test_backward_compatibility_error():
    """Test that old parameter names are not accepted (cause ValueError due to missing pixel_values)."""
    config = SMTConfig(
        d_model=64,
        num_hidden_layers=2,
        dim_ff=128,
        num_attn_heads=2,
        maxlen=50,
        encoder_model_name_or_path="facebook/convnextv2-tiny-1k-224",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SMTModelForCausalLM(config)

    # Old parameter name 'encoder_input' is ignored (in **kwargs), causing ValueError
    with pytest.raises(ValueError, match="pixel_values or encoder_outputs"):
        model(
            encoder_input=torch.randn(2, 3, 64, 64),
            labels=torch.randint(0, 100, (2, 10)),
        )
