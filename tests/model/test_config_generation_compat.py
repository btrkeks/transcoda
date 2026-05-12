import pytest

from src.model.configuration_transcoda import TranscodaConfig


def test_config_exposes_vocab_size_for_transformers_generation():
    config = TranscodaConfig(out_categories=321)
    assert config.vocab_size == 321
    assert config.get_text_config().vocab_size == 321


def test_config_rejects_mismatched_vocab_size_and_out_categories():
    with pytest.raises(ValueError, match="must match"):
        TranscodaConfig(out_categories=320, vocab_size=321)
