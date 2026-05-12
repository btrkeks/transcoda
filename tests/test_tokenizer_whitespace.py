"""
Tests to ensure tokenizer preserves newlines and tabs.

This prevents regression where whitespace characters might be stripped during tokenization,
which would prevent the model from learning to produce properly formatted output.
"""

import pytest
from tokenizers import Tokenizer


@pytest.fixture
def bpe_tokenizer():
    """Load the BPE tokenizer used for training."""
    return Tokenizer.from_file("vocab/bpe3k-splitspaces-tokenizer.json")


def test_tokenizer_preserves_kern_whitespace_roundtrip(bpe_tokenizer):
    """Test that kern newlines and tabs survive tokenization."""
    text = "*clefF4\t*clefG2\n*k[b-e-a-d-]\t*k[b-e-a-d-]\n*M9/16\t*M9/16"
    encoding = bpe_tokenizer.encode(text)

    assert encoding.tokens.count("\n") == 2
    assert encoding.tokens.count("\t") == 3
    decoded = bpe_tokenizer.decode(encoding.ids, skip_special_tokens=True)
    assert decoded == text


def test_tokenizer_whitespace_in_vocabulary(bpe_tokenizer):
    """Test that newline and tab characters exist in the vocabulary."""
    vocab = bpe_tokenizer.get_vocab()

    assert "\n" in vocab, "Newline character must be in vocabulary"
    assert "\t" in vocab, "Tab character must be in vocabulary"

    newline_id = vocab["\n"]
    tab_id = vocab["\t"]

    assert isinstance(newline_id, int)
    assert isinstance(tab_id, int)
    assert newline_id >= 0
    assert tab_id >= 0
