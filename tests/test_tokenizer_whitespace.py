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
    return Tokenizer.from_file("vocab/grandstaff-bpe4k-newline_tab_split-tokenizer.json")


def test_tokenizer_preserves_newlines(bpe_tokenizer):
    """Test that newlines are preserved as separate tokens."""
    text = "*clefG2\n*k[]"
    encoding = bpe_tokenizer.encode(text)

    # Newline should be a separate token
    assert "\n" in encoding.tokens, "Newline should be preserved as a token"

    # Decoding should restore the original text
    decoded = bpe_tokenizer.decode(encoding.ids, skip_special_tokens=True)
    assert "\n" in decoded, "Decoded text should contain newline"


def test_tokenizer_preserves_tabs(bpe_tokenizer):
    """Test that tabs are preserved as separate tokens."""
    text = "*clefF4\t*clefG2"
    encoding = bpe_tokenizer.encode(text)

    # Tab should be a separate token
    assert "\t" in encoding.tokens, "Tab should be preserved as a token"

    # Decoding should restore the original text
    decoded = bpe_tokenizer.decode(encoding.ids, skip_special_tokens=True)
    assert "\t" in decoded, "Decoded text should contain tab"


def test_tokenizer_preserves_mixed_whitespace(bpe_tokenizer):
    """Test that mixed newlines and tabs are preserved."""
    text = "*clefF4\t*clefG2\n*k[b-e-a-d-]\t*k[b-e-a-d-]\n*M9/16\t*M9/16"
    encoding = bpe_tokenizer.encode(text)

    # Count whitespace tokens
    newline_count = encoding.tokens.count("\n")
    tab_count = encoding.tokens.count("\t")

    assert newline_count == 2, f"Expected 2 newlines, found {newline_count}"
    assert tab_count == 3, f"Expected 3 tabs, found {tab_count}"

    # Decoding should restore the original text exactly
    decoded = bpe_tokenizer.decode(encoding.ids, skip_special_tokens=True)
    assert decoded == text, "Decoded text should match original exactly"


def test_tokenizer_whitespace_in_vocabulary(bpe_tokenizer):
    """Test that newline and tab characters exist in the vocabulary."""
    vocab = bpe_tokenizer.get_vocab()

    assert "\n" in vocab, "Newline character must be in vocabulary"
    assert "\t" in vocab, "Tab character must be in vocabulary"

    # Check that they have valid token IDs
    newline_id = vocab["\n"]
    tab_id = vocab["\t"]

    assert isinstance(newline_id, int), "Newline should have an integer ID"
    assert isinstance(tab_id, int), "Tab should have an integer ID"
    assert newline_id >= 0, "Newline ID should be non-negative"
    assert tab_id >= 0, "Tab ID should be non-negative"


def test_tokenizer_roundtrip_kern_format(bpe_tokenizer):
    """Test that a realistic kern transcription survives tokenization roundtrip."""
    # Realistic kern format with both newlines and tabs
    kern_text = """*clefF4\t*clefG2
*k[]
*M4/4
4c\t4cc
4d\t4dd
=1
4e\t4ee
4f\t4ff"""

    encoding = bpe_tokenizer.encode(kern_text)
    decoded = bpe_tokenizer.decode(encoding.ids, skip_special_tokens=True)

    # The decoded text should match the original
    assert decoded == kern_text, "Kern format should survive tokenization roundtrip"

    # Verify structure is preserved
    assert decoded.count("\n") == kern_text.count("\n"), "All newlines should be preserved"
    assert decoded.count("\t") == kern_text.count("\t"), "All tabs should be preserved"
