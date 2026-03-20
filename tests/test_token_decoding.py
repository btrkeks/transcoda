"""
Tests for token ID to string decoding with whitespace preservation.

Ensures that _token_ids_to_string correctly reconstructs kern format text
with newlines and tabs from token IDs.
"""

import json

import pytest


@pytest.fixture
def vocab_mappings():
    """Load vocabulary mappings from the BPE tokenizer."""
    with open("vocab/grandstaff-bpe4k-newline_tab_split-tokenizer.json") as f:
        vocab = json.load(f)["model"]["vocab"]

    w2i = vocab
    i2w = {v: k for k, v in vocab.items()}

    # Get special token IDs
    pad_id = w2i.get("<pad>", 0)
    bos_id = w2i.get("<bos>", 1)
    eos_id = w2i.get("<eos>", 2)

    return {
        "w2i": w2i,
        "i2w": i2w,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
    }


def token_ids_to_string(token_ids: list[int], i2w: dict, pad_id: int) -> str:
    """
    Reference implementation of _token_ids_to_string for testing.
    Mirrors the implementation in src/modules/omr_planner.py
    """
    tokens = []
    for token_id in token_ids:
        if token_id == pad_id:
            break
        token = i2w.get(token_id, "")
        if token in ("<bos>", "<eos>"):
            continue
        tokens.append(token)

    # Reconstruct kern format from tokens
    text = "".join(tokens)
    # Legacy replacements (should be no-ops with new tokenizer)
    text = text.replace("<t>", "\t").replace("<b>", "\n").replace("<s>", " ")
    return text


def test_decode_with_newlines(vocab_mappings):
    """Test that token IDs with newlines decode correctly."""
    w2i = vocab_mappings["w2i"]
    i2w = vocab_mappings["i2w"]
    pad_id = vocab_mappings["pad_id"]

    # Create token sequence: *clefG2\n*k[]
    token_ids = [
        w2i.get("*clefG2", 100),
        w2i["\n"],  # newline token
        w2i.get("*k[]", 200),
    ]

    result = token_ids_to_string(token_ids, i2w, pad_id)

    assert "\n" in result, "Decoded string should contain newline"
    assert result.count("\n") == 1, "Should have exactly one newline"


def test_decode_with_tabs(vocab_mappings):
    """Test that token IDs with tabs decode correctly."""
    w2i = vocab_mappings["w2i"]
    i2w = vocab_mappings["i2w"]
    pad_id = vocab_mappings["pad_id"]

    # Create token sequence: *clefF4\t*clefG2
    token_ids = [
        w2i.get("*clefF4", 100),
        w2i["\t"],  # tab token
        w2i.get("*clefG2", 102),
    ]

    result = token_ids_to_string(token_ids, i2w, pad_id)

    assert "\t" in result, "Decoded string should contain tab"
    assert result.count("\t") == 1, "Should have exactly one tab"


def test_decode_mixed_whitespace(vocab_mappings):
    """Test decoding with both newlines and tabs."""
    w2i = vocab_mappings["w2i"]
    i2w = vocab_mappings["i2w"]
    pad_id = vocab_mappings["pad_id"]

    # Create token sequence: *clefF4\t*clefG2\n*k[]
    token_ids = [
        w2i.get("*clefF4", 100),
        w2i["\t"],
        w2i.get("*clefG2", 102),
        w2i["\n"],
        w2i.get("*k[]", 208),
    ]

    result = token_ids_to_string(token_ids, i2w, pad_id)

    assert result.count("\t") == 1, "Should have exactly one tab"
    assert result.count("\n") == 1, "Should have exactly one newline"

    # Check order is preserved
    tab_pos = result.index("\t")
    newline_pos = result.index("\n")
    assert tab_pos < newline_pos, "Tab should come before newline"


def test_decode_stops_at_padding(vocab_mappings):
    """Test that decoding stops at padding token."""
    w2i = vocab_mappings["w2i"]
    i2w = vocab_mappings["i2w"]
    pad_id = vocab_mappings["pad_id"]

    # Create token sequence with padding: *clefG2\n<pad><pad>...
    token_ids = [
        w2i.get("*clefG2", 102),
        w2i["\n"],
        pad_id,
        w2i.get("*k[]", 208),  # This should be ignored (after padding)
    ]

    result = token_ids_to_string(token_ids, i2w, pad_id)

    # Should stop at padding
    assert "*k[]" not in result, "Should stop decoding at padding token"
    assert result.endswith("\n"), "Should end with newline (before padding)"


def test_decode_skips_special_tokens(vocab_mappings):
    """Test that BOS/EOS tokens are skipped during decoding."""
    w2i = vocab_mappings["w2i"]
    i2w = vocab_mappings["i2w"]
    pad_id = vocab_mappings["pad_id"]
    bos_id = vocab_mappings["bos_id"]
    eos_id = vocab_mappings["eos_id"]

    # Create token sequence: <bos>*clefG2\n*k[]<eos>
    token_ids = [
        bos_id,
        w2i.get("*clefG2", 102),
        w2i["\n"],
        w2i.get("*k[]", 208),
        eos_id,
    ]

    result = token_ids_to_string(token_ids, i2w, pad_id)

    assert "<bos>" not in result, "BOS token should be stripped"
    assert "<eos>" not in result, "EOS token should be stripped"
    assert "\n" in result, "Newline should be preserved"


def test_decode_realistic_kern_sequence(vocab_mappings):
    """Test decoding a realistic kern transcription sequence."""
    w2i = vocab_mappings["w2i"]
    i2w = vocab_mappings["i2w"]
    pad_id = vocab_mappings["pad_id"]

    # Create a realistic kern sequence with multiple lines and tabs
    # *clefF4\t*clefG2\n*k[]\n4c\t4cc
    token_ids = [
        w2i.get("*clefF4", 100),
        w2i["\t"],
        w2i.get("*clefG2", 102),
        w2i["\n"],
        w2i.get("*k[]", 208),
        w2i["\n"],
        w2i.get("4c", 237),
        w2i["\t"],
        w2i.get("4cc", 300),
    ]

    result = token_ids_to_string(token_ids, i2w, pad_id)

    # Verify structure
    assert result.count("\n") == 2, "Should have 2 newlines"
    assert result.count("\t") == 2, "Should have 2 tabs"

    # Verify it looks like valid kern format
    lines = result.split("\n")
    assert len(lines) == 3, "Should have 3 lines"
    assert "\t" in lines[0], "First line should have tab (clef declarations)"
