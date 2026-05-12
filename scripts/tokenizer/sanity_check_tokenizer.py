#!/usr/bin/env python3
"""Sanity check for BPE tokenizers."""

import argparse
from pathlib import Path

from transformers import PreTrainedTokenizerFast


def sanity_check_tokenizer(tokenizer_path: str | Path):
    """Run sanity checks on a tokenizer."""
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    print("\n" + "=" * 80)
    print("TOKENIZER SANITY CHECK")
    print("=" * 80)

    # 1. Check vocab size
    vocab_size = len(tokenizer)
    print(f"\n1. Vocabulary size: {vocab_size}")

    # 2. Check special tokens
    print("\n2. Special tokens:")
    print(f"   PAD: '{tokenizer.pad_token}' -> ID {tokenizer.pad_token_id}")
    print(f"   BOS: '{tokenizer.bos_token}' -> ID {tokenizer.bos_token_id}")
    print(f"   EOS: '{tokenizer.eos_token}' -> ID {tokenizer.eos_token_id}")
    print(f"   UNK: '{tokenizer.unk_token}' -> ID {tokenizer.unk_token_id}")

    # Verify special token IDs
    issues = []
    if tokenizer.pad_token_id != 0:
        issues.append(f"PAD token ID should be 0, got {tokenizer.pad_token_id}")
    if tokenizer.bos_token_id != 1:
        issues.append(f"BOS token ID should be 1, got {tokenizer.bos_token_id}")
    if tokenizer.eos_token_id != 2:
        issues.append(f"EOS token ID should be 2, got {tokenizer.eos_token_id}")
    if tokenizer.unk_token_id != 3:
        issues.append(f"UNK token ID should be 3, got {tokenizer.unk_token_id}")

    if issues:
        print("   ❌ ISSUES FOUND:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✓ All special tokens have correct IDs")

    # 3. Test basic encoding/decoding
    print("\n3. Basic encode/decode test:")
    test_strings = [
        "4c 4d 4e 4f",
        "=1\t=1\n*-\t*-",
        "4.cc\t4dd\n8ee\t8ff",
        "*clefG2\t*clefF4\n*k[f#c#]\t*k[f#c#]\n*M4/4\t*M4/4",
    ]

    all_passed = True
    for test_str in test_strings:
        encoded = tokenizer.encode(test_str, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)

        # Check roundtrip (some whitespace normalization is expected)
        roundtrip_ok = decoded == test_str or decoded.strip() == test_str.strip()

        status = "✓" if roundtrip_ok else "❌"
        print(f"   {status} Input: {repr(test_str)}")
        print(
            f"      Tokens: {len(encoded)} tokens -> IDs: {encoded[:10]}{'...' if len(encoded) > 10 else ''}"
        )
        print(f"      Output: {repr(decoded)}")

        if not roundtrip_ok:
            all_passed = False
            print("      ❌ Roundtrip failed!")

    if not all_passed:
        print("\n   ⚠️  Some roundtrip tests failed")

    # 4. Test special tokens handling
    print("\n4. Special tokens handling:")
    text = "4c 4d"
    with_special = tokenizer.encode(text, add_special_tokens=True)
    without_special = tokenizer.encode(text, add_special_tokens=False)

    print(f"   Without special tokens: {without_special}")
    print(f"   With special tokens:    {with_special}")

    expected_with = [tokenizer.bos_token_id] + without_special + [tokenizer.eos_token_id]
    if with_special == expected_with:
        print("   ✓ BOS/EOS tokens correctly added")
    else:
        print(f"   ❌ Expected {expected_with}, got {with_special}")

    # 5. Test batch encoding
    print("\n5. Batch encoding test:")
    batch = ["4c 4d", "8ee 8ff", "2gg"]
    encoded_batch = tokenizer(batch, padding=True, return_tensors="pt")

    print(f"   Input IDs shape: {encoded_batch['input_ids'].shape}")
    print(f"   Attention mask shape: {encoded_batch['attention_mask'].shape}")
    print("   ✓ Batch encoding works")

    # 6. Sample some random tokens from vocab
    print("\n6. Sample vocabulary tokens:")
    import random

    sample_size = min(20, vocab_size - 4)  # Exclude special tokens
    sample_ids = random.sample(range(4, vocab_size), sample_size)

    for token_id in sample_ids[:10]:  # Show first 10
        token = tokenizer.decode([token_id])
        print(f"   ID {token_id:5d}: {repr(token)}")

    # 7. Check for common Humdrum tokens
    print("\n7. Checking common Humdrum kern tokens:")
    common_tokens = [
        "*",
        "**kern",
        "=",
        "4",
        "8",
        "c",
        "d",
        "e",
        "f",
        "g",
        "\t",
        "\n",
        ".",
        "-",
        "#",
    ]

    for token in common_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        status = "✓" if decoded == token else "⚠️"
        print(f"   {status} '{token}' -> {encoded} -> '{decoded}'")

    print("\n" + "=" * 80)
    print("SANITY CHECK COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run sanity checks on a tokenizer")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="vocab/grandstaff-bpe4k",
        help="Path to the tokenizer directory",
    )
    args = parser.parse_args()

    sanity_check_tokenizer(args.tokenizer_path)


if __name__ == "__main__":
    main()
