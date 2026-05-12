#!/usr/bin/env python3
"""
Convert old NumPy vocabulary format to HuggingFace tokenizer format.

This script converts the simple w2i/i2w NumPy dictionaries to a proper
HuggingFace tokenizer with special tokens.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def create_hf_tokenizer_json(vocab_dict: dict, special_tokens: dict) -> dict:
    """
    Create a HuggingFace tokenizer JSON structure from a vocabulary dictionary.

    Args:
        vocab_dict: Dictionary mapping tokens to IDs
        special_tokens: Dictionary of special token names to their content

    Returns:
        Dictionary representing the HuggingFace tokenizer format
    """
    # Create added_tokens list for special tokens
    added_tokens = []
    for special_id, (_name, content) in enumerate(special_tokens.items()):
        added_tokens.append(
            {
                "id": special_id,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )

    # Build the vocabulary with special tokens first
    full_vocab = {}
    offset = len(special_tokens)

    # Add special tokens to vocab
    for idx, (_name, content) in enumerate(special_tokens.items()):
        full_vocab[content] = idx

    # Add regular tokens to vocab (offsetting their IDs)
    for token, original_id in vocab_dict.items():
        # Skip if token is already a special token
        if token not in special_tokens.values():
            full_vocab[token] = original_id + offset - 1  # -1 because old vocab starts at 1

    # Create the tokenizer structure
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFKC"}]},
        "pre_tokenizer": None,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<eos>", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<eos>", "type_id": 0}},
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 0}},
                {"SpecialToken": {"id": "<eos>", "type_id": 0}},
            ],
            "special_tokens": {
                "<bos>": {"id": "<bos>", "ids": [1], "tokens": ["<bos>"]},
                "<eos>": {"id": "<eos>", "ids": [2], "tokens": ["<eos>"]},
            },
        },
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": full_vocab, "unk_token": "<unk>"},
    }

    return tokenizer_json


def convert_vocab(w2i_path: Path, output_path: Path):
    """
    Convert NumPy vocabulary to HuggingFace tokenizer format.

    Args:
        w2i_path: Path to the w2i.npy file
        output_path: Path where the tokenizer.json should be saved
    """
    # Load the old vocabulary
    w2i = np.load(w2i_path, allow_pickle=True).item()

    print(f"Loaded vocabulary from {w2i_path}")
    print(f"  Original vocab size: {len(w2i)}")

    # Define special tokens (matching the new format)
    special_tokens = {"pad": "<pad>", "bos": "<bos>", "eos": "<eos>", "unk": "<unk>"}

    # Create the HuggingFace tokenizer structure
    tokenizer_json = create_hf_tokenizer_json(w2i, special_tokens)

    # Save the tokenizer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    print(f"Saved HuggingFace tokenizer to {output_path}")
    print(f"  Total vocab size (with special tokens): {len(tokenizer_json['model']['vocab'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert old NumPy vocabulary to HuggingFace tokenizer format"
    )
    parser.add_argument(
        "--w2i_path",
        type=Path,
        default=Path("vocab_old/GrandStaffw2i.npy"),
        help="Path to the w2i.npy file",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("vocab/grandstaff-wordlevel-tokenizer.json"),
        help="Output path for the HuggingFace tokenizer JSON",
    )

    args = parser.parse_args()

    if not args.w2i_path.exists():
        print(f"Error: {args.w2i_path} does not exist")
        return 1

    convert_vocab(args.w2i_path, args.output_path)
    return 0


if __name__ == "__main__":
    exit(main())
