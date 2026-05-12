"""Shared tokenizer/vocab compatibility helpers for training and inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.artifacts import _hash_vocab_dict


def vocab_from_tokenizer(tokenizer: Any) -> tuple[dict[str, int], dict[int, str]]:
    """Return token->id and id->token mappings from a HF tokenizer instance."""
    w2i = tokenizer.get_vocab()
    i2w = {int(idx): token for token, idx in w2i.items()}
    return w2i, i2w


def resolve_vocab_dir(data_config: Mapping[str, Any]) -> str:
    """Resolve tokenizer directory from run artifact data config."""
    vocab_dir = data_config.get("vocab_dir")
    if not isinstance(vocab_dir, str) or not vocab_dir.strip():
        raise ValueError("Missing required 'vocab_dir' in run artifact data config.")
    return vocab_dir


def assert_vocab_hashes_match(
    *,
    expected_w2i_hash: str,
    expected_i2w_hash: str,
    w2i: Mapping[int | str, int | str],
    i2w: Mapping[int | str, int | str],
    context_label: str,
) -> None:
    """Validate current vocab mappings against expected artifact hashes."""
    actual_w2i_hash = _hash_vocab_dict(w2i)
    actual_i2w_hash = _hash_vocab_dict(i2w)

    if expected_w2i_hash == actual_w2i_hash and expected_i2w_hash == actual_i2w_hash:
        return

    raise ValueError(
        f"{context_label} tokenizer mismatch detected.\n"
        "Expected vocab hashes from checkpoint run artifact:\n"
        f"  w2i_hash={expected_w2i_hash}\n"
        f"  i2w_hash={expected_i2w_hash}\n"
        "Loaded tokenizer hashes:\n"
        f"  w2i_hash={actual_w2i_hash}\n"
        f"  i2w_hash={actual_i2w_hash}\n\n"
        "This usually means tokenizer files changed after checkpoint training. "
        "Use the tokenizer version that matches the checkpoint run artifact."
    )
