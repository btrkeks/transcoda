"""Adapter for converting SMT tokenizer to xgrammar TokenizerInfo."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xgrammar as xgr


def create_tokenizer_info(
    vocab_dir: str,
    vocab_size: int,
) -> "xgr.TokenizerInfo":
    """Create xgrammar TokenizerInfo from saved HuggingFace tokenizer.

    Loads the BPE tokenizer from the vocab directory and wraps it for use
    with xgrammar's grammar compilation.

    Args:
        vocab_dir: Path to directory containing tokenizer.json
        vocab_size: Total vocabulary size (model's out_categories).
            May be larger than tokenizer vocab due to padding.

    Returns:
        xgr.TokenizerInfo suitable for grammar compilation.

    Raises:
        ImportError: If xgrammar is not installed.
    """
    import xgrammar as xgr
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(vocab_dir)
    return xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
