"""Grammar-constrained decoding for **kern generation.

This module provides xgrammar integration for ensuring syntactically valid
**kern output during model inference. The grammar is specified in GBNF format
and compiled once per model lifetime.

Requires xgrammar to be installed: uv sync --group grammar

Example:
    from src.grammar import GrammarProvider

    provider = GrammarProvider("grammars/kern.gbnf", vocab_dir, vocab_size)

    # In generation:
    logits_processor = provider.create_logits_processor(pad_token_id=pad_token_id)
    output = model.generate(..., logits_processor=[logits_processor])
"""

from .constraint_factory import ConstrainedDecodingFactory, ConstraintBundle
from .provider import GrammarProvider

__all__ = ["ConstraintBundle", "ConstrainedDecodingFactory", "GrammarProvider"]
