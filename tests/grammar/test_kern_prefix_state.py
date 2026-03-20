from __future__ import annotations

import pytest

from src.grammar.kern_prefix_state import KernPrefixState, KernPrefixStateError
from src.grammar.stateful_kern_logits_processor import TokenizerConstraintContext


def test_split_star_dash_across_tokens():
    state = KernPrefixState()

    state.accept_token_text("*")
    state.accept_token_text("-")
    preview = state.accept_token_text("\n")

    assert preview is not None
    assert preview.fields == ("*-",)


def test_empty_field_after_tab_is_rejected():
    state = KernPrefixState()
    state.accept_token_text("4c")
    state.accept_token_text("\t")

    with pytest.raises(KernPrefixStateError, match="empty field"):
        state.accept_token_text("\t")


def test_line_close_requires_non_empty_field():
    state = KernPrefixState()

    with pytest.raises(KernPrefixStateError, match="empty field"):
        state.accept_token_text("\n")


def test_tokenizer_context_requires_dedicated_delimiters():
    with pytest.raises(ValueError, match="missing dedicated"):
        TokenizerConstraintContext.from_i2w(
            i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c"},
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )


def test_tokenizer_context_rejects_embedded_delimiters():
    with pytest.raises(ValueError, match="embedded tabs/newlines"):
        TokenizerConstraintContext.from_i2w(
            i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "\t", 4: "\n", 5: "bad\tok"},
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
