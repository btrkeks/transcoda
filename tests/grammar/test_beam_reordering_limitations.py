from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from src.grammar import GrammarProvider
from src.grammar.stateful_kern_logits_processor import StatefulKernLogitsProcessor


def _xgrammar_installed() -> bool:
    try:
        import xgrammar  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _xgrammar_installed(),
    reason="xgrammar not installed (install with: uv sync --group grammar)",
)


def _allowed_token_ids(
    compiled_grammar,
    vocab_size: int,
    *,
    accepted_prefix: list[int] | None = None,
) -> list[int]:
    import xgrammar as xgr

    matcher = xgr.GrammarMatcher(compiled_grammar)
    for token_id in accepted_prefix or []:
        assert matcher.accept_token(int(token_id))

    bitmask = xgr.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(bitmask, 0)

    scores = torch.zeros((1, vocab_size), dtype=torch.float32)
    xgr.apply_token_bitmask_inplace(scores, bitmask)
    return torch.nonzero(scores[0] != float("-inf"), as_tuple=True)[0].tolist()


def _find_divergent_root_tokens(compiled_grammar, vocab_size: int) -> tuple[int, int]:
    root_allowed = _allowed_token_ids(compiled_grammar, vocab_size)
    for token_a in root_allowed:
        allowed_after_a = set(_allowed_token_ids(compiled_grammar, vocab_size, accepted_prefix=[token_a]))
        for token_b in root_allowed:
            if token_a == token_b:
                continue
            if token_b in allowed_after_a:
                continue

            allowed_after_b = set(
                _allowed_token_ids(compiled_grammar, vocab_size, accepted_prefix=[token_b])
            )
            if token_a not in allowed_after_b:
                return int(token_a), int(token_b)

    raise AssertionError("Failed to find a divergent pair of valid root tokens for beam-reorder test.")


def test_grammar_processor_rejects_beam_style_row_reordering() -> None:
    vocab_dir = Path("./vocab/bpe3k-splitspaces")
    grammar_path = Path("./grammars/kern.gbnf")
    if not vocab_dir.exists() or not grammar_path.exists():
        pytest.skip("Required vocab/grammar assets are unavailable in this checkout")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(vocab_dir))
    assert tokenizer.bos_token_id is not None
    assert tokenizer.pad_token_id is not None

    provider = GrammarProvider(
        grammar_path=str(grammar_path),
        vocab_dir=str(vocab_dir),
        vocab_size=int(tokenizer.vocab_size),
    )
    processor = provider.create_logits_processor(pad_token_id=int(tokenizer.pad_token_id))

    token_a, token_b = _find_divergent_root_tokens(provider.compiled_grammar, int(tokenizer.vocab_size))
    bos_token_id = int(tokenizer.bos_token_id)
    scores = torch.zeros((2, int(tokenizer.vocab_size)), dtype=torch.float32)

    # Step 0: initialize row-local matcher state.
    processor(torch.tensor([[bos_token_id], [bos_token_id]], dtype=torch.long), scores.clone())

    # Step 1: let the two rows diverge onto distinct valid prefixes.
    processor(torch.tensor([[bos_token_id, token_a], [bos_token_id, token_b]], dtype=torch.long), scores.clone())

    # Step 2: simulate beam search reordering hypotheses between rows.
    with pytest.raises(AssertionError, match="Grammar matcher rejected sampled token"):
        processor(torch.tensor([[bos_token_id, token_b], [bos_token_id, token_a]], dtype=torch.long), scores.clone())


class _BranchRule:
    def __init__(self) -> None:
        self._first_text: str | None = None

    @property
    def terminated(self) -> bool:
        return False

    def on_text_appended(self, prefix_state) -> None:
        if self._first_text is None:
            self._first_text = prefix_state.current_field_buffer

    def on_tab_appended(self, prefix_state) -> None:
        return None

    def on_line_closed(self, fields: tuple[str, ...]) -> None:
        return None

    def can_accept_tab(self, prefix_state) -> bool:
        return True

    def can_close_line(self, fields: tuple[str, ...]) -> bool:
        return True

    def can_end_sequence(self, fields: tuple[str, ...]) -> bool:
        return True

    def mask_scores(self, prefix_state, row_scores: torch.FloatTensor, context) -> None:
        if self._first_text == "A":
            row_scores[8] = float("-inf")
        elif self._first_text == "B":
            row_scores[7] = float("-inf")


def test_semantic_processor_keeps_stale_state_after_beam_style_row_reordering() -> None:
    processor = StatefulKernLogitsProcessor(
        i2w={
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            3: "\t",
            4: "\n",
            5: "A",
            6: "B",
            7: "X",
            8: "Y",
        },
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        rule_factories=[_BranchRule],
    )
    scores = torch.zeros((2, 9), dtype=torch.float32)

    processor(torch.tensor([[1], [1]], dtype=torch.long), scores.clone())
    processor(torch.tensor([[1, 5], [1, 6]], dtype=torch.long), scores.clone())

    swapped_scores = processor(torch.tensor([[1, 6], [1, 5]], dtype=torch.long), scores.clone())

    # The row histories now end with B and A respectively, but the processor
    # still masks according to the original row-local state established above.
    assert swapped_scores[0, 7] != float("-inf")
    assert swapped_scores[0, 8] == float("-inf")
    assert swapped_scores[1, 7] == float("-inf")
    assert swapped_scores[1, 8] != float("-inf")
