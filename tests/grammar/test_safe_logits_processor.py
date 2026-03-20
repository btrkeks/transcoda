from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from src.grammar import GrammarProvider


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


def test_grammar_logits_processor_tolerates_pad_on_finished_rows():
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
    processor = provider.create_logits_processor(
        pad_token_id=int(tokenizer.pad_token_id),
        collect_stats=True,
    )

    first_allowed = _allowed_token_ids(
        provider.compiled_grammar,
        int(tokenizer.vocab_size),
    )[0]
    second_allowed = _allowed_token_ids(
        provider.compiled_grammar,
        int(tokenizer.vocab_size),
        accepted_prefix=[int(first_allowed)],
    )[0]

    # Step 0: processor prefill.
    vocab_size = int(tokenizer.vocab_size)
    dummy_scores = torch.zeros((2, vocab_size), dtype=torch.float32)
    input_ids = torch.tensor(
        [[int(tokenizer.bos_token_id)], [int(tokenizer.bos_token_id)]],
        dtype=torch.long,
    )
    _ = processor(input_ids, dummy_scores.clone())

    # Step 1: both rows decode the same grammar-valid token.
    input_ids = torch.tensor(
        [
            [int(tokenizer.bos_token_id), int(first_allowed)],
            [int(tokenizer.bos_token_id), int(first_allowed)],
        ],
        dtype=torch.long,
    )
    _ = processor(input_ids, dummy_scores.clone())

    # Step 2: row 0 is externally finished and receives PAD while row 1 continues.
    # The processor must not assert on PAD for row 0.
    input_ids = torch.tensor(
        [
            [int(tokenizer.bos_token_id), int(first_allowed), int(tokenizer.pad_token_id)],
            [int(tokenizer.bos_token_id), int(first_allowed), int(second_allowed)],
        ],
        dtype=torch.long,
    )
    _ = processor(input_ids, dummy_scores.clone())

    stats = processor.stats()
    assert stats["calls"] == 3
    assert stats["rows_processed"] == 6
    assert stats["externally_finished_rows"] == 1
    assert stats["total_ms"] >= 0.0
