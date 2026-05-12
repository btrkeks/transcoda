from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from src.grammar.spine_structure_processor import SpineStructureLogitsProcessor

_VOCAB_DIR = Path("./vocab/bpe3k-splitspaces")


@pytest.fixture
def production_tokenizer() -> PreTrainedTokenizerFast:
    if not _VOCAB_DIR.exists():
        pytest.skip("Production tokenizer assets are unavailable in this checkout")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(_VOCAB_DIR))
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        pytest.skip("Tokenizer is missing BOS/EOS ids")
    return tokenizer


def _make_processor(tokenizer: PreTrainedTokenizerFast) -> SpineStructureLogitsProcessor:
    vocab = tokenizer.get_vocab()
    i2w = {token_id: token for token, token_id in vocab.items()}
    return SpineStructureLogitsProcessor(
        i2w=i2w,
        bos_token_id=int(tokenizer.bos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
    )


def _scores_for_prefix(
    processor: SpineStructureLogitsProcessor,
    tokenizer: PreTrainedTokenizerFast,
    prefix_text: str,
) -> torch.Tensor:
    prefix_ids = [int(tokenizer.bos_token_id)] + tokenizer.encode(prefix_text, add_special_tokens=False)
    scores = None
    vocab_size = int(tokenizer.vocab_size)
    for end in range(1, len(prefix_ids) + 1):
        input_ids = torch.tensor([prefix_ids[:end]], dtype=torch.long)
        scores = processor(input_ids, torch.zeros((1, vocab_size), dtype=torch.float32))
    assert scores is not None
    return scores[0]


def test_newline_masked_when_line_is_too_narrow_for_active_spines(production_tokenizer):
    processor = _make_processor(production_tokenizer)

    scores = _scores_for_prefix(processor, production_tokenizer, "4c\t4e\t4g\n4c")

    assert scores[int(production_tokenizer.convert_tokens_to_ids("\n"))] == float("-inf")


def test_newline_and_eos_masked_immediately_after_tab(production_tokenizer):
    processor = _make_processor(production_tokenizer)

    scores = _scores_for_prefix(processor, production_tokenizer, "4c\t4e\n4c\t")

    assert scores[int(production_tokenizer.convert_tokens_to_ids("\n"))] == float("-inf")
    assert scores[int(production_tokenizer.eos_token_id)] == float("-inf")


def test_tab_masked_once_required_number_of_fields_is_reached(production_tokenizer):
    processor = _make_processor(production_tokenizer)

    scores = _scores_for_prefix(processor, production_tokenizer, "4c\t4e\n4g\t4a")

    assert scores[int(production_tokenizer.convert_tokens_to_ids("\t"))] == float("-inf")


def test_newline_masked_for_invalid_merge_pattern(production_tokenizer):
    processor = _make_processor(production_tokenizer)

    scores = _scores_for_prefix(processor, production_tokenizer, "4c\t4e\n*v\t*")

    assert scores[int(production_tokenizer.convert_tokens_to_ids("\n"))] == float("-inf")


def test_only_eos_remains_after_valid_terminating_line(production_tokenizer):
    processor = _make_processor(production_tokenizer)

    scores = _scores_for_prefix(processor, production_tokenizer, "4c\t4e\n*-\t*-\n")

    eos_token_id = int(production_tokenizer.eos_token_id)
    allowed = torch.nonzero(scores != float("-inf"), as_tuple=True)[0].tolist()
    assert allowed == [eos_token_id]
