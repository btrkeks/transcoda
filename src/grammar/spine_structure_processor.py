"""Compatibility wrapper around the shared stateful **kern processor."""

from __future__ import annotations

import torch
import transformers

from .spine_structure_rule import SpineStructureRule
from .stateful_kern_logits_processor import StatefulKernLogitsProcessor


class SpineStructureLogitsProcessor(transformers.LogitsProcessor):
    """Mask tokens that would violate context-dependent spine invariants."""

    def __init__(
        self,
        *,
        i2w: dict[int, str],
        bos_token_id: int | None,
        eos_token_id: int | None,
        pad_token_id: int | None,
    ) -> None:
        self._delegate = StatefulKernLogitsProcessor(
            i2w=i2w,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            rule_factories=[SpineStructureRule],
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self._delegate(input_ids, scores)
