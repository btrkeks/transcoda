"""Keep greedy constrained decoding out of prefixes that no token can continue.

The grammar and the spine rules check each token against the characters it
appends, not against what the vocabulary can spell next. After ``*`` the
grammar allows the token ``k``, because ``*k`` starts ``*k[...]``, but the
vocabulary has no token that starts with ``[``. A model that wants ``*kcancel``
picks ``k`` and the next step has nothing left to sample.

This processor runs last. For each row it takes the token greedy decoding is
about to pick and checks, on forked constraint state, that some continuation
reaches the end of the field. If none does, it masks that token and checks the
next best one. A token that ends a field (tab, newline, EOS) is accepted as
soon as every constraint allows it. When the search budget runs out the token
is kept, so a token is only ever masked when it provably dead-ends.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
import transformers


@runtime_checkable
class ForkableConstraint(Protocol):
    """A per-row token mask whose row state can be copied and advanced off-line."""

    def fork_row(self, row: int) -> object | None: ...

    def clone_fork(self, fork: object) -> object: ...

    def advance_fork(self, fork: object, token_id: int) -> bool: ...

    def mask_fork(self, fork: object, row_scores: torch.FloatTensor) -> None: ...


class DeadEndLookaheadLogitsProcessor(transformers.LogitsProcessor):
    """Mask greedy picks from which every continuation is masked."""

    def __init__(
        self,
        constraints: Sequence[ForkableConstraint],
        *,
        field_end_token_ids: Sequence[int],
        pad_token_id: int | None,
        max_depth: int = 8,
        max_expansions: int = 64,
    ) -> None:
        if not constraints:
            raise ValueError("DeadEndLookaheadLogitsProcessor needs at least one constraint")
        self._constraints = list(constraints)
        self._field_end_token_ids = frozenset(int(token_id) for token_id in field_end_token_ids)
        self._pad_token_id = pad_token_id
        self._max_depth = int(max_depth)
        self._max_expansions = int(max_expansions)
        self._vocab_size = 0
        self._stats = {"masked_dead_tokens": 0, "budget_exhausted": 0, "expansions": 0}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._vocab_size = int(scores.shape[1])
        for row in range(int(scores.shape[0])):
            self._guard_row(row, scores[row])
        return scores

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _guard_row(self, row: int, row_scores: torch.FloatTensor) -> None:
        forks: list[object] | None = None
        while True:
            token_id = int(torch.argmax(row_scores).item())
            if not torch.isfinite(row_scores[token_id]) or token_id == self._pad_token_id:
                return
            if token_id in self._field_end_token_ids:
                return
            if forks is None:
                forks = [constraint.fork_row(row) for constraint in self._constraints]
                if any(fork is None for fork in forks):
                    return
            budget = [self._max_expansions]
            if self._can_finish_field(forks, token_id, self._max_depth, budget):
                return
            row_scores[token_id] = float("-inf")
            self._stats["masked_dead_tokens"] += 1

    def _can_finish_field(
        self,
        forks: list[object],
        token_id: int,
        depth: int,
        budget: list[int],
    ) -> bool:
        children = [
            constraint.clone_fork(fork)
            for constraint, fork in zip(self._constraints, forks, strict=True)
        ]
        for constraint, child in zip(self._constraints, children, strict=True):
            if not constraint.advance_fork(child, token_id):
                return False
        if depth <= 0 or budget[0] <= 0:
            self._stats["budget_exhausted"] += 1
            return True
        budget[0] -= 1
        self._stats["expansions"] += 1

        allowed = self._allowed_after(children)
        if not allowed.isdisjoint(self._field_end_token_ids):
            return True
        return any(
            self._can_finish_field(children, next_id, depth - 1, budget)
            for next_id in sorted(allowed)
            if next_id != self._pad_token_id
        )

    def _allowed_after(self, forks: list[object]) -> set[int]:
        scores = torch.zeros(self._vocab_size, dtype=torch.float32)
        for constraint, fork in zip(self._constraints, forks, strict=True):
            constraint.mask_fork(fork, scores)
        return set(torch.isfinite(scores).nonzero().flatten().tolist())
