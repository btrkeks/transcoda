"""HuggingFace logits processor with xgrammar constraints and PAD-safe behavior.

This wraps xgrammar matcher updates while tolerating PAD tokens that can appear
for sequences stopped externally (e.g. per-row stopping criteria) while other
rows in the same batch keep decoding.
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import TYPE_CHECKING

import torch
import transformers

if TYPE_CHECKING:
    import xgrammar as xgr


def _available_cpu_count() -> int:
    """Return the CPUs this process can actually schedule work on."""
    try:
        affinity_count = max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        affinity_count = max(1, os.cpu_count() or 1)

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            affinity_count = min(affinity_count, max(1, int(slurm_cpus)))
        except ValueError:
            pass
    return affinity_count


class GrammarConstrainedLogitsProcessor(transformers.LogitsProcessor):
    """Apply xgrammar token masks during generation.

    Compared to `xgrammar.contrib.hf.LogitsProcessor`, this variant accepts an
    optional `pad_token_id` and treats PAD on a non-terminated matcher as an
    external-finish signal for that row, instead of asserting.
    """

    def __init__(
        self,
        compiled_grammar: xgr.CompiledGrammar | list[xgr.CompiledGrammar],
        *,
        pad_token_id: int | None = None,
        collect_stats: bool = False,
    ) -> None:
        import xgrammar as xgr

        self._xgr = xgr
        self.matchers: list[xgr.GrammarMatcher] = []
        self._batch_matcher: xgr.BatchGrammarMatcher | None = None
        self._batch_matcher_threads = 0
        self.compiled_grammars: list[xgr.CompiledGrammar] = (
            compiled_grammar if isinstance(compiled_grammar, list) else [compiled_grammar]
        )
        self.full_vocab_size = self.compiled_grammars[0].tokenizer_info.vocab_size
        self.token_bitmask = None
        self._fork_bitmask = None
        self.prefilled = False
        self.batch_size = 0
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self._externally_finished_rows: list[bool] = []
        self.collect_stats = bool(collect_stats)
        self._stats = {
            "calls": 0,
            "rows_processed": 0,
            "matcher_state_advance_ms": 0.0,
            "bitmask_fill_ms": 0.0,
            "bitmask_apply_ms": 0.0,
            "externally_finished_rows": 0,
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._stats["calls"] += 1
        self._stats["rows_processed"] += int(input_ids.shape[0])
        # Lazily initialize matchers and bitmask on first call.
        if len(self.matchers) == 0:
            self.batch_size = int(input_ids.shape[0])
            self.compiled_grammars = (
                self.compiled_grammars
                if len(self.compiled_grammars) > 1
                else self.compiled_grammars * self.batch_size
            )
            if len(self.compiled_grammars) != self.batch_size:
                raise AssertionError(
                    "The number of compiled grammars must be equal to the batch size."
                )
            self.matchers = [
                self._xgr.GrammarMatcher(self.compiled_grammars[i]) for i in range(self.batch_size)
            ]
            self._batch_matcher_threads = min(self.batch_size, _available_cpu_count())
            self._batch_matcher = self._xgr.BatchGrammarMatcher(
                max_threads=self._batch_matcher_threads
            )
            self.token_bitmask = self._xgr.allocate_token_bitmask(
                self.batch_size, self.full_vocab_size
            )
            self._externally_finished_rows = [False] * self.batch_size

        if int(input_ids.shape[0]) != self.batch_size:
            raise RuntimeError(
                "Expect input_ids.shape[0] to be LogitsProcessor.batch_size."
                + f"Got {input_ids.shape[0]} for the former, and {self.batch_size} for the latter."
            )

        if not self.prefilled:
            self.prefilled = True
        else:
            advance_started_at = perf_counter() if self.collect_stats else 0.0
            active_rows: list[int] = []
            active_matchers: list[xgr.GrammarMatcher] = []
            sampled_tokens: list[int] = []
            latest_tokens = input_ids[:, -1].tolist()
            for i in range(self.batch_size):
                if self._externally_finished_rows[i] or self.matchers[i].is_terminated():
                    continue

                sampled_token = int(latest_tokens[i])
                if self.pad_token_id is not None and sampled_token == self.pad_token_id:
                    # HF generate can append PAD to rows that were stopped by external
                    # stopping criteria before grammar reaches a terminal state.
                    self._externally_finished_rows[i] = True
                    self._stats["externally_finished_rows"] += 1
                    continue

                active_rows.append(i)
                active_matchers.append(self.matchers[i])
                sampled_tokens.append(sampled_token)

            accepted = (
                self._xgr.BatchGrammarMatcher.batch_accept_token(active_matchers, sampled_tokens)
                if active_matchers
                else []
            )
            for i, sampled_token, was_accepted in zip(
                active_rows, sampled_tokens, accepted, strict=True
            ):
                if not was_accepted:
                    raise AssertionError(
                        f"Grammar matcher rejected sampled token id={sampled_token} at row={i}."
                    )
            if self.collect_stats:
                self._stats["matcher_state_advance_ms"] += (
                    perf_counter() - advance_started_at
                ) * 1000.0

        fill_started_at = perf_counter() if self.collect_stats else 0.0
        active_rows = []
        active_matchers = []
        for i in range(self.batch_size):
            if self._externally_finished_rows[i] or self.matchers[i].is_terminated():
                continue
            active_rows.append(i)
            active_matchers.append(self.matchers[i])
        if active_matchers:
            assert self._batch_matcher is not None
            self._batch_matcher.batch_fill_next_token_bitmask(
                active_matchers,
                self.token_bitmask,
                indices=active_rows,
            )
        if self.collect_stats:
            self._stats["bitmask_fill_ms"] += (perf_counter() - fill_started_at) * 1000.0

        # xgrammar supports masking on CUDA/CPU tensors.
        device_type = scores.device.type
        if device_type != "cuda":
            scores = scores.to("cpu")
        apply_started_at = perf_counter() if self.collect_stats else 0.0
        self._xgr.apply_token_bitmask_inplace(scores, self.token_bitmask.to(scores.device))
        if self.collect_stats:
            self._stats["bitmask_apply_ms"] += (perf_counter() - apply_started_at) * 1000.0
        if device_type != "cuda":
            scores = scores.to(device_type)

        if any(self._externally_finished_rows):
            finished_mask = torch.tensor(
                self._externally_finished_rows,
                device=scores.device,
                dtype=torch.bool,
            )
            if self.pad_token_id is not None and 0 <= self.pad_token_id < scores.shape[1]:
                scores[finished_mask] = float("-inf")
                scores[finished_mask, self.pad_token_id] = 0.0
            else:
                scores[finished_mask] = 0.0

        return scores

    def fork_row(self, row: int) -> xgr.GrammarMatcher | None:
        """Copy one active row's matcher for lookahead; None once the row has finished."""
        if not self.matchers:
            return None
        if self._externally_finished_rows[row] or self.matchers[row].is_terminated():
            return None
        return self.matchers[row].fork()

    def clone_fork(self, fork: xgr.GrammarMatcher) -> xgr.GrammarMatcher:
        return fork.fork()

    def advance_fork(self, fork: xgr.GrammarMatcher, token_id: int) -> bool:
        return bool(fork.accept_token(token_id))

    def mask_fork(self, fork: xgr.GrammarMatcher, row_scores: torch.FloatTensor) -> None:
        """Mask the tokens the grammar rejects after the forked prefix."""
        if self._fork_bitmask is None:
            self._fork_bitmask = self._xgr.allocate_token_bitmask(1, self.full_vocab_size)
        fork.fill_next_token_bitmask(self._fork_bitmask)
        self._xgr.apply_token_bitmask_inplace(
            row_scores.unsqueeze(0), self._fork_bitmask.to(row_scores.device)
        )

    def stats(self) -> dict[str, float | int]:
        return {
            **self._stats,
            "batch_matcher_threads": self._batch_matcher_threads,
            "total_ms": float(
                self._stats["matcher_state_advance_ms"]
                + self._stats["bitmask_fill_ms"]
                + self._stats["bitmask_apply_ms"]
            ),
        }
