"""Hard decode-time guard against degenerate identical-line loops."""

from __future__ import annotations

import torch
import transformers


class LineLoopStoppingCriteria(transformers.StoppingCriteria):
    """Stop sequences that emit the same complete line too many times in a row.

    Observed runaway validation decodes repeat one line 23-200 times until the
    generation cap while gold Polish scores top out at 5 consecutive identical
    lines (experiments/analysis/05roh5s9.md). Rows track line state
    incrementally from newly appended tokens, which assumes stable row
    identity across steps; the owning provider forces greedy decoding through
    has_generation_constraint to guarantee that.
    """

    def __init__(
        self,
        *,
        i2w: dict[int, str],
        max_identical_line_run: int,
        bos_token_id: int | None,
        eos_token_id: int | None,
        pad_token_id: int | None,
    ) -> None:
        if max_identical_line_run < 2:
            raise ValueError("max_identical_line_run must be >= 2")
        self._i2w = i2w
        self._max_run = int(max_identical_line_run)
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._consumed = 0
        self._line_buffers: list[list[str]] | None = None
        self._last_lines: list[str] = []
        self._run_lengths: list[int] = []
        self._stopped: list[bool] = []

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> torch.BoolTensor:
        batch_size = int(input_ids.shape[0])
        if self._line_buffers is None:
            self._line_buffers = [[] for _ in range(batch_size)]
            self._last_lines = [""] * batch_size
            self._run_lengths = [0] * batch_size
            self._stopped = [False] * batch_size
        elif batch_size != len(self._line_buffers):
            raise RuntimeError(
                "Expect input_ids.shape[0] to match LineLoopStoppingCriteria batch size. "
                f"Got {batch_size} and {len(self._line_buffers)}."
            )

        seq_len = int(input_ids.shape[1])
        for position in range(self._consumed, seq_len):
            column = input_ids[:, position].tolist()
            for row, token_id in enumerate(column):
                if not self._stopped[row]:
                    self._consume(row, int(token_id))
        self._consumed = seq_len

        return torch.tensor(self._stopped, device=input_ids.device, dtype=torch.bool)

    def _consume(self, row: int, token_id: int) -> None:
        if token_id in (self._bos_token_id, self._eos_token_id, self._pad_token_id):
            return
        token_text = self._i2w.get(token_id)
        if token_text is None:
            return
        assert self._line_buffers is not None
        if token_text != "\n":
            self._line_buffers[row].append(token_text)
            return

        line = "".join(self._line_buffers[row])
        self._line_buffers[row].clear()
        if line and line == self._last_lines[row]:
            self._run_lengths[row] += 1
        else:
            self._run_lengths[row] = 1
        self._last_lines[row] = line
        if self._run_lengths[row] >= self._max_run:
            self._stopped[row] = True
