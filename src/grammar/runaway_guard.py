"""Runaway decoding guardrails for grammar-constrained generation.

This module adds a secondary HuggingFace logits processor that masks
high-risk control tokens when repeated/control-heavy generation patterns
suggest a degenerative loop.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from transformers import LogitsProcessor

_VALID_STRICTNESS = {"lenient", "moderate", "strict"}


@dataclass(frozen=True)
class RunawayGuardConfig:
    """Thresholds used by the runaway-breaker logits processor."""

    max_same_control_token: int
    max_control_lines_streak: int
    max_spine_splits: int
    max_spine_merges: int
    max_ottava_markers: int
    max_tuplet_markers: int
    max_tremolo_markers: int


@dataclass
class _SequenceState:
    processed_len: int = 0
    last_control_token_id: int | None = None
    same_control_run_len: int = 0
    control_line_streak: int = 0
    line_buffer: str = ""
    class_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


_STRICTNESS_PRESETS: dict[str, dict[str, int]] = {
    "lenient": {
        "max_same_control_token": 12,
        "max_control_lines_streak": 10,
        "max_spine_splits": 96,
        "max_spine_merges": 128,
        "max_ottava_markers": 32,
        "max_tuplet_markers": 24,
        "max_tremolo_markers": 24,
    },
    "moderate": {
        "max_same_control_token": 8,
        "max_control_lines_streak": 6,
        "max_spine_splits": 64,
        "max_spine_merges": 96,
        "max_ottava_markers": 16,
        "max_tuplet_markers": 12,
        "max_tremolo_markers": 12,
    },
    "strict": {
        "max_same_control_token": 5,
        "max_control_lines_streak": 4,
        "max_spine_splits": 40,
        "max_spine_merges": 64,
        "max_ottava_markers": 10,
        "max_tuplet_markers": 8,
        "max_tremolo_markers": 8,
    },
}

_OVERRIDE_FIELDS: tuple[str, ...] = (
    "runaway_guard_max_same_control_token",
    "runaway_guard_max_control_lines_streak",
    "runaway_guard_max_spine_splits",
    "runaway_guard_max_spine_merges",
    "runaway_guard_max_ottava_markers",
    "runaway_guard_max_tuplet_markers",
    "runaway_guard_max_tremolo_markers",
)


def resolve_runaway_guard_config(training_cfg: Any) -> RunawayGuardConfig:
    """Resolve strictness preset + optional explicit overrides."""
    strictness = str(getattr(training_cfg, "runaway_guard_strictness", "moderate"))
    if strictness not in _VALID_STRICTNESS:
        raise ValueError(
            f"Unsupported runaway_guard_strictness '{strictness}'. "
            f"Expected one of {sorted(_VALID_STRICTNESS)}."
        )

    resolved = dict(_STRICTNESS_PRESETS[strictness])
    for field_name in _OVERRIDE_FIELDS:
        value = getattr(training_cfg, field_name, None)
        if value is None:
            continue
        resolved[field_name.replace("runaway_guard_", "")] = int(value)

    return RunawayGuardConfig(**resolved)


class RunawayBreakerLogitsProcessor(LogitsProcessor):
    """Mask runaway-prone control markers during generation.

    The processor is intentionally conservative:
    - it only masks control tokens (never content tokens),
    - it never masks EOS/BOS,
    - if parsing state becomes inconsistent it falls back to no-op for safety.
    """

    _SPINE_SPLIT_MARKERS = {"*^"}
    _SPINE_MERGE_MARKERS = {"*v"}
    _OTTAVA_MARKERS = {
        "*8va",
        "*X8va",
        "*8ba",
        "*X8ba",
        "*15va",
        "*X15va",
        "*15ba",
        "*X15ba",
        "*15ma",
        "*X15ma",
    }
    _TUPLET_MARKERS = {"*Xtuplet", "*tuplet"}
    _TREMOLO_MARKERS = {"*tremolo", "*Xtremolo"}

    def __init__(
        self,
        tokenizer_i2w: dict[int, str],
        bos_token_id: int,
        eos_token_id: int,
        config: RunawayGuardConfig,
    ) -> None:
        self._i2w = tokenizer_i2w
        self._bos_token_id = int(bos_token_id)
        self._eos_token_id = int(eos_token_id)
        self._config = config

        self._marker_to_class: dict[str, str] = {}
        for marker in self._SPINE_SPLIT_MARKERS:
            self._marker_to_class[marker] = "spine_splits"
        for marker in self._SPINE_MERGE_MARKERS:
            self._marker_to_class[marker] = "spine_merges"
        for marker in self._OTTAVA_MARKERS:
            self._marker_to_class[marker] = "ottava_markers"
        for marker in self._TUPLET_MARKERS:
            self._marker_to_class[marker] = "tuplet_markers"
        for marker in self._TREMOLO_MARKERS:
            self._marker_to_class[marker] = "tremolo_markers"

        self._token_ids_by_marker: dict[str, set[int]] = defaultdict(set)
        self._token_ids_by_class: dict[str, set[int]] = defaultdict(set)
        self._all_control_token_ids: set[int] = set()
        self._star_token_ids: set[int] = set()

        for token_id, token_text in self._i2w.items():
            marker = token_text.strip()
            if marker == "*":
                self._star_token_ids.add(int(token_id))
            if marker not in self._marker_to_class:
                continue

            tid = int(token_id)
            klass = self._marker_to_class[marker]
            self._token_ids_by_marker[marker].add(tid)
            self._token_ids_by_class[klass].add(tid)
            self._all_control_token_ids.add(tid)

        self._state_by_batch_index: dict[int, _SequenceState] = {}

    def _get_or_reset_state(self, batch_index: int, seq_len: int) -> _SequenceState:
        state = self._state_by_batch_index.get(batch_index)
        if state is None:
            state = _SequenceState()
            self._state_by_batch_index[batch_index] = state
            return state

        # Defensive reset if sequence is truncated/restarted.
        if seq_len < state.processed_len:
            state = _SequenceState()
            self._state_by_batch_index[batch_index] = state
        return state

    def _process_line(self, line: str, state: _SequenceState) -> None:
        line_fields = [field.strip() for field in line.split("\t")]
        if not line_fields:
            state.control_line_streak = 0
            return

        has_control_marker = False
        control_only = True

        for value in line_fields:
            if not value:
                continue
            if value == "*":
                continue

            marker_class = self._marker_to_class.get(value)
            if marker_class is None:
                control_only = False
                continue

            has_control_marker = True
            state.class_counts[marker_class] += 1

        if has_control_marker and control_only:
            state.control_line_streak += 1
        else:
            state.control_line_streak = 0

    def _ingest_token(self, token_id: int, state: _SequenceState) -> None:
        token_text = self._i2w.get(int(token_id), "")
        if int(token_id) in {self._bos_token_id, self._eos_token_id}:
            return
        if token_text in {"<bos>", "<eos>", "<pad>"}:
            return

        if int(token_id) in self._all_control_token_ids:
            if state.last_control_token_id == int(token_id):
                state.same_control_run_len += 1
            else:
                state.last_control_token_id = int(token_id)
                state.same_control_run_len = 1
        else:
            state.last_control_token_id = None
            state.same_control_run_len = 0

        if not token_text:
            return

        state.line_buffer += token_text

        # Keep incomplete line buffer bounded if no newline appears for a while.
        if len(state.line_buffer) > 4096 and "\n" not in state.line_buffer:
            state.line_buffer = state.line_buffer[-1024:]

        while "\n" in state.line_buffer:
            line, remainder = state.line_buffer.split("\n", 1)
            state.line_buffer = remainder
            self._process_line(line.rstrip("\r"), state)

    def _blocked_ids_for_state(self, state: _SequenceState) -> set[int]:
        blocked_ids: set[int] = set()

        if (
            state.last_control_token_id is not None
            and state.same_control_run_len >= self._config.max_same_control_token
        ):
            blocked_ids.add(state.last_control_token_id)

        threshold_map = {
            "spine_splits": self._config.max_spine_splits,
            "spine_merges": self._config.max_spine_merges,
            "ottava_markers": self._config.max_ottava_markers,
            "tuplet_markers": self._config.max_tuplet_markers,
            "tremolo_markers": self._config.max_tremolo_markers,
        }
        for klass, threshold in threshold_map.items():
            if state.class_counts.get(klass, 0) >= threshold:
                blocked_ids.update(self._token_ids_by_class.get(klass, set()))

        if state.control_line_streak >= self._config.max_control_lines_streak:
            blocked_ids.update(self._all_control_token_ids)

        # Never block special IDs.
        blocked_ids.discard(self._bos_token_id)
        blocked_ids.discard(self._eos_token_id)

        return blocked_ids

    def __call__(self, input_ids, scores):
        if input_ids.ndim != 2 or scores.ndim != 2:
            return scores

        batch_size = input_ids.shape[0]
        for batch_index in range(batch_size):
            try:
                seq_len = int(input_ids[batch_index].shape[0])
                state = self._get_or_reset_state(batch_index, seq_len)

                if seq_len > state.processed_len:
                    new_ids = input_ids[batch_index, state.processed_len:seq_len].tolist()
                    for token_id in new_ids:
                        self._ingest_token(int(token_id), state)
                    state.processed_len = seq_len

                blocked_ids = self._blocked_ids_for_state(state)
                if blocked_ids:
                    # Keep null interpretation '*' available even when control-line streak triggers.
                    blocked_ids.difference_update(self._star_token_ids)
                    if blocked_ids:
                        scores[batch_index, list(blocked_ids)] = float("-inf")
            except Exception:
                # Safety fallback: never break generation from guardrail errors.
                continue

        return scores
