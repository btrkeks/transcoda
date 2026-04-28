"""Runaway decoding diagnostics for validation/test monitoring.

This module provides lightweight per-sample heuristics to detect generation
runaways (length blow-up, repetition loops, and missing EOS at max length).
It also exposes reusable text-level probes for benchmark-only catastrophic
loop recovery.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


_VALID_STRICTNESS = {"lenient", "moderate", "strict"}

_STRICTNESS_PRESETS: dict[str, dict[str, float | int]] = {
    "lenient": {
        "max_len_ratio": 2.5,
        "repeat_ngram_size": 4,
        "repeat_ngram_max_occurrences": 8,
        "max_identical_line_run": 10,
    },
    "moderate": {
        "max_len_ratio": 1.8,
        "repeat_ngram_size": 4,
        "repeat_ngram_max_occurrences": 5,
        "max_identical_line_run": 6,
    },
    "strict": {
        "max_len_ratio": 1.4,
        "repeat_ngram_size": 4,
        "repeat_ngram_max_occurrences": 3,
        "max_identical_line_run": 4,
    },
}


@dataclass(frozen=True)
class RunawayMonitorConfig:
    """Thresholds controlling runaway detection sensitivity."""

    max_len_ratio: float
    repeat_ngram_size: int
    repeat_ngram_max_occurrences: int
    max_identical_line_run: int
    flag_no_eos_at_max_length: bool


@dataclass(frozen=True)
class RunawayTextDiagnostics:
    """Text-level repetition diagnostics for finalized output."""

    max_identical_line_run: int
    max_repeated_ngram_occurrences: int
    repeated_ngram_size: int
    repeated_ngram_line_coverage: float
    repeat_loop: bool
    repeat_loop_reason: str | None


@dataclass(frozen=True)
class CatastrophicLoopConfig:
    """Conservative thresholds for benchmark-only catastrophic loop recovery."""

    max_identical_line_run: int = 8
    repeated_ngram_size: int = 4
    min_repeated_ngram_occurrences: int = 12
    min_repeated_ngram_line_coverage: float = 0.33


@dataclass(frozen=True)
class CatastrophicLoopDiagnostics:
    """Benchmark-only catastrophic repetition diagnostics."""

    max_identical_line_run: int
    max_repeated_ngram_occurrences: int
    repeated_ngram_size: int
    repeated_ngram_line_coverage: float
    repeat_loop: bool
    repeat_loop_reason: str | None


@dataclass(frozen=True)
class RunawayMonitorSampleDiagnostics:
    """Per-sample runaway diagnostics."""

    is_runaway: bool
    length_ratio: float
    length_blowup: bool
    repeat_loop: bool
    no_eos_at_max_length: bool
    max_length_hit: bool


class RunawayTextProbe:
    """Reusable text-level repetition probe."""

    def __init__(
        self,
        *,
        repeat_ngram_size: int,
        repeat_ngram_max_occurrences: int,
        max_identical_line_run: int,
    ) -> None:
        self._repeat_ngram_size = int(repeat_ngram_size)
        self._repeat_ngram_max_occurrences = int(repeat_ngram_max_occurrences)
        self._max_identical_line_run = int(max_identical_line_run)

    @classmethod
    def from_monitor_config(cls, config: RunawayMonitorConfig) -> "RunawayTextProbe":
        return cls(
            repeat_ngram_size=config.repeat_ngram_size,
            repeat_ngram_max_occurrences=config.repeat_ngram_max_occurrences,
            max_identical_line_run=config.max_identical_line_run,
        )

    def analyze_text(self, text: str) -> RunawayTextDiagnostics:
        lines = [line for line in text.split("\n") if line]
        if not lines:
            return RunawayTextDiagnostics(
                max_identical_line_run=0,
                max_repeated_ngram_occurrences=0,
                repeated_ngram_size=self._repeat_ngram_size,
                repeated_ngram_line_coverage=0.0,
                repeat_loop=False,
                repeat_loop_reason=None,
            )

        run_len = 1
        max_run = 1
        for prev, current in zip(lines, lines[1:], strict=False):
            if current == prev:
                run_len += 1
                if run_len > max_run:
                    max_run = run_len
            else:
                run_len = 1

        max_occurrences = 0
        n = self._repeat_ngram_size
        if len(lines) >= n:
            occurrences: dict[tuple[str, ...], int] = defaultdict(int)
            for idx in range(len(lines) - n + 1):
                ngram = tuple(lines[idx : idx + n])
                occurrences[ngram] += 1
                if occurrences[ngram] > max_occurrences:
                    max_occurrences = occurrences[ngram]

        coverage = float(max_occurrences) / float(len(lines)) if lines else 0.0
        repeat_loop = False
        repeat_loop_reason: str | None = None
        if max_run >= self._max_identical_line_run:
            repeat_loop = True
            repeat_loop_reason = "identical_line_run"
        elif max_occurrences >= self._repeat_ngram_max_occurrences:
            repeat_loop = True
            repeat_loop_reason = "repeated_ngram"

        return RunawayTextDiagnostics(
            max_identical_line_run=max_run,
            max_repeated_ngram_occurrences=max_occurrences,
            repeated_ngram_size=n,
            repeated_ngram_line_coverage=coverage,
            repeat_loop=repeat_loop,
            repeat_loop_reason=repeat_loop_reason,
        )


def resolve_runaway_monitor_config(training_cfg: Any) -> RunawayMonitorConfig:
    """Resolve strictness preset + optional explicit monitor overrides."""
    strictness = str(getattr(training_cfg, "runaway_monitor_strictness", "moderate"))
    if strictness not in _VALID_STRICTNESS:
        raise ValueError(
            f"Unsupported runaway_monitor_strictness '{strictness}'. "
            f"Expected one of {sorted(_VALID_STRICTNESS)}."
        )

    resolved = dict(_STRICTNESS_PRESETS[strictness])

    max_len_ratio = getattr(training_cfg, "runaway_monitor_max_len_ratio", None)
    if max_len_ratio is not None:
        resolved["max_len_ratio"] = float(max_len_ratio)

    repeat_ngram_size = getattr(training_cfg, "runaway_monitor_repeat_ngram_size", None)
    if repeat_ngram_size is not None:
        resolved["repeat_ngram_size"] = int(repeat_ngram_size)

    repeat_ngram_max_occurrences = getattr(
        training_cfg, "runaway_monitor_repeat_ngram_max_occurrences", None
    )
    if repeat_ngram_max_occurrences is not None:
        resolved["repeat_ngram_max_occurrences"] = int(repeat_ngram_max_occurrences)

    max_identical_line_run = getattr(training_cfg, "runaway_monitor_max_identical_line_run", None)
    if max_identical_line_run is not None:
        resolved["max_identical_line_run"] = int(max_identical_line_run)

    flag_no_eos = bool(getattr(training_cfg, "runaway_monitor_flag_no_eos_at_max_length", True))

    return RunawayMonitorConfig(
        max_len_ratio=float(resolved["max_len_ratio"]),
        repeat_ngram_size=int(resolved["repeat_ngram_size"]),
        repeat_ngram_max_occurrences=int(resolved["repeat_ngram_max_occurrences"]),
        max_identical_line_run=int(resolved["max_identical_line_run"]),
        flag_no_eos_at_max_length=flag_no_eos,
    )


def analyze_catastrophic_repetition(
    text: str,
    config: CatastrophicLoopConfig | None = None,
) -> CatastrophicLoopDiagnostics:
    """Analyze finalized text with a conservative catastrophic-loop policy."""
    resolved = config or CatastrophicLoopConfig()
    diagnostics = RunawayTextProbe(
        repeat_ngram_size=resolved.repeated_ngram_size,
        repeat_ngram_max_occurrences=resolved.min_repeated_ngram_occurrences,
        max_identical_line_run=resolved.max_identical_line_run,
    ).analyze_text(text)

    repeat_loop = False
    repeat_loop_reason: str | None = None
    if diagnostics.max_identical_line_run >= resolved.max_identical_line_run:
        repeat_loop = True
        repeat_loop_reason = "identical_line_run"
    elif (
        diagnostics.max_repeated_ngram_occurrences >= resolved.min_repeated_ngram_occurrences
        and diagnostics.repeated_ngram_size == resolved.repeated_ngram_size
        and diagnostics.repeated_ngram_line_coverage >= resolved.min_repeated_ngram_line_coverage
    ):
        repeat_loop = True
        repeat_loop_reason = "repeated_ngram"

    return CatastrophicLoopDiagnostics(
        max_identical_line_run=diagnostics.max_identical_line_run,
        max_repeated_ngram_occurrences=diagnostics.max_repeated_ngram_occurrences,
        repeated_ngram_size=diagnostics.repeated_ngram_size,
        repeated_ngram_line_coverage=diagnostics.repeated_ngram_line_coverage,
        repeat_loop=repeat_loop,
        repeat_loop_reason=repeat_loop_reason,
    )


class RunawayMonitorTracker:
    """Accumulates runaway diagnostics and computes aggregate metrics."""

    def __init__(
        self,
        *,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        i2w: dict[int, str],
        config: RunawayMonitorConfig,
    ) -> None:
        self._pad_id = pad_id
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._i2w = i2w
        self._config = config
        self._text_probe = RunawayTextProbe.from_monitor_config(config)
        self.reset()

    def reset(self) -> None:
        self._total_samples = 0
        self._runaway_count = 0
        self._length_blowup_count = 0
        self._repeat_loop_count = 0
        self._no_eos_at_max_length_count = 0
        self._max_length_hit_count = 0
        self._len_ratios: list[float] = []

    def _trim_ids(self, seq: list[int]) -> tuple[list[int], bool]:
        out: list[int] = []
        has_eos = False

        for token_id in seq:
            if token_id == self._pad_id:
                break
            if token_id == self._bos_id:
                continue
            if token_id == self._eos_id:
                has_eos = True
                break
            if token_id < 0:
                continue
            out.append(int(token_id))

        return out, has_eos

    def _generated_length_before_pad(self, seq: list[int]) -> int:
        """Count generated tokens up to first PAD (matches HF max_length semantics)."""
        count = 0
        for token_id in seq:
            if token_id == self._pad_id:
                break
            count += 1
        return count

    def _ids_to_text(self, seq: list[int]) -> str:
        tokens: list[str] = []
        for token_id in seq:
            token = self._i2w.get(token_id, "")
            if not token or token in {"<bos>", "<eos>", "<pad>"}:
                continue
            tokens.append(token)
        return "".join(tokens)

    def analyze_sample(
        self,
        *,
        pred_ids: list[int],
        target_ids: list[int],
        max_length_cap: int,
    ) -> RunawayMonitorSampleDiagnostics:
        """Compute runaway diagnostics for a single prediction/target pair."""
        pred_trimmed, has_eos = self._trim_ids(pred_ids)
        target_trimmed, _ = self._trim_ids(target_ids)

        target_len = max(1, len(target_trimmed))
        pred_len = len(pred_trimmed)
        length_ratio = float(pred_len) / float(target_len)
        length_blowup = length_ratio >= self._config.max_len_ratio

        generated_len = self._generated_length_before_pad(pred_ids)
        max_length_hit = (not has_eos) and generated_len >= int(max_length_cap)
        no_eos_at_max_length = self._config.flag_no_eos_at_max_length and max_length_hit

        pred_text = self._ids_to_text(pred_trimmed)
        repeat_loop = self._text_probe.analyze_text(pred_text).repeat_loop

        is_runaway = length_blowup or repeat_loop or no_eos_at_max_length
        return RunawayMonitorSampleDiagnostics(
            is_runaway=is_runaway,
            length_ratio=length_ratio,
            length_blowup=length_blowup,
            repeat_loop=repeat_loop,
            no_eos_at_max_length=no_eos_at_max_length,
            max_length_hit=max_length_hit,
        )

    def update_batch(
        self, preds, targets, *, max_length_cap: int
    ) -> list[RunawayMonitorSampleDiagnostics]:
        """Update aggregate state from a batch of predictions and targets."""
        pred_list = preds.detach().cpu().tolist()
        target_list = targets.detach().cpu().tolist()
        diagnostics = []
        for pred_ids, target_ids in zip(pred_list, target_list, strict=False):
            diag = self.analyze_sample(
                pred_ids=pred_ids,
                target_ids=target_ids,
                max_length_cap=max_length_cap,
            )
            diagnostics.append(diag)

            self._total_samples += 1
            self._len_ratios.append(diag.length_ratio)

            if diag.is_runaway:
                self._runaway_count += 1
            if diag.length_blowup:
                self._length_blowup_count += 1
            if diag.repeat_loop:
                self._repeat_loop_count += 1
            if diag.no_eos_at_max_length:
                self._no_eos_at_max_length_count += 1
            if diag.max_length_hit:
                self._max_length_hit_count += 1
        return diagnostics

    @staticmethod
    def _p95(values: list[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * 0.95) - 1))
        return float(sorted_values[index])

    def compute(self) -> dict[str, float | int]:
        """Compute aggregate runaway metrics."""
        if self._total_samples == 0:
            return {
                "runaway_rate": 0.0,
                "runaway_count": 0,
                "runaway_samples": 0,
                "runaway_length_blowup_rate": 0.0,
                "runaway_repeat_loop_rate": 0.0,
                "runaway_no_eos_at_max_length_rate": 0.0,
                "runaway_max_length_hit_rate": 0.0,
                "runaway_len_ratio_p95": 0.0,
            }

        total = float(self._total_samples)

        return {
            "runaway_rate": 100.0 * float(self._runaway_count) / total,
            "runaway_count": int(self._runaway_count),
            "runaway_samples": int(self._total_samples),
            "runaway_length_blowup_rate": 100.0 * float(self._length_blowup_count) / total,
            "runaway_repeat_loop_rate": 100.0 * float(self._repeat_loop_count) / total,
            "runaway_no_eos_at_max_length_rate": 100.0
            * float(self._no_eos_at_max_length_count)
            / total,
            "runaway_max_length_hit_rate": 100.0 * float(self._max_length_hit_count) / total,
            "runaway_len_ratio_p95": self._p95(self._len_ratios),
        }
