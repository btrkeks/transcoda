"""Shared stateful logits processor for semantic **kern decoding rules."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter
from typing import Any, Protocol

import torch
import transformers

from .kern_prefix_state import KernPrefixState, KernPrefixStateError


class KernSemanticRule(Protocol):
    """Per-sequence semantic rule used by StatefulKernLogitsProcessor."""

    @property
    def terminated(self) -> bool:
        ...

    def on_text_appended(self, prefix_state: KernPrefixState) -> None:
        ...

    def on_tab_appended(self, prefix_state: KernPrefixState) -> None:
        ...

    def on_line_closed(self, fields: tuple[str, ...]) -> None:
        ...

    def can_accept_tab(self, prefix_state: KernPrefixState) -> bool:
        ...

    def can_close_line(self, fields: tuple[str, ...]) -> bool:
        ...

    def can_end_sequence(self, fields: tuple[str, ...]) -> bool:
        ...

    def mask_scores(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: "TokenizerConstraintContext",
    ) -> None:
        ...


SemanticRuleFactory = Callable[[], KernSemanticRule]


@dataclass(frozen=True)
class RhythmTokenMetadata:
    """Precomputed token buckets used by rhythm-aware masking."""

    data_token_ids: list[int]
    non_grace_data_token_ids: list[int]
    duration_start_token_ids: list[int]
    non_grace_token_ids_by_sig_class: dict[str, list[int]]
    token_ids_by_sig_class: dict[str, list[int]]
    parsed_duration_token_ids: dict[Fraction, list[int]]


@dataclass(frozen=True)
class InterpretationTokenMetadata:
    """Precomputed token buckets used by interpretation-transition biasing."""

    interpretation_token_ids: list[int]
    spine_op_token_ids: list[int]
    non_spine_interp_token_ids: list[int]
    null_interpretation_token_ids: list[int]
    clef_token_ids: list[int]
    barline_token_ids: list[int]
    non_control_data_token_ids: list[int]


@dataclass(frozen=True)
class TokenizerConstraintContext:
    """Tokenizer metadata shared by semantic logits processors."""

    i2w: dict[int, str]
    bos_token_id: int | None
    eos_token_id: int
    pad_token_id: int | None
    tab_token_id: int
    newline_token_id: int
    all_token_ids: tuple[int, ...]
    rhythm: RhythmTokenMetadata
    interpretation: InterpretationTokenMetadata

    @classmethod
    def from_i2w(
        cls,
        *,
        i2w: dict[int, str],
        bos_token_id: int | None,
        eos_token_id: int | None,
        pad_token_id: int | None,
    ) -> "TokenizerConstraintContext":
        normalized = {int(token_id): text for token_id, text in i2w.items()}
        if eos_token_id is None:
            raise ValueError("StatefulKernLogitsProcessor requires eos_token_id")
        tab_token_id = _find_unique_token_id(normalized, "\t")
        newline_token_id = _find_unique_token_id(normalized, "\n")
        _validate_token_texts(normalized.values())
        pad_id = None if pad_token_id is None else int(pad_token_id)
        bos_id = None if bos_token_id is None else int(bos_token_id)
        eos_id = int(eos_token_id)
        return cls(
            i2w=normalized,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            tab_token_id=tab_token_id,
            newline_token_id=newline_token_id,
            all_token_ids=tuple(sorted(normalized)),
            rhythm=_build_rhythm_token_metadata(
                normalized,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                tab_token_id=tab_token_id,
                newline_token_id=newline_token_id,
            ),
            interpretation=_build_interpretation_token_metadata(
                normalized,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                tab_token_id=tab_token_id,
                newline_token_id=newline_token_id,
            ),
        )


def _find_unique_token_id(i2w: dict[int, str], token_text: str) -> int:
    matches = [token_id for token_id, text in i2w.items() if text == token_text]
    if not matches:
        raise ValueError(f"Tokenizer is missing dedicated {token_text!r} token.")
    if len(matches) > 1:
        raise ValueError(f"Tokenizer has multiple dedicated {token_text!r} tokens: {matches}")
    return int(matches[0])


def _validate_token_texts(token_texts: Iterable[str]) -> None:
    invalid = []
    for token in token_texts:
        if token in {"\t", "\n"}:
            continue
        if "\t" in token or "\n" in token:
            invalid.append(token)
    if invalid:
        preview = ", ".join(repr(token) for token in invalid[:5])
        raise ValueError(
            "Tokenizer contains tokens with embedded tabs/newlines, which is unsupported: "
            f"{preview}"
        )


def _token_sig_char_class(token_text: str) -> str:
    stripped = token_text.strip().lstrip("[](){}_")
    if not stripped:
        return "empty"
    first = stripped[0]
    if first.isdigit():
        return "digit"
    if first == ".":
        return "dot"
    if first == "%":
        return "percent"
    return "other"


def _parse_duration_value_for_metadata(token_text: str) -> Fraction | None:
    token = token_text.strip()
    if token == ".":
        return None
    if "q" in token or "Q" in token:
        return None
    if token.startswith("=") or token.startswith("*") or token.startswith("!"):
        return None
    if " " in token:
        first = token.split(" ")[0]
        return _parse_duration_value_for_metadata(first)

    stripped = token.lstrip("[](){}_")
    number = []
    index = 0
    while index < len(stripped) and stripped[index].isdigit():
        number.append(stripped[index])
        index += 1
    if not number:
        return None
    numerator = int("".join(number))
    if numerator == 0:
        return Fraction(0, 1)

    if index < len(stripped) and stripped[index] == "%":
        index += 1
        denom_digits = []
        while index < len(stripped) and stripped[index].isdigit():
            denom_digits.append(stripped[index])
            index += 1
        if not denom_digits:
            return None
        denominator = int("".join(denom_digits))
        if denominator == 0:
            return None
        dots = 0
        while index < len(stripped) and stripped[index] == ".":
            dots += 1
            index += 1
        base = Fraction(numerator, denominator)
        return _apply_duration_dots_for_metadata(base, dots)

    dots = 0
    while index < len(stripped) and stripped[index] == ".":
        dots += 1
        index += 1
    base = Fraction(1, numerator)
    return _apply_duration_dots_for_metadata(base, dots)


def _apply_duration_dots_for_metadata(base: Fraction, dots: int) -> Fraction:
    total = base
    current = base
    for _ in range(dots):
        current /= 2
        total += current
    return total


def _starts_duration_candidate_for_metadata(token_text: str) -> bool:
    stripped = token_text.strip().lstrip("[](){}_")
    return bool(stripped) and stripped[0].isdigit()


def _build_rhythm_token_metadata(
    i2w: dict[int, str],
    *,
    bos_token_id: int | None,
    eos_token_id: int,
    pad_token_id: int | None,
    tab_token_id: int,
    newline_token_id: int,
) -> RhythmTokenMetadata:
    excluded_ids = {eos_token_id, tab_token_id, newline_token_id}
    if bos_token_id is not None:
        excluded_ids.add(bos_token_id)
    if pad_token_id is not None:
        excluded_ids.add(pad_token_id)

    data_token_ids: list[int] = []
    non_grace_data_token_ids: list[int] = []
    duration_start_token_ids: list[int] = []
    token_ids_by_sig_class: dict[str, list[int]] = {
        "digit": [],
        "dot": [],
        "percent": [],
        "other": [],
        "empty": [],
    }
    non_grace_token_ids_by_sig_class: dict[str, list[int]] = {
        "digit": [],
        "dot": [],
        "percent": [],
        "other": [],
        "empty": [],
    }
    parsed_duration_token_ids: dict[Fraction, list[int]] = {}

    for token_id, token_text in sorted(i2w.items()):
        if token_id in excluded_ids:
            continue
        if token_text.startswith("*") or token_text.startswith("="):
            continue

        data_token_ids.append(token_id)
        sig_class = _token_sig_char_class(token_text)
        token_ids_by_sig_class[sig_class].append(token_id)

        if _starts_duration_candidate_for_metadata(token_text):
            duration_start_token_ids.append(token_id)

        is_grace = ("q" in token_text) or ("Q" in token_text)
        if not is_grace:
            non_grace_data_token_ids.append(token_id)
            non_grace_token_ids_by_sig_class[sig_class].append(token_id)

        duration = _parse_duration_value_for_metadata(token_text)
        if duration is not None:
            parsed_duration_token_ids.setdefault(duration, []).append(token_id)

    return RhythmTokenMetadata(
        data_token_ids=data_token_ids,
        non_grace_data_token_ids=non_grace_data_token_ids,
        duration_start_token_ids=duration_start_token_ids,
        non_grace_token_ids_by_sig_class=non_grace_token_ids_by_sig_class,
        token_ids_by_sig_class=token_ids_by_sig_class,
        parsed_duration_token_ids=parsed_duration_token_ids,
    )


def _build_interpretation_token_metadata(
    i2w: dict[int, str],
    *,
    bos_token_id: int | None,
    eos_token_id: int,
    pad_token_id: int | None,
    tab_token_id: int,
    newline_token_id: int,
) -> InterpretationTokenMetadata:
    excluded_ids = {eos_token_id, tab_token_id, newline_token_id}
    if bos_token_id is not None:
        excluded_ids.add(bos_token_id)
    if pad_token_id is not None:
        excluded_ids.add(pad_token_id)

    interpretation_token_ids: list[int] = []
    spine_op_token_ids: list[int] = []
    non_spine_interp_token_ids: list[int] = []
    null_interpretation_token_ids: list[int] = []
    clef_token_ids: list[int] = []
    barline_token_ids: list[int] = []
    non_control_data_token_ids: list[int] = []
    spine_op_tokens = {"*", "*^", "*v", "*-"}

    for token_id, token_text in sorted(i2w.items()):
        if token_id in excluded_ids:
            continue

        token = token_text.strip()
        if token.startswith("*"):
            interpretation_token_ids.append(token_id)
            if token in spine_op_tokens:
                spine_op_token_ids.append(token_id)
            else:
                non_spine_interp_token_ids.append(token_id)
            if token == "*":
                null_interpretation_token_ids.append(token_id)
            if token.startswith("*clef"):
                clef_token_ids.append(token_id)
            continue

        if token.startswith("="):
            barline_token_ids.append(token_id)
            continue

        non_control_data_token_ids.append(token_id)

    return InterpretationTokenMetadata(
        interpretation_token_ids=interpretation_token_ids,
        spine_op_token_ids=spine_op_token_ids,
        non_spine_interp_token_ids=non_spine_interp_token_ids,
        null_interpretation_token_ids=null_interpretation_token_ids,
        clef_token_ids=clef_token_ids,
        barline_token_ids=barline_token_ids,
        non_control_data_token_ids=non_control_data_token_ids,
    )
class StatefulKernLogitsProcessor(transformers.LogitsProcessor):
    """Apply shared delimiter-aware semantic masking over **kern prefixes."""

    def __init__(
        self,
        *,
        i2w: dict[int, str],
        bos_token_id: int | None,
        eos_token_id: int | None,
        pad_token_id: int | None,
        rule_factories: Sequence[SemanticRuleFactory],
        collect_stats: bool = False,
    ) -> None:
        self.context = TokenizerConstraintContext.from_i2w(
            i2w=i2w,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        self._rule_factories = list(rule_factories)
        self.prefilled = False
        self.batch_size = 0
        self.prefix_states: list[KernPrefixState] = []
        self.rule_sets: list[list[KernSemanticRule]] = []
        self._inactive_rows: list[bool] = []
        self.collect_stats = bool(collect_stats)
        self._stats = {
            "calls": 0,
            "rows_processed": 0,
            "advance_row_ms": 0.0,
            "mask_row_ms": 0.0,
            "inactive_rows": 0,
            "terminated_rows": 0,
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._stats["calls"] += 1
        self._stats["rows_processed"] += int(input_ids.shape[0])
        if not self.prefix_states:
            self.batch_size = int(input_ids.shape[0])
            self.prefix_states = [KernPrefixState() for _ in range(self.batch_size)]
            self.rule_sets = [
                [factory() for factory in self._rule_factories] for _ in range(self.batch_size)
            ]
            self._inactive_rows = [False] * self.batch_size

        if int(input_ids.shape[0]) != self.batch_size:
            raise RuntimeError(
                "Expect input_ids.shape[0] to match StatefulKernLogitsProcessor.batch_size. "
                f"Got {input_ids.shape[0]} and {self.batch_size}."
            )

        if not self.prefilled:
            self.prefilled = True
        else:
            for row in range(self.batch_size):
                if self._inactive_rows[row]:
                    continue
                started_at = perf_counter() if self.collect_stats else 0.0
                self._advance_row(row=row, token_id=int(input_ids[row][-1].item()))
                if self.collect_stats:
                    self._stats["advance_row_ms"] += (perf_counter() - started_at) * 1000.0

        for row in range(self.batch_size):
            if self._inactive_rows[row]:
                continue
            started_at = perf_counter() if self.collect_stats else 0.0
            self._mask_row(scores[row], self.prefix_states[row], self.rule_sets[row])
            if self.collect_stats:
                self._stats["mask_row_ms"] += (perf_counter() - started_at) * 1000.0

        self._stats["inactive_rows"] = sum(1 for inactive in self._inactive_rows if inactive)
        self._stats["terminated_rows"] = sum(
            1 for rules in self.rule_sets if any(rule.terminated for rule in rules)
        )

        if any(self._inactive_rows):
            finished_mask = torch.tensor(
                self._inactive_rows,
                device=scores.device,
                dtype=torch.bool,
            )
            if 0 <= self.context.eos_token_id < scores.shape[1]:
                scores[finished_mask] = float("-inf")
                scores[finished_mask, self.context.eos_token_id] = 0.0
        return scores

    def _advance_row(self, *, row: int, token_id: int) -> None:
        if self.context.pad_token_id is not None and token_id == self.context.pad_token_id:
            self._inactive_rows[row] = True
            return
        if token_id == self.context.eos_token_id:
            self._inactive_rows[row] = True
            return
        if self.context.bos_token_id is not None and token_id == self.context.bos_token_id:
            return

        token_text = self.context.i2w.get(token_id)
        if token_text is None:
            raise AssertionError(f"Missing token text for token id={token_id}.")

        prefix_state = self.prefix_states[row]
        rules = self.rule_sets[row]
        try:
            if token_text == "\t":
                prefix_state.append_tab()
                for rule in rules:
                    rule.on_tab_appended(prefix_state)
                return
            if token_text == "\n":
                preview = prefix_state.consume_line_close()
                for rule in rules:
                    rule.on_line_closed(preview.fields)
                return

            prefix_state.append_text(token_text)
            for rule in rules:
                rule.on_text_appended(prefix_state)
        except (KernPrefixStateError, ValueError) as exc:
            raise AssertionError(
                f"Stateful **kern processor rejected sampled token id={token_id} at row={row}: {exc}"
            ) from exc

    def _mask_row(
        self,
        row_scores: torch.FloatTensor,
        prefix_state: KernPrefixState,
        rules: list[KernSemanticRule],
    ) -> None:
        if any(rule.terminated for rule in rules):
            row_scores[:] = float("-inf")
            row_scores[self.context.eos_token_id] = 0.0
            return

        invalid_ids: set[int] = set()

        if prefix_state.current_field_buffer == "":
            invalid_ids.add(self.context.tab_token_id)
            invalid_ids.add(self.context.newline_token_id)
            invalid_ids.add(self.context.eos_token_id)
        else:
            if not all(rule.can_accept_tab(prefix_state) for rule in rules):
                invalid_ids.add(self.context.tab_token_id)

            try:
                fields = prefix_state.preview_line_close().fields
            except KernPrefixStateError:
                invalid_ids.add(self.context.newline_token_id)
                invalid_ids.add(self.context.eos_token_id)
            else:
                if not all(rule.can_close_line(fields) for rule in rules):
                    invalid_ids.add(self.context.newline_token_id)
                if not all(rule.can_end_sequence(fields) for rule in rules):
                    invalid_ids.add(self.context.eos_token_id)

        for token_id in invalid_ids:
            if 0 <= token_id < row_scores.shape[0]:
                row_scores[token_id] = float("-inf")

        for rule in rules:
            rule.mask_scores(prefix_state, row_scores, self.context)

    def stats(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **self._stats,
            "total_ms": float(self._stats["advance_row_ms"] + self._stats["mask_row_ms"]),
        }
        rule_stats = self._collect_rule_stats()
        if rule_stats:
            payload["rules"] = rule_stats
        return payload

    def _collect_rule_stats(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for rules in self.rule_sets:
            for rule in rules:
                stats_fn = getattr(rule, "stats", None)
                if not callable(stats_fn):
                    continue
                _merge_rule_stats(merged, stats_fn())
        return merged


def _merge_rule_stats(destination: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        existing = destination.get(key)
        if isinstance(value, dict):
            nested = existing if isinstance(existing, dict) else {}
            destination[key] = nested
            _merge_rule_stats(nested, value)
            continue
        if isinstance(value, bool):
            destination[key] = int(existing or 0) + int(value)
            continue
        if isinstance(value, int):
            destination[key] = int(existing or 0) + value
            continue
        if isinstance(value, float):
            destination[key] = float(existing or 0.0) + value
            continue
        destination[key] = value
