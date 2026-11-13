"""Shared stateful logits processor for semantic **kern decoding rules."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter
from typing import Any, Protocol

import torch
import transformers

from .kern_prefix_state import KernPrefixState, KernPrefixStateError, PrefixPositionKind
from .spine_lineage import LINEAGE_FIELDS


class KernSemanticRule(Protocol):
    """Per-sequence semantic rule used by StatefulKernLogitsProcessor."""

    @property
    def humdrum_terminated(self) -> bool:
        ...

    @property
    def terminated(self) -> bool:
        ...

    @property
    def supports_empty_fields(self) -> bool:
        ...

    def on_text_appended(self, prefix_state: KernPrefixState) -> None:
        ...

    def on_tab_appended(self, prefix_state: KernPrefixState) -> None:
        ...

    def on_line_closed(self, fields: tuple[str, ...]) -> None:
        ...

    def can_accept_tab(self, prefix_state: KernPrefixState) -> bool:
        ...

    def can_accept_empty_field(self, prefix_state: KernPrefixState) -> bool:
        ...

    def can_close_line(self, fields: tuple[str, ...]) -> bool:
        ...

    def can_close_line_with_empty_field(
        self,
        prefix_state: KernPrefixState,
        fields: tuple[str, ...],
    ) -> bool:
        ...

    def can_end_sequence(self, fields: tuple[str, ...]) -> bool:
        ...

    def can_end_at_line_boundary(self, prefix_state: KernPrefixState) -> bool:
        ...

    def mask_scores(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
    ) -> None:
        ...


class KernSemanticRuleBase:
    """No-op semantic rule defaults for tests and simple future rules."""

    @property
    def humdrum_terminated(self) -> bool:
        return False

    @property
    def terminated(self) -> bool:
        return self.humdrum_terminated

    @property
    def supports_empty_fields(self) -> bool:
        return False

    def on_text_appended(self, prefix_state: KernPrefixState) -> None:
        return None

    def on_tab_appended(self, prefix_state: KernPrefixState) -> None:
        return None

    def on_line_closed(self, fields: tuple[str, ...]) -> None:
        return None

    def can_accept_tab(self, prefix_state: KernPrefixState) -> bool:
        return True

    def can_accept_empty_field(self, prefix_state: KernPrefixState) -> bool:
        return True

    def can_close_line(self, fields: tuple[str, ...]) -> bool:
        return True

    def can_close_line_with_empty_field(
        self,
        prefix_state: KernPrefixState,
        fields: tuple[str, ...],
    ) -> bool:
        return True

    def can_end_sequence(self, fields: tuple[str, ...]) -> bool:
        return True

    def can_end_at_line_boundary(self, prefix_state: KernPrefixState) -> bool:
        return True

    def mask_scores(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
    ) -> None:
        return None


SemanticRuleFactory = Callable[[], KernSemanticRule]


@dataclass
class SemanticRowFork:
    """Detached copy of one row's prefix and rule state, for lookahead."""

    prefix_state: KernPrefixState
    rules: list[KernSemanticRule]


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
    lineage_field_tokens: dict[str, tuple[tuple[int, str], ...]]

    @classmethod
    def from_i2w(
        cls,
        *,
        i2w: dict[int, str],
        bos_token_id: int | None,
        eos_token_id: int | None,
        pad_token_id: int | None,
    ) -> TokenizerConstraintContext:
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
            lineage_field_tokens=_build_field_prefix_tokens(normalized, LINEAGE_FIELDS),
        )


def _find_unique_token_id(i2w: dict[int, str], token_text: str) -> int:
    matches = [token_id for token_id, text in i2w.items() if text == token_text]
    if not matches:
        raise ValueError(f"Tokenizer is missing dedicated {token_text!r} token.")
    if len(matches) > 1:
        raise ValueError(f"Tokenizer has multiple dedicated {token_text!r} tokens: {matches}")
    return int(matches[0])


def _build_field_prefix_tokens(
    i2w: dict[int, str],
    targets: frozenset[str],
) -> dict[str, tuple[tuple[int, str], ...]]:
    """Map each proper prefix of a target field to the tokens that extend it.

    Each entry pairs a token id with the field text after appending it.
    """
    alphabet = set("".join(targets))
    table: dict[str, list[tuple[int, str]]] = {}
    for token_id, text in sorted(i2w.items()):
        if not text or not set(text) <= alphabet:
            continue
        for target in targets:
            for start in range(len(target)):
                if target.startswith(text, start):
                    entry = (token_id, target[: start + len(text)])
                    entries = table.setdefault(target[:start], [])
                    if entry not in entries:
                        entries.append(entry)
    return {prefix: tuple(entries) for prefix, entries in table.items()}


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
    number_text = "".join(number)
    numerator = int(number_text)
    if numerator == 0:
        if number_text not in {"0", "00", "000"}:
            return None
        dots = 0
        while index < len(stripped) and stripped[index] == ".":
            dots += 1
            index += 1
        base = Fraction(2 ** len(number_text), 1)
        return _apply_duration_dots_for_metadata(base, dots)

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
        base = Fraction(denominator, numerator)
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
            "humdrum_terminated_rows": 0,
            "constraint_dead_ends": 0,
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
        humdrum_terminated_rows = sum(
            1 for rules in self.rule_sets if _has_humdrum_terminated_rule(rules)
        )
        self._stats["humdrum_terminated_rows"] = humdrum_terminated_rows
        self._stats["terminated_rows"] = humdrum_terminated_rows

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

        try:
            self._advance_state(self.prefix_states[row], self.rule_sets[row], token_id)
        except (KernPrefixStateError, ValueError) as exc:
            raise AssertionError(
                f"Stateful **kern processor rejected sampled token id={token_id} at row={row}: {exc}"
            ) from exc

    def _advance_state(
        self,
        prefix_state: KernPrefixState,
        rules: list[KernSemanticRule],
        token_id: int,
    ) -> None:
        token_text = self.context.i2w.get(token_id)
        if token_text is None:
            raise AssertionError(f"Missing token text for token id={token_id}.")

        if token_text == "\t":
            if prefix_state.current_field_buffer == "":
                if not _can_accept_empty_field(prefix_state, rules):
                    raise ValueError("empty field separator is not valid here")
            elif not all(rule.can_accept_tab(prefix_state) for rule in rules):
                raise ValueError("tab is not valid here")
            prefix_state.append_tab()
            for rule in rules:
                rule.on_tab_appended(prefix_state)
            return
        if token_text == "\n":
            preview = prefix_state.preview_line_close()
            if prefix_state.current_field_buffer == "":
                if not _can_close_line_with_empty_field(prefix_state, preview.fields, rules):
                    raise ValueError("line close with empty field is not valid here")
            elif not all(rule.can_close_line(preview.fields) for rule in rules):
                raise ValueError("line close is not valid here")
            prefix_state.consume_line_close()
            for rule in rules:
                rule.on_line_closed(preview.fields)
            return

        prefix_state.append_text(token_text)
        for rule in rules:
            rule.on_text_appended(prefix_state)

    def fork_row(self, row: int) -> SemanticRowFork | None:
        """Copy one active row's state for lookahead; None once the row has finished."""
        if not self.prefix_states or self._inactive_rows[row]:
            return None
        return SemanticRowFork(
            prefix_state=copy.deepcopy(self.prefix_states[row]),
            rules=copy.deepcopy(self.rule_sets[row]),
        )

    def clone_fork(self, fork: SemanticRowFork) -> SemanticRowFork:
        return copy.deepcopy(fork)

    def advance_fork(self, fork: SemanticRowFork, token_id: int) -> bool:
        """Advance a forked row by one text, tab or newline token; False if it is rejected."""
        try:
            self._advance_state(fork.prefix_state, fork.rules, token_id)
        except (KernPrefixStateError, ValueError):
            return False
        return True

    def mask_fork(self, fork: SemanticRowFork, row_scores: torch.FloatTensor) -> None:
        """Mask what this processor would mask after the forked prefix, with no dead-end fallback."""
        self._apply_row_masks(row_scores, fork.prefix_state, fork.rules)

    def _mask_row(
        self,
        row_scores: torch.FloatTensor,
        prefix_state: KernPrefixState,
        rules: list[KernSemanticRule],
    ) -> None:
        self._apply_row_masks(row_scores, prefix_state, rules)
        if not torch.isfinite(row_scores).any():
            if self.context.pad_token_id is None:
                raise RuntimeError(
                    "Semantic constraint stack reached a dead end without a padding sentinel."
                )
            self._stats["constraint_dead_ends"] += 1
            row_scores[self.context.pad_token_id] = 0.0

    def _apply_row_masks(
        self,
        row_scores: torch.FloatTensor,
        prefix_state: KernPrefixState,
        rules: list[KernSemanticRule],
    ) -> None:
        if _has_humdrum_terminated_rule(rules):
            row_scores[:] = float("-inf")
            row_scores[self.context.eos_token_id] = 0.0
            return

        invalid_ids = self._invalid_ids_for_position(prefix_state, rules)
        if self.context.bos_token_id is not None:
            invalid_ids.add(self.context.bos_token_id)
        if self.context.pad_token_id is not None:
            invalid_ids.add(self.context.pad_token_id)

        for token_id in invalid_ids:
            if 0 <= token_id < row_scores.shape[0]:
                row_scores[token_id] = float("-inf")

        for rule in rules:
            rule.mask_scores(prefix_state, row_scores, self.context)

    def _invalid_ids_for_position(
        self,
        prefix_state: KernPrefixState,
        rules: list[KernSemanticRule],
    ) -> set[int]:
        if prefix_state.position_kind is PrefixPositionKind.FIELD_TEXT:
            return self._invalid_ids_for_field_text_position(prefix_state, rules)
        return self._invalid_ids_for_empty_position(prefix_state, rules)

    def _invalid_ids_for_empty_position(
        self,
        prefix_state: KernPrefixState,
        rules: list[KernSemanticRule],
    ) -> set[int]:
        invalid_ids: set[int] = set()
        if not _can_accept_empty_field(prefix_state, rules):
            invalid_ids.add(self.context.tab_token_id)
        if not self._can_close_line_at_empty_position(prefix_state, rules):
            invalid_ids.add(self.context.newline_token_id)
        if not self._can_emit_eos_at_empty_position(prefix_state, rules):
            invalid_ids.add(self.context.eos_token_id)
        return invalid_ids

    def _can_close_line_at_empty_position(
        self,
        prefix_state: KernPrefixState,
        rules: list[KernSemanticRule],
    ) -> bool:
        if prefix_state.position_kind is PrefixPositionKind.LINE_BOUNDARY:
            return False
        fields = prefix_state.preview_line_close().fields
        return _can_close_line_with_empty_field(prefix_state, fields, rules)

    def _can_emit_eos_at_empty_position(
        self,
        prefix_state: KernPrefixState,
        rules: list[KernSemanticRule],
    ) -> bool:
        if prefix_state.position_kind is PrefixPositionKind.LINE_BOUNDARY:
            return all(rule.can_end_at_line_boundary(prefix_state) for rule in rules)
        if prefix_state.position_kind is PrefixPositionKind.START:
            return False
        fields = prefix_state.preview_line_close().fields
        return _can_close_line_with_empty_field(prefix_state, fields, rules) and all(
            rule.can_end_sequence(fields) for rule in rules
        )

    def _invalid_ids_for_field_text_position(
        self,
        prefix_state: KernPrefixState,
        rules: list[KernSemanticRule],
    ) -> set[int]:
        invalid_ids: set[int] = set()

        if not all(rule.can_accept_tab(prefix_state) for rule in rules):
            invalid_ids.add(self.context.tab_token_id)

        try:
            fields = prefix_state.preview_line_close().fields
        except KernPrefixStateError:
            invalid_ids.add(self.context.newline_token_id)
            invalid_ids.add(self.context.eos_token_id)
            return invalid_ids

        if not all(rule.can_close_line(fields) for rule in rules):
            invalid_ids.add(self.context.newline_token_id)
        if not all(rule.can_end_sequence(fields) for rule in rules):
            invalid_ids.add(self.context.eos_token_id)
        return invalid_ids

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


def _has_humdrum_terminated_rule(rules: list[KernSemanticRule]) -> bool:
    return any(rule.humdrum_terminated for rule in rules)


def _has_empty_field_support(rules: list[KernSemanticRule]) -> bool:
    return any(rule.supports_empty_fields for rule in rules)


def _can_accept_empty_field(
    prefix_state: KernPrefixState,
    rules: list[KernSemanticRule],
) -> bool:
    if not _has_empty_field_support(rules):
        return False
    return all(rule.can_accept_empty_field(prefix_state) for rule in rules)


def _can_close_line_with_empty_field(
    prefix_state: KernPrefixState,
    fields: tuple[str, ...],
    rules: list[KernSemanticRule],
) -> bool:
    if not _has_empty_field_support(rules):
        return False
    return all(
        rule.can_close_line_with_empty_field(prefix_state, fields)
        for rule in rules
    )


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
