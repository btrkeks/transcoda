"""Rhythm-aware semantic masking for stateful **kern decoding."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import torch

from src.core.spine_state import advance_keep_mask

from .kern_prefix_state import KernPrefixState
from .stateful_kern_logits_processor import KernSemanticRule, TokenizerConstraintContext

RhythmLineType = Literal[
    "empty",
    "exclusive_interpretation",
    "tandem_interpretation",
    "barline",
    "data",
    "spine_terminator",
]

class RhythmRuleError(ValueError):
    """Raised when semantic rhythm constraints are violated."""


@dataclass(frozen=True)
class TimeSignature:
    """Time signature semantics used by the Python rhythm rule."""

    beats: int
    beat_unit: int

    @classmethod
    def parse(cls, value: str) -> "TimeSignature":
        rest = value.removeprefix("*M")
        if rest == value:
            raise RhythmRuleError(f"missing *M prefix: {value}")
        beats_str, sep, beat_unit_str = rest.partition("/")
        if sep != "/":
            raise RhythmRuleError(f"expected beats/unit format: {value}")
        if not beats_str or not beat_unit_str:
            raise RhythmRuleError(f"invalid time signature: {value}")
        try:
            beats = int(beats_str)
            beat_unit = int(beat_unit_str)
        except ValueError as exc:
            raise RhythmRuleError(f"invalid time signature: {value}") from exc
        if beats <= 0 or beat_unit <= 0 or (beat_unit & (beat_unit - 1)) != 0:
            raise RhythmRuleError(f"invalid time signature: {value}")
        return cls(beats=beats, beat_unit=beat_unit)

    @classmethod
    def is_time_signature(cls, value: str) -> bool:
        rest = value.removeprefix("*M")
        if rest == value:
            return False
        beats_str, sep, beat_unit_str = rest.partition("/")
        return bool(sep == "/" and beats_str.isdigit() and beat_unit_str.isdigit())

    @classmethod
    def is_time_signature_candidate(cls, value: str) -> bool:
        return value.startswith("*M") and len(value) > 2 and value[2].isdigit()

    def measure_duration(self) -> Fraction:
        return Fraction(self.beats, self.beat_unit)


@dataclass
class _RhythmSpineState:
    time_signature: TimeSignature | None = None
    current_measure_expected: Fraction | None = None
    current_measure_time_sig: TimeSignature | None = None
    measure_duration: Fraction = Fraction(0, 1)
    outstanding_duration: Fraction = Fraction(0, 1)
    has_data_in_measure: bool = False
    measure_number: int = 1
    is_kern: bool = False
    tie_carryover: Fraction = Fraction(0, 1)
    has_ambiguous_merge_carry: bool = False

    def add_duration(self, duration: Fraction) -> None:
        self.measure_duration += duration
        self.has_data_in_measure = True

    def start_measure(self) -> None:
        self.current_measure_expected = (
            None if self.time_signature is None else self.time_signature.measure_duration()
        )
        self.current_measure_time_sig = self.time_signature
        self.has_data_in_measure = False

    def reset_measure(self) -> None:
        self.measure_duration = self.tie_carryover
        self.tie_carryover = Fraction(0, 1)
        self.outstanding_duration = Fraction(0, 1)
        self.has_ambiguous_merge_carry = False
        self.measure_number += 1
        self.start_measure()


@dataclass
class _RepeatTracker:
    initial_pickup: list[Fraction] = field(default_factory=list)
    expected_duration: list[Fraction] = field(default_factory=list)
    has_initial_pickup: bool = False
    awaiting_section_pickup: bool = False
    section_pickup: list[Fraction] = field(default_factory=list)
    has_section_pickup: bool = False

    def record_initial_pickup(self, durations: list[Fraction]) -> None:
        self.initial_pickup = list(durations)
        self.has_initial_pickup = True

    def on_start_repeat(self, expected: list[Fraction]) -> None:
        self.expected_duration = list(expected)

    def get_initial_pickup(self, index: int) -> Fraction:
        if index >= len(self.initial_pickup):
            return Fraction(0, 1)
        return self.initial_pickup[index]

    def mark_awaiting_section_pickup(self) -> None:
        self.awaiting_section_pickup = True

    def clear_awaiting_section_pickup(self) -> None:
        self.awaiting_section_pickup = False

    def record_section_pickup(self, durations: list[Fraction]) -> None:
        self.section_pickup = list(durations)
        self.has_section_pickup = True
        self.awaiting_section_pickup = False

    def get_section_pickup(self, index: int) -> Fraction:
        if index >= len(self.section_pickup):
            return Fraction(0, 1)
        return self.section_pickup[index]

    def clear_section_pickup(self) -> None:
        self.section_pickup.clear()
        self.has_section_pickup = False
        self.awaiting_section_pickup = False


@dataclass
class _RepeatInfo:
    is_end_repeat: bool = False
    is_start_repeat: bool = False


RhythmEventKind = Literal["duration", "continuation", "grace", "other"]
RhythmPrefixKind = Literal[
    "empty_or_wrapper",
    "other",
    "continuation",
    "grace",
    "duration_numerator",
    "duration_after_percent",
    "duration_denominator",
    "duration_dots",
    "duration_fixed",
]


@dataclass(frozen=True)
class _RhythmEvent:
    kind: RhythmEventKind
    duration: Fraction | None = None


@dataclass(frozen=True)
class _RhythmPrefixInfo:
    kind: RhythmPrefixKind
    duration: Fraction | None = None

    @property
    def starts_duration(self) -> bool:
        return self.kind.startswith("duration_")


@dataclass
class RhythmRule(KernSemanticRule):
    """Stateful rhythm semantics ported from the Rust checker for supported spine ops."""

    allow_anacrusis: bool = True
    allow_incomplete_final: bool = True
    allow_repeat_pairing: bool = True
    spines: list[_RhythmSpineState] = field(default_factory=list)
    repeat_tracker: _RepeatTracker = field(default_factory=_RepeatTracker)
    is_first_barline: bool = True
    pending_final_short_errors: list[tuple[int, Fraction, Fraction]] = field(default_factory=list)

    @property
    def terminated(self) -> bool:
        return False

    def on_text_appended(self, prefix_state: KernPrefixState) -> None:
        return None

    def on_tab_appended(self, prefix_state: KernPrefixState) -> None:
        return None

    def on_line_closed(self, fields: tuple[str, ...]) -> None:
        self._apply_closed_line(fields, end_of_sequence=False)

    def can_accept_tab(self, prefix_state: KernPrefixState) -> bool:
        return True

    def can_close_line(self, fields: tuple[str, ...]) -> bool:
        clone = deepcopy(self)
        try:
            clone._apply_closed_line(fields, end_of_sequence=False)
        except RhythmRuleError:
            return False
        return True

    def can_end_sequence(self, fields: tuple[str, ...]) -> bool:
        clone = deepcopy(self)
        try:
            clone._apply_closed_line(fields, end_of_sequence=True)
        except RhythmRuleError:
            return False
        return True

    def mask_scores(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
    ) -> None:
        if not self.spines:
            return

        spine_index = len(prefix_state.completed_fields)
        if spine_index >= len(self.spines):
            return
        spine = self.spines[spine_index]
        expected = spine.current_measure_expected
        if not spine.is_kern:
            return

        line_type = _infer_partial_line_type(prefix_state)
        if line_type not in {"unknown", "data"}:
            return
        prefix_info = _analyze_rhythm_prefix(prefix_state.current_field_buffer)
        rhythm = context.rhythm

        if spine.outstanding_duration > 0 and not spine.has_ambiguous_merge_carry:
            if prefix_info.kind == "empty_or_wrapper":
                _mask_token_ids(row_scores, rhythm.duration_start_token_ids)
            elif prefix_info.starts_duration:
                _mask_token_ids(row_scores, rhythm.data_token_ids)

        if expected is None:
            return

        current_total = spine.measure_duration
        if prefix_info.kind == "empty_or_wrapper":
            for duration, token_ids in rhythm.parsed_duration_token_ids.items():
                if current_total + duration > expected:
                    _mask_token_ids(row_scores, token_ids)
            return

        if prefix_info.kind == "duration_fixed":
            if prefix_info.duration is not None and current_total + prefix_info.duration > expected:
                _mask_token_ids(row_scores, rhythm.non_grace_data_token_ids)
            return

        if prefix_info.kind == "duration_numerator":
            self._mask_fixed_duration_bucket(
                row_scores,
                expected=expected,
                current_total=current_total,
                duration=prefix_info.duration,
                token_ids=(
                    rhythm.non_grace_token_ids_by_sig_class["other"]
                    + rhythm.non_grace_token_ids_by_sig_class["empty"]
                ),
            )
            self._mask_candidate_tokens(
                prefix_state.current_field_buffer,
                row_scores,
                context,
                current_total=current_total,
                expected=expected,
                token_ids=(
                    rhythm.token_ids_by_sig_class["digit"]
                    + rhythm.token_ids_by_sig_class["dot"]
                    + rhythm.token_ids_by_sig_class["percent"]
                ),
            )
            return

        if prefix_info.kind == "duration_after_percent":
            self._mask_candidate_tokens(
                prefix_state.current_field_buffer,
                row_scores,
                context,
                current_total=current_total,
                expected=expected,
                token_ids=rhythm.token_ids_by_sig_class["digit"],
            )
            return

        if prefix_info.kind == "duration_denominator":
            self._mask_fixed_duration_bucket(
                row_scores,
                expected=expected,
                current_total=current_total,
                duration=prefix_info.duration,
                token_ids=(
                    rhythm.non_grace_token_ids_by_sig_class["other"]
                    + rhythm.non_grace_token_ids_by_sig_class["empty"]
                    + rhythm.non_grace_token_ids_by_sig_class["percent"]
                ),
            )
            self._mask_candidate_tokens(
                prefix_state.current_field_buffer,
                row_scores,
                context,
                current_total=current_total,
                expected=expected,
                token_ids=(
                    rhythm.token_ids_by_sig_class["digit"]
                    + rhythm.token_ids_by_sig_class["dot"]
                ),
            )
            return

        if prefix_info.kind == "duration_dots":
            self._mask_fixed_duration_bucket(
                row_scores,
                expected=expected,
                current_total=current_total,
                duration=prefix_info.duration,
                token_ids=(
                    rhythm.non_grace_token_ids_by_sig_class["other"]
                    + rhythm.non_grace_token_ids_by_sig_class["empty"]
                    + rhythm.non_grace_token_ids_by_sig_class["percent"]
                    + rhythm.non_grace_token_ids_by_sig_class["digit"]
                ),
            )
            self._mask_candidate_tokens(
                prefix_state.current_field_buffer,
                row_scores,
                context,
                current_total=current_total,
                expected=expected,
                token_ids=rhythm.token_ids_by_sig_class["dot"],
            )

    def _mask_fixed_duration_bucket(
        self,
        row_scores: torch.FloatTensor,
        *,
        expected: Fraction,
        current_total: Fraction,
        duration: Fraction | None,
        token_ids: list[int],
    ) -> None:
        if duration is None or current_total + duration <= expected:
            return
        _mask_token_ids(row_scores, token_ids)

    def _mask_candidate_tokens(
        self,
        prefix: str,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
        *,
        current_total: Fraction,
        expected: Fraction,
        token_ids: list[int],
    ) -> None:
        if not token_ids:
            return
        for token_id in token_ids:
            if token_id >= row_scores.shape[0]:
                continue
            candidate_text = prefix + context.i2w[token_id]
            duration = _parse_duration_value(candidate_text)
            if duration is None:
                continue
            if current_total + duration > expected:
                row_scores[token_id] = float("-inf")

    def _apply_closed_line(self, fields: tuple[str, ...], *, end_of_sequence: bool) -> None:
        line_type = _classify_line(fields)
        if line_type == "empty":
            if end_of_sequence:
                self._finalize_eof(last_line_type=line_type)
            return

        if self.pending_final_short_errors and line_type not in {"barline", "spine_terminator"}:
            raise RhythmRuleError("pending final-measure errors before more content")

        if not self.spines and line_type != "exclusive_interpretation":
            # Online decoding emits body-only **kern without a leading **kern header.
            # Seed a synthetic all-**kern header so tandem/data lines can be checked safely.
            self._initialize_spines(tuple(["**kern"] * len(fields)))

        if line_type == "exclusive_interpretation":
            self._initialize_spines(fields)
        elif line_type == "tandem_interpretation":
            self._process_tandem(fields)
        elif line_type == "data":
            self._process_data(fields)
        elif line_type == "barline":
            self._process_barline(fields)
        elif line_type == "spine_terminator":
            self._process_spine_terminator(fields)
        else:
            raise RhythmRuleError(f"unsupported line type: {line_type}")

        if end_of_sequence and line_type != "spine_terminator":
            self._finalize_eof(last_line_type=line_type)

    def _initialize_spines(self, fields: tuple[str, ...]) -> None:
        self.spines = [_RhythmSpineState(is_kern=(field == "**kern")) for field in fields]
        self.repeat_tracker = _RepeatTracker()
        self.is_first_barline = True
        self.pending_final_short_errors.clear()

    def _process_tandem(self, fields: tuple[str, ...]) -> None:
        has_spine_op = any(field in {"*^", "*v"} for field in fields)
        if has_spine_op:
            self.spines = _advance_spines(self.spines, fields)
        if len(fields) == len(self.spines):
            for index, field in enumerate(fields):
                spine = self.spines[index]
                if TimeSignature.is_time_signature(field):
                    ts = TimeSignature.parse(field)
                    spine.time_signature = ts
                    if not spine.has_data_in_measure:
                        spine.current_measure_expected = ts.measure_duration()
                        spine.current_measure_time_sig = ts
                elif TimeSignature.is_time_signature_candidate(field):
                    raise RhythmRuleError(f"invalid time signature: {field}")

    def _process_data(self, fields: tuple[str, ...]) -> None:
        events: list[_RhythmEvent | None] = [None] * len(fields)
        positive_durations: list[Fraction] = []
        has_grace_event = False
        has_duration_event = False

        for spine_idx, field in enumerate(fields):
            if spine_idx >= len(self.spines):
                break
            spine = self.spines[spine_idx]
            if not spine.is_kern:
                continue
            event = _parse_rhythm_event(field)
            events[spine_idx] = event
            if event.kind == "grace":
                has_grace_event = True
            elif event.kind == "duration":
                has_duration_event = True

        is_grace_only_alignment_line = has_grace_event and not has_duration_event
        is_timed_alignment_line = has_duration_event

        for spine_idx, field in enumerate(fields):
            if spine_idx >= len(self.spines):
                break
            spine = self.spines[spine_idx]
            if not spine.is_kern:
                continue
            event = events[spine_idx]
            assert event is not None

            if event.kind == "continuation":
                if spine.outstanding_duration <= 0:
                    if is_grace_only_alignment_line or (
                        is_timed_alignment_line
                        and spine.has_data_in_measure
                        and spine.current_measure_expected is not None
                    ):
                        continue
                    raise RhythmRuleError("continuation token requires active carry")
                if not is_grace_only_alignment_line:
                    positive_durations.append(spine.outstanding_duration)
                continue

            if event.kind == "duration":
                if spine.has_ambiguous_merge_carry:
                    spine.outstanding_duration = Fraction(0, 1)
                    spine.has_ambiguous_merge_carry = False
                if spine.outstanding_duration > 0:
                    raise RhythmRuleError("duration token started before prior carry resolved")
                assert event.duration is not None
                spine.add_duration(event.duration)
                positive_durations.append(event.duration)
                continue

            if event.kind == "grace":
                if spine.outstanding_duration > 0 and not is_grace_only_alignment_line:
                    positive_durations.append(spine.outstanding_duration)
                continue

            if spine.has_ambiguous_merge_carry:
                spine.outstanding_duration = Fraction(0, 1)
                spine.has_ambiguous_merge_carry = False
            if spine.outstanding_duration > 0:
                raise RhythmRuleError("active carry must continue with '.' or grace token")

        step = min(positive_durations, default=Fraction(0, 1))
        if step < 0:
            raise RhythmRuleError("negative rhythm step is invalid")

        for spine_idx, event in enumerate(events):
            if spine_idx >= len(self.spines):
                break
            spine = self.spines[spine_idx]
            if not spine.is_kern:
                continue

            if event is None:
                spine.outstanding_duration = Fraction(0, 1)
                continue

            if event.kind == "continuation":
                if spine.outstanding_duration > 0 and not is_grace_only_alignment_line:
                    spine.outstanding_duration -= step
            elif event.kind == "duration":
                assert event.duration is not None
                spine.outstanding_duration = event.duration - step
            elif event.kind == "grace":
                if spine.outstanding_duration > 0 and not is_grace_only_alignment_line:
                    spine.outstanding_duration -= step
            else:
                if spine.outstanding_duration > 0:
                    spine.outstanding_duration -= step

            if spine.outstanding_duration < 0:
                raise RhythmRuleError("line synchronization overran outstanding duration")

    def _process_barline(self, fields: tuple[str, ...]) -> None:
        if self.pending_final_short_errors:
            raise RhythmRuleError("pending final-measure errors committed before next barline")
        is_explicit_final = _is_explicit_final_barline(fields[0])
        if is_explicit_final:
            for spine in self.spines:
                if spine.is_kern:
                    spine.outstanding_duration = Fraction(0, 1)
                    spine.has_ambiguous_merge_carry = False
        for spine in self.spines:
            if spine.is_kern and spine.has_ambiguous_merge_carry:
                spine.outstanding_duration = Fraction(0, 1)
                spine.has_ambiguous_merge_carry = False
        if any(spine.is_kern and spine.outstanding_duration > 0 for spine in self.spines):
            raise RhythmRuleError("barline arrived before outstanding duration resolved")

        repeat = _parse_repeat_info(fields[0])
        has_content = any(spine.is_kern and spine.measure_duration != 0 for spine in self.spines)
        if self.is_first_barline and not has_content:
            for spine in self.spines:
                spine.start_measure()
            return

        kern_indices = [index for index, spine in enumerate(self.spines) if spine.is_kern]
        spine_durations = [self.spines[index].measure_duration for index in kern_indices]
        expected_durations = [
            self.spines[index].current_measure_expected or Fraction(0, 1) for index in kern_indices
        ]

        if repeat.is_start_repeat:
            self.repeat_tracker.on_start_repeat(expected_durations)

        deferred_short_errors: list[tuple[int, Fraction, Fraction]] = []

        for local_idx, spine_idx in enumerate(kern_indices):
            spine = self.spines[spine_idx]
            expected = spine.current_measure_expected
            if expected is None:
                continue

            actual = spine.measure_duration
            if actual == expected:
                continue

            is_first = self.is_first_barline
            is_section_start = _is_section_start_barline(fields[0], repeat=repeat)
            is_short = actual < expected
            at_end_repeat = repeat.is_end_repeat and is_short

            should_report = True
            if is_first and self.allow_anacrusis and is_short:
                should_report = False
            elif self.repeat_tracker.awaiting_section_pickup and self.allow_anacrusis and is_short:
                should_report = False
            elif is_explicit_final and self.allow_incomplete_final and is_short:
                should_report = False
            elif (
                at_end_repeat
                and self.allow_repeat_pairing
                and (
                    self.repeat_tracker.has_section_pickup or self.repeat_tracker.has_initial_pickup
                )
            ):
                pickup = (
                    self.repeat_tracker.get_section_pickup(local_idx)
                    if self.repeat_tracker.has_section_pickup
                    else self.repeat_tracker.get_initial_pickup(local_idx)
                )
                if pickup + actual == expected:
                    should_report = False
                else:
                    raise RhythmRuleError("repeat pickup pairing failed")
            elif is_section_start and self.allow_anacrusis and is_short:
                should_report = False

            if should_report:
                if not is_explicit_final and self.allow_incomplete_final and is_short:
                    deferred_short_errors.append((spine_idx, expected, actual))
                else:
                    raise RhythmRuleError("measure duration mismatch at barline")

        self.pending_final_short_errors = deferred_short_errors

        if self.is_first_barline and self.allow_anacrusis:
            has_incomplete = any(
                (self.spines[index].current_measure_expected is not None)
                and self.spines[index].measure_duration < self.spines[index].current_measure_expected
                for index in kern_indices
            )
            if has_incomplete:
                self.repeat_tracker.record_initial_pickup(spine_durations)

        if self.repeat_tracker.awaiting_section_pickup and self.allow_anacrusis:
            has_incomplete = any(
                (self.spines[index].current_measure_expected is not None)
                and self.spines[index].measure_duration < self.spines[index].current_measure_expected
                for index in kern_indices
            )
            if has_incomplete:
                self.repeat_tracker.record_section_pickup(spine_durations)
            else:
                self.repeat_tracker.clear_awaiting_section_pickup()

        if repeat.is_end_repeat:
            self.repeat_tracker.clear_section_pickup()

        is_section_boundary = _is_section_boundary_barline(fields[0])
        if (repeat.is_start_repeat or is_section_boundary) and self.allow_anacrusis:
            self.repeat_tracker.mark_awaiting_section_pickup()

        for spine in self.spines:
            spine.reset_measure()
        self.is_first_barline = False

    def _process_spine_terminator(self, fields: tuple[str, ...]) -> None:
        _ = fields
        if any(spine.is_kern and spine.outstanding_duration > 0 for spine in self.spines):
            raise RhythmRuleError("terminator arrived before outstanding duration resolved")
        kern_indices = [index for index, spine in enumerate(self.spines) if spine.is_kern]
        has_content = any(self.spines[index].measure_duration != 0 for index in kern_indices)

        if has_content:
            if self.pending_final_short_errors:
                raise RhythmRuleError("deferred short-measure errors before final terminator")
            for spine_idx in kern_indices:
                spine = self.spines[spine_idx]
                if spine.measure_duration == 0:
                    continue
                expected = spine.current_measure_expected
                if expected is None:
                    continue
                actual = spine.measure_duration
                if actual != expected:
                    is_short = actual < expected
                    if not (self.allow_incomplete_final and is_short):
                        raise RhythmRuleError("final measure duration mismatch before *-")
        else:
            if self.pending_final_short_errors and not self.allow_incomplete_final:
                raise RhythmRuleError("final short measure is not allowed")
            self.pending_final_short_errors.clear()

    def _finalize_eof(self, *, last_line_type: RhythmLineType) -> None:
        if last_line_type == "data":
            if any(spine.is_kern and spine.outstanding_duration > 0 for spine in self.spines):
                raise RhythmRuleError("terminal data line leaves unresolved carry")
            kern_indices = [index for index, spine in enumerate(self.spines) if spine.is_kern]
            for spine_idx in kern_indices:
                spine = self.spines[spine_idx]
                expected = spine.current_measure_expected
                if expected is None:
                    continue
                actual = spine.measure_duration
                if actual != expected:
                    is_short = actual < expected
                    if not (self.allow_incomplete_final and is_short):
                        raise RhythmRuleError("terminal data line leaves invalid final measure")
            self.pending_final_short_errors.clear()
            return

        if last_line_type == "barline":
            if self.pending_final_short_errors and not self.allow_incomplete_final:
                raise RhythmRuleError("pending final-measure errors at EOF")
            self.pending_final_short_errors.clear()
            return

        if self.pending_final_short_errors:
            raise RhythmRuleError("pending final-measure errors at EOF")


def _classify_line(fields: tuple[str, ...]) -> RhythmLineType:
    if not fields:
        return "empty"
    first = fields[0].strip()
    if not first or first.startswith("!"):
        return "empty"
    if first.startswith("**"):
        return "exclusive_interpretation"
    if first == "*-":
        return "spine_terminator"
    if first.startswith("*"):
        return "tandem_interpretation"
    if first.startswith("="):
        return "barline"
    return "data"


def _infer_partial_line_type(prefix_state: KernPrefixState) -> Literal["unknown", "tandem", "barline", "data"]:
    first = None
    if prefix_state.completed_fields:
        first = prefix_state.completed_fields[0]
    elif prefix_state.current_field_buffer:
        first = prefix_state.current_field_buffer

    if first is None or first == "":
        return "unknown"
    if first.startswith("*"):
        return "tandem"
    if first.startswith("="):
        return "barline"
    return "data"


def _is_explicit_final_barline(token: str) -> bool:
    return token == "==" or token == "=:|!"


def _is_section_start_barline(token: str, *, repeat: _RepeatInfo | None = None) -> bool:
    repeat = repeat or _parse_repeat_info(token)
    return "||" in token or repeat.is_start_repeat


def _is_section_boundary_barline(token: str) -> bool:
    return _is_explicit_final_barline(token) or "||" in token


def _parse_repeat_info(token: str) -> _RepeatInfo:
    return _RepeatInfo(
        is_end_repeat=(":|" in token) or (":!" in token),
        is_start_repeat=("|:" in token) or ("!:" in token),
    )


def _advance_spines(spines: list[_RhythmSpineState], fields: tuple[str, ...]) -> list[_RhythmSpineState]:
    keep_mask = [spine.is_kern for spine in spines]
    advance_keep_mask(keep_mask, fields)

    new_spines: list[_RhythmSpineState] = []
    has_ambiguous_structural_carry = False
    old_idx = 0
    token_idx = 0
    while token_idx < len(fields) and old_idx < len(spines):
        token = fields[token_idx]
        if token == "*^":
            spine = deepcopy(spines[old_idx])
            if spine.outstanding_duration > 0:
                spine.has_ambiguous_merge_carry = True
                has_ambiguous_structural_carry = True
            new_spines.append(spine)
            new_spines.append(deepcopy(spine))
            old_idx += 1
            token_idx += 1
            continue
        if token == "*v":
            merge_start = old_idx
            merge_count = 0
            while token_idx < len(fields) and fields[token_idx] == "*v":
                merge_count += 1
                token_idx += 1
            if merge_count < 2:
                raise RhythmRuleError("Invalid merge operation")
            if merge_start + merge_count > len(spines):
                raise RhythmRuleError("not enough spines to merge")
            merged = deepcopy(spines[merge_start])
            merged_states = spines[merge_start : merge_start + merge_count]
            for index in range(merge_start + 1, merge_start + merge_count):
                if spines[index].measure_duration > merged.measure_duration:
                    merged.measure_duration = spines[index].measure_duration
                if spines[index].outstanding_duration > merged.outstanding_duration:
                    merged.outstanding_duration = spines[index].outstanding_duration
            merged.has_ambiguous_merge_carry = _has_ambiguous_merge_carry(merged_states)
            has_ambiguous_structural_carry = (
                has_ambiguous_structural_carry or merged.has_ambiguous_merge_carry
            )
            new_spines.append(merged)
            old_idx += merge_count
            continue
        if token == "*-":
            old_idx += 1
            token_idx += 1
            continue
        new_spines.append(deepcopy(spines[old_idx]))
        old_idx += 1
        token_idx += 1
    if has_ambiguous_structural_carry:
        for spine in new_spines:
            if spine.is_kern and spine.outstanding_duration > 0:
                spine.has_ambiguous_merge_carry = True
    return new_spines

def _parse_duration_value(token: str) -> Fraction | None:
    token = token.strip()
    if token == ".":
        return None
    if "q" in token or "Q" in token:
        return None
    if token.startswith("=") or token.startswith("*") or token.startswith("!"):
        return None
    if " " in token:
        first = token.split(" ")[0]
        return _parse_duration_value(first)

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
        return _apply_dots(base, dots)

    dots = 0
    while index < len(stripped) and stripped[index] == ".":
        dots += 1
        index += 1
    base = Fraction(1, numerator)
    return _apply_dots(base, dots)

def _parse_rhythm_event(token: str) -> _RhythmEvent:
    stripped = token.strip()
    if stripped == ".":
        return _RhythmEvent(kind="continuation")
    if "q" in stripped or "Q" in stripped:
        return _RhythmEvent(kind="grace")
    duration = _parse_duration_value(stripped)
    if duration is not None:
        return _RhythmEvent(kind="duration", duration=duration)
    return _RhythmEvent(kind="other")

def _starts_duration_candidate(token: str) -> bool:
    stripped = token.strip().lstrip("[](){}_")
    return bool(stripped) and stripped[0].isdigit()


def _mask_token_ids(row_scores: torch.FloatTensor, token_ids: list[int]) -> None:
    valid_ids = [token_id for token_id in token_ids if token_id < row_scores.shape[0]]
    if valid_ids:
        row_scores[valid_ids] = float("-inf")


def _analyze_rhythm_prefix(token: str) -> _RhythmPrefixInfo:
    stripped = token.strip()
    if stripped == ".":
        return _RhythmPrefixInfo(kind="continuation")
    if "q" in stripped or "Q" in stripped:
        return _RhythmPrefixInfo(kind="grace")

    normalized = stripped.lstrip("[](){}_")
    if not normalized:
        return _RhythmPrefixInfo(kind="empty_or_wrapper")
    if not normalized[0].isdigit():
        return _RhythmPrefixInfo(kind="other")

    digits_end = 0
    while digits_end < len(normalized) and normalized[digits_end].isdigit():
        digits_end += 1

    numerator = int(normalized[:digits_end])
    if numerator == 0:
        current_duration = Fraction(0, 1)
    else:
        current_duration = Fraction(1, numerator)

    if digits_end == len(normalized):
        return _RhythmPrefixInfo(kind="duration_numerator", duration=current_duration)

    current = normalized[digits_end]
    if current == "%":
        index = digits_end + 1
        denom_start = index
        while index < len(normalized) and normalized[index].isdigit():
            index += 1
        if index == denom_start:
            return _RhythmPrefixInfo(kind="duration_after_percent")

        denominator = int(normalized[denom_start:index])
        if denominator == 0:
            return _RhythmPrefixInfo(kind="other")

        dots = 0
        while index < len(normalized) and normalized[index] == ".":
            dots += 1
            index += 1
        current_duration = _apply_dots(Fraction(numerator, denominator), dots)
        if index == len(normalized):
            kind: RhythmPrefixKind = "duration_dots" if dots > 0 else "duration_denominator"
            return _RhythmPrefixInfo(kind=kind, duration=current_duration)
        return _RhythmPrefixInfo(kind="duration_fixed", duration=current_duration)

    dots = 0
    index = digits_end
    while index < len(normalized) and normalized[index] == ".":
        dots += 1
        index += 1
    current_duration = _apply_dots(current_duration, dots)
    if index == len(normalized):
        if dots > 0:
            return _RhythmPrefixInfo(kind="duration_dots", duration=current_duration)
        return _RhythmPrefixInfo(kind="duration_fixed", duration=current_duration)
    if dots > 0:
        return _RhythmPrefixInfo(kind="duration_fixed", duration=current_duration)
    return _RhythmPrefixInfo(kind="duration_fixed", duration=current_duration)


def _has_ambiguous_merge_carry(spines: list[_RhythmSpineState]) -> bool:
    if not spines:
        return False
    if all(spine.outstanding_duration == 0 for spine in spines):
        return False
    first_state = (spines[0].measure_duration, spines[0].outstanding_duration)
    return any(
        (spine.measure_duration, spine.outstanding_duration) != first_state for spine in spines[1:]
    )


def _apply_dots(base: Fraction, dots: int) -> Fraction:
    total = base
    current = base
    for _ in range(dots):
        current /= 2
        total += current
    return total
