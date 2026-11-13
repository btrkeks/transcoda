"""Keep every spine on one Humdrum timeline during constrained decoding.

kern_parser lowers a transcoda-v2 prediction to Humdrum **kern, and Verovio's
humlib then walks each spine and writes the spine's running duration onto every
line that carries one of its notes, rests or the final ``*-``
(``HumdrumFileStructure::prepareDurations``). Two spines that write different
times onto one line raise "Inconsistent rhythm analysis", after which Verovio
may crash. Barlines and interpretations carry no time, so humlib never checks
them directly.

V2.1 often misreads a duration (a sixteenth-note run as ``12 24 24``), so one
voice runs ahead of another. Masking the decoder back onto the timeline does
not work: denied a barline or a note, it writes filler (spine operations,
grace notes, endless dots) until it runs out of tokens, and spelling the
missing time as rests in its own context makes it copy those rests. So the
decoder keeps its own reading, and the rule tracks what that reading implies.
Each column has a next onset, in quarter notes:

- A data line sits at the latest onset among its regular notes and rests. Every
  column still before that time gets an implied invisible rest (``Nryy``, the
  dialect's own gap filler) up to it. The decoder never sees these rests.
- Barlines, system breaks and the end of the sequence imply rests that align
  every column. ``MM3r`` rests do too, then fill every column with the same
  count and wait for a barline.
- ``*^`` children inherit their parent's onset. Before a ``*v`` run, implied
  rests bring the run's columns to its latest onset: humlib follows the left
  column through a merge, so they must agree. Such a rest may sit out of time
  order with other columns, which humlib accepts.
- Grace notes last zero time and sit at their own column's onset. On a line
  with regular notes, a grace note's column must not be later than the line
  (an earlier one is rested up to it). The grace notes of a grace-only line
  share one onset.
- Durations take at most three dots, as in every reference label.

``spell_timeline`` writes the implied rests into the finalized text, so the
output keeps every token the decoder chose. A reference label implies no rest.
The rule aligns time slices only. Durations stay free, so it never checks a
measure against its time signature.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

import torch

from .kern_prefix_state import KernPrefixState
from .spine_lineage import MERGE_FIELD
from .stateful_kern_logits_processor import KernSemanticRuleBase, TokenizerConstraintContext

_SPINE_OP_FIELDS = frozenset({"*", "*^", MERGE_FIELD})
_DURATION_CHARS = frozenset("0123456789.%")
_DURATION = re.compile(r"(\d+)(?:%(\d+))?(\.*)")
_SPECIAL_TOKENS = frozenset({"\t", "\n", "<bos>", "<eos>", "<pad>"})
_MAX_DOTS = 3


class FieldKind(StrEnum):
    """What a (partial) field contributes to the timeline."""

    NULL = "null"
    REGULAR = "regular"
    GRACE = "grace"
    MULTI_REST = "multi_rest"
    # Duration characters only: the field may still become regular or grace.
    PENDING = "pending"
    BARLINE = "barline"
    COMMENT = "comment"
    INTERPRETATION = "interpretation"


_DATA_KINDS = frozenset({FieldKind.REGULAR, FieldKind.GRACE, FieldKind.MULTI_REST})


def field_kind(text: str) -> FieldKind:
    if text in ("", "."):
        return FieldKind.NULL
    first = text[0]
    if first == "=":
        return FieldKind.BARLINE
    if first == "!":
        return FieldKind.COMMENT
    if first == "*":
        return FieldKind.INTERPRETATION
    if first == "M":
        return FieldKind.MULTI_REST
    return _kind_after_duration(text)


def _kind_after_duration(text: str) -> FieldKind:
    rest = text.lstrip("0123456789.%")
    if not rest:
        return FieldKind.PENDING
    if rest[0] in "qQ":
        return FieldKind.GRACE
    return FieldKind.REGULAR


def duration_in_quarters(field: str) -> Fraction:
    """Duration of a compact data field's leading **kern recip (`4.`, `12`, `7%2`, `00`)."""
    match = _DURATION.match(field)
    if match is None:
        raise ValueError(f"data field has no duration: {field!r}")
    number, denominator, dots = match.groups()
    if denominator is not None:
        whole = Fraction(int(denominator), int(number))
    elif int(number) == 0:
        whole = Fraction(2 ** len(number))
    else:
        whole = Fraction(1, int(number))
    total = step = whole
    for _ in dots:
        step /= 2
        total += step
    return total * 4


def recip(quarters: Fraction) -> str:
    """The **kern recip for a duration in quarter notes: `4`, `8.`, `2..`, `0`, or `8%5`."""
    whole = quarters / 4
    count, unit = whole.numerator, whole.denominator
    if count == 1:
        return str(unit)
    if count == 3 and unit % 2 == 0:
        return f"{unit // 2}."
    if count == 7 and unit % 4 == 0:
        return f"{unit // 4}.."
    if unit == 1 and count in (2, 4):
        return "0" * (count // 2)
    return f"{unit}%{count}"


@dataclass(frozen=True)
class Timeline:
    """Next onset of each active column, and whether a multi-measure rest awaits its barline."""

    onsets: tuple[Fraction, ...]
    multi_rest_open: bool = False

    @classmethod
    def initial(cls, width: int) -> Timeline:
        return cls(onsets=(Fraction(0),) * width)

    def allowed_kinds(self, completed_fields: Sequence[str]) -> frozenset[FieldKind]:
        """Field kinds the next column may take, given the fields already on the line."""
        column = len(completed_fields)
        if column >= len(self.onsets):
            return frozenset()
        kinds = [field_kind(field) for field in completed_fields]
        structural = frozenset({FieldKind.INTERPRETATION, FieldKind.BARLINE, FieldKind.COMMENT})
        if kinds and kinds[0] not in _DATA_KINDS and kinds[0] is not FieldKind.NULL:
            return structural
        if self.multi_rest_open:
            return frozenset() if column else structural
        if FieldKind.MULTI_REST in kinds:
            return frozenset({FieldKind.MULTI_REST})
        onset = self.onsets[column]
        regular = [self.onsets[i] for i, kind in enumerate(kinds) if kind is FieldKind.REGULAR]
        grace = [self.onsets[i] for i, kind in enumerate(kinds) if kind is FieldKind.GRACE]
        allowed: set[FieldKind] = {FieldKind.NULL}
        if column == 0:
            allowed |= structural | {FieldKind.MULTI_REST}
        # Checked column by column, so every prefix of a line can still close.
        if not grace or onset >= max(grace):
            allowed.add(FieldKind.REGULAR)
        if (regular and onset <= max(regular)) or (not regular and set(grace) <= {onset}):
            allowed.add(FieldKind.GRACE)
        if FieldKind.REGULAR in allowed or FieldKind.GRACE in allowed:
            allowed.add(FieldKind.PENDING)
        return frozenset(allowed)

    def close(self, fields: Sequence[str]) -> tuple[tuple[str, ...], Timeline]:
        """Apply one closed line: the implied rest lines that precede it, and the next timeline.

        Raise ValueError when the line breaks a rule the decoder is masked to keep.
        """
        if len(fields) == 1 and fields[0].startswith("!"):
            return self.align() if fields[0].startswith("!!LO:LB") else ((), self)
        if fields[0].startswith("="):
            return self.align()[0], Timeline.initial(len(self.onsets))
        if len(fields) != len(self.onsets):
            raise ValueError(f"expected {len(self.onsets)} fields, got {len(fields)}")
        if fields[0].startswith("*"):
            if all(field in _SPINE_OP_FIELDS for field in fields):
                return self._apply_spine_ops(fields)
            return (), self
        for column, field in enumerate(fields):
            if field_kind(field) not in self.allowed_kinds(fields[:column]):
                raise ValueError(f"column {column} {field!r} breaks the timeline {self.onsets}")
        kinds = [field_kind(field) for field in fields]
        if FieldKind.MULTI_REST in kinds:
            if len(set(fields)) != 1:
                raise ValueError("a multi-measure rest line needs the same rest in every column")
            rests, aligned = self.align()
            return rests, Timeline(aligned.onsets, multi_rest_open=True)
        regular = [column for column, kind in enumerate(kinds) if kind is FieldKind.REGULAR]
        if not regular:
            return (), self
        line_time = max(self.onsets[column] for column in regular)
        onsets = [max(onset, line_time) for onset in self.onsets]
        rests = self._rests_to(tuple(onsets))
        for column in regular:
            onsets[column] = line_time + duration_in_quarters(fields[column])
        return rests, Timeline(tuple(onsets))

    def align(self) -> tuple[tuple[str, ...], Timeline]:
        """Implied rest lines that bring every column to the latest onset, and the aligned timeline."""
        if self.multi_rest_open:
            return (), self
        aligned = (max(self.onsets),) * len(self.onsets)
        return self._rests_to(aligned), Timeline(aligned)

    def _rests_to(self, targets: tuple[Fraction, ...]) -> tuple[str, ...]:
        """Implied rest lines that carry each column from its onset to its target, in onset order."""
        pairs = list(zip(self.onsets, targets, strict=True))
        return tuple(
            "\t".join(
                recip(target - onset) + "ryy" if onset == start and target > onset else ""
                for onset, target in pairs
            )
            for start in sorted({onset for onset, target in pairs if onset < target})
        )

    def _apply_spine_ops(self, fields: Sequence[str]) -> tuple[tuple[str, ...], Timeline]:
        targets = list(self.onsets)
        onsets: list[Fraction] = []
        column = 0
        while column < len(fields):
            field = fields[column]
            if field == MERGE_FIELD:
                end = column
                while end < len(fields) and fields[end] == MERGE_FIELD:
                    end += 1
                latest = max(self.onsets[column:end])
                targets[column:end] = [latest] * (end - column)
                onsets.append(latest)
                column = end
                continue
            onsets.extend([self.onsets[column]] * (2 if field == "*^" else 1))
            column += 1
        return self._rests_to(tuple(targets)), Timeline(tuple(onsets), self.multi_rest_open)


@dataclass(frozen=True)
class _TokenTables:
    """Token ids grouped by the field kind they produce from an empty or duration-only field."""

    from_empty: dict[FieldKind, torch.Tensor]
    after_duration: dict[FieldKind, torch.Tensor]
    # Token ids that start with dots, by how many.
    leading_dots: dict[int, torch.Tensor]


_TABLES: dict[int, tuple[TokenizerConstraintContext, _TokenTables]] = {}


def _token_tables(context: TokenizerConstraintContext) -> _TokenTables:
    cached = _TABLES.get(id(context))
    if cached is not None and cached[0] is context:
        return cached[1]
    from_empty: dict[FieldKind, list[int]] = {}
    after_duration: dict[FieldKind, list[int]] = {}
    leading_dots: dict[int, list[int]] = {}
    for token_id, text in context.i2w.items():
        if text in _SPECIAL_TOKENS or not text:
            continue
        from_empty.setdefault(field_kind(text), []).append(token_id)
        after_duration.setdefault(_kind_after_duration(text), []).append(token_id)
        dots = len(text) - len(text.lstrip("."))
        if dots:
            leading_dots.setdefault(dots, []).append(token_id)

    def tensors(groups: dict[FieldKind, list[int]]) -> dict[FieldKind, torch.Tensor]:
        return {kind: torch.tensor(sorted(ids), dtype=torch.long) for kind, ids in groups.items()}

    tables = _TokenTables(
        from_empty=tensors(from_empty),
        after_duration=tensors(after_duration),
        leading_dots={
            dots: torch.tensor(sorted(ids), dtype=torch.long) for dots, ids in leading_dots.items()
        },
    )
    _TABLES[id(context)] = (context, tables)
    return tables


def _mask(row_scores: torch.FloatTensor, token_ids: torch.Tensor) -> None:
    if token_ids.numel():
        row_scores[token_ids.to(row_scores.device)] = float("-inf")


@dataclass(frozen=True)
class TimelineRuleFactory:
    def __call__(self) -> TimelineRule:
        return TimelineRule()


@dataclass
class TimelineRule(KernSemanticRuleBase):
    """Per-sequence timeline state; ``timeline`` is None until the header's first line closes."""

    timeline: Timeline | None = None

    def on_line_closed(self, fields: tuple[str, ...]) -> None:
        self.timeline = self._close(fields)[1]

    def can_accept_tab(self, prefix_state: KernPrefixState) -> bool:
        if prefix_state.current_field_buffer == "":
            return self.can_accept_empty_field(prefix_state)
        return self._field_complete(prefix_state)

    def can_accept_empty_field(self, prefix_state: KernPrefixState) -> bool:
        if self.timeline is None:
            return True
        return FieldKind.NULL in self.timeline.allowed_kinds(prefix_state.completed_fields)

    def can_close_line(self, fields: tuple[str, ...]) -> bool:
        try:
            self._close(fields)
        except ValueError:
            return False
        return True

    def can_close_line_with_empty_field(
        self,
        prefix_state: KernPrefixState,
        fields: tuple[str, ...],
    ) -> bool:
        return self.can_accept_empty_field(prefix_state) and self.can_close_line(fields)

    def can_end_sequence(self, fields: tuple[str, ...]) -> bool:
        return self.can_close_line(fields)

    def mask_scores(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
    ) -> None:
        timeline = self.timeline
        if timeline is None:
            return
        completed = prefix_state.completed_fields
        buffer = prefix_state.current_field_buffer
        tables = _token_tables(context)
        allowed = timeline.allowed_kinds(completed)
        if buffer == "":
            for kind, token_ids in tables.from_empty.items():
                if kind not in allowed:
                    _mask(row_scores, token_ids)
            return
        kind = field_kind(buffer)
        if kind is FieldKind.PENDING:
            for after, token_ids in tables.after_duration.items():
                if after not in allowed:
                    _mask(row_scores, token_ids)
            trailing_dots = len(buffer) - len(buffer.rstrip("."))
            for dots, token_ids in tables.leading_dots.items():
                if trailing_dots + dots > _MAX_DOTS:
                    _mask(row_scores, token_ids)
        elif kind is FieldKind.MULTI_REST and completed:
            self._mask_multi_rest_spelling(buffer, completed[0], row_scores, context)

    def _field_complete(self, prefix_state: KernPrefixState) -> bool:
        completed = prefix_state.completed_fields
        buffer = prefix_state.current_field_buffer
        if self.timeline is None or not completed or field_kind(buffer) is not FieldKind.MULTI_REST:
            return True
        return buffer == completed[0]

    def _mask_multi_rest_spelling(
        self,
        buffer: str,
        first_field: str,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
    ) -> None:
        for token_id, text in context.i2w.items():
            if text in _SPECIAL_TOKENS:
                continue
            if not first_field.startswith(buffer + text) and 0 <= token_id < row_scores.shape[0]:
                row_scores[token_id] = float("-inf")

    def _close(self, fields: Sequence[str]) -> tuple[tuple[str, ...], Timeline]:
        timeline = self.timeline if self.timeline is not None else Timeline.initial(len(fields))
        return timeline.close(fields)


def spell_timeline(text: str) -> str:
    """Write the invisible rests the timeline implies into decoded text, ending aligned.

    Raise ValueError on a line the timeline rule would have masked.
    """
    spelled: list[str] = []
    timeline: Timeline | None = None
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if body:
            fields = body.split("\t")
            rests, timeline = (timeline or Timeline.initial(len(fields))).close(fields)
            spelled.extend(rest + "\n" for rest in rests)
        spelled.append(line)
    rests = timeline.align()[0] if timeline is not None else ()
    if rests and spelled and not spelled[-1].endswith("\n"):
        spelled[-1] += "\n"
    spelled.extend(rest + "\n" for rest in rests)
    return "".join(spelled)
