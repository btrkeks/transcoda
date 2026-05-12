"""HoistSpineSplitsToBarline pass - moves split records after prior barlines."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.kern_utils import is_bar_line, is_spinesplit_line
from src.core.spine_state import (
    InvalidSpineOperationError,
    UnsupportedSpineManipulatorError,
    advance_spine_count,
    is_interpretation_record,
)

from ..base import NormalizationContext

_LEADING_DURATION_WRAPPERS = "[](){}_&"
_DURATION_RE = re.compile(r"^\d+(?:%\d+)?\.*")


@dataclass(frozen=True)
class _SplitEvent:
    index: int
    fields: list[str]
    post_width: int


def _duration_text_from_token(token: str) -> str | None:
    index = 0
    while index < len(token) and token[index] in _LEADING_DURATION_WRAPPERS:
        index += 1

    match = _DURATION_RE.match(token[index:])
    if not match:
        return None
    return match.group(0)


def _duration_text_from_field(field: str) -> str | None:
    if field in {"", "."}:
        return None

    durations: set[str] = set()
    for token in field.split(" "):
        duration = _duration_text_from_token(token)
        if duration is None:
            return None
        durations.add(duration)

    if len(durations) != 1:
        return None
    return next(iter(durations))


def _is_data_line(line: str) -> bool:
    if not line:
        return False
    return not line.startswith(("!", "*", "="))


def _compute_post_split_width(fields: list[str]) -> int:
    return sum(2 if field == "*^" else 1 for field in fields)


def _expand_fields_for_split(fields: list[str], split_fields: list[str]) -> list[str] | None:
    if len(fields) != len(split_fields):
        return None

    expanded: list[str] = []
    for field, split_field in zip(fields, split_fields, strict=True):
        expanded.append(field)
        if split_field != "*^":
            continue

        duration = _duration_text_from_field(field)
        if duration is None:
            return None
        expanded.append(f"{duration}ryy")

    return expanded


def _advance_width(current_width: int, fields: list[str]) -> int | None:
    if not is_interpretation_record(fields):
        return current_width
    try:
        return advance_spine_count(current_width, fields)
    except (
        InvalidSpineOperationError,
        UnsupportedSpineManipulatorError,
        ValueError,
    ):
        return None


def _rewrite_chunk_after_barline(lines: list[str], initial_width: int) -> list[str]:
    split_events: list[_SplitEvent] = []
    split_indices: set[int] = set()
    data_by_index: dict[int, list[str]] = {}
    current_width = initial_width
    blocked_before_split = False
    blocked_after_split = False

    for index, line in enumerate(lines):
        fields = line.split("\t")

        if is_spinesplit_line(line):
            if blocked_before_split or blocked_after_split or len(fields) != current_width:
                return lines
            event = _SplitEvent(
                index=index,
                fields=fields,
                post_width=_compute_post_split_width(fields),
            )
            split_events.append(event)
            split_indices.add(index)
            current_width = event.post_width
            continue

        if _is_data_line(line):
            if len(fields) != current_width:
                return lines
            data_by_index[index] = fields
            continue

        next_width = _advance_width(current_width, fields)
        if next_width is None:
            return lines
        current_width = next_width

        if split_events:
            blocked_after_split = True
        else:
            blocked_before_split = True

    if not split_events:
        return lines

    rewritten: list[str] = ["\t".join(event.fields) for event in split_events]
    for index, line in enumerate(lines):
        if index in split_indices:
            continue

        fields = data_by_index.get(index)
        if fields is None:
            rewritten.append(line)
            continue

        expanded = fields
        for event in split_events:
            if event.index <= index:
                continue
            next_expanded = _expand_fields_for_split(expanded, event.fields)
            if next_expanded is None:
                return lines
            expanded = next_expanded
        rewritten.append("\t".join(expanded))

    return rewritten


class HoistSpineSplitsToBarline:
    """
    Move safe spine-split records to immediately after the previous barline.

    Data rows between the barline and original split are expanded to the
    post-split width. Newly-created right-hand spines are filled with invisible
    rests matching the split source field's duration. Ambiguous chunks are left
    unchanged.
    """

    name = "hoist_spine_splits_to_barline"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Hoist safe spine splits to the start of their measure chunk."""
        if not text:
            return text

        result: list[str] = []
        chunk: list[str] = []
        previous_barline_width: int | None = None

        def flush_chunk() -> None:
            nonlocal chunk
            if previous_barline_width is None:
                result.extend(chunk)
            else:
                result.extend(_rewrite_chunk_after_barline(chunk, previous_barline_width))
            chunk = []

        for line in text.split("\n"):
            if is_bar_line(line):
                flush_chunk()
                result.append(line)
                previous_barline_width = len(line.split("\t"))
                continue
            chunk.append(line)

        flush_chunk()
        return "\n".join(result)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Validation is handled by the final ValidateSpineOperations pass."""
        pass
