"""DelaySpineMergesToBarline pass - moves merge records to the next barline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.kern_utils import is_bar_line, is_spinemerge_line

from ..base import NormalizationContext

_LEADING_DURATION_WRAPPERS = "[](){}_&"
_DURATION_RE = re.compile(r"^\d+(?:%\d+)?\.*")


@dataclass(frozen=True)
class _MergeMapping:
    pre_to_post: list[int]
    merged_away_indices: set[int]
    post_width: int


def _build_merge_mapping(fields: list[str]) -> _MergeMapping:
    pre_to_post: list[int] = []
    merged_away_indices: set[int] = set()
    post_index = 0
    index = 0

    while index < len(fields):
        if fields[index] != "*v":
            pre_to_post.append(post_index)
            post_index += 1
            index += 1
            continue

        group_start = index
        while index < len(fields) and fields[index] == "*v":
            pre_to_post.append(post_index)
            if index > group_start:
                merged_away_indices.add(index)
            index += 1
        post_index += 1

    return _MergeMapping(
        pre_to_post=pre_to_post,
        merged_away_indices=merged_away_indices,
        post_width=post_index,
    )


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


def _expand_data_line(line: str, mapping: _MergeMapping) -> str | None:
    fields = line.split("\t")
    if len(fields) != mapping.post_width:
        return None

    expanded: list[str] = []
    rest_by_post_index: dict[int, str] = {}
    for pre_index, post_index in enumerate(mapping.pre_to_post):
        field = fields[post_index]
        if pre_index not in mapping.merged_away_indices:
            expanded.append(field)
            continue

        rest = rest_by_post_index.get(post_index)
        if rest is None:
            duration = _duration_text_from_field(field)
            if duration is None:
                return None
            rest = f"{duration}ryy"
            rest_by_post_index[post_index] = rest
        expanded.append(rest)

    return "\t".join(expanded)


class DelaySpineMergesToBarline:
    """
    Move pure spine-merge records to the next barline when safe.

    Data rows between the original merge and barline are expanded back to the
    pre-merge width. The merged-away spines are filled with invisible rests
    matching the surviving merged field's duration. Ambiguous regions are left
    unchanged.
    """

    name = "delay_spine_merges_to_barline"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Delay safe spine merges until immediately before the next barline."""
        if not text:
            return text

        result: list[str] = []
        pending_merge_line: str | None = None
        pending_mapping: _MergeMapping | None = None
        pending_original_lines: list[str] = []
        pending_expanded_lines: list[str] = []

        def flush_pending() -> None:
            nonlocal pending_merge_line, pending_mapping, pending_original_lines
            nonlocal pending_expanded_lines
            if pending_merge_line is not None:
                result.append(pending_merge_line)
                result.extend(pending_original_lines)
            pending_merge_line = None
            pending_mapping = None
            pending_original_lines = []
            pending_expanded_lines = []

        for line in text.split("\n"):
            if pending_merge_line is None:
                if is_spinemerge_line(line):
                    pending_merge_line = line
                    pending_mapping = _build_merge_mapping(line.split("\t"))
                else:
                    result.append(line)
                continue

            assert pending_mapping is not None
            if is_bar_line(line):
                if len(line.split("\t")) == pending_mapping.post_width:
                    result.extend(pending_expanded_lines)
                    result.append(pending_merge_line)
                    result.append(line)
                    pending_merge_line = None
                    pending_mapping = None
                    pending_original_lines = []
                    pending_expanded_lines = []
                else:
                    flush_pending()
                    result.append(line)
                continue

            if not _is_data_line(line):
                flush_pending()
                result.append(line)
                continue

            expanded = _expand_data_line(line, pending_mapping)
            if expanded is None:
                flush_pending()
                result.append(line)
                continue

            pending_original_lines.append(line)
            pending_expanded_lines.append(expanded)

        flush_pending()
        return "\n".join(result)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Validation is handled by the final ValidateSpineOperations pass."""
        pass
