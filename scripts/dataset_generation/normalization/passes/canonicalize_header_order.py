"""Pass to canonicalize the order of header interpretation lines.

This pass reorders header interpretation lines (stria, staff, clef, key, meter)
into the canonical order specified by the **kern grammar:

1. stria-line (optional) - *stria4, *stria6
2. staff-line (optional) - *staff1 through *staff6, or *
3. clef-line (required) - *clefF4, *clefG2, etc.
4. key-line (required) - *k[...]
5. meter-line (optional) - *M{num}/{denom} or *met(...) mensuration signs
6. xtuplet-line (optional) - *Xtuplet (marks spines containing tuplets)
"""

from __future__ import annotations

import re

from ..base import NormalizationContext

# Priority order for header line types (lower = earlier in output)
PRIORITY = {
    "stria": 0,
    "staff": 1,
    "clef": 2,
    "key": 3,
    "meter": 4,
    "xtuplet": 5,
}

# Patterns for detecting header interpretation types
# These match tokens that appear in spines
_STRIA_PATTERN = re.compile(r"^\*stria\d+$")
_STAFF_PATTERN = re.compile(r"^\*staff\d+$")
_CLEF_PATTERN = re.compile(r"^\*clef[A-Ga-gvX]v?\d?$")
_KEY_PATTERN = re.compile(r"^\*k\[.*\]$")
_METER_PATTERN = re.compile(r"^\*M\d+/\d+$")
_MENSURATION_PATTERN = re.compile(r"^\*met\(.+\)$")
_XTUPLET_PATTERN = re.compile(r"^\*Xtuplet$")

# Spine manipulators that should not be reordered
_SPINE_MANIPULATORS = {"*^", "*v", "*x", "*+", "*-"}


def _classify_token(token: str) -> str | None:
    """
    Classify a single token as a header type.

    Returns the header type name or None if not a header token.
    """
    if token == "*":
        return "null"  # Null interpretation, compatible with any header type

    if token in _SPINE_MANIPULATORS:
        return None  # Spine manipulator, not a header

    if _STRIA_PATTERN.match(token):
        return "stria"
    if _STAFF_PATTERN.match(token):
        return "staff"
    if _CLEF_PATTERN.match(token):
        return "clef"
    if _KEY_PATTERN.match(token):
        return "key"
    if _METER_PATTERN.match(token):
        return "meter"
    if _MENSURATION_PATTERN.match(token):
        return "meter"
    if _XTUPLET_PATTERN.match(token):
        return "xtuplet"

    return None


def _classify_line(line: str) -> str | None:
    """
    Classify a line by its header type.

    A line is classified as a header type if ALL spines match the same
    header pattern (or are null '*'). Returns None if the line is not
    a header line or contains mixed header types.
    """
    if not line.startswith("*"):
        return None

    spines = line.split("\t")
    header_type = None

    for spine in spines:
        token_type = _classify_token(spine)

        if token_type is None:
            # Token is not a recognized header type
            return None

        if token_type == "null":
            # Null interpretation is compatible with any header type
            continue

        if header_type is None:
            header_type = token_type
        elif header_type != token_type:
            # Mixed header types in the same line
            return None

    return header_type


def _find_header_block(lines: list[str]) -> tuple[int, int]:
    """
    Find the contiguous block of header interpretation lines.

    Returns (start_index, end_index) where end_index is exclusive.
    The header block starts at the first header line and ends when
    we encounter a non-header line (barline, data, etc.).
    """
    start = None
    end = None

    for i, line in enumerate(lines):
        line_type = _classify_line(line)

        if line_type is not None:
            if start is None:
                start = i
            end = i + 1
        elif start is not None:
            # We've found the end of the header block
            break

    if start is None or end is None:
        return (0, 0)

    return (start, end)


def _sort_header_lines(lines: list[str]) -> list[str]:
    """
    Sort header lines by their canonical priority.

    Lines are sorted stably by their header type priority.
    """
    classified = []
    for line in lines:
        line_type = _classify_line(line)
        priority = PRIORITY.get(line_type, 99) if line_type else 99
        classified.append((priority, line))

    # Sort by priority (stable sort preserves original order for equal priorities)
    classified.sort(key=lambda x: x[0])

    return [line for _, line in classified]


class CanonicalizeHeaderOrder:
    """
    Reorder header interpretation lines to canonical order.

    This pass ensures that header lines (stria, part, staff, clef, key, meter)
    appear in the canonical order defined by the kern grammar. This is important
    for grammar validation to succeed and for consistent model training data.

    Example transformation:
        Input:
            *k[f#c#]	*k[f#c#]
            *clefG2	*clefF4
            *M4/4	*M4/4
            =1	=1

        Output:
            *clefG2	*clefF4
            *k[f#c#]	*k[f#c#]
            *M4/4	*M4/4
            =1	=1
    """

    name = "canonicalize_header_order"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Transform by reordering header lines to canonical order."""
        lines = text.split("\n")

        # Find the header block
        start, end = _find_header_block(lines)

        if start == end:
            # No header block found
            return text

        # Extract and sort the header lines
        header_lines = lines[start:end]
        sorted_headers = _sort_header_lines(header_lines)

        # Reconstruct the text
        result_lines = lines[:start] + sorted_headers + lines[end:]

        return "\n".join(result_lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Verify that header lines are in canonical order."""
        lines = text.split("\n")
        start, end = _find_header_block(lines)

        if start == end:
            return

        last_priority = -1
        for line in lines[start:end]:
            line_type = _classify_line(line)
            if line_type is None:
                continue
            priority = PRIORITY.get(line_type, 99)
            if priority < last_priority:
                raise ValueError(
                    f"Header lines are not in canonical order: "
                    f"found {line_type} after a higher-priority header type"
                )
            last_priority = priority
