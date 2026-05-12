"""Pass to merge consecutive clef lines in the header.

This pass handles files converted by musicxml2hum that may have multiple
consecutive clef interpretation lines before any data records. The grammar
only allows ONE clef line in the header, so these must be merged into a
single line keeping the last non-null clef for each spine.

Example:
    Input:
        *clefF4	*clefG2	*clefG2	*clefF4
        *	*	*clefF4	*clefGv2

    Output:
        *clefF4	*clefG2	*clefF4	*clefGv2
"""

from __future__ import annotations

import re

from ..base import NormalizationContext

# Pattern matching clef interpretations
# Matches: *clefG2, *clefF4, *clefC3, *clefGv2 (vocal), *clefX (percussion), etc.
_CLEF_PATTERN = re.compile(r"^\*clef[A-Ga-gvX]v?\d?$")


def _is_clef_token(token: str) -> bool:
    """Check if a token is a clef interpretation."""
    return bool(_CLEF_PATTERN.match(token))


def _is_clef_line(line: str) -> bool:
    """
    Check if a line is a clef interpretation line.

    A clef line contains only clef tokens and null interpretations (*).
    At least one token must be a clef (not all nulls).
    """
    if not line.startswith("*"):
        return False

    tokens = line.split("\t")
    has_clef = False

    for token in tokens:
        if token == "*":
            continue
        elif _is_clef_token(token):
            has_clef = True
        else:
            # Token is neither null nor clef
            return False

    return has_clef


def _find_header_end(lines: list[str]) -> int:
    """
    Find the end of the header block (index of first barline or data line).

    Returns the index of the first non-header line, or len(lines) if all
    lines are header lines.
    """
    for i, line in enumerate(lines):
        # Data lines start with a note token (digit for duration)
        # Barlines start with =
        # Also stop on empty lines
        if not line:
            continue
        if line.startswith("="):
            return i
        # Data lines start with digits (note durations) or dots (continuation)
        if line[0].isdigit() or line.startswith("."):
            return i
        # Check if first token looks like data (not an interpretation)
        first_token = line.split("\t")[0] if "\t" in line else line
        if not first_token.startswith("*") and not first_token.startswith("!"):
            return i

    return len(lines)


def _find_consecutive_clef_groups(
    lines: list[str], header_end: int
) -> list[tuple[int, int]]:
    """
    Find groups of consecutive clef lines within the header.

    Returns list of (start, end) tuples where end is exclusive.
    Only returns groups with 2 or more consecutive clef lines.
    """
    groups: list[tuple[int, int]] = []
    i = 0

    while i < header_end:
        if _is_clef_line(lines[i]):
            start = i
            # Find end of consecutive clef lines
            while i < header_end and _is_clef_line(lines[i]):
                i += 1
            end = i
            # Only include groups with 2+ lines
            if end - start >= 2:
                groups.append((start, end))
        else:
            i += 1

    return groups


def _merge_clef_lines(lines: list[str]) -> str:
    """
    Merge multiple clef lines into a single line.

    For each spine position, keeps the last non-null clef value.
    """
    if not lines:
        return ""

    # Split each line into tokens
    all_tokens = [line.split("\t") for line in lines]

    # Verify all lines have the same number of spines
    num_spines = len(all_tokens[0])
    for tokens in all_tokens[1:]:
        if len(tokens) != num_spines:
            # Spine count mismatch - return first line unchanged
            return lines[0]

    # For each spine, find the last non-null clef
    merged: list[str] = []
    for spine_idx in range(num_spines):
        last_clef = "*"  # Default to null if no clef found
        for tokens in all_tokens:
            token = tokens[spine_idx]
            if _is_clef_token(token):
                last_clef = token
        merged.append(last_clef)

    return "\t".join(merged)


class MergeHeaderClefLines:
    """
    Merge consecutive clef interpretation lines in header into single line.

    This pass identifies consecutive clef lines in the header block and
    merges them by taking the last non-null clef for each spine position.
    Mid-piece clef changes are not affected.

    Example transformation:
        Input:
            *part1	*part2
            *clefF4	*clefG2
            *	*clefF4
            *k[]	*k[]
            =1	=1

        Output:
            *part1	*part2
            *clefF4	*clefF4
            *k[]	*k[]
            =1	=1
    """

    name = "merge_header_clef_lines"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Merge consecutive clef lines in the header."""
        lines = text.split("\n")

        # Find header end
        header_end = _find_header_end(lines)

        # Find groups of consecutive clef lines
        groups = _find_consecutive_clef_groups(lines, header_end)

        if not groups:
            return text

        # Process groups in reverse order to maintain indices
        result_lines = lines[:]
        for start, end in reversed(groups):
            merged = _merge_clef_lines(result_lines[start:end])
            result_lines[start:end] = [merged]

        return "\n".join(result_lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Verify there are no consecutive clef lines in the header."""
        lines = text.split("\n")
        header_end = _find_header_end(lines)
        groups = _find_consecutive_clef_groups(lines, header_end)

        if groups:
            raise ValueError(
                f"Found {len(groups)} group(s) of consecutive clef lines "
                f"in header that should have been merged"
            )
