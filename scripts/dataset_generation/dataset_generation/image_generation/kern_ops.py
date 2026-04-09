"""Kern operations for synthetic data generation.

This module contains functions for manipulating **kern notation that are
specific to synthetic data generation (not general normalization).
"""

from __future__ import annotations

import re

# Re-export shared helpers from core module for backward compatibility
from src.core.kern_utils import (
    extract_pitch,
    get_duration_of_token,
    is_bar_line,
    is_grace_note,
    is_note_token,
    is_spinemerge_line as _core_is_spinemerge_line,
    is_spinesplit_line as _core_is_spinesplit_line,
    is_terminator_line,
    split_into_same_spine_nr_chunks_and_measures,
)

# Backward compatibility alias for old name
split_into_same_splite_nr_chunks_and_measures = split_into_same_spine_nr_chunks_and_measures

__all__ = [
    "drop_header_strings",
    "is_interpretation_token",
    "remove_bar_from_end",
    "remove_consecutive_newlines",
    "remove_first_time_signature",
    # Re-exported from src.core.kern_utils
    "extract_pitch",
    "get_duration_of_token",
    "is_bar_line",
    "is_grace_note",
    "is_note_token",
    "is_spinemerge_line",
    "is_spinesplit_line",
    "is_terminator_line",
    "split_into_same_spine_nr_chunks_and_measures",
    "split_into_same_splite_nr_chunks_and_measures",  # Backward compat alias
]

_TIME_SIG_RE = re.compile(r"(^|\t)\*M\d+/\d+(?:\+\d+/\d+)*(?=\t|$)")


def is_spinesplit_line(line: str) -> bool:
    stripped = line.strip()
    return "\t" in stripped and _core_is_spinesplit_line(stripped)


def is_spinemerge_line(line: str) -> bool:
    stripped = line.strip()
    return "\t" in stripped and _core_is_spinemerge_line(stripped)


def is_interpretation_token(token: str) -> bool:
    """Check if a token is an interpretation token (clef, key sig, time sig)."""
    f = token.strip()
    if not f:
        # empty field — treat as header-ish (doesn't stop the header block)
        return True
    if f.startswith("*clef"):
        return True
    if f.startswith("*k["):
        return True
    # *M4/4, *M17/16 etc. (field[0]=='*', field[1]=='M', field[2] digit)
    if f.startswith("*M") and len(f) > 2 and f[2].isdigit():
        return True
    return False


def drop_header_strings(s: str) -> str:
    """Remove all clef, key signature, and time signature lines from the top of a kern string.

    Tokens removed:
    - clef (*clefG2, *clefF4, *clefC3, etc.)
    - key signature (*k[f#c#g#d#a#e#b#], *k[], etc.)
    - time signature (*M4/4, *M3/4, *M6/8, etc.)

    Args:
        s: Kern data as a string with newline-separated lines

    Returns:
        Filtered kern data with header lines removed
    """
    if not s:
        return ""

    lines = s.splitlines()
    for i, line in enumerate(lines):
        fields = line.split("\t")

        # if every field is an interpretation token, skip; otherwise stop here
        if all(is_interpretation_token(f) for f in fields):
            continue
        # first line that contains a non-header field -> return from here onward
        return "\n".join(lines[i:])
    return ""  # All lines were header lines


def remove_first_time_signature(s: str) -> str:
    """Remove the first line that contains a genuine time signature token.

    Matches *M<num>/<num> (optionally additive like *M3/8+3/8) in any spine.
    Does NOT match tempo tokens like *MM120.
    """
    if not s:
        return s
    lines = s.split("\n")
    for i, line in enumerate(lines):
        if _TIME_SIG_RE.search(line):
            return "\n".join(lines[:i] + lines[i + 1 :])
    return s


def remove_consecutive_newlines(s: str) -> str:
    """Replace multiple consecutive newlines with a single newline."""
    return re.sub(r"\n+", "\n", s)


def _find_bar_to_split_at(lines: list[str]) -> int | None:
    """Find the index of the bar line after which should be split."""

    # Skip all structure lines at the end
    for i in range(len(lines) - 1, -1, -1):
        if not (is_bar_line(lines[i]) or is_terminator_line(lines[i])):
            break
    else:
        return None

    # Find first bar line from the end
    for j in range(i, -1, -1):
        if is_bar_line(lines[j]):
            return j

    return None


def remove_bar_from_end(transcription: str) -> str:
    """Remove the last measure from a **kern transcription.

    Removes the last single bar line (=\\t=) and all content until the next
    bar line or terminator. Also fixes field counts on structural lines to
    match the spine count at that point in the transcription.

    Args:
        transcription: **kern transcription as a string

    Returns:
        Transcription with last measure removed
    """
    if not transcription:
        return ""

    lines = transcription.split("\n")

    # Find the index after which to remove everything
    split_idx = _find_bar_to_split_at(lines)

    # If no index found, return unchanged
    if split_idx is None:
        return transcription

    # Build the result
    result_lines: list[str] = []

    # Keep all lines before the split index (inclusive)
    result_lines.extend(lines[: split_idx + 1])

    # Add terminator line
    num_spines = result_lines[-1].count("\t") + 1
    result_lines.append("\t".join(["*-"] * num_spines))

    # Join lines and clean up trailing whitespace
    result = "\n".join(result_lines)
    return result.rstrip()
