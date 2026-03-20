"""Shared utility functions for parsing and inspecting **kern notation.

These pure functions are used across both normalization and synthetic data
generation modules. They have no dependencies on other project modules.
"""

from __future__ import annotations

import re

__all__ = [
    "extract_pitch",
    "get_duration_of_token",
    "is_bar_line",
    "is_grace_note",
    "is_note_token",
    "is_spinemerge_line",
    "is_spinesplit_line",
    "strip_tie_beam_markers_from_kern_text",
    "strip_tie_beam_markers_from_note_token",
    "is_terminator_line",
    "split_into_same_spine_nr_chunks_and_measures",
]

_TIE_BEAM_MARKERS = "[]_LJKk"
_TIE_BEAM_REMOVE_TABLE = str.maketrans("", "", _TIE_BEAM_MARKERS)


def get_duration_of_token(token: str) -> int | None:
    """Extract the duration from a **kern token.

    Args:
        token: A single **kern notes/rests token delimited by spaces

    Returns:
        Duration as an integer, or None if not a valid notes/rests token
    """
    # Strip tie markers ([, ]) and slur markers ((, )) before matching duration
    # These can appear before the duration in tokens like [8ee-J or (8b-L
    token_stripped = token.lstrip("[]()")

    # Match duration at the start of the token
    match = re.match(r"^(\d+)", token_stripped)
    if match:
        return int(match.group(1))
    return None


def is_note_token(token: str) -> bool:
    """Check if a token represents a note in **kern.

    A note token contains pitch letters (a-g or A-G) and does not start
    with '*' (which indicates tandem interpretations).
    """
    if not token or token.startswith("*") or token.startswith("!"):
        return False
    return any(c in token for c in "abcdefgABCDEFG")


def is_grace_note(token: str) -> bool:
    """Check if a token is a grace note (contains 'q')."""
    return "q" in token


def strip_tie_beam_markers_from_note_token(token: str) -> str:
    """Remove tie and beam markers from a single **kern note token."""
    if not token:
        return token
    return token.translate(_TIE_BEAM_REMOVE_TABLE)


def strip_tie_beam_markers_from_kern_text(text: str) -> str:
    """Remove tie/beam markers from note tokens while preserving all other tokens."""
    if not text:
        return text

    output_lines: list[str] = []
    for line in text.splitlines():
        fields = line.split("\t")
        output_fields: list[str] = []
        for field in fields:
            tokens = field.split(" ")
            output_tokens = [
                strip_tie_beam_markers_from_note_token(token) if is_note_token(token) else token
                for token in tokens
            ]
            output_fields.append(" ".join(output_tokens))
        output_lines.append("\t".join(output_fields))
    return "\n".join(output_lines)


def extract_pitch(token: str) -> str | None:
    """Extract the pitch part from a **kern note token.

    Args:
        token: A single **kern note/rests token delimited by spaces

    Returns:
        The pitch part of the token, or None if not a valid note/rests token
    """
    # Strip leading tie markers ([, ]) and slur markers ((, ))
    token = token.lstrip("[]()")
    n = len(token)
    i = 0

    # 1) Skip leading duration digits and augmentation dots (e.g., '16', '4.')
    while i < n and (token[i].isdigit() or token[i] == "."):
        i += 1

    # 2) Collect consecutive pitch letters (octave via repetition)
    j = i
    while j < n and ("A" <= token[j] <= "G" or "a" <= token[j] <= "g"):
        j += 1
    if j == i:
        return None  # no pitch letters present

    # 3) Collect accidentals immediately following the pitch
    k = j
    while k < n and token[k] in {"#", "-", "n"}:
        k += 1

    pitch = token[i:k] if k > i else None
    return pitch


def is_spinesplit_line(line: str) -> bool:
    """Check if line only contains spine split symbols (*^, *)."""
    columns = line.split("\t")
    if any(col == "*^" for col in columns) and all(col == "*^" or col == "*" for col in columns):
        return True
    return False


def is_spinemerge_line(line: str) -> bool:
    """Check if line only contains spine merge symbols (*v, *)."""
    columns = line.split("\t")
    if any(col == "*v" for col in columns) and all(col == "*v" or col == "*" for col in columns):
        return True
    return False


def is_bar_line(line: str) -> bool:
    """Check if line is a bar line (all fields start with =)."""
    fields = line.split("\t")
    return bool(fields) and all(f.startswith("=") for f in fields)


def is_terminator_line(line: str) -> bool:
    """Check if line is a terminator line (all fields are *-)."""
    fields = line.split("\t")
    return bool(fields) and all(f == "*-" for f in fields)


def split_into_same_spine_nr_chunks_and_measures(krn: str) -> list[str]:
    """Split a **kern transcription into chunks with uniform spine count.

    Splits by measures and spine splits/merges such that every returned
    segment has the same number of spines throughout.

    Args:
        krn: **kern transcription as a string

    Returns:
        List of chunks, each with consistent spine count
    """
    result: list[str] = []
    current_chunk: list[str] = []
    for line in krn.splitlines():
        current_chunk.append(line)

        if is_bar_line(line) or is_spinesplit_line(line) or is_spinemerge_line(line):
            # This one is still included in this part. The next part will start after it.
            result.append("\n".join(current_chunk) + "\n")
            current_chunk = []

    if current_chunk:  # leftover after the last boundary
        result.append("\n".join(current_chunk).rstrip())

    # Validation: ensure each chunk has consistent spine count
    for part in result:
        lines = part.splitlines()
        num_tabs = lines[0].count("\t")
        assert lines, "Empty measure part found"
        assert all(line.count("\t") == num_tabs for line in lines), (
            "Inconsistent number of spines in measure part"
        )

    return result
