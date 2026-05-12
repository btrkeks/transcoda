"""Shared utilities for kern token manipulation."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from src.core.kern_utils import is_bar_line, is_note_token

__all__ = [
    "TokenPosition",
    "find_note_tokens",
    "find_barline_indices",
    "append_to_token",
    "apply_suffix_to_notes",
    "get_local_spine_count",
]

# Characters that should come AFTER articulations in a note token
# These are beam markers, tie markers, slur markers, and stem directions
_TRAILING_MARKERS = frozenset("LJK[]()/_\\")


@dataclass(frozen=True)
class TokenPosition:
    """Location of a token in a kern document.

    Attributes:
        line_idx: Zero-based line index.
        col_idx: Zero-based column (spine) index.
        token: The token string at this position.
    """

    line_idx: int
    col_idx: int
    token: str


def find_note_tokens(krn: str) -> list[TokenPosition]:
    """Find all note tokens that can receive articulations.

    Skips rests (tokens containing 'r') and grace notes (containing 'q').

    Args:
        krn: The kern string to search.

    Returns:
        List of TokenPosition objects for each note token found.
    """
    positions: list[TokenPosition] = []

    for line_idx, line in enumerate(krn.splitlines()):
        # Skip interpretation lines and barlines
        if not line or line.startswith("*") or line.startswith("="):
            continue

        for col_idx, token in enumerate(line.split("\t")):
            if _is_articulatable_note(token):
                positions.append(TokenPosition(line_idx, col_idx, token))

    return positions


TIME_SIG_PATTERN = re.compile(r"^\*M\d+/\d+$")


def is_time_signature_line(line: str) -> bool:
    """We simply consider a line to be a time signature line if its first token is a signature."""
    return is_time_signature_token(line.split("\t")[0])


def is_time_signature_token(token: str) -> bool:
    return bool(TIME_SIG_PATTERN.match(token))


def _is_articulatable_note(token: str) -> bool:
    """Check if a token is a note that can receive articulations.

    Returns True for regular notes, False for rests, grace notes,
    and non-note tokens.
    """
    if not is_note_token(token):
        return False
    # Skip rests
    if "r" in token.lower():
        return False
    # Skip grace notes
    if "q" in token:
        return False
    return True


def find_barline_indices(krn: str) -> list[int]:
    """Find line indices of barlines.

    Useful for placing tempo or expression markings at measure boundaries.

    Args:
        krn: The kern string to search.

    Returns:
        List of zero-based line indices where barlines occur.
    """
    indices: list[int] = []
    for line_idx, line in enumerate(krn.splitlines()):
        if is_bar_line(line):
            indices.append(line_idx)
    return indices


def append_to_token(token: str, suffix: str) -> str:
    """Append an articulation symbol to a note token.

    Inserts the suffix before trailing markers (beam, tie, slur, stem).
    For example, appending "'" to "8ee-LJ" produces "8ee-'LJ".

    Args:
        token: The note token to modify.
        suffix: The articulation symbol to append (e.g., "'", "z", ";").

    Returns:
        The modified token with suffix inserted.
    """
    # Find where trailing markers begin
    insert_pos = len(token)
    for i in range(len(token) - 1, -1, -1):
        if token[i] in _TRAILING_MARKERS:
            insert_pos = i
        else:
            break

    return token[:insert_pos] + suffix + token[insert_pos:]


def apply_suffix_to_notes(
    krn: str,
    positions: list[TokenPosition],
    suffix: str,
) -> str:
    """Apply a suffix to notes at specified positions.

    Args:
        krn: The kern string to modify.
        positions: List of TokenPosition objects indicating where to apply.
        suffix: The articulation symbol to append to each note.

    Returns:
        Modified kern string with suffixes applied.
    """
    if not positions:
        return krn

    # Build a set of (line_idx, col_idx) for fast lookup
    position_set = {(p.line_idx, p.col_idx) for p in positions}

    lines = krn.splitlines()
    result_lines: list[str] = []

    for line_idx, line in enumerate(lines):
        # Check if any positions are on this line
        if not any(line_idx == p.line_idx for p in positions):
            result_lines.append(line)
            continue

        # Process each column
        columns = line.split("\t")
        new_columns: list[str] = []

        for col_idx, token in enumerate(columns):
            if (line_idx, col_idx) in position_set:
                new_columns.append(append_to_token(token, suffix))
            else:
                new_columns.append(token)

        result_lines.append("\t".join(new_columns))

    return "\n".join(result_lines)


def sample_positions(
    positions: list[TokenPosition],
    probability: float,
) -> list[TokenPosition]:
    """Randomly sample positions based on probability.

    Args:
        positions: List of candidate positions.
        probability: Probability of selecting each position (0.0 to 1.0).

    Returns:
        Subset of positions that were selected.
    """
    return [p for p in positions if random.random() < probability]


def insert_interpretation_line(
    krn: str,
    line_idx: int,
    tokens: list[str],
) -> str:
    """Insert an interpretation line (starting with *) at the given index.

    The tokens list should have one entry per spine. Use "*" for spines
    that don't receive the interpretation.

    Args:
        krn: The kern string to modify.
        line_idx: Zero-based line index where to insert.
        tokens: List of interpretation tokens, one per spine.

    Returns:
        Modified kern string with interpretation line inserted.
    """
    lines = krn.splitlines()
    new_line = "\t".join(tokens)
    lines.insert(line_idx, new_line)
    return "\n".join(lines)


def get_spine_count(krn: str) -> int:
    """Get the number of spines in a kern string.

    Returns the spine count from the first non-empty line.
    """
    for line in krn.splitlines():
        if line.strip():
            return line.count("\t") + 1
    return 0


def get_local_spine_count(lines: list[str], idx: int, fallback: int = 0) -> int:
    """Infer active local spine count around an insertion index.

    Looks forward first (preferred for insertions before content), then
    backward, while skipping global comments and blank lines.
    """
    for j in range(idx, len(lines)):
        candidate = lines[j]
        if candidate.strip() and not candidate.startswith("!"):
            return candidate.count("\t") + 1
    for j in range(idx - 1, -1, -1):
        candidate = lines[j]
        if candidate.strip() and not candidate.startswith("!"):
            return candidate.count("\t") + 1
    return max(0, fallback)
