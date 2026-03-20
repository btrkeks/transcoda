"""Xtuplet marking augmentation."""

from __future__ import annotations

import random
import re

from .kern_utils import get_spine_count

__all__ = ["apply_xtuplet"]

# Token for marking a spine as containing tuplets
XTUPLET = "*Xtuplet"

# Regex to extract rhythm value (leading digits) from a note token
_RHYTHM_PATTERN = re.compile(r"^(\d+)")


def apply_xtuplet(
    krn: str,
    per_spine_probability: float = 0.5,
) -> str:
    """Add *Xtuplet markers to spines that contain tuplet notes.

    Inserts an interpretation line with *Xtuplet for spines that contain
    tuplet rhythms (non-power-of-2 values like 3, 6, 12, etc.).

    Args:
        krn: The kern string to modify.
        per_spine_probability: Probability of marking each spine that
            contains tuplets. Defaults to 0.5.

    Returns:
        Modified kern string with *Xtuplet markers.
    """
    if not krn:
        return krn

    lines = krn.splitlines()
    spine_count = get_spine_count(krn)

    if spine_count == 0:
        return krn

    # Find which spines contain tuplets
    spines_with_tuplets = _find_spines_with_tuplets(lines, spine_count)

    if not spines_with_tuplets:
        return krn

    # Find insertion point (after headers, before spine manipulators)
    insert_idx = _find_insertion_point(lines)

    # Build the interpretation line
    # Each spine that has tuplets gets *Xtuplet with per_spine_probability
    tokens: list[str] = []
    for spine_idx in range(spine_count):
        if spine_idx in spines_with_tuplets and random.random() < per_spine_probability:
            tokens.append(XTUPLET)
        else:
            tokens.append("*")

    # Only insert if at least one spine gets *Xtuplet
    if XTUPLET not in tokens:
        return krn

    xtuplet_line = "\t".join(tokens)
    lines.insert(insert_idx, xtuplet_line)

    return "\n".join(lines)


def _find_spines_with_tuplets(lines: list[str], spine_count: int) -> set[int]:
    """Find which spine indices contain tuplet notes.

    Scans each column position and checks if any note has a tuplet rhythm.

    Args:
        lines: The lines of the kern document.
        spine_count: Number of original spines.

    Returns:
        Set of spine indices (0-based) that contain tuplets.
    """
    spines_with_tuplets: set[int] = set()

    for line in lines:
        # Skip interpretation lines, barlines, and empty lines
        if not line or line.startswith("*") or line.startswith("="):
            continue

        tokens = line.split("\t")
        for col_idx, token in enumerate(tokens):
            # Only check columns within original spine count
            if col_idx >= spine_count:
                break
            if _has_tuplet_rhythm(token):
                spines_with_tuplets.add(col_idx)

    return spines_with_tuplets


def _has_tuplet_rhythm(token: str) -> bool:
    """Check if a token has a tuplet rhythm value.

    A rhythm is a tuplet if it's not a power of 2 (e.g., 3, 5, 6, 7, 9, 10, 12).
    Regular rhythms are powers of 2: 1, 2, 4, 8, 16, 32, 64.

    Args:
        token: A kern note token (e.g., "12g-L", "4c").

    Returns:
        True if the token has a tuplet rhythm, False otherwise.
    """
    match = _RHYTHM_PATTERN.match(token)
    if not match:
        return False

    rhythm = int(match.group(1))
    return not _is_power_of_two(rhythm)


def _is_power_of_two(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _find_insertion_point(lines: list[str]) -> int:
    """Find the line index to insert the *Xtuplet interpretation line.

    Returns the index after header interpretation lines (clef, key, time sig)
    but before spine manipulators (*^, *v) or musical content.

    Args:
        lines: The lines of the kern document.

    Returns:
        The line index where *Xtuplet should be inserted.
    """
    for i, line in enumerate(lines):
        if not line:
            continue

        # Skip header line
        if line.startswith("**"):
            continue

        # Skip interpretation lines that are header-type
        if line.startswith("*") and not line.startswith("*-"):
            tokens = line.split("\t")
            is_header = all(_is_header_interpretation(t) for t in tokens)
            if is_header:
                continue

        # Found first non-header line, insert before it
        return i

    # If we get here, insert at the end
    return len(lines)


def _is_header_interpretation(token: str) -> bool:
    """Check if a token is a header interpretation (clef, key, time sig, etc.).

    Args:
        token: An interpretation token.

    Returns:
        True if this is a header interpretation, False otherwise.
    """
    if token == "*":
        return True
    if token.startswith("*clef"):
        return True
    if token.startswith("*k["):
        return True
    # Time signature: *M followed by a digit
    if token.startswith("*M") and len(token) > 2 and token[2].isdigit():
        return True
    # Part designations
    if token.startswith("*part"):
        return True
    # Staff numbers
    if token.startswith("*staff"):
        return True
    # Instrument designations
    if token.startswith("*I"):
        return True
    return False
