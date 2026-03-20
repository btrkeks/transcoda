"""Tempo marking augmentations."""

from __future__ import annotations

import random
import re

__all__ = ["apply_tempo_markings", "has_tempo_markings"]

# Curated Italian tempo vocabulary for OMD reference records.
OMD_TEMPO_MARKINGS: tuple[str, ...] = (
    "Largo",
    "Adagio",
    "Andante",
    "Moderato",
    "Allegro",
    "Presto",
)

# BPM range for optional metronome (*MM) tandem interpretation.
BPM_RANGE = (40, 180)

_OMD_RE = re.compile(r"^!!!OMD:", re.MULTILINE)
_MM_TOKEN_RE = re.compile(r"(^|\t)\*MM[^\t\n]*(?=\t|$)", re.MULTILINE)


def has_tempo_markings(krn: str) -> bool:
    """Return True when any OMD or *MM tempo metadata is already present."""
    return bool(_OMD_RE.search(krn) or _MM_TOKEN_RE.search(krn))


def apply_tempo_markings(krn: str, include_mm_probability: float = 0.35) -> str:
    """Add render-only tempo metadata (OMD + optional *MM).

    Inserts:
    - one `!!!OMD: <tempo>` reference record near the top
    - an optional numeric `*MM<bpm>` interpretation line in all spines

    Args:
        krn: The kern string to modify.
        include_mm_probability: Probability of adding a numeric *MM line.

    Returns:
        Modified kern string with tempo metadata, or unchanged when tempo
        metadata already exists.
    """
    if not 0.0 <= include_mm_probability <= 1.0:
        raise ValueError(
            "include_mm_probability must be in [0.0, 1.0], "
            f"got {include_mm_probability}"
        )
    if not krn:
        return krn
    if has_tempo_markings(krn):
        return krn

    lines = krn.splitlines()

    tempo_text = random.choice(OMD_TEMPO_MARKINGS)
    omd_line = f"!!!OMD: {tempo_text}"
    lines.insert(_find_omd_insertion_point(lines), omd_line)

    if random.random() < include_mm_probability:
        spine_count = _get_spine_count(lines)
        if spine_count > 0:
            bpm = random.randint(*BPM_RANGE)
            mm_line = "\t".join([f"*MM{bpm}"] + ["*"] * (spine_count - 1))
            lines.insert(_find_tempo_insertion_point(lines), mm_line)

    return "\n".join(lines)


def _find_omd_insertion_point(lines: list[str]) -> int:
    """Insert OMD after leading comments/reference records."""
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if line.startswith("!"):
            continue
        return i
    return 0


def _get_spine_count(lines: list[str]) -> int:
    """Return spine count from the first non-comment, non-empty line."""
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("!"):
            continue
        return line.count("\t") + 1
    return 0


def _find_tempo_insertion_point(lines: list[str]) -> int:
    """Find the best line index to insert a tempo marking.

    Returns the index after header interpretation lines (clef, key, time sig)
    but before the first musical content.
    """
    for i, line in enumerate(lines):
        if not line.strip():
            continue

        # Skip comments/reference records
        if line.startswith("!"):
            continue

        # Skip header line
        if line.startswith("**"):
            continue

        # Skip interpretation lines (clef, key sig, time sig)
        if line.startswith("*") and not line.startswith("*-"):
            # Check if this is a header interpretation
            tokens = line.split("\t")
            is_header = all(_is_header_interpretation(t) for t in tokens)
            if is_header:
                continue

        # Found first non-header line, insert before it
        return i

    # If we get here, insert at the end
    return len(lines)


def _is_header_interpretation(token: str) -> bool:
    """Check if a token is a header interpretation (clef, key, time sig)."""
    if token == "*":
        return True
    if token.startswith("*clef"):
        return True
    if token.startswith("*k["):
        return True
    if token.startswith("*M") and len(token) > 2 and token[2].isdigit():
        return True
    return False
