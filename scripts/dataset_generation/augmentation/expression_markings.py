"""Expression marking augmentations."""

from __future__ import annotations

import random

from .kern_utils import find_barline_indices, get_local_spine_count, get_spine_count

__all__ = ["apply_expression_markings"]

# Common expression markings found in classical music
EXPRESSION_MARKINGS: tuple[str, ...] = (
    "espressivo",
    "cantabile",
    "dolce",
    "con brio",
    "con moto",
    "con fuoco",
    "con anima",
    "leggiero",
    "sostenuto",
    "grazioso",
    "maestoso",
    "tranquillo",
    "agitato",
    "appassionato",
    "scherzando",
    "brillante",
)


def apply_expression_markings(
    krn: str,
    *,
    rng: random.Random | None = None,
) -> str:
    """Add a random expression marking to the transcription.

    Inserts an expression marking at a random measure boundary.

    Args:
        krn: The kern string to modify.
        rng: Optional seeded RNG for deterministic sampling in tests.

    Returns:
        Modified kern string with expression marking.
    """
    if not krn:
        return krn

    lines = krn.splitlines()
    spine_count = get_spine_count(krn)
    choose = rng.choice if rng is not None else random.choice

    if spine_count == 0:
        return krn

    # Find barline positions as potential insertion points
    barline_indices = find_barline_indices(krn)

    # Choose insertion point
    if barline_indices:
        # Insert after a random barline (at the start of a new measure)
        insert_idx = choose(barline_indices) + 1
    else:
        # No barlines found, insert near the beginning
        insert_idx = _find_first_content_line(lines)

    # Choose a random expression marking
    expression = choose(EXPRESSION_MARKINGS)
    expression_token = f"*>{expression}"

    # Create expression line using local active spine width at insertion point.
    local_spine_count = get_local_spine_count(lines, insert_idx, fallback=spine_count)
    expression_line = "\t".join([expression_token] + ["*"] * (max(1, local_spine_count) - 1))

    # Insert the expression line
    if insert_idx <= len(lines):
        lines.insert(insert_idx, expression_line)

    return "\n".join(lines)


def _find_first_content_line(lines: list[str]) -> int:
    """Find the index of the first line with musical content."""
    for i, line in enumerate(lines):
        if not line:
            continue
        # Skip header and interpretation lines
        if line.startswith("**") or line.startswith("*"):
            continue
        # Skip barlines
        if line.startswith("="):
            continue
        # Found content
        return i
    return len(lines)
