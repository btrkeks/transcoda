"""Pedal marking augmentations."""

from __future__ import annotations

import random

from .kern_utils import find_barline_indices, get_local_spine_count, get_spine_count

__all__ = ["apply_pedaling"]

# Pedal interpretation tokens
PEDAL_DOWN = "*ped"
PEDAL_UP = "*Xped"


def apply_pedaling(
    krn: str,
    measures_probability: float = 0.3,
    *,
    rng: random.Random | None = None,
) -> str:
    """Add pedal markings to random measures.

    Inserts *ped at the start of selected measures and *Xped before
    the following barline.

    Args:
        krn: The kern string to modify.
        measures_probability: Probability of adding pedal to each measure.
        rng: Optional seeded RNG for deterministic sampling in tests.

    Returns:
        Modified kern string with pedal markings.
    """
    if not krn:
        return krn

    lines = krn.splitlines()
    spine_count = get_spine_count(krn)
    rand = rng.random if rng is not None else random.random

    if spine_count == 0:
        return krn

    # Find barline positions
    barline_indices = find_barline_indices(krn)

    if len(barline_indices) < 2:
        # Need at least 2 barlines to create a pedal span
        return krn

    # Select measures to add pedaling
    # A measure spans from one barline to the next
    insertions: list[tuple[int, str]] = []

    for i in range(len(barline_indices) - 1):
        if rand() >= measures_probability:
            continue

        start_barline = barline_indices[i]
        end_barline = barline_indices[i + 1]

        # Insert *ped after the start barline
        ped_down_idx = start_barline + 1
        # Insert *Xped before the end barline
        ped_up_idx = end_barline

        # Add to insertions list (we'll process in reverse order)
        insertions.append((ped_down_idx, PEDAL_DOWN))
        insertions.append((ped_up_idx, PEDAL_UP))

    if not insertions:
        return krn

    # Sort by index in reverse order so insertions don't shift later indices
    insertions.sort(key=lambda x: x[0], reverse=True)

    # Create interpretation line template
    def make_pedal_line(token: str, local_spine_count: int) -> str:
        return "\t".join([token] + ["*"] * (max(1, local_spine_count) - 1))

    # Insert pedal lines
    for idx, token in insertions:
        pedal_line = make_pedal_line(
            token,
            get_local_spine_count(lines, idx, fallback=spine_count),
        )
        lines.insert(idx, pedal_line)

    return "\n".join(lines)
