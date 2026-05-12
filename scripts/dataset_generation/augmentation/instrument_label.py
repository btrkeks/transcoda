"""Instrument label augmentations."""

from __future__ import annotations

import re

__all__ = ["apply_instrument_label_piano"]

_INSTRUMENT_TOKEN_RE = re.compile(r"(^|\t)\*I[^\t\n]*(?=\t|$)", re.MULTILINE)


def apply_instrument_label_piano(krn: str) -> str:
    """Insert a `*I"Piano` interpretation line across all spines.

    The insertion is intended for render-only augmentation. If an instrument
    interpretation already exists, the input is returned unchanged.
    """
    if not krn:
        return krn

    if _INSTRUMENT_TOKEN_RE.search(krn):
        return krn

    lines = krn.splitlines()
    spine_count = _get_spine_count(krn)
    if spine_count == 0:
        return krn

    piano_line = "\t".join(['*I"Piano'] * spine_count)

    insert_idx = 0
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        if line.startswith("**"):
            insert_idx = idx + 1
        else:
            insert_idx = idx
        break

    lines.insert(insert_idx, piano_line)
    return "\n".join(lines)


def _get_spine_count(krn: str) -> int:
    """Return spine count from the first non-empty line."""
    for line in krn.splitlines():
        if line.strip():
            return line.count("\t") + 1
    return 0
