"""Remove beginning time signature augmentation"""

from __future__ import annotations

from .kern_utils import (
    is_time_signature_line,
)

__all__ = ["apply_remove_beginning_time_signature"]


def apply_remove_beginning_time_signature(krn: str, per_note_probability: float = 0.1) -> str:
    """Remove the time signature at the very beginning of the piece, if present."""
    lines = krn.splitlines()

    for line in lines:
        if not line.startswith("*") or line.startswith("!"):
            break

        if is_time_signature_line(line):
            lines.remove(line)
            break

    return "\n".join(lines)
