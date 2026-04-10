"""Utilities for end-of-sequence postprocessing in page-faithful **kern text."""

from __future__ import annotations

from src.core.kern_utils import is_terminator_line
from src.core.spine_state import advance_spine_count, is_interpretation_record


def strip_terminal_terminator_lines(text: str) -> str:
    """Remove trailing spine-terminator lines (`*-`) from a **kern string.

    Only trailing terminator lines are removed; interior `*-` lines (if any) are left
    untouched.
    """
    if not text:
        return text

    lines = text.rstrip("\n").split("\n")
    while lines and is_terminator_line(lines[-1]):
        lines.pop()
    return "\n".join(lines)


def append_terminator_if_missing(text: str) -> str:
    """Append a spine terminator line when the text does not already end with one."""
    if not text:
        return text

    lines = text.rstrip("\n").split("\n")
    if not lines:
        return text

    if is_terminator_line(lines[-1]):
        return "\n".join(lines)

    spine_count = _infer_terminal_spine_count(lines)
    if spine_count is None or spine_count <= 0:
        spine_count = lines[-1].count("\t") + 1
    terminator = "\t".join(["*-"] * spine_count)
    return "\n".join([*lines, terminator])


def _infer_terminal_spine_count(lines: list[str]) -> int | None:
    active_spines: int | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("!!"):
            continue

        fields = line.split("\t")
        if active_spines is None:
            active_spines = len(fields)
        elif len(fields) != active_spines:
            return None

        if not is_interpretation_record(fields):
            continue
        try:
            active_spines = advance_spine_count(active_spines, fields)
        except ValueError:
            return None
    return active_spines
