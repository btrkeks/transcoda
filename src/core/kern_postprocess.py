"""Utilities for end-of-sequence postprocessing in page-faithful **kern text."""

from __future__ import annotations

from src.core.kern_utils import is_terminator_line


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

    spine_count = lines[-1].count("\t") + 1
    terminator = "\t".join(["*-"] * spine_count)
    return "\n".join([*lines, terminator])
