"""Shared helpers for trailing ``**dynam`` spine render-only augmentations."""

from __future__ import annotations

from src.core.kern_utils import is_bar_line, is_terminator_line

__all__ = [
    "default_dynam_token",
    "find_trailing_dynam_spine_index",
    "infer_trailing_spine_index",
    "is_eligible_data_line",
    "line_has_writable_dynam_null",
]


def is_eligible_data_line(line: str) -> bool:
    """Return True when line can host visual-only dynamic tokens."""
    if not line:
        return False
    if line.startswith("*") or line.startswith("=") or line.startswith("!"):
        return False

    tokens = [token.strip() for token in line.split("\t")]
    return any(token and token != "." for token in tokens)


def default_dynam_token(line: str) -> str | None:
    """Return default trailing ``**dynam`` token for a kern line."""
    if not line:
        return None
    if line.startswith("!!"):
        # Global comments are not spine-aligned records.
        return None
    if line.startswith("**"):
        return "**dynam"
    if is_terminator_line(line):
        return "*-"
    if is_bar_line(line):
        return line.split("\t")[0]
    if line.startswith("*"):
        return "*"
    if _is_local_comment_line(line):
        return line.split("\t")[0]
    return "."


def find_trailing_dynam_spine_index(lines: list[str]) -> int | None:
    """Find trailing ``**dynam`` spine index from the header, if present."""
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("!!"):
            continue
        if not line.startswith("**"):
            return None
        tokens = [token.strip() for token in line.split("\t")]
        if tokens and tokens[-1] == "**dynam":
            return len(tokens) - 1
        return None
    return None


def infer_trailing_spine_index(lines: list[str]) -> int | None:
    """Infer trailing spine index from first local, spine-aligned record."""
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("!!"):
            continue
        return line.count("\t")
    return None


def line_has_writable_dynam_null(line: str, dynam_idx: int) -> bool:
    """Return True when line has writable ``.`` at the ``**dynam`` index."""
    if not line or line.startswith("!!"):
        return False
    fields = line.split("\t")
    if len(fields) <= dynam_idx:
        return False
    return fields[dynam_idx].strip() == "."


def _is_local_comment_line(line: str) -> bool:
    fields = line.split("\t")
    return bool(fields) and all(field.startswith("!") for field in fields)
