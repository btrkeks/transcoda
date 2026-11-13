"""Base protocols for pre-parser kern cleanup passes."""

from __future__ import annotations

from typing import Any, Protocol


class CleanupContext(dict[str, Any]):
    """Scratchpad for cleanup passes to share run-local metadata."""


class CleanupPass(Protocol):
    """Protocol for a text-to-text cleanup pass."""

    name: str

    def transform(self, text: str, ctx: CleanupContext) -> str:
        """Return cleaned text."""
        ...
