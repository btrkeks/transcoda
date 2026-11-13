"""Pipeline for composing pre-parser kern cleanup passes."""

from __future__ import annotations

from collections.abc import Iterable

from .base import CleanupContext, CleanupPass


class CleanupPipeline:
    """Run cleanup passes in order."""

    def __init__(self, passes: Iterable[CleanupPass]):
        self.passes = list(passes)

    def __call__(self, text: str, ctx: CleanupContext | None = None) -> str:
        if ctx is None:
            ctx = CleanupContext()

        for cleanup_pass in self.passes:
            text = cleanup_pass.transform(text, ctx)

        return text

    def get_pass_names(self) -> list[str]:
        """Return pass names in execution order."""
        return [cleanup_pass.name for cleanup_pass in self.passes]
