"""Strip terminal spine-terminator lines from normalized **kern text."""

from __future__ import annotations

from src.core.kern_postprocess import strip_terminal_terminator_lines
from src.core.kern_utils import is_terminator_line

from ..base import NormalizationContext


class StripTerminalTerminator:
    """Remove trailing `*-` lines to produce page-faithful targets."""

    name = "strip_terminal_terminator"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Strip trailing terminator lines from the transcription."""
        return strip_terminal_terminator_lines(text)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Ensure terminal terminator lines were stripped."""
        if not text:
            return
        lines = text.rstrip("\n").split("\n")
        assert not is_terminator_line(lines[-1]), "Trailing *- line must be stripped"
