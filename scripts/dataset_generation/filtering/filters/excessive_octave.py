"""Filter for detecting corrupted pitches with excessive octave repetitions."""

import re
from pathlib import Path

from ..base import FilterContext, FilterResult


class ExcessiveOctaveFilter:
    """
    Reject files with corrupted pitch notation (excessive letter repetitions).

    Some MusicXML source files contain corrupted <display-octave> values
    (e.g., 1434 instead of 4), which musicxml2hum faithfully converts to
    kern notation with 1000+ repeated pitch letters (e.g., 4ggggggggg...).

    Valid range: a-aaaaaa (6 letters max for extreme but valid octaves up to octave 9).
    """

    name = "excessive_octave"

    def __init__(self, max_repetitions: int = 6):
        """
        Initialize the excessive octave filter.

        Args:
            max_repetitions: Maximum allowed pitch letter repetitions.
                            Default 6 covers up to octave 9 (e.g., cccccc).
        """
        self.max_repetitions = max_repetitions
        self.pattern = re.compile(rf"[a-gA-G]{{{max_repetitions + 1},}}")

    def check(self, path: Path, ctx: FilterContext) -> FilterResult:
        """
        Check if a file contains excessive pitch letter repetitions.

        Args:
            path: Path to the file to check
            ctx: Filter context for content caching

        Returns:
            FilterResult indicating pass or fail
        """
        content = ctx.get_content(path)

        if content is None:
            error = ctx.get_content_error() or "Unknown read error"
            return FilterResult.fail(error)

        match = self.pattern.search(content)
        if match:
            matched_text = match.group(0)
            display_text = matched_text[:20] + "..." if len(matched_text) > 20 else matched_text
            return FilterResult.fail(
                f"Excessive pitch repetitions: '{display_text}' ({len(matched_text)} chars)",
                matched_position=match.start(),
                repetition_count=len(matched_text),
            )

        return FilterResult.pass_()
