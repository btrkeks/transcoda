"""Filter for excessive accidentals caused by malformed transposition in source files."""

import re
from pathlib import Path

from ..base import FilterContext, FilterResult

# Pattern to match notes with more than 2 consecutive sharps or flats
# Matches: pitch letter (a-g or A-G) followed by 3+ sharps or 3+ flats
EXCESSIVE_ACCIDENTALS_PATTERN = re.compile(r"[a-gA-G](#{3,}|-{3,})")


class AccidentalsFilter:
    """
    Filter that rejects files with excessive consecutive accidentals.

    This catches corruption from musicxml2hum when processing malformed
    transposition specifications (e.g., diatonic: 1, chromatic: 0) which
    can produce notes with hundreds of sharps like 'a################'.

    Triple+ accidentals are extremely rare in legitimate music notation
    and almost always indicate conversion errors.
    """

    name = "accidentals"

    def __init__(self, max_consecutive: int = 2):
        """
        Initialize the accidentals filter.

        Args:
            max_consecutive: Maximum allowed consecutive sharps or flats.
                             Default is 2 (double sharp/flat is valid).
        """
        self.max_consecutive = max_consecutive
        # Build pattern dynamically based on max_consecutive
        min_count = max_consecutive + 1
        self.pattern = re.compile(
            rf"[a-gA-G](#{{{min_count},}}|-{{{min_count},}})"
        )

    def check(self, path: Path, ctx: FilterContext) -> FilterResult:
        """
        Check if a file contains excessive consecutive accidentals.

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
            # Extract the matched text for the error message
            matched_text = match.group(0)
            # Truncate if excessively long
            if len(matched_text) > 20:
                matched_text = matched_text[:17] + "..."

            return FilterResult.fail(
                f"Excessive accidentals found: '{matched_text}'",
                matched_position=match.start(),
                matched_text=match.group(0)[:50],  # Limit stored text
            )

        return FilterResult.pass_()
