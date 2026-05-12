"""Filter for files containing rational durations (mensural notation)."""

import re
from pathlib import Path

from ..base import FilterContext, FilterResult

# Pattern matches rational duration syntax: digits % digits
# Examples: 2%5, 3%7, 4%9 (used in mensural notation like 2%5r for rests)
RATIONAL_DURATION_PATTERN = re.compile(r"\d+%\d+")


class RationalDurationFilter:
    """
    Filter that rejects files containing rational durations.

    Rational durations (e.g., 2%5r = "2/5 of a breve rest") are used in
    mensural/early music notation. The '%' character is not in the model's
    tokenizer vocabulary, causing these tokens to become <unk>.
    """

    name = "rational_duration"

    def check(self, path: Path, ctx: FilterContext) -> FilterResult:
        """
        Check if a file contains rational duration notation.

        Args:
            path: Path to the file to check
            ctx: Filter context for content caching

        Returns:
            FilterResult indicating pass (no rational durations) or fail
        """
        content = ctx.get_content(path)

        if content is None:
            error = ctx.get_content_error() or "Unknown read error"
            return FilterResult.fail(error)

        match = RATIONAL_DURATION_PATTERN.search(content)
        if match:
            return FilterResult.fail(
                f"Contains rational duration: {match.group()!r}",
                pattern=match.group(),
                position=match.start(),
            )

        return FilterResult.pass_()
