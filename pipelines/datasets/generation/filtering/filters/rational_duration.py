"""Filter for files containing rational durations (mensural notation)."""

import re
from pathlib import Path

from ..base import FilterContext, FilterResult

# Pattern matches rational duration syntax: digits % digits
# Examples: 2%5, 3%7, 4%9 (used in mensural notation like 2%5r for rests)
RATIONAL_DURATION_PATTERN = re.compile(rb"\d+%\d+")


class RationalDurationFilter:
    """
    Filter that rejects files containing rational durations.

    Rational durations (e.g., 2%5r = 5/2 whole-note units) are used in
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
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return FilterResult.fail("File not found")
        except OSError as e:
            return FilterResult.fail(f"OS error: {e}")

        if b"%" not in content:
            return FilterResult.pass_()

        match = RATIONAL_DURATION_PATTERN.search(content)
        if match:
            duration_text = match.group().decode("ascii")
            return FilterResult.fail(
                f"Contains rational duration: {duration_text!r}",
                pattern=duration_text,
                position=match.start(),
            )

        return FilterResult.pass_()
