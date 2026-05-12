"""UTF-8 encoding validation filter."""

from pathlib import Path

from ..base import FilterContext, FilterResult


class UTF8Filter:
    """
    Filter that checks if a file is valid UTF-8.

    This filter uses the FilterContext's content caching mechanism,
    which attempts to read the file as UTF-8 and caches the result.
    """

    name = "utf8"

    def check(self, path: Path, ctx: FilterContext) -> FilterResult:
        """
        Check if a file can be read as valid UTF-8.

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

        return FilterResult.pass_()
