"""Spine termination validation filter."""

from pathlib import Path

from ..base import FilterContext, FilterResult


class TerminationFilter:
    """
    Filter that checks if a kern file has proper spine terminators.

    A valid kern file should end with a data line where all tokens
    are spine terminators (*-). Trailing comment lines (starting with !!!)
    are skipped when finding the last data line.
    """

    name = "termination"

    def check(self, path: Path, ctx: FilterContext) -> FilterResult:
        """
        Check if a file has proper spine terminators.

        Args:
            path: Path to the file to check
            ctx: Filter context for content caching

        Returns:
            FilterResult indicating pass or fail
        """
        content = ctx.get_content(path)

        if content is None:
            return FilterResult.fail("Could not read file content")

        text = content.rstrip("\n")
        if not text:
            return FilterResult.fail("Empty file")

        lines = text.split("\n")

        # Skip trailing comment lines (!!!)
        while lines and lines[-1].startswith("!!!"):
            lines.pop()

        if not lines:
            return FilterResult.fail("File contains only comments")

        last_line = lines[-1]
        tokens = last_line.split("\t")

        if not tokens:
            return FilterResult.fail("Last data line is empty")

        if not all(t == "*-" for t in tokens):
            non_terminators = [t for t in tokens if t != "*-"]
            return FilterResult.fail(
                "Missing spine terminators",
                last_line=last_line,
                non_terminators=non_terminators[:5],  # Limit for readability
            )

        return FilterResult.pass_()
