"""Filter for files with too many kern staves."""

from pathlib import Path

from ..base import FilterContext, FilterResult


class StavesFilter:
    """
    Filter that rejects files with 8 or more exact **kern spines.

    The stave count is taken from the first tab-separated line containing an
    exact **kern token. Other spine types, including **ekern and versioned
    **kern variants, are ignored.
    """

    name = "staves"
    max_staves = 7

    def check(self, path: Path, ctx: FilterContext) -> FilterResult:
        saw_line = False

        try:
            with path.open("rb") as f:
                for line in f:
                    saw_line = True
                    tokens = line.rstrip(b"\n").split(b"\t")
                    num_staves = sum(1 for token in tokens if token == b"**kern")
                    if num_staves == 0:
                        continue

                    if num_staves > self.max_staves:
                        return FilterResult.fail(
                            "Too many staves",
                            num_staves=num_staves,
                            max_staves=self.max_staves,
                        )

                    return FilterResult.pass_()
        except FileNotFoundError:
            return FilterResult.fail("File not found")
        except OSError as e:
            return FilterResult.fail(f"OS error: {e}")

        if not saw_line:
            return FilterResult.fail("Empty file")
        return FilterResult.fail("No **kern declaration found")
