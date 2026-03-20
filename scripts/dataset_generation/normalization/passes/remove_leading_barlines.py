"""RemoveLeadingBarlines pass - removes barlines before first musical content."""

from ..base import NormalizationContext


def is_bar_line(line: str) -> bool:
    """Check if all fields in a line are barlines (start with =)."""
    fields = line.split("\t")
    return bool(fields) and all(f.startswith("=") for f in fields)


def is_data_line(line: str) -> bool:
    """Check if line contains musical data (notes/rests, not interpretations or barlines)."""
    if not line:
        return False
    fields = line.split("\t")
    if not fields:
        return False
    # Not a data line if all fields are null tokens
    if all(f == "." for f in fields):
        return False
    # Not a data line if any field starts with * (interpretation) or = (barline)
    first_field = fields[0]
    if first_field.startswith("*") or first_field.startswith("="):
        return False
    return True


class RemoveLeadingBarlines:
    """
    Remove barlines that appear before the first musical data content.

    In Humdrum kern notation, barlines are lines where all fields start with '='.
    Sometimes a barline appears after the header/interpretation lines but before
    any actual musical data (notes, rests). This pass removes such leading barlines.

    Example:
        Input:
            **kern\\t**kern
            *clefF4\\t*clefG2
            *M3/4\\t*M3/4
            =\\t=
            4c\\t4e

        Output:
            **kern\\t**kern
            *clefF4\\t*clefG2
            *M3/4\\t*M3/4
            4c\\t4e

    The barline "=\\t=" is removed because it appears before any note data.
    Barlines that appear after the first data line are preserved.
    """

    name = "remove_leading_barlines"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed for this pass."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """
        Remove barlines that appear before the first musical data line.

        Args:
            text: The kern notation string to process
            ctx: The normalization context (not used)

        Returns:
            The kern notation string with leading barlines removed
        """
        if not text:
            return text

        lines = text.split("\n")

        # Find index of first data line
        first_data_idx = None
        for i, line in enumerate(lines):
            if is_data_line(line):
                first_data_idx = i
                break

        if first_data_idx is None:
            # No data lines found, return as-is
            return text

        # Filter out barlines before first data line
        result_lines = []
        for i, line in enumerate(lines):
            if i < first_data_idx and is_bar_line(line):
                continue  # Skip leading barlines
            result_lines.append(line)

        return "\n".join(result_lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """
        Validate that no barlines appear before the first data line.

        Raises:
            ValueError: If a barline is found before musical data
        """
        if not text:
            return

        lines = text.split("\n")
        for line in lines:
            if is_data_line(line):
                break
            if is_bar_line(line):
                raise ValueError("Barline found before first musical data")
