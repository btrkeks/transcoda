"""RemoveNullLines pass - removes lines containing only null tokens (.)."""

from ..base import NormalizationContext


class RemoveNullLines:
    """
    Removes lines from kern notation that contain only null tokens (.).

    In Humdrum kern notation, a '.' (dot) is a null token that means
    "continue from the previous event". Lines where ALL spines contain
    only null tokens provide no additional information and can be safely
    removed during normalization.

    Example:
        Input:
            **kern\t**kern
            *clefF4\t*clefG2
            .\t.
            4c 4e 4g\t4cc 4ee

        Output:
            **kern\t**kern
            *clefF4\t*clefG2
            4c 4e 4g\t4cc 4ee

    The line ".\t." is removed because all tokens are null.
    Lines like ".\t4c\t." are preserved because they contain a note.

    Example:
        from normalization import Pipeline
        from normalization.passes import RemoveNullLines

        pipeline = Pipeline([RemoveNullLines()])
        result = pipeline("4c\\t4e\\n.\\t.\\n4d\\t4f")
        # Result: "4c\\t4e\\n4d\\t4f" (null-only line removed)
    """

    name = "remove_null_lines"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed for this pass."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """
        Remove lines containing only null tokens (.).

        Args:
            text: The kern notation string to process
            ctx: The normalization context (not used)

        Returns:
            The kern notation string with null-only lines removed
        """
        if not text:
            return text

        # Split into lines
        lines = text.split("\n")

        # Filter out lines where all tokens are "."
        filtered_lines = []
        for line in lines:
            # Split by tabs to get individual spine tokens
            tokens = line.split("\t")

            # Check if all tokens are null tokens (.)
            if all(token == "." for token in tokens):
                # This is a null-only line, skip it
                continue

            # Keep this line
            filtered_lines.append(line)

        # Join back with newlines
        return "\n".join(filtered_lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """
        Validate that the output is still valid kern notation.

        For this pass, we just ensure that no null-only lines remain.
        """
        if not text:
            return

        lines = text.split("\n")
        for i, line in enumerate(lines):
            tokens = line.split("\t")
            if all(token == "." for token in tokens):
                raise ValueError(
                    f"Line {i + 1} still contains only null tokens after RemoveNullLines pass"
                )
