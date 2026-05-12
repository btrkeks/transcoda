"""SymbolsBeforeSplit pass - reorders interpretation lines before spine splits."""

from src.core.kern_utils import is_spinesplit_line

from ..base import NormalizationContext


class SymbolsBeforeSplit:
    """
    Reorders tandem interpretations to appear before spine split symbols.

    In Humdrum kern notation, spine manipulation symbols (*^, *v, etc.) can appear
    before or after tandem interpretations (clefs, key signatures, time signatures).
    For model training, we decide to place them BEFORE the spine splits in order to
    have more accurate validation errors.

    This pass detects such cases and reorders the lines accordingly.

    Example:
        Input:
            *\t*^
            *clefF4\t*clefG2\t*clefG2
            *k[b-]\t*k[b-]\t*k[b-]

        Output:
            *clefF4\t*clefG2
            *k[b-]\t*k[b-]
            *\t*^
    """

    name = "symbols_before_split"

    def _is_interpretation_line(self, line: str) -> bool:
        """Check if all columns are interpretation tokens (start with * followed by a letter)."""
        if not line:
            return False
        columns = line.split("\t")
        return all(col.startswith("*") and len(col) >= 2 and col[1].isalpha() for col in columns)

    def _find_split_indices(self, line: str) -> list[int]:
        """Find indices of a line that are spine split tokens."""
        split_indices = []
        for i, token in enumerate(line.split("\t")):
            if token == "*^":
                split_indices.append(i)
        return split_indices

    def _maybe_move_interpretation_before_split(
        self, lines: list[str], split_line_index: int
    ) -> list[str]:
        """ "Moves the line at index split_index + 1 before the split line at split_index, if syntax allows."""
        split_token_indices = self._find_split_indices(lines[split_line_index])
        assert split_token_indices, "No split indices found in supposed split line. This is a bug."

        interpretation_tokens = lines[split_line_index + 1].split("\t")

        # Docs: "Splitting a spine causes a new spine (to the right of the current spine) to be spawned."
        # Thus, we remove tokens to the right of each split index if they can be merged, i.e. if they are the same.
        num_successful_splits = 0
        for j, split_idx in reversed(list(enumerate(split_token_indices))):
            expanded_idx = split_idx + j  # j = count of prior splits (lower indices)
            if expanded_idx + 1 < len(interpretation_tokens):
                if interpretation_tokens[expanded_idx + 1] == interpretation_tokens[expanded_idx]:
                    del interpretation_tokens[expanded_idx + 1]
                    num_successful_splits += 1

        if num_successful_splits == len(split_token_indices):
            # All splits were successful, we can move the line
            new_interpretation_line = "\t".join(interpretation_tokens)
            split_line = lines[split_line_index]
            lines[split_line_index] = new_interpretation_line
            lines[split_line_index + 1] = split_line
            return lines

        # If there were any unsuccessful splits, we do not move the line
        return lines

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No-op preparation for now."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Reorder interpretation lines to appear before spine splits."""
        lines = text.split("\n")
        i = 0

        while i < len(lines) - 1:
            line = lines[i]

            if is_spinesplit_line(line):
                if self._is_interpretation_line(lines[i + 1]):
                    lines = self._maybe_move_interpretation_before_split(lines, i)

            i = i + 1

        return "\n".join(lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No-op validation for now."""
        pass
