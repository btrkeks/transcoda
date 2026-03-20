"""CapSlurs pass - caps slurs at 2 per note token."""

from __future__ import annotations

from src.core.kern_utils import is_note_token

from ..base import NormalizationContext


class CapSlurs:
    """
    Caps slurs at a maximum of 2 per note token.

    Malformed **kern data may contain excessive slur markers (e.g., '((((((4ff').
    This pass removes excess '(' and ')' characters, keeping only the first
    MAX_SLURS of each type per token.

    This pass should run BEFORE FixSlurs, which handles orphan/empty slurs,
    since capping should happen first.
    """

    name = "cap_slurs"
    MAX_SLURS = 2

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Cap slurs at MAX_SLURS per token."""
        # Build 3-level nested structure: lines -> fields (tabs) -> tokens (spaces)
        grid: list[list[list[str]]] = [
            [field.split(" ") for field in line.split("\t")] for line in text.splitlines()
        ]

        for i, row in enumerate(grid):
            for j, field in enumerate(row):
                for k, token in enumerate(field):
                    if not is_note_token(token):
                        continue

                    # Count slurs in token
                    open_count = token.count("(")
                    close_count = token.count(")")

                    # Skip if already within limits
                    if open_count <= self.MAX_SLURS and close_count <= self.MAX_SLURS:
                        continue

                    # Remove all slurs, then add back capped amounts
                    stripped = token.replace("(", "").replace(")", "")
                    capped_opens = "(" * min(open_count, self.MAX_SLURS)
                    capped_closes = ")" * min(close_count, self.MAX_SLURS)

                    # Reconstruct: opens before token content, closes after
                    grid[i][j][k] = capped_opens + stripped + capped_closes

        return "\n".join("\t".join(" ".join(field) for field in row) for row in grid)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass
