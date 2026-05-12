"""FixSlurs pass - removes orphan slur markers."""

from __future__ import annotations

from src.core.kern_utils import is_note_token

from ..base import NormalizationContext


def _opens_slur(field: str) -> bool:
    return "(" in field


def _closes_slur(field: str) -> bool:
    return ")" in field


class FixSlurs:
    """
    Fixes slur syntax by removing invalid slur markers.

    This pass handles two cases:
    1. Empty slurs: tokens with both '(' and ')' (e.g., '(4AA)' or '4AA()')
       are meaningless since a slur cannot start and end on the same note.
       Both markers are removed.
    2. Orphan closing slurs: ')' markers without a matching '(' are removed.

    Unlike ties (which connect the same pitch), slurs connect phrases of notes
    regardless of pitch. This pass tracks a global open slur count and removes
    closing slurs that don't have a corresponding opening slur.

    Slurs can be nested - a slur can be opened while another is already open.
    The count tracks how many slurs are currently open, and closing slurs are
    only valid when there's at least one open slur.
    """

    name = "fix_slurs"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Remove orphan closing slurs."""
        # Build 3-level nested structure: lines -> fields (tabs) -> tokens (spaces)
        grid: list[list[list[str]]] = [
            [field.split(" ") for field in line.split("\t")] for line in text.splitlines()
        ]
        open_slur_count = 0

        for i, row in enumerate(grid):
            for j, field in enumerate(row):
                for k, token in enumerate(field):
                    if not is_note_token(token):
                        continue

                    # Remove empty slurs (open and close on same note is meaningless)
                    if _opens_slur(token) and _closes_slur(token):
                        grid[i][j][k] = token.replace("(", "").replace(")", "")
                        continue

                    if _opens_slur(token):
                        open_slur_count += 1

                    if _closes_slur(token):
                        if open_slur_count <= 0:
                            # Slur is being closed without being opened
                            # Remove the closing slur
                            grid[i][j][k] = token.replace(")", "")
                        else:
                            open_slur_count -= 1

        return "\n".join("\t".join(" ".join(field) for field in row) for row in grid)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass
