"""RemoveNullTies pass - canonicalizes repeated ties and removes null ties/slurs."""

from __future__ import annotations

import re

from src.core.kern_utils import is_note_token

from ..base import NormalizationContext


class RemoveNullTies:
    """
    Canonicalizes repeated ties and removes null ties/slurs on note tokens.

    In **kern notation, a tie that opens and immediately closes on the same
    token (e.g., `4G[]` or `4G[[]]`) is invalid and should be removed.
    Similarly, a slur that opens and closes on the same token (e.g., `4G()`
    or `4G(())`) is meaningless and should be removed.
    This pass also canonicalizes repeated tie markers to the single-marker
    form required by our grammar: `[[ -> [`, `]] -> ]`, `__ -> _`.
    Slur multiplicity (`((`, `))`) is intentionally preserved.

    This pass should run after CanonicalizeNoteOrder so that tie/slur markers
    are already in their canonical positions.

    Examples:
        4G[[ -> 4G[
        4G]] -> 4G]
        4G__ -> 4G_
        4G[] -> 4G
        4G[[]] -> 4G
        4G#[]L -> 4G#L
        4G() -> 4G
        4G(()) -> 4G
        4G()[] -> 4G
    """

    name = "remove_null_ties"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Canonicalize repeated ties and remove null ties/slurs from note tokens."""
        # Build 3-level nested structure: lines -> fields (tabs) -> tokens (spaces)
        grid: list[list[list[str]]] = [
            [field.split(" ") for field in line.split("\t")] for line in text.splitlines()
        ]

        for i, row in enumerate(grid):
            for j, field in enumerate(row):
                for k, token in enumerate(field):
                    if not is_note_token(token):
                        continue

                    # Collapse repeated tie markers to grammar-canonical single markers.
                    # We do this before null cleanup so patterns like [[]] become []
                    # and are then removed in the next step.
                    collapsed_ties = re.sub(r"\[{2,}", "[", token)
                    collapsed_ties = re.sub(r"\]{2,}", "]", collapsed_ties)
                    collapsed_ties = re.sub(r"_{2,}", "_", collapsed_ties)

                    # Remove double null ties/slurs first, then single null ties/slurs
                    # Order matters: [[]] should become "" not "[]"
                    modified = (
                        collapsed_ties.replace("[[]]", "")
                        .replace("[]", "")
                        .replace("(())", "")
                        .replace("()", "")
                    )
                    if modified != token:
                        grid[i][j][k] = modified

        return "\n".join("\t".join(" ".join(field) for field in row) for row in grid)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Optionally verify no null ties remain."""
        # This validation is informational - null ties shouldn't exist after transform
        pass
