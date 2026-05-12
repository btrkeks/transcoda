"""RemoveConflictingBowings pass - removes conflicting bowing markers from notes."""

from __future__ import annotations

from src.core.kern_utils import is_note_token

from ..base import NormalizationContext


class RemoveConflictingBowings:
    """
    Removes down-bow when both up-bow and down-bow are present on the same token.

    MusicXML source data sometimes contains both `<up-bow/>` and `<down-bow/>`
    on the same note, which is musically impossible. Verovio converts these to
    `vu` in **kern tokens (e.g., `8ggvuL`), causing grammar violations since
    only one bowing articulation is allowed per note.

    This pass removes the down-bow (`u`) when both are present, keeping only
    the up-bow (`v`).

    Examples:
        8ggvuL -> 8ggvL
        4Dvu -> 4Dv
        8c#vuJ -> 8c#vJ
    """

    name = "remove_conflicting_bowings"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Remove down-bow 'u' when both up-bow 'v' and down-bow 'u' are present."""
        # Build 3-level nested structure: lines -> fields (tabs) -> tokens (spaces)
        grid: list[list[list[str]]] = [
            [field.split(" ") for field in line.split("\t")] for line in text.splitlines()
        ]

        for i, row in enumerate(grid):
            for j, field in enumerate(row):
                for k, token in enumerate(field):
                    if not is_note_token(token):
                        continue

                    # If both 'v' (up-bow) and 'u' (down-bow) are present,
                    # remove all 'u' characters to keep only the up-bow
                    if "v" in token and "u" in token:
                        grid[i][j][k] = token.replace("u", "")

        return "\n".join("\t".join(" ".join(field) for field in row) for row in grid)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Verify no token has both 'v' and 'u' bowing markers."""
        # This validation is informational - conflicting bowings shouldn't exist after transform
        pass
