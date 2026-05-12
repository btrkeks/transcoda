"""RemoveGraceRests pass - replaces grace rest tokens with null tokens.

Grace rests (e.g., qryy@, qr) are used for spine alignment in **kern when
one voice has grace notes and another doesn't. They are not visually inferable
from sheet music and must be removed before further normalization.
"""

from __future__ import annotations

import re

from ..base import NormalizationContext

# Matches grace rest tokens: optional duration, then q (grace marker), then r (rest),
# then any trailing modifiers. The q must be followed by r (not a pitch letter).
_GRACE_REST_RE = re.compile(r"^\d*\.*q+r")


def _is_grace_rest(token: str) -> bool:
    """Check if a token is a grace rest."""
    if not token or token == "." or token.startswith("*") or token.startswith("="):
        return False
    return bool(_GRACE_REST_RE.match(token))


class RemoveGraceRests:
    """
    Replace grace rest tokens with null tokens (.).

    Grace rests (e.g., qr, qryy@) are structural placeholders used for spine
    alignment when one voice has grace notes and another doesn't. They have no
    visual representation in sheet music and are not valid grammar tokens.

    This pass must run BEFORE CanonicalizeNoteOrder, which would otherwise
    mangle grace rests into bare 'r' (rest without duration).

    Examples:
        qr -> .
        qr@ -> .
        qryy@ -> .
        4qr -> .
    """

    name = "remove_grace_rests"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Replace grace rest tokens with null tokens."""
        grid: list[list[list[str]]] = [
            [field.split(" ") for field in line.split("\t")]
            for line in text.splitlines()
        ]

        for i, row in enumerate(grid):
            for j, field in enumerate(row):
                for k, token in enumerate(field):
                    if _is_grace_rest(token):
                        grid[i][j][k] = "."

        return "\n".join(
            "\t".join(" ".join(field) for field in row) for row in grid
        )

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass
