"""RemoveContradictoryAccidentals pass - resolves contradictory accidental combinations."""

from __future__ import annotations

from src.core.kern_utils import is_note_token

from ..base import NormalizationContext

# Contradictory accidental patterns (longer patterns first for greedy matching)
# After CanonicalizeNoteOrder, accidentals are grouped together after pitch
_CONTRADICTORY_PATTERNS = [
    ("##n", "n"),  # double sharp + natural → natural
    ("--n", "n"),  # double flat + natural → natural
    ("#n", "n"),  # sharp + natural → natural
    ("-n", "n"),  # flat + natural → natural
    ("#-", "-"),  # sharp + flat → flat (rightmost wins)
    ("-#", "#"),  # flat + sharp → sharp (rightmost wins)
]


class RemoveContradictoryAccidentals:
    """
    Resolves contradictory accidental combinations in note tokens.

    MusicXML source data sometimes contains both `<alter>` (sounding pitch) and
    `<accidental>` (display) that contradict each other. For example, a note with
    `<alter>1</alter>` (C#) and `<accidental>natural</accidental>` produces `c#n`
    after musicxml2hum conversion.

    This pass resolves contradictions by keeping the rightmost accidental, which
    represents the visual/display intention.

    Must run AFTER CanonicalizeNoteOrder, which guarantees all accidentals are
    grouped together immediately after the pitch.

    Examples:
        16c#n  -> 16cn
        4e-n   -> 4en
        4c##n  -> 4cn
        4c--n  -> 4cn
    """

    name = "remove_contradictory_accidentals"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Resolve contradictory accidentals in note tokens."""
        grid: list[list[list[str]]] = [
            [field.split(" ") for field in line.split("\t")]
            for line in text.splitlines()
        ]

        for i, row in enumerate(grid):
            for j, field in enumerate(row):
                for k, token in enumerate(field):
                    if not is_note_token(token):
                        continue
                    resolved = _resolve_accidentals(token)
                    if resolved != token:
                        grid[i][j][k] = resolved

        return "\n".join(
            "\t".join(" ".join(field) for field in row) for row in grid
        )

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass


def _resolve_accidentals(token: str) -> str:
    """Replace contradictory accidental combinations with the resolved accidental."""
    for pattern, replacement in _CONTRADICTORY_PATTERNS:
        if pattern in token:
            token = token.replace(pattern, replacement)
    return token
