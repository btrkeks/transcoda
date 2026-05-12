"""FixTies pass - removes orphan tie markers."""

from __future__ import annotations

from src.core.kern_utils import extract_pitch, is_note_token

from ..base import NormalizationContext


def _opens_tie(field: str) -> bool:
    return "[" in field


def _closes_tie(field: str) -> bool:
    return "]" in field


class FixTies:
    """
    Fixes tie syntax by removing orphan closing ties (]) that don't have
    matching opening ties ([).

    In **kern notation, ties connect notes of the same pitch across time.
    An opening tie marker '[' should be followed by a closing tie marker ']'
    on a note of the same pitch. This pass removes closing tie markers that
    don't have a corresponding opening tie on the same pitch.

    Also removes incorrect underscore ('_') usage when a token already has
    a tie marker ([ or ]).

    This pass tracks open ties by pitch value (e.g., "ee-" for E-flat in octave 4).
    When a closing tie is encountered for a pitch that doesn't have an open tie,
    the closing tie marker is removed.
    """

    name = "fix_ties"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Remove orphan closing ties."""
        # Build 3-level nested structure: lines -> fields (tabs) -> tokens (spaces)
        grid: list[list[list[str]]] = [
            [field.split(" ") for field in line.split("\t")] for line in text.splitlines()
        ]
        open_pitches: set[str] = set()

        for i, row in enumerate(grid):
            for j, field in enumerate(row):
                for k, token in enumerate(field):
                    if not is_note_token(token):
                        continue

                    # Fix all instances of wrong _ usage
                    if "_" in token and ("[" in token or "]" in token):
                        grid[i][j][k] = token.replace("_", "")

                    if _opens_tie(token):
                        pitch = extract_pitch(token)
                        if pitch:
                            open_pitches.add(pitch)

                    if _closes_tie(token):
                        pitch = extract_pitch(token)

                        if pitch not in open_pitches:
                            # Tie is being closed without being opened
                            # Remove the closing tie
                            grid[i][j][k] = token.replace("]", "")
                        else:
                            open_pitches.remove(pitch)

        return "\n".join("\t".join(" ".join(field) for field in row) for row in grid)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass
