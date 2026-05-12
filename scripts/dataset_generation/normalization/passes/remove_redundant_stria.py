"""RemoveRedundantStria pass - removes redundant stria lines outside the header."""

from __future__ import annotations

import re

from ..base import NormalizationContext

# Matches stria tokens: *stria1, *stria3, *stria4, *stria6
_STRIA_TOKEN_RE = re.compile(r"^\*stria\d+$")

# Matches lines containing only * tokens (for cleanup after converting redundant stria)
_ALL_STAR_LINE_RE = re.compile(r"(?m)^\*(?:\t\*)*(?:\r?\n|$)")


def _is_stria_token(token: str) -> bool:
    """Check if a token is a stria interpretation (*stria1, *stria3, etc.)."""
    return bool(_STRIA_TOKEN_RE.match(token))


def _is_data_or_barline(line: str) -> bool:
    """Check if a line is data (notes/rests) or a barline."""
    if not line or line.startswith("*") or line.startswith("!"):
        return False
    # Barlines start with = in at least one field
    fields = line.split("\t")
    if any(f.startswith("=") for f in fields):
        return True
    # Data lines have note/rest tokens (not starting with * or !)
    return True


def _is_stria_line(fields: list[str]) -> bool:
    """Check if a line consists only of stria tokens and null interpretations."""
    if not fields:
        return False
    has_stria = False
    for field in fields:
        if _is_stria_token(field):
            has_stria = True
        elif field != "*":
            return False
    return has_stria


class RemoveRedundantStria:
    """
    Removes redundant *stria lines that appear mid-piece with unchanged values.

    Stria (staff line count) is typically set once in the header. When it appears
    again mid-piece with the same value (or null), it's redundant and can be removed.

    The pass:
    1. Scans the header to find canonical stria values per spine
    2. Identifies mid-piece stria lines
    3. Replaces redundant stria lines (same values as header) with null interpretations
    4. Preserves legitimate stria changes (different values)
    5. Removes all-null-interpretation lines created by the replacement
    """

    name = "remove_redundant_stria"

    def __init__(self) -> None:
        self._header_end_idx: int = 0
        self._canonical_stria: list[str | None] = []

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """Parse header to find canonical stria values per spine."""
        lines = text.split("\n")
        self._canonical_stria = []
        self._header_end_idx = 0

        # Find header end (first data or barline)
        for i, line in enumerate(lines):
            if _is_data_or_barline(line):
                self._header_end_idx = i
                break
        else:
            # No data found, entire file is header
            self._header_end_idx = len(lines)

        # Find initial stria values in header
        for i in range(self._header_end_idx):
            line = lines[i]
            if not line:
                continue
            fields = line.split("\t")
            if _is_stria_line(fields):
                # Found the stria line in header
                self._canonical_stria = [
                    f if _is_stria_token(f) else None for f in fields
                ]
                break

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Remove redundant stria lines, replacing with null interpretations."""
        if not self._canonical_stria:
            # No stria in header, nothing to do
            return text

        lines = text.split("\n")
        result_lines = []

        for i, line in enumerate(lines):
            # Keep header lines as-is
            if i < self._header_end_idx:
                result_lines.append(line)
                continue

            if not line:
                result_lines.append(line)
                continue

            fields = line.split("\t")

            # Check if this is a stria line
            if not _is_stria_line(fields):
                result_lines.append(line)
                continue

            # Check if it's redundant (all spines match canonical or are null)
            is_redundant = True
            new_canonical: list[str | None] = list(self._canonical_stria)

            for j, field in enumerate(fields):
                if j >= len(self._canonical_stria):
                    # More spines than expected, keep the line
                    is_redundant = False
                    break

                if field == "*":
                    # Null interpretation, matches anything
                    continue

                canonical = self._canonical_stria[j]
                if _is_stria_token(field):
                    if canonical is None or field != canonical:
                        # Different stria value, this is a real change
                        is_redundant = False
                        new_canonical[j] = field

            if is_redundant:
                # Replace with null interpretations
                null_line = "\t".join("*" for _ in fields)
                result_lines.append(null_line)
            else:
                # Update canonical values and keep the line
                self._canonical_stria = new_canonical
                result_lines.append(line)

        # Remove all-star lines created by our replacements
        result = "\n".join(result_lines)
        return _ALL_STAR_LINE_RE.sub("", result)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass
