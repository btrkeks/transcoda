"""RemoveRedundantTimeSignatures pass - removes meter lines when equivalent mensuration exists."""

from __future__ import annotations

import re

from ..base import NormalizationContext

# Mensuration sign mappings to equivalent meters (case-insensitive)
_MENSURATION_TO_METER: dict[str, str] = {
    "*met(c)": "*M4/4",
    "*met(C)": "*M4/4",
    "*met(c|)": "*M2/2",
    "*met(C|)": "*M2/2",
}

# Regex patterns
_MENSURATION_RE = re.compile(r"^\*met\([cC]\|?\)$")
_METER_RE = re.compile(r"^\*M\d+/\d+$")
_SPINE_OP_RE = re.compile(r"^\*[\^vx+\-]$")

# Matches lines containing only * tokens (for cleanup after removing meter lines)
_ALL_STAR_LINE_RE = re.compile(r"(?m)^\*(?:\t\*)*(?:\r?\n|$)")


def _is_mensuration_token(token: str) -> bool:
    """Check if a token is a mensuration sign."""
    return bool(_MENSURATION_RE.match(token))


def _is_meter_token(token: str) -> bool:
    """Check if a token is a meter/time signature."""
    return bool(_METER_RE.match(token))


def _is_spine_op(token: str) -> bool:
    """Check if a token is a spine manipulation operator."""
    return bool(_SPINE_OP_RE.match(token))


def _is_data_or_barline(line: str) -> bool:
    """Check if a line is data (notes/rests) or a barline."""
    if not line or line.startswith("*") or line.startswith("!"):
        return False
    fields = line.split("\t")
    # Barlines start with = in at least one field
    if any(f.startswith("=") for f in fields):
        return True
    # Data lines have note/rest tokens
    return True


def _get_equivalent_meter(mensuration: str) -> str | None:
    """Get the equivalent meter for a mensuration sign."""
    return _MENSURATION_TO_METER.get(mensuration)


class RemoveRedundantTimeSignatures:
    """
    Removes redundant meter lines when an equivalent mensuration sign is present.

    When both a mensuration sign (*met(c)) and its equivalent time signature (*M4/4)
    are present, Verovio renders only the mensuration sign. This creates a mismatch
    between the rendered image and the transcription.

    This pass removes the meter line when:
    1. A mensuration sign line precedes the meter line
    2. No data/barlines/spine-ops occur between them
    3. ALL spines either match their expected equivalent meter or are null

    Mappings:
    - *met(c) / *met(C) -> *M4/4
    - *met(c|) / *met(C|) -> *M2/2
    """

    name = "remove_redundant_time_signatures"

    def __init__(self) -> None:
        # Pending expected meters per spine (None = no pending mensuration)
        self._pending_meters: list[str | None] = []

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """Initialize state for transformation."""
        self._pending_meters = []

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Remove meter lines that are redundant due to equivalent mensuration signs."""
        lines = text.split("\n")
        result_lines = []
        pending_meters: list[str | None] = []

        for line in lines:
            if not line:
                result_lines.append(line)
                continue

            fields = line.split("\t")

            # Reset tracking on data lines, barlines, or spine operations
            if _is_data_or_barline(line):
                pending_meters = []
                result_lines.append(line)
                continue

            # Check for spine manipulation operators
            if any(_is_spine_op(f) for f in fields):
                pending_meters = []
                result_lines.append(line)
                continue

            # Check if this is a mensuration line
            is_mensuration_line = False
            has_mensuration = False
            new_pending: list[str | None] = []

            for field in fields:
                if _is_mensuration_token(field):
                    has_mensuration = True
                    new_pending.append(_get_equivalent_meter(field))
                elif field == "*":
                    new_pending.append(None)  # Null matches anything
                else:
                    new_pending.append(None)

            if has_mensuration:
                # Check if ALL non-null tokens are mensuration tokens
                is_mensuration_line = all(
                    _is_mensuration_token(f) or f == "*" for f in fields
                )

            if is_mensuration_line:
                pending_meters = new_pending
                result_lines.append(line)
                continue

            # Check if this is a meter line with pending mensuration
            if pending_meters:
                is_meter_line = all(_is_meter_token(f) or f == "*" for f in fields)
                has_meter = any(_is_meter_token(f) for f in fields)

                if is_meter_line and has_meter:
                    # Check if all spines match their expected meters (or both are null)
                    is_redundant = True

                    if len(fields) != len(pending_meters):
                        # Spine count mismatch, keep the line
                        is_redundant = False
                    else:
                        for i, field in enumerate(fields):
                            expected = pending_meters[i]
                            if field == "*":
                                # Null field - matches if expected is also null
                                continue
                            elif expected is None:
                                # We have a meter but no expected value - not redundant
                                is_redundant = False
                                break
                            elif field != expected:
                                # Meter doesn't match expected
                                is_redundant = False
                                break

                    if is_redundant:
                        # Replace with null interpretations
                        null_line = "\t".join("*" for _ in fields)
                        result_lines.append(null_line)
                        pending_meters = []
                        continue

            result_lines.append(line)

        # Remove all-star lines created by our replacements
        result = "\n".join(result_lines)
        return _ALL_STAR_LINE_RE.sub("", result)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass
