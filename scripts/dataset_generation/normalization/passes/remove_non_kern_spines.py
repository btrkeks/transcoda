"""RemoveNonKernSpines pass - removes non-**kern spines (e.g., **text) from transcriptions."""

from __future__ import annotations

import re

from src.core.spine_state import (
    UnsupportedSpineManipulatorError,
    advance_keep_mask,
    is_interpretation_record,
)

from ..base import NormalizationContext


class UnsafeSpineStructureError(ValueError):
    """Raised when safe non-kern spine removal cannot be guaranteed."""

    pass


# Matches the exclusive interpretation line (e.g., **kern\t**text\t**kern)
_EXCLUSIVE_INTERP_LINE = re.compile(r"^(\*\*[^\t\n]+(?:\t\*\*[^\t\n]+)*)$", re.MULTILINE)

# Matches **kern or **ekern spines, including versioned variants like **kern_1.0.
_KERN_SPINE_TOKEN = re.compile(r"^\*\*e?kern(?:_[^\t\n]+)?$")


class RemoveNonKernSpines:
    """
    Removes non-**kern spines from transcriptions.

    This pass handles files with mixed spine types (e.g., **kern + **text for lyrics)
    by removing columns that are not **kern spines.

    Safety: Spine structure can change mid-file due to manipulators. This pass
    tracks structure changes for supported operations (*^, *v, *-) and raises
    UnsafeSpineStructureError for unsupported manipulator types (*x, *+) or
    malformed mixed manipulator records.

    Operations:
    1. Detect exclusive interpretation line to find spine types
    2. Remove columns corresponding to non-**kern spines from all lines
    3. Track spine structure changes on interpretation records
    """

    name = "remove_non_kern_spines"

    def __init__(self) -> None:
        self._initial_keep_mask: list[bool] | None = None
        self._has_non_kern: bool = False

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """Analyze initial spine structure and mark non-kern columns."""
        # Find exclusive interpretation line
        match = _EXCLUSIVE_INTERP_LINE.search(text)
        if not match:
            # No spine declarations found - nothing to do
            self._has_non_kern = False
            self._initial_keep_mask = None
            return

        spine_line = match.group(1)
        spines = spine_line.split("\t")

        # Track which columns are **kern/**ekern (including versioned variants)
        self._initial_keep_mask = [bool(_KERN_SPINE_TOKEN.match(s)) for s in spines]
        self._has_non_kern = not all(self._initial_keep_mask)

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Remove non-kern columns from all lines."""
        if not self._has_non_kern or self._initial_keep_mask is None:
            return text

        keep_mask = list(self._initial_keep_mask)
        lines = text.split("\n")
        result_lines = []

        for line in lines:
            # Skip empty lines and comments
            if not line or line.startswith("!"):
                result_lines.append(line)
                continue

            fields = line.split("\t")

            # Handle lines with unexpected width conservatively (e.g., malformed global lines).
            if len(fields) != len(keep_mask):
                result_lines.append(line)
                continue

            # Keep only kern columns
            kern_fields = [field for field, keep in zip(fields, keep_mask, strict=True) if keep]
            result_lines.append("\t".join(kern_fields))

            if is_interpretation_record(fields):
                try:
                    keep_mask = advance_keep_mask(keep_mask, fields)
                except UnsupportedSpineManipulatorError as exc:
                    raise UnsafeSpineStructureError(
                        "Cannot safely remove non-kern spines: unsupported spine manipulators "
                        "(*x, *+) are not handled."
                    ) from exc
                except ValueError as exc:
                    raise UnsafeSpineStructureError(f"Cannot safely remove non-kern spines: {exc}") from exc

        return "\n".join(result_lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass
