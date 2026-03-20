"""CanonicalizeBarlines pass - normalizes non-standard barline tokens to grammar-compliant forms."""

from __future__ import annotations

from ..base import NormalizationContext

# Map of non-standard barline tokens to their canonical forms.
_BARLINE_MAP = {
    "=|!": "=",  # Strip visual modifiers (thin|thick) from regular barline
    "==:|!": "=:|!",  # Strip extra = prefix from end-repeat barline
}


class CanonicalizeBarlines:
    """
    Normalizes non-standard barline tokens to grammar-compliant forms.

    Applies per-field replacement on barline lines (lines where all fields
    start with '='), mapping non-standard tokens to canonical equivalents:

    - =|!   → =    (visual rendering modifiers, not in grammar)
    - ==:|! → =:|! (extra = prefix on end-repeat barline)

    Unlike NormalizeFinalBarline, this pass affects ALL barline lines,
    not just the final one.
    """

    name = "canonicalize_barlines"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Replace non-standard barline tokens with canonical forms."""
        if not text:
            return text

        lines = text.splitlines()

        for i, line in enumerate(lines):
            fields = line.split("\t")
            if not all(f.startswith("=") for f in fields):
                continue

            new_fields = [_BARLINE_MAP.get(f, f) for f in fields]
            if new_fields != fields:
                lines[i] = "\t".join(new_fields)

        return "\n".join(lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Assert no non-standard barline tokens remain."""
        if not text:
            return

        non_standard = set(_BARLINE_MAP.keys())
        for line in text.splitlines():
            fields = line.split("\t")
            if not all(f.startswith("=") for f in fields):
                continue
            for f in fields:
                assert (
                    f not in non_standard
                ), f"Non-standard barline token remains: {f!r} in line {line!r}"
