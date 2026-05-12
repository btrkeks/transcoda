"""NormalizeFinalBarline pass - normalizes non-canonical final barlines."""

from __future__ import annotations

from ..base import NormalizationContext

# Barline tokens that should be rewritten to == on the final barline line.
_REPLACE_TOKENS = {"=||", "="}

# Barline tokens containing a start-repeat component that should become =:|!
_START_REPEAT_TOKENS = {"=!|:", "=:|!|:", "=:!|:"}


class NormalizeFinalBarline:
    """
    Normalizes non-canonical final barlines.

    Handles these cases on the last barline line:
    1. =|| or bare = → == (canonical final barline)
    2. =!|: → =:|! (start repeat flipped to end repeat)
    3. =:|!|: or =:!|: → =:|! (end-and-start repeat stripped to end repeat)

    Only the last barline line is affected — interior barlines are left alone.

    Note: ==:|! → =:|! is handled by CanonicalizeBarlines which runs first.
    """

    name = "normalize_final_barline"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Replace non-canonical barlines with == on the final barline line."""
        if not text:
            return text

        lines = text.splitlines()

        # Find the last barline line (all spines start with '=')
        last_barline_idx = None
        for i in range(len(lines) - 1, -1, -1):
            fields = lines[i].split("\t")
            if all(f.startswith("=") for f in fields):
                last_barline_idx = i
                break

        if last_barline_idx is None:
            return text

        # Pass 1: Replace =|| and bare = with == (requires all spines to match)
        fields = lines[last_barline_idx].split("\t")
        if all(f in _REPLACE_TOKENS for f in fields):
            lines[last_barline_idx] = "\t".join("==" for _ in fields)

        # Pass 2: Normalize repeat barlines that contain a start-repeat component
        # A piece cannot end with a start-repeat; strip it to just the end-repeat.
        fields = lines[last_barline_idx].split("\t")
        if any(f in _START_REPEAT_TOKENS for f in fields):
            lines[last_barline_idx] = "\t".join(
                "=:|!" if f in _START_REPEAT_TOKENS else f for f in fields
            )

        return "\n".join(lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Assert no non-canonical barline remains as the final barline line."""
        if not text:
            return

        lines = text.splitlines()

        # Find the last barline line
        for i in range(len(lines) - 1, -1, -1):
            fields = lines[i].split("\t")
            if all(f.startswith("=") for f in fields):
                assert not all(
                    f in _REPLACE_TOKENS for f in fields
                ), f"Final barline line still contains non-canonical barline: {lines[i]!r}"
                assert not any(
                    f in _START_REPEAT_TOKENS for f in fields
                ), f"Final barline line still contains start-repeat barline: {lines[i]!r}"
                break
