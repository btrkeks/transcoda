"""NormalizeNullKeysig pass - replaces null interpretations with explicit *k[] in key signature lines."""

import re

from ..base import NormalizationContext

_KEYSIG_PATTERN = re.compile(r"^\*k\[.*\]$")


class NormalizeNullKeysig:
    """
    Replace * with *k[] in lines containing key signatures.

    In Humdrum kern notation, key signatures are represented as *k[...] where
    the brackets contain the accidentals. When one spine has a key signature
    but another has just a null interpretation (*), the grammar requires an
    explicit empty key signature (*k[]) instead.

    Example:
        Input:
            **kern\t**kern
            *clefF4\t*clefG2
            *\t*k[b-e-]

        Output:
            **kern\t**kern
            *clefF4\t*clefG2
            *k[]\t*k[b-e-]

    The null interpretation (*) is replaced with *k[] because the line
    contains a key signature in another spine.

    Example:
        from normalization import Pipeline
        from normalization.passes import NormalizeNullKeysig

        pipeline = Pipeline([NormalizeNullKeysig()])
        result = pipeline("*\\t*k[f#]")
        # Result: "*k[]\\t*k[f#]"
    """

    name = "normalize_null_keysig"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed for this pass."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """
        Replace null interpretations with *k[] in key signature lines.

        Args:
            text: The kern notation string to process
            ctx: The normalization context (not used)

        Returns:
            The kern notation string with null interpretations replaced
        """
        if not text:
            return text

        lines = text.split("\n")
        result_lines = []

        for line in lines:
            tokens = line.split("\t")
            # Check if any token is a key signature
            if any(_KEYSIG_PATTERN.match(t) for t in tokens):
                # Replace null interpretations with explicit empty key signature
                tokens = ["*k[]" if t == "*" else t for t in tokens]
            result_lines.append("\t".join(tokens))

        return "\n".join(result_lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed for this pass."""
        pass
