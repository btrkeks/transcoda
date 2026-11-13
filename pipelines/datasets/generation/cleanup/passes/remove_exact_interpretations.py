"""Remove exact tandem interpretation fields that the parser should not model."""

from __future__ import annotations

from ..base import CleanupContext

DEFAULT_REMOVED_INTERPRETATIONS = frozenset({"*staff0", "*Trd0c0"})


class RemoveExactInterpretations:
    """Replace configured interpretation fields with null interpretations."""

    name = "remove_exact_interpretations"

    def __init__(
        self,
        removed_interpretations: frozenset[str] = DEFAULT_REMOVED_INTERPRETATIONS,
    ) -> None:
        self.removed_interpretations = removed_interpretations

    def transform(self, text: str, ctx: CleanupContext) -> str:
        removed_counts = {token: 0 for token in sorted(self.removed_interpretations)}
        output_lines: list[str] = []

        for line in text.splitlines():
            output_fields: list[str] = []
            for field in line.split("\t"):
                if field in self.removed_interpretations:
                    removed_counts[field] += 1
                    output_fields.append("*")
                else:
                    output_fields.append(field)

            output_lines.append("\t".join(output_fields))

        ctx[self.name] = {
            "removed_total": sum(removed_counts.values()),
            "removed_counts": removed_counts,
        }
        return "\n".join(output_lines)
