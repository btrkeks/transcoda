"""CanonicalizeSpineSplits pass - coalesces equivalent adjacent spine splits."""

from src.core.kern_utils import is_spinesplit_line

from ..base import NormalizationContext


def _split_fields(line: str) -> list[str]:
    return line.split("\t")


def _expanded_original_indices(split_by_original: list[bool]) -> list[int]:
    """Map post-split field positions back to original spine indices."""
    indices: list[int] = []
    for original_index, is_split in enumerate(split_by_original):
        indices.append(original_index)
        if is_split:
            indices.append(original_index)
    return indices


def _try_coalesce_split_lines(previous: str, following: str) -> str | None:
    """Return a coalesced split line when two adjacent split lines are equivalent."""
    previous_fields = _split_fields(previous)
    following_fields = _split_fields(following)
    split_by_original = [field == "*^" for field in previous_fields]
    original_indices = _expanded_original_indices(split_by_original)

    if len(following_fields) != len(original_indices):
        return None

    next_split_by_original = split_by_original[:]
    for field_index, field in enumerate(following_fields):
        if field != "*^":
            continue

        original_index = original_indices[field_index]
        if split_by_original[original_index]:
            return None
        next_split_by_original[original_index] = True

    return "\t".join("*^" if is_split else "*" for is_split in next_split_by_original)


class CanonicalizeSpineSplits:
    """
    Coalesces adjacent split-only interpretation records when topology-equivalent.

    A later split line can be folded into an earlier split line only when every
    later split targets an original spine that was not already split. Splits on
    newly-created subspines are intentionally preserved because they are not
    equivalent to simultaneous splits of the original spines.
    """

    name = "canonicalize_spine_splits"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed for this pass."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Coalesce safe adjacent spine-split lines."""
        if not text:
            return text

        result: list[str] = []
        for line in text.split("\n"):
            if result and is_spinesplit_line(result[-1]) and is_spinesplit_line(line):
                coalesced = _try_coalesce_split_lines(result[-1], line)
                if coalesced is not None:
                    result[-1] = coalesced
                    continue
            result.append(line)

        return "\n".join(result)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Validation is handled by the final ValidateSpineOperations pass."""
        pass
