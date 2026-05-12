"""MergeSplitNormalizer pass - normalizes redundant spine merge/split patterns around barlines."""

from enum import Enum, auto

from src.core.kern_utils import (
    is_bar_line,
    is_spinemerge_line,
    is_spinesplit_line,
    is_terminator_line,
)

from ..base import NormalizationContext


class _State(Enum):
    NORMAL = auto()
    MERGING = auto()
    SAW_BARLINE = auto()


def _get_spine_count(line: str) -> int:
    """Get the number of spines (tab-separated fields) in a line."""
    return line.count("\t") + 1


def _compute_post_merge_spine_count(merge_line: str) -> int:
    """Compute spine count after applying a merge line.

    Adjacent *v tokens merge into one spine.
    """
    tokens = merge_line.split("\t")
    count = 0
    i = 0
    while i < len(tokens):
        if tokens[i] == "*v":
            count += 1
            while i < len(tokens) and tokens[i] == "*v":
                i += 1
            continue
        else:
            count += 1
            i += 1
    return count


def _compute_post_split_spine_count(split_line: str) -> int:
    """Compute spine count after applying a split line.

    Each *^ creates an additional spine.
    """
    tokens = split_line.split("\t")
    return sum(2 if t == "*^" else 1 for t in tokens)


def _create_barline_line(token: str, spine_count: int) -> str:
    """Create a barline with the given token repeated for each spine."""
    return "\t".join([token] * spine_count)


def _create_terminator_line(spine_count: int) -> str:
    """Create a terminator line with the given spine count."""
    return "\t".join(["*-"] * spine_count)


class MergeSplitNormalizer:
    """
    Normalizes redundant spine merge/split patterns around barlines in **kern notation.

    This pass uses a state machine to handle patterns where spines are temporarily
    merged before a barline and then split (or terminated) after. These patterns
    can be simplified to just the barline across all original spines.

    The state machine tracks:
    - NORMAL: Looking for merge lines
    - MERGING: Accumulating merge lines, tracking original spine count
    - SAW_BARLINE: After merge(s) and barline, looking for split/terminate

    Three types of patterns are handled:

    Type 1 - Merge → Barline → Split:
        When split restores original spine count, collapse to barline at original count.
        Example: "*v\\t*v\\t*\\n=\\t=\\n*\\t*^" → "=\\t=\\t="

    Type 2 - Merge → Barline → Terminate:
        Collapse to barline + terminator at original spine count.
        Example: "*\\t*v\\t*v\\n=\\t=\\n*-\\t*-" → "=\\t=\\t=\\n*-\\t*-\\t*-"

    Type 3 - Merge → Terminate (no barline):
        Collapse to terminator at original spine count.
        Example: "*\\t*v\\t*v\\n*-\\t*-" → "*-\\t*-\\t*-"

    This approach handles N-spine patterns (not just 3-spine) and chained merges.
    """

    name = "merge_split_normalizer"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed for this pass."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """
        Normalize spine merge/split patterns around barlines.

        Args:
            text: The kern notation string to process
            ctx: The normalization context (not used)

        Returns:
            The kern notation string with merge/split patterns normalized
        """
        if not text:
            return text

        lines = text.split("\n")
        result: list[str] = []

        state = _State.NORMAL
        original_spine_count = 0
        current_spine_count = 0
        accumulated_merges: list[str] = []
        barline_token = ""

        for line in lines:
            if state == _State.NORMAL:
                if is_spinemerge_line(line):
                    state = _State.MERGING
                    original_spine_count = _get_spine_count(line)
                    current_spine_count = _compute_post_merge_spine_count(line)
                    accumulated_merges = [line]
                else:
                    result.append(line)

            elif state == _State.MERGING:
                if is_spinemerge_line(line):
                    accumulated_merges.append(line)
                    current_spine_count = _compute_post_merge_spine_count(line)
                elif is_bar_line(line):
                    state = _State.SAW_BARLINE
                    barline_token = line.split("\t")[0]
                elif is_terminator_line(line):
                    result.append(_create_terminator_line(original_spine_count))
                    state = _State.NORMAL
                else:
                    result.extend(accumulated_merges)
                    result.append(line)
                    state = _State.NORMAL

            elif state == _State.SAW_BARLINE:
                if is_spinesplit_line(line):
                    post_split = _compute_post_split_spine_count(line)
                    if post_split == original_spine_count:
                        result.append(
                            _create_barline_line(barline_token, original_spine_count)
                        )
                    else:
                        result.extend(accumulated_merges)
                        result.append(
                            _create_barline_line(barline_token, current_spine_count)
                        )
                        result.append(line)
                    state = _State.NORMAL
                elif is_terminator_line(line):
                    result.append(
                        _create_barline_line(barline_token, original_spine_count)
                    )
                    result.append(_create_terminator_line(original_spine_count))
                    state = _State.NORMAL
                else:
                    result.extend(accumulated_merges)
                    result.append(
                        _create_barline_line(barline_token, current_spine_count)
                    )
                    if is_spinemerge_line(line):
                        state = _State.MERGING
                        original_spine_count = current_spine_count
                        current_spine_count = _compute_post_merge_spine_count(line)
                        accumulated_merges = [line]
                    else:
                        result.append(line)
                        state = _State.NORMAL

        # Handle incomplete sequences at EOF
        if state == _State.MERGING:
            result.extend(accumulated_merges)
        elif state == _State.SAW_BARLINE:
            result.extend(accumulated_merges)
            result.append(_create_barline_line(barline_token, current_spine_count))

        return "\n".join(result)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Validation is optional for this pass."""
        pass
