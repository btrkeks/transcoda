"""RepairInterpretationSpacing pass - fixes malformed interpretation lines.

External data (e.g., PRAIG/polish-scores) sometimes has interpretation lines
where spaces are used instead of tabs as field delimiters. This breaks
spine-aware processing in downstream passes like FixNoteBeams.

Example:
    Malformed:  "*\t*Xtuplet    *"   (1 tab, 4 spaces)
    Repaired:   "*\t*Xtuplet\t*"     (2 tabs)
"""

from __future__ import annotations

import re

from src.core.spine_state import UnsupportedSpineManipulatorError, advance_spine_count

from ..base import NormalizationContext

# Match 2+ consecutive spaces that could be misused as tab delimiters
_MULTIPLE_SPACES_RE = re.compile(r" {2,}")


def _is_interpretation_line(line: str) -> bool:
    """Check if all fields in line are interpretation tokens (start with *).

    Args:
        line: A kern line (possibly malformed)

    Returns:
        True if this appears to be an interpretation line
    """
    if not line or not line.strip():
        return False
    # Split by tabs to get current fields
    fields = line.split("\t")
    # All non-empty fields must start with *
    return all(f.strip().startswith("*") for f in fields if f.strip())


def _repair_line(line: str, expected_tabs: int) -> str:
    """Attempt to repair a line by converting multiple spaces to tabs.

    Args:
        line: The potentially malformed line
        expected_tabs: Expected number of tab characters

    Returns:
        Repaired line with correct tab count

    Raises:
        ValueError: If repair cannot produce expected tab count
    """
    actual_tabs = line.count("\t")

    if actual_tabs == expected_tabs:
        return line

    # Try replacing multiple consecutive spaces with tabs
    repaired = _MULTIPLE_SPACES_RE.sub("\t", line)

    if repaired.count("\t") == expected_tabs:
        return repaired

    # If still wrong, try splitting on any whitespace and rejoining with tabs
    tokens = line.split()
    if len(tokens) == expected_tabs + 1:
        return "\t".join(tokens)

    repaired_count = repaired.count("\t")
    raise ValueError(
        f"Cannot repair interpretation line to have {expected_tabs} tabs. "
        f"Original ({actual_tabs} tabs): {repr(line)}, "
        f"After repair attempt ({repaired_count} tabs): {repr(repaired)}, "
        f"Tokens found: {len(tokens)}"
    )


class RepairInterpretationSpacing:
    """
    Repairs malformed interpretation lines in **kern data from external sources.

    This pass tracks spine count through *^ (split) and *v (merge) operations
    to determine the expected number of tab delimiters, then repairs lines
    where the actual tab count doesn't match.

    This pass should run early in the pipeline, after CleanupKern (so ekern
    markers are removed) but before any passes that rely on tab-based spine
    counting (like FixNoteBeams).
    """

    name = "repair_interpretation_spacing"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """Analyze text to determine initial spine count."""
        lines = text.split("\n")

        # Find initial spine count from first non-empty line
        for line in lines:
            if line.strip():
                ctx[self.name] = {"initial_spines": line.count("\t") + 1}
                return

        ctx[self.name] = {"initial_spines": 1}

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Repair interpretation lines with incorrect spacing."""
        lines = text.split("\n")
        result_lines: list[str] = []

        current_spines = ctx.get(self.name, {}).get("initial_spines", 2)

        for line_index, line in enumerate(lines, start=1):
            if not line.strip():
                result_lines.append(line)
                continue

            expected_tabs = current_spines - 1
            actual_tabs = line.count("\t")

            # Check if this is an interpretation line that needs repair
            if _is_interpretation_line(line) and actual_tabs != expected_tabs:
                line = _repair_line(line, expected_tabs)

            result_lines.append(line)

            # Update spine count for next line based on this line's operations
            if line.strip():
                try:
                    current_spines = advance_spine_count(current_spines, line.split("\t"))
                except UnsupportedSpineManipulatorError as exc:
                    raise ValueError(
                        f"Unsupported spine manipulator on line {line_index}: {exc}"
                    ) from exc
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid spine structure on line {line_index}: {exc}"
                    ) from exc

        return "\n".join(result_lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Validate that all interpretation lines have consistent spine count."""
        lines = text.split("\n")
        current_spines = ctx.get(self.name, {}).get("initial_spines", 2)

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            expected_tabs = current_spines - 1
            actual_tabs = line.count("\t")

            if _is_interpretation_line(line) and actual_tabs != expected_tabs:
                raise ValueError(
                    f"Line {i + 1} still has incorrect tab count after repair. "
                    f"Expected {expected_tabs} tabs ({current_spines} spines), "
                    f"got {actual_tabs}. Line: {repr(line)}"
                )

            # Update spine count
            if line.strip():
                try:
                    current_spines = advance_spine_count(current_spines, line.split("\t"))
                except UnsupportedSpineManipulatorError as exc:
                    raise ValueError(f"Unsupported spine manipulator on line {i + 1}: {exc}") from exc
                except ValueError as exc:
                    raise ValueError(f"Invalid spine structure on line {i + 1}: {exc}") from exc
