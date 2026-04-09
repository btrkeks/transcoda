"""Helpers for stitching multiple **kern snippets into one transcription."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.core.kern_utils import is_bar_line, is_terminator_line
from src.core.spine_state import advance_spine_count, is_interpretation_record

Ancestry = tuple[int, ...]
Lineage = tuple[Ancestry, ...]


@dataclass(frozen=True)
class SpineTopologySummary:
    """Summarize the boundary spine widths of a **kern transcription."""

    initial_spine_count: int | None
    terminal_spine_count: int | None


@dataclass(frozen=True)
class SpineTopologyDiagnostic:
    """Describe the first topology parsing failure in a **kern transcription."""

    reason_code: str
    message: str
    line_number: int | None = None
    line_text: str | None = None
    expected_spine_count: int | None = None
    actual_spine_count: int | None = None


@dataclass(frozen=True)
class _LineageAnalysis:
    """Boundary ancestry state inferred from a **kern transcription."""

    initial_lineage: Lineage | None
    terminal_lineage: Lineage | None


@dataclass(frozen=True)
class _LineageAnalysisResult:
    analysis: _LineageAnalysis | None
    diagnostic: SpineTopologyDiagnostic | None


def _split_text_preserving_trailing_newline(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _is_ignored_line(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("!!")


def _initial_lineage(width: int) -> Lineage:
    return tuple((idx,) for idx in range(width))


def _merge_ancestries(ancestries: Sequence[Ancestry]) -> Ancestry:
    merged: list[int] = []
    seen: set[int] = set()
    for ancestry in ancestries:
        for root in ancestry:
            if root in seen:
                continue
            seen.add(root)
            merged.append(root)
    return tuple(merged)


def _lineage_signature(lineage: Lineage) -> tuple[int, ...]:
    return tuple(ancestry[0] for ancestry in lineage)


def _advance_lineage(lineage: Lineage, fields: Sequence[str]) -> Lineage:
    expected_width = advance_spine_count(len(lineage), fields)

    next_lineage: list[Ancestry] = []
    i = 0
    fields_list = list(fields)
    while i < len(fields_list):
        token = fields_list[i]
        if token == "*v":
            j = i
            while j < len(fields_list) and fields_list[j] == "*v":
                j += 1
            next_lineage.append(_merge_ancestries(lineage[i:j]))
            i = j
            continue
        if token == "*^":
            next_lineage.extend([lineage[i], lineage[i]])
            i += 1
            continue
        if token == "*-":
            i += 1
            continue
        next_lineage.append(lineage[i])
        i += 1

    if len(next_lineage) != expected_width:
        raise ValueError(
            f"Lineage width mismatch: expected {expected_width}, got {len(next_lineage)}"
        )
    return tuple(next_lineage)


def _analyze_lineage_result(text: str) -> _LineageAnalysisResult:
    current_lineage: Lineage | None = None
    initial_lineage: Lineage | None = None
    terminal_lineage: Lineage | None = None

    for line_number, line in enumerate(_split_text_preserving_trailing_newline(text), start=1):
        if _is_ignored_line(line):
            continue

        fields = line.split("\t")
        if current_lineage is None:
            current_lineage = _initial_lineage(len(fields))
            initial_lineage = current_lineage
            terminal_lineage = current_lineage

        if len(fields) != len(current_lineage):
            return _LineageAnalysisResult(
                analysis=None,
                diagnostic=SpineTopologyDiagnostic(
                    reason_code="width_mismatch",
                    message=(
                        f"line {line_number}: expected {len(current_lineage)} spine field(s), "
                        f"got {len(fields)}"
                    ),
                    line_number=line_number,
                    line_text=line,
                    expected_spine_count=len(current_lineage),
                    actual_spine_count=len(fields),
                ),
            )

        if not is_interpretation_record(fields):
            terminal_lineage = current_lineage
            continue

        try:
            current_lineage = _advance_lineage(current_lineage, fields)
        except ValueError as exc:
            return _LineageAnalysisResult(
                analysis=None,
                diagnostic=SpineTopologyDiagnostic(
                    reason_code="invalid_interpretation_record",
                    message=f"line {line_number}: {exc}",
                    line_number=line_number,
                    line_text=line,
                ),
            )

        if not is_terminator_line(line):
            terminal_lineage = current_lineage

    analysis = _LineageAnalysis(
        initial_lineage=initial_lineage,
        terminal_lineage=terminal_lineage,
    )
    if analysis.initial_lineage is None or analysis.terminal_lineage is None:
        return _LineageAnalysisResult(
            analysis=None,
            diagnostic=SpineTopologyDiagnostic(
                reason_code="empty_source",
                message="source does not contain any significant records",
            ),
        )
    return _LineageAnalysisResult(analysis=analysis, diagnostic=None)


def _analyze_lineage(text: str) -> _LineageAnalysis | None:
    return _analyze_lineage_result(text).analysis


def diagnose_spine_topology(text: str) -> SpineTopologyDiagnostic | None:
    """Return the first topology parsing diagnostic for ``text``, if any."""
    return _analyze_lineage_result(text).diagnostic


def _equal_ancestry_runs(lineage: Lineage) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    signature = _lineage_signature(lineage)
    i = 0
    while i < len(signature):
        j = i + 1
        while j < len(signature) and signature[j] == signature[i]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _select_merge_runs(lineage: Lineage) -> tuple[tuple[int, int], ...]:
    selected: list[tuple[int, int]] = []
    for start, end in _equal_ancestry_runs(lineage):
        if end - start < 2:
            continue
        if selected and start == selected[-1][1]:
            continue
        selected.append((start, end))
    return tuple(selected)


def _apply_selected_merge_runs(lineage: Lineage, runs: Sequence[tuple[int, int]]) -> Lineage:
    if not runs:
        return lineage

    runs_by_start = {start: end for start, end in runs}
    next_lineage: list[Ancestry] = []
    i = 0
    while i < len(lineage):
        end = runs_by_start.get(i)
        if end is None:
            next_lineage.append(lineage[i])
            i += 1
            continue
        next_lineage.append(lineage[i])
        i = end
    return tuple(next_lineage)


def _build_merge_row(width: int, runs: Sequence[tuple[int, int]]) -> str:
    fields = ["*"] * width
    for start, end in runs:
        for idx in range(start, end):
            fields[idx] = "*v"
    return "\t".join(fields)


def _collapse_fields(fields: Sequence[str], lineage: Lineage) -> list[str]:
    collapsed: list[str] = []
    for start, _ in _equal_ancestry_runs(lineage):
        collapsed.append(fields[start])
    return collapsed


def _find_repair_anchor(lines: Sequence[str]) -> tuple[int, tuple[int, ...]] | None:
    significant = [(idx, line) for idx, line in enumerate(lines) if not _is_ignored_line(line)]
    if not significant:
        return None

    trailing_terminators: list[int] = []
    cursor = len(significant) - 1
    while cursor >= 0 and is_terminator_line(significant[cursor][1]):
        trailing_terminators.append(significant[cursor][0])
        cursor -= 1

    if cursor < 0:
        return None

    anchor_idx, anchor_line = significant[cursor]
    if not is_bar_line(anchor_line):
        return None

    return anchor_idx, tuple(reversed(trailing_terminators))


def summarize_spine_topology(text: str) -> SpineTopologySummary:
    """Return the initial and terminal active spine counts for ``text``."""
    analysis = _analyze_lineage(text)
    if analysis is None or analysis.initial_lineage is None or analysis.terminal_lineage is None:
        return SpineTopologySummary(initial_spine_count=None, terminal_spine_count=None)

    return SpineTopologySummary(
        initial_spine_count=len(analysis.initial_lineage),
        terminal_spine_count=len(analysis.terminal_lineage),
    )


def restore_terminal_spine_count_before_final_barline(text: str) -> str:
    """Restore the terminal spine width to the initial width before snippet end."""
    analysis = _analyze_lineage(text)
    if analysis is None or analysis.initial_lineage is None or analysis.terminal_lineage is None:
        return text

    initial_lineage = analysis.initial_lineage
    terminal_lineage = analysis.terminal_lineage
    if len(initial_lineage) >= len(terminal_lineage):
        return text

    lines = _split_text_preserving_trailing_newline(text)
    repair_anchor = _find_repair_anchor(lines)
    if repair_anchor is None:
        return text

    barline_idx, trailing_terminator_indices = repair_anchor
    barline_fields = lines[barline_idx].split("\t")
    if len(barline_fields) != len(terminal_lineage):
        return text

    merge_rows: list[str] = []
    working_lineage = terminal_lineage
    initial_signature = _lineage_signature(initial_lineage)
    while _lineage_signature(working_lineage) != initial_signature:
        merge_runs = _select_merge_runs(working_lineage)
        if not merge_runs:
            return text
        merge_rows.append(_build_merge_row(len(working_lineage), merge_runs))
        working_lineage = _apply_selected_merge_runs(working_lineage, merge_runs)

    lines[barline_idx] = "\t".join(_collapse_fields(barline_fields, terminal_lineage))
    for terminator_idx in trailing_terminator_indices:
        terminator_fields = lines[terminator_idx].split("\t")
        if len(terminator_fields) != len(terminal_lineage):
            return text
        lines[terminator_idx] = "\t".join(_collapse_fields(terminator_fields, terminal_lineage))

    lines[barline_idx:barline_idx] = merge_rows
    return "\n".join(lines)
