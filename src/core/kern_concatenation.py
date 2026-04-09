"""Helpers for stitching multiple **kern snippets into one transcription."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.spine_state import advance_spine_count, is_interpretation_record


@dataclass(frozen=True)
class SpineTopologySummary:
    """Summarize the boundary spine widths of a **kern transcription."""

    initial_spine_count: int | None
    terminal_spine_count: int | None


def summarize_spine_topology(text: str) -> SpineTopologySummary:
    """Return the initial and terminal active spine counts for ``text``."""
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("!!")]
    if not lines:
        return SpineTopologySummary(initial_spine_count=None, terminal_spine_count=None)

    current_spines: int | None = None
    initial_spines: int | None = None
    for line in lines:
        fields = line.split("\t")
        if current_spines is None:
            current_spines = len(fields)
            initial_spines = current_spines
        if is_interpretation_record(fields) and "*-" not in fields:
            current_spines = advance_spine_count(current_spines, fields)
        elif fields and all(field == "*-" for field in fields):
            continue
        else:
            current_spines = len(fields)

    return SpineTopologySummary(
        initial_spine_count=initial_spines,
        terminal_spine_count=current_spines,
    )


def restore_terminal_spine_count_before_final_barline(text: str) -> str:
    """Restore the terminal spine width to the initial width before snippet end.

    This is the seam the dataset-generation composer should call after concatenating
    raw snippet bodies. The eventual implementation will synthesize terminal ``*v``
    merge records immediately before the final barline when a snippet ends in a
    wider split topology than it started with.

    For now the helper is intentionally a no-op so we can land the call site and
    tests without changing existing dataset behavior.
    """
    topology = summarize_spine_topology(text)
    if topology.initial_spine_count is None or topology.terminal_spine_count is None:
        return text
    if topology.initial_spine_count == topology.terminal_spine_count:
        return text
    # TODO: insert terminal merge synthesis before the final barline.
    return text
