"""Helpers for computing cheap source-file statistics used in scheduling and filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.kern_utils import is_bar_line


@dataclass(frozen=True)
class KernSourceStats:
    """Cheap structural stats derived from a normalized **kern source file."""

    non_empty_line_count: int
    measure_count: int


def compute_kern_source_stats(path: Path) -> KernSourceStats:
    """Count non-empty lines and barline-based measures for one normalized **kern file."""
    non_empty_line_count = 0
    measure_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            non_empty_line_count += 1
            if is_bar_line(line):
                measure_count += 1
    return KernSourceStats(
        non_empty_line_count=non_empty_line_count,
        measure_count=measure_count,
    )
