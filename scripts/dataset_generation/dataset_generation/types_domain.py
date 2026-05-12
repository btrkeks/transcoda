"""Core domain types — source entries, sample plans, and enumerations.

Stable building blocks that render/event/outcome types build on. No
intra-package dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

AcceptanceAction = Literal[
    "accept_without_truncation",
    "accept_with_truncation",
    "reject",
]
TruncationMode = Literal["forbidden", "preferred", "required"]
AugmentationBand = Literal["roomy", "balanced", "tight"]
PreferredFiveSixStatus = Literal[
    "preferred_5_6_accepted_full",
    "preferred_5_6_rescued",
    "preferred_5_6_truncated",
    "preferred_5_6_failed",
]


class AttemptStageName(StrEnum):
    FULL = "full"
    FULL_LAYOUT_RESCUE = "full_layout_rescue"
    TRUNCATION_CANDIDATE = "truncation_candidate"
    TRUNCATION_CANDIDATE_LAYOUT_RESCUE = "truncation_candidate_layout_rescue"


@dataclass(frozen=True)
class SourceEntry:
    entry_idx: int
    path: Path
    source_id: str
    root_dir: Path
    root_label: str
    measure_count: int
    non_empty_line_count: int
    has_header: bool
    initial_spine_count: int
    terminal_spine_count: int
    restored_terminal_spine_count: int


@dataclass(frozen=True)
class SourceSegment:
    source_id: str
    path: Path
    order: int


@dataclass(frozen=True)
class SamplePlan:
    sample_id: str
    seed: int
    segments: tuple[SourceSegment, ...]
    label_transcription: str
    source_measure_count: int
    source_non_empty_line_count: int
    source_max_initial_spine_count: int
    segment_count: int
