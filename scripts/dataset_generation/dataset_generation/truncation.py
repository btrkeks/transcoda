"""Truncation policy and prefix candidate generation."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.types_domain import TruncationMode
from scripts.dataset_generation.dataset_generation.types_render import (
    RenderResult,
    SvgLayoutDiagnostics,
)
from src.core.kern_postprocess import resolve_terminal_active_spine_count
from src.core.kern_utils import (
    is_bar_line,
    is_spinemerge_line,
    is_spinesplit_line,
    is_terminator_line,
    split_into_same_spine_nr_chunks_and_measures,
)

_DURATION_BEARING_NOTE_OR_REST_RE = re.compile(r"\d+(?:%\d+)?\.*[A-GRra-gr]")


@dataclass(frozen=True)
class PrefixTruncationCandidate:
    """A prefix-based truncation candidate aligned to measure/chunk boundaries."""

    transcription: str
    chunk_count: int
    total_chunks: int
    ratio: float
    origin_line_indices: tuple[int, ...]


@dataclass(frozen=True)
class PrefixTruncationSpace:
    """Cached chunk-aligned truncation search space for one transcription."""

    chunks: tuple[str, ...]
    chunk_origin_line_indices: tuple[tuple[int, ...], ...]

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def candidate_for_chunk_count(
        self,
        chunk_count: int,
    ) -> PrefixTruncationCandidate | None:
        total_chunks = self.total_chunks
        if total_chunks == 0:
            return None
        if not 1 <= chunk_count <= total_chunks:
            raise ValueError(
                f"chunk_count must be in [1, {total_chunks}], got {chunk_count}"
            )

        transcription, origin_line_indices = _strip_trailing_spine_transition_lines(
            "".join(self.chunks[:chunk_count]).rstrip("\n").splitlines(),
            [
                line_idx
                for chunk_origin_line_indices in self.chunk_origin_line_indices[:chunk_count]
                for line_idx in chunk_origin_line_indices
            ],
        )
        if not transcription.strip():
            return None
        ratio = float(chunk_count) / float(total_chunks)
        return PrefixTruncationCandidate(
            transcription=transcription,
            chunk_count=chunk_count,
            total_chunks=total_chunks,
            ratio=ratio,
            origin_line_indices=origin_line_indices,
        )


@dataclass(frozen=True)
class TruncationProbeResult:
    """Result of probing one truncation candidate during search."""

    candidate: PrefixTruncationCandidate
    accepted: bool
    rejection_reason: str | None
    decision_reason: str | None
    render_result: RenderResult | None = None


@dataclass(frozen=True)
class TruncationSearchResult:
    """Outcome of bounded truncation search over canonical prefix candidates."""

    selected_candidate: PrefixTruncationCandidate | None
    selected_probe: TruncationProbeResult | None
    probes: tuple[TruncationProbeResult, ...]
    exhausted_budget: bool


def build_prefix_truncation_space(kern_text: str) -> PrefixTruncationSpace:
    chunks = tuple(split_into_same_spine_nr_chunks_and_measures(kern_text))
    chunk_origin_line_indices: list[tuple[int, ...]] = []
    current_chunk_origin_line_indices: list[int] = []
    for line_idx, line in enumerate(kern_text.splitlines()):
        current_chunk_origin_line_indices.append(line_idx)
        if is_bar_line(line) or is_spinesplit_line(line) or is_spinemerge_line(line):
            chunk_origin_line_indices.append(tuple(current_chunk_origin_line_indices))
            current_chunk_origin_line_indices = []
    if current_chunk_origin_line_indices:
        chunk_origin_line_indices.append(tuple(current_chunk_origin_line_indices))

    return PrefixTruncationSpace(
        chunks=chunks,
        chunk_origin_line_indices=tuple(chunk_origin_line_indices),
    )


def _strip_trailing_spine_transition_lines(
    lines: list[str],
    origin_line_indices: list[int],
) -> tuple[str, tuple[int, ...]]:
    trimmed_lines = list(lines)
    trimmed_origin_line_indices = list(origin_line_indices)
    while trimmed_lines and (
        is_spinesplit_line(trimmed_lines[-1]) or is_spinemerge_line(trimmed_lines[-1])
    ):
        trimmed_lines.pop()
        trimmed_origin_line_indices.pop()
    return "\n".join(trimmed_lines), tuple(trimmed_origin_line_indices)


def truncate_by_chunk_count(kern_text: str, chunk_count: int) -> tuple[str, float]:
    space = build_prefix_truncation_space(kern_text)
    candidate = space.candidate_for_chunk_count(chunk_count)
    if candidate is None:
        return "", 0.0
    return candidate.transcription, candidate.ratio


def build_canonical_prefix_candidates(kern_text: str) -> list[PrefixTruncationCandidate]:
    """Return unique prefix candidates ordered from shortest to longest."""
    space = build_prefix_truncation_space(kern_text)
    total_chunks = space.total_chunks
    if total_chunks <= 1:
        return []

    by_transcription: dict[str, PrefixTruncationCandidate] = {}
    for chunk_count in range(1, total_chunks):
        candidate = space.candidate_for_chunk_count(chunk_count)
        if candidate is None:
            continue
        existing = by_transcription.get(candidate.transcription)
        if existing is None or candidate.chunk_count > existing.chunk_count:
            by_transcription[candidate.transcription] = candidate

    return sorted(
        by_transcription.values(),
        key=lambda candidate: (candidate.chunk_count, candidate.ratio, candidate.transcription),
    )


def build_prefix_truncation_candidates(
    kern_text: str,
    *,
    max_trials: int,
) -> list[PrefixTruncationCandidate]:
    if max_trials < 1:
        raise ValueError(f"max_trials must be >= 1, got {max_trials}")

    canonical_candidates = build_canonical_prefix_candidates(kern_text)
    if not canonical_candidates:
        return []
    return list(reversed(canonical_candidates[-max_trials:]))


def classify_truncation_mode(
    diagnostics: SvgLayoutDiagnostics,
    recipe: ProductionRecipe,
) -> TruncationMode:
    if diagnostics.page_count > 1:
        return "required"
    if diagnostics.system_count > recipe.truncation.required_over_systems:
        return "required"
    if (
        recipe.truncation.preferred_min_systems
        <= diagnostics.system_count
        <= recipe.truncation.preferred_max_systems
    ):
        return "preferred"
    return "forbidden"


def build_prefix_candidates(
    label_transcription: str,
    recipe: ProductionRecipe,
) -> list[PrefixTruncationCandidate]:
    return build_prefix_truncation_candidates(
        label_transcription,
        max_trials=recipe.truncation.max_candidate_trials,
    )


def validate_truncation_candidate_terminal_state(text: str) -> str | None:
    """Return a rejection reason when candidate terminal spine state is invalid."""
    if not text.strip():
        return "invalid_terminal_spine_state"

    active_spines = resolve_terminal_active_spine_count(text)
    if active_spines is None or active_spines <= 0:
        return "invalid_terminal_spine_state"

    lines = text.rstrip("\n").splitlines()
    if not lines:
        return "invalid_terminal_spine_state"
    if is_terminator_line(lines[-1]):
        terminator_width = lines[-1].count("\t") + 1
        if terminator_width != active_spines:
            return "invalid_terminal_spine_state"
    if is_spinesplit_line(lines[-1]) or is_spinemerge_line(lines[-1]):
        return "invalid_terminal_spine_state"
    return None


def validate_truncation_candidate_contains_music(text: str) -> str | None:
    """Return a rejection reason when a candidate has no duration-bearing data."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.startswith("!")
            or line.startswith("*")
            or is_bar_line(line)
            or is_spinesplit_line(line)
            or is_spinemerge_line(line)
        ):
            continue
        for field in raw_line.split("\t"):
            for token in field.split():
                if token == "." or token.startswith("*") or token.startswith("!"):
                    continue
                if _DURATION_BEARING_NOTE_OR_REST_RE.search(token):
                    return None
    return "non_musical_truncation_candidate"


def find_best_truncation_candidate(
    kern_text: str,
    *,
    max_trials: int,
    probe_candidate: Callable[[PrefixTruncationCandidate], TruncationProbeResult],
    local_refinement_radius: int = 2,
) -> TruncationSearchResult:
    """Probe canonical candidates with bounded binary search and local refinement."""
    if max_trials < 1:
        raise ValueError(f"max_trials must be >= 1, got {max_trials}")
    if local_refinement_radius < 0:
        raise ValueError(
            f"local_refinement_radius must be >= 0, got {local_refinement_radius}"
        )

    candidates = build_canonical_prefix_candidates(kern_text)
    if not candidates:
        return TruncationSearchResult(
            selected_candidate=None,
            selected_probe=None,
            probes=(),
            exhausted_budget=False,
        )

    ordered_probes: list[TruncationProbeResult] = []
    probe_cache: dict[int, TruncationProbeResult] = {}
    exhausted_budget = False

    def probe_index(index: int) -> TruncationProbeResult | None:
        nonlocal exhausted_budget
        if index in probe_cache:
            return probe_cache[index]
        if len(ordered_probes) >= max_trials:
            exhausted_budget = True
            return None
        probe = probe_candidate(candidates[index])
        probe_cache[index] = probe
        ordered_probes.append(probe)
        return probe

    low = 0
    high = len(candidates) - 1
    best_accepted_idx: int | None = None
    first_rejected_mid: int | None = None

    while low <= high:
        mid = (low + high) // 2
        probe = probe_index(mid)
        if probe is None:
            break
        if probe.accepted:
            best_accepted_idx = mid
            low = mid + 1
        else:
            if first_rejected_mid is None:
                first_rejected_mid = mid
            high = mid - 1

    refinement_indices: list[int] = []
    if best_accepted_idx is not None:
        refinement_indices.extend(
            range(
                max(0, best_accepted_idx - local_refinement_radius),
                min(len(candidates) - 1, best_accepted_idx + local_refinement_radius) + 1,
            )
        )
    elif first_rejected_mid is not None:
        refinement_indices.extend(
            range(
                max(0, first_rejected_mid - local_refinement_radius),
                min(len(candidates) - 1, first_rejected_mid + local_refinement_radius) + 1,
            )
        )
    if first_rejected_mid is not None and first_rejected_mid - 1 >= 0:
        refinement_indices.append(first_rejected_mid - 1)

    seen_indices: set[int] = set()
    for index in refinement_indices:
        if index in seen_indices:
            continue
        seen_indices.add(index)
        if probe_index(index) is None:
            break

    accepted_indices = [index for index, probe in probe_cache.items() if probe.accepted]
    if accepted_indices:
        selected_idx = max(accepted_indices)
        selected_candidate = candidates[selected_idx]
        selected_probe = probe_cache[selected_idx]
    else:
        selected_candidate = None
        selected_probe = None

    return TruncationSearchResult(
        selected_candidate=selected_candidate,
        selected_probe=selected_probe,
        probes=tuple(ordered_probes),
        exhausted_budget=exhausted_budget,
    )
