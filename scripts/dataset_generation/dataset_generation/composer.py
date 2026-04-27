"""Sample planning and whole-file composition."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.source_index import SourceIndex
from scripts.dataset_generation.dataset_generation.types_domain import (
    SamplePlan,
    SourceEntry,
    SourceSegment,
)
from src.core.kern_concatenation import restore_terminal_spine_count_before_final_barline
from src.core.kern_utils import is_spinemerge_line, is_spinesplit_line, is_terminator_line

_CLEF_RE = re.compile(r"^\*clef[A-Ga-gvX]v?\d?$")
_KEYSIG_RE = re.compile(r"^\*k\[.*\]$")
_METER_RE = re.compile(r"^\*M\d+/\d+$")


@dataclass(frozen=True)
class BoundaryState:
    clef: tuple[str | None, ...]
    key: tuple[str | None, ...]
    meter: tuple[str | None, ...]

    @property
    def spine_count(self) -> int:
        return len(self.clef)


@dataclass(frozen=True)
class StructuredSamplePlan:
    sample_id: str
    seed: int
    entry_ids: tuple[int, ...]
    segments: tuple[SourceSegment, ...]
    source_measure_count: int
    source_non_empty_line_count: int
    source_max_initial_spine_count: int
    segment_count: int


def compose_label_transcription(entries: Sequence[SourceEntry]) -> str:
    if not entries:
        raise ValueError("compose_label_transcription requires at least one source entry")

    prepared_segments: list[str] = []
    current_state: BoundaryState | None = None
    for idx, entry in enumerate(entries):
        text = entry.path.read_text(encoding="utf-8")
        if idx < len(entries) - 1:
            text = restore_terminal_spine_count_before_final_barline(text)
        raw_lines = _split_text_lines(text)
        segment_lines, scan_initial_state = _prepare_segment_lines(
            raw_lines,
            current_state=current_state,
            strip_header=idx > 0,
            strip_terminator=idx < len(entries) - 1,
        )
        current_state = _scan_boundary_state(segment_lines, initial_state=scan_initial_state)
        segment = "\n".join(segment_lines).strip("\n")
        if segment:
            prepared_segments.append(segment)

    composed = "\n".join(part.rstrip("\n") for part in prepared_segments).rstrip("\n")
    if not composed:
        raise ValueError("Composed transcription is empty")
    composed = restore_terminal_spine_count_before_final_barline(composed)
    return composed + "\n"


def plan_sample(
    source_index: SourceIndex,
    recipe: ProductionRecipe,
    *,
    sample_idx: int,
    base_seed: int = 0,
    excluded_entry_ids: set[int] | frozenset[int] | None = None,
) -> SamplePlan:
    structured = plan_sample_structure(
        source_index,
        recipe,
        sample_idx=sample_idx,
        base_seed=base_seed,
        excluded_entry_ids=excluded_entry_ids,
    )
    return materialize_sample_plan(source_index, structured)


def plan_sample_structure(
    source_index: SourceIndex,
    recipe: ProductionRecipe,
    *,
    sample_idx: int,
    base_seed: int = 0,
    excluded_entry_ids: set[int] | frozenset[int] | None = None,
) -> StructuredSamplePlan:
    seed = derive_sample_seed(base_seed=base_seed, sample_idx=sample_idx)
    rng = random.Random(seed)
    entry_ids = tuple(
        _choose_entries(
            source_index,
            recipe,
            rng,
            excluded_entry_ids=excluded_entry_ids,
        )
    )
    segments = tuple(
        SourceSegment(
            source_id=source_index.entries[entry_idx].source_id,
            path=source_index.entries[entry_idx].path,
            order=order,
        )
        for order, entry_idx in enumerate(entry_ids)
    )
    entries = tuple(source_index.entries[entry_idx] for entry_idx in entry_ids)
    return StructuredSamplePlan(
        sample_id=f"sample_{sample_idx:08d}",
        seed=seed,
        entry_ids=entry_ids,
        segments=segments,
        source_measure_count=sum(entry.measure_count for entry in entries),
        source_non_empty_line_count=sum(entry.non_empty_line_count for entry in entries),
        source_max_initial_spine_count=max(entry.initial_spine_count for entry in entries),
        segment_count=len(entries),
    )


def materialize_sample_plan(
    source_index: SourceIndex,
    structured: StructuredSamplePlan,
) -> SamplePlan:
    entries = tuple(source_index.entries[entry_idx] for entry_idx in structured.entry_ids)
    label_transcription = compose_label_transcription(entries)
    return SamplePlan(
        sample_id=structured.sample_id,
        seed=structured.seed,
        segments=structured.segments,
        label_transcription=label_transcription,
        source_measure_count=structured.source_measure_count,
        source_non_empty_line_count=structured.source_non_empty_line_count,
        source_max_initial_spine_count=structured.source_max_initial_spine_count,
        segment_count=structured.segment_count,
    )


def derive_sample_seed(*, base_seed: int, sample_idx: int) -> int:
    return ((int(base_seed) & 0xFFFFFFFF) * 1_000_003 + int(sample_idx)) & 0xFFFFFFFF


def _choose_entries(
    source_index: SourceIndex,
    recipe: ProductionRecipe,
    rng: random.Random,
    *,
    excluded_entry_ids: set[int] | frozenset[int] | None = None,
) -> Sequence[int]:
    excluded_ids = (
        frozenset(int(entry_idx) for entry_idx in excluded_entry_ids)
        if excluded_entry_ids is not None
        else frozenset()
    )
    if not source_index.entries:
        raise ValueError("Cannot compose from an empty source index")

    target_segments = _weighted_choice(recipe.composition.segment_count_weights, rng)
    target_segments = max(1, min(target_segments, len(source_index.entries)))
    target_total_measures = rng.randint(
        recipe.composition.min_total_measures,
        recipe.composition.max_total_measures,
    )

    anchor_entry_idx = _pick_anchor_entry_idx(source_index, excluded_ids, rng)
    anchor = source_index.entries[anchor_entry_idx]
    chosen_entry_ids: list[int] = [anchor_entry_idx]
    unavailable_entry_ids = set(excluded_ids)
    unavailable_entry_ids.add(anchor_entry_idx)
    current_total = anchor.measure_count
    current_boundary_spine_count = anchor.restored_terminal_spine_count

    while len(chosen_entry_ids) < target_segments:
        candidate_entry_idx = _pick_next_entry(
            source_index=source_index,
            unavailable_entry_ids=unavailable_entry_ids,
            current_total=current_total,
            current_boundary_spine_count=current_boundary_spine_count,
            target_total_measures=target_total_measures,
            max_total_measures=recipe.composition.max_total_measures,
            rng=rng,
        )
        if candidate_entry_idx is None:
            break
        candidate = source_index.entries[candidate_entry_idx]
        chosen_entry_ids.append(candidate_entry_idx)
        unavailable_entry_ids.add(candidate_entry_idx)
        current_total += candidate.measure_count
        current_boundary_spine_count = candidate.restored_terminal_spine_count

    if current_total < recipe.composition.min_total_measures and len(chosen_entry_ids) < target_segments:
        for _ in range(recipe.composition.max_selection_attempts):
            candidate_entry_idx = _pick_next_entry(
                source_index=source_index,
                unavailable_entry_ids=unavailable_entry_ids,
                current_total=current_total,
                current_boundary_spine_count=current_boundary_spine_count,
                target_total_measures=recipe.composition.min_total_measures,
                max_total_measures=recipe.composition.max_total_measures,
                rng=rng,
            )
            if candidate_entry_idx is None:
                break
            candidate = source_index.entries[candidate_entry_idx]
            chosen_entry_ids.append(candidate_entry_idx)
            unavailable_entry_ids.add(candidate_entry_idx)
            current_total += candidate.measure_count
            current_boundary_spine_count = candidate.restored_terminal_spine_count
            if (
                current_total >= recipe.composition.min_total_measures
                or len(chosen_entry_ids) >= target_segments
            ):
                break

    return tuple(chosen_entry_ids)


def _pick_anchor_entry_idx(
    source_index: SourceIndex,
    excluded_entry_ids: frozenset[int],
    rng: random.Random,
) -> int:
    available_entry_ids = tuple(
        entry.entry_idx
        for entry in source_index.entries
        if entry.entry_idx not in excluded_entry_ids
    )
    if available_entry_ids:
        return available_entry_ids[rng.randrange(len(available_entry_ids))]
    raise ValueError("Cannot compose from an empty source index")


def _weighted_choice(
    weighted_values: Sequence[tuple[int, float]],
    rng: random.Random,
) -> int:
    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight <= 0.0:
        raise ValueError("segment_count_weights must contain positive total weight")
    roll = rng.random() * total_weight
    cumulative = 0.0
    for value, weight in weighted_values:
        cumulative += weight
        if roll <= cumulative:
            return value
    return weighted_values[-1][0]


def _pick_next_entry(
    *,
    source_index: SourceIndex,
    unavailable_entry_ids: set[int],
    current_total: int,
    current_boundary_spine_count: int,
    target_total_measures: int,
    max_total_measures: int,
    rng: random.Random,
) -> int | None:
    compatible_by_measure = _compatible_entry_ids_by_measure_count(
        source_index=source_index,
        initial_spine_count=current_boundary_spine_count,
    )
    if not compatible_by_measure:
        return None

    scored: list[tuple[float, int]] = []
    target_remaining = target_total_measures - current_total
    measure_counts_by_distance: dict[int, list[int]] = defaultdict(list)
    for measure_count in compatible_by_measure:
        projected_total = current_total + measure_count
        if projected_total > max_total_measures and current_total >= target_total_measures:
            continue
        distance = abs(target_remaining - measure_count)
        measure_counts_by_distance[distance].append(measure_count)

    for distance in sorted(measure_counts_by_distance):
        for measure_count in measure_counts_by_distance[distance]:
            for entry_idx in _sample_available_entry_ids(
                compatible_by_measure[measure_count],
                unavailable_entry_ids=unavailable_entry_ids,
                rng=rng,
                limit=max(4, 8 - len(scored)),
            ):
                jitter = rng.random() * 0.25
                scored.append((distance + jitter, entry_idx))
        if len(scored) >= 4:
            break

    if not scored:
        return None

    scored.sort(key=lambda item: item[0])
    shortlist = [entry_idx for _, entry_idx in scored[: min(4, len(scored))]]
    return shortlist[rng.randrange(len(shortlist))]


def _compatible_entry_ids_by_measure_count(
    *,
    source_index: SourceIndex,
    initial_spine_count: int,
) -> dict[int, tuple[int, ...]]:
    cache_key = int(initial_spine_count)
    cached = source_index.compatible_entry_ids_by_measure_count_cache.get(cache_key)
    if cached is not None:
        return cached

    grouped: dict[int, list[int]] = defaultdict(list)
    for entry_idx in source_index.entry_indices_by_initial_spine_count.get(
        initial_spine_count,
        (),
    ):
        entry = source_index.entries[entry_idx]
        grouped[int(entry.measure_count)].append(entry_idx)
    indexed = {
        measure_count: tuple(entry_ids)
        for measure_count, entry_ids in sorted(grouped.items())
    }
    source_index.compatible_entry_ids_by_measure_count_cache[cache_key] = indexed
    return indexed


def _sample_available_entry_ids(
    entry_ids: tuple[int, ...],
    *,
    unavailable_entry_ids: set[int],
    rng: random.Random,
    limit: int,
) -> list[int]:
    if not entry_ids or limit <= 0:
        return []

    selected: list[int] = []
    start = rng.randrange(len(entry_ids))
    for offset in range(len(entry_ids)):
        entry_idx = entry_ids[(start + offset) % len(entry_ids)]
        if entry_idx in unavailable_entry_ids:
            continue
        selected.append(entry_idx)
        if len(selected) >= limit:
            break
    return selected


def _normalize_segment_text(
    text: str,
    *,
    strip_header: bool,
    strip_terminator: bool,
) -> str:
    lines = _split_text_lines(text)
    if strip_header:
        lines = _strip_leading_header(lines)
    if strip_terminator:
        lines = _strip_trailing_terminator(lines)
    return "\n".join(line for line in lines if line is not None).strip("\n")


def _prepare_segment_lines(
    raw_lines: Sequence[str],
    *,
    current_state: BoundaryState | None,
    strip_header: bool,
    strip_terminator: bool,
) -> tuple[list[str], BoundaryState | None]:
    lines = list(raw_lines)
    scan_initial_state = current_state
    if strip_header:
        if current_state is None:
            raise ValueError("Cannot strip follow-up headers without an active boundary state")
        next_spine_count = _infer_spine_count(lines)
        if next_spine_count is None:
            return [], current_state
        header_lines, body_lines = _split_leading_header(lines)
        next_initial_state = _scan_boundary_state(
            header_lines,
            initial_state=_empty_boundary_state(next_spine_count),
        )
        assert next_initial_state is not None
        if next_spine_count != current_state.spine_count:
            raise ValueError(
                "Boundary spine count mismatch after terminal restoration: "
                f"{current_state.spine_count} -> {next_spine_count}"
            )
        transition_lines = _build_transition_lines(current_state, next_initial_state)
        lines = transition_lines + body_lines
        scan_initial_state = current_state
    if strip_terminator:
        lines = _strip_trailing_terminator(lines)
    return lines, scan_initial_state


def _split_text_lines(text: str) -> list[str]:
    return [line.rstrip("\n") for line in text.replace("\r\n", "\n").split("\n")]


def _split_leading_header(lines: Sequence[str]) -> tuple[list[str], list[str]]:
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if line.startswith("!") or line.startswith("*"):
            idx += 1
            continue
        break
    return list(lines[:idx]), list(lines[idx:])


def _strip_leading_header(lines: Sequence[str]) -> list[str]:
    _, body_lines = _split_leading_header(lines)
    return body_lines


def _strip_trailing_terminator(lines: Sequence[str]) -> list[str]:
    stripped = list(lines)
    while stripped and not stripped[-1].strip():
        stripped.pop()
    if stripped and is_terminator_line(stripped[-1]):
        stripped.pop()
    while stripped and not stripped[-1].strip():
        stripped.pop()
    return stripped


def _infer_spine_count(lines: Sequence[str]) -> int | None:
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        return line.count("\t") + 1
    return None


def _empty_boundary_state(spine_count: int) -> BoundaryState:
    empty = tuple(None for _ in range(spine_count))
    return BoundaryState(clef=empty, key=empty, meter=empty)


def _scan_boundary_state(
    lines: Sequence[str],
    *,
    initial_state: BoundaryState | None = None,
) -> BoundaryState | None:
    state = initial_state
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        if state is None:
            state = _empty_boundary_state(line.count("\t") + 1)
        elif line.count("\t") + 1 != state.spine_count:
            raise ValueError(
                "Encountered unexpected spine-count change while scanning composed segment"
            )
        state = _apply_tracked_updates(state, line)
        if is_spinesplit_line(line):
            state = _apply_spine_split(state, line.split("\t"))
        elif is_spinemerge_line(line):
            state = _apply_spine_merge(state, line.split("\t"))
    return state


def _apply_tracked_updates(state: BoundaryState, line: str) -> BoundaryState:
    tokens = line.split("\t")
    if len(tokens) != state.spine_count:
        raise ValueError("Tracked update line does not match active spine count")
    return BoundaryState(
        clef=_update_category(state.clef, tokens, _is_clef_token),
        key=_update_category(state.key, tokens, _is_keysig_token),
        meter=_update_category(state.meter, tokens, _is_meter_token),
    )


def _update_category(
    current: tuple[str | None, ...],
    tokens: Sequence[str],
    matcher: Callable[[str], bool],
) -> tuple[str | None, ...]:
    updated = list(current)
    for idx, token in enumerate(tokens):
        if matcher(token):
            updated[idx] = token
    return tuple(updated)


def _apply_spine_split(state: BoundaryState, tokens: Sequence[str]) -> BoundaryState:
    return BoundaryState(
        clef=_expand_for_split(state.clef, tokens),
        key=_expand_for_split(state.key, tokens),
        meter=_expand_for_split(state.meter, tokens),
    )


def _expand_for_split(values: tuple[str | None, ...], tokens: Sequence[str]) -> tuple[str | None, ...]:
    expanded: list[str | None] = []
    for value, token in zip(values, tokens, strict=True):
        expanded.append(value)
        if token == "*^":
            expanded.append(value)
    return tuple(expanded)


def _apply_spine_merge(state: BoundaryState, tokens: Sequence[str]) -> BoundaryState:
    return BoundaryState(
        clef=_collapse_for_merge(state.clef, tokens),
        key=_collapse_for_merge(state.key, tokens),
        meter=_collapse_for_merge(state.meter, tokens),
    )


def _collapse_for_merge(values: tuple[str | None, ...], tokens: Sequence[str]) -> tuple[str | None, ...]:
    collapsed: list[str | None] = []
    idx = 0
    while idx < len(tokens):
        if tokens[idx] != "*v":
            collapsed.append(values[idx])
            idx += 1
            continue
        end = idx
        merged_values: list[str | None] = []
        while end < len(tokens) and tokens[end] == "*v":
            merged_values.append(values[end])
            end += 1
        merged_value = merged_values[0] if merged_values else None
        if any(value != merged_value for value in merged_values[1:]):
            merged_value = None
        collapsed.append(merged_value)
        idx = end
    return tuple(collapsed)


def _build_transition_lines(current_state: BoundaryState, next_state: BoundaryState) -> list[str]:
    if current_state.spine_count != next_state.spine_count:
        raise ValueError(
            "Cannot build boundary transitions for snippets with different spine counts"
        )
    transition_lines: list[str] = []
    for current_values, next_values in (
        (current_state.clef, next_state.clef),
        (current_state.key, next_state.key),
        (current_state.meter, next_state.meter),
    ):
        line_tokens = [
            next_token if next_token is not None and next_token != current_token else "*"
            for current_token, next_token in zip(current_values, next_values, strict=True)
        ]
        if any(token != "*" for token in line_tokens):
            transition_lines.append("\t".join(line_tokens))
    return transition_lines


def _is_clef_token(token: str) -> bool:
    return bool(_CLEF_RE.match(token))


def _is_keysig_token(token: str) -> bool:
    return bool(_KEYSIG_RE.match(token))


def _is_meter_token(token: str) -> bool:
    return bool(_METER_RE.match(token))
