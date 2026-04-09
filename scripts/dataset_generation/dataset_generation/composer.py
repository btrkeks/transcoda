"""Sample planning and whole-file composition."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from src.core.kern_utils import is_spinemerge_line, is_spinesplit_line, is_terminator_line

from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.source_index import SourceIndex
from scripts.dataset_generation.dataset_generation.types import (
    SamplePlan,
    SourceEntry,
    SourceSegment,
)

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


def compose_label_transcription(entries: Sequence[SourceEntry]) -> str:
    if not entries:
        raise ValueError("compose_label_transcription requires at least one source entry")

    prepared_segments: list[str] = []
    current_state: BoundaryState | None = None
    for idx, entry in enumerate(entries):
        text = Path(entry.path).read_text(encoding="utf-8")
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
    return composed + "\n"


def plan_sample(
    source_index: SourceIndex,
    recipe: ProductionRecipe,
    *,
    sample_idx: int,
    base_seed: int = 0,
    excluded_paths: set[Path] | None = None,
) -> SamplePlan:
    seed = derive_sample_seed(base_seed=base_seed, sample_idx=sample_idx)
    rng = random.Random(seed)
    entries = tuple(_choose_entries(source_index.entries, recipe, rng, excluded_paths=excluded_paths))
    label_transcription = compose_label_transcription(entries)
    segments = tuple(
        SourceSegment(source_id=entry.source_id, path=entry.path, order=idx)
        for idx, entry in enumerate(entries)
    )
    return SamplePlan(
        sample_id=f"sample_{sample_idx:08d}",
        seed=seed,
        segments=segments,
        label_transcription=label_transcription,
        source_measure_count=sum(entry.measure_count for entry in entries),
        source_non_empty_line_count=sum(entry.non_empty_line_count for entry in entries),
        source_max_initial_spine_count=max(entry.initial_spine_count for entry in entries),
        segment_count=len(entries),
    )


def derive_sample_seed(*, base_seed: int, sample_idx: int) -> int:
    return ((int(base_seed) & 0xFFFFFFFF) * 1_000_003 + int(sample_idx)) & 0xFFFFFFFF


def _choose_entries(
    available_entries: Sequence[SourceEntry],
    recipe: ProductionRecipe,
    rng: random.Random,
    *,
    excluded_paths: set[Path] | None = None,
) -> Sequence[SourceEntry]:
    filtered_entries = tuple(
        entry for entry in available_entries if excluded_paths is None or entry.path not in excluded_paths
    )
    if not filtered_entries:
        raise ValueError("Cannot compose from an empty source index")

    target_segments = _weighted_choice(recipe.composition.segment_count_weights, rng)
    target_segments = max(1, min(target_segments, len(filtered_entries)))
    target_total_measures = rng.randint(
        recipe.composition.min_total_measures,
        recipe.composition.max_total_measures,
    )

    anchor = filtered_entries[rng.randrange(len(filtered_entries))]
    chosen: list[SourceEntry] = [anchor]
    remaining = [entry for entry in filtered_entries if entry.path != anchor.path]
    current_total = anchor.measure_count

    while len(chosen) < target_segments and remaining:
        candidate = _pick_next_entry(
            remaining=remaining,
            current_total=current_total,
            target_total_measures=target_total_measures,
            max_total_measures=recipe.composition.max_total_measures,
            rng=rng,
        )
        if candidate is None:
            break
        chosen.append(candidate)
        remaining = [entry for entry in remaining if entry.path != candidate.path]
        current_total += candidate.measure_count

    if current_total < recipe.composition.min_total_measures and len(chosen) < target_segments:
        for _ in range(recipe.composition.max_selection_attempts):
            candidate = _pick_next_entry(
                remaining=remaining,
                current_total=current_total,
                target_total_measures=recipe.composition.min_total_measures,
                max_total_measures=recipe.composition.max_total_measures,
                rng=rng,
            )
            if candidate is None:
                break
            chosen.append(candidate)
            remaining = [entry for entry in remaining if entry.path != candidate.path]
            current_total += candidate.measure_count
            if (
                current_total >= recipe.composition.min_total_measures
                or len(chosen) >= target_segments
            ):
                break

    return tuple(chosen)


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
    remaining: Sequence[SourceEntry],
    current_total: int,
    target_total_measures: int,
    max_total_measures: int,
    rng: random.Random,
) -> SourceEntry | None:
    if not remaining:
        return None

    scored: list[tuple[float, SourceEntry]] = []
    for entry in remaining:
        projected_total = current_total + entry.measure_count
        if projected_total > max_total_measures and current_total >= target_total_measures:
            continue
        distance = abs(target_total_measures - projected_total)
        jitter = rng.random() * 0.25
        scored.append((distance + jitter, entry))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0])
    shortlist = [entry for _, entry in scored[: min(4, len(scored))]]
    return shortlist[rng.randrange(len(shortlist))]


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
        if next_spine_count == current_state.spine_count:
            transition_lines = _build_transition_lines(current_state, next_initial_state)
            lines = transition_lines + body_lines
            scan_initial_state = current_state
        else:
            # Temporary permissive mode: for mismatched widths, drop follow-up headers and
            # concatenate the raw body while resetting tracked state to the new snippet.
            lines = body_lines
            scan_initial_state = next_initial_state
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
