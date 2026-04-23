"""Build deterministic render-only transcriptions from planned label transcriptions."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from scripts.dataset_generation.augmentation.articulations import ACCENT, SFORZANDO
from scripts.dataset_generation.augmentation.dynam_spine import (
    default_dynam_token,
    find_trailing_dynam_spine_index,
    is_eligible_data_line,
)
from scripts.dataset_generation.augmentation.hairpins import _MAX_SPAN_STEPS
from scripts.dataset_generation.augmentation.kern_utils import (
    append_to_token,
    find_barline_indices,
    find_note_tokens,
    get_local_spine_count,
    get_spine_count,
)
from scripts.dataset_generation.augmentation.pedaling import PEDAL_DOWN, PEDAL_UP
from scripts.dataset_generation.augmentation.render_dynamic_marks import _DYNAMIC_MARK_TOKENS
from scripts.dataset_generation.augmentation.tempo_markings import BPM_RANGE, OMD_TEMPO_MARKINGS
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from src.core.kern_postprocess import append_terminator_if_missing

_INSTRUMENT_TOKEN_RE = re.compile(r'(^|\t)\*I[^\t\n]*(?=\t|$)', re.MULTILINE)
_OMD_RE = re.compile(r"^!!!OMD:", re.MULTILINE)
_MM_TOKEN_RE = re.compile(r"(^|\t)\*MM[^\t\n]*(?=\t|$)", re.MULTILINE)


@dataclass(frozen=True)
class NoteSuffixPlan:
    origin_line_idx: int
    col_idx: int
    suffix: str


@dataclass(frozen=True)
class PedalSpanPlan:
    start_barline_line_idx: int
    end_barline_line_idx: int


@dataclass(frozen=True)
class HairpinPlan:
    origin_line_indices: tuple[int, ...]
    is_crescendo: bool


@dataclass(frozen=True)
class DynamicMarkPlan:
    origin_line_idx: int
    token: str


@dataclass(frozen=True)
class TempoPlan:
    tempo_text: str
    bpm: int | None


@dataclass(frozen=True)
class RenderAugmentationPlan:
    note_suffixes: tuple[NoteSuffixPlan, ...] = ()
    pedal_spans: tuple[PedalSpanPlan, ...] = ()
    include_instrument_label: bool = False
    tempo: TempoPlan | None = None
    hairpins: tuple[HairpinPlan, ...] = ()
    dynamic_marks: tuple[DynamicMarkPlan, ...] = ()

    @property
    def has_trailing_dynam_spine(self) -> bool:
        return bool(self.hairpins or self.dynamic_marks)


@dataclass
class _AnchoredLine:
    origin_line_idx: int | None
    text: str


def build_render_transcription(
    label_transcription: str,
    recipe: ProductionRecipe,
    *,
    seed: int,
) -> str:
    rng = random.Random(seed)
    augmentation_plan = sample_render_augmentation_plan(
        label_transcription,
        recipe,
        rng=rng,
    )
    source_line_indices = tuple(range(len(label_transcription.splitlines())))
    return materialize_render_transcription(
        label_transcription,
        recipe,
        augmentation_plan=augmentation_plan,
        source_line_indices=source_line_indices,
    )


def sample_render_augmentation_plan(
    label_transcription: str,
    recipe: ProductionRecipe,
    *,
    rng: random.Random,
) -> RenderAugmentationPlan:
    policy = recipe.render_only_aug
    note_positions = find_note_tokens(label_transcription)

    note_suffixes: list[NoteSuffixPlan] = []
    if rng.random() < policy.render_sforzando_probability:
        note_suffixes.extend(
            NoteSuffixPlan(
                origin_line_idx=position.line_idx,
                col_idx=position.col_idx,
                suffix=SFORZANDO,
            )
            for position in note_positions
            if rng.random() < policy.render_sforzando_per_note_probability
        )
    if rng.random() < policy.render_accent_probability:
        note_suffixes.extend(
            NoteSuffixPlan(
                origin_line_idx=position.line_idx,
                col_idx=position.col_idx,
                suffix=ACCENT,
            )
            for position in note_positions
            if rng.random() < policy.render_accent_per_note_probability
        )

    pedal_spans: list[PedalSpanPlan] = []
    barline_indices = find_barline_indices(label_transcription)
    if (
        len(barline_indices) >= 2
        and rng.random() < policy.render_pedals_probability
    ):
        for idx in range(len(barline_indices) - 1):
            if rng.random() >= policy.render_pedals_measures_probability:
                continue
            pedal_spans.append(
                PedalSpanPlan(
                    start_barline_line_idx=barline_indices[idx],
                    end_barline_line_idx=barline_indices[idx + 1],
                )
            )

    include_instrument_label = rng.random() < policy.render_instrument_piano_probability

    tempo: TempoPlan | None = None
    if (
        rng.random() < policy.render_tempo_probability
        and not has_tempo_markings(label_transcription)
    ):
        bpm = None
        if rng.random() < policy.render_tempo_include_mm_probability:
            bpm = rng.randint(*BPM_RANGE)
        tempo = TempoPlan(
            tempo_text=rng.choice(OMD_TEMPO_MARKINGS),
            bpm=bpm,
        )

    lines = label_transcription.splitlines()
    eligible_line_indices = tuple(
        idx for idx, line in enumerate(lines) if is_eligible_data_line(line)
    )

    hairpins: list[HairpinPlan] = []
    occupied_dynam_line_indices: set[int] = set()
    if (
        len(eligible_line_indices) >= 2
        and rng.random() < policy.render_hairpins_probability
    ):
        for start_step, end_step in _sample_non_overlapping_spans(
            num_steps=len(eligible_line_indices),
            max_spans=policy.render_hairpins_max_spans,
            rng=rng,
        ):
            origin_line_indices = eligible_line_indices[start_step : end_step + 1]
            hairpins.append(
                HairpinPlan(
                    origin_line_indices=origin_line_indices,
                    is_crescendo=(rng.random() < 0.5),
                )
            )
            occupied_dynam_line_indices.update(origin_line_indices)

    dynamic_marks: list[DynamicMarkPlan] = []
    available_dynamic_lines = [
        line_idx
        for line_idx in eligible_line_indices
        if line_idx not in occupied_dynam_line_indices
    ]
    if (
        available_dynamic_lines
        and rng.random() < policy.render_dynamic_marks_probability
    ):
        target_mark_count = min(
            rng.randint(
                policy.render_dynamic_marks_min_count,
                policy.render_dynamic_marks_max_count,
            ),
            len(available_dynamic_lines),
        )
        for origin_line_idx in rng.sample(available_dynamic_lines, k=target_mark_count):
            dynamic_marks.append(
                DynamicMarkPlan(
                    origin_line_idx=origin_line_idx,
                    token=rng.choice(_DYNAMIC_MARK_TOKENS),
                )
            )

    return RenderAugmentationPlan(
        note_suffixes=tuple(note_suffixes),
        pedal_spans=tuple(pedal_spans),
        include_instrument_label=include_instrument_label,
        tempo=tempo,
        hairpins=tuple(hairpins),
        dynamic_marks=tuple(dynamic_marks),
    )


def materialize_render_transcription(
    label_transcription: str,
    recipe: ProductionRecipe,
    *,
    augmentation_plan: RenderAugmentationPlan,
    source_line_indices: tuple[int, ...] | None = None,
) -> str:
    del recipe
    source_lines = label_transcription.splitlines()
    if source_line_indices is None:
        source_line_indices = tuple(range(len(source_lines)))
    if len(source_line_indices) != len(source_lines):
        raise ValueError("source_line_indices must align with label_transcription lines")

    lines = [
        _AnchoredLine(origin_line_idx=origin_idx, text=line)
        for origin_idx, line in zip(source_line_indices, source_lines, strict=True)
    ]
    lines = _apply_note_suffixes(lines, augmentation_plan.note_suffixes)
    lines = _apply_pedaling(lines, augmentation_plan.pedal_spans)
    lines = _apply_instrument_label(lines, include=augmentation_plan.include_instrument_label)
    lines = _apply_tempo(lines, augmentation_plan.tempo)
    lines = _apply_trailing_dynam_spine(
        lines,
        hairpins=augmentation_plan.hairpins,
        dynamic_marks=augmentation_plan.dynamic_marks,
    )

    text = "\n".join(line.text for line in lines)
    return append_terminator_if_missing(
        ensure_render_header(
            text,
            last_spine_type="dynam" if augmentation_plan.has_trailing_dynam_spine else None,
        )
    )


def ensure_render_header(
    content: str,
    *,
    last_spine_type: str | None = None,
) -> str:
    first_data_like_line: str | None = None
    for line in content.splitlines():
        if not line.strip():
            continue
        if line.startswith("!!"):
            continue
        first_data_like_line = line
        break

    if first_data_like_line is None or first_data_like_line.startswith("**"):
        return content

    num_spines = first_data_like_line.count("\t") + 1
    if last_spine_type == "dynam":
        if num_spines < 2:
            raise ValueError("Cannot build mixed header with fewer than 2 spines")
        header_tokens = ["**kern"] * (num_spines - 1) + ["**dynam"]
    else:
        header_tokens = ["**kern"] * num_spines
    return "\t".join(header_tokens) + "\n" + content


def has_tempo_markings(krn: str) -> bool:
    return bool(_OMD_RE.search(krn) or _MM_TOKEN_RE.search(krn))


def _apply_note_suffixes(
    lines: list[_AnchoredLine],
    note_suffixes: tuple[NoteSuffixPlan, ...],
) -> list[_AnchoredLine]:
    if not note_suffixes:
        return lines

    suffixes_by_line_col: dict[tuple[int, int], list[str]] = {}
    for plan in note_suffixes:
        suffixes_by_line_col.setdefault((plan.origin_line_idx, plan.col_idx), []).append(plan.suffix)

    result: list[_AnchoredLine] = []
    for line in lines:
        if line.origin_line_idx is None:
            result.append(line)
            continue
        columns = line.text.split("\t")
        updated = False
        for col_idx, token in enumerate(columns):
            suffixes = suffixes_by_line_col.get((line.origin_line_idx, col_idx))
            if not suffixes:
                continue
            for suffix in suffixes:
                token = append_to_token(token, suffix)
            columns[col_idx] = token
            updated = True
        result.append(
            _AnchoredLine(
                origin_line_idx=line.origin_line_idx,
                text="\t".join(columns) if updated else line.text,
            )
        )
    return result


def _apply_pedaling(
    lines: list[_AnchoredLine],
    pedal_spans: tuple[PedalSpanPlan, ...],
) -> list[_AnchoredLine]:
    if not pedal_spans:
        return lines

    indexed_lines = list(lines)
    insertions: list[tuple[int, str]] = []
    for span in pedal_spans:
        start_idx = _find_record_index_for_origin(indexed_lines, span.start_barline_line_idx)
        end_idx = _find_record_index_for_origin(indexed_lines, span.end_barline_line_idx)
        if start_idx is None or end_idx is None or end_idx <= start_idx:
            continue
        insertions.append(
            (
                start_idx + 1,
                _make_interpretation_line(
                    indexed_lines,
                    insertion_idx=start_idx + 1,
                    token=PEDAL_DOWN,
                ),
            )
        )
        insertions.append(
            (
                end_idx,
                _make_interpretation_line(
                    indexed_lines,
                    insertion_idx=end_idx,
                    token=PEDAL_UP,
                ),
            )
        )

    if not insertions:
        return indexed_lines

    for insert_idx, text in sorted(insertions, key=lambda item: item[0], reverse=True):
        indexed_lines.insert(insert_idx, _AnchoredLine(origin_line_idx=None, text=text))
    return indexed_lines


def _apply_instrument_label(
    lines: list[_AnchoredLine],
    *,
    include: bool,
) -> list[_AnchoredLine]:
    if not include:
        return lines
    text = "\n".join(line.text for line in lines)
    if _INSTRUMENT_TOKEN_RE.search(text):
        return lines

    spine_count = get_spine_count(text)
    if spine_count == 0:
        return lines

    piano_line = "\t".join(['*I"Piano'] * spine_count)
    insert_idx = 0
    for idx, line in enumerate(lines):
        if not line.text.strip():
            continue
        if line.text.startswith("**"):
            insert_idx = idx + 1
        else:
            insert_idx = idx
        break
    result = list(lines)
    result.insert(insert_idx, _AnchoredLine(origin_line_idx=None, text=piano_line))
    return result


def _apply_tempo(
    lines: list[_AnchoredLine],
    tempo: TempoPlan | None,
) -> list[_AnchoredLine]:
    if tempo is None:
        return lines

    text = "\n".join(line.text for line in lines)
    if has_tempo_markings(text):
        return lines

    result = list(lines)
    result.insert(
        _find_omd_insertion_point(result),
        _AnchoredLine(origin_line_idx=None, text=f"!!!OMD: {tempo.tempo_text}"),
    )
    if tempo.bpm is not None:
        spine_count = get_spine_count("\n".join(line.text for line in result))
        if spine_count > 0:
            mm_line = "\t".join([f"*MM{tempo.bpm}"] + ["*"] * (spine_count - 1))
            result.insert(
                _find_tempo_insertion_point(result),
                _AnchoredLine(origin_line_idx=None, text=mm_line),
            )
    return result


def _apply_trailing_dynam_spine(
    lines: list[_AnchoredLine],
    *,
    hairpins: tuple[HairpinPlan, ...],
    dynamic_marks: tuple[DynamicMarkPlan, ...],
) -> list[_AnchoredLine]:
    if not hairpins and not dynamic_marks:
        return lines

    raw_lines = [line.text for line in lines]
    trailing_dynam_idx = find_trailing_dynam_spine_index(raw_lines)
    index_by_origin = {
        line.origin_line_idx: idx
        for idx, line in enumerate(lines)
        if line.origin_line_idx is not None
    }

    if trailing_dynam_idx is None:
        dynam_tokens = [default_dynam_token(line.text) for line in lines]
    else:
        dynam_tokens = []
        for line in lines:
            fields = line.text.split("\t")
            dynam_tokens.append(fields[trailing_dynam_idx] if len(fields) > trailing_dynam_idx else None)

    for hairpin in hairpins:
        indices = [index_by_origin.get(origin_idx) for origin_idx in hairpin.origin_line_indices]
        if any(index is None for index in indices):
            continue
        current_indices = [index for index in indices if index is not None]
        if not current_indices:
            continue
        start_token, continuation_token, end_token = (
            ("<", "(", "[") if hairpin.is_crescendo else (">", ")", "]")
        )
        dynam_tokens[current_indices[0]] = start_token
        for current_idx in current_indices[1:-1]:
            dynam_tokens[current_idx] = continuation_token
        if len(current_indices) > 1:
            dynam_tokens[current_indices[-1]] = end_token

    for dynamic_mark in dynamic_marks:
        current_idx = index_by_origin.get(dynamic_mark.origin_line_idx)
        if current_idx is None:
            continue
        if dynam_tokens[current_idx] not in {None, "."}:
            continue
        dynam_tokens[current_idx] = dynamic_mark.token

    result: list[_AnchoredLine] = []
    for line, dynam_token in zip(lines, dynam_tokens, strict=True):
        if trailing_dynam_idx is None:
            if dynam_token is None:
                result.append(line)
            elif line.text:
                result.append(
                    _AnchoredLine(
                        origin_line_idx=line.origin_line_idx,
                        text=f"{line.text}\t{dynam_token}",
                    )
                )
            else:
                result.append(
                    _AnchoredLine(origin_line_idx=line.origin_line_idx, text=dynam_token)
                )
            continue

        fields = line.text.split("\t")
        if len(fields) > trailing_dynam_idx and dynam_token is not None:
            fields[trailing_dynam_idx] = dynam_token
            result.append(
                _AnchoredLine(
                    origin_line_idx=line.origin_line_idx,
                    text="\t".join(fields),
                )
            )
        else:
            result.append(line)
    return result


def _find_record_index_for_origin(
    lines: list[_AnchoredLine],
    origin_line_idx: int,
) -> int | None:
    for idx, line in enumerate(lines):
        if line.origin_line_idx == origin_line_idx:
            return idx
    return None


def _make_interpretation_line(
    lines: list[_AnchoredLine],
    *,
    insertion_idx: int,
    token: str,
) -> str:
    local_spine_count = get_local_spine_count([line.text for line in lines], insertion_idx)
    return "\t".join([token] + ["*"] * (max(1, local_spine_count) - 1))


def _find_omd_insertion_point(lines: list[_AnchoredLine]) -> int:
    for idx, line in enumerate(lines):
        text = line.text
        if not text.strip():
            continue
        if text.startswith("!"):
            continue
        return idx
    return 0


def _find_tempo_insertion_point(lines: list[_AnchoredLine]) -> int:
    for idx, line in enumerate(lines):
        text = line.text
        if not text.strip():
            continue
        if text.startswith("!"):
            continue
        if text.startswith("**"):
            continue
        if text.startswith("*") and not text.startswith("*-"):
            tokens = text.split("\t")
            if all(_is_header_interpretation(token) for token in tokens):
                continue
        return idx
    return len(lines)


def _is_header_interpretation(token: str) -> bool:
    if token == "*":
        return True
    if token.startswith("*clef"):
        return True
    if token.startswith("*k["):
        return True
    if token.startswith("*M") and len(token) > 2 and token[2].isdigit():
        return True
    return False


def _sample_non_overlapping_spans(
    *,
    num_steps: int,
    max_spans: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    max_possible = min(max_spans, num_steps // 2)
    if max_possible < 1:
        return []

    target_span_count = rng.randint(1, max_possible)
    available_ranges: list[tuple[int, int]] = [(0, num_steps - 1)]
    spans: list[tuple[int, int]] = []

    for _ in range(target_span_count):
        candidate_ranges = [item for item in available_ranges if (item[1] - item[0] + 1) >= 2]
        if not candidate_ranges:
            break

        range_start, range_end = rng.choice(candidate_ranges)
        range_len = range_end - range_start + 1
        span_len = rng.randint(2, min(_MAX_SPAN_STEPS, range_len))
        span_start = rng.randint(range_start, range_end - span_len + 1)
        span_end = span_start + span_len - 1
        spans.append((span_start, span_end))

        updated_ranges: list[tuple[int, int]] = []
        for current_start, current_end in available_ranges:
            if current_end < span_start or current_start > span_end:
                updated_ranges.append((current_start, current_end))
                continue
            if current_start <= span_start - 1:
                updated_ranges.append((current_start, span_start - 1))
            if span_end + 1 <= current_end:
                updated_ranges.append((span_end + 1, current_end))
        available_ranges = updated_ranges

    spans.sort()
    return spans
