"""Build render-only transcriptions from planned label transcriptions."""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np

from scripts.dataset_generation.augmentation.articulations import apply_accents, apply_sforzandos
from scripts.dataset_generation.augmentation.hairpins import apply_render_hairpins
from scripts.dataset_generation.augmentation.instrument_label import apply_instrument_label_piano
from scripts.dataset_generation.augmentation.pedaling import apply_pedaling
from scripts.dataset_generation.augmentation.render_dynamic_marks import apply_render_dynamic_marks
from scripts.dataset_generation.augmentation.tempo_markings import apply_tempo_markings
from scripts.dataset_generation.dataset_generation_new.recipe import ProductionRecipe
from src.core.kern_postprocess import append_terminator_if_missing


def build_render_transcription(
    label_transcription: str,
    recipe: ProductionRecipe,
    *,
    seed: int,
) -> str:
    policy = recipe.render_only_aug
    with _seeded_random(seed):
        render_transcription = label_transcription
        if _roll(policy.render_pedals_probability):
            render_transcription = apply_pedaling(
                render_transcription,
                measures_probability=policy.render_pedals_measures_probability,
            )
        if _roll(policy.render_instrument_piano_probability):
            render_transcription = apply_instrument_label_piano(render_transcription)
        if _roll(policy.render_sforzando_probability):
            render_transcription = apply_sforzandos(
                render_transcription,
                per_note_probability=policy.render_sforzando_per_note_probability,
            )
        if _roll(policy.render_accent_probability):
            render_transcription = apply_accents(
                render_transcription,
                per_note_probability=policy.render_accent_per_note_probability,
            )
        if _roll(policy.render_tempo_probability):
            render_transcription = apply_tempo_markings(
                render_transcription,
                include_mm_probability=policy.render_tempo_include_mm_probability,
            )

        hairpins_added = False
        hairpins_version = apply_render_hairpins(
            render_transcription,
            sample_probability=policy.render_hairpins_probability,
            max_spans=policy.render_hairpins_max_spans,
        )
        if hairpins_version != render_transcription:
            hairpins_added = True
            render_transcription = hairpins_version

        dynamic_marks_version = apply_render_dynamic_marks(
            render_transcription,
            sample_probability=policy.render_dynamic_marks_probability,
            min_marks=policy.render_dynamic_marks_min_count,
            max_marks=policy.render_dynamic_marks_max_count,
            assume_trailing_dynam=hairpins_added,
        )
        dynamic_marks_added = dynamic_marks_version != render_transcription
        render_transcription = dynamic_marks_version

        return append_terminator_if_missing(
            ensure_render_header(
                render_transcription,
                last_spine_type="dynam" if (hairpins_added or dynamic_marks_added) else None,
            )
        )


def _roll(probability: float) -> bool:
    return probability > 0.0 and float(np.random.random()) < probability


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


@contextmanager
def _seeded_random(seed: int):
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
