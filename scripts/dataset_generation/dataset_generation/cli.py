"""CLI adapter for dataset generation service."""

from __future__ import annotations

from scripts.dataset_generation.data_spec import DEFAULT_DATA_SPEC_PATH
from scripts.dataset_generation.dataset_generation.config import (
    build_generation_run_config,
    coerce_boolish,
)
from scripts.dataset_generation.dataset_generation.service import run_generation


def main(
    *kern_dirs: str,
    dataset_preset: str | None = None,
    output_dir: str | None = None,
    num_samples: int = 10_000_000,
    target_accepted_samples: int | None = None,
    max_scheduled_tasks: int | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    data_spec_path: str = str(DEFAULT_DATA_SPEC_PATH),
    strict_data_spec: bool = True,
    num_workers: int | None = None,
    variants_per_file: int | None = None,
    adaptive_variants_enabled: bool | str | None = None,
    overflow_truncation_enabled: bool = True,
    overflow_truncation_max_trials: int | None = None,
    augment_seed: int | None = None,
    render_pedals_enabled: bool | str | None = None,
    render_pedals_probability: float = 0.20,
    render_pedals_measures_probability: float = 0.3,
    render_instrument_piano_enabled: bool | str | None = None,
    render_instrument_piano_probability: float = 0.15,
    render_sforzando_enabled: bool | str | None = None,
    render_sforzando_probability: float = 0.20,
    render_sforzando_per_note_probability: float = 0.03,
    render_accent_enabled: bool | str | None = None,
    render_accent_probability: float = 0.10,
    render_accent_per_note_probability: float = 0.015,
    render_tempo_enabled: bool | str | None = None,
    render_tempo_probability: float = 0.12,
    render_tempo_include_mm_probability: float = 0.35,
    render_hairpins_enabled: bool | str | None = None,
    render_hairpins_probability: float = 0.25,
    render_hairpins_max_spans: int = 2,
    render_dynamic_marks_enabled: bool | str | None = None,
    render_dynamic_marks_probability: float = 0.15,
    render_dynamic_marks_min_count: int = 1,
    render_dynamic_marks_max_count: int = 2,
    courtesy_naturals_probability: float | None = None,
    disable_offline_image_augmentations: bool | str | None = None,
    geom_x_squeeze_prob: float = 0.45,
    geom_x_squeeze_min_scale: float = 0.70,
    geom_x_squeeze_max_scale: float = 0.95,
    geom_x_squeeze_apply_in_conservative: bool = True,
    geom_x_squeeze_preview_force_scale: float | None = None,
    target_min_systems: int | None = None,
    target_max_systems: int | None = None,
    render_layout_profile: str | None = None,
    prefilter_min_non_empty_lines: int | None = None,
    prefilter_max_non_empty_lines: int | None = None,
    prefilter_min_measure_count: int | None = None,
    prefilter_max_measure_count: int | None = None,
    progress_enabled: bool = True,
    progress_update_interval_seconds: int = 30,
    profile_enabled: bool = False,
    profile_out_dir: str | None = None,
    profile_log_every: int = 100,
    profile_sample_limit: int = 500,
    profile_capture_per_sample: bool = False,
    failure_policy: str | None = None,
    resume_mode: str = "auto",
    quarantine_in: str | None = None,
    quarantine_out: str | None = None,
    artifacts_out_dir: str | None = None,
    start_method: str = "auto",
    quiet: bool = False,
):
    """Generate a dataset from normalized **kern files."""
    config = build_generation_run_config(
        kern_dirs=tuple(kern_dirs),
        dataset_preset=dataset_preset,
        output_dir=output_dir,
        num_samples=num_samples,
        target_accepted_samples=target_accepted_samples,
        max_scheduled_tasks=max_scheduled_tasks,
        image_width=image_width,
        image_height=image_height,
        data_spec_path=data_spec_path,
        strict_data_spec=coerce_boolish(strict_data_spec),
        num_workers=num_workers,
        variants_per_file=variants_per_file,
        adaptive_variants_enabled=(
            None if adaptive_variants_enabled is None else coerce_boolish(adaptive_variants_enabled)
        ),
        overflow_truncation_enabled=coerce_boolish(overflow_truncation_enabled),
        overflow_truncation_max_trials=overflow_truncation_max_trials,
        augment_seed=augment_seed,
        render_pedals_enabled=(
            None if render_pedals_enabled is None else coerce_boolish(render_pedals_enabled)
        ),
        render_pedals_probability=render_pedals_probability,
        render_pedals_measures_probability=render_pedals_measures_probability,
        render_instrument_piano_enabled=(
            None
            if render_instrument_piano_enabled is None
            else coerce_boolish(render_instrument_piano_enabled)
        ),
        render_instrument_piano_probability=render_instrument_piano_probability,
        render_sforzando_enabled=(
            None if render_sforzando_enabled is None else coerce_boolish(render_sforzando_enabled)
        ),
        render_sforzando_probability=render_sforzando_probability,
        render_sforzando_per_note_probability=render_sforzando_per_note_probability,
        render_accent_enabled=(
            None if render_accent_enabled is None else coerce_boolish(render_accent_enabled)
        ),
        render_accent_probability=render_accent_probability,
        render_accent_per_note_probability=render_accent_per_note_probability,
        render_tempo_enabled=(
            None if render_tempo_enabled is None else coerce_boolish(render_tempo_enabled)
        ),
        render_tempo_probability=render_tempo_probability,
        render_tempo_include_mm_probability=render_tempo_include_mm_probability,
        render_hairpins_enabled=(
            None if render_hairpins_enabled is None else coerce_boolish(render_hairpins_enabled)
        ),
        render_hairpins_probability=render_hairpins_probability,
        render_hairpins_max_spans=render_hairpins_max_spans,
        render_dynamic_marks_enabled=(
            None
            if render_dynamic_marks_enabled is None
            else coerce_boolish(render_dynamic_marks_enabled)
        ),
        render_dynamic_marks_probability=render_dynamic_marks_probability,
        render_dynamic_marks_min_count=render_dynamic_marks_min_count,
        render_dynamic_marks_max_count=render_dynamic_marks_max_count,
        courtesy_naturals_probability=courtesy_naturals_probability,
        disable_offline_image_augmentations=(
            None
            if disable_offline_image_augmentations is None
            else coerce_boolish(disable_offline_image_augmentations)
        ),
        geom_x_squeeze_prob=geom_x_squeeze_prob,
        geom_x_squeeze_min_scale=geom_x_squeeze_min_scale,
        geom_x_squeeze_max_scale=geom_x_squeeze_max_scale,
        geom_x_squeeze_apply_in_conservative=coerce_boolish(
            geom_x_squeeze_apply_in_conservative
        ),
        geom_x_squeeze_preview_force_scale=geom_x_squeeze_preview_force_scale,
        target_min_systems=target_min_systems,
        target_max_systems=target_max_systems,
        render_layout_profile=render_layout_profile,
        prefilter_min_non_empty_lines=prefilter_min_non_empty_lines,
        prefilter_max_non_empty_lines=prefilter_max_non_empty_lines,
        prefilter_min_measure_count=prefilter_min_measure_count,
        prefilter_max_measure_count=prefilter_max_measure_count,
        progress_enabled=coerce_boolish(progress_enabled),
        progress_update_interval_seconds=progress_update_interval_seconds,
        profile_enabled=coerce_boolish(profile_enabled),
        profile_out_dir=profile_out_dir,
        profile_log_every=profile_log_every,
        profile_sample_limit=profile_sample_limit,
        profile_capture_per_sample=coerce_boolish(profile_capture_per_sample),
        failure_policy=failure_policy,
        resume_mode=resume_mode,
        quarantine_in=quarantine_in,
        quarantine_out=quarantine_out,
        artifacts_out_dir=artifacts_out_dir,
        start_method=start_method,
        quiet=coerce_boolish(quiet),
    )
    run_generation(config)
    return None
