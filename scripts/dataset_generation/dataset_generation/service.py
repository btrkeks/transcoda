"""Execution service for synthetic dataset generation."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import datasets
from datasets import Features, Value
from rich.console import Console

from scripts.dataset_generation.dataset_generation.config import (
    GenerationRunConfig,
    configure_start_method,
    load_quarantine_list,
    resolve_failure_policy,
    resolve_kern_dirs,
    resolve_worker_config,
)
from scripts.dataset_generation.dataset_generation.file_generator import FileDataGenerator
from scripts.dataset_generation.dataset_generation.resumable_dataset import (
    ResumableDatasetRunStore,
    compute_generation_config_fingerprint,
)
from scripts.dataset_generation.dataset_generation.reporting import (
    build_info_summary,
    write_info_and_latest_pointer,
    write_primary_artifacts,
    write_profile_artifacts,
)
from scripts.dataset_generation.dataset_generation.run_context import RunContext, build_run_context
from scripts.dataset_generation.dataset_generation.variant_policy import FIXED_VARIANTS_PER_FILE


@dataclass(frozen=True)
class DatasetGenerationRunResult:
    """Structured run result returned by the generation service."""

    run_context: RunContext
    total_samples: int
    total_size_gb: float
    runtime_seconds: dict[str, float]
    profile_enabled: bool
    profile_written: bool


def _format_system_histogram(histogram: dict[str, object]) -> str:
    if not histogram:
        return "none"
    sorted_items: list[tuple[int, int]] = []
    for key, value in histogram.items():
        try:
            sorted_items.append((int(key), int(value)))
        except (TypeError, ValueError):
            continue
    sorted_items.sort(key=lambda item: item[0])
    if not sorted_items:
        return "none"
    return " ".join(f"{systems}:{count}" for systems, count in sorted_items)


def _print_run_configuration(
    *,
    console: Console,
    config: GenerationRunConfig,
    run_context: RunContext,
    resolved_failure_policy: object,
    preloaded_quarantine_files: list[str],
    configured_start_method: str,
    worker_config: object,
    kern_dir_paths: list[object],
) -> None:
    console.print("[bold green]Generating synthetic dataset...")
    console.print(
        f"Dataset preset: {config.dataset_preset}"
        if config.dataset_preset is not None
        else "Dataset preset: custom"
    )
    console.print(f"Kern directories: {[str(d) for d in kern_dir_paths]}")
    if config.effective_target_accepted_samples is not None:
        console.print(f"Accepted target: {config.effective_target_accepted_samples:,}")
        console.print(
            "Max scheduled tasks: "
            f"{config.effective_max_scheduled_tasks:,}"
            if config.effective_max_scheduled_tasks is not None
            else "Max scheduled tasks: unbounded"
        )
    else:
        console.print(f"Max samples: {config.effective_num_samples:,}")
    console.print(f"Run artifacts dir: {run_context.run_artifacts_dir}")
    if config.profile_enabled:
        console.print(
            f"Profiling: enabled (requested={config.requested_num_samples:,}, "
            f"cap={config.profile_sample_limit:,}, out={run_context.profile_path})"
        )
    console.print(
        "Failure policy: "
        f"{resolved_failure_policy.name} "
        f"(timeout={resolved_failure_policy.task_timeout_seconds}, "
        f"retry_timeout={resolved_failure_policy.max_task_retries_timeout}, "
        f"retry_expired={resolved_failure_policy.max_task_retries_expired})"
    )
    if run_context.resolved_quarantine_in:
        console.print(
            "Preloaded quarantined files: "
            f"{len(preloaded_quarantine_files):,} "
            f"(from {run_context.resolved_quarantine_in})"
        )
    else:
        console.print(f"Preloaded quarantined files: {len(preloaded_quarantine_files):,}")
    console.print(f"Multiprocessing start method: {configured_start_method}")
    console.print(f"Variants per file: fixed at {FIXED_VARIANTS_PER_FILE}")
    console.print("Adaptive variants enabled: False (simplified fixed policy)")
    console.print(f"Courtesy naturals (label-side): {worker_config.courtesy_naturals_probability}")
    console.print(
        "Overflow truncation rescue: "
        f"{config.overflow_truncation_enabled} "
        f"(max_trials={config.overflow_truncation_max_trials})"
    )
    console.print(f"Image size: {worker_config.image_width}x{worker_config.image_height}")
    console.print(
        "Render pedals: "
        f"{worker_config.render_pedals_enabled} "
        f"(sample prob={worker_config.render_pedals_probability}, "
        f"measure prob={worker_config.render_pedals_measures_probability})"
    )
    console.print(
        'Render instrument *I"Piano: '
        f"{worker_config.render_instrument_piano_enabled} "
        f"(sample prob={worker_config.render_instrument_piano_probability})"
    )
    console.print(
        "Render sforzando z: "
        f"{worker_config.render_sforzando_enabled} "
        f"(sample prob={worker_config.render_sforzando_probability}, "
        f"per-note prob={worker_config.render_sforzando_per_note_probability})"
    )
    console.print(
        "Render accent ^: "
        f"{worker_config.render_accent_enabled} "
        f"(sample prob={worker_config.render_accent_probability}, "
        f"per-note prob={worker_config.render_accent_per_note_probability})"
    )
    console.print(
        "Render tempo (OMD + optional *MM): "
        f"{worker_config.render_tempo_enabled} "
        f"(sample prob={worker_config.render_tempo_probability}, "
        f"include-mm prob={worker_config.render_tempo_include_mm_probability})"
    )
    console.print(
        "Render hairpins **dynam: "
        f"{worker_config.render_hairpins_enabled} "
        f"(sample prob={worker_config.render_hairpins_probability}, "
        f"max spans={worker_config.render_hairpins_max_spans})"
    )
    console.print(
        "Render dynamic marks **dynam: "
        f"{worker_config.render_dynamic_marks_enabled} "
        f"(sample prob={worker_config.render_dynamic_marks_probability}, "
        f"count range=[{worker_config.render_dynamic_marks_min_count}, "
        f"{worker_config.render_dynamic_marks_max_count}])"
    )
    console.print(
        "Offline image augmentations (OpenCV + Augraphy): "
        f"{not worker_config.disable_offline_image_augmentations}"
    )
    console.print(
        "Geom x-squeeze: "
        f"prob={worker_config.geom_x_squeeze_prob}, "
        f"range=({worker_config.geom_x_squeeze_min_scale}, "
        f"{worker_config.geom_x_squeeze_max_scale}), "
        f"conservative={worker_config.geom_x_squeeze_apply_in_conservative}, "
        f"force={worker_config.geom_x_squeeze_preview_force_scale}"
    )
    console.print(f"Render layout profile: {worker_config.render_layout_profile}")
    console.print(
        "Target system band: "
        f"{worker_config.target_min_systems}-{worker_config.target_max_systems}"
        if worker_config.target_min_systems is not None and worker_config.target_max_systems is not None
        else "Target system band: disabled"
    )
    console.print(
        "Prefilter min non-empty lines: "
        f"{config.prefilter_min_non_empty_lines}"
        if config.prefilter_min_non_empty_lines is not None
        else "Prefilter min non-empty lines: disabled"
    )
    console.print(
        "Prefilter max non-empty lines: "
        f"{config.prefilter_max_non_empty_lines}"
        if config.prefilter_max_non_empty_lines is not None
        else "Prefilter max non-empty lines: disabled"
    )
    console.print(
        "Prefilter min measure count: "
        f"{config.prefilter_min_measure_count}"
        if config.prefilter_min_measure_count is not None
        else "Prefilter min measure count: disabled"
    )
    console.print(
        "Prefilter max measure count: "
        f"{config.prefilter_max_measure_count}"
        if config.prefilter_max_measure_count is not None
        else "Prefilter max measure count: disabled"
    )
    console.print(f"Live progress enabled: {config.progress_enabled}")
    console.print(
        "Live progress interval: "
        f"{config.progress_update_interval_seconds}s"
        if config.progress_enabled
        else "Live progress interval: disabled"
    )


def run_generation(
    config: GenerationRunConfig,
    *,
    console: Console | None = None,
) -> DatasetGenerationRunResult:
    """Execute one end-to-end dataset generation run."""
    active_console = console or Console()
    run_context = build_run_context(config)
    configured_start_method = configure_start_method(config.start_method)
    failure_policy = resolve_failure_policy(config.failure_policy)
    preloaded_quarantine_files = load_quarantine_list(run_context.resolved_quarantine_in)
    kern_dir_paths = resolve_kern_dirs(config.kern_dirs)
    worker_config = resolve_worker_config(config)
    config_fingerprint = compute_generation_config_fingerprint(
        config=config,
        resolved_kern_dirs=kern_dir_paths,
    )

    if not config.quiet:
        _print_run_configuration(
            console=active_console,
            config=config,
            run_context=run_context,
            resolved_failure_policy=failure_policy,
            preloaded_quarantine_files=preloaded_quarantine_files,
            configured_start_method=configured_start_method,
            worker_config=worker_config,
            kern_dir_paths=kern_dir_paths,
        )

    features = Features(
        {
            "image": datasets.Image(mode="RGB"),
            "transcription": Value("string"),
            "source": Value("string"),
            "source_dataset": Value("string"),
            "source_split": Value("string"),
            "sample_id": Value("string"),
            "curation_stage": Value("string"),
            "source_domain": Value("string"),
            "actual_system_count": Value("int32"),
            "truncation_applied": Value("bool"),
            "truncation_ratio": Value("float32"),
            "render_layout_profile": Value("string"),
            "bottom_whitespace_ratio": Value("float32"),
            "vertical_fill_ratio": Value("float32"),
            "source_non_empty_line_count": Value("int32"),
            "source_measure_count": Value("int32"),
        }
    )
    resume_store = ResumableDatasetRunStore(
        run_context=run_context,
        features=features,
        config_fingerprint=config_fingerprint,
        resume_mode=config.resume_mode,
    )
    resume_state = resume_store.prepare()
    generator_worker_kwargs = worker_config.to_dict()
    generator_worker_kwargs["deterministic_seed_salt"] = config_fingerprint

    generator = FileDataGenerator(
        kern_dirs=kern_dir_paths,
        num_workers=config.num_workers,
        variants_per_file=config.variants_per_file,
        adaptive_variants_enabled=config.adaptive_variants_enabled,
        progress_enabled=config.progress_enabled,
        progress_update_interval_seconds=config.progress_update_interval_seconds,
        progress_path=run_context.progress_path,
        progress_run_id=run_context.run_id,
        progress_output_dir=str(run_context.output_path),
        progress_run_artifacts_dir=str(run_context.run_artifacts_dir),
        target_accepted_samples=config.effective_target_accepted_samples,
        max_scheduled_tasks=config.effective_max_scheduled_tasks,
        prefilter_min_non_empty_lines=config.prefilter_min_non_empty_lines,
        prefilter_max_non_empty_lines=config.prefilter_max_non_empty_lines,
        prefilter_min_measure_count=config.prefilter_min_measure_count,
        prefilter_max_measure_count=config.prefilter_max_measure_count,
        task_timeout_seconds=failure_policy.task_timeout_seconds,
        max_task_retries_timeout=failure_policy.max_task_retries_timeout,
        max_task_retries_expired=failure_policy.max_task_retries_expired,
        initial_quarantined_files=preloaded_quarantine_files,
        profile_log_every=config.profile_log_every,
        profile_capture_per_sample=config.profile_capture_per_sample,
        resume_state=resume_state,
        resume_observer=resume_store,
        **generator_worker_kwargs,
    )

    if not config.quiet:
        active_console.print(f"Found {len(generator.file_paths):,} kern files")
        policy_summary = generator.variant_policy_summary
        active_console.print(
            "Effective variant policy: "
            f"{policy_summary.get('policy')} "
            f"(mean variants/file={float(policy_summary.get('mean_variants_per_file', 0.0)):.2f}, "
            f"total tasks={int(policy_summary.get('total_available_tasks', 0)):,})"
        )

    run_start_ns = time.perf_counter_ns()
    generation_start_ns = time.perf_counter_ns()
    finalized_dataset: dict[str, object] | None = None
    try:
        for _sample in generator.generate(
            num_samples=config.effective_num_samples,
            target_accepted_samples=config.effective_target_accepted_samples,
            max_scheduled_tasks=config.effective_max_scheduled_tasks,
        ):
            del _sample
        generation_elapsed_s = (time.perf_counter_ns() - generation_start_ns) / 1_000_000_000.0

        if not config.quiet and generator.stats.has_failures:
            stats = generator.stats
            active_console.print(
                f"[yellow]Stats: {stats.successful} successful, "
                f"{stats.overflows} overflows, {stats.invalid} invalid, "
                f"{stats.rejected_sparse} sparse rejects, "
                f"{stats.rejected_render_fit} fit rejects, "
                f"{stats.errors} errors, {stats.timeouts} timeouts, "
                f"{stats.expired_workers} expired workers"
            )

        if not config.quiet:
            active_console.print(f"Finalizing dataset in {run_context.output_path}...")
        finalize_start_ns = time.perf_counter_ns()
        finalized_dataset = resume_store.finalize(generator=generator)
        save_elapsed_s = (time.perf_counter_ns() - finalize_start_ns) / 1_000_000_000.0
        generator.emit_external_progress("completed")
        total_elapsed_s = (time.perf_counter_ns() - run_start_ns) / 1_000_000_000.0
    except BaseException as exc:
        resume_store.mark_terminal_status(
            generator=generator,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
        )
        generator.emit_external_progress(
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        raise

    assert finalized_dataset is not None
    total_size_bytes = int(finalized_dataset["total_size_bytes"])
    total_size_gb = total_size_bytes / (1024**3)
    runtime_seconds = {
        "generation": generation_elapsed_s,
        "save_to_disk": save_elapsed_s,
        "finalize_dataset": save_elapsed_s,
        "total": total_elapsed_s,
    }

    write_primary_artifacts(
        run_context=run_context,
        quarantine_summary=generator.last_quarantine_summary,
        failure_summary=generator.last_failure_summary,
        layout_summary=generator.last_layout_summary,
        scheduler_summary=generator.last_scheduler_summary,
        write_latest_quarantine=config.quarantine_out is None,
    )

    info_summary = build_info_summary(
        config=config,
        run_context=run_context,
        worker_config=worker_config,
        configured_start_method=configured_start_method,
        failure_policy=failure_policy,
        runtime_seconds=runtime_seconds,
        total_samples=generator.stats.successful,
        total_size_gb=total_size_gb,
        resolved_quarantine_in=run_context.resolved_quarantine_in,
        resolved_kern_dirs=kern_dir_paths,
        variant_policy_summary=generator.variant_policy_summary,
        prefilter_summary=generator.prefilter_summary,
        resumable_state={
            "config_fingerprint": config_fingerprint,
            "resume_mode": config.resume_mode,
            "resume_session_id": resume_store.resume_session_id,
            "resumed": resume_store.resumed,
            "resumed_from_session_id": resume_store.resumed_from_session_id,
            "committed_shard_count": len(finalized_dataset["shard_lengths"]),
            "completed_sample_count": generator.stats.successful,
            "completion_flag": True,
            "resume_db_path": str(run_context.resume_db_path),
        },
    )
    write_info_and_latest_pointer(
        run_context=run_context,
        info_summary=info_summary,
    )

    profile_written = False
    if config.profile_enabled and generator.profile_report is not None:
        write_profile_artifacts(
            run_context=run_context,
            profile_report=generator.profile_report,
            info_summary=info_summary,
            stats=generator.stats,
            generation_elapsed_s=generation_elapsed_s,
            total_samples=generator.stats.successful,
            argv=list(sys.argv),
        )
        profile_written = True

    if not config.quiet:
        active_console.print("[bold green]Dataset generation complete!")
        active_console.print(f"Total size: {total_size_gb:.2f} GB")
        active_console.print(f"Total samples: {generator.stats.successful:,}")
        active_console.print(f"Run artifacts: {run_context.run_artifacts_dir}")
        active_console.print(f"Failure summary: {run_context.failure_summary_path}")
        active_console.print(f"Scheduler summary: {run_context.scheduler_summary_path}")
        layout_summary = generator.last_layout_summary
        histogram = layout_summary.get("systems_histogram", {})
        known_count = int(layout_summary.get("with_known_system_count", 0))
        unknown_count = int(layout_summary.get("with_unknown_system_count", 0))
        active_console.print(
            "Accepted systems histogram: "
            f"{_format_system_histogram(histogram)} "
            f"(known={known_count:,}, unknown={unknown_count:,})"
        )
        active_console.print(
            "Accepted >=6 systems: "
            f"{int(layout_summary.get('accepted_ge_6_systems', 0)):,} "
            f"({float(layout_summary.get('accepted_ge_6_systems_rate', 0.0)):.2%})"
        )
        target_band = layout_summary.get("target_system_band")
        if isinstance(target_band, dict):
            active_console.print(
                "Accepted in target band: "
                f"{int(layout_summary.get('accepted_in_target_band', 0)):,} "
                f"({float(layout_summary.get('accepted_in_target_band_rate', 0.0)):.2%}) "
                f"for {target_band.get('min')}-{target_band.get('max')}"
            )
        bottom_stats = layout_summary.get("bottom_whitespace_ratio_stats", {})
        fill_stats = layout_summary.get("vertical_fill_ratio_stats", {})
        active_console.print(
            "Bottom whitespace ratio: "
            f"mean={float(bottom_stats.get('mean', 0.0)):.2%} "
            f"p50={float(bottom_stats.get('p50', 0.0)):.2%} "
            f"p95={float(bottom_stats.get('p95', 0.0)):.2%}"
        )
        active_console.print(
            "Vertical fill ratio: "
            f"mean={float(fill_stats.get('mean', 0.0)):.2%} "
            f"p50={float(fill_stats.get('p50', 0.0)):.2%} "
            f"p95={float(fill_stats.get('p95', 0.0)):.2%}"
        )
        active_console.print(f"Layout summary: {run_context.layout_summary_path}")
        if profile_written:
            active_console.print(f"Profile artifacts: {run_context.profile_path}")

    return DatasetGenerationRunResult(
        run_context=run_context,
        total_samples=generator.stats.successful,
        total_size_gb=total_size_gb,
        runtime_seconds=runtime_seconds,
        profile_enabled=config.profile_enabled,
        profile_written=profile_written,
    )
