"""FileDataGenerator for generating samples from kern files on disk."""

from __future__ import annotations

import contextlib
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pebble import ProcessExpired, ProcessPool

from scripts.dataset_generation.dataset_generation.base import GenerationStats
from scripts.dataset_generation.dataset_generation.config import coerce_boolish
from scripts.dataset_generation.dataset_generation.failures import (
    WorkerFailureReason,
    build_failure_reason_counts,
    legacy_message_to_failure_reason,
    record_worker_failure,
)
from scripts.dataset_generation.dataset_generation.progress import (
    build_generation_progress_snapshot,
    format_generation_progress_line,
    write_progress_snapshot_atomic,
)
from scripts.dataset_generation.dataset_generation.resumable_dataset import GeneratorResumeState
from scripts.dataset_generation.dataset_generation.source_stats import (
    KernSourceStats,
    compute_kern_source_stats,
)
from scripts.dataset_generation.dataset_generation.variant_policy import (
    FIXED_VARIANTS_PER_FILE,
    build_fixed_variant_plan,
)
from scripts.dataset_generation.dataset_generation.worker import (
    generate_sample_from_path_outcome,
    init_file_worker,
)
from scripts.dataset_generation.dataset_generation.worker_models import (
    PROFILE_STAGE_NAMES,
    SampleFailure,
    SampleSuccess,
    WorkerInitConfig,
)

# Timeout in seconds for each sample generation task.
DEFAULT_TASK_TIMEOUT_SECONDS = 60
# Retry task once on transient process failures before counting as terminal failure.
DEFAULT_MAX_TASK_RETRIES = 1
# Keep only a small number of in-flight tasks to bound memory usage.
MAX_IN_FLIGHT_MULTIPLIER = 2


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summarize_series(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    total = float(sum(values))
    return {
        "count": len(values),
        "total_ms": total,
        "mean_ms": total / len(values),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "p50_ms": _percentile(values, 50.0),
        "p95_ms": _percentile(values, 95.0),
        "p99_ms": _percentile(values, 99.0),
    }


def _summarize_ratio_series(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": len(values),
        "mean": float(sum(values)) / len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
    }


def _build_layout_summary(
    *,
    accepted_samples_total: int,
    systems_histogram: Counter[int],
    unknown_system_count_successes: int,
    target_min_systems: int | None,
    target_max_systems: int | None,
    bottom_whitespace_ratios: list[float],
    vertical_fill_ratios: list[float],
) -> dict[str, Any]:
    known_count = int(sum(int(count) for count in systems_histogram.values()))
    ge_6_count = int(
        sum(int(count) for systems, count in systems_histogram.items() if int(systems) >= 6)
    )
    ge_6_rate = (
        float(ge_6_count / accepted_samples_total) if accepted_samples_total > 0 else 0.0
    )
    accepted_in_target_band = 0
    if target_min_systems is not None and target_max_systems is not None:
        accepted_in_target_band = int(
            sum(
                int(count)
                for systems, count in systems_histogram.items()
                if int(target_min_systems) <= int(systems) <= int(target_max_systems)
            )
        )
    accepted_in_target_band_rate = (
        float(accepted_in_target_band / accepted_samples_total)
        if accepted_samples_total > 0
        else 0.0
    )
    return {
        "version": 1,
        "population": "accepted_samples",
        "accepted_samples_total": int(accepted_samples_total),
        "with_known_system_count": known_count,
        "with_unknown_system_count": int(unknown_system_count_successes),
        "systems_histogram": {
            str(system_count): int(count)
            for system_count, count in sorted(
                systems_histogram.items(),
                key=lambda item: int(item[0]),
            )
        },
        "accepted_ge_6_systems": ge_6_count,
        "accepted_ge_6_systems_rate": ge_6_rate,
        "target_system_band": (
            {"min": int(target_min_systems), "max": int(target_max_systems)}
            if target_min_systems is not None and target_max_systems is not None
            else None
        ),
        "accepted_in_target_band": accepted_in_target_band,
        "accepted_in_target_band_rate": accepted_in_target_band_rate,
        "bottom_whitespace_ratio_stats": _summarize_ratio_series(bottom_whitespace_ratios),
        "vertical_fill_ratio_stats": _summarize_ratio_series(vertical_fill_ratios),
    }


def _bucket_for_worker_result(result: object) -> str:
    if isinstance(result, SampleSuccess):
        return "success"
    if isinstance(result, SampleFailure):
        return result.code
    if isinstance(result, tuple) and len(result) == 3 and isinstance(result[1], str):
        image, trans_or_error, _source = result
        if image is not None:
            return "success"
        return legacy_message_to_failure_reason(trans_or_error)
    return "unknown_result"


def _summarize_running_series_map(
    values_by_key: dict[str, _RunningSeriesSummary],
) -> dict[str, dict[str, float | int]]:
    return {
        key: values_by_key[key].as_dict()
        for key in sorted(values_by_key)
    }


def _build_scheduler_summary(
    *,
    accepted_target: int | None,
    max_scheduled_tasks: int | None,
    stop_condition: str,
    file_states: dict[Path, _FileRunState],
    timing_by_outcome: dict[str, dict[str, _RunningSeriesSummary]],
    worker_stage_by_outcome: dict[str, dict[str, _RunningSeriesSummary]],
    total_available_tasks: int,
    submitted_tasks: int,
    terminal_completed_tasks: int,
    successful_samples: int,
) -> dict[str, Any]:
    attempted_counts = [state.attempted for state in file_states.values()]
    success_counts = [state.successful for state in file_states.values()]
    files_early_stopped = [state for state in file_states.values() if state.early_stopped]
    files_with_success = [state for state in file_states.values() if state.successful > 0]
    high_yield = sorted(
        file_states.values(),
        key=lambda state: (-state.smoothed_success_rate, state.attempted, str(state.path)),
    )[:10]
    low_yield = sorted(
        file_states.values(),
        key=lambda state: (state.smoothed_success_rate, -state.attempted, str(state.path)),
    )[:10]
    return {
        "version": 1,
        "stop_condition": stop_condition,
        "accepted_target": accepted_target,
        "max_scheduled_tasks": max_scheduled_tasks,
        "total_available_tasks": int(total_available_tasks),
        "submitted_tasks": int(submitted_tasks),
        "terminal_completed_tasks": int(terminal_completed_tasks),
        "successful_samples": int(successful_samples),
        "file_counts": {
            "total": len(file_states),
            "schedulable_remaining": int(
                sum(
                    1
                    for state in file_states.values()
                    if state.remaining_variants > 0
                    and not state.in_flight
                    and not state.quarantined
                    and not state.early_stopped
                )
            ),
            "with_at_least_one_success": len(files_with_success),
            "early_stopped": len(files_early_stopped),
            "quarantined": int(sum(1 for state in file_states.values() if state.quarantined)),
        },
        "per_file_attempts": _summarize_ratio_series([float(value) for value in attempted_counts]),
        "per_file_successes": _summarize_ratio_series([float(value) for value in success_counts]),
        "top_high_yield_files": [
            {
                "source": str(state.path),
                "attempted": state.attempted,
                "successful": state.successful,
                "failed": state.failed,
                "remaining_variants": state.remaining_variants,
                "smoothed_success_rate": state.smoothed_success_rate,
            }
            for state in high_yield
            if state.attempted > 0
        ],
        "top_low_yield_files": [
            {
                "source": str(state.path),
                "attempted": state.attempted,
                "successful": state.successful,
                "failed": state.failed,
                "remaining_variants": state.remaining_variants,
                "smoothed_success_rate": state.smoothed_success_rate,
            }
            for state in low_yield
            if state.attempted > 0
        ],
        "timing_by_outcome": {
            outcome: _summarize_running_series_map(series_by_metric)
            for outcome, series_by_metric in sorted(timing_by_outcome.items())
        },
        "worker_stage_timing_by_outcome": {
            outcome: _summarize_running_series_map(series_by_stage)
            for outcome, series_by_stage in sorted(worker_stage_by_outcome.items())
        },
    }


def _sample_id_from_relative_source(relative_source: Path, variant_idx: int) -> str:
    relative_stem = relative_source.with_suffix("")
    normalized = "__".join(relative_stem.parts)
    return f"{normalized}__v{variant_idx}"


def _has_target_system_band(
    target_min_systems: int | None,
    target_max_systems: int | None,
) -> bool:
    return target_min_systems is not None and target_max_systems is not None


def _system_count_in_target_band(
    system_count: int | None,
    *,
    target_min_systems: int | None,
    target_max_systems: int | None,
) -> bool:
    if not _has_target_system_band(target_min_systems, target_max_systems):
        return True
    if system_count is None:
        return False
    assert target_min_systems is not None
    assert target_max_systems is not None
    return target_min_systems <= int(system_count) <= target_max_systems


def _system_band_failure_code(
    system_count: int | None,
    *,
    target_min_systems: int | None,
    target_max_systems: int | None,
    truncation_attempted: bool = False,
) -> str:
    if not _has_target_system_band(target_min_systems, target_max_systems):
        return "system_band_rejected"
    if system_count is None:
        return "system_band_truncation_exhausted" if truncation_attempted else "system_band_rejected"
    assert target_min_systems is not None
    assert target_max_systems is not None
    if int(system_count) < target_min_systems:
        return "system_band_below_min"
    if int(system_count) > target_max_systems:
        return "system_band_above_max"
    return "system_band_rejected"


def _build_timeout_event(
    *,
    file_path: Path,
    variant_idx: int,
    retry_count: int,
    queue_wait_ms: float,
    will_retry: bool,
) -> dict[str, Any]:
    return {
        "event": "timeout",
        "timestamp": time.time(),
        "file_path": str(file_path),
        "variant_idx": variant_idx,
        "retry_count": retry_count,
        "will_retry": will_retry,
        "queue_wait_ms": queue_wait_ms,
    }


def _build_process_expired_event(
    *,
    exc: ProcessExpired,
    file_path: Path,
    variant_idx: int,
    retry_count: int,
    queue_wait_ms: float,
    will_retry: bool,
) -> dict[str, Any]:
    return {
        "event": "process_expired",
        "timestamp": time.time(),
        "file_path": str(file_path),
        "variant_idx": variant_idx,
        "retry_count": retry_count,
        "will_retry": will_retry,
        "queue_wait_ms": queue_wait_ms,
        "exception_type": type(exc).__name__,
        "exception_repr": repr(exc),
        "exception_args": [repr(arg) for arg in getattr(exc, "args", ())],
        "pid": getattr(exc, "pid", None),
        "exitcode": getattr(exc, "exitcode", None),
    }


def _build_quarantine_event(
    *,
    file_path: Path,
    reason: str,
    trigger_event: str,
    retry_count: int,
    queue_wait_ms: float,
    dropped_pending_tasks: int,
    exitcode: int | None = None,
) -> dict[str, Any]:
    return {
        "event": "file_quarantined",
        "timestamp": time.time(),
        "file_path": str(file_path),
        "reason": reason,
        "trigger_event": trigger_event,
        "retry_count": retry_count,
        "queue_wait_ms": queue_wait_ms,
        "dropped_pending_tasks": dropped_pending_tasks,
        "exitcode": exitcode,
    }


def _future_result_with_timeout(future: Any, timeout_s: float) -> Any:
    """Call future.result(timeout=...) with compatibility fallback for test doubles."""
    try:
        return future.result(timeout=timeout_s)
    except TypeError:
        return future.result()


@dataclass(frozen=True)
class GeneratorRunResult:
    """Structured per-run summary exported by FileDataGenerator."""

    stats: dict[str, int]
    quarantine_summary: dict[str, Any]
    failure_summary: dict[str, Any]
    layout_summary: dict[str, Any]
    scheduler_summary: dict[str, Any]
    profile_report: dict[str, Any] | None


@dataclass
class _FileRunState:
    path: Path
    total_variants: int
    next_variant_idx: int = 0
    attempted: int = 0
    successful: int = 0
    failed: int = 0
    in_flight: bool = False
    quarantined: bool = False
    early_stopped: bool = False

    @property
    def remaining_variants(self) -> int:
        return max(0, int(self.total_variants) - int(self.next_variant_idx))

    @property
    def has_bootstrapped(self) -> bool:
        return self.attempted > 0 or self.in_flight or self.next_variant_idx > 0

    @property
    def has_any_success(self) -> bool:
        return self.successful > 0

    @property
    def smoothed_success_rate(self) -> float:
        return float(self.successful + 1) / float(self.attempted + 2)


@dataclass
class _RunningSeriesSummary:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float | None = None
    max_ms: float | None = None

    def add(self, value_ms: float) -> None:
        value = float(value_ms)
        self.count += 1
        self.total_ms += value
        self.min_ms = value if self.min_ms is None else min(self.min_ms, value)
        self.max_ms = value if self.max_ms is None else max(self.max_ms, value)

    def as_dict(self) -> dict[str, float | int]:
        mean_ms = self.total_ms / self.count if self.count > 0 else 0.0
        return {
            "count": int(self.count),
            "total_ms": float(self.total_ms),
            "mean_ms": float(mean_ms),
            "min_ms": float(self.min_ms) if self.min_ms is not None else 0.0,
            "max_ms": float(self.max_ms) if self.max_ms is not None else 0.0,
        }


class FileDataGenerator:
    """Multiprocessing-based generator that reads kern files from disk.

    This class generates samples by reading existing .krn files and rendering
    them as images. Each file can produce one or more augmented variants.

    Usage:
        generator = FileDataGenerator(
            kern_dirs="data/interim/pdmx/train",
            image_width=1050,
            image_height=620,
            num_workers=4
        )

        for sample in generator.generate(num_samples=1000):
            # sample is a dict with 'image' (bytes) and 'transcription' (str)
            process_sample(sample)
    """

    def __init__(
        self,
        kern_dirs: str | Path | list[str | Path],
        image_width: int = 1050,
        image_height: int | None = None,
        num_workers: int | None = None,
        variants_per_file: int = 3,
        adaptive_variants_enabled: bool = False,
        augment_seed: int | None = None,
        render_pedals_enabled: bool = True,
        render_pedals_probability: float = 0.20,
        render_pedals_measures_probability: float = 0.3,
        render_instrument_piano_enabled: bool = True,
        render_instrument_piano_probability: float = 0.15,
        render_sforzando_enabled: bool = True,
        render_sforzando_probability: float = 0.20,
        render_sforzando_per_note_probability: float = 0.03,
        render_accent_enabled: bool = True,
        render_accent_probability: float = 0.10,
        render_accent_per_note_probability: float = 0.015,
        render_tempo_enabled: bool = True,
        render_tempo_probability: float = 0.12,
        render_tempo_include_mm_probability: float = 0.35,
        render_hairpins_enabled: bool = True,
        render_hairpins_probability: float = 0.25,
        render_hairpins_max_spans: int = 2,
        render_dynamic_marks_enabled: bool = True,
        render_dynamic_marks_probability: float = 0.15,
        render_dynamic_marks_min_count: int = 1,
        render_dynamic_marks_max_count: int = 2,
        courtesy_naturals_probability: float = 0.15,
        disable_offline_image_augmentations: bool = False,
        geom_x_squeeze_prob: float = 0.45,
        geom_x_squeeze_min_scale: float = 0.70,
        geom_x_squeeze_max_scale: float = 0.95,
        geom_x_squeeze_apply_in_conservative: bool = True,
        geom_x_squeeze_preview_force_scale: float | None = None,
        target_min_systems: int | None = None,
        target_max_systems: int | None = None,
        render_layout_profile: str = "default",
        prefilter_min_non_empty_lines: int | None = None,
        prefilter_max_non_empty_lines: int | None = None,
        prefilter_min_measure_count: int | None = None,
        prefilter_max_measure_count: int | None = None,
        progress_enabled: bool = True,
        progress_update_interval_seconds: int = 30,
        progress_path: str | Path | None = None,
        progress_run_id: str | None = None,
        progress_output_dir: str | None = None,
        progress_run_artifacts_dir: str | None = None,
        target_accepted_samples: int | None = None,
        max_scheduled_tasks: int | None = None,
        overflow_truncation_enabled: bool = True,
        overflow_truncation_max_trials: int = 24,
        task_timeout_seconds: int = DEFAULT_TASK_TIMEOUT_SECONDS,
        max_task_retries_timeout: int = DEFAULT_MAX_TASK_RETRIES,
        max_task_retries_expired: int = DEFAULT_MAX_TASK_RETRIES,
        initial_quarantined_files: list[str | Path] | set[str | Path] | None = None,
        profile_enabled: bool = False,
        profile_log_every: int = 100,
        profile_capture_per_sample: bool = False,
        resume_state: GeneratorResumeState | None = None,
        resume_observer: Any | None = None,
        deterministic_seed_salt: str | None = None,
    ):
        """Initialize the file-based data generator.

        Args:
            kern_dirs: Path or list of paths to directories containing .krn files.
            image_width: Target image width in pixels (default: 1050).
            image_height: Target image height in pixels (default: None, content-determined).
            num_workers: Number of worker processes (default: min(4, CPU count - 1)).
            variants_per_file: Legacy CLI/config knob. The current scheduler
                uses a fixed 3 variants per source file.
            adaptive_variants_enabled: Legacy knob retained for CLI
                compatibility. Adaptive planning is currently disabled.
            augment_seed: Optional seed for deterministic augmentation.
            deterministic_seed_salt: Optional salt used to derive stable
                per-sample RNG seeds across resume sessions.
            render_pedals_enabled: Whether to inject pedal markings for rendering only.
            render_pedals_probability: Per-sample probability of applying
                pedal augmentation before the per-measure pedal probability.
            render_pedals_measures_probability: Per-measure probability of inserting pedal
                markings for rendering only.
            render_instrument_piano_enabled: Whether to inject *I"Piano labels for
                rendering only.
            render_instrument_piano_probability: Per-sample probability of inserting
                *I"Piano labels for rendering only.
            render_sforzando_enabled: Whether to inject note-level z articulations
                for rendering only.
            render_sforzando_probability: Per-sample probability of inserting
                note-level z articulations.
            render_sforzando_per_note_probability: Per-note probability for z
                insertion when enabled for a sample.
            render_accent_enabled: Whether to inject note-level ^ accents for
                rendering only.
            render_accent_probability: Per-sample probability of inserting
                note-level ^ accents.
            render_accent_per_note_probability: Per-note probability for ^
                insertion when enabled for a sample.
            render_tempo_enabled: Whether to inject OMD/*MM tempo metadata for
                rendering only.
            render_tempo_probability: Per-sample probability of applying
                render-only tempo augmentation.
            render_tempo_include_mm_probability: Conditional probability of
                adding numeric *MM when tempo augmentation is applied.
            render_hairpins_enabled: Whether to inject a temporary trailing
                ``**dynam`` spine with hairpins for rendering only.
            render_hairpins_probability: Per-sample probability of applying
                render-only hairpin augmentation.
            render_hairpins_max_spans: Maximum non-overlapping hairpin spans
                sampled when hairpin augmentation is applied.
            render_dynamic_marks_enabled: Whether to inject canonical terraced
                dynamic marks in a trailing ``**dynam`` spine for rendering only.
            render_dynamic_marks_probability: Per-sample probability of applying
                render-only terraced dynamic marks.
            render_dynamic_marks_min_count: Minimum dynamic marks to inject
                when dynamic-mark augmentation is applied.
            render_dynamic_marks_max_count: Maximum dynamic marks to inject
                when dynamic-mark augmentation is applied.
            courtesy_naturals_probability: Per-sample probability of applying
                courtesy naturals render-only augmentation.
            disable_offline_image_augmentations: Skip raster-stage geometric
                OpenCV augmentation and Augraphy document artifacts.
            geom_x_squeeze_prob: Probability of applying explicit horizontal squeeze.
            geom_x_squeeze_min_scale: Minimum horizontal squeeze scale.
            geom_x_squeeze_max_scale: Maximum horizontal squeeze scale.
            geom_x_squeeze_apply_in_conservative: Whether squeeze can be applied during
                conservative geometric retry pass.
            geom_x_squeeze_preview_force_scale: Optional forced squeeze scale for debugging.
            target_min_systems: Optional inclusive lower bound on accepted system count.
            target_max_systems: Optional inclusive upper bound on accepted system count.
            render_layout_profile: Render option sampling profile.
            prefilter_min_non_empty_lines: Optional minimum non-empty line threshold
                applied before variant planning.
            prefilter_max_non_empty_lines: Optional maximum non-empty line threshold
                applied before variant planning.
            prefilter_min_measure_count: Optional minimum measure-count threshold
                applied before variant planning.
            prefilter_max_measure_count: Optional maximum measure-count threshold
                applied before variant planning.
            progress_enabled: Whether to emit periodic accepted-sample progress.
            progress_update_interval_seconds: Minimum interval between live
                progress emissions in seconds.
            progress_path: Optional heartbeat JSON path updated during generation.
            progress_run_id: Optional run identifier included in the heartbeat.
            progress_output_dir: Output dataset directory recorded in the heartbeat.
            progress_run_artifacts_dir: Run artifacts directory recorded in the heartbeat.
            target_accepted_samples: Optional accepted-sample stopping target.
            max_scheduled_tasks: Optional cap on total scheduled tasks.
            overflow_truncation_enabled: Whether to try truncating multi-page
                samples to a single-page prefix before dropping.
            overflow_truncation_max_trials: Max truncation prefix attempts per
                multi-page sample.
            task_timeout_seconds: Per-task timeout used for pool.schedule(..., timeout=...).
            max_task_retries_timeout: Number of retry attempts after timeout failures.
            max_task_retries_expired: Number of retry attempts after ProcessExpired failures.
            initial_quarantined_files: Optional file paths to skip scheduling at start.
            profile_enabled: Enable detailed worker/orchestrator timing capture.
            profile_log_every: Emit profile progress every N completed tasks when profiling.
            profile_capture_per_sample: Include per-sample profiling records in the final report.
            resume_state: Persisted scheduler and acceptance state to resume from.
            resume_observer: Callback used to persist terminal task state.

        Raises:
            ValueError: If no .krn files are found in any of the directories.
        """
        from multiprocessing import cpu_count

        # Handle single path or list of paths
        if isinstance(kern_dirs, (str, Path)):
            kern_dirs = [kern_dirs]

        self.kern_dirs = [Path(d) for d in kern_dirs]
        self.num_workers = num_workers or min(4, max(1, cpu_count() - 1))
        self.variants_per_file = FIXED_VARIANTS_PER_FILE
        self.adaptive_variants_enabled = False
        self.worker_config = WorkerInitConfig(
            image_width=image_width,
            image_height=image_height,
            augment_seed=augment_seed,
            deterministic_seed_salt=deterministic_seed_salt,
            render_pedals_enabled=render_pedals_enabled,
            render_pedals_probability=render_pedals_probability,
            render_pedals_measures_probability=render_pedals_measures_probability,
            render_instrument_piano_enabled=render_instrument_piano_enabled,
            render_instrument_piano_probability=render_instrument_piano_probability,
            render_sforzando_enabled=render_sforzando_enabled,
            render_sforzando_probability=render_sforzando_probability,
            render_sforzando_per_note_probability=render_sforzando_per_note_probability,
            render_accent_enabled=render_accent_enabled,
            render_accent_probability=render_accent_probability,
            render_accent_per_note_probability=render_accent_per_note_probability,
            render_tempo_enabled=render_tempo_enabled,
            render_tempo_probability=render_tempo_probability,
            render_tempo_include_mm_probability=render_tempo_include_mm_probability,
            render_hairpins_enabled=render_hairpins_enabled,
            render_hairpins_probability=render_hairpins_probability,
            render_hairpins_max_spans=render_hairpins_max_spans,
            render_dynamic_marks_enabled=render_dynamic_marks_enabled,
            render_dynamic_marks_probability=render_dynamic_marks_probability,
            render_dynamic_marks_min_count=render_dynamic_marks_min_count,
            render_dynamic_marks_max_count=render_dynamic_marks_max_count,
            courtesy_naturals_probability=courtesy_naturals_probability,
            disable_offline_image_augmentations=disable_offline_image_augmentations,
            geom_x_squeeze_prob=geom_x_squeeze_prob,
            geom_x_squeeze_min_scale=geom_x_squeeze_min_scale,
            geom_x_squeeze_max_scale=geom_x_squeeze_max_scale,
            geom_x_squeeze_apply_in_conservative=geom_x_squeeze_apply_in_conservative,
            geom_x_squeeze_preview_force_scale=geom_x_squeeze_preview_force_scale,
            target_min_systems=target_min_systems,
            target_max_systems=target_max_systems,
            render_layout_profile=render_layout_profile,
            overflow_truncation_enabled=overflow_truncation_enabled,
            overflow_truncation_max_trials=overflow_truncation_max_trials,
            profile_enabled=profile_enabled,
        )
        self.image_width = self.worker_config.image_width
        self.image_height = self.worker_config.image_height
        self.augment_seed = self.worker_config.augment_seed
        self.render_pedals_enabled = self.worker_config.render_pedals_enabled
        self.render_pedals_probability = self.worker_config.render_pedals_probability
        self.render_pedals_measures_probability = self.worker_config.render_pedals_measures_probability
        self.render_instrument_piano_enabled = self.worker_config.render_instrument_piano_enabled
        self.render_instrument_piano_probability = self.worker_config.render_instrument_piano_probability
        self.render_sforzando_enabled = self.worker_config.render_sforzando_enabled
        self.render_sforzando_probability = self.worker_config.render_sforzando_probability
        self.render_sforzando_per_note_probability = (
            self.worker_config.render_sforzando_per_note_probability
        )
        self.render_accent_enabled = self.worker_config.render_accent_enabled
        self.render_accent_probability = self.worker_config.render_accent_probability
        self.render_accent_per_note_probability = self.worker_config.render_accent_per_note_probability
        self.render_tempo_enabled = self.worker_config.render_tempo_enabled
        self.render_tempo_probability = self.worker_config.render_tempo_probability
        self.render_tempo_include_mm_probability = self.worker_config.render_tempo_include_mm_probability
        self.render_hairpins_enabled = self.worker_config.render_hairpins_enabled
        self.render_hairpins_probability = self.worker_config.render_hairpins_probability
        self.render_hairpins_max_spans = self.worker_config.render_hairpins_max_spans
        self.render_dynamic_marks_enabled = self.worker_config.render_dynamic_marks_enabled
        self.render_dynamic_marks_probability = self.worker_config.render_dynamic_marks_probability
        self.render_dynamic_marks_min_count = self.worker_config.render_dynamic_marks_min_count
        self.render_dynamic_marks_max_count = self.worker_config.render_dynamic_marks_max_count
        self.courtesy_naturals_probability = self.worker_config.courtesy_naturals_probability
        self.disable_offline_image_augmentations = (
            self.worker_config.disable_offline_image_augmentations
        )
        self.geom_x_squeeze_prob = self.worker_config.geom_x_squeeze_prob
        self.geom_x_squeeze_min_scale = self.worker_config.geom_x_squeeze_min_scale
        self.geom_x_squeeze_max_scale = self.worker_config.geom_x_squeeze_max_scale
        self.geom_x_squeeze_apply_in_conservative = (
            self.worker_config.geom_x_squeeze_apply_in_conservative
        )
        self.geom_x_squeeze_preview_force_scale = self.worker_config.geom_x_squeeze_preview_force_scale
        self.target_min_systems = self.worker_config.target_min_systems
        self.target_max_systems = self.worker_config.target_max_systems
        self.render_layout_profile = self.worker_config.render_layout_profile
        self.prefilter_min_non_empty_lines = (
            None
            if prefilter_min_non_empty_lines is None
            else int(prefilter_min_non_empty_lines)
        )
        self.prefilter_max_non_empty_lines = (
            None
            if prefilter_max_non_empty_lines is None
            else int(prefilter_max_non_empty_lines)
        )
        self.prefilter_min_measure_count = (
            None if prefilter_min_measure_count is None else int(prefilter_min_measure_count)
        )
        self.prefilter_max_measure_count = (
            None if prefilter_max_measure_count is None else int(prefilter_max_measure_count)
        )
        self.progress_enabled = coerce_boolish(progress_enabled)
        self.progress_update_interval_seconds = int(progress_update_interval_seconds)
        self.progress_path = Path(progress_path) if progress_path is not None else None
        self.progress_run_id = progress_run_id or "unknown"
        self.progress_output_dir = progress_output_dir or ""
        self.progress_run_artifacts_dir = progress_run_artifacts_dir or ""
        self.target_accepted_samples = (
            None if target_accepted_samples is None else int(target_accepted_samples)
        )
        self.max_scheduled_tasks = (
            None if max_scheduled_tasks is None else int(max_scheduled_tasks)
        )
        self.overflow_truncation_enabled = self.worker_config.overflow_truncation_enabled
        self.overflow_truncation_max_trials = self.worker_config.overflow_truncation_max_trials
        self.profile_enabled = self.worker_config.profile_enabled
        self.profile_log_every = max(1, profile_log_every)
        self.profile_capture_per_sample = profile_capture_per_sample
        self.resume_state = resume_state
        self.resume_observer = resume_observer
        self.deterministic_seed_salt = deterministic_seed_salt
        self.profile_report: dict[str, Any] | None = None
        self.task_timeout_seconds = int(task_timeout_seconds)
        self.max_task_retries_timeout = int(max_task_retries_timeout)
        self.max_task_retries_expired = int(max_task_retries_expired)
        if self.task_timeout_seconds <= 0:
            raise ValueError("task_timeout_seconds must be > 0")
        if self.max_task_retries_timeout < 0:
            raise ValueError("max_task_retries_timeout must be >= 0")
        if self.max_task_retries_expired < 0:
            raise ValueError("max_task_retries_expired must be >= 0")
        if self.prefilter_min_non_empty_lines is not None and self.prefilter_min_non_empty_lines < 1:
            raise ValueError("prefilter_min_non_empty_lines must be >= 1 when provided")
        if self.prefilter_max_non_empty_lines is not None and self.prefilter_max_non_empty_lines < 1:
            raise ValueError("prefilter_max_non_empty_lines must be >= 1 when provided")
        if self.prefilter_min_measure_count is not None and self.prefilter_min_measure_count < 1:
            raise ValueError("prefilter_min_measure_count must be >= 1 when provided")
        if self.prefilter_max_measure_count is not None and self.prefilter_max_measure_count < 1:
            raise ValueError("prefilter_max_measure_count must be >= 1 when provided")
        if (
            self.prefilter_min_non_empty_lines is not None
            and self.prefilter_max_non_empty_lines is not None
            and self.prefilter_min_non_empty_lines > self.prefilter_max_non_empty_lines
        ):
            raise ValueError(
                "prefilter_min_non_empty_lines must be <= prefilter_max_non_empty_lines"
            )
        if (
            self.prefilter_min_measure_count is not None
            and self.prefilter_max_measure_count is not None
            and self.prefilter_min_measure_count > self.prefilter_max_measure_count
        ):
            raise ValueError(
                "prefilter_min_measure_count must be <= prefilter_max_measure_count"
            )
        if self.progress_update_interval_seconds < 1:
            raise ValueError("progress_update_interval_seconds must be >= 1")
        if self.target_accepted_samples is not None and self.target_accepted_samples < 1:
            raise ValueError("target_accepted_samples must be >= 1 when provided")
        if self.max_scheduled_tasks is not None and self.max_scheduled_tasks < 1:
            raise ValueError("max_scheduled_tasks must be >= 1 when provided")
        self.initial_quarantined_files = {
            str(Path(path))
            for path in (initial_quarantined_files or [])
        }
        self.last_quarantine_summary: dict[str, Any] = {
            "files": sorted(self.initial_quarantined_files),
            "files_count": len(self.initial_quarantined_files),
            "skipped_tasks": 0,
            "dropped_pending_tasks": 0,
            "preloaded_files_count": len(self.initial_quarantined_files),
        }
        self.last_failure_summary: dict[str, Any] = {
            "requested_tasks": 0,
            "submitted_tasks": 0,
            "successful_samples": 0,
            "failed_samples_total": 0,
            "skipped_samples_total": 0,
            "accepted_target": self.target_accepted_samples,
            "max_scheduled_tasks": self.max_scheduled_tasks,
            "stop_condition": "not_started",
            "truncation": {
                "attempted": 0,
                "rescued": 0,
                "failed": 0,
            },
            "failure_reason_counts": {
                "multi_page": 0,
                "invalid_kern": 0,
                "sparse_render": 0,
                "render_fit": 0,
                "render_rejected": 0,
                "system_band_below_min": 0,
                "system_band_above_max": 0,
                "system_band_truncation_exhausted": 0,
                "system_band_rejected": 0,
                "processing_error": 0,
                "unknown_result": 0,
                "timeout": 0,
                "process_expired": 0,
            },
            "quarantine": {
                "skipped_tasks": 0,
                "dropped_pending_tasks": 0,
                "skipped_tasks_by_reason": {},
                "dropped_pending_tasks_by_reason": {},
            },
        }
        self.prefilter_summary: dict[str, Any] = {
            "min_non_empty_lines": self.prefilter_min_non_empty_lines,
            "max_non_empty_lines": self.prefilter_max_non_empty_lines,
            "min_measure_count": self.prefilter_min_measure_count,
            "max_measure_count": self.prefilter_max_measure_count,
            "original_file_count": 0,
            "retained_file_count": 0,
            "filtered_out_file_count": 0,
        }
        self.last_layout_summary: dict[str, Any] = _build_layout_summary(
            accepted_samples_total=0,
            systems_histogram=Counter(),
            unknown_system_count_successes=0,
            target_min_systems=self.target_min_systems,
            target_max_systems=self.target_max_systems,
            bottom_whitespace_ratios=[],
            vertical_fill_ratios=[],
        )
        self.last_scheduler_summary: dict[str, Any] = {
            "version": 1,
            "stop_condition": "not_started",
            "accepted_target": self.target_accepted_samples,
            "max_scheduled_tasks": self.max_scheduled_tasks,
            "total_available_tasks": int(getattr(self, "total_available_tasks", 0)),
            "submitted_tasks": 0,
            "terminal_completed_tasks": 0,
            "successful_samples": 0,
            "file_counts": {
                "total": 0,
                "schedulable_remaining": 0,
                "with_at_least_one_success": 0,
                "early_stopped": 0,
                "quarantined": 0,
            },
            "per_file_attempts": _summarize_ratio_series([]),
            "per_file_successes": _summarize_ratio_series([]),
            "top_high_yield_files": [],
            "top_low_yield_files": [],
            "timing_by_outcome": {},
            "worker_stage_timing_by_outcome": {},
        }
        self.last_run_result: GeneratorRunResult | None = None
        self._run_failure_reason_counts: Counter[str] = Counter()
        self._run_truncation_counts: Counter[str] = Counter()
        self._accepted_system_count_histogram: Counter[int] = Counter()
        self._accepted_unknown_system_count_successes = 0
        self._accepted_bottom_whitespace_ratios: list[float] = []
        self._accepted_vertical_fill_ratios: list[float] = []
        self._progress_started_at: float | None = None
        self._last_progress_update_at: float | None = None
        self._progress_target_samples = 0
        self._progress_max_scheduled_tasks: int | None = None
        self._progress_submitted_tasks = 0
        self._progress_terminal_completed_tasks = 0
        self._progress_in_flight_tasks = 0
        self._progress_stop_condition: str | None = None
        self.stats = GenerationStats()

        if self.variants_per_file < 1:
            raise ValueError("variants_per_file must be >= 1")
        self.worker_config.validate()

        # Collect files from all directories.
        file_paths: list[Path] = []
        source_root_by_file: dict[Path, Path] = {}
        for kern_dir in self.kern_dirs:
            for file_path in kern_dir.glob("*.krn"):
                file_paths.append(file_path)
                source_root_by_file[file_path] = kern_dir
        file_paths = sorted(file_paths)
        original_file_count = len(file_paths)
        self._source_stats_by_file: dict[Path, KernSourceStats] = {}
        if (
            self.prefilter_min_non_empty_lines is not None
            or self.prefilter_max_non_empty_lines is not None
            or self.prefilter_min_measure_count is not None
            or self.prefilter_max_measure_count is not None
        ):
            retained_paths: list[Path] = []
            for file_path in file_paths:
                source_stats = compute_kern_source_stats(file_path)
                self._source_stats_by_file[file_path] = source_stats
                if (
                    self.prefilter_min_non_empty_lines is not None
                    and source_stats.non_empty_line_count < self.prefilter_min_non_empty_lines
                ):
                    continue
                if (
                    self.prefilter_max_non_empty_lines is not None
                    and source_stats.non_empty_line_count > self.prefilter_max_non_empty_lines
                ):
                    continue
                if (
                    self.prefilter_min_measure_count is not None
                    and source_stats.measure_count < self.prefilter_min_measure_count
                ):
                    continue
                if (
                    self.prefilter_max_measure_count is not None
                    and source_stats.measure_count > self.prefilter_max_measure_count
                ):
                    continue
                retained_paths.append(file_path)
            file_paths = retained_paths
        self.file_paths = file_paths
        self._source_root_by_file = source_root_by_file
        self.prefilter_summary = {
            "min_non_empty_lines": self.prefilter_min_non_empty_lines,
            "max_non_empty_lines": self.prefilter_max_non_empty_lines,
            "min_measure_count": self.prefilter_min_measure_count,
            "max_measure_count": self.prefilter_max_measure_count,
            "original_file_count": original_file_count,
            "retained_file_count": len(self.file_paths),
            "filtered_out_file_count": original_file_count - len(self.file_paths),
        }

        if not self.file_paths:
            raise ValueError(f"No .krn files found in {kern_dirs}")

        self.variant_count_by_file, self.variant_policy_summary = build_fixed_variant_plan(
            self.file_paths,
            FIXED_VARIANTS_PER_FILE,
        )
        self.total_available_tasks = int(sum(self.variant_count_by_file.values()))

    def _record_failure_from_code(self, code: WorkerFailureReason, detail: str | None = None) -> None:
        record_worker_failure(
            stats=self.stats,
            reason_counts=self._run_failure_reason_counts,
            reason=code,
            reason_detail=detail,
        )

    def _record_failure_from_legacy_message(self, message: str) -> None:
        reason = legacy_message_to_failure_reason(message)
        detail: str | None = None
        if message.startswith("Reject:system_band_truncation_exhausted:"):
            detail = message.split("Reject:system_band_truncation_exhausted:", 1)[1]
        self._record_failure_from_code(reason, detail)

    def _record_profile_outcome(
        self,
        result: object,
        profile_data: dict[str, Any],
    ) -> None:
        outcome_counts: Counter[str] = profile_data["outcome_counts"]
        failure_code_counts: Counter[str] = profile_data["failure_code_counts"]
        worker_stage_values: dict[str, list[float]] = profile_data["worker_stage_values"]
        per_sample: list[dict[str, Any]] | None = profile_data["per_sample"]

        worker_profile = None
        failure_code: str | None = None
        if isinstance(result, SampleSuccess):
            outcome_counts["success"] += 1
            worker_profile = result.profile
        elif isinstance(result, SampleFailure):
            outcome_counts["failure"] += 1
            failure_code = result.code
            failure_code_counts[result.code] += 1
            worker_profile = result.profile
        elif isinstance(result, tuple) and len(result) == 3 and isinstance(result[1], str):
            img, trans_or_error, source = result
            if img is not None:
                outcome_counts["success"] += 1
            else:
                outcome_counts["failure"] += 1
                failure_code = legacy_message_to_failure_reason(trans_or_error)
                failure_code_counts[failure_code] += 1
                if per_sample is not None:
                    per_sample.append(
                        {
                            "source": source,
                            "failure_code": failure_code,
                            "failure_stage": None,
                        }
                    )
            return
        else:
            outcome_counts["failure"] += 1
            failure_code_counts["unknown_result"] += 1
            return

        if worker_profile is None:
            return
        for stage_name in PROFILE_STAGE_NAMES:
            stage_value = float(worker_profile.stages_ms.get(stage_name, 0.0))
            worker_stage_values[stage_name].append(stage_value)
        if per_sample is not None:
            per_sample.append(
                {
                    "source": getattr(result, "filename", None),
                    "failure_code": failure_code,
                    "failure_stage": worker_profile.failure_stage,
                    "stages_ms": dict(worker_profile.stages_ms),
                }
            )

    def _build_sample_record(
        self,
        *,
        file_path: Path,
        variant_idx: int,
        image: bytes,
        transcription: str,
        actual_system_count: int | None,
        truncation_applied: bool,
        truncation_ratio: float | None,
        bottom_whitespace_ratio: float | None,
        vertical_fill_ratio: float | None,
    ) -> dict[str, Any]:
        source_root = self._source_root_by_file[file_path]
        relative_source = file_path.relative_to(source_root)
        source_stats = self._source_stats_by_file.get(file_path)
        if source_stats is None:
            source_stats = compute_kern_source_stats(file_path)
            self._source_stats_by_file[file_path] = source_stats
        return {
            "image": image,
            "transcription": transcription,
            "source": str(relative_source),
            "source_dataset": source_root.parent.name,
            "source_split": "train",
            "sample_id": _sample_id_from_relative_source(relative_source, variant_idx),
            "curation_stage": "synthetic",
            "source_domain": "synth",
            "actual_system_count": actual_system_count,
            "truncation_applied": truncation_applied,
            "truncation_ratio": truncation_ratio,
            "render_layout_profile": self.render_layout_profile,
            "bottom_whitespace_ratio": bottom_whitespace_ratio,
            "vertical_fill_ratio": vertical_fill_ratio,
            "source_non_empty_line_count": source_stats.non_empty_line_count,
            "source_measure_count": source_stats.measure_count,
        }

    def _build_sample_id(self, *, file_path: Path, variant_idx: int) -> str:
        source_root = self._source_root_by_file[file_path]
        relative_source = file_path.relative_to(source_root)
        return _sample_id_from_relative_source(relative_source, variant_idx)

    def _apply_resume_state(self) -> tuple[int, set[str], dict[str, str]]:
        resumed_submitted = 0
        quarantined_files: set[str] = set(self.initial_quarantined_files)
        quarantine_reason_by_file: dict[str, str] = {
            file_path: "preloaded" for file_path in self.initial_quarantined_files
        }
        resume_state = self.resume_state
        if resume_state is None:
            return resumed_submitted, quarantined_files, quarantine_reason_by_file

        self._run_failure_reason_counts.update(resume_state.failure_reason_counts)
        self._run_truncation_counts.update(resume_state.truncation_counts)
        self._accepted_system_count_histogram.update(
            {int(key): int(value) for key, value in resume_state.accepted_system_histogram.items()}
        )
        self._accepted_unknown_system_count_successes = int(
            resume_state.accepted_unknown_system_count_successes
        )
        self._accepted_bottom_whitespace_ratios.extend(
            float(value) for value in resume_state.accepted_bottom_whitespace_ratios
        )
        self._accepted_vertical_fill_ratios.extend(
            float(value) for value in resume_state.accepted_vertical_fill_ratios
        )
        self.stats.successful = int(resume_state.successful_samples)
        self.stats.overflows = int(resume_state.failure_reason_counts.get("multi_page", 0))
        self.stats.invalid = int(resume_state.failure_reason_counts.get("invalid_kern", 0))
        self.stats.rejected_sparse = int(resume_state.failure_reason_counts.get("sparse_render", 0))
        self.stats.rejected_render_fit = int(resume_state.failure_reason_counts.get("render_fit", 0))
        self.stats.errors = (
            int(resume_state.failure_reason_counts.get("processing_error", 0))
            + int(resume_state.failure_reason_counts.get("unknown_result", 0))
        )
        self.stats.timeouts = int(resume_state.failure_reason_counts.get("timeout", 0))
        self.stats.expired_workers = int(resume_state.failure_reason_counts.get("process_expired", 0))

        for file_key, persisted in resume_state.file_states.items():
            file_path = Path(file_key)
            state = self._file_states.get(file_path)
            if state is None:
                continue
            state.next_variant_idx = min(
                int(persisted.get("next_variant_idx", 0)),
                int(state.total_variants),
            )
            state.attempted = int(persisted.get("attempted", 0))
            state.successful = int(persisted.get("successful", 0))
            state.failed = int(persisted.get("failed", 0))
            state.quarantined = bool(persisted.get("quarantined", False))
            state.early_stopped = bool(persisted.get("early_stopped", False))
            resumed_submitted += state.attempted
            if state.quarantined:
                file_key_str = str(file_path)
                quarantined_files.add(file_key_str)
                quarantine_reason_by_file.setdefault(file_key_str, "resumed")

        return resumed_submitted, quarantined_files, quarantine_reason_by_file

    def _consume_worker_result(
        self,
        result: object,
        *,
        file_path: Path,
        variant_idx: int,
    ) -> dict[str, Any] | None:
        if isinstance(result, SampleSuccess):
            if not _system_count_in_target_band(
                result.actual_system_count,
                target_min_systems=self.target_min_systems,
                target_max_systems=self.target_max_systems,
            ):
                if result.truncation_applied:
                    self._run_truncation_counts["attempted"] += 1
                    self._run_truncation_counts["failed"] += 1
                self._record_failure_from_code(
                    _system_band_failure_code(
                        result.actual_system_count,
                        target_min_systems=self.target_min_systems,
                        target_max_systems=self.target_max_systems,
                        truncation_attempted=result.truncation_applied,
                    ),
                )
                return None
            self.stats.successful += 1
            if result.truncation_applied:
                self._run_truncation_counts["attempted"] += 1
                self._run_truncation_counts["rescued"] += 1
            if result.actual_system_count is None:
                self._accepted_unknown_system_count_successes += 1
            else:
                self._accepted_system_count_histogram[int(result.actual_system_count)] += 1
            if result.bottom_whitespace_ratio is not None:
                self._accepted_bottom_whitespace_ratios.append(float(result.bottom_whitespace_ratio))
            if result.vertical_fill_ratio is not None:
                self._accepted_vertical_fill_ratios.append(float(result.vertical_fill_ratio))
            return self._build_sample_record(
                file_path=file_path,
                variant_idx=variant_idx,
                image=result.image,
                transcription=result.transcription,
                actual_system_count=result.actual_system_count,
                truncation_applied=result.truncation_applied,
                truncation_ratio=result.truncation_ratio,
                bottom_whitespace_ratio=result.bottom_whitespace_ratio,
                vertical_fill_ratio=result.vertical_fill_ratio,
            )

        if isinstance(result, SampleFailure):
            if result.truncation_attempted:
                self._run_truncation_counts["attempted"] += 1
                self._run_truncation_counts["failed"] += 1
            self._record_failure_from_code(result.code, result.detail)
            return None

        if (
            isinstance(result, tuple)
            and len(result) == 3
            and isinstance(result[1], str)
        ):
            img, trans_or_error, _source = result
            if img is not None:
                self.stats.successful += 1
                self._accepted_unknown_system_count_successes += 1
                return self._build_sample_record(
                    file_path=file_path,
                    variant_idx=variant_idx,
                    image=img,
                    transcription=trans_or_error,
                    actual_system_count=None,
                    truncation_applied=False,
                    truncation_ratio=None,
                    bottom_whitespace_ratio=None,
                    vertical_fill_ratio=None,
                )
            self._record_failure_from_legacy_message(trans_or_error)
            return None

        self.stats.errors += 1
        self._run_failure_reason_counts["unknown_result"] += 1
        return None

    def _current_progress_snapshot(self, *, status: str) -> dict[str, Any]:
        started_at = self._progress_started_at or time.time()
        last_update_at = time.time()
        remaining_schedulable_files = 0
        files_early_stopped = 0
        files_with_at_least_one_success = 0
        file_states: dict[Path, _FileRunState] = getattr(self, "_file_states", {})
        if file_states:
            files_early_stopped = int(sum(1 for state in file_states.values() if state.early_stopped))
            files_with_at_least_one_success = int(
                sum(1 for state in file_states.values() if state.successful > 0)
            )
            remaining_schedulable_files = int(
                sum(
                    1
                    for state in file_states.values()
                    if state.remaining_variants > 0
                    and not state.in_flight
                    and not state.quarantined
                    and not state.early_stopped
                )
            )
        snapshot = build_generation_progress_snapshot(
            run_id=self.progress_run_id,
            status=status,
            started_at=started_at,
            last_update_at=last_update_at,
            target_samples=self._progress_target_samples,
            accepted_target=self.target_accepted_samples,
            max_scheduled_tasks=self._progress_max_scheduled_tasks,
            accepted_samples=self.stats.successful,
            submitted_tasks=self._progress_submitted_tasks,
            terminal_completed_tasks=self._progress_terminal_completed_tasks,
            in_flight_tasks=self._progress_in_flight_tasks,
            stop_condition=self._progress_stop_condition,
            remaining_schedulable_files=remaining_schedulable_files,
            files_early_stopped=files_early_stopped,
            files_with_at_least_one_success=files_with_at_least_one_success,
            truncation_counts=dict(self._run_truncation_counts),
            failure_reason_counts=build_failure_reason_counts(self._run_failure_reason_counts),
            output_dir=self.progress_output_dir,
            run_artifacts_dir=self.progress_run_artifacts_dir,
            progress_path=str(self.progress_path) if self.progress_path is not None else "",
        )
        self._last_progress_update_at = last_update_at
        return snapshot

    def _emit_progress_snapshot(self, *, status: str, force: bool = False) -> None:
        if not self.progress_enabled:
            return
        now = time.time()
        if (
            not force
            and self._last_progress_update_at is not None
            and (now - self._last_progress_update_at) < self.progress_update_interval_seconds
        ):
            return
        snapshot = self._current_progress_snapshot(status=status)
        print(format_generation_progress_line(snapshot))
        if self.progress_path is not None:
            write_progress_snapshot_atomic(self.progress_path, snapshot)

    def emit_external_progress(self, status: str) -> None:
        self._emit_progress_snapshot(status=status, force=True)

    def _resolve_stop_limits(self, num_samples: int) -> tuple[int | None, int]:
        accepted_target = self.target_accepted_samples
        max_scheduled_tasks = self.max_scheduled_tasks
        if accepted_target is None and max_scheduled_tasks is None:
            max_scheduled_tasks = int(num_samples)
        elif max_scheduled_tasks is None and accepted_target is not None:
            max_scheduled_tasks = self.total_available_tasks
        assert max_scheduled_tasks is not None
        return accepted_target, min(int(max_scheduled_tasks), self.total_available_tasks)

    def _select_next_file_state(self) -> _FileRunState | None:
        file_states: dict[Path, _FileRunState] = getattr(self, "_file_states", {})
        bootstrap_candidates = [
            state
            for state in file_states.values()
            if state.remaining_variants > 0
            and not state.in_flight
            and not state.quarantined
            and not state.early_stopped
            and not state.has_bootstrapped
        ]
        if bootstrap_candidates:
            return min(bootstrap_candidates, key=lambda state: str(state.path))

        ranked_candidates = [
            state
            for state in file_states.values()
            if state.remaining_variants > 0
            and not state.in_flight
            and not state.quarantined
            and not state.early_stopped
        ]
        if not ranked_candidates:
            return None

        def _rank_key(state: _FileRunState) -> tuple[int, float, int, str]:
            if state.successful > 0:
                priority_group = 0
            elif state.attempted >= 3:
                priority_group = 2
            else:
                priority_group = 1
            return (
                priority_group,
                -state.smoothed_success_rate,
                state.attempted,
                str(state.path),
            )

        return min(ranked_candidates, key=_rank_key)

    def generate(
        self,
        num_samples: int,
        target_accepted_samples: int | None = None,
        max_scheduled_tasks: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Generate samples from kern files.

        Args:
            num_samples: Legacy task-budget parameter used when accepted-target
                mode is disabled.
            target_accepted_samples: Optional accepted-sample stopping target.
            max_scheduled_tasks: Optional cap on scheduled tasks.

        Yields:
            Dictionaries with 'image' (bytes) and 'transcription' (str) keys.
        """
        # Reset stats for this run
        self.stats.successful = 0
        self.stats.overflows = 0
        self.stats.errors = 0
        self.stats.timeouts = 0
        self.stats.expired_workers = 0
        self.stats.invalid = 0
        self.stats.rejected_sparse = 0
        self.stats.rejected_render_fit = 0
        self.profile_report = None
        self.last_run_result = None
        self._run_failure_reason_counts.clear()
        self._run_truncation_counts.clear()
        self._accepted_system_count_histogram.clear()
        self._accepted_unknown_system_count_successes = 0
        self._accepted_bottom_whitespace_ratios.clear()
        self._accepted_vertical_fill_ratios.clear()
        if target_accepted_samples is not None:
            self.target_accepted_samples = int(target_accepted_samples)
        if max_scheduled_tasks is not None:
            self.max_scheduled_tasks = int(max_scheduled_tasks)

        accepted_target, total_tasks = self._resolve_stop_limits(num_samples)
        self._file_states = {
            file_path: _FileRunState(
                path=file_path,
                total_variants=int(self.variant_count_by_file[file_path]),
            )
            for file_path in self.file_paths
        }
        resumed_submitted, quarantined_files, quarantine_reason_by_file = (
            self._apply_resume_state()
        )
        self._progress_started_at = time.time()
        self._last_progress_update_at = None
        self._progress_target_samples = (
            int(accepted_target) if accepted_target is not None else int(total_tasks)
        )
        self._progress_max_scheduled_tasks = int(total_tasks)
        self._progress_submitted_tasks = int(resumed_submitted)
        self._progress_terminal_completed_tasks = int(resumed_submitted)
        self._progress_in_flight_tasks = 0
        self._progress_stop_condition = None

        def task_iter():
            for file_path in self.file_paths:
                for variant_idx in range(self.variant_count_by_file[file_path]):
                    yield file_path, variant_idx

        counters = {
            "timeout_terminal": 0,
            "expired_terminal": 0,
            "timeout_retry": 0,
            "expired_retry": 0,
        }
        quarantine_skipped_by_reason: Counter[str] = Counter()
        quarantine_dropped_pending_by_reason: Counter[str] = Counter()
        quarantine_skipped_tasks = 0
        quarantine_dropped_pending_total = 0
        max_in_flight = max(1, self.num_workers * MAX_IN_FLIGHT_MULTIPLIER)
        profile_data: dict[str, Any] | None = None
        if self.profile_enabled:
            profile_data = {
                "worker_stage_values": {stage_name: [] for stage_name in PROFILE_STAGE_NAMES},
                "queue_wait_ms": [],
                "future_block_ms": [],
                "future_total_ms": [],
                "result_handling_ms": [],
                "in_flight_depth": [],
                "scheduler_idle_ms": [],
                "outcome_counts": Counter(),
                "failure_code_counts": Counter(),
                "retry_counts": Counter(),
                "timeout_events": [],
                "process_expired_events": [],
                "quarantine_events": [],
                "per_sample": [] if self.profile_capture_per_sample else None,
            }
        scheduler_timing_by_outcome: dict[str, dict[str, _RunningSeriesSummary]] = {}
        scheduler_worker_stage_by_outcome: dict[str, dict[str, _RunningSeriesSummary]] = {}

        def _record_scheduler_metric(outcome: str, metric: str, value_ms: float) -> None:
            scheduler_timing_by_outcome.setdefault(outcome, {})
            scheduler_timing_by_outcome[outcome].setdefault(metric, _RunningSeriesSummary())
            scheduler_timing_by_outcome[outcome][metric].add(value_ms)

        def _record_scheduler_worker_stages(outcome: str, result: object) -> None:
            worker_profile = getattr(result, "profile", None)
            if worker_profile is None:
                return
            scheduler_worker_stage_by_outcome.setdefault(outcome, {})
            for stage_name in PROFILE_STAGE_NAMES:
                scheduler_worker_stage_by_outcome[outcome].setdefault(
                    stage_name,
                    _RunningSeriesSummary(),
                )
                scheduler_worker_stage_by_outcome[outcome][stage_name].add(
                    float(worker_profile.stages_ms.get(stage_name, 0.0))
                )

        self._emit_progress_snapshot(status="running", force=True)
        pending: list[tuple[Any, int, Path, int, int]] = []
        submitted = int(resumed_submitted)
        terminal_completed = int(resumed_submitted)
        stop_condition = "candidate_pool_exhausted"

        try:
            with ProcessPool(
                max_workers=self.num_workers,
                max_tasks=200,
                initializer=init_file_worker,
                initargs=(self.worker_config,),
                ) as pool:
                del task_iter

                for file_path in quarantined_files:
                    state = self._file_states.get(Path(file_path))
                    if state is not None:
                        state.quarantined = True

                def _next_schedulable_task() -> tuple[Path, int] | None:
                    nonlocal quarantine_skipped_tasks
                    selected_state = self._select_next_file_state()
                    if selected_state is None:
                        return None
                    if str(selected_state.path) in quarantined_files:
                        quarantine_skipped_tasks += max(1, selected_state.remaining_variants)
                        quarantine_reason = quarantine_reason_by_file.get(
                            str(selected_state.path),
                            "unknown",
                        )
                        quarantine_skipped_by_reason[quarantine_reason] += max(
                            1,
                            selected_state.remaining_variants,
                        )
                        selected_state.quarantined = True
                        return _next_schedulable_task()
                    variant_idx = selected_state.next_variant_idx
                    selected_state.next_variant_idx += 1
                    selected_state.in_flight = True
                    return selected_state.path, variant_idx

                def _drop_pending_for_file(target_file_path: Path) -> int:
                    retained: list[tuple[Any, int, Path, int, int]] = []
                    dropped = 0
                    target_key = str(target_file_path)
                    for item in pending:
                        if str(item[2]) == target_key:
                            dropped += 1
                        else:
                            retained.append(item)
                    pending[:] = retained
                    target_state = self._file_states.get(target_file_path)
                    if target_state is not None:
                        target_state.in_flight = False
                    return dropped

                def _quarantine_file(
                    *,
                    file_path: Path,
                    reason: str,
                    trigger_event: str,
                    retry_count: int,
                    queue_wait_ms: float,
                    exitcode: int | None = None,
                ) -> None:
                    nonlocal quarantine_dropped_pending_total
                    file_key = str(file_path)
                    if file_key in quarantined_files:
                        return
                    quarantined_files.add(file_key)
                    quarantine_reason_by_file[file_key] = reason
                    target_state = self._file_states.get(file_path)
                    if target_state is not None:
                        target_state.quarantined = True
                        target_state.in_flight = False
                    dropped_pending_tasks = _drop_pending_for_file(file_path)
                    quarantine_dropped_pending_total += dropped_pending_tasks
                    quarantine_dropped_pending_by_reason[reason] += dropped_pending_tasks
                    if profile_data is not None:
                        profile_data["quarantine_events"].append(
                            _build_quarantine_event(
                                file_path=file_path,
                                reason=reason,
                                trigger_event=trigger_event,
                                retry_count=retry_count,
                                queue_wait_ms=queue_wait_ms,
                                dropped_pending_tasks=dropped_pending_tasks,
                                exitcode=exitcode,
                            )
                        )

                def _schedule_task(
                    *,
                    file_path: Path,
                    variant_idx: int,
                    retry_count: int,
                ) -> tuple[Any, int, Path, int, int]:
                    sample_id = self._build_sample_id(file_path=file_path, variant_idx=variant_idx)
                    return (
                        pool.schedule(
                            generate_sample_from_path_outcome,
                            args=(file_path, variant_idx, sample_id),
                            timeout=self.task_timeout_seconds,
                        ),
                        time.perf_counter_ns(),
                        file_path,
                        variant_idx,
                        retry_count,
                    )

                def _pop_next_resolved_task() -> tuple[
                    tuple[Any, int, Path, int, int],
                    int,
                    int,
                    object | None,
                    BaseException | None,
                ] | None:
                    """Return the next resolved task record, preferring already-ready futures."""
                    for task_record in pending:
                        future = task_record[0]
                        done_fn = getattr(future, "done", None)
                        if callable(done_fn):
                            with contextlib.suppress(Exception):
                                if not done_fn():
                                    continue
                        before_result_ns = time.perf_counter_ns()
                        try:
                            result = future.result()
                            after_result_ns = time.perf_counter_ns()
                            return task_record, before_result_ns, after_result_ns, result, None
                        except TimeoutError:
                            after_result_ns = time.perf_counter_ns()
                            return task_record, before_result_ns, after_result_ns, None, TimeoutError()
                        except ProcessExpired as exc:
                            after_result_ns = time.perf_counter_ns()
                            return task_record, before_result_ns, after_result_ns, None, exc

                    task_record = pending[0]
                    future = task_record[0]
                    before_result_ns = time.perf_counter_ns()
                    try:
                        result = _future_result_with_timeout(future, 0.05)
                        after_result_ns = time.perf_counter_ns()
                        return task_record, before_result_ns, after_result_ns, result, None
                    except TimeoutError:
                        return None
                    except ProcessExpired as exc:
                        after_result_ns = time.perf_counter_ns()
                        return task_record, before_result_ns, after_result_ns, None, exc

                for _ in range(min(max_in_flight, max(0, total_tasks - submitted))):
                    scheduled_task = _next_schedulable_task()
                    if scheduled_task is None:
                        break
                    file_path, variant_idx = scheduled_task
                    pending.append(
                        _schedule_task(
                            file_path=file_path,
                            variant_idx=variant_idx,
                            retry_count=0,
                        )
                    )

                submitted += len(pending)
                self._progress_submitted_tasks = submitted
                self._progress_in_flight_tasks = len(pending)
                self._emit_progress_snapshot(status="running")

                while pending or submitted < total_tasks:
                    if accepted_target is not None and self.stats.successful >= accepted_target:
                        stop_condition = "accepted_target_reached"
                        break
                    if not pending:
                        idle_start_ns = time.perf_counter_ns()
                        next_task = _next_schedulable_task()
                        if next_task is None:
                            break
                        next_file_path, next_variant_idx = next_task
                        pending.append(
                            _schedule_task(
                                file_path=next_file_path,
                                variant_idx=next_variant_idx,
                                retry_count=0,
                            )
                        )
                        submitted += 1
                        self._progress_submitted_tasks = submitted
                        self._progress_in_flight_tasks = len(pending)
                        if profile_data is not None:
                            profile_data["scheduler_idle_ms"].append(
                                (time.perf_counter_ns() - idle_start_ns) / 1_000_000.0
                            )
                        self._emit_progress_snapshot(status="running")
                        continue

                    in_flight_depth = len(pending)
                    poll_start_ns = time.perf_counter_ns()
                    resolved = _pop_next_resolved_task()
                    if resolved is None:
                        if profile_data is not None:
                            profile_data["scheduler_idle_ms"].append(
                                (time.perf_counter_ns() - poll_start_ns) / 1_000_000.0
                            )
                        self._progress_submitted_tasks = submitted
                        self._progress_terminal_completed_tasks = terminal_completed
                        self._progress_in_flight_tasks = len(pending)
                        self._emit_progress_snapshot(status="running")
                        continue
                    task_record, before_result_ns, after_result_ns, resolved_result, resolved_exc = resolved
                    pending.remove(task_record)
                    _future, scheduled_ns, file_path, variant_idx, retry_count = task_record
                    queue_wait_ms = (before_result_ns - scheduled_ns) / 1_000_000.0
                    should_resubmit = False
                    result = None
                    outcome_bucket = "unknown_result"
                    state = self._file_states[file_path]
                    state.in_flight = False

                    if resolved_exc is None:
                        result = resolved_result
                        outcome_bucket = _bucket_for_worker_result(result)
                    elif isinstance(resolved_exc, TimeoutError):
                        will_retry = retry_count < self.max_task_retries_timeout
                        outcome_bucket = "timeout"
                        if will_retry:
                            counters["timeout_retry"] += 1
                            should_resubmit = True
                            if profile_data is not None:
                                profile_data["retry_counts"]["timeout"] += 1
                        else:
                            counters["timeout_terminal"] += 1
                            self._run_failure_reason_counts["timeout"] += 1
                            if profile_data is not None:
                                profile_data["outcome_counts"]["timeout"] += 1
                        if profile_data is not None:
                            profile_data["timeout_events"].append(
                                _build_timeout_event(
                                    file_path=file_path,
                                    variant_idx=variant_idx,
                                    retry_count=retry_count,
                                    queue_wait_ms=queue_wait_ms,
                                    will_retry=will_retry,
                                )
                            )
                        if not will_retry:
                            _quarantine_file(
                                file_path=file_path,
                                reason="terminal_timeout",
                                trigger_event="timeout",
                                retry_count=retry_count,
                                queue_wait_ms=queue_wait_ms,
                            )
                    elif isinstance(resolved_exc, ProcessExpired):
                        exc = resolved_exc
                        will_retry = retry_count < self.max_task_retries_expired
                        outcome_bucket = "process_expired"
                        if will_retry:
                            counters["expired_retry"] += 1
                            should_resubmit = True
                            if profile_data is not None:
                                profile_data["retry_counts"]["expired"] += 1
                        else:
                            counters["expired_terminal"] += 1
                            self._run_failure_reason_counts["process_expired"] += 1
                            if profile_data is not None:
                                profile_data["outcome_counts"]["expired"] += 1
                        if profile_data is not None:
                            profile_data["process_expired_events"].append(
                                _build_process_expired_event(
                                    exc=exc,
                                    file_path=file_path,
                                    variant_idx=variant_idx,
                                    retry_count=retry_count,
                                    queue_wait_ms=queue_wait_ms,
                                    will_retry=will_retry,
                                )
                            )
                        if not will_retry:
                            _quarantine_file(
                                file_path=file_path,
                                reason="terminal_process_expired",
                                trigger_event="process_expired",
                                retry_count=retry_count,
                                queue_wait_ms=queue_wait_ms,
                                exitcode=getattr(exc, "exitcode", None),
                            )
                    else:
                        raise resolved_exc

                    future_block_ms = (after_result_ns - before_result_ns) / 1_000_000.0
                    future_total_ms = (after_result_ns - scheduled_ns) / 1_000_000.0
                    _record_scheduler_metric(outcome_bucket, "queue_wait_ms", queue_wait_ms)
                    _record_scheduler_metric(outcome_bucket, "future_block_ms", future_block_ms)
                    _record_scheduler_metric(outcome_bucket, "future_total_ms", future_total_ms)
                    _record_scheduler_metric(outcome_bucket, "in_flight_depth", float(in_flight_depth))
                    if profile_data is not None:
                        profile_data["in_flight_depth"].append(float(in_flight_depth))
                        profile_data["queue_wait_ms"].append(queue_wait_ms)
                        profile_data["future_block_ms"].append(future_block_ms)
                        profile_data["future_total_ms"].append(future_total_ms)

                    handle_start_ns = time.perf_counter_ns()
                    if should_resubmit:
                        state.in_flight = True
                        pending.append(
                            _schedule_task(
                                file_path=file_path,
                                variant_idx=variant_idx,
                                retry_count=retry_count + 1,
                            )
                        )
                    elif result is not None:
                        if profile_data is not None:
                            self._record_profile_outcome(result, profile_data)
                        _record_scheduler_worker_stages(outcome_bucket, result)
                        sample = self._consume_worker_result(
                            result,
                            file_path=file_path,
                            variant_idx=variant_idx,
                        )
                        state.attempted += 1
                        if sample is not None:
                            state.successful += 1
                        else:
                            state.failed += 1
                        if self.resume_observer is not None:
                            self.resume_observer.observe_terminal_task(
                                generator=self,
                                file_path=file_path,
                                file_state=state,
                                sample=sample,
                            )
                        if sample is not None:
                            yield sample
                        terminal_completed += 1
                    else:
                        state.attempted += 1
                        state.failed += 1
                        if self.resume_observer is not None:
                            self.resume_observer.observe_terminal_task(
                                generator=self,
                                file_path=file_path,
                                file_state=state,
                                sample=None,
                            )
                        terminal_completed += 1
                    result_handling_ms = (time.perf_counter_ns() - handle_start_ns) / 1_000_000.0
                    _record_scheduler_metric(outcome_bucket, "result_handling_ms", result_handling_ms)
                    if profile_data is not None:
                        profile_data["result_handling_ms"].append(result_handling_ms)
                    if state.successful == 0 and state.attempted >= 8:
                        state.early_stopped = True

                    while submitted < total_tasks and len(pending) < max_in_flight:
                        if accepted_target is not None and self.stats.successful >= accepted_target:
                            stop_condition = "accepted_target_reached"
                            break
                        next_task = _next_schedulable_task()
                        if next_task is None:
                            break
                        next_file_path, next_variant_idx = next_task
                        pending.append(
                            _schedule_task(
                                file_path=next_file_path,
                                variant_idx=next_variant_idx,
                                retry_count=0,
                            )
                        )
                        submitted += 1

                    self._progress_submitted_tasks = submitted
                    self._progress_terminal_completed_tasks = terminal_completed
                    self._progress_in_flight_tasks = len(pending)
                    self._emit_progress_snapshot(status="running")

                    if (
                        profile_data is not None
                        and self.profile_log_every > 0
                        and terminal_completed > 0
                        and terminal_completed % self.profile_log_every == 0
                    ):
                        outcome_counts: Counter[str] = profile_data["outcome_counts"]
                        print(
                            "[profile] "
                            f"completed={terminal_completed}/{total_tasks} "
                            f"success={outcome_counts.get('success', 0)} "
                            f"failure={outcome_counts.get('failure', 0)} "
                            f"timeout={outcome_counts.get('timeout', 0)} "
                            f"expired={outcome_counts.get('expired', 0)}",
                        )
                else:
                    stop_condition = "task_budget_exhausted"
        except BaseException as exc:
            self._progress_submitted_tasks = submitted
            self._progress_terminal_completed_tasks = terminal_completed
            self._progress_in_flight_tasks = len(pending)
            self._progress_stop_condition = (
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            )
            self._emit_progress_snapshot(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                force=True,
            )
            raise

        for _future, _scheduled_ns, file_path, _variant_idx, _retry_count in pending:
            state = self._file_states.get(file_path)
            if state is not None:
                state.in_flight = False
        if accepted_target is not None and self.stats.successful >= accepted_target:
            stop_condition = "accepted_target_reached"
        elif submitted >= total_tasks:
            stop_condition = "task_budget_exhausted"
        else:
            stop_condition = "candidate_pool_exhausted"

        # Copy counters to stats
        self.stats.timeouts = counters["timeout_terminal"]
        self.stats.expired_workers = counters["expired_terminal"]
        quarantine_summary = {
            "files": sorted(quarantined_files),
            "files_count": len(quarantined_files),
            "skipped_tasks": quarantine_skipped_tasks,
            "dropped_pending_tasks": quarantine_dropped_pending_total,
            "preloaded_files_count": len(self.initial_quarantined_files),
        }
        self.last_quarantine_summary = quarantine_summary
        failure_reason_counts = build_failure_reason_counts(self._run_failure_reason_counts)
        failed_samples_total = int(sum(failure_reason_counts.values()))
        self.last_failure_summary = {
            "requested_tasks": int(total_tasks),
            "submitted_tasks": int(submitted),
            "successful_samples": int(self.stats.successful),
            "failed_samples_total": failed_samples_total,
            "skipped_samples_total": int(
                failed_samples_total
                + quarantine_skipped_tasks
                + quarantine_dropped_pending_total
            ),
            "accepted_target": accepted_target,
            "max_scheduled_tasks": int(total_tasks),
            "stop_condition": stop_condition,
            "truncation": {
                "attempted": int(self._run_truncation_counts.get("attempted", 0)),
                "rescued": int(self._run_truncation_counts.get("rescued", 0)),
                "failed": int(self._run_truncation_counts.get("failed", 0)),
            },
            "failure_reason_counts": failure_reason_counts,
            "quarantine": {
                "skipped_tasks": int(quarantine_skipped_tasks),
                "dropped_pending_tasks": int(quarantine_dropped_pending_total),
                "skipped_tasks_by_reason": {
                    reason: int(count)
                    for reason, count in sorted(quarantine_skipped_by_reason.items())
                },
                "dropped_pending_tasks_by_reason": {
                    reason: int(count)
                    for reason, count in sorted(quarantine_dropped_pending_by_reason.items())
                },
            },
        }
        self.last_layout_summary = _build_layout_summary(
            accepted_samples_total=self.stats.successful,
            systems_histogram=self._accepted_system_count_histogram,
            unknown_system_count_successes=self._accepted_unknown_system_count_successes,
            target_min_systems=self.target_min_systems,
            target_max_systems=self.target_max_systems,
            bottom_whitespace_ratios=self._accepted_bottom_whitespace_ratios,
            vertical_fill_ratios=self._accepted_vertical_fill_ratios,
        )
        self.last_scheduler_summary = _build_scheduler_summary(
            accepted_target=accepted_target,
            max_scheduled_tasks=int(total_tasks),
            stop_condition=stop_condition,
            file_states=self._file_states,
            timing_by_outcome=scheduler_timing_by_outcome,
            worker_stage_by_outcome=scheduler_worker_stage_by_outcome,
            total_available_tasks=self.total_available_tasks,
            submitted_tasks=submitted,
            terminal_completed_tasks=terminal_completed,
            successful_samples=self.stats.successful,
        )
        self._progress_submitted_tasks = submitted
        self._progress_terminal_completed_tasks = terminal_completed
        self._progress_in_flight_tasks = 0
        self._progress_stop_condition = stop_condition
        self._emit_progress_snapshot(status="completed", force=True)
        if profile_data is not None:
            worker_stage_values: dict[str, list[float]] = profile_data["worker_stage_values"]
            self.profile_report = {
                "total_tasks": total_tasks,
                "submitted_tasks": submitted if total_tasks > 0 else 0,
                "outcome_counts": dict(profile_data["outcome_counts"]),
                "failure_code_counts": dict(profile_data["failure_code_counts"]),
                "retry_counts": dict(profile_data["retry_counts"]),
                "timeout_events": list(profile_data["timeout_events"]),
                "process_expired_events": list(profile_data["process_expired_events"]),
                "quarantine_events": list(profile_data["quarantine_events"]),
                "quarantine": quarantine_summary,
                "worker_stage_stats": {
                    stage_name: _summarize_series(stage_values)
                    for stage_name, stage_values in worker_stage_values.items()
                },
                "orchestrator_stats": {
                    "queue_wait_ms": _summarize_series(profile_data["queue_wait_ms"]),
                    "future_block_ms": _summarize_series(profile_data["future_block_ms"]),
                    "future_total_ms": _summarize_series(profile_data["future_total_ms"]),
                    "result_handling_ms": _summarize_series(profile_data["result_handling_ms"]),
                    "in_flight_depth": _summarize_series(profile_data["in_flight_depth"]),
                    "scheduler_idle_ms": _summarize_series(profile_data["scheduler_idle_ms"]),
                },
                "per_sample": profile_data["per_sample"],
            }
        self.last_run_result = GeneratorRunResult(
            stats={
                "successful": int(self.stats.successful),
                "overflows": int(self.stats.overflows),
                "errors": int(self.stats.errors),
                "timeouts": int(self.stats.timeouts),
                "expired_workers": int(self.stats.expired_workers),
                "invalid": int(self.stats.invalid),
                "rejected_sparse": int(self.stats.rejected_sparse),
                "rejected_render_fit": int(self.stats.rejected_render_fit),
            },
            quarantine_summary=dict(self.last_quarantine_summary),
            failure_summary=dict(self.last_failure_summary),
            layout_summary=dict(self.last_layout_summary),
            scheduler_summary=dict(self.last_scheduler_summary),
            profile_report=self.profile_report,
        )
