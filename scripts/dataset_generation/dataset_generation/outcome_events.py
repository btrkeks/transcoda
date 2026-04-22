"""Outcome event and summary helpers for dataset generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from scripts.dataset_generation.dataset_generation.io import append_jsonl
from scripts.dataset_generation.dataset_generation.run_context import RunContext
from scripts.dataset_generation.dataset_generation.types import (
    AugmentationTraceEvent,
    FailureTraceEvent,
    SamplePlan,
    SuccessTraceEvent,
    WorkerFailure,
    WorkerSuccess,
)


class TaskTraceLike(Protocol):
    sample_idx: int
    plan: SamplePlan
    target_bucket: int | None
    planned_line_count: int | None
    candidate_in_target_range: bool | None


@dataclass(frozen=True)
class TaskTraceContext:
    sample_idx: int
    source_paths: tuple[str, ...]
    target_bucket: int | None
    planned_line_count: int | None
    candidate_in_target_range: bool | None


def write_outcome_events(
    *,
    run_context: RunContext,
    outcome: WorkerSuccess | WorkerFailure,
    task: TaskTraceLike | None,
    committed_to_dataset: bool,
) -> None:
    if outcome.verovio_diagnostics:
        append_jsonl(
            run_context.verovio_events_path,
            [asdict(event) for event in outcome.verovio_diagnostics],
        )
    if isinstance(outcome, WorkerFailure):
        append_jsonl(
            run_context.failure_events_path,
            [asdict(build_failure_trace_event(outcome=outcome, task=task))],
        )
        return
    append_jsonl(
        run_context.success_events_path,
        [
            asdict(
                build_success_trace_event(
                    outcome=outcome,
                    task=task,
                    committed_to_dataset=committed_to_dataset,
                )
            )
        ],
    )
    if outcome.augmentation_trace is not None:
        append_jsonl(
            run_context.augmentation_events_path,
            [asdict(outcome.augmentation_trace)],
        )


def build_failure_trace_event(
    *,
    outcome: WorkerFailure,
    task: TaskTraceLike | None,
) -> FailureTraceEvent:
    trace_context = _task_trace_context(task=task, sample_id=outcome.sample_id)
    return FailureTraceEvent(
        event="failure_trace",
        sample_id=outcome.sample_id,
        sample_idx=trace_context.sample_idx,
        source_paths=trace_context.source_paths,
        target_bucket=trace_context.target_bucket,
        planned_line_count=trace_context.planned_line_count,
        candidate_in_target_range=trace_context.candidate_in_target_range,
        failure_reason=outcome.failure_reason,
        truncation_mode=outcome.truncation_mode,
        truncation_attempted=outcome.truncation_attempted,
        preferred_5_6_rescue_attempted=outcome.preferred_5_6_rescue_attempted,
        preferred_5_6_rescue_succeeded=outcome.preferred_5_6_rescue_succeeded,
        preferred_5_6_status=outcome.preferred_5_6_status,
        attempts=outcome.failure_attempts,
    )


def build_success_trace_event(
    *,
    outcome: WorkerSuccess,
    task: TaskTraceLike | None,
    committed_to_dataset: bool,
) -> SuccessTraceEvent:
    trace_context = _task_trace_context(task=task, sample_id=outcome.sample.sample_id)
    return SuccessTraceEvent(
        event="success_trace",
        sample_id=outcome.sample.sample_id,
        sample_idx=trace_context.sample_idx,
        source_paths=trace_context.source_paths,
        target_bucket=trace_context.target_bucket,
        planned_line_count=trace_context.planned_line_count,
        candidate_in_target_range=trace_context.candidate_in_target_range,
        committed_to_dataset=committed_to_dataset,
        full_render_system_count=outcome.full_render_system_count,
        full_render_content_height_px=outcome.full_render_content_height_px,
        full_render_vertical_fill_ratio=outcome.full_render_vertical_fill_ratio,
        full_render_rejection_reason=outcome.full_render_rejection_reason,
        accepted_render_system_count=outcome.accepted_render_system_count,
        truncation_attempted=outcome.truncation_attempted,
        truncation_rescued=outcome.truncation_rescued,
        preferred_5_6_rescue_attempted=outcome.preferred_5_6_rescue_attempted,
        preferred_5_6_rescue_succeeded=outcome.preferred_5_6_rescue_succeeded,
        preferred_5_6_status=outcome.preferred_5_6_status,
        initial_kern_spine_count=outcome.sample.initial_kern_spine_count,
        segment_count=outcome.sample.segment_count,
        source_non_empty_line_count=outcome.sample.source_non_empty_line_count,
        truncation_applied=outcome.sample.truncation_applied,
        truncation_reason=outcome.sample.truncation_reason,
        truncation_ratio=outcome.sample.truncation_ratio,
    )


def update_augmentation_summary_counters(
    *,
    counters: dict[str, object],
    trace: AugmentationTraceEvent,
) -> None:
    final_geometry_counts: Counter = counters["final_geometry_counts"]  # type: ignore[assignment]
    oob_failure_reason_counts: Counter = counters["oob_failure_reason_counts"]  # type: ignore[assignment]
    outer_gate_failure_reason_counts: Counter = counters["outer_gate_failure_reason_counts"]  # type: ignore[assignment]

    if not trace.outer_gate.passed:
        final_geometry_counts["base_image_returned"] += 1
    elif trace.final_geometry_applied:
        final_geometry_counts["geometry_survived"] += 1
    else:
        final_geometry_counts["geometry_discarded"] += 1

    if trace.initial_oob_gate.failure_reason is not None:
        oob_failure_reason_counts[trace.initial_oob_gate.failure_reason] += 1
    if trace.retry_oob_gate is not None and trace.retry_oob_gate.failure_reason is not None:
        oob_failure_reason_counts[trace.retry_oob_gate.failure_reason] += 1
    if trace.outer_gate.failure_reason is not None:
        outer_gate_failure_reason_counts[trace.outer_gate.failure_reason] += 1


def _task_trace_context(*, task: TaskTraceLike | None, sample_id: str) -> TaskTraceContext:
    sample_idx = int(sample_id.split("_")[-1])
    source_paths: tuple[str, ...] = ()
    target_bucket = None
    planned_line_count = None
    candidate_in_target_range = None
    if task is not None:
        sample_idx = task.sample_idx
        source_paths = tuple(str(Path(segment.path).resolve()) for segment in task.plan.segments)
        target_bucket = task.target_bucket
        planned_line_count = task.planned_line_count
        candidate_in_target_range = task.candidate_in_target_range
    return TaskTraceContext(
        sample_idx=sample_idx,
        source_paths=source_paths,
        target_bucket=target_bucket,
        planned_line_count=planned_line_count,
        candidate_in_target_range=candidate_in_target_range,
    )
