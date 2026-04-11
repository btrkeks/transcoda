"""Production executor for the rewritten dataset-generation pipeline."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, TimeoutError, wait
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import Features, Image, Sequence, Value
from pebble import ProcessExpired, ProcessPool

from scripts.dataset_generation.dataset_generation.augmentation import augment_accepted_render
from scripts.dataset_generation.dataset_generation.crash_repro import write_crash_artifact
from scripts.dataset_generation.dataset_generation.failure_policy import (
    FailurePolicySettings,
    resolve_failure_policy,
)
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
)
from scripts.dataset_generation.dataset_generation.io import append_jsonl, write_json
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.renderer import render_sample
from scripts.dataset_generation.dataset_generation.resume_store import (
    ResumableShardStore,
    RuntimeSnapshot,
    compute_config_fingerprint,
    write_run_info,
)
from scripts.dataset_generation.dataset_generation.run_context import RunContext, build_run_context
from scripts.dataset_generation.dataset_generation.source_index import (
    InvalidSourceDiagnostic,
    build_source_index,
)
from scripts.dataset_generation.dataset_generation.system_balance import (
    DEFAULT_CANDIDATE_PLAN_COUNT,
    DEFAULT_TOKENIZER_DIR,
    choose_balanced_plan,
    compute_recipe_fingerprint,
    compute_tokenizer_fingerprint,
    load_bundled_system_balance_spec,
    resolve_tokenizer_dir,
    spine_class_for_count,
)
from scripts.dataset_generation.dataset_generation.types import (
    ResumeSnapshot,
    SamplePlan,
    WorkerFailure,
    WorkerSuccess,
)
from scripts.dataset_generation.dataset_generation.worker import (
    evaluate_sample_plan,
    init_generation_worker,
    outcome_to_dataset_row,
    process_sample_plan,
)

_MAX_TASKS_PER_WORKER = 200


@dataclass(frozen=True)
class ExecutionSummary:
    input_dirs: tuple[Path, ...]
    output_dir: Path
    requested_samples: int
    attempted_samples: int
    accepted_samples: int
    rejected_samples: int
    run_artifacts_dir: Path


@dataclass
class ScheduledTask:
    future: object
    sample_idx: int
    plan: SamplePlan
    scheduled_ns: int
    target_bucket: int | None = None
    planned_line_count: int | None = None
    candidate_in_target_range: bool | None = None
    timeout_retries: int = 0
    expired_retries: int = 0


@dataclass(frozen=True)
class PlannedTask:
    sample_idx: int
    plan: SamplePlan
    target_bucket: int | None = None
    planned_line_count: int | None = None
    candidate_in_target_range: bool | None = None


@dataclass(frozen=True)
class SystemBalanceRuntime:
    mode: str
    spec_path: Path | None = None
    spec_fingerprint: str | None = None
    tokenizer_path: Path | None = None
    tokenizer_fingerprint: str | None = None
    recipe_fingerprint: str | None = None
    candidate_plan_count: int = DEFAULT_CANDIDATE_PLAN_COUNT
    spec: object | None = None


def run_dataset_generation(
    input_dirs: str | Path | tuple[str | Path, ...] | list[str | Path],
    output_dir: str | Path,
    *,
    target_samples: int,
    num_workers: int = 1,
    artifacts_out_dir: str | Path | None = None,
    resume_mode: str = "auto",
    base_seed: int = 0,
    max_attempts: int | None = None,
    failure_policy: str = "balanced",
    quarantine_in: str | Path | None = None,
    quarantine_out: str | Path | None = None,
    quiet: bool = False,
    capture_verovio_diagnostics: bool = True,
    recipe: ProductionRecipe | None = None,
    renderer: VerovioRenderer | None = None,
    render_fn: Callable[..., object] | None = None,
    augment_fn: Callable[..., object] | None = None,
) -> ExecutionSummary:
    active_recipe = recipe or ProductionRecipe()
    normalized_input_dirs = _normalize_input_dirs(input_dirs)
    source_index = build_source_index(*normalized_input_dirs)
    auto_quarantined_source_count = len(source_index.invalid_sources)
    auto_quarantined_source_examples = _serialize_invalid_source_examples(
        source_index.invalid_sources
    )
    auto_quarantined_paths = {diagnostic.path for diagnostic in source_index.invalid_sources}

    if target_samples < 1:
        raise ValueError("target_samples must be >= 1")

    resolved_num_workers = max(1, int(num_workers))
    attempt_budget = (
        max_attempts
        if max_attempts is not None
        else max(target_samples, target_samples * active_recipe.max_attempt_multiplier)
    )
    if attempt_budget < target_samples:
        raise ValueError("max_attempts must be >= target_samples")

    resolved_failure_policy = resolve_failure_policy(failure_policy)
    run_context = build_run_context(
        output_dir=output_dir,
        artifacts_out_dir=artifacts_out_dir,
    )
    features = Features(
        {
            "image": Image(mode="RGB"),
            "transcription": Value("string"),
            "sample_id": Value("string"),
            "initial_kern_spine_count": Value("int32"),
            "source_ids": Sequence(Value("string")),
            "segment_count": Value("int32"),
            "source_measure_count": Value("int32"),
            "source_non_empty_line_count": Value("int32"),
            "svg_system_count": Value("int32"),
            "truncation_applied": Value("bool"),
            "truncation_reason": Value("string"),
            "truncation_ratio": Value("float32"),
            "vertical_fill_ratio": Value("float32"),
            "bottom_whitespace_ratio": Value("float32"),
            "bottom_whitespace_px": Value("int32"),
            "top_whitespace_px": Value("int32"),
            "content_height_px": Value("int32"),
            "recipe_version": Value("string"),
        }
    )
    balance_runtime = _resolve_system_balance_runtime(
        recipe=active_recipe,
        quiet=quiet,
    )
    config_fingerprint = compute_config_fingerprint(
        input_dirs=tuple(str(path) for path in normalized_input_dirs),
        base_seed=base_seed,
        recipe=active_recipe,
        extra_config={
            "system_balance": _serialize_system_balance_runtime(balance_runtime),
        },
    )
    resume_store = ResumableShardStore(
        run_context=run_context,
        features=features,
        config_fingerprint=config_fingerprint,
        resume_mode=resume_mode,
    )
    resume_snapshot = resume_store.prepare()

    counters = _build_runtime_counters(resume_snapshot)
    counters["quarantined_sources"].update(_load_quarantine_sources(quarantine_in))
    counters["quarantined_sources"].update(auto_quarantined_paths)
    quarantine_out_path = _resolve_quarantine_out_path(
        quarantine_out=quarantine_out,
        run_context=run_context,
    )
    _write_quarantined_sources(
        quarantined_sources=counters["quarantined_sources"],  # type: ignore[arg-type]
        destination=quarantine_out_path,
    )

    next_to_commit = counters["next_sample_idx"]
    next_to_schedule = next_to_commit
    pending_terminal: dict[int, WorkerSuccess | WorkerFailure] = {}
    pending_rows: list[dict[str, object]] = []
    futures_by_handle: dict[object, ScheduledTask] = {}
    tasks_by_sample_idx: dict[int, ScheduledTask] = {}
    active_workers = 0
    progress_interval_seconds = 30.0
    last_progress_at = 0.0
    generation_start = time.perf_counter()

    use_process_pool = (
        resolved_num_workers > 1
        and render_fn is None
        and augment_fn is None
        and renderer is None
    )
    sync_renderer = renderer
    sync_render_callable = render_fn or render_sample
    sync_augment_callable = augment_fn or augment_accepted_render

    if not quiet:
        print(
            f"Indexed {len(source_index.entries)} valid source(s); "
            f"auto-quarantined {auto_quarantined_source_count} invalid source(s)"
        )
        for diagnostic in source_index.invalid_sources[:3]:
            print(
                f"  auto-quarantined: {diagnostic.path} "
                f"({diagnostic.reason_code}: {diagnostic.message})"
            )
        print(
            f"Generating rewrite dataset from {len(source_index.entries)} valid source(s) "
            f"into {run_context.output_path} with {resolved_num_workers} worker(s)"
        )

    try:
        if not _has_schedulable_entries(
            source_index=source_index,
            quarantined_sources=counters["quarantined_sources"],  # type: ignore[arg-type]
        ):
            raise RuntimeError(
                "No schedulable sources remain after applying quarantine "
                f"({auto_quarantined_source_count} auto-quarantined invalid source(s))"
            )
        if not use_process_pool:
            if sync_renderer is None:
                sync_renderer = VerovioRenderer()
            while counters["accepted_samples"] < target_samples and next_to_schedule < attempt_budget:
                plan = _plan_with_quarantine(
                    source_index=source_index,
                    recipe=active_recipe,
                    sample_idx=next_to_schedule,
                    base_seed=base_seed,
                    quarantined_sources=counters["quarantined_sources"],  # type: ignore[arg-type]
                    system_balance_runtime=balance_runtime,
                    accepted_system_histogram=counters["accepted_system_histogram"],  # type: ignore[arg-type]
                )
                _record_planning_choice(counters=counters, task=plan)
                outcome = evaluate_sample_plan(
                    plan.plan,
                    recipe=active_recipe,
                    renderer=sync_renderer,
                    render_fn=sync_render_callable,
                    augment_fn=sync_augment_callable,
                    capture_verovio_diagnostics=capture_verovio_diagnostics,
                )
                pending_terminal[next_to_schedule] = outcome
                next_to_schedule += 1
                next_to_commit_ref = [next_to_commit]
                pending_rows = _commit_contiguous_results(
                    pending_terminal=pending_terminal,
                    pending_rows=pending_rows,
                    counters=counters,
                    next_to_commit_ref=next_to_commit_ref,
                    recipe=active_recipe,
                    target_samples=target_samples,
                    run_context=run_context,
                )
                next_to_commit = next_to_commit_ref[0]
                last_progress_at_ref = [last_progress_at]
                _maybe_flush_and_report(
                    resume_store=resume_store,
                    run_context=run_context,
                    counters=counters,
                    pending_rows=pending_rows,
                    active_workers=0,
                    target_samples=target_samples,
                    last_progress_at_ref=last_progress_at_ref,
                    progress_interval_seconds=progress_interval_seconds,
                    quiet=quiet,
                    quarantine_out_path=quarantine_out_path,
                )
                last_progress_at = last_progress_at_ref[0]
            if pending_rows:
                resume_store.commit(snapshot=_snapshot_from_counters(counters), sample_rows=pending_rows)
                pending_rows = []
        else:
            max_in_flight = max(resolved_num_workers * 4, resolved_num_workers)
            with ProcessPool(
                max_workers=resolved_num_workers,
                max_tasks=_MAX_TASKS_PER_WORKER,
                initializer=init_generation_worker,
                initargs=(active_recipe, capture_verovio_diagnostics),
                context=mp.get_context("spawn"),
            ) as pool:
                while True:
                    while (
                        len(futures_by_handle) < max_in_flight
                        and next_to_schedule < attempt_budget
                        and counters["accepted_samples"] < target_samples
                    ):
                        plan = _plan_with_quarantine(
                            source_index=source_index,
                            recipe=active_recipe,
                            sample_idx=next_to_schedule,
                            base_seed=base_seed,
                            quarantined_sources=counters["quarantined_sources"],  # type: ignore[arg-type]
                            system_balance_runtime=balance_runtime,
                            accepted_system_histogram=counters["accepted_system_histogram"],  # type: ignore[arg-type]
                        )
                        _record_planning_choice(counters=counters, task=plan)
                        task = _schedule_task(
                            pool=pool,
                            plan=plan.plan,
                            failure_policy=resolved_failure_policy,
                            target_bucket=plan.target_bucket,
                            planned_line_count=plan.planned_line_count,
                            candidate_in_target_range=plan.candidate_in_target_range,
                        )
                        futures_by_handle[task.future] = task
                        tasks_by_sample_idx[task.sample_idx] = task
                        next_to_schedule += 1

                    if not futures_by_handle:
                        break

                    done, _ = wait(
                        tuple(futures_by_handle),
                        timeout=0.5,
                        return_when=FIRST_COMPLETED,
                    )
                    active_workers = len(futures_by_handle) - len(done)
                    if not done:
                        last_progress_at_ref = [last_progress_at]
                        _maybe_flush_and_report(
                            resume_store=resume_store,
                            run_context=run_context,
                            counters=counters,
                            pending_rows=pending_rows,
                            active_workers=active_workers,
                            target_samples=target_samples,
                            last_progress_at_ref=last_progress_at_ref,
                            progress_interval_seconds=progress_interval_seconds,
                            quiet=quiet,
                            quarantine_out_path=quarantine_out_path,
                        )
                        last_progress_at = last_progress_at_ref[0]
                        continue

                    for future in done:
                        # A terminal failure earlier in this same done batch can quarantine an
                        # overlapping sibling and remove its future from the tracking map before
                        # we iterate to it here.
                        task = futures_by_handle.pop(future, None)
                        if task is None:
                            continue
                        tasks_by_sample_idx.pop(task.sample_idx, None)
                        before_result_ns = time.perf_counter_ns()
                        queue_wait_ms = (before_result_ns - task.scheduled_ns) / 1_000_000.0

                        try:
                            outcome = future.result()
                        except TimeoutError:
                            maybe_rescheduled = _maybe_retry_task(
                                failure_kind="timeout",
                                task=task,
                                failure_policy=resolved_failure_policy,
                                pool=pool,
                                counters=counters,
                                pending_terminal=pending_terminal,
                                futures_by_handle=futures_by_handle,
                                tasks_by_sample_idx=tasks_by_sample_idx,
                            )
                            if maybe_rescheduled:
                                append_jsonl(
                                    run_context.timeout_events_path,
                                    [
                                        _build_timeout_event(
                                            task=task,
                                            queue_wait_ms=queue_wait_ms,
                                            will_retry=True,
                                        )
                                    ],
                                )
                            else:
                                dropped_pending_tasks = _quarantine_pending_tasks(
                                    trigger_task=task,
                                    counters=counters,
                                    pending_terminal=pending_terminal,
                                    futures_by_handle=futures_by_handle,
                                    tasks_by_sample_idx=tasks_by_sample_idx,
                                )
                                crash_artifact_path, crash_repro_stage_count = _write_terminal_crash_artifact(
                                    run_context=run_context,
                                    recipe=active_recipe,
                                    task=task,
                                    event_type="timeout",
                                    retry_count=task.timeout_retries,
                                    will_retry=False,
                                    queue_wait_ms=queue_wait_ms,
                                    dropped_pending_tasks=dropped_pending_tasks,
                                    counters=counters,
                                )
                                append_jsonl(
                                    run_context.timeout_events_path,
                                    [
                                        _build_timeout_event(
                                            task=task,
                                            queue_wait_ms=queue_wait_ms,
                                            will_retry=False,
                                            dropped_pending_tasks=dropped_pending_tasks,
                                            crash_artifact_json_path=str(crash_artifact_path),
                                            crash_repro_stage_count=crash_repro_stage_count,
                                        )
                                    ],
                                )
                        except ProcessExpired as exc:
                            maybe_rescheduled = _maybe_retry_task(
                                failure_kind="process_expired",
                                task=task,
                                failure_policy=resolved_failure_policy,
                                pool=pool,
                                counters=counters,
                                pending_terminal=pending_terminal,
                                futures_by_handle=futures_by_handle,
                                tasks_by_sample_idx=tasks_by_sample_idx,
                            )
                            if maybe_rescheduled:
                                append_jsonl(
                                    run_context.process_expired_events_path,
                                    [
                                        _build_process_expired_event(
                                            task=task,
                                            queue_wait_ms=queue_wait_ms,
                                            exc=exc,
                                            will_retry=True,
                                        )
                                    ],
                                )
                            else:
                                dropped_pending_tasks = _quarantine_pending_tasks(
                                    trigger_task=task,
                                    counters=counters,
                                    pending_terminal=pending_terminal,
                                    futures_by_handle=futures_by_handle,
                                    tasks_by_sample_idx=tasks_by_sample_idx,
                                )
                                crash_artifact_path, crash_repro_stage_count = _write_terminal_crash_artifact(
                                    run_context=run_context,
                                    recipe=active_recipe,
                                    task=task,
                                    event_type="process_expired",
                                    retry_count=task.expired_retries,
                                    will_retry=False,
                                    queue_wait_ms=queue_wait_ms,
                                    dropped_pending_tasks=dropped_pending_tasks,
                                    counters=counters,
                                    exc=exc,
                                )
                                append_jsonl(
                                    run_context.process_expired_events_path,
                                    [
                                        _build_process_expired_event(
                                            task=task,
                                            queue_wait_ms=queue_wait_ms,
                                            exc=exc,
                                            will_retry=False,
                                            dropped_pending_tasks=dropped_pending_tasks,
                                            crash_artifact_json_path=str(crash_artifact_path),
                                            crash_repro_stage_count=crash_repro_stage_count,
                                        )
                                    ],
                                )
                        except BaseException as exc:
                            pending_terminal[task.sample_idx] = WorkerFailure(
                                sample_id=task.plan.sample_id,
                                failure_reason=f"task_error:{type(exc).__name__}",
                                truncation_attempted=False,
                            )
                            dropped_pending_tasks = _quarantine_pending_tasks(
                                trigger_task=task,
                                counters=counters,
                                pending_terminal=pending_terminal,
                                futures_by_handle=futures_by_handle,
                                tasks_by_sample_idx=tasks_by_sample_idx,
                            )
                        else:
                            pending_terminal[task.sample_idx] = outcome

                    next_to_commit_ref = [next_to_commit]
                    pending_rows = _commit_contiguous_results(
                        pending_terminal=pending_terminal,
                        pending_rows=pending_rows,
                        counters=counters,
                        next_to_commit_ref=next_to_commit_ref,
                        recipe=active_recipe,
                        target_samples=target_samples,
                        run_context=run_context,
                    )
                    next_to_commit = next_to_commit_ref[0]

                    last_progress_at_ref = [last_progress_at]
                    _maybe_flush_and_report(
                        resume_store=resume_store,
                        run_context=run_context,
                        counters=counters,
                        pending_rows=pending_rows,
                        active_workers=active_workers,
                        target_samples=target_samples,
                        last_progress_at_ref=last_progress_at_ref,
                        progress_interval_seconds=progress_interval_seconds,
                        quiet=quiet,
                        quarantine_out_path=quarantine_out_path,
                    )
                    last_progress_at = last_progress_at_ref[0]

                    if counters["accepted_samples"] >= target_samples and not futures_by_handle:
                        break

            if pending_rows:
                resume_store.commit(snapshot=_snapshot_from_counters(counters), sample_rows=pending_rows)
                pending_rows = []

        resume_store.commit(snapshot=_snapshot_from_counters(counters), sample_rows=[])
        generation_seconds = time.perf_counter() - generation_start
        finalization = resume_store.finalize()
        runtime_seconds = {
            "generation": generation_seconds,
            "total": generation_seconds,
        }
        _write_quarantined_sources(
            quarantined_sources=counters["quarantined_sources"],  # type: ignore[arg-type]
            destination=quarantine_out_path,
        )
        write_run_info(
            run_context=run_context,
            recipe=active_recipe,
            input_dirs=tuple(str(path) for path in normalized_input_dirs),
            output_dir=str(run_context.output_path),
            num_workers=resolved_num_workers,
            target_samples=target_samples,
            max_attempts=attempt_budget,
            base_seed=base_seed,
            resume_mode=resume_mode,
            failure_policy=resolved_failure_policy.as_dict(),
            capture_verovio_diagnostics=capture_verovio_diagnostics,
            quarantine_in=str(Path(quarantine_in).expanduser().resolve()) if quarantine_in else None,
            quarantine_out=str(quarantine_out_path),
            system_balance=_serialize_system_balance_runtime(balance_runtime),
            runtime_seconds=runtime_seconds,
            layout_summary=_build_layout_summary(counters),
            snapshot=finalization["snapshot"],
            finalization=finalization,
            auto_quarantined_source_count=auto_quarantined_source_count,
            invalid_source_examples=auto_quarantined_source_examples,
        )
        write_json(
            run_context.latest_run_path,
            {
                "run_id": run_context.run_id,
                "run_artifacts_dir": str(run_context.run_artifacts_dir),
                "info_path": str(run_context.info_path),
                "updated_at": time.time(),
            },
        )
        return ExecutionSummary(
            input_dirs=tuple(Path(path) for path in normalized_input_dirs),
            output_dir=run_context.output_path,
            requested_samples=target_samples,
            attempted_samples=int(counters["next_sample_idx"]),
            accepted_samples=int(counters["accepted_samples"]),
            rejected_samples=int(counters["rejected_samples"]),
            run_artifacts_dir=run_context.run_artifacts_dir,
        )
    except BaseException:
        if pending_rows:
            resume_store.commit(snapshot=_snapshot_from_counters(counters), sample_rows=pending_rows)
        _write_quarantined_sources(
            quarantined_sources=counters["quarantined_sources"],  # type: ignore[arg-type]
            destination=quarantine_out_path,
        )
        resume_store.mark_terminal_status(status="failed")
        write_run_info(
            run_context=run_context,
            recipe=active_recipe,
            input_dirs=tuple(str(path) for path in normalized_input_dirs),
            output_dir=str(run_context.output_path),
            num_workers=resolved_num_workers,
            target_samples=target_samples,
            max_attempts=attempt_budget,
            base_seed=base_seed,
            resume_mode=resume_mode,
            failure_policy=resolved_failure_policy.as_dict(),
            capture_verovio_diagnostics=capture_verovio_diagnostics,
            quarantine_in=str(Path(quarantine_in).expanduser().resolve()) if quarantine_in else None,
            quarantine_out=str(quarantine_out_path),
            system_balance=_serialize_system_balance_runtime(balance_runtime),
            layout_summary=_build_layout_summary(counters),
            snapshot=_snapshot_from_counters(counters),
            auto_quarantined_source_count=auto_quarantined_source_count,
            invalid_source_examples=auto_quarantined_source_examples,
        )
        raise


def _normalize_input_dirs(
    input_dirs: str | Path | tuple[str | Path, ...] | list[str | Path],
) -> tuple[Path, ...]:
    if isinstance(input_dirs, (str, Path)):
        values = (input_dirs,)
    else:
        values = tuple(input_dirs)
    if not values:
        raise ValueError("At least one input directory is required")
    return tuple(Path(value).expanduser().resolve() for value in values)


def _load_quarantine_sources(path: str | Path | None) -> set[Path]:
    if path is None:
        return set()
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_values = payload
    elif isinstance(payload, dict):
        raw_values = (
            payload.get("quarantined_sources")
            or payload.get("sources")
            or payload.get("paths")
            or payload.get("files")
            or []
        )
    else:
        raise ValueError("Unsupported quarantine input payload")
    return {Path(value).expanduser().resolve() for value in raw_values}


def _resolve_quarantine_out_path(
    *,
    quarantine_out: str | Path | None,
    run_context: RunContext,
) -> Path:
    if quarantine_out is None:
        return run_context.quarantined_sources_path
    return Path(quarantine_out).expanduser().resolve()


def _write_quarantined_sources(*, quarantined_sources: set[Path], destination: Path) -> None:
    write_json(
        destination,
        {
            "quarantined_sources": sorted(str(path) for path in quarantined_sources),
        },
    )


def _serialize_invalid_source_examples(
    diagnostics: tuple[InvalidSourceDiagnostic, ...], *, limit: int = 5
) -> list[dict[str, object]]:
    return [
        {
            "path": str(diagnostic.path),
            "root_dir": str(diagnostic.root_dir),
            "root_label": diagnostic.root_label,
            "reason_code": diagnostic.reason_code,
            "message": diagnostic.message,
        }
        for diagnostic in diagnostics[:limit]
    ]


def _has_schedulable_entries(*, source_index, quarantined_sources: set[Path]) -> bool:
    return any(entry.path not in quarantined_sources for entry in source_index.entries)


def _plan_with_quarantine(
    *,
    source_index,
    recipe: ProductionRecipe,
    sample_idx: int,
    base_seed: int,
    quarantined_sources: set[Path],
    system_balance_runtime: SystemBalanceRuntime,
    accepted_system_histogram: Counter,
) -> PlannedTask:
    try:
        if system_balance_runtime.mode != "spine_aware_line_proxy":
            raise RuntimeError(
                f"Unsupported mandatory system balance mode: {system_balance_runtime.mode!r}"
            )
        assert system_balance_runtime.spec is not None
        selected = choose_balanced_plan(
            source_index=source_index,
            recipe=recipe,
            sample_idx=sample_idx,
            base_seed=base_seed,
            excluded_paths=quarantined_sources,
            spec=system_balance_runtime.spec,  # type: ignore[arg-type]
            accepted_system_histogram=accepted_system_histogram,
            candidate_plan_count=system_balance_runtime.candidate_plan_count,
        )
        return PlannedTask(
            sample_idx=sample_idx,
            plan=selected.plan,
            target_bucket=selected.target_bucket,
            planned_line_count=selected.line_count,
            candidate_in_target_range=selected.in_target_range,
        )
    except ValueError as exc:
        available_entries = [
            entry for entry in source_index.entries if entry.path not in quarantined_sources
        ]
        if not available_entries:
            raise RuntimeError("No schedulable sources remain after applying quarantine") from exc
        raise RuntimeError(
            f"Failed to build a valid balanced plan after exhausting candidate plans: {exc}"
        ) from exc


def _schedule_task(
    *,
    pool: ProcessPool,
    plan: SamplePlan,
    failure_policy: FailurePolicySettings,
    target_bucket: int | None = None,
    planned_line_count: int | None = None,
    candidate_in_target_range: bool | None = None,
    timeout_retries: int = 0,
    expired_retries: int = 0,
) -> ScheduledTask:
    future = pool.schedule(
        process_sample_plan,
        args=(plan,),
        timeout=failure_policy.task_timeout_seconds,
    )
    return ScheduledTask(
        future=future,
        sample_idx=int(plan.sample_id.split("_")[-1]),
        plan=plan,
        scheduled_ns=time.perf_counter_ns(),
        target_bucket=target_bucket,
        planned_line_count=planned_line_count,
        candidate_in_target_range=candidate_in_target_range,
        timeout_retries=timeout_retries,
        expired_retries=expired_retries,
    )


def _maybe_retry_task(
    *,
    failure_kind: str,
    task: ScheduledTask,
    failure_policy: FailurePolicySettings,
    pool: ProcessPool,
    counters: dict[str, object],
    pending_terminal: dict[int, WorkerSuccess | WorkerFailure],
    futures_by_handle: dict[object, ScheduledTask],
    tasks_by_sample_idx: dict[int, ScheduledTask],
) -> bool:
    retry_counts: Counter = counters["retry_counts"]  # type: ignore[assignment]
    if failure_kind == "timeout":
        if task.timeout_retries >= failure_policy.max_task_retries_timeout:
            pending_terminal[task.sample_idx] = WorkerFailure(
                sample_id=task.plan.sample_id,
                failure_reason="timeout",
                truncation_attempted=False,
            )
            return False
        retried_task = _schedule_task(
            pool=pool,
            plan=task.plan,
            failure_policy=failure_policy,
            target_bucket=task.target_bucket,
            planned_line_count=task.planned_line_count,
            candidate_in_target_range=task.candidate_in_target_range,
            timeout_retries=task.timeout_retries + 1,
            expired_retries=task.expired_retries,
        )
        retry_counts["timeout"] += 1
    else:
        if task.expired_retries >= failure_policy.max_task_retries_expired:
            pending_terminal[task.sample_idx] = WorkerFailure(
                sample_id=task.plan.sample_id,
                failure_reason="process_expired",
                truncation_attempted=False,
            )
            return False
        retried_task = _schedule_task(
            pool=pool,
            plan=task.plan,
            failure_policy=failure_policy,
            target_bucket=task.target_bucket,
            planned_line_count=task.planned_line_count,
            candidate_in_target_range=task.candidate_in_target_range,
            timeout_retries=task.timeout_retries,
            expired_retries=task.expired_retries + 1,
        )
        retry_counts["process_expired"] += 1

    futures_by_handle[retried_task.future] = retried_task
    tasks_by_sample_idx[retried_task.sample_idx] = retried_task
    return True


def _quarantine_pending_tasks(
    *,
    trigger_task: ScheduledTask,
    counters: dict[str, object],
    pending_terminal: dict[int, WorkerSuccess | WorkerFailure],
    futures_by_handle: dict[object, ScheduledTask],
    tasks_by_sample_idx: dict[int, ScheduledTask],
) -> int:
    quarantined_sources: set[Path] = counters["quarantined_sources"]  # type: ignore[assignment]
    new_sources = {segment.path.resolve() for segment in trigger_task.plan.segments}
    quarantined_sources.update(new_sources)

    dropped_pending_tasks = 0
    for sample_idx, task in list(tasks_by_sample_idx.items()):
        if sample_idx == trigger_task.sample_idx:
            continue
        if not _task_overlaps_sources(task=task, source_paths=new_sources):
            continue
        futures_by_handle.pop(task.future, None)
        tasks_by_sample_idx.pop(sample_idx, None)
        task.future.cancel()
        pending_terminal[sample_idx] = WorkerFailure(
            sample_id=task.plan.sample_id,
            failure_reason="quarantined",
            truncation_attempted=False,
        )
        dropped_pending_tasks += 1
    return dropped_pending_tasks


def _task_overlaps_sources(*, task: ScheduledTask, source_paths: set[Path]) -> bool:
    return any(segment.path.resolve() in source_paths for segment in task.plan.segments)


def _resolve_system_balance_runtime(
    *,
    recipe: ProductionRecipe,
    quiet: bool,
) -> SystemBalanceRuntime:
    try:
        spec = load_bundled_system_balance_spec()
        resolved_tokenizer_path = resolve_tokenizer_dir(DEFAULT_TOKENIZER_DIR)
        tokenizer_fingerprint = compute_tokenizer_fingerprint(resolved_tokenizer_path)
    except Exception as exc:
        raise RuntimeError(f"Bundled system balance spec could not be loaded: {exc}") from exc

    expected_recipe_fingerprint = compute_recipe_fingerprint(recipe)
    if spec.mode != "spine_aware_line_proxy":
        raise RuntimeError(
            f"Bundled system balance spec has unsupported mode: {spec.mode!r}"
        )
    if spec.recipe_fingerprint != expected_recipe_fingerprint:
        raise RuntimeError(
            "Bundled system balance spec is incompatible with the current recipe fingerprint"
        )
    if spec.tokenizer_fingerprint != tokenizer_fingerprint:
        raise RuntimeError(
            "Bundled system balance spec is incompatible with the current tokenizer fingerprint"
        )

    spec_path = spec.path
    spec_fingerprint = hashlib_sha256(spec_path.read_bytes())
    if not quiet:
        print(f"Using bundled system balance spec: {spec_path}")
    return SystemBalanceRuntime(
        mode="spine_aware_line_proxy",
        spec_path=spec_path,
        spec_fingerprint=spec_fingerprint,
        tokenizer_path=resolved_tokenizer_path,
        tokenizer_fingerprint=tokenizer_fingerprint,
        recipe_fingerprint=spec.recipe_fingerprint,
        candidate_plan_count=spec.candidate_plan_count,
        spec=spec,
    )


def _serialize_system_balance_runtime(runtime: SystemBalanceRuntime) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": runtime.mode,
        "mandatory": True,
    }
    if runtime.spec_path is not None:
        payload["spec_path"] = str(runtime.spec_path)
    if runtime.spec_fingerprint is not None:
        payload["spec_fingerprint"] = runtime.spec_fingerprint
    if runtime.tokenizer_path is not None:
        payload["tokenizer_path"] = str(runtime.tokenizer_path)
    if runtime.tokenizer_fingerprint is not None:
        payload["tokenizer_fingerprint"] = runtime.tokenizer_fingerprint
    if runtime.recipe_fingerprint is not None:
        payload["recipe_fingerprint"] = runtime.recipe_fingerprint
    if runtime.mode == "spine_aware_line_proxy":
        payload["candidate_plan_count"] = runtime.candidate_plan_count
    return payload


def _record_planning_choice(*, counters: dict[str, object], task: PlannedTask) -> None:
    if task.target_bucket is None:
        return
    requested_histogram: Counter = counters["requested_target_bucket_histogram"]  # type: ignore[assignment]
    requested_histogram[int(task.target_bucket)] += 1
    candidate_hit_counts: Counter = counters["candidate_hit_counts"]  # type: ignore[assignment]
    if task.candidate_in_target_range:
        candidate_hit_counts["inside_target_bucket"] += 1
    else:
        candidate_hit_counts["outside_target_bucket"] += 1


def hashlib_sha256(raw_bytes: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw_bytes).hexdigest()


def _build_runtime_counters(snapshot: ResumeSnapshot | None) -> dict[str, object]:
    if snapshot is None:
        return {
            "next_sample_idx": 0,
            "accepted_samples": 0,
            "rejected_samples": 0,
            "failure_reason_counts": Counter(),
            "truncation_counts": Counter({"attempted": 0, "rescued": 0, "failed": 0}),
            "full_render_system_histogram": Counter(),
            "accepted_system_histogram": defaultdict(Counter),
            "truncated_output_system_histogram": Counter(),
            "preferred_5_6_counts": Counter(
                {
                    "preferred_5_6_accepted_full": 0,
                    "preferred_5_6_rescued": 0,
                    "preferred_5_6_truncated": 0,
                    "preferred_5_6_failed": 0,
                }
            ),
            "bottom_whitespace_px_histogram": Counter(),
            "top_whitespace_px_histogram": Counter(),
            "content_height_px_histogram": Counter(),
            "terminal_timeout_crash_artifacts": 0,
            "terminal_process_expired_crash_artifacts": 0,
            "requested_target_bucket_histogram": Counter(),
            "candidate_hit_counts": Counter(),
            "retry_counts": Counter(),
            "quarantined_sources": set(),
            "augmentation_outcome_counts": Counter(),
            "augmentation_band_counts": Counter(),
            "augmentation_branch_counts": Counter(),
        }
    return {
        "next_sample_idx": snapshot.next_sample_idx,
        "accepted_samples": snapshot.accepted_samples,
        "rejected_samples": snapshot.rejected_samples,
        "failure_reason_counts": Counter(snapshot.failure_reason_counts),
        "truncation_counts": Counter(snapshot.truncation_counts),
        "full_render_system_histogram": Counter(
            {int(key): int(value) for key, value in snapshot.full_render_system_histogram.items()}
        ),
        "accepted_system_histogram": defaultdict(
            Counter,
            {
                str(spine_cls): Counter(
                    {int(bucket): int(count) for bucket, count in bucket_counts.items()}
                )
                for spine_cls, bucket_counts in snapshot.accepted_system_histogram.items()
            },
        ),
        "truncated_output_system_histogram": Counter(
            {
                int(key): int(value)
                for key, value in snapshot.truncated_output_system_histogram.items()
            }
        ),
        "preferred_5_6_counts": Counter(snapshot.preferred_5_6_counts),
        "bottom_whitespace_px_histogram": Counter(
            {
                int(key): int(value)
                for key, value in snapshot.bottom_whitespace_px_histogram.items()
            }
        ),
        "top_whitespace_px_histogram": Counter(
            {
                int(key): int(value)
                for key, value in snapshot.top_whitespace_px_histogram.items()
            }
        ),
        "content_height_px_histogram": Counter(
            {
                int(key): int(value)
                for key, value in snapshot.content_height_px_histogram.items()
            }
        ),
        "terminal_timeout_crash_artifacts": int(snapshot.terminal_timeout_crash_artifacts),
        "terminal_process_expired_crash_artifacts": int(
            snapshot.terminal_process_expired_crash_artifacts
        ),
        "requested_target_bucket_histogram": Counter(
            {
                int(key): int(value)
                for key, value in snapshot.requested_target_bucket_histogram.items()
            }
        ),
        "candidate_hit_counts": Counter(snapshot.candidate_hit_counts),
        "retry_counts": Counter(snapshot.retry_counts),
        "quarantined_sources": {Path(path).expanduser().resolve() for path in snapshot.quarantined_sources},
        "augmentation_outcome_counts": Counter(snapshot.augmentation_outcome_counts),
        "augmentation_band_counts": Counter(snapshot.augmentation_band_counts),
        "augmentation_branch_counts": Counter(snapshot.augmentation_branch_counts),
    }


def _commit_contiguous_results(
    *,
    pending_terminal: dict[int, WorkerSuccess | WorkerFailure],
    pending_rows: list[dict[str, object]],
    counters: dict[str, object],
    next_to_commit_ref: list[int],
    recipe: ProductionRecipe,
    target_samples: int,
    run_context: RunContext,
) -> list[dict[str, object]]:
    next_to_commit = next_to_commit_ref[0]
    while next_to_commit in pending_terminal:
        outcome = pending_terminal.pop(next_to_commit)
        _write_verovio_events(run_context=run_context, outcome=outcome)
        _write_augmentation_events(run_context=run_context, outcome=outcome)
        truncation_counts: Counter = counters["truncation_counts"]  # type: ignore[assignment]
        failure_counts: Counter = counters["failure_reason_counts"]  # type: ignore[assignment]
        full_render_histogram: Counter = counters["full_render_system_histogram"]  # type: ignore[assignment]
        system_histogram: defaultdict[str, Counter] = counters["accepted_system_histogram"]  # type: ignore[assignment]
        truncated_output_histogram: Counter = counters["truncated_output_system_histogram"]  # type: ignore[assignment]
        preferred_5_6_counts: Counter = counters["preferred_5_6_counts"]  # type: ignore[assignment]
        bottom_whitespace_px_histogram: Counter = counters["bottom_whitespace_px_histogram"]  # type: ignore[assignment]
        top_whitespace_px_histogram: Counter = counters["top_whitespace_px_histogram"]  # type: ignore[assignment]
        content_height_px_histogram: Counter = counters["content_height_px_histogram"]  # type: ignore[assignment]
        if outcome.truncation_attempted:
            truncation_counts["attempted"] += 1
        if outcome.truncation_rescued:
            truncation_counts["rescued"] += 1
        elif outcome.truncation_attempted:
            truncation_counts["failed"] += 1
        if outcome.full_render_system_count is not None:
            full_render_histogram[int(outcome.full_render_system_count)] += 1
        if outcome.preferred_5_6_status is not None:
            preferred_5_6_counts[str(outcome.preferred_5_6_status)] += 1

        if isinstance(outcome, WorkerSuccess):
            if int(counters["accepted_samples"]) < target_samples:
                pending_rows.append(outcome_to_dataset_row(outcome, recipe=recipe))
                counters["accepted_samples"] = int(counters["accepted_samples"]) + 1
                accepted_spine_class = spine_class_for_count(
                    outcome.sample.initial_kern_spine_count
                )
                system_histogram[accepted_spine_class][int(outcome.sample.system_count)] += 1
                if outcome.sample.bottom_whitespace_px is not None:
                    bottom_whitespace_px_histogram[int(outcome.sample.bottom_whitespace_px)] += 1
                if outcome.sample.top_whitespace_px is not None:
                    top_whitespace_px_histogram[int(outcome.sample.top_whitespace_px)] += 1
                if outcome.sample.content_height_px is not None:
                    content_height_px_histogram[int(outcome.sample.content_height_px)] += 1
                if outcome.sample.truncation_applied:
                    truncated_output_histogram[int(outcome.sample.system_count)] += 1
                if outcome.augmentation_trace is not None:
                    aug_outcome_counts: Counter = counters["augmentation_outcome_counts"]  # type: ignore[assignment]
                    aug_band_counts: Counter = counters["augmentation_band_counts"]  # type: ignore[assignment]
                    aug_branch_counts: Counter = counters["augmentation_branch_counts"]  # type: ignore[assignment]
                    aug_outcome_counts[outcome.augmentation_trace.final_outcome] += 1
                    aug_band_counts[outcome.augmentation_trace.band] += 1
                    aug_branch_counts[outcome.augmentation_trace.branch] += 1
            else:
                failure_counts["discarded_after_target"] += 1
        else:
            counters["rejected_samples"] = int(counters["rejected_samples"]) + 1
            failure_counts[str(outcome.failure_reason)] += 1
        next_to_commit += 1

    counters["next_sample_idx"] = next_to_commit
    next_to_commit_ref[0] = next_to_commit
    return pending_rows


def _write_verovio_events(
    *,
    run_context: RunContext,
    outcome: WorkerSuccess | WorkerFailure,
) -> None:
    if not outcome.verovio_diagnostics:
        return
    append_jsonl(
        run_context.verovio_events_path,
        [asdict(event) for event in outcome.verovio_diagnostics],
    )


def _write_augmentation_events(
    *,
    run_context: RunContext,
    outcome: WorkerSuccess | WorkerFailure,
) -> None:
    if not isinstance(outcome, WorkerSuccess) or outcome.augmentation_trace is None:
        return
    append_jsonl(
        run_context.augmentation_events_path,
        [asdict(outcome.augmentation_trace)],
    )


def _maybe_flush_and_report(
    *,
    resume_store: ResumableShardStore,
    run_context,
    counters: dict[str, object],
    pending_rows: list[dict[str, object]],
    active_workers: int,
    target_samples: int,
    last_progress_at_ref: list[float],
    progress_interval_seconds: float,
    quiet: bool,
    quarantine_out_path: Path,
) -> None:
    now = time.time()
    if pending_rows:
        rows_to_commit = list(pending_rows)
        pending_rows.clear()
        resume_store.commit(snapshot=_snapshot_from_counters(counters), sample_rows=rows_to_commit)
    if (
        now - last_progress_at_ref[0] >= progress_interval_seconds
        or int(counters["accepted_samples"]) >= target_samples
    ):
        progress_payload = {
            "attempted_samples": int(counters["next_sample_idx"]),
            "accepted_samples": int(counters["accepted_samples"]),
            "rejected_samples": int(counters["rejected_samples"]),
            "active_workers": active_workers,
            "failure_reason_counts": dict(counters["failure_reason_counts"]),
            "truncation_counts": dict(counters["truncation_counts"]),
            "full_render_system_histogram": {
                str(key): int(value)
                for key, value in dict(counters["full_render_system_histogram"]).items()
            },
            "accepted_system_histogram": {
                str(spine_cls): {
                    str(bucket): int(count)
                    for bucket, count in dict(bucket_counts).items()
                }
                for spine_cls, bucket_counts in dict(counters["accepted_system_histogram"]).items()
            },
            "truncated_output_system_histogram": {
                str(key): int(value)
                for key, value in dict(counters["truncated_output_system_histogram"]).items()
            },
            "preferred_5_6_counts": dict(counters["preferred_5_6_counts"]),
            "requested_target_bucket_histogram": {
                str(key): int(value)
                for key, value in dict(counters["requested_target_bucket_histogram"]).items()
            },
            "candidate_hit_counts": dict(counters["candidate_hit_counts"]),
            "retry_counts": dict(counters["retry_counts"]),
            "quarantined_source_count": len(counters["quarantined_sources"]),
            "terminal_timeout_crash_artifacts": int(counters["terminal_timeout_crash_artifacts"]),
            "terminal_process_expired_crash_artifacts": int(
                counters["terminal_process_expired_crash_artifacts"]
            ),
            "augmentation_outcome_counts": dict(counters["augmentation_outcome_counts"]),
            "augmentation_band_counts": dict(counters["augmentation_band_counts"]),
            "augmentation_branch_counts": dict(counters["augmentation_branch_counts"]),
        }
        progress_payload.update(_build_layout_summary(counters))
        write_json(run_context.progress_path, progress_payload)
        _write_quarantined_sources(
            quarantined_sources=counters["quarantined_sources"],  # type: ignore[arg-type]
            destination=quarantine_out_path,
        )
        last_progress_at_ref[0] = now
        if not quiet:
            print(
                "Progress: "
                f"{progress_payload['accepted_samples']}/{target_samples} accepted, "
                f"{progress_payload['attempted_samples']} attempted"
            )


def _snapshot_from_counters(counters: dict[str, object]) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        next_sample_idx=int(counters["next_sample_idx"]),
        accepted_samples=int(counters["accepted_samples"]),
        rejected_samples=int(counters["rejected_samples"]),
        failure_reason_counts=dict(counters["failure_reason_counts"]),
        truncation_counts=dict(counters["truncation_counts"]),
        full_render_system_histogram={
            str(key): int(value)
            for key, value in dict(counters["full_render_system_histogram"]).items()
        },
        accepted_system_histogram={
            str(spine_cls): {
                str(bucket): int(count)
                for bucket, count in dict(bucket_counts).items()
            }
            for spine_cls, bucket_counts in dict(counters["accepted_system_histogram"]).items()
        },
        truncated_output_system_histogram={
            str(key): int(value)
            for key, value in dict(counters["truncated_output_system_histogram"]).items()
        },
        preferred_5_6_counts=dict(counters["preferred_5_6_counts"]),
        bottom_whitespace_px_histogram={
            str(key): int(value)
            for key, value in dict(counters["bottom_whitespace_px_histogram"]).items()
        },
        top_whitespace_px_histogram={
            str(key): int(value)
            for key, value in dict(counters["top_whitespace_px_histogram"]).items()
        },
        content_height_px_histogram={
            str(key): int(value)
            for key, value in dict(counters["content_height_px_histogram"]).items()
        },
        terminal_timeout_crash_artifacts=int(counters["terminal_timeout_crash_artifacts"]),
        terminal_process_expired_crash_artifacts=int(
            counters["terminal_process_expired_crash_artifacts"]
        ),
        requested_target_bucket_histogram={
            str(key): int(value)
            for key, value in dict(counters["requested_target_bucket_histogram"]).items()
        },
        candidate_hit_counts=dict(counters["candidate_hit_counts"]),
        retry_counts=dict(counters["retry_counts"]),
        quarantined_sources=tuple(
            sorted(str(path) for path in counters["quarantined_sources"])  # type: ignore[arg-type]
        ),
        augmentation_outcome_counts=dict(counters["augmentation_outcome_counts"]),
        augmentation_band_counts=dict(counters["augmentation_band_counts"]),
        augmentation_branch_counts=dict(counters["augmentation_branch_counts"]),
    )


def _build_layout_summary(counters: dict[str, object]) -> dict[str, object]:
    bottom_histogram: Counter = counters["bottom_whitespace_px_histogram"]  # type: ignore[assignment]
    top_histogram: Counter = counters["top_whitespace_px_histogram"]  # type: ignore[assignment]
    content_histogram: Counter = counters["content_height_px_histogram"]  # type: ignore[assignment]
    page_height = 1485.0
    return {
        "layout_summary_version": 1,
        "layout_summary_population": "accepted_samples",
        "accepted_samples_total": int(counters["accepted_samples"]),
        "bottom_whitespace_px_stats": _summarize_histogram(bottom_histogram),
        "top_whitespace_px_stats": _summarize_histogram(top_histogram),
        "content_height_px_stats": _summarize_histogram(content_histogram),
        "bottom_whitespace_ratio_stats": _summarize_scaled_histogram(
            bottom_histogram,
            scale=page_height,
        ),
        "vertical_fill_ratio_stats": _summarize_scaled_histogram(
            content_histogram,
            scale=page_height,
        ),
    }


def _summarize_histogram(histogram: Counter) -> dict[str, float | int]:
    if not histogram:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    total = int(sum(int(count) for count in histogram.values()))
    weighted_sum = float(sum(int(value) * int(count) for value, count in histogram.items()))
    sorted_items = sorted((int(value), int(count)) for value, count in histogram.items())
    return {
        "count": total,
        "mean": weighted_sum / total,
        "min": float(sorted_items[0][0]),
        "max": float(sorted_items[-1][0]),
        "p50": _percentile_from_histogram(sorted_items, total, 50.0),
        "p95": _percentile_from_histogram(sorted_items, total, 95.0),
        "p99": _percentile_from_histogram(sorted_items, total, 99.0),
    }


def _summarize_scaled_histogram(histogram: Counter, *, scale: float) -> dict[str, float | int]:
    raw = _summarize_histogram(histogram)
    if raw["count"] == 0:
        return raw
    return {
        "count": raw["count"],
        "mean": float(raw["mean"]) / scale,
        "min": float(raw["min"]) / scale,
        "max": float(raw["max"]) / scale,
        "p50": float(raw["p50"]) / scale,
        "p95": float(raw["p95"]) / scale,
        "p99": float(raw["p99"]) / scale,
    }


def _percentile_from_histogram(
    sorted_items: list[tuple[int, int]],
    total_count: int,
    percentile: float,
) -> float:
    if total_count <= 0:
        return 0.0
    rank = max(1, int(round((percentile / 100.0) * total_count)))
    cumulative = 0
    for value, count in sorted_items:
        cumulative += count
        if cumulative >= rank:
            return float(value)
    return float(sorted_items[-1][0])


def _build_timeout_event(
    *,
    task: ScheduledTask,
    queue_wait_ms: float,
    will_retry: bool,
    dropped_pending_tasks: int = 0,
    crash_artifact_json_path: str | None = None,
    crash_repro_stage_count: int = 0,
) -> dict[str, object]:
    return {
        "event": "timeout",
        "timestamp": time.time(),
        "sample_id": task.plan.sample_id,
        "sample_idx": task.sample_idx,
        "source_paths": [str(segment.path) for segment in task.plan.segments],
        "retry_count": task.timeout_retries,
        "will_retry": will_retry,
        "queue_wait_ms": queue_wait_ms,
        "dropped_pending_tasks": dropped_pending_tasks,
        "crash_artifact_json_path": crash_artifact_json_path,
        "crash_repro_stage_count": int(crash_repro_stage_count),
    }


def _build_process_expired_event(
    *,
    task: ScheduledTask,
    queue_wait_ms: float,
    exc: ProcessExpired,
    will_retry: bool,
    dropped_pending_tasks: int = 0,
    crash_artifact_json_path: str | None = None,
    crash_repro_stage_count: int = 0,
) -> dict[str, object]:
    return {
        "event": "process_expired",
        "timestamp": time.time(),
        "sample_id": task.plan.sample_id,
        "sample_idx": task.sample_idx,
        "source_paths": [str(segment.path) for segment in task.plan.segments],
        "retry_count": task.expired_retries,
        "will_retry": will_retry,
        "queue_wait_ms": queue_wait_ms,
        "dropped_pending_tasks": dropped_pending_tasks,
        "exception_type": type(exc).__name__,
        "exception_repr": repr(exc),
        "exception_args": [repr(arg) for arg in getattr(exc, "args", ())],
        "pid": getattr(exc, "pid", None),
        "exitcode": getattr(exc, "exitcode", None),
        "crash_artifact_json_path": crash_artifact_json_path,
        "crash_repro_stage_count": int(crash_repro_stage_count),
    }


def _write_terminal_crash_artifact(
    *,
    run_context: RunContext,
    recipe: ProductionRecipe,
    task: ScheduledTask,
    event_type: str,
    retry_count: int,
    will_retry: bool,
    queue_wait_ms: float,
    dropped_pending_tasks: int,
    counters: dict[str, object],
    exc: ProcessExpired | None = None,
) -> tuple[Path, int]:
    if event_type == "timeout":
        counters["terminal_timeout_crash_artifacts"] = int(
            counters["terminal_timeout_crash_artifacts"]
        ) + 1
    else:
        counters["terminal_process_expired_crash_artifacts"] = int(
            counters["terminal_process_expired_crash_artifacts"]
        ) + 1
    exception_payload = _build_exception_payload(exc) if exc is not None else None
    return write_crash_artifact(
        run_context=run_context,
        recipe=recipe,
        plan=task.plan,
        sample_idx=task.sample_idx,
        event_type=event_type,
        retry_count=retry_count,
        will_retry=will_retry,
        queue_wait_ms=queue_wait_ms,
        dropped_pending_tasks=dropped_pending_tasks,
        target_bucket=task.target_bucket,
        planned_line_count=task.planned_line_count,
        candidate_in_target_range=task.candidate_in_target_range,
        exception_payload=exception_payload,
    )


def _build_exception_payload(exc: ProcessExpired | None) -> dict[str, object] | None:
    if exc is None:
        return None
    return {
        "exception_type": type(exc).__name__,
        "exception_repr": repr(exc),
        "exception_args": [repr(arg) for arg in getattr(exc, "args", ())],
        "pid": getattr(exc, "pid", None),
        "exitcode": getattr(exc, "exitcode", None),
    }
