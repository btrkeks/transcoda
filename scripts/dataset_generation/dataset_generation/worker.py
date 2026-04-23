"""Worker-side sample evaluation for production dataset generation."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from scripts.dataset_generation.dataset_generation.attempts import (
    AttemptLedger,
    ExecutedRenderAttempt,
    RenderAttemptPlan,
    execute_render_attempt,
    finalize_render_attempt,
)
from scripts.dataset_generation.dataset_generation.augmentation import augment_accepted_render
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
)
from scripts.dataset_generation.dataset_generation.io import append_jsonl, encode_jpeg_image
from scripts.dataset_generation.dataset_generation.policy import (
    finalize_failure_reason,
    is_in_preferred_band,
    layout_rescue_seed,
    should_attempt_layout_rescue,
)
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.records import build_dataset_row
from scripts.dataset_generation.dataset_generation.render_transcription import (
    ensure_render_header,
    materialize_render_transcription,
)
from scripts.dataset_generation.dataset_generation.renderer import (
    SampleRenderContext,
    prepare_sample_render_context,
    render_sample,
    render_sample_with_layout_rescue,
)
from scripts.dataset_generation.dataset_generation.truncation import (
    PrefixTruncationCandidate,
    TruncationProbeResult,
    classify_truncation_mode,
    find_best_truncation_candidate,
    validate_truncation_candidate_terminal_state,
)
from scripts.dataset_generation.dataset_generation.types_domain import AttemptStageName, SamplePlan
from scripts.dataset_generation.dataset_generation.types_events import (
    AugmentationPreviewArtifacts,
    AugmentationTraceEvent,
    VerovioDiagnosticEvent,
)
from scripts.dataset_generation.dataset_generation.types_outcomes import (
    WorkerFailure,
    WorkerOutcome,
    WorkerSuccess,
)
from scripts.dataset_generation.dataset_generation.types_render import (
    AcceptedSample,
    AugmentedRenderResult,
    RenderResult,
    SvgLayoutDiagnostics,
)

_WORKER_RECIPE: ProductionRecipe | None = None
_WORKER_RENDERER: VerovioRenderer | None = None
_WORKER_CAPTURE_VEROVIO_DIAGNOSTICS = True
_WORKER_STAGE_EVENTS_PATH: Path | None = None
_RATIONAL_DURATION_PATTERN = re.compile(r"\d%-?\d")


def compute_initial_kern_spine_count(transcription: str) -> int:
    for raw_line in transcription.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!!"):
            continue
        return raw_line.count("\t") + 1
    raise ValueError("Cannot infer initial kern spine count from empty transcription")


def is_valid_kern(content: str) -> tuple[bool, str | None]:
    if _RATIONAL_DURATION_PATTERN.search(content):
        return False, "rational duration (corrupted source)"
    return True, None


def ensure_kern_header(content: str) -> str:
    return ensure_render_header(content)


def init_generation_worker(
    recipe: ProductionRecipe,
    capture_verovio_diagnostics: bool = True,
    stage_events_path: str | Path | None = None,
) -> None:
    global _WORKER_RECIPE, _WORKER_RENDERER, _WORKER_CAPTURE_VEROVIO_DIAGNOSTICS, _WORKER_STAGE_EVENTS_PATH
    _WORKER_RECIPE = recipe
    _WORKER_RENDERER = VerovioRenderer()
    _WORKER_CAPTURE_VEROVIO_DIAGNOSTICS = capture_verovio_diagnostics
    _WORKER_STAGE_EVENTS_PATH = (
        Path(stage_events_path).expanduser().resolve() if stage_events_path is not None else None
    )


def process_sample_plan(plan: SamplePlan) -> WorkerOutcome:
    assert _WORKER_RECIPE is not None, "Worker recipe not initialized"
    assert _WORKER_RENDERER is not None, "Worker renderer not initialized"
    return evaluate_sample_plan(
        plan,
        recipe=_WORKER_RECIPE,
        renderer=_WORKER_RENDERER,
        capture_verovio_diagnostics=_WORKER_CAPTURE_VEROVIO_DIAGNOSTICS,
    )


def process_calibration_sample_plan(plan: SamplePlan) -> WorkerOutcome:
    assert _WORKER_RECIPE is not None, "Worker recipe not initialized"
    assert _WORKER_RENDERER is not None, "Worker renderer not initialized"
    return evaluate_sample_plan(
        plan,
        recipe=_WORKER_RECIPE,
        renderer=_WORKER_RENDERER,
        augment_fn=_skip_calibration_image_bytes,
        capture_verovio_diagnostics=_WORKER_CAPTURE_VEROVIO_DIAGNOSTICS,
    )


def _skip_calibration_image_bytes(
    plan: SamplePlan,
    render_result: RenderResult,
    recipe: ProductionRecipe,
) -> bytes:
    del plan, render_result, recipe
    return b""


def evaluate_sample_plan(
    plan: SamplePlan,
    *,
    recipe: ProductionRecipe,
    renderer: VerovioRenderer,
    render_fn: Callable[..., RenderResult] = render_sample,
    rescue_render_fn: Callable[..., RenderResult] | None = None,
    augment_fn: Callable[..., object] = augment_accepted_render,
    capture_verovio_diagnostics: bool = True,
    stage_events_path: str | Path | None = None,
) -> WorkerOutcome:
    attempt_ledger = AttemptLedger()
    resolved_stage_events_path = _resolve_stage_events_path(stage_events_path)
    render_context = prepare_sample_render_context(
        plan.label_transcription,
        recipe,
        seed=plan.seed,
    )
    render_transcription = materialize_render_transcription(
        plan.label_transcription,
        recipe,
        augmentation_plan=render_context.augmentation_plan,
        source_line_indices=tuple(range(len(plan.label_transcription.splitlines()))),
    )
    full_attempt = _execute_attempt(
        sample_plan=plan,
        attempt_ledger=attempt_ledger,
        attempt_plan=RenderAttemptPlan(
            stage=AttemptStageName.FULL,
            seed=plan.seed,
            render_transcription=render_transcription,
            truncation_applied=False,
        ),
        recipe=recipe,
        renderer=renderer,
        render_callable=render_fn,
        capture_verovio_diagnostics=capture_verovio_diagnostics,
        stage_events_path=resolved_stage_events_path,
        render_context=render_context,
    )
    full_render = full_attempt.render_result
    full_decision = full_attempt.decision
    truncation_mode = classify_truncation_mode(full_render.svg_diagnostics, recipe)
    full_render_system_count = full_render.svg_diagnostics.system_count
    full_render_content_height_px = full_render.content_height_px
    full_render_vertical_fill_ratio = full_render.vertical_fill_ratio
    full_render_rejection_reason = full_render.rejection_reason
    full_render_in_preferred_band = truncation_mode == "preferred" and is_in_preferred_band(
        full_render_system_count,
        recipe,
    )
    preferred_5_6_rescue_attempted = False
    preferred_5_6_rescue_succeeded = False
    truncation_attempted = False

    if full_decision.action == "accept_without_truncation":
        preferred_status = (
            "preferred_5_6_accepted_full"
            if truncation_mode == "preferred"
            and recipe.truncation.preferred_min_systems
            <= full_render_system_count
            <= recipe.truncation.preferred_max_systems
            else None
        )
        sample, aug_trace, aug_preview = _finalize_sample(
            plan=plan,
            render_result=full_render,
            transcription=plan.label_transcription,
            truncation_applied=False,
            truncation_ratio=None,
            truncation_reason=None,
            recipe=recipe,
            augment_fn=augment_fn,
        )
        return WorkerSuccess(
            sample=sample,
            truncation_attempted=False,
            truncation_rescued=False,
            full_render_system_count=full_render_system_count,
            full_render_content_height_px=full_render_content_height_px,
            full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
            full_render_rejection_reason=full_render_rejection_reason,
            accepted_render_system_count=full_render_system_count,
            preferred_5_6_rescue_attempted=False,
            preferred_5_6_rescue_succeeded=False,
            preferred_5_6_status=preferred_status,
            verovio_diagnostics=attempt_ledger.verovio_tuple(),
            augmentation_trace=aug_trace,
            augmentation_preview=aug_preview,
        )

    if should_attempt_layout_rescue(full_render, recipe):
        preferred_5_6_rescue_attempted = full_render_in_preferred_band
        rescue_attempt = _execute_layout_rescue_attempt(
            sample_plan=plan,
            attempt_ledger=attempt_ledger,
            base_seed=plan.seed,
            stage=AttemptStageName.FULL_LAYOUT_RESCUE,
            render_transcription=render_transcription,
            recipe=recipe,
            renderer=renderer,
            render_fn=render_fn,
            rescue_render_fn=rescue_render_fn,
            truncation_applied=False,
            capture_verovio_diagnostics=capture_verovio_diagnostics,
            stage_events_path=resolved_stage_events_path,
            render_context=render_context,
        )
        rescue_render = rescue_attempt.render_result
        rescue_decision = rescue_attempt.decision
        if rescue_decision.action == "accept_without_truncation":
            preferred_5_6_rescue_succeeded = full_render_in_preferred_band
            sample, aug_trace, aug_preview = _finalize_sample(
                plan=plan,
                render_result=rescue_render,
                transcription=plan.label_transcription,
                truncation_applied=False,
                truncation_ratio=None,
                truncation_reason=None,
                recipe=recipe,
                augment_fn=augment_fn,
            )
            return WorkerSuccess(
                sample=sample,
                truncation_attempted=False,
                truncation_rescued=False,
                full_render_system_count=full_render_system_count,
                full_render_content_height_px=full_render_content_height_px,
                full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
                full_render_rejection_reason=full_render_rejection_reason,
                accepted_render_system_count=rescue_render.svg_diagnostics.system_count,
                preferred_5_6_rescue_attempted=full_render_in_preferred_band,
                preferred_5_6_rescue_succeeded=full_render_in_preferred_band,
                preferred_5_6_status=(
                    "preferred_5_6_rescued" if full_render_in_preferred_band else None
                ),
                verovio_diagnostics=attempt_ledger.verovio_tuple(),
                augmentation_trace=aug_trace,
                augmentation_preview=aug_preview,
            )

    if truncation_mode in {"preferred", "required"}:
        search_result = find_best_truncation_candidate(
            plan.label_transcription,
            max_trials=recipe.truncation.max_candidate_trials,
            probe_candidate=lambda candidate: _probe_truncation_candidate(
                sample_plan=plan,
                attempt_ledger=attempt_ledger,
                candidate=candidate,
                recipe=recipe,
                renderer=renderer,
                render_fn=render_fn,
                rescue_render_fn=rescue_render_fn,
                capture_verovio_diagnostics=capture_verovio_diagnostics,
                stage_events_path=resolved_stage_events_path,
                render_context=render_context,
            ),
        )
        truncation_attempted = bool(search_result.probes)
        if (
            search_result.selected_candidate is not None
            and search_result.selected_probe is not None
            and search_result.selected_probe.render_result is not None
        ):
            return _build_truncated_success(
                plan=plan,
                recipe=recipe,
                augment_fn=augment_fn,
                render_result=search_result.selected_probe.render_result,
                transcription=search_result.selected_candidate.transcription,
                truncation_ratio=search_result.selected_candidate.ratio,
                truncation_attempted=truncation_attempted,
                full_render_system_count=full_render_system_count,
                full_render_content_height_px=full_render_content_height_px,
                full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
                full_render_rejection_reason=full_render_rejection_reason,
                preferred_5_6_rescue_attempted=preferred_5_6_rescue_attempted,
                preferred_5_6_rescue_succeeded=preferred_5_6_rescue_succeeded,
                preferred_5_6_status=(
                    "preferred_5_6_truncated" if full_render_in_preferred_band else None
                ),
                verovio_events=attempt_ledger.verovio_tuple(),
            )

    failure_reason = finalize_failure_reason(
        full_decision_reason=full_decision.reason,
        full_render_rejection_reason=full_render.rejection_reason,
        truncation_attempted=truncation_attempted,
        truncation_mode=truncation_mode,
    )
    return WorkerFailure(
        sample_id=plan.sample_id,
        failure_reason=failure_reason,
        truncation_attempted=truncation_attempted,
        truncation_rescued=False,
        truncation_mode=truncation_mode,
        full_render_system_count=full_render_system_count,
        full_render_content_height_px=full_render_content_height_px,
        full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
        full_render_rejection_reason=full_render_rejection_reason,
        accepted_render_system_count=None,
        preferred_5_6_rescue_attempted=preferred_5_6_rescue_attempted,
        preferred_5_6_rescue_succeeded=False,
        preferred_5_6_status=(
            "preferred_5_6_failed" if full_render_in_preferred_band else None
        ),
        failure_attempts=attempt_ledger.failure_tuple(),
        verovio_diagnostics=attempt_ledger.verovio_tuple(),
    )


def _build_truncated_success(
    *,
    plan: SamplePlan,
    recipe: ProductionRecipe,
    augment_fn: Callable[..., object],
    render_result: RenderResult,
    transcription: str,
    truncation_ratio: float,
    truncation_attempted: bool,
    full_render_system_count: int | None,
    full_render_content_height_px: int | None,
    full_render_vertical_fill_ratio: float | None,
    full_render_rejection_reason: str | None,
    preferred_5_6_rescue_attempted: bool,
    preferred_5_6_rescue_succeeded: bool,
    preferred_5_6_status: str | None,
    verovio_events: tuple[VerovioDiagnosticEvent, ...],
) -> WorkerSuccess:
    sample, aug_trace, aug_preview = _finalize_sample(
        plan=plan,
        render_result=render_result,
        transcription=transcription,
        truncation_applied=True,
        truncation_ratio=truncation_ratio,
        truncation_reason="system_count_policy",
        recipe=recipe,
        augment_fn=augment_fn,
    )
    return WorkerSuccess(
        sample=sample,
        truncation_attempted=truncation_attempted,
        truncation_rescued=True,
        full_render_system_count=full_render_system_count,
        full_render_content_height_px=full_render_content_height_px,
        full_render_vertical_fill_ratio=full_render_vertical_fill_ratio,
        full_render_rejection_reason=full_render_rejection_reason,
        accepted_render_system_count=render_result.svg_diagnostics.system_count,
        preferred_5_6_rescue_attempted=preferred_5_6_rescue_attempted,
        preferred_5_6_rescue_succeeded=preferred_5_6_rescue_succeeded,
        preferred_5_6_status=preferred_5_6_status,
        verovio_diagnostics=verovio_events,
        augmentation_trace=aug_trace,
        augmentation_preview=aug_preview,
    )


def _execute_attempt(
    *,
    sample_plan: SamplePlan,
    attempt_ledger: AttemptLedger,
    attempt_plan: RenderAttemptPlan,
    recipe: ProductionRecipe,
    renderer: VerovioRenderer,
    render_callable: Callable[..., RenderResult],
    capture_verovio_diagnostics: bool,
    stage_events_path: Path | None,
    render_context: SampleRenderContext | None = None,
) -> ExecutedRenderAttempt:
    stage_started_at = time.time()
    _write_worker_stage_event(
        sample_plan=sample_plan,
        attempt_plan=attempt_plan,
        phase="started",
        stage_events_path=stage_events_path,
        started_at=stage_started_at,
    )
    attempt = execute_render_attempt(
        sample_plan=sample_plan,
        attempt_plan=attempt_plan,
        recipe=recipe,
        renderer=renderer,
        render_callable=render_callable,
        capture_verovio_diagnostics=capture_verovio_diagnostics,
        context=render_context,
    )
    _write_worker_stage_event(
        sample_plan=sample_plan,
        attempt_plan=attempt_plan,
        phase="completed",
        stage_events_path=stage_events_path,
        started_at=stage_started_at,
        attempt=attempt,
    )
    attempt_ledger.record(attempt)
    return attempt


def _probe_truncation_candidate(
    *,
    sample_plan: SamplePlan,
    attempt_ledger: AttemptLedger,
    candidate: PrefixTruncationCandidate,
    recipe: ProductionRecipe,
    renderer: VerovioRenderer,
    render_fn: Callable[..., RenderResult],
    rescue_render_fn: Callable[..., RenderResult] | None,
    capture_verovio_diagnostics: bool,
    stage_events_path: Path | None,
    render_context: SampleRenderContext,
) -> TruncationProbeResult:
    candidate_seed = (sample_plan.seed + candidate.chunk_count * 17) & 0xFFFFFFFF
    candidate_render_transcription = materialize_render_transcription(
        candidate.transcription,
        recipe,
        augmentation_plan=render_context.augmentation_plan,
        source_line_indices=candidate.origin_line_indices,
    )
    attempt_plan = RenderAttemptPlan(
        stage=AttemptStageName.TRUNCATION_CANDIDATE,
        seed=candidate_seed,
        render_transcription=candidate_render_transcription,
        truncation_applied=True,
        chunk_count=candidate.chunk_count,
        total_chunks=candidate.total_chunks,
        ratio=candidate.ratio,
    )
    validation_reason = validate_truncation_candidate_terminal_state(
        candidate_render_transcription
    )
    if validation_reason is not None:
        structural_attempt = _record_structural_rejection_attempt(
            sample_plan=sample_plan,
            attempt_ledger=attempt_ledger,
            attempt_plan=attempt_plan,
            recipe=recipe,
            rejection_reason=validation_reason,
            stage_events_path=stage_events_path,
        )
        return _build_probe_result(candidate, structural_attempt)

    candidate_attempt = _execute_attempt(
        sample_plan=sample_plan,
        attempt_ledger=attempt_ledger,
        attempt_plan=attempt_plan,
        recipe=recipe,
        renderer=renderer,
        render_callable=render_fn,
        capture_verovio_diagnostics=capture_verovio_diagnostics,
        stage_events_path=stage_events_path,
        render_context=render_context,
    )
    if candidate_attempt.decision.action == "accept_with_truncation":
        return _build_probe_result(candidate, candidate_attempt)

    candidate_render = candidate_attempt.render_result
    if should_attempt_layout_rescue(candidate_render, recipe):
        rescued_candidate_attempt = _execute_layout_rescue_attempt(
            sample_plan=sample_plan,
            attempt_ledger=attempt_ledger,
            base_seed=candidate_seed,
            stage=AttemptStageName.TRUNCATION_CANDIDATE_LAYOUT_RESCUE,
            render_transcription=candidate_render_transcription,
            recipe=recipe,
            renderer=renderer,
            render_fn=render_fn,
            rescue_render_fn=rescue_render_fn,
            truncation_applied=True,
            chunk_count=candidate.chunk_count,
            total_chunks=candidate.total_chunks,
            ratio=candidate.ratio,
            capture_verovio_diagnostics=capture_verovio_diagnostics,
            stage_events_path=stage_events_path,
            render_context=render_context,
        )
        if rescued_candidate_attempt.decision.action == "accept_with_truncation":
            return _build_probe_result(candidate, rescued_candidate_attempt)
        return _build_probe_result(candidate, rescued_candidate_attempt)

    return _build_probe_result(candidate, candidate_attempt)


def _record_structural_rejection_attempt(
    *,
    sample_plan: SamplePlan,
    attempt_ledger: AttemptLedger,
    attempt_plan: RenderAttemptPlan,
    recipe: ProductionRecipe,
    rejection_reason: str,
    stage_events_path: Path | None,
) -> ExecutedRenderAttempt:
    stage_started_at = time.time()
    _write_worker_stage_event(
        sample_plan=sample_plan,
        attempt_plan=attempt_plan,
        phase="started",
        stage_events_path=stage_events_path,
        started_at=stage_started_at,
    )
    attempt = finalize_render_attempt(
        sample_plan=sample_plan,
        attempt_plan=attempt_plan,
        recipe=recipe,
        render_result=RenderResult(
            image=None,
            render_layers=None,
            svg_diagnostics=SvgLayoutDiagnostics(system_count=0, page_count=0),
            bottom_whitespace_ratio=None,
            vertical_fill_ratio=None,
            rejection_reason=rejection_reason,
        ),
    )
    _write_worker_stage_event(
        sample_plan=sample_plan,
        attempt_plan=attempt_plan,
        phase="completed",
        stage_events_path=stage_events_path,
        started_at=stage_started_at,
        attempt=attempt,
    )
    attempt_ledger.record(attempt)
    return attempt


def _build_probe_result(
    candidate: PrefixTruncationCandidate,
    attempt: ExecutedRenderAttempt,
) -> TruncationProbeResult:
    return TruncationProbeResult(
        candidate=candidate,
        accepted=attempt.decision.action == "accept_with_truncation",
        rejection_reason=attempt.render_result.rejection_reason,
        decision_reason=attempt.decision.reason,
        render_result=(
            attempt.render_result if attempt.decision.action == "accept_with_truncation" else None
        ),
    )


def _execute_layout_rescue_attempt(
    *,
    sample_plan: SamplePlan,
    attempt_ledger: AttemptLedger,
    base_seed: int,
    stage: AttemptStageName,
    render_transcription: str,
    recipe: ProductionRecipe,
    renderer: VerovioRenderer,
    render_fn: Callable[..., RenderResult],
    rescue_render_fn: Callable[..., RenderResult] | None,
    truncation_applied: bool,
    chunk_count: int | None = None,
    total_chunks: int | None = None,
    ratio: float | None = None,
    capture_verovio_diagnostics: bool,
    stage_events_path: Path | None,
    render_context: SampleRenderContext | None = None,
) -> ExecutedRenderAttempt:
    rescue_callable = rescue_render_fn
    if rescue_callable is None and render_fn is render_sample:
        rescue_callable = render_sample_with_layout_rescue
    return _execute_attempt(
        sample_plan=sample_plan,
        attempt_ledger=attempt_ledger,
        attempt_plan=RenderAttemptPlan(
            stage=stage,
            seed=layout_rescue_seed(base_seed),
            render_transcription=render_transcription,
            truncation_applied=truncation_applied,
            chunk_count=chunk_count,
            total_chunks=total_chunks,
            ratio=ratio,
        ),
        recipe=recipe,
        renderer=renderer,
        render_callable=rescue_callable or render_fn,
        capture_verovio_diagnostics=capture_verovio_diagnostics,
        stage_events_path=stage_events_path,
        render_context=render_context,
    )


def _finalize_sample(
    *,
    plan: SamplePlan,
    render_result: RenderResult,
    transcription: str,
    truncation_applied: bool,
    truncation_ratio: float | None,
    truncation_reason: str | None,
    recipe: ProductionRecipe,
    augment_fn: Callable[..., object],
) -> tuple[AcceptedSample, AugmentationTraceEvent | None, AugmentationPreviewArtifacts | None]:
    augmented_result = augment_fn(plan, render_result, recipe)
    augmentation_preview: AugmentationPreviewArtifacts | None = None
    if isinstance(augmented_result, AugmentedRenderResult):
        augmented_image = augmented_result.final_image
        aug_trace = augmented_result.trace
        if (
            isinstance(augmented_result.final_image, np.ndarray)
            and isinstance(augmented_result.base_image, np.ndarray)
            and isinstance(augmented_result.pre_augraphy_image, np.ndarray)
        ):
            augmentation_preview = AugmentationPreviewArtifacts(
                base_image_jpeg=encode_jpeg_image(augmented_result.base_image),
                pre_augraphy_image_jpeg=encode_jpeg_image(augmented_result.pre_augraphy_image),
                final_image_jpeg=encode_jpeg_image(augmented_result.final_image),
            )
    elif isinstance(augmented_result, tuple):
        augmented_image, aug_trace = augmented_result
    else:
        augmented_image = augmented_result
        aug_trace = None
    if not isinstance(augmented_image, bytes):
        image_bytes = encode_jpeg_image(augmented_image)
    else:
        image_bytes = augmented_image
    sample = AcceptedSample(
        sample_id=plan.sample_id,
        label_transcription=transcription,
        image_bytes=image_bytes,
        initial_kern_spine_count=compute_initial_kern_spine_count(transcription),
        segment_count=plan.segment_count,
        source_ids=tuple(segment.source_id for segment in plan.segments),
        source_measure_count=plan.source_measure_count,
        source_non_empty_line_count=plan.source_non_empty_line_count,
        system_count=render_result.svg_diagnostics.system_count,
        truncation_applied=truncation_applied,
        truncation_reason=truncation_reason,
        truncation_ratio=truncation_ratio,
        bottom_whitespace_ratio=render_result.bottom_whitespace_ratio,
        vertical_fill_ratio=render_result.vertical_fill_ratio,
        bottom_whitespace_px=render_result.bottom_whitespace_px,
        top_whitespace_px=render_result.top_whitespace_px,
        content_height_px=render_result.content_height_px,
    )
    return sample, aug_trace, augmentation_preview


def outcome_to_dataset_row(
    outcome: WorkerSuccess,
    *,
    recipe: ProductionRecipe,
) -> dict[str, object]:
    return build_dataset_row(outcome.sample, recipe=recipe)


def _resolve_stage_events_path(stage_events_path: str | Path | None) -> Path | None:
    if stage_events_path is not None:
        return Path(stage_events_path).expanduser().resolve()
    return _WORKER_STAGE_EVENTS_PATH


def _write_worker_stage_event(
    *,
    sample_plan: SamplePlan,
    attempt_plan: RenderAttemptPlan,
    phase: str,
    stage_events_path: Path | None,
    started_at: float,
    attempt: ExecutedRenderAttempt | None = None,
) -> None:
    if stage_events_path is None:
        return
    sample_idx = int(sample_plan.sample_id.split("_")[-1])
    event: dict[str, object] = {
        "event": "worker_stage",
        "timestamp": time.time(),
        "sample_id": sample_plan.sample_id,
        "sample_idx": sample_idx,
        "source_paths": [str(Path(segment.path).resolve()) for segment in sample_plan.segments],
        "stage": str(attempt_plan.stage),
        "phase": phase,
        "seed": attempt_plan.seed,
        "pid": os.getpid(),
        "truncation_applied": attempt_plan.truncation_applied,
        "truncation_chunk_count": attempt_plan.chunk_count,
        "truncation_total_chunks": attempt_plan.total_chunks,
        "truncation_ratio": attempt_plan.ratio,
    }
    if attempt is not None:
        event.update(
            {
                "duration_ms": max(0.0, (time.time() - started_at) * 1000.0),
                "accepted": attempt.decision.action != "reject",
                "decision_action": attempt.decision.action,
                "decision_reason": attempt.decision.reason,
                "system_count": attempt.render_result.svg_diagnostics.system_count,
                "page_count": attempt.render_result.svg_diagnostics.page_count,
                "content_height_px": attempt.render_result.content_height_px,
                "vertical_fill_ratio": attempt.render_result.vertical_fill_ratio,
                "render_rejection_reason": attempt.render_result.rejection_reason,
            }
        )
    append_jsonl(stage_events_path, [event])
