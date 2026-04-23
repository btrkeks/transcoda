"""Attempt execution and ledger helpers for dataset generation."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from scripts.dataset_generation.dataset_generation.acceptance import decide_acceptance
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
)
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.renderer import SampleRenderContext
from scripts.dataset_generation.dataset_generation.types_domain import AttemptStageName, SamplePlan
from scripts.dataset_generation.dataset_generation.types_events import (
    FailureRenderAttempt,
    VerovioDiagnosticEvent,
)
from scripts.dataset_generation.dataset_generation.types_render import (
    AcceptanceDecision,
    RenderResult,
)


@dataclass(frozen=True)
class RenderAttemptPlan:
    stage: AttemptStageName
    seed: int
    render_transcription: str
    truncation_applied: bool
    chunk_count: int | None = None
    total_chunks: int | None = None
    ratio: float | None = None


@dataclass(frozen=True)
class ExecutedRenderAttempt:
    plan: RenderAttemptPlan
    render_result: RenderResult
    decision: AcceptanceDecision
    failure_attempt: FailureRenderAttempt
    verovio_events: tuple[VerovioDiagnosticEvent, ...]


@dataclass
class AttemptLedger:
    verovio_events: list[VerovioDiagnosticEvent] = field(default_factory=list)
    failure_attempts: list[FailureRenderAttempt] = field(default_factory=list)

    def record(self, attempt: ExecutedRenderAttempt) -> None:
        self.verovio_events.extend(attempt.verovio_events)
        self.failure_attempts.append(attempt.failure_attempt)

    def verovio_tuple(self) -> tuple[VerovioDiagnosticEvent, ...]:
        return tuple(self.verovio_events)

    def failure_tuple(self) -> tuple[FailureRenderAttempt, ...]:
        return tuple(self.failure_attempts)


def execute_render_attempt(
    *,
    sample_plan: SamplePlan,
    attempt_plan: RenderAttemptPlan,
    recipe: ProductionRecipe,
    renderer: VerovioRenderer,
    render_callable: Callable[..., RenderResult],
    capture_verovio_diagnostics: bool,
    context: SampleRenderContext | None = None,
) -> ExecutedRenderAttempt:
    render_result = _call_render(
        render_callable,
        attempt_plan.render_transcription,
        recipe,
        seed=attempt_plan.seed,
        renderer=renderer,
        capture_verovio_diagnostics=capture_verovio_diagnostics,
        context=context,
    )
    return finalize_render_attempt(
        sample_plan=sample_plan,
        attempt_plan=attempt_plan,
        recipe=recipe,
        render_result=render_result,
    )


def finalize_render_attempt(
    *,
    sample_plan: SamplePlan,
    attempt_plan: RenderAttemptPlan,
    recipe: ProductionRecipe,
    render_result: RenderResult,
) -> ExecutedRenderAttempt:
    decision = decide_acceptance(
        render_result,
        recipe,
        truncation_applied=attempt_plan.truncation_applied,
    )
    return ExecutedRenderAttempt(
        plan=attempt_plan,
        render_result=render_result,
        decision=decision,
        failure_attempt=_build_failure_attempt(attempt_plan=attempt_plan, render_result=render_result, decision=decision),
        verovio_events=_build_verovio_events(sample_plan=sample_plan, attempt_plan=attempt_plan, result=render_result),
    )


def _call_render(
    render_callable: Callable[..., RenderResult],
    render_transcription: str,
    recipe: ProductionRecipe,
    *,
    seed: int,
    renderer: VerovioRenderer,
    capture_verovio_diagnostics: bool,
    context: SampleRenderContext | None,
) -> RenderResult:
    kwargs = {
        "seed": seed,
        "renderer": renderer,
    }
    if _callable_accepts_parameter(render_callable, "capture_verovio_diagnostics"):
        kwargs["capture_verovio_diagnostics"] = capture_verovio_diagnostics
    if context is not None and _callable_accepts_parameter(render_callable, "context"):
        kwargs["context"] = context
    return render_callable(render_transcription, recipe, **kwargs)


def _callable_accepts_parameter(render_callable: Callable[..., object], name: str) -> bool:
    try:
        signature = inspect.signature(render_callable)
    except (TypeError, ValueError):
        return False
    if name in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _build_verovio_events(
    *,
    sample_plan: SamplePlan,
    attempt_plan: RenderAttemptPlan,
    result: RenderResult,
) -> tuple[VerovioDiagnosticEvent, ...]:
    sample_idx = int(sample_plan.sample_id.split("_")[-1])
    source_paths = tuple(str(Path(segment.path).resolve()) for segment in sample_plan.segments)
    return tuple(
        VerovioDiagnosticEvent(
            event="verovio_diagnostic",
            sample_id=sample_plan.sample_id,
            sample_idx=sample_idx,
            source_paths=source_paths,
            stage=attempt_plan.stage,
            seed=attempt_plan.seed,
            render_attempt_idx=diagnostic.render_attempt_idx,
            diagnostic_kind=diagnostic.diagnostic_kind,
            raw_message=diagnostic.raw_message,
            near_line=diagnostic.near_line,
            expected_duration_from_start=diagnostic.expected_duration_from_start,
            found_duration_from_start=diagnostic.found_duration_from_start,
            line_text=diagnostic.line_text,
            truncation_chunk_count=attempt_plan.chunk_count,
            truncation_total_chunks=attempt_plan.total_chunks,
            truncation_ratio=attempt_plan.ratio,
        )
        for diagnostic in result.verovio_diagnostics
    )


def _build_failure_attempt(
    *,
    attempt_plan: RenderAttemptPlan,
    render_result: RenderResult,
    decision: AcceptanceDecision,
) -> FailureRenderAttempt:
    return FailureRenderAttempt(
        stage=attempt_plan.stage,
        seed=attempt_plan.seed,
        chunk_count=attempt_plan.chunk_count,
        total_chunks=attempt_plan.total_chunks,
        ratio=attempt_plan.ratio,
        system_count=render_result.svg_diagnostics.system_count,
        page_count=render_result.svg_diagnostics.page_count,
        content_height_px=render_result.content_height_px,
        vertical_fill_ratio=render_result.vertical_fill_ratio,
        render_rejection_reason=render_result.rejection_reason,
        decision_reason=decision.reason,
        accepted=decision.action != "reject",
        verovio_diagnostic_count=len(render_result.verovio_diagnostics),
    )
