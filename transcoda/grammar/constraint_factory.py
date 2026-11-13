"""Factory for assembling constrained-decoding runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from transcoda.model.generation_policy import GenerationSettings, enforce_constraint_safe_settings

from .dead_end_lookahead import DeadEndLookaheadLogitsProcessor, ForkableConstraint
from .line_loop_guard import LineLoopStoppingCriteria
from .spine_structure_rule import (
    BarlineWidthMode,
    EmptyFieldMode,
    SpineStructureRuleFactory,
)
from .stateful_kern_logits_processor import SemanticRuleFactory, StatefulKernLogitsProcessor
from .timeline_rule import TimelineRuleFactory


@dataclass(frozen=True)
class ConstraintBundle:
    """Assembled generation-time constraint components."""

    logits_processors: list[Any] | None
    stopping_criteria: list[Any] | None
    generation_settings: GenerationSettings
    semantic_rule_factories: tuple[SemanticRuleFactory, ...]
    active_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintBuildContext:
    """Shared inputs used by constraint providers."""

    i2w: dict[int, str]
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    collect_stats: bool


@dataclass(frozen=True)
class ConstraintProviderResult:
    """Partial constraint stack contribution from one provider."""

    logits_processors: tuple[Any, ...] = ()
    stopping_criteria: tuple[Any, ...] = ()
    semantic_rule_factories: tuple[SemanticRuleFactory, ...] = ()
    active_constraints: tuple[str, ...] = ()
    has_generation_constraint: bool = False


class ConstraintProvider(Protocol):
    """Provider for one independent validation decode constraint."""

    def build(self, context: ConstraintBuildContext) -> ConstraintProviderResult:
        ...


class GrammarConstraintProvider:
    """Provider for grammar-constrained decoding."""

    def __init__(self, grammar_provider: Any | None) -> None:
        self._grammar_provider = grammar_provider

    def build(self, context: ConstraintBuildContext) -> ConstraintProviderResult:
        if self._grammar_provider is None:
            return ConstraintProviderResult()
        return ConstraintProviderResult(
            logits_processors=(
                self._grammar_provider.create_logits_processor(
                    pad_token_id=context.pad_token_id,
                    collect_stats=context.collect_stats,
                ),
            ),
            active_constraints=("grammar",),
            has_generation_constraint=True,
        )


class SpineStructureConstraintProvider:
    """Provider for active-spine-width invariants."""

    def __init__(
        self,
        *,
        enabled: bool,
        barline_width_mode: str = BarlineWidthMode.PER_SPINE,
        empty_field_mode: str = EmptyFieldMode.FORBID,
    ) -> None:
        self._enabled = bool(enabled)
        spine_rule_factory = SpineStructureRuleFactory(
            barline_width_mode=barline_width_mode,
            empty_field_mode=empty_field_mode,
        )
        # Sounding columns hold empty fields, so the timeline needs the data-only dialect.
        self._rule_factories: tuple[SemanticRuleFactory, ...] = (spine_rule_factory,)
        self._active_constraints: tuple[str, ...] = ("spine_structure",)
        if spine_rule_factory.empty_field_mode is EmptyFieldMode.DATA_ONLY:
            self._rule_factories += (TimelineRuleFactory(),)
            self._active_constraints += ("timeline",)

    def build(self, context: ConstraintBuildContext) -> ConstraintProviderResult:
        if not self._enabled:
            return ConstraintProviderResult()

        semantic_rule_factories = self._rule_factories
        return ConstraintProviderResult(
            logits_processors=(
                StatefulKernLogitsProcessor(
                    i2w=context.i2w,
                    bos_token_id=context.bos_token_id,
                    eos_token_id=context.eos_token_id,
                    pad_token_id=context.pad_token_id,
                    rule_factories=semantic_rule_factories,
                    collect_stats=context.collect_stats,
                ),
            ),
            semantic_rule_factories=semantic_rule_factories,
            active_constraints=self._active_constraints,
            has_generation_constraint=True,
        )


class LineLoopGuardConstraintProvider:
    """Provider for the hard identical-line runaway stop."""

    def __init__(self, *, max_identical_line_run: int | None) -> None:
        self._max_identical_line_run = max_identical_line_run

    def build(self, context: ConstraintBuildContext) -> ConstraintProviderResult:
        if self._max_identical_line_run is None:
            return ConstraintProviderResult()
        return ConstraintProviderResult(
            stopping_criteria=(
                LineLoopStoppingCriteria(
                    i2w=context.i2w,
                    max_identical_line_run=self._max_identical_line_run,
                    bos_token_id=context.bos_token_id,
                    eos_token_id=context.eos_token_id,
                    pad_token_id=context.pad_token_id,
                ),
            ),
            active_constraints=("line_loop_guard",),
            # Incremental per-row line state requires stable row identity, so
            # force the greedy constrained mode like the other providers.
            has_generation_constraint=True,
        )


class ConstraintStack:
    """Compose independent constraint providers into one generation bundle."""

    def __init__(self, providers: list[ConstraintProvider], context: ConstraintBuildContext) -> None:
        self._providers = list(providers)
        self._context = context

    def build(self, settings: GenerationSettings) -> ConstraintBundle:
        logits_processors: list[Any] = []
        stopping_criteria: list[Any] = []
        semantic_rule_factories: list[SemanticRuleFactory] = []
        active_constraints: list[str] = []
        has_generation_constraint = False

        for provider in self._providers:
            result = provider.build(self._context)
            logits_processors.extend(result.logits_processors)
            stopping_criteria.extend(result.stopping_criteria)
            semantic_rule_factories.extend(result.semantic_rule_factories)
            active_constraints.extend(result.active_constraints)
            has_generation_constraint = (
                has_generation_constraint or result.has_generation_constraint
            )

        generation_settings = enforce_constraint_safe_settings(
            settings,
            has_constraints=has_generation_constraint,
        )
        # Constraints force greedy decoding, so the lookahead checks the argmax.
        forkable = [p for p in logits_processors if isinstance(p, ForkableConstraint)]
        if forkable:
            logits_processors.append(
                DeadEndLookaheadLogitsProcessor(
                    forkable,
                    field_end_token_ids=self._field_end_token_ids(),
                    pad_token_id=self._context.pad_token_id,
                )
            )

        return ConstraintBundle(
            logits_processors=logits_processors or None,
            stopping_criteria=stopping_criteria or None,
            generation_settings=generation_settings,
            semantic_rule_factories=tuple(semantic_rule_factories),
            active_constraints=tuple(active_constraints),
        )

    def _field_end_token_ids(self) -> list[int]:
        token_ids = [
            int(token_id) for token_id, text in self._context.i2w.items() if text in {"\t", "\n"}
        ]
        if self._context.eos_token_id is not None:
            token_ids.append(int(self._context.eos_token_id))
        return token_ids


class ConstrainedDecodingFactory:
    """Build a consistent constrained-decoding stack for inference."""

    def __init__(
        self,
        *,
        grammar_provider: Any | None,
        i2w: dict[int, str],
        bos_token_id: int | None,
        eos_token_id: int | None,
        pad_token_id: int | None,
        use_spine_structure_constraints: bool,
        spine_structure_barline_width: str = BarlineWidthMode.PER_SPINE,
        spine_structure_empty_field_mode: str = EmptyFieldMode.FORBID,
        max_identical_line_run: int | None = None,
        collect_stats: bool = False,
    ) -> None:
        self._grammar_provider = grammar_provider
        self._i2w = i2w
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._use_spine_structure_constraints = bool(use_spine_structure_constraints)
        self._spine_structure_rule_factory = SpineStructureRuleFactory(
            barline_width_mode=spine_structure_barline_width,
            empty_field_mode=spine_structure_empty_field_mode,
        )
        self._collect_stats = bool(collect_stats)
        self._stack = ConstraintStack(
            providers=[
                GrammarConstraintProvider(grammar_provider),
                SpineStructureConstraintProvider(
                    enabled=use_spine_structure_constraints,
                    barline_width_mode=spine_structure_barline_width,
                    empty_field_mode=spine_structure_empty_field_mode,
                ),
                LineLoopGuardConstraintProvider(
                    max_identical_line_run=max_identical_line_run,
                ),
            ],
            context=ConstraintBuildContext(
                i2w=i2w,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                collect_stats=collect_stats,
            ),
        )

    def build(self, settings: GenerationSettings) -> ConstraintBundle:
        return self._stack.build(settings)

    def _resolve_semantic_rule_factories(self) -> list[SemanticRuleFactory]:
        structured_rule_factories: list[SemanticRuleFactory] = []
        if self._use_spine_structure_constraints:
            structured_rule_factories.append(self._spine_structure_rule_factory)
        return structured_rule_factories
