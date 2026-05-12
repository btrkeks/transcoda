"""Factory for assembling constrained-decoding runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from src.model.generation_policy import GenerationSettings, enforce_constraint_safe_settings

from .interpretation_transition_rule import (
    InterpretationTransitionConfig,
    InterpretationTransitionRule,
)
from .runaway_guard import RunawayBreakerLogitsProcessor, RunawayGuardConfig
from .spine_structure_rule import SpineStructureRule
from .stateful_kern_logits_processor import SemanticRuleFactory, StatefulKernLogitsProcessor


@dataclass(frozen=True)
class ConstraintBundle:
    """Assembled generation-time constraint components."""

    logits_processors: list[Any] | None
    stopping_criteria: list[Any] | None
    generation_settings: GenerationSettings
    semantic_rule_factories: tuple[SemanticRuleFactory, ...]


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
        use_interpretation_transition_constraints: bool,
        use_spine_structure_constraints: bool,
        use_rhythm_constraints: bool,
        interpretation_transition_config: InterpretationTransitionConfig | None = None,
        runaway_guard_enabled: bool = False,
        runaway_guard_config: RunawayGuardConfig | None = None,
        collect_stats: bool = False,
    ) -> None:
        self._grammar_provider = grammar_provider
        self._i2w = i2w
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._use_interpretation_transition_constraints = bool(
            use_interpretation_transition_constraints
        )
        self._use_spine_structure_constraints = bool(use_spine_structure_constraints)
        # Backward-compatible config surface only; inference no longer attaches RhythmRule.
        self._use_rhythm_constraints = bool(use_rhythm_constraints)
        self._interpretation_transition_config = (
            interpretation_transition_config or InterpretationTransitionConfig()
        )
        self._runaway_guard_enabled = bool(runaway_guard_enabled)
        self._runaway_guard_config = runaway_guard_config
        self._collect_stats = bool(collect_stats)

    def build(self, settings: GenerationSettings) -> ConstraintBundle:
        logits_processors: list[Any] = []
        stopping_criteria: list[Any] = []
        has_constraints = self._grammar_provider is not None
        structured_rule_factories = self._resolve_semantic_rule_factories()

        if self._grammar_provider is not None:
            logits_processors.append(
                self._grammar_provider.create_logits_processor(
                    pad_token_id=self._pad_token_id,
                    collect_stats=self._collect_stats,
                )
            )

            if structured_rule_factories:
                logits_processors.append(
                    StatefulKernLogitsProcessor(
                        i2w=self._i2w,
                        bos_token_id=self._bos_token_id,
                        eos_token_id=self._eos_token_id,
                        pad_token_id=self._pad_token_id,
                        rule_factories=structured_rule_factories,
                        collect_stats=self._collect_stats,
                    )
                )

            if self._runaway_guard_enabled and self._runaway_guard_config is not None:
                logits_processors.append(
                    RunawayBreakerLogitsProcessor(
                        tokenizer_i2w=self._i2w,
                        bos_token_id=int(self._bos_token_id),
                        eos_token_id=int(self._eos_token_id),
                        config=self._runaway_guard_config,
                    )
                )

        generation_settings = enforce_constraint_safe_settings(
            settings,
            has_constraints=has_constraints,
        )

        return ConstraintBundle(
            logits_processors=logits_processors or None,
            stopping_criteria=stopping_criteria or None,
            generation_settings=generation_settings,
            semantic_rule_factories=tuple(structured_rule_factories),
        )

    def _resolve_semantic_rule_factories(self) -> list[SemanticRuleFactory]:
        structured_rule_factories: list[SemanticRuleFactory] = []
        if self._use_interpretation_transition_constraints:
            structured_rule_factories.append(
                partial(
                    InterpretationTransitionRule,
                    config=self._interpretation_transition_config,
                )
            )
        if self._use_spine_structure_constraints:
            structured_rule_factories.append(SpineStructureRule)
        # RhythmRule is intentionally preserved in the repo but disabled in the
        # shared inference-time constraint stack.
        return structured_rule_factories
