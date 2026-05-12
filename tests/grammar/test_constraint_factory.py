from __future__ import annotations

from src.grammar.constraint_factory import ConstrainedDecodingFactory
from src.grammar.interpretation_transition_rule import (
    InterpretationTransitionConfig,
    InterpretationTransitionRule,
)
from src.grammar.runaway_guard import RunawayBreakerLogitsProcessor, RunawayGuardConfig
from src.grammar.spine_structure_rule import SpineStructureRule
from src.grammar.stateful_kern_logits_processor import StatefulKernLogitsProcessor
from src.model.generation_policy import GenerationSettings


class _GrammarProviderStub:
    def __init__(self, sentinel):
        self.sentinel = sentinel

    def create_logits_processor(self, *, pad_token_id=None, collect_stats=False):
        assert pad_token_id == 0
        assert collect_stats is False
        return self.sentinel


def _settings(strategy: str = "beam", num_beams: int = 4) -> GenerationSettings:
    return GenerationSettings(
        strategy=strategy,
        num_beams=num_beams,
        length_penalty=1.0,
        repetition_penalty=1.1,
        early_stopping=True,
        num_return_sequences=1,
        use_cache=True,
        do_sample=False,
    )


def _guard_config() -> RunawayGuardConfig:
    return RunawayGuardConfig(
        max_same_control_token=8,
        max_control_lines_streak=6,
        max_spine_splits=64,
        max_spine_merges=96,
        max_ottava_markers=16,
        max_tuplet_markers=12,
        max_tremolo_markers=12,
    )


def test_factory_builds_grammar_spine_and_runaway_stack():
    sentinel = object()
    factory = ConstrainedDecodingFactory(
        grammar_provider=_GrammarProviderStub(sentinel),
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "\t", 4: "\n", 5: "4c", 6: "*^"},
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_interpretation_transition_constraints=True,
        use_spine_structure_constraints=True,
        use_rhythm_constraints=False,
        interpretation_transition_config=InterpretationTransitionConfig(),
        runaway_guard_enabled=True,
        runaway_guard_config=_guard_config(),
    )

    bundle = factory.build(_settings())

    assert bundle.generation_settings.strategy == "greedy"
    assert bundle.logits_processors is not None
    assert bundle.logits_processors[0] is sentinel
    assert isinstance(bundle.logits_processors[1], StatefulKernLogitsProcessor)
    assert isinstance(bundle.logits_processors[2], RunawayBreakerLogitsProcessor)
    assert bundle.stopping_criteria is None
    assert len(bundle.semantic_rule_factories) == 2
    assert isinstance(bundle.semantic_rule_factories[0](), InterpretationTransitionRule)
    assert bundle.semantic_rule_factories[0]().config == InterpretationTransitionConfig()
    assert bundle.semantic_rule_factories[1] is SpineStructureRule


def test_factory_skips_semantic_stack_when_grammar_is_disabled():
    factory = ConstrainedDecodingFactory(
        grammar_provider=None,
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "\t", 4: "\n", 5: "4c"},
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_interpretation_transition_constraints=False,
        use_spine_structure_constraints=True,
        use_rhythm_constraints=True,
        runaway_guard_enabled=True,
        runaway_guard_config=_guard_config(),
    )

    bundle = factory.build(_settings(strategy="beam", num_beams=4))

    assert bundle.logits_processors is None
    assert bundle.stopping_criteria is None
    assert bundle.generation_settings.strategy == "beam"
    assert bundle.generation_settings.num_beams == 4
    assert bundle.semantic_rule_factories == (SpineStructureRule,)
