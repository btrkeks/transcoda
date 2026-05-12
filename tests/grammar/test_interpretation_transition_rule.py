from __future__ import annotations

from functools import partial

import torch

from src.grammar.interpretation_transition_rule import (
    InterpretationTransitionConfig,
    InterpretationTransitionRule,
)
from src.grammar.kern_prefix_state import KernPrefixState
from src.grammar.semantic_sequence_finalizer import finalize_kern_sequence_text
from src.grammar.stateful_kern_logits_processor import (
    StatefulKernLogitsProcessor,
    TokenizerConstraintContext,
)


def _i2w() -> dict[int, str]:
    return {
        0: "<pad>",
        1: "<bos>",
        2: "<eos>",
        3: "\t",
        4: "\n",
        5: "*",
        6: "*clefG2",
        7: "*^",
        8: "*v",
        9: "*-",
        10: "=1",
        11: ".",
        12: "4c",
        13: "2r",
        14: "*M4/4",
    }


def _context() -> TokenizerConstraintContext:
    return TokenizerConstraintContext.from_i2w(
        i2w=_i2w(),
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def test_tokenizer_context_builds_interpretation_buckets() -> None:
    context = _context()

    assert context.interpretation.interpretation_token_ids == [5, 6, 7, 8, 9, 14]
    assert context.interpretation.spine_op_token_ids == [5, 7, 8, 9]
    assert context.interpretation.non_spine_interp_token_ids == [6, 14]
    assert context.interpretation.null_interpretation_token_ids == [5]
    assert context.interpretation.clef_token_ids == [6]
    assert context.interpretation.barline_token_ids == [10]
    assert context.interpretation.non_control_data_token_ids == [11, 12, 13]


def test_rule_tracks_line_types_and_topology_age() -> None:
    rule = InterpretationTransitionRule()

    rule.on_line_closed(("4c", "."))
    assert rule.active_spines == 2
    assert rule.last_closed_line_type == "data"
    assert rule.last_line_had_dot_fields is True
    assert rule.recent_spine_topology_change_age is None

    rule.on_line_closed(("*^", "*"))
    assert rule.active_spines == 3
    assert rule.previous_closed_line_type == "data"
    assert rule.last_closed_line_type == "spine_op"
    assert rule.recent_spine_topology_change_age == 0

    rule.on_line_closed(("*", "*clefG2", "*"))
    assert rule.last_closed_line_type == "interp"
    assert rule.previous_closed_line_type == "spine_op"
    assert rule.recent_spine_topology_change_age == 1

    rule.on_line_closed(("4c", "4c", "2r"))
    assert rule.last_closed_line_type == "data"
    assert rule.recent_spine_topology_change_age == 2

    rule.on_line_closed(("4c", "4c", "4c"))
    assert rule.recent_spine_topology_change_age is None


def test_transition_context_biases_interpretation_start_and_preserves_masks() -> None:
    rule = InterpretationTransitionRule()
    rule.active_spines = 2
    rule.last_closed_line_type = "spine_op"
    prefix_state = KernPrefixState()
    context = _context()
    scores = torch.full((15,), float("-inf"))
    scores[5] = 0.1
    scores[6] = 0.0
    scores[7] = 0.3
    scores[9] = -0.2
    scores[10] = 0.2
    scores[12] = 0.4
    scores[13] = 0.0
    scores[14] = float("-inf")

    rule.mask_scores(prefix_state, scores, context)

    assert scores[6] == 1.5
    assert scores[5] == 1.1
    assert scores[12] == -0.6
    assert scores[10] == -0.8
    assert scores[7] == 0.3
    assert scores[9] == -0.2
    assert scores[14] == float("-inf")

    stats = rule.stats()
    assert stats["transition_context_activations"] == 1
    assert stats["first_field_interp_bias_applied"] == 1
    assert stats["first_field_bias_flipped_top_choice"] == 1


def test_rule_does_not_bias_single_spine_or_non_transition_context() -> None:
    context = _context()
    prefix_state = KernPrefixState()
    base_scores = torch.arange(15, dtype=torch.float32)

    single_spine_rule = InterpretationTransitionRule(active_spines=1, last_closed_line_type="spine_op")
    single_scores = base_scores.clone()
    single_spine_rule.mask_scores(prefix_state, single_scores, context)
    assert torch.equal(single_scores, base_scores)

    ordinary_rule = InterpretationTransitionRule(active_spines=3, last_closed_line_type="data")
    ordinary_scores = base_scores.clone()
    ordinary_rule.mask_scores(prefix_state, ordinary_scores, context)
    assert torch.equal(ordinary_scores, base_scores)


def test_rule_stops_biasing_after_line_commits_to_data() -> None:
    rule = InterpretationTransitionRule(active_spines=2, last_closed_line_type="spine_op")
    context = _context()
    prefix_state = KernPrefixState()
    prefix_state.append_text("4c")
    rule.on_text_appended(prefix_state)
    scores = torch.arange(15, dtype=torch.float32)

    rule.mask_scores(prefix_state, scores, context)

    assert rule.line_mode == "data"
    assert torch.equal(scores, torch.arange(15, dtype=torch.float32))
    assert rule.stats()["first_field_started_as_data"] == 1


def test_regression_like_context_prefers_interpretation_start() -> None:
    processor = StatefulKernLogitsProcessor(
        i2w=_i2w(),
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        rule_factories=[
            partial(
                InterpretationTransitionRule,
                config=InterpretationTransitionConfig(),
            )
        ],
        collect_stats=True,
    )

    prefix_ids = [1, 12, 3, 13, 4, 5, 3, 7, 4]
    scores = None
    for end in range(1, len(prefix_ids) + 1):
        input_ids = torch.tensor([prefix_ids[:end]], dtype=torch.long)
        base_scores = torch.zeros((1, 15), dtype=torch.float32)
        base_scores[0, 12] = 0.4
        base_scores[0, 6] = 0.0
        scores = processor(input_ids, base_scores)

    assert scores is not None
    assert scores[0, 6] > scores[0, 12]

    stats = processor.stats()
    assert stats["rules"]["transition_context_activations"] == 1
    assert stats["rules"]["first_field_interp_bias_applied"] == 1


def test_finalizer_accepts_configured_rule_factories() -> None:
    result = finalize_kern_sequence_text(
        text="*clefG2\t*clefG2\n4c\t4c\n",
        saw_eos=False,
        hit_max_length=False,
        rule_factories=(
            partial(
                InterpretationTransitionRule,
                config=InterpretationTransitionConfig(),
            ),
        ),
    )

    assert result.text == "*clefG2\t*clefG2\n4c\t4c\n*-\t*-"
    assert result.appended_terminator is True
