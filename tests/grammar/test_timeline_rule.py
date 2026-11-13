from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from transcoda.grammar.constraint_factory import ConstrainedDecodingFactory
from transcoda.grammar.provider import GrammarProvider
from transcoda.grammar.semantic_sequence_finalizer import finalize_kern_sequence_text
from transcoda.grammar.spine_structure_rule import SpineStructureRuleFactory
from transcoda.grammar.stateful_kern_logits_processor import StatefulKernLogitsProcessor
from transcoda.grammar.timeline_rule import (
    TimelineRuleFactory,
    duration_in_quarters,
    recip,
    spell_timeline,
)
from transcoda.model.generation_policy import GenerationSettings

_GRAMMAR = Path("grammars/kern-transcoda-v2.gbnf")

HEADER = "*clefF4\t*clefG2\n*k[]\t*k[]\n*M2/4\t*M2/4\n"


@pytest.mark.parametrize(
    ("field", "quarters"),
    [
        ("4c", Fraction(1)),
        ("4.c", Fraction(3, 2)),
        ("2..c", Fraction(7, 2)),
        ("12cL", Fraction(1, 3)),
        ("24r", Fraction(1, 6)),
        ("20c", Fraction(1, 5)),
        ("28c", Fraction(1, 7)),
        ("7%2c", Fraction(8, 7)),
        ("0r", Fraction(8)),
        ("00c", Fraction(16)),
        ("4ryy", Fraction(1)),
        ("8c e g", Fraction(1, 2)),
    ],
)
def test_durations_follow_kern_recip(field, quarters):
    assert duration_in_quarters(field) == quarters


@pytest.mark.parametrize(
    "quarters",
    [Fraction(1), Fraction(1, 2), Fraction(3, 2), Fraction(7, 2), Fraction(1, 3), Fraction(5, 4),
     Fraction(8), Fraction(16), Fraction(5, 2), Fraction(2, 3), Fraction(3, 16)],
)
def test_recip_spells_any_gap_as_one_duration(quarters):
    spelled = recip(quarters)
    assert duration_in_quarters(spelled + "ryy") == quarters
    assert spelled.count(".") <= 2


def test_a_consistent_text_implies_no_rest():
    for text in (
        HEADER + "2C\t4c\n\t8d\n\t8e\n=\n",
        HEADER + "4C\t\n\t4c\n4D\t4d\n=\n",
        # Chopin Op. 48: grace notes in a sounding column, before the other hand's next chord.
        HEADER + "4C\t4.c\n\t16qqd\n4D\t\n\t8e\n=\n",
        HEADER + "*\t*^\n2C\t4c\t2e\n\t4d\t\n*\t*v\t*v\n=\n",
        HEADER + "MM3r\tMM3r\n=\n2C\t2c\n=\n",
    ):
        assert spell_timeline(text) == text


def test_a_barline_break_or_end_rests_the_lagging_column_first():
    assert spell_timeline(HEADER + "2C\t4c\n=\n") == HEADER + "2C\t4c\n\t4ryy\n=\n"
    assert spell_timeline(HEADER + "2C\t4c\n!!LO:LB:g=original\n=\n") == (
        HEADER + "2C\t4c\n\t4ryy\n!!LO:LB:g=original\n=\n"
    )
    assert spell_timeline(HEADER + "2C\t4c\n") == HEADER + "2C\t4c\n\t4ryy\n"
    assert spell_timeline(HEADER + "2C\t4c") == HEADER + "2C\t4c\n\t4ryy\n"


def test_a_note_in_a_column_that_runs_ahead_rests_the_others_up_to_it():
    # A sixteenth run read as a triplet leaves the right hand a third of a quarter behind.
    text = HEADER + "4C\t12c\n\t24d\n\t24e\n4D\t4f\n=\n"
    assert spell_timeline(text) == HEADER + "4C\t12c\n\t24d\n\t24e\n\t12ryy\n4D\t4f\n=\n"


def test_rests_for_columns_at_different_onsets_come_in_onset_order():
    text = HEADER + "*\t*^\n2C\t8c\t4e\n=\n"
    assert spell_timeline(text) == HEADER + "*\t*^\n2C\t8c\t4e\n\t4.ryy\t\n\t\t4ryy\n=\n"


def test_a_merge_rests_its_columns_up_to_the_latest_one():
    text = HEADER + "*\t*^\n2C\t4c\t2e\n*\t*v\t*v\n=\n"
    assert spell_timeline(text) == HEADER + "*\t*^\n2C\t4c\t2e\n\t4ryy\t\n*\t*v\t*v\n=\n"


def test_grace_notes_sit_at_their_columns_onset():
    # A grace note in a column behind the line is rested up to the line first.
    assert spell_timeline(HEADER + "4C\t8c\n4D\t8qd\n\t4e\n=\n") == (
        HEADER + "4C\t8c\n\t8ryy\n4D\t8qd\n\t4e\n=\n"
    )
    with pytest.raises(ValueError):
        spell_timeline(HEADER + "4C\t2c\n4D\t8qd\n")
    with pytest.raises(ValueError):
        spell_timeline(HEADER + "4C\t8c\n8qD\t8qd\n")


def test_multi_measure_rests_align_first_then_wait_for_the_barline():
    assert spell_timeline(HEADER + "4C\t2c\nMM3r\tMM3r\n=\n") == (
        HEADER + "4C\t2c\n4ryy\t\nMM3r\tMM3r\n=\n"
    )
    for broken in ("MM3r\tMM2r\n", "MM3r\t2c\n", "MM3r\tMM3r\n2C\t2c\n"):
        with pytest.raises(ValueError):
            spell_timeline(HEADER + broken)


def test_finalizer_spells_implied_rests_only_under_the_timeline_rule():
    text = HEADER + "2C\t2c\n=\n2C\t4c\n\t8d\n\t8"
    timeline = finalize_kern_sequence_text(
        text=text, saw_eos=False, hit_max_length=True, rule_factories=(TimelineRuleFactory(),)
    )
    assert timeline.text == HEADER + "2C\t2c\n=\n2C\t4c\n\t8d\n\t8ryy\n"
    assert timeline.truncated
    plain = finalize_kern_sequence_text(text=text, saw_eos=False, hit_max_length=True)
    assert plain.text == HEADER + "2C\t2c\n=\n2C\t4c\n\t8d\n"


_TOY = [
    "<bos>", "<eos>", "<pad>", "\t", "\n", "*clefF4", "*clefG2", "*k[]", "*", "*^", "*v",
    "v", "=", "==", "!!LO:LB", "2", "4", "8", "4c", "2c", "8c", "4C", "2C", "q", "c", "C",
    "MM3r", "ryy", "4ryy", ".",
]


def _allowed_after(tokens: list[str]) -> set[str]:
    i2w = dict(enumerate(_TOY))
    ids = {text: token_id for token_id, text in i2w.items()}
    processor = StatefulKernLogitsProcessor(
        i2w=i2w,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        rule_factories=[
            SpineStructureRuleFactory(barline_width_mode="compact", empty_field_mode="data_only"),
            TimelineRuleFactory(),
        ],
    )
    prefix = [0]
    for index in range(len(tokens) + 1):
        if index:
            prefix.append(ids[tokens[index - 1]])
        scores = torch.zeros((1, len(_TOY)))
        processor(torch.tensor([prefix]), scores)
    return {_TOY[i] for i in torch.isfinite(scores[0]).nonzero().flatten().tolist()}


_TOY_HEADER = ["*clefF4", "\t", "*clefG2", "\n", "*k[]", "\t", "*k[]", "\n"]
_MISALIGNED = [*_TOY_HEADER, "2C", "\t", "4c", "\n"]


def test_processor_leaves_a_misaligned_decoder_free():
    assert {"=", "!!LO:LB", "<eos>", "MM3r", "4C", "\t", "*"} <= _allowed_after(_MISALIGNED)
    assert {"4C", "2C", "8"} <= _allowed_after([*_MISALIGNED, "4c", "\t"])
    assert "*v" in _allowed_after([*_MISALIGNED, "*", "\t", "*^", "\n", "*", "\t"])


def test_processor_keeps_a_grace_note_off_a_line_that_ends_before_its_column():
    ahead = [*_TOY_HEADER, "4C", "\t", "2c", "\n", "4C", "\t", "8"]
    assert "q" not in _allowed_after(ahead)
    assert "c" in _allowed_after(ahead)
    behind = [*_MISALIGNED, "4C", "\t", "8"]
    assert {"q", "c"} <= _allowed_after(behind)


def test_a_pending_duration_takes_at_most_three_dots():
    line = [*_MISALIGNED, "\t", "8"]
    assert "." in _allowed_after([*line, ".", "."])
    assert "." not in _allowed_after([*line, ".", ".", "."])
