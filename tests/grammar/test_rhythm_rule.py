from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from src.grammar.kern_prefix_state import KernPrefixState
from src.grammar.rhythm_rule import RhythmRule, RhythmRuleError, TimeSignature
from src.grammar.stateful_kern_logits_processor import TokenizerConstraintContext


def _accept_line(rule: RhythmRule, fields: list[str]) -> None:
    rule.on_line_closed(tuple(fields))


@lru_cache(maxsize=1)
def _ground_truth_samples() -> dict[str, str]:
    sample_path = (
        Path(__file__).resolve().parents[2]
        / "analysis/benchmark_constraint_vs_unconstrained_20260313/synth_ground_truth.jsonl"
    )
    return {
        entry["sample_key"]: entry["ground_truth"]
        for entry in (json.loads(line) for line in sample_path.read_text().splitlines())
    }


def _assert_ground_truth_sample_reachable(sample_key: str) -> None:
    rule = RhythmRule()
    for line_no, line in enumerate(_ground_truth_samples()[sample_key].splitlines(), start=1):
        fields = tuple(line.split("\t"))
        assert rule.can_close_line(fields), f"{sample_key} rejected at line {line_no}: {line}"
        rule.on_line_closed(fields)


def _context() -> TokenizerConstraintContext:
    return TokenizerConstraintContext.from_i2w(
        i2w={
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            3: "\t",
            4: "\n",
            5: "4c",
            6: "2g",
            7: "4g",
            8: "=1",
            9: "=:|!|:",
            10: "16g",
            11: "32g",
            12: ".",
            13: "[",
            14: "g",
            15: "6",
            16: "%",
            17: "q",
            18: "8",
            19: "4",
        },
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def _warmup_single_spine(rule: RhythmRule, meter: str = "*M4/4") -> None:
    _accept_line(rule, ["**kern"])
    _accept_line(rule, [meter])


def _warmup_nonfirst_short_final_measure(rule: RhythmRule) -> None:
    _warmup_single_spine(rule)
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["=1"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])


def test_overfill_tokens_masked_immediately():
    rule = RhythmRule()
    ctx = _context()
    _warmup_single_spine(rule)
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])

    prefix = KernPrefixState()
    scores = torch.zeros((10,), dtype=torch.float32)
    rule.mask_scores(prefix, scores, ctx)

    assert scores[6] == float("-inf")
    assert scores[7] != float("-inf")


def test_underfilled_nonfinal_measure_fails_on_next_barline():
    rule = RhythmRule()
    _warmup_single_spine(rule)
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["=1"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["=2"])

    assert rule.can_close_line(("=3",)) is False


def test_continuation_requires_active_carry() -> None:
    rule = RhythmRule()
    _warmup_single_spine(rule)

    assert rule.can_close_line((".",)) is False


def test_grace_alignment_placeholder_allowed_at_measure_start() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**kern"])
    _accept_line(rule, ["*M2/4", "*M2/4"])

    assert rule.can_close_line((".", "eeq")) is True


def test_grace_alignment_placeholder_allowed_mid_measure_without_carry() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**kern"])
    _accept_line(rule, ["*M4/4", "*M4/4"])
    _accept_line(rule, ["4c", "4e"])

    assert rule.can_close_line((".", "bq")) is True


def test_grace_alignment_placeholder_allowed_when_grace_is_in_first_spine() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**kern"])
    _accept_line(rule, ["*M4/4", "*M4/4"])

    assert rule.can_close_line(("B-q", ".")) is True


def test_grace_alignment_placeholder_allowed_in_three_spines() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**kern", "**kern"])
    _accept_line(rule, ["*M4/4", "*M4/4", "*M4/4"])

    assert rule.can_close_line((".", "ddq", ".")) is True


def test_grace_alignment_placeholders_do_not_advance_measure_duration() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**kern"])
    _accept_line(rule, ["*M2/4", "*M2/4"])
    _accept_line(rule, [".", "eeq"])

    assert rule.spines[0].measure_duration == 0
    assert rule.spines[1].measure_duration == 0
    assert rule.spines[0].outstanding_duration == 0
    assert rule.spines[1].outstanding_duration == 0


def test_grace_only_lines_preserve_existing_carry() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**kern"])
    _accept_line(rule, ["*M4/4", "*M4/4"])
    _accept_line(rule, ["4AA", "8a"])

    assert rule.spines[0].outstanding_duration == pytest.approx(1 / 8)

    _accept_line(rule, [".", "ccc#qLLL"])

    assert rule.spines[0].outstanding_duration == pytest.approx(1 / 8)
    assert rule.spines[1].outstanding_duration == 0


def test_reduced_000018_sequence_remains_reachable_after_grace_only_lines() -> None:
    rule = RhythmRule()
    for fields in [
        ["**kern", "**kern"],
        ["*M4/4", "*M4/4"],
        ["8d", "16ff#L"],
        [".", "32ddL"],
        [".", "32bJJJ"],
        ["8e", "8cc#L"],
        [".", "bq"],
        ["8EJ", "8bJ"],
        [".", "aqLLL"],
        [".", "bqJJJ"],
        ["4AA", "8a"],
    ]:
        _accept_line(rule, fields)

    assert rule.spines[0].outstanding_duration == pytest.approx(1 / 8)

    _accept_line(rule, [".", "ccc#qLLL"])
    _accept_line(rule, [".", "dddqJJJ"])

    assert rule.spines[0].outstanding_duration == pytest.approx(1 / 8)
    assert rule.can_close_line((".", "8eee")) is True


def test_zero_carry_placeholder_rejected_at_measure_start_on_timed_line() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**kern"])
    _accept_line(rule, ["*M4/4", "*M4/4"])

    assert rule.can_close_line((".", "4c")) is False


def test_explicit_onset_before_prior_carry_resolves_is_rejected() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**kern"])
    _accept_line(rule, ["8c", "16e"])

    assert rule.can_close_line(("8d", "4f")) is False


@pytest.mark.parametrize("sample_key", ["001350", "003669", "005580", "005911", "006675"])
def test_reported_zero_carry_placeholder_samples_are_reachable(sample_key: str) -> None:
    _assert_ground_truth_sample_reachable(sample_key)


def test_valid_pickup_allowed_on_first_barline():
    rule = RhythmRule()
    _warmup_single_spine(rule)
    _accept_line(rule, ["4c"])

    assert rule.can_close_line(("=1",)) is True


def test_valid_incomplete_final_allowed_at_eos():
    rule = RhythmRule()
    _warmup_single_spine(rule)

    assert rule.can_end_sequence(("4c",)) is True


def test_terminal_short_measure_after_end_repeat_barline_is_allowed_at_eof() -> None:
    rule = RhythmRule()
    _warmup_nonfirst_short_final_measure(rule)

    assert rule.can_end_sequence(("=:|!",)) is True


def test_terminal_short_measure_after_plain_barline_is_allowed_at_eof() -> None:
    rule = RhythmRule()
    _warmup_nonfirst_short_final_measure(rule)

    assert rule.can_end_sequence(("=",)) is True


def test_terminal_short_measure_after_double_barline_remains_allowed_at_eof() -> None:
    rule = RhythmRule()
    _warmup_nonfirst_short_final_measure(rule)

    assert rule.can_end_sequence(("==",)) is True


def test_terminal_short_measure_before_more_content_remains_invalid() -> None:
    rule = RhythmRule()
    _warmup_nonfirst_short_final_measure(rule)
    _accept_line(rule, ["="])

    assert rule.can_close_line(("4c",)) is False
    assert rule.can_close_line(("*",)) is False


@pytest.mark.parametrize("barline", ["=||", "=!|:", "=:|!|:"])
def test_section_start_short_measure_allows_following_data(barline: str) -> None:
    rule = RhythmRule()
    _warmup_nonfirst_short_final_measure(rule)
    _accept_line(rule, [barline])

    assert rule.pending_final_short_errors == []
    assert rule.can_close_line(("24ddLL",)) is True


def test_section_start_short_measure_allows_following_tandem_interpretation() -> None:
    rule = RhythmRule()
    _warmup_nonfirst_short_final_measure(rule)
    _accept_line(rule, ["=||"])

    assert rule.pending_final_short_errors == []
    assert rule.can_close_line(("*clefG2",)) is True


def test_plain_short_measure_before_tandem_interpretation_remains_invalid() -> None:
    rule = RhythmRule()
    _warmup_nonfirst_short_final_measure(rule)
    _accept_line(rule, ["="])

    assert rule.can_close_line(("*clefG2",)) is False


def test_reduced_004461_section_continuation_remains_reachable() -> None:
    rule = RhythmRule()
    for fields in [
        ["**kern"],
        ["*M3/4"],
        ["4g"],
        ["4g"],
        ["4g"],
        ["=1"],
        ["4g"],
        ["=||"],
        ["24ddLL"],
        ["24ee"],
        ["24ff#JJ"],
    ]:
        assert rule.can_close_line(tuple(fields)) is True
        _accept_line(rule, fields)

    assert rule.can_close_line(("=",)) is True


def test_explicit_final_barline_does_not_leave_pending_short_measure_error() -> None:
    rule = RhythmRule()
    _warmup_nonfirst_short_final_measure(rule)
    _accept_line(rule, ["=:|!"])

    assert rule.pending_final_short_errors == []
    assert rule.can_close_line(("==",)) is True


def test_repeat_pairing_accepted_and_rejected():
    valid = RhythmRule()
    _warmup_single_spine(valid, meter="*M6/8")
    _accept_line(valid, ["8c"])
    _accept_line(valid, ["=1"])
    _accept_line(valid, ["5%8c"])

    assert valid.can_close_line(("=:|!|:",)) is True

    invalid = RhythmRule()
    _warmup_single_spine(invalid, meter="*M6/8")
    _accept_line(invalid, ["8c"])
    _accept_line(invalid, ["=1"])
    _accept_line(invalid, ["4c"])

    assert invalid.can_close_line(("=:|!|:",)) is False


def test_time_signature_changes_apply_to_current_measure_before_data():
    rule = RhythmRule()
    _accept_line(rule, ["**kern"])
    _accept_line(rule, ["*M4/4"])
    _accept_line(rule, ["*M3/4"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])

    assert rule.can_close_line(("=1",)) is True


def test_time_signature_parse_rejects_incomplete_denominator() -> None:
    with pytest.raises(RhythmRuleError, match=r"\*M4/$"):
        TimeSignature.parse("*M4/")


def test_time_signature_matcher_requires_complete_beats_and_unit() -> None:
    assert TimeSignature.is_time_signature("*M9/16") is True
    assert TimeSignature.is_time_signature("*M9") is False
    assert TimeSignature.is_time_signature("*M9/") is False


def test_malformed_time_signature_candidate_is_rejected_without_value_error() -> None:
    rule = RhythmRule()

    assert rule.can_close_line(("*M9/16", "*M9")) is False
    assert rule.can_close_line(("*M9/16", "*M9/")) is False


def test_split_and_merge_preserve_measure_state():
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**kern"])
    _accept_line(rule, ["*M4/4", "*M4/4"])
    _accept_line(rule, ["4c", "4e"])
    _accept_line(rule, ["*^", "*"])

    assert len(rule.spines) == 3
    assert rule.spines[0].measure_duration == rule.spines[1].measure_duration == pytest.approx(1 / 4)

    _accept_line(rule, ["4d", "4f", "4g"])
    _accept_line(rule, ["*v", "*v", "*"])

    assert len(rule.spines) == 2
    assert rule.spines[0].measure_duration == pytest.approx(1 / 2)


def test_ambiguous_merge_carry_remains_deferred_until_a_fresh_resolution() -> None:
    rule = RhythmRule()
    lines = _ground_truth_samples()["006093"].splitlines()

    for line in lines[:59]:
        fields = tuple(line.split("\t"))
        assert rule.can_close_line(fields) is True
        rule.on_line_closed(fields)

    assert rule.spines[1].has_ambiguous_merge_carry is True
    next_fields = tuple(lines[59].split("\t"))
    assert rule.can_close_line(next_fields) is True

    rule.on_line_closed(next_fields)

    assert rule.spines[1].has_ambiguous_merge_carry is True


def test_split_derived_three_spine_overactivity_is_rejected() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**kern"])
    _accept_line(rule, ["*", "*^"])
    _accept_line(rule, ["4c", "4e", "16g"])
    _accept_line(rule, [".", ".", "16a"])

    assert rule.can_close_line((".", "16b", "16c")) is False


def test_barline_rejected_when_merge_preserves_unanimous_unresolved_carry() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**kern"])
    _accept_line(rule, ["*M4/4", "*M4/4"])
    _accept_line(rule, ["*^", "*"])
    _accept_line(rule, ["4c", "4d", "8e"])
    _accept_line(rule, ["*v", "*v", "*"])

    assert rule.can_close_line(("=1", "=1")) is False


@pytest.mark.parametrize("sample_key", ["004805", "005370", "006035", "006062", "006746", "006751"])
def test_reported_merged_duration_restart_samples_are_reachable(sample_key: str) -> None:
    _assert_ground_truth_sample_reachable(sample_key)


@pytest.mark.parametrize("sample_key", ["004921", "005928", "006421"])
def test_reported_merged_barline_samples_are_reachable(sample_key: str) -> None:
    _assert_ground_truth_sample_reachable(sample_key)


def test_reported_merged_continuation_sample_remains_reachable() -> None:
    _assert_ground_truth_sample_reachable("006093")


def test_non_kern_spines_are_ignored():
    rule = RhythmRule()
    _accept_line(rule, ["**kern", "**text"])
    _accept_line(rule, ["*M4/4", "*"])
    _accept_line(rule, ["4c", "lyric"])

    assert rule.can_close_line(("=1", "=1")) is True


def test_body_only_spine_op_line_bootstraps_synthetic_header():
    rule = RhythmRule()

    assert rule.can_close_line(("*", "*^")) is True

    rule.on_line_closed(("*", "*^"))

    assert len(rule.spines) == 3


def test_terminal_single_spine_note_can_end_sequence() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern"])
    _accept_line(rule, ["*M4/4"])

    assert rule.can_end_sequence(("4c",)) is True


def test_repeat_pairing_failure_is_rejected() -> None:
    rule = RhythmRule()
    _accept_line(rule, ["**kern"])
    _accept_line(rule, ["*M6/8"])
    _accept_line(rule, ["8c"])
    _accept_line(rule, ["=1"])
    _accept_line(rule, ["4c"])

    assert rule.can_close_line(("=:|!|:",)) is False


def test_mask_rejects_duration_and_continuation_that_break_carry_rules() -> None:
    rule = RhythmRule()
    ctx = _context()
    _accept_line(rule, ["**kern", "**kern"])
    _accept_line(rule, ["8c", "16e"])

    prefix = KernPrefixState()
    scores = torch.zeros((13,), dtype=torch.float32)
    rule.mask_scores(prefix, scores, ctx)

    assert scores[12] != float("-inf")
    assert scores[10] == float("-inf")


def test_late_measure_tail_burst_is_masked_when_budget_is_exhausted() -> None:
    rule = RhythmRule()
    ctx = _context()
    _warmup_single_spine(rule)
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["16g"])

    prefix = KernPrefixState()
    scores = torch.zeros((13,), dtype=torch.float32)
    rule.mask_scores(prefix, scores, ctx)

    assert scores[11] != float("-inf")
    assert scores[7] == float("-inf")


def test_wrapper_only_prefix_uses_same_duration_budget_masking() -> None:
    rule = RhythmRule()
    ctx = _context()
    _warmup_single_spine(rule)
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])

    prefix = KernPrefixState(current_field_buffer="[")
    scores = torch.zeros((20,), dtype=torch.float32)
    rule.mask_scores(prefix, scores, ctx)

    assert scores[6] == float("-inf")
    assert scores[18] != float("-inf")


def test_partial_duration_prefix_masks_only_tokens_that_keep_overfill() -> None:
    rule = RhythmRule()
    ctx = _context()
    _warmup_single_spine(rule)
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])

    prefix = KernPrefixState(current_field_buffer="1")
    scores = torch.zeros((20,), dtype=torch.float32)
    rule.mask_scores(prefix, scores, ctx)

    assert scores[14] == float("-inf")
    assert scores[13] == float("-inf")
    assert scores[15] != float("-inf")
    assert scores[16] != float("-inf")


def test_fixed_duration_prefix_masks_non_grace_tail_tokens_when_overfilled() -> None:
    rule = RhythmRule()
    ctx = _context()
    _warmup_single_spine(rule)
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])
    _accept_line(rule, ["4c"])

    prefix = KernPrefixState(current_field_buffer="1c")
    scores = torch.zeros((20,), dtype=torch.float32)
    rule.mask_scores(prefix, scores, ctx)

    assert scores[14] == float("-inf")
    assert scores[13] == float("-inf")
    assert scores[17] != float("-inf")


def test_active_carry_masks_partial_duration_prefix_but_ambiguous_carry_does_not() -> None:
    ctx = _context()

    strict_rule = RhythmRule()
    _accept_line(strict_rule, ["**kern", "**kern"])
    _accept_line(strict_rule, ["8c", "16e"])
    strict_prefix = KernPrefixState(current_field_buffer="1")
    strict_scores = torch.zeros((20,), dtype=torch.float32)
    strict_rule.mask_scores(strict_prefix, strict_scores, ctx)

    assert strict_scores[14] == float("-inf")
    assert strict_scores[17] == float("-inf")

    relaxed_rule = RhythmRule()
    _accept_line(relaxed_rule, ["**kern", "**kern"])
    _accept_line(relaxed_rule, ["*^", "*"])
    _accept_line(relaxed_rule, ["4c", "4d", "8e"])
    _accept_line(relaxed_rule, ["*", "*v", "*v"])
    relaxed_prefix = KernPrefixState(current_field_buffer="1")
    relaxed_scores = torch.zeros((20,), dtype=torch.float32)
    relaxed_rule.mask_scores(relaxed_prefix, relaxed_scores, ctx)

    assert relaxed_rule.spines[1].has_ambiguous_merge_carry is True
    assert relaxed_scores[14] != float("-inf")
    assert relaxed_scores[17] != float("-inf")
