from __future__ import annotations

from src.grammar.rhythm_rule import RhythmRule
from src.grammar.semantic_sequence_finalizer import finalize_kern_sequence_text
from src.grammar.spine_structure_rule import SpineStructureRule

_RULES = (SpineStructureRule, RhythmRule)


def test_trailing_dangling_tab_is_trimmed() -> None:
    result = finalize_kern_sequence_text(
        text="*clefF4\t*clefG2\n4c\t",
        saw_eos=False,
        hit_max_length=True,
        rule_factories=_RULES,
    )

    assert result.text == "*clefF4\t*clefG2\n*-\t*-"
    assert result.trimmed_incomplete_tail is True
    assert result.appended_terminator is True
    assert result.truncated is True


def test_partial_final_line_is_trimmed() -> None:
    result = finalize_kern_sequence_text(
        text="*clefF4\t*clefG2\n*M4/4\t*M4/4\n*^",
        saw_eos=False,
        hit_max_length=True,
        rule_factories=_RULES,
    )

    assert result.text == "*clefF4\t*clefG2\n*M4/4\t*M4/4\n*-\t*-"
    assert result.trimmed_incomplete_tail is True
    assert result.appended_terminator is True


def test_complete_line_without_terminator_gets_legal_terminator() -> None:
    result = finalize_kern_sequence_text(
        text="*clefF4\t*clefG2\n*M4/4\t*M4/4\n4c\t4e\n",
        saw_eos=False,
        hit_max_length=False,
        rule_factories=_RULES,
    )

    assert result.text.endswith("\n*-\t*-")
    assert result.trimmed_incomplete_tail is False
    assert result.appended_terminator is True


def test_invalid_tail_rolls_back_without_inventing_structure() -> None:
    result = finalize_kern_sequence_text(
        text="*clefF4\t*clefG2\n*v\t*",
        saw_eos=False,
        hit_max_length=True,
        rule_factories=_RULES,
    )

    assert result.text == "*clefF4\t*clefG2\n*-\t*-"
    assert result.trimmed_incomplete_tail is True
    assert result.appended_terminator is True


def test_max_length_without_eos_marks_truncation() -> None:
    result = finalize_kern_sequence_text(
        text="*clefF4\t*clefG2\n4c\t",
        saw_eos=False,
        hit_max_length=True,
        rule_factories=_RULES,
    )

    assert result.hit_max_length is True
    assert result.saw_eos is False
    assert result.truncated is True


def test_complete_eos_terminated_sequence_is_preserved() -> None:
    result = finalize_kern_sequence_text(
        text="*-\t*-",
        saw_eos=True,
        hit_max_length=False,
        rule_factories=_RULES,
    )

    assert result.text == "*-\t*-"
    assert result.trimmed_incomplete_tail is False
    assert result.appended_terminator is False
    assert result.truncated is False
