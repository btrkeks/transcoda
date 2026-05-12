"""Tests for StripTerminalTerminator normalization pass."""

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import StripTerminalTerminator


def test_pass_exists():
    pass_obj = StripTerminalTerminator()
    assert pass_obj.name == "strip_terminal_terminator"


def test_strips_single_spine_terminator():
    pass_obj = StripTerminalTerminator()
    ctx = NormalizationContext()
    input_text = "4c\n=1\n*-"
    expected = "4c\n=1"

    pass_obj.prepare(input_text, ctx)
    result = pass_obj.transform(input_text, ctx)
    pass_obj.validate(result, ctx)
    assert result == expected


def test_strips_multi_spine_terminator():
    pass_obj = StripTerminalTerminator()
    ctx = NormalizationContext()
    input_text = "4c\t4e\n==\t==\n*-\t*-"
    expected = "4c\t4e\n==\t=="

    pass_obj.prepare(input_text, ctx)
    result = pass_obj.transform(input_text, ctx)
    pass_obj.validate(result, ctx)
    assert result == expected


def test_noop_when_no_terminator_present():
    pass_obj = StripTerminalTerminator()
    ctx = NormalizationContext()
    input_text = "4c\t4e\n=\t="

    pass_obj.prepare(input_text, ctx)
    result = pass_obj.transform(input_text, ctx)
    pass_obj.validate(result, ctx)
    assert result == input_text


def test_idempotent():
    pass_obj = StripTerminalTerminator()
    ctx = NormalizationContext()
    input_text = "4c\n=1\n*-"

    first = pass_obj.transform(input_text, ctx)
    second = pass_obj.transform(first, ctx)
    assert first == second
