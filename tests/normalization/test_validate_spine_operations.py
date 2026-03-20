"""Tests for ValidateSpineOperations pass."""

import pytest

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import ValidateSpineOperations
from src.core.spine_state import InvalidSpineOperationError, UnsupportedSpineManipulatorError


def _run_pass(text: str) -> str:
    pass_obj = ValidateSpineOperations()
    ctx = NormalizationContext()
    pass_obj.prepare(text, ctx)
    out = pass_obj.transform(text, ctx)
    pass_obj.validate(out, ctx)
    return out


def test_pass_exists():
    pass_obj = ValidateSpineOperations()
    assert pass_obj.name == "validate_spine_operations"


def test_accepts_valid_consecutive_manipulator_rows():
    text = (
        "*clefF4\t*clefG2\n"
        "*k[]\t*k[]\n"
        "*M4/4\t*M4/4\n"
        "*^\t*\n"
        "*\t*^\t*\n"
        "4c\t4e\t4g\t4b\n"
        "*v\t*v\t*\t*\n"
        "*\t*v\t*v\n"
        "4c\t4g\n"
    )
    assert _run_pass(text) == text


def test_rejects_orphan_merge_with_line_context():
    text = (
        "*clefF4\t*clefG2\n"
        "*k[]\t*k[]\n"
        "*M4/4\t*M4/4\n"
        "*\t*v\n"
    )
    with pytest.raises(InvalidSpineOperationError, match=r"Line 4: .*Invalid merge operation"):
        _run_pass(text)


def test_rejects_spine_width_mismatch():
    text = (
        "*clefF4\t*clefG2\n"
        "*k[]\t*k[]\n"
        "*M4/4\t*M4/4\n"
        "4c\t4e\t4g\n"
    )
    with pytest.raises(InvalidSpineOperationError, match=r"Line 4: expected 2 spine fields, got 3"):
        _run_pass(text)


def test_unsupported_spine_manipulators_still_raise_specific_error():
    text = (
        "*clefF4\t*clefG2\n"
        "*k[]\t*k[]\n"
        "*M4/4\t*M4/4\n"
        "*x\t*x\n"
    )
    with pytest.raises(UnsupportedSpineManipulatorError, match=r"Line 4: Unsupported spine manipulators"):
        _run_pass(text)
