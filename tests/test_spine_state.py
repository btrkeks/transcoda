"""Tests for shared spine-state transition helpers."""

import pytest

from src.core.spine_state import (
    InvalidSpineOperationError,
    UnsupportedSpineManipulatorError,
    advance_keep_mask,
    advance_spine_count,
    is_interpretation_record,
)


def test_is_interpretation_record():
    assert is_interpretation_record(["*clefG2", "*M4/4"])
    assert is_interpretation_record(["**kern", "**text"])
    assert not is_interpretation_record(["4c", "*"])
    assert not is_interpretation_record([])


def test_advance_spine_count_split_merge_terminate():
    assert advance_spine_count(3, ["*", "*^", "*"]) == 4
    assert advance_spine_count(4, ["*v", "*v", "*", "*"]) == 3
    assert advance_spine_count(3, ["*-", "*", "*-"]) == 1


def test_advance_spine_count_non_interpretation_passthrough():
    assert advance_spine_count(3, ["4c", "4e", "4g"]) == 3


def test_advance_keep_mask_split():
    keep_mask = [True, False, True]
    fields = ["*", "*^", "*"]
    assert advance_keep_mask(keep_mask, fields) == [True, False, False, True]


def test_advance_keep_mask_merge_groups():
    keep_mask = [False, False, True, False, True]
    fields = ["*v", "*v", "*", "*v", "*v"]
    assert advance_keep_mask(keep_mask, fields) == [False, True, True]


@pytest.mark.parametrize(
    "fields",
    [
        ["*", "*v", "*"],
        ["*v", "*", "*"],
        ["*", "*", "*v"],
        ["*v"],
    ],
)
def test_orphan_merge_groups_raise(fields):
    with pytest.raises(InvalidSpineOperationError, match=r"Invalid merge operation"):
        advance_spine_count(len(fields), fields)
    with pytest.raises(InvalidSpineOperationError, match=r"Invalid merge operation"):
        advance_keep_mask([True] * len(fields), fields)


def test_advance_keep_mask_terminate():
    keep_mask = [True, False, True]
    fields = ["*-", "*", "*-"]
    assert advance_keep_mask(keep_mask, fields) == [False]


def test_unsupported_manipulators_raise():
    with pytest.raises(UnsupportedSpineManipulatorError, match=r"\*x"):
        advance_spine_count(2, ["*x", "*x"])
    with pytest.raises(UnsupportedSpineManipulatorError, match=r"\*\+"):
        advance_keep_mask([True, False], ["*+", "*+"])


def test_mixed_supported_manipulators_are_applied_in_one_record():
    fields = ["*^", "*v", "*v", "*"]
    assert advance_spine_count(4, fields) == 4
    assert advance_keep_mask([True, False, True, False], fields) == [True, True, True, False]


def test_interpretation_width_mismatch_raises():
    with pytest.raises(ValueError, match="width mismatch"):
        advance_spine_count(2, ["*", "*", "*"])
    with pytest.raises(ValueError, match="width mismatch"):
        advance_keep_mask([True], ["*", "*"])
