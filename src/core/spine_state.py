"""Helpers for tracking Humdrum spine transitions on interpretation records."""

from __future__ import annotations

from collections.abc import Sequence

_UNSUPPORTED_SPINE_MANIPULATORS = {"*x", "*+"}
_SUPPORTED_SPINE_MANIPULATORS = {"*^", "*v", "*-"}
_ALL_SPINE_MANIPULATORS = _SUPPORTED_SPINE_MANIPULATORS | _UNSUPPORTED_SPINE_MANIPULATORS


class UnsupportedSpineManipulatorError(ValueError):
    """Raised when unsupported spine manipulators are encountered."""

    pass


class InvalidSpineOperationError(ValueError):
    """Raised when a supported spine operation record is semantically invalid."""

    pass


def is_interpretation_record(fields: Sequence[str]) -> bool:
    """Return True when all fields are interpretation tokens."""
    return bool(fields) and all(field.startswith("*") for field in fields)


def _ensure_supported_spine_manipulators(fields: Sequence[str]) -> None:
    """Validate that a record does not contain unsupported manipulators."""
    active = {field for field in fields if field in _ALL_SPINE_MANIPULATORS}
    unsupported = sorted(active & _UNSUPPORTED_SPINE_MANIPULATORS)
    if unsupported:
        ops = ", ".join(unsupported)
        raise UnsupportedSpineManipulatorError(f"Unsupported spine manipulators: {ops}")


def validate_spine_operation_record(fields: Sequence[str]) -> None:
    """Validate semantic constraints for spine manipulator records.

    Supported mixed records are allowed, but `*v` tokens must appear in merge
    groups of at least two adjacent tokens.
    """
    fields_list = list(fields)
    i = 0
    while i < len(fields_list):
        if fields_list[i] != "*v":
            i += 1
            continue

        j = i
        while j < len(fields_list) and fields_list[j] == "*v":
            j += 1
        run_len = j - i
        if run_len < 2:
            raise InvalidSpineOperationError(
                f"Invalid merge operation: '*v' at field {i + 1} is not adjacent to another '*v'."
            )
        i = j


def advance_spine_count(current_spines: int, fields: Sequence[str]) -> int:
    """Advance the spine count after applying an interpretation record."""
    if current_spines < 0:
        raise ValueError(f"current_spines must be >= 0, got {current_spines}")
    if not is_interpretation_record(fields):
        return current_spines
    if len(fields) != current_spines:
        raise ValueError(
            f"Interpretation record width mismatch: expected {current_spines} fields, "
            f"got {len(fields)}"
        )

    _ensure_supported_spine_manipulators(fields)
    validate_spine_operation_record(fields)

    next_spines = 0
    i = 0
    fields_list = list(fields)
    while i < len(fields_list):
        token = fields_list[i]
        if token == "*v":
            while i < len(fields_list) and fields_list[i] == "*v":
                i += 1
            next_spines += 1
            continue
        if token == "*^":
            next_spines += 2
            i += 1
            continue
        if token == "*-":
            i += 1
            continue
        next_spines += 1
        i += 1
    return next_spines


def advance_keep_mask(keep_mask: Sequence[bool], fields: Sequence[str]) -> list[bool]:
    """Advance a spine keep-mask after applying an interpretation record."""
    if not is_interpretation_record(fields):
        return list(keep_mask)
    if len(fields) != len(keep_mask):
        raise ValueError(
            f"Interpretation record width mismatch: expected {len(keep_mask)} fields, "
            f"got {len(fields)}"
        )

    _ensure_supported_spine_manipulators(fields)
    validate_spine_operation_record(fields)

    next_mask = []
    i = 0
    fields_list = list(fields)
    mask_list = list(keep_mask)
    while i < len(fields_list):
        token = fields_list[i]
        if token == "*v":
            j = i
            keep_any = False
            while j < len(fields_list) and fields_list[j] == "*v":
                keep_any = keep_any or mask_list[j]
                j += 1
            next_mask.append(keep_any)
            i = j
            continue
        if token == "*^":
            keep = mask_list[i]
            next_mask.extend([keep, keep])
            i += 1
            continue
        if token == "*-":
            i += 1
            continue
        next_mask.append(mask_list[i])
        i += 1
    return next_mask
