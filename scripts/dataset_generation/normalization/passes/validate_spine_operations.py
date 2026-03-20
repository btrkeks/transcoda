"""Validate semantic correctness of spine operations in normalized **kern text."""

from __future__ import annotations

from src.core.spine_state import (
    InvalidSpineOperationError,
    UnsupportedSpineManipulatorError,
    advance_spine_count,
    is_interpretation_record,
)

from ..base import NormalizationContext


class ValidateSpineOperations:
    """Final-pass validator for spine widths and manipulation semantics."""

    name = "validate_spine_operations"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No-op preparation."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """This pass only validates; it does not transform text."""
        return text

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """Validate per-line spine widths and manipulator semantics."""
        lines = text.split("\n")
        current_spines: int | None = None

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            fields = line.split("\t")
            if current_spines is None:
                current_spines = len(fields)

            assert current_spines is not None
            if len(fields) != current_spines:
                raise InvalidSpineOperationError(
                    f"Line {line_number}: expected {current_spines} spine fields, "
                    f"got {len(fields)}. Row: {line!r}"
                )

            if not is_interpretation_record(fields):
                continue

            try:
                current_spines = advance_spine_count(current_spines, fields)
            except UnsupportedSpineManipulatorError as exc:
                raise UnsupportedSpineManipulatorError(
                    f"Line {line_number}: {exc}. Row: {line!r}"
                ) from exc
            except InvalidSpineOperationError as exc:
                raise InvalidSpineOperationError(
                    f"Line {line_number}: {exc}. Row: {line!r}"
                ) from exc
            except ValueError as exc:
                raise InvalidSpineOperationError(
                    f"Line {line_number}: invalid spine structure ({exc}). Row: {line!r}"
                ) from exc
