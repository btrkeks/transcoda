"""Semantic width-preservation rule for stateful **kern decoding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from src.core.spine_state import advance_spine_count

from .kern_prefix_state import KernPrefixState
from .stateful_kern_logits_processor import KernSemanticRule, TokenizerConstraintContext

_SPINE_OP_FIELDS = {"*", "*^", "*v", "*-"}


class SpineLineKind(str, Enum):
    """Structural class of the line currently being decoded."""

    UNKNOWN = "unknown"
    SPINE_OP = "spine_op"
    FIXED_WIDTH = "fixed_width"


def _is_spine_op_prefix(text: str) -> bool:
    return any(field.startswith(text) for field in _SPINE_OP_FIELDS)


@dataclass(frozen=True)
class SpineLineClosePreview:
    """Computed outcome of closing a line under spine-width semantics."""

    fields: tuple[str, ...]
    line_kind: SpineLineKind
    prior_spines: int
    next_spines: int
    terminated: bool


@dataclass
class SpineStructureRule(KernSemanticRule):
    """Track width-aware spine invariants for one generated sequence."""

    active_spines: int | None = None
    current_line_kind: SpineLineKind = SpineLineKind.UNKNOWN
    _terminated: bool = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    def on_text_appended(self, prefix_state: KernPrefixState) -> None:
        self.current_line_kind = self._infer_line_kind(prefix_state)

    def on_tab_appended(self, prefix_state: KernPrefixState) -> None:
        self.current_line_kind = self._infer_line_kind(prefix_state)

    def on_line_closed(self, fields: tuple[str, ...]) -> None:
        preview = self.preview_line_close(fields)
        self.active_spines = preview.next_spines
        self._terminated = preview.terminated
        self.current_line_kind = SpineLineKind.UNKNOWN

    def can_accept_tab(self, prefix_state: KernPrefixState) -> bool:
        if prefix_state.current_field_buffer == "":
            return False
        if self.active_spines is None:
            return True
        return len(prefix_state.completed_fields) + 1 < self.active_spines

    def can_close_line(self, fields: tuple[str, ...]) -> bool:
        try:
            self.preview_line_close(fields)
        except ValueError:
            return False
        return True

    def can_end_sequence(self, fields: tuple[str, ...]) -> bool:
        return self.can_close_line(fields)

    def mask_scores(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
    ) -> None:
        if prefix_state.current_field_buffer == "":
            return
        if not self.can_accept_tab(prefix_state):
            row_scores[context.tab_token_id] = float("-inf")

    def preview_line_close(self, fields: tuple[str, ...]) -> SpineLineClosePreview:
        prior_spines = self.active_spines if self.active_spines is not None else len(fields)
        if self.active_spines is not None and len(fields) != self.active_spines:
            raise ValueError(
                f"expected {self.active_spines} fields before line close, got {len(fields)}"
            )

        if all(field in _SPINE_OP_FIELDS for field in fields):
            line_kind = SpineLineKind.SPINE_OP
            try:
                next_spines = advance_spine_count(prior_spines, fields)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        else:
            line_kind = SpineLineKind.FIXED_WIDTH
            next_spines = prior_spines

        return SpineLineClosePreview(
            fields=fields,
            line_kind=line_kind,
            prior_spines=prior_spines,
            next_spines=next_spines,
            terminated=next_spines == 0,
        )

    def _infer_line_kind(self, prefix_state: KernPrefixState) -> SpineLineKind:
        if not prefix_state.completed_fields and prefix_state.current_field_buffer == "":
            return SpineLineKind.UNKNOWN
        if all(
            field in _SPINE_OP_FIELDS for field in prefix_state.completed_fields
        ) and _is_spine_op_prefix(prefix_state.current_field_buffer):
            return SpineLineKind.SPINE_OP
        return SpineLineKind.FIXED_WIDTH
