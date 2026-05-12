"""Structural bias rule for interpretation transitions at new-line start."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import torch

from src.core.spine_state import advance_spine_count

from .kern_prefix_state import KernPrefixState
from .stateful_kern_logits_processor import KernSemanticRule, TokenizerConstraintContext

ClosedLineType = Literal["data", "bar", "interp", "spine_op", "unknown"]
LineMode = Literal["unknown", "data", "bar", "interp", "spine_op"]

_SPINE_OP_FIELDS = {"*", "*^", "*v"}
_WIDTH_TRACKING_FIELDS = _SPINE_OP_FIELDS | {"*-"}


@dataclass(frozen=True)
class InterpretationTransitionConfig:
    """Score-shaping knobs for line-start interpretation transitions."""

    non_spine_bonus: float = 1.5
    null_interp_bonus: float = 1.0
    data_start_penalty: float = -1.0
    barline_start_penalty: float = -1.0


def resolve_interpretation_transition_config(training_cfg: object) -> InterpretationTransitionConfig:
    """Resolve interpretation-transition config from training config values."""

    return InterpretationTransitionConfig(
        non_spine_bonus=float(
            getattr(training_cfg, "interpretation_transition_non_spine_bonus", 1.5)
        ),
        null_interp_bonus=float(
            getattr(training_cfg, "interpretation_transition_null_interp_bonus", 1.0)
        ),
        data_start_penalty=float(
            getattr(training_cfg, "interpretation_transition_data_start_penalty", -1.0)
        ),
        barline_start_penalty=float(
            getattr(training_cfg, "interpretation_transition_barline_start_penalty", -1.0)
        ),
    )


@dataclass
class InterpretationTransitionRule(KernSemanticRule):
    """Bias first-field choices toward interpretation starts in transition-heavy contexts."""

    config: InterpretationTransitionConfig = field(default_factory=InterpretationTransitionConfig)
    active_spines: int | None = None
    last_closed_line_type: ClosedLineType = "unknown"
    previous_closed_line_type: ClosedLineType = "unknown"
    last_line_had_dot_fields: bool = False
    recent_spine_topology_change_age: int | None = None
    line_mode: LineMode = "unknown"
    _stats: Counter[str] = field(default_factory=Counter)
    _activation_reason_counts: Counter[str] = field(default_factory=Counter)

    @property
    def terminated(self) -> bool:
        return self.active_spines == 0

    def on_text_appended(self, prefix_state: KernPrefixState) -> None:
        if self.line_mode != "unknown":
            return
        if prefix_state.completed_fields:
            return
        if prefix_state.current_field_buffer == "":
            return

        self.line_mode = _infer_line_mode(prefix_state.current_field_buffer)
        if self.line_mode == "interp":
            self._stats["first_field_started_as_interp"] += 1
        else:
            self._stats["first_field_started_as_data"] += 1

    def on_tab_appended(self, prefix_state: KernPrefixState) -> None:
        if self.line_mode != "unknown":
            return
        if not prefix_state.completed_fields:
            return
        self.line_mode = _infer_line_mode(prefix_state.completed_fields[0])

    def on_line_closed(self, fields: tuple[str, ...]) -> None:
        prior_spines = self.active_spines if self.active_spines is not None else len(fields)
        self.previous_closed_line_type = self.last_closed_line_type
        self.last_closed_line_type = _classify_closed_line(fields)
        self.last_line_had_dot_fields = any(field == "." for field in fields)
        self.recent_spine_topology_change_age = _next_topology_change_age(
            fields,
            self.recent_spine_topology_change_age,
        )
        self.line_mode = "unknown"

        if fields and all(field in _WIDTH_TRACKING_FIELDS for field in fields):
            self.active_spines = advance_spine_count(prior_spines, fields)
        elif self.active_spines is None:
            self.active_spines = len(fields)

    def can_accept_tab(self, prefix_state: KernPrefixState) -> bool:
        return True

    def can_close_line(self, fields: tuple[str, ...]) -> bool:
        return True

    def can_end_sequence(self, fields: tuple[str, ...]) -> bool:
        return True

    def mask_scores(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
    ) -> None:
        if not _is_first_field_choice(prefix_state):
            return
        if self.line_mode != "unknown":
            return
        if self.active_spines is None or self.active_spines < 2:
            return

        reason_codes = self._transition_reason_codes()
        if not reason_codes:
            return

        self._stats["transition_context_activations"] += 1
        for reason in reason_codes:
            self._activation_reason_counts[reason] += 1

        top_before = _top_finite_token_id(row_scores)
        applied = False
        applied |= _apply_bias(
            row_scores,
            context.interpretation.non_spine_interp_token_ids,
            self.config.non_spine_bonus,
        )
        applied |= _apply_bias(
            row_scores,
            context.interpretation.null_interpretation_token_ids,
            self.config.null_interp_bonus,
        )
        applied |= _apply_bias(
            row_scores,
            context.interpretation.non_control_data_token_ids,
            self.config.data_start_penalty,
        )
        applied |= _apply_bias(
            row_scores,
            context.interpretation.barline_token_ids,
            self.config.barline_start_penalty,
        )
        if not applied:
            return

        self._stats["first_field_interp_bias_applied"] += 1
        top_after = _top_finite_token_id(row_scores)
        if top_before is not None and top_after is not None and top_before != top_after:
            self._stats["first_field_bias_flipped_top_choice"] += 1

    def stats(self) -> dict[str, object]:
        payload: dict[str, object] = dict(self._stats)
        if self._activation_reason_counts:
            payload["activation_reason_counts"] = dict(self._activation_reason_counts)
        return payload

    def _transition_reason_codes(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.last_closed_line_type == "spine_op":
            reasons.append("after_spine_op")
        if self.previous_closed_line_type == "spine_op":
            reasons.append("after_previous_spine_op")
        if self.last_closed_line_type == "interp":
            reasons.append("after_interp")
        if self.last_line_had_dot_fields:
            reasons.append("after_dot_heavy_data")
        if self.recent_spine_topology_change_age is not None:
            reasons.append("recent_topology_change")
        return tuple(reasons)


def _is_first_field_choice(prefix_state: KernPrefixState) -> bool:
    return not prefix_state.completed_fields and prefix_state.current_field_buffer == ""


def _infer_line_mode(field_text: str) -> LineMode:
    token = field_text.strip()
    if token in _SPINE_OP_FIELDS:
        return "spine_op"
    if token.startswith("*"):
        return "interp"
    if token.startswith("="):
        return "bar"
    return "data"


def _classify_closed_line(fields: tuple[str, ...]) -> ClosedLineType:
    if not fields:
        return "unknown"
    if all(field in _SPINE_OP_FIELDS for field in fields):
        return "spine_op"
    if all(field == "*-" for field in fields):
        return "unknown"
    if all(field.startswith("*") for field in fields):
        return "interp"
    if all(field.startswith("=") for field in fields):
        return "bar"
    if any(field.startswith("*") or field.startswith("=") for field in fields):
        return "unknown"
    return "data"


def _next_topology_change_age(
    fields: tuple[str, ...],
    previous_age: int | None,
) -> int | None:
    if any(field in {"*^", "*v"} for field in fields):
        return 0
    if previous_age is None:
        return None
    next_age = previous_age + 1
    return next_age if next_age <= 2 else None


def _apply_bias(row_scores: torch.FloatTensor, token_ids: list[int], amount: float) -> bool:
    applied = False
    for token_id in token_ids:
        if token_id < 0 or token_id >= row_scores.shape[0]:
            continue
        if not torch.isfinite(row_scores[token_id]):
            continue
        row_scores[token_id] += amount
        applied = True
    return applied


def _top_finite_token_id(row_scores: torch.FloatTensor) -> int | None:
    finite_mask = torch.isfinite(row_scores)
    if not bool(finite_mask.any()):
        return None
    candidate_ids = torch.nonzero(finite_mask, as_tuple=False).flatten()
    best_index = int(torch.argmax(row_scores[candidate_ids]).item())
    return int(candidate_ids[best_index].item())
