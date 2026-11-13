"""Semantic width-preservation rule for stateful **kern decoding."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum

import torch

from .kern_prefix_state import KernPrefixState
from .spine_lineage import LINEAGE_FIELDS, MERGE_FIELD, SHARED_FIELD_PREFIXES, SpineLineage
from .stateful_kern_logits_processor import KernSemanticRuleBase, TokenizerConstraintContext

# The model dialect terminates with EOS, not Humdrum ``*-``. Keep ``*-`` out
# of structural spine operations so it never forces decoder termination.
_SPINE_MERGE_FIELD = MERGE_FIELD
_SPINE_OP_FIELDS = {"*", "*^", _SPINE_MERGE_FIELD}
_SYSTEM_BREAK_PREFIX = "!!LO:LB"
_SYSTEM_BREAK_GROUP_PREFIX = "!!LO:LB:g="
_SYSTEM_BREAK_GROUP_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
)


class BarlineWidthMode(StrEnum):
    """Allowed target dialect widths for barline lines."""

    PER_SPINE = "per_spine"
    COMPACT = "compact"
    BOTH = "both"


class EmptyFieldMode(StrEnum):
    """Allowed target dialect modes for omitted null data fields."""

    FORBID = "forbid"
    DATA_ONLY = "data_only"


@dataclass(frozen=True)
class BarlineWidthPolicy:
    """Width policy for barline lines in the model target dialect."""

    mode: BarlineWidthMode = BarlineWidthMode.PER_SPINE

    @classmethod
    def from_value(cls, value: BarlineWidthMode | str) -> BarlineWidthPolicy:
        try:
            mode = value if isinstance(value, BarlineWidthMode) else BarlineWidthMode(str(value))
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in BarlineWidthMode)
            raise ValueError(f"barline_width_mode must be one of {allowed}") from exc
        return cls(mode=mode)

    def allows_width(self, width: int, active_spines: int | None) -> bool:
        if active_spines is None:
            return width == 1
        if self.mode is BarlineWidthMode.PER_SPINE:
            return width == active_spines
        if self.mode is BarlineWidthMode.COMPACT:
            return width == 1
        if self.mode is BarlineWidthMode.BOTH:
            return width in {1, active_spines}
        raise AssertionError(f"Unhandled barline width mode: {self.mode}")

    def can_accept_tab(self, current_width: int, active_spines: int | None) -> bool:
        if active_spines is None:
            return False
        if self.mode is BarlineWidthMode.COMPACT:
            return False
        return current_width < active_spines

    def expected_width_description(self, active_spines: int) -> str:
        if self.mode is BarlineWidthMode.PER_SPINE:
            return f"{active_spines} fields"
        if self.mode is BarlineWidthMode.COMPACT:
            return "1 field"
        if self.mode is BarlineWidthMode.BOTH:
            return f"1 or {active_spines} fields"
        raise AssertionError(f"Unhandled barline width mode: {self.mode}")


@dataclass(frozen=True)
class EmptyFieldPolicy:
    """Policy for empty fields created by compact omitted-null data records."""

    mode: EmptyFieldMode = EmptyFieldMode.FORBID

    @classmethod
    def from_value(cls, value: EmptyFieldMode | str) -> EmptyFieldPolicy:
        try:
            mode = value if isinstance(value, EmptyFieldMode) else EmptyFieldMode(str(value))
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in EmptyFieldMode)
            raise ValueError(f"empty_field_mode must be one of {allowed}") from exc
        return cls(mode=mode)

    @property
    def supports_empty_fields(self) -> bool:
        return self.mode is EmptyFieldMode.DATA_ONLY


class SpineLineKind(StrEnum):
    """Structural class of the line currently being decoded."""

    UNKNOWN = "unknown"
    SYSTEM_BREAK = "system_break"
    BARLINE = "barline"
    SPINE_OP = "spine_op"
    FIXED_WIDTH = "fixed_width"


def _is_spine_op_prefix(text: str) -> bool:
    return any(field.startswith(text) for field in _SPINE_OP_FIELDS)


def _is_system_break_namespace(text: str) -> bool:
    return text == _SYSTEM_BREAK_PREFIX or text.startswith(f"{_SYSTEM_BREAK_PREFIX}:")


def _is_system_break_line(text: str) -> bool:
    if text == _SYSTEM_BREAK_PREFIX:
        return True
    if not text.startswith(_SYSTEM_BREAK_GROUP_PREFIX):
        return False
    group = text.removeprefix(_SYSTEM_BREAK_GROUP_PREFIX)
    return bool(group) and all(char in _SYSTEM_BREAK_GROUP_CHARS for char in group)


def _is_system_break_prefix(text: str) -> bool:
    if not text.startswith("!!"):
        return False
    if _SYSTEM_BREAK_PREFIX.startswith(text):
        return True
    if _SYSTEM_BREAK_GROUP_PREFIX.startswith(text):
        return True
    if text.startswith(_SYSTEM_BREAK_GROUP_PREFIX):
        group = text.removeprefix(_SYSTEM_BREAK_GROUP_PREFIX)
        return all(char in _SYSTEM_BREAK_GROUP_CHARS for char in group)
    return False


def _is_barline_field(text: str) -> bool:
    return text.startswith("=")


def _is_barline_line(fields: tuple[str, ...]) -> bool:
    return bool(fields) and all(_is_barline_field(field) for field in fields)


def _is_barline_prefix(prefix_state: KernPrefixState) -> bool:
    return (
        bool(prefix_state.current_field_buffer)
        and _is_barline_field(prefix_state.current_field_buffer)
        and all(_is_barline_field(field) for field in prefix_state.completed_fields)
    )


def _has_empty_field(fields: tuple[str, ...] | list[str]) -> bool:
    return any(field == "" for field in fields)


def _is_structural_field(field: str) -> bool:
    return field.startswith(("*", "=", "!"))


def _has_structural_field(fields: tuple[str, ...] | list[str]) -> bool:
    return any(_is_structural_field(field) for field in fields if field)


def _trailing_merge_run_length(fields: tuple[str, ...] | list[str]) -> int:
    run_length = 0
    for field in reversed(fields):
        if field != _SPINE_MERGE_FIELD:
            break
        run_length += 1
    return run_length


def _is_delimiter_or_special_token(token_text: str) -> bool:
    return token_text in {"\t", "\n", "<bos>", "<eos>", "<pad>"}


@dataclass(frozen=True)
class SpineLineClosePreview:
    """Computed outcome of closing a line under spine-width semantics."""

    fields: tuple[str, ...]
    line_kind: SpineLineKind
    prior_spines: int
    next_lineage: SpineLineage
    establishes_spine_width: bool = True

    @property
    def next_spines(self) -> int:
        return len(self.next_lineage.tracks)

    @property
    def humdrum_terminated(self) -> bool:
        return self.next_spines == 0

    @property
    def terminated(self) -> bool:
        return self.humdrum_terminated


@dataclass(frozen=True)
class SpineStructureRuleFactory:
    """Callable factory carrying the target dialect options for the rule."""

    barline_width_mode: BarlineWidthMode | str = BarlineWidthMode.PER_SPINE
    empty_field_mode: EmptyFieldMode | str = EmptyFieldMode.FORBID

    def __post_init__(self) -> None:
        barline_policy = BarlineWidthPolicy.from_value(self.barline_width_mode)
        empty_field_policy = EmptyFieldPolicy.from_value(self.empty_field_mode)
        object.__setattr__(self, "barline_width_mode", barline_policy.mode)
        object.__setattr__(self, "empty_field_mode", empty_field_policy.mode)

    def __call__(self) -> SpineStructureRule:
        return SpineStructureRule(
            barline_width_mode=self.barline_width_mode,
            empty_field_mode=self.empty_field_mode,
        )


@dataclass
class SpineStructureRule(KernSemanticRuleBase):
    """Track width-aware spine invariants for one generated sequence."""

    barline_width_mode: BarlineWidthMode | str = BarlineWidthMode.PER_SPINE
    empty_field_mode: EmptyFieldMode | str = EmptyFieldMode.FORBID
    lineage: SpineLineage | None = None
    initial_spines: int | None = None
    current_line_kind: SpineLineKind = SpineLineKind.UNKNOWN
    _humdrum_terminated: bool = False
    _pending_system_break: bool = False
    _barline_width_policy: BarlineWidthPolicy = dataclass_field(init=False)
    _empty_field_policy: EmptyFieldPolicy = dataclass_field(init=False)

    def __post_init__(self) -> None:
        self._barline_width_policy = BarlineWidthPolicy.from_value(self.barline_width_mode)
        self._empty_field_policy = EmptyFieldPolicy.from_value(self.empty_field_mode)
        self.barline_width_mode = self._barline_width_policy.mode
        self.empty_field_mode = self._empty_field_policy.mode

    @property
    def active_spines(self) -> int | None:
        return None if self.lineage is None else len(self.lineage.tracks)

    @property
    def humdrum_terminated(self) -> bool:
        return self._humdrum_terminated

    @property
    def terminated(self) -> bool:
        return self.humdrum_terminated

    @property
    def supports_empty_fields(self) -> bool:
        return self._empty_field_policy.supports_empty_fields

    def on_text_appended(self, prefix_state: KernPrefixState) -> None:
        self.current_line_kind = self._infer_line_kind(prefix_state)

    def on_tab_appended(self, prefix_state: KernPrefixState) -> None:
        self.current_line_kind = self._infer_line_kind(prefix_state)

    def on_line_closed(self, fields: tuple[str, ...]) -> None:
        preview = self.preview_line_close(fields)
        if self.initial_spines is None and preview.establishes_spine_width:
            self.initial_spines = preview.prior_spines
        if preview.establishes_spine_width or self.lineage is not None:
            self.lineage = preview.next_lineage
        self._humdrum_terminated = preview.humdrum_terminated
        self._pending_system_break = preview.line_kind is SpineLineKind.SYSTEM_BREAK
        self.current_line_kind = SpineLineKind.UNKNOWN

    def can_accept_tab(self, prefix_state: KernPrefixState) -> bool:
        if prefix_state.current_field_buffer == "":
            return self.can_accept_empty_field(prefix_state)
        if _has_empty_field(prefix_state.completed_fields):
            if not self._empty_field_policy.supports_empty_fields:
                return False
            if self.active_spines is None:
                return False
            if _has_structural_field(
                [*prefix_state.completed_fields, prefix_state.current_field_buffer]
            ):
                return False
            return len(prefix_state.completed_fields) + 1 < self.active_spines
        if (
            not prefix_state.completed_fields
            and _is_system_break_prefix(prefix_state.current_field_buffer)
        ):
            return False
        if _is_barline_prefix(prefix_state):
            current_width = len(prefix_state.completed_fields) + 1
            return self._barline_width_policy.can_accept_tab(
                current_width,
                self.active_spines,
            )
        if self.active_spines is None:
            return True
        return len(prefix_state.completed_fields) + 1 < self.active_spines

    def can_accept_empty_field(self, prefix_state: KernPrefixState) -> bool:
        if not self._empty_field_policy.supports_empty_fields:
            return False
        if prefix_state.current_field_buffer != "":
            return False
        if self.active_spines is None:
            return False

        candidate_fields = [*prefix_state.completed_fields, ""]
        current_width = len(candidate_fields)
        if current_width >= self.active_spines:
            return False
        return not _has_structural_field(candidate_fields)

    def can_close_line(self, fields: tuple[str, ...]) -> bool:
        try:
            self.preview_line_close(fields)
        except ValueError:
            return False
        return True

    def can_close_line_with_empty_field(
        self,
        prefix_state: KernPrefixState,
        fields: tuple[str, ...],
    ) -> bool:
        _ = prefix_state
        return self.can_close_line(fields)

    def can_end_sequence(self, fields: tuple[str, ...]) -> bool:
        try:
            preview = self.preview_line_close(fields)
        except ValueError:
            return False
        return preview.line_kind is not SpineLineKind.SYSTEM_BREAK

    def can_end_at_line_boundary(self, prefix_state: KernPrefixState) -> bool:
        return (
            not self._pending_system_break
            and prefix_state.closed_line_count > 0
            and self.active_spines is not None
        )

    def mask_scores(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
    ) -> None:
        self._mask_merge_dead_ends(prefix_state, row_scores, context)
        self._mask_structural_tokens_after_empty_field(prefix_state, row_scores, context)
        self._mask_non_barline_tokens_after_system_break(
            prefix_state,
            row_scores,
            context,
        )
        if prefix_state.current_field_buffer == "":
            return
        if not self.can_accept_tab(prefix_state):
            row_scores[context.tab_token_id] = float("-inf")

    def _mask_non_barline_tokens_after_system_break(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
    ) -> None:
        if not self._pending_system_break:
            return
        if prefix_state.current_field_buffer:
            return
        for token_id, token_text in context.i2w.items():
            if _is_delimiter_or_special_token(token_text):
                continue
            if token_text.startswith("="):
                continue
            if 0 <= token_id < row_scores.shape[0]:
                row_scores[token_id] = float("-inf")

    def _mask_structural_tokens_after_empty_field(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
    ) -> None:
        if not self._empty_field_policy.supports_empty_fields:
            return
        if not _has_empty_field(prefix_state.completed_fields):
            return
        if prefix_state.current_field_buffer and not _is_structural_field(
            prefix_state.current_field_buffer
        ):
            return
        for token_id, token_text in context.i2w.items():
            if _is_delimiter_or_special_token(token_text):
                continue
            candidate = prefix_state.current_field_buffer + token_text
            if _is_structural_field(candidate) and 0 <= token_id < row_scores.shape[0]:
                row_scores[token_id] = float("-inf")

    def _mask_merge_dead_ends(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
    ) -> None:
        lineage = self.lineage
        if lineage is None:
            return

        column = len(prefix_state.completed_fields)
        if column >= len(lineage.tracks):
            return

        if _trailing_merge_run_length(prefix_state.completed_fields) == 1:
            self._mask_tokens_that_cannot_complete_field(
                prefix_state,
                row_scores,
                context,
                required_fields=(_SPINE_MERGE_FIELD,),
            )
            return

        sibling_spellings = lineage.sibling_spellings(column, prefix_state.completed_fields)
        if sibling_spellings is not None:
            self._mask_tokens_that_cannot_complete_field(
                prefix_state,
                row_scores,
                context,
                required_fields=sibling_spellings,
            )
            return

        forbidden = lineage.forbidden_fields(column, prefix_state.completed_fields)
        allowed = LINEAGE_FIELDS - forbidden
        for token_id, candidate in context.lineage_field_tokens.get(
            prefix_state.current_field_buffer, ()
        ):
            if candidate in SHARED_FIELD_PREFIXES:
                continue
            if any(field.startswith(candidate) for field in allowed):
                continue
            if any(field.startswith(candidate) for field in forbidden):
                row_scores[token_id] = float("-inf")

    def _mask_tokens_that_cannot_complete_field(
        self,
        prefix_state: KernPrefixState,
        row_scores: torch.FloatTensor,
        context: TokenizerConstraintContext,
        *,
        required_fields: tuple[str, ...],
    ) -> None:
        current = prefix_state.current_field_buffer
        for token_id, token_text in context.i2w.items():
            if current in required_fields and _is_delimiter_or_special_token(token_text):
                continue
            candidate = current + token_text
            if any(field.startswith(candidate) for field in required_fields):
                continue
            if 0 <= token_id < row_scores.shape[0]:
                row_scores[token_id] = float("-inf")

    def preview_line_close(self, fields: tuple[str, ...]) -> SpineLineClosePreview:
        lineage = self.lineage if self.lineage is not None else SpineLineage.initial(len(fields))
        prior_spines = len(lineage.tracks)
        if self._pending_system_break and not _is_barline_line(fields):
            raise ValueError("system break must be followed immediately by a barline")
        if _has_empty_field(fields):
            return self._preview_empty_data_line(fields, prior_spines=prior_spines)
        if len(fields) == 1 and _is_system_break_line(fields[0]):
            return SpineLineClosePreview(
                fields=fields,
                line_kind=SpineLineKind.SYSTEM_BREAK,
                prior_spines=prior_spines,
                next_lineage=lineage,
                establishes_spine_width=False,
            )
        if len(fields) > 1 and any(_is_system_break_namespace(field) for field in fields):
            raise ValueError("system break lines must be a single field")
        if _is_barline_line(fields):
            if not self._barline_width_policy.allows_width(len(fields), self.active_spines):
                expected = (
                    "1 field"
                    if self.active_spines is None
                    else self._barline_width_policy.expected_width_description(
                        self.active_spines
                    )
                )
                raise ValueError(f"barline line expected {expected}, got {len(fields)}")
            return SpineLineClosePreview(
                fields=fields,
                line_kind=SpineLineKind.BARLINE,
                prior_spines=prior_spines,
                next_lineage=lineage,
                establishes_spine_width=False,
            )
        if len(fields) != prior_spines:
            raise ValueError(f"expected {prior_spines} fields before line close, got {len(fields)}")

        if all(field in _SPINE_OP_FIELDS for field in fields):
            line_kind = SpineLineKind.SPINE_OP
        else:
            line_kind = SpineLineKind.FIXED_WIDTH
        return SpineLineClosePreview(
            fields=fields,
            line_kind=line_kind,
            prior_spines=prior_spines,
            next_lineage=lineage.advance(fields),
        )

    def _preview_empty_data_line(
        self,
        fields: tuple[str, ...],
        *,
        prior_spines: int,
    ) -> SpineLineClosePreview:
        if not self._empty_field_policy.supports_empty_fields:
            raise ValueError("empty fields are disabled")
        if self.lineage is None:
            raise ValueError("empty data fields require known active spine count")
        if len(fields) != prior_spines:
            raise ValueError(f"expected {prior_spines} fields before line close, got {len(fields)}")
        if not any(fields):
            raise ValueError("data records cannot contain only empty fields")
        if _has_structural_field(fields):
            raise ValueError("empty fields are only allowed in data records")
        return SpineLineClosePreview(
            fields=fields,
            line_kind=SpineLineKind.FIXED_WIDTH,
            prior_spines=prior_spines,
            next_lineage=self.lineage,
        )

    def _infer_line_kind(self, prefix_state: KernPrefixState) -> SpineLineKind:
        if not prefix_state.completed_fields and prefix_state.current_field_buffer == "":
            return SpineLineKind.UNKNOWN
        if (
            not prefix_state.completed_fields
            and _is_system_break_prefix(prefix_state.current_field_buffer)
        ):
            return SpineLineKind.SYSTEM_BREAK
        if _is_barline_prefix(prefix_state):
            return SpineLineKind.BARLINE
        if all(
            field in _SPINE_OP_FIELDS for field in prefix_state.completed_fields
        ) and _is_spine_op_prefix(prefix_state.current_field_buffer):
            return SpineLineKind.SPINE_OP
        return SpineLineKind.FIXED_WIDTH
