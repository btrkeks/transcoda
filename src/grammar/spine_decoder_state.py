"""Compatibility wrapper for width-aware **kern decoding state."""

from __future__ import annotations

from dataclasses import dataclass, field

from .kern_prefix_state import KernPrefixState, KernPrefixStateError
from .spine_structure_rule import SpineLineKind, SpineLineClosePreview, SpineStructureRule


class SpineDecoderStateError(ValueError):
    """Raised when a partial decode violates spine-structure invariants."""


@dataclass
class SpineDecoderState:
    """Track spine width and line state for one generated sequence."""

    prefix_state: KernPrefixState = field(default_factory=KernPrefixState)
    rule: SpineStructureRule = field(default_factory=SpineStructureRule)

    @property
    def active_spines(self) -> int | None:
        return self.rule.active_spines

    @property
    def completed_fields(self) -> list[str]:
        return self.prefix_state.completed_fields

    @property
    def current_field_buffer(self) -> str:
        return self.prefix_state.current_field_buffer

    @property
    def current_line_kind(self) -> SpineLineKind:
        return self.rule.current_line_kind

    @property
    def terminated(self) -> bool:
        return self.rule.terminated

    def append_text(self, text: str) -> None:
        try:
            self.prefix_state.append_text(text)
            self.rule.on_text_appended(self.prefix_state)
        except (KernPrefixStateError, ValueError) as exc:
            raise SpineDecoderStateError(str(exc)) from exc

    def append_tab(self) -> None:
        try:
            self.prefix_state.append_tab()
            self.rule.on_tab_appended(self.prefix_state)
        except (KernPrefixStateError, ValueError) as exc:
            raise SpineDecoderStateError(str(exc)) from exc

    def append_newline(self) -> SpineLineClosePreview:
        try:
            preview = self.preview_line_close()
            self.prefix_state.consume_line_close()
            self.rule.on_line_closed(preview.fields)
            return preview
        except (KernPrefixStateError, ValueError) as exc:
            raise SpineDecoderStateError(str(exc)) from exc

    def accept_token_text(self, text: str) -> SpineLineClosePreview | None:
        if text == "\t":
            self.append_tab()
            return None
        if text == "\n":
            return self.append_newline()
        self.append_text(text)
        return None

    def can_accept_tab(self) -> bool:
        return self.rule.can_accept_tab(self.prefix_state)

    def preview_line_close(self) -> SpineLineClosePreview:
        try:
            fields = self.prefix_state.preview_line_close().fields
            return self.rule.preview_line_close(fields)
        except (KernPrefixStateError, ValueError) as exc:
            raise SpineDecoderStateError(str(exc)) from exc
