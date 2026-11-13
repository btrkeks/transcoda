"""Delimiter-safe prefix buffering for stateful **kern decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class KernPrefixStateError(ValueError):
    """Raised when prefix buffering invariants are violated."""


class PrefixPositionKind(StrEnum):
    """Delimiter position of the current decode prefix."""

    START = "start"
    FIELD_TEXT = "field_text"
    AFTER_TAB = "after_tab"
    LINE_BOUNDARY = "line_boundary"


@dataclass(frozen=True)
class LineBufferClosePreview:
    """Current line fields if the line were closed now."""

    fields: tuple[str, ...]


@dataclass
class KernPrefixState:
    """Track the delimiter-structured text prefix for one decoded sequence."""

    completed_fields: list[str] = field(default_factory=list)
    current_field_buffer: str = ""
    closed_line_count: int = 0

    @property
    def position_kind(self) -> PrefixPositionKind:
        if self.current_field_buffer:
            return PrefixPositionKind.FIELD_TEXT
        if self.completed_fields:
            return PrefixPositionKind.AFTER_TAB
        if self.closed_line_count > 0:
            return PrefixPositionKind.LINE_BOUNDARY
        return PrefixPositionKind.START

    def append_text(self, text: str) -> None:
        if "\t" in text or "\n" in text:
            raise KernPrefixStateError("append_text requires delimiter-free text")
        self.current_field_buffer += text

    def append_tab(self) -> None:
        self.completed_fields.append(self.current_field_buffer)
        self.current_field_buffer = ""

    def preview_line_close(self) -> LineBufferClosePreview:
        return LineBufferClosePreview(
            fields=tuple(self.completed_fields + [self.current_field_buffer]),
        )

    def consume_line_close(self) -> LineBufferClosePreview:
        preview = self.preview_line_close()
        self.completed_fields.clear()
        self.current_field_buffer = ""
        self.closed_line_count += 1
        return preview

    def accept_token_text(self, text: str) -> LineBufferClosePreview | None:
        if text == "\t":
            self.append_tab()
            return None
        if text == "\n":
            return self.consume_line_close()
        self.append_text(text)
        return None
