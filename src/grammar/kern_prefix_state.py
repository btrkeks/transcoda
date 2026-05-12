"""Delimiter-safe prefix buffering for stateful **kern decoding."""

from __future__ import annotations

from dataclasses import dataclass, field


class KernPrefixStateError(ValueError):
    """Raised when prefix buffering invariants are violated."""


@dataclass(frozen=True)
class LineBufferClosePreview:
    """Current line fields if the line were closed now."""

    fields: tuple[str, ...]


@dataclass
class KernPrefixState:
    """Track the delimiter-structured text prefix for one decoded sequence."""

    completed_fields: list[str] = field(default_factory=list)
    current_field_buffer: str = ""

    def append_text(self, text: str) -> None:
        if "\t" in text or "\n" in text:
            raise KernPrefixStateError("append_text requires delimiter-free text")
        self.current_field_buffer += text

    def append_tab(self) -> None:
        if self.current_field_buffer == "":
            raise KernPrefixStateError("cannot append tab after an empty field")
        self.completed_fields.append(self.current_field_buffer)
        self.current_field_buffer = ""

    def preview_line_close(self) -> LineBufferClosePreview:
        if self.current_field_buffer == "":
            raise KernPrefixStateError("cannot close a line with an empty field")
        return LineBufferClosePreview(
            fields=tuple(self.completed_fields + [self.current_field_buffer]),
        )

    def consume_line_close(self) -> LineBufferClosePreview:
        preview = self.preview_line_close()
        self.completed_fields.clear()
        self.current_field_buffer = ""
        return preview

    def accept_token_text(self, text: str) -> LineBufferClosePreview | None:
        if text == "\t":
            self.append_tab()
            return None
        if text == "\n":
            return self.consume_line_close()
        self.append_text(text)
        return None
