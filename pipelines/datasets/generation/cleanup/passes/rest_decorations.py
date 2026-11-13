"""Remove unsupported decoration markers from rest tokens."""

from __future__ import annotations

from ..base import CleanupContext

REST_DECORATION_CHARS = "()[]_LJ@Z<,"
_REST_DECORATION_TRANS = str.maketrans({char: None for char in REST_DECORATION_CHARS})


def _is_data_field(field: str) -> bool:
    return bool(field) and not field.startswith(("*", "=", "!")) and field != "."


def _is_rest_token(token: str) -> bool:
    return "r" in token


class RemoveRestDecorations:
    """Strip parser-irrelevant markers from rest tokens only."""

    name = "remove_rest_decorations"

    def transform(self, text: str, ctx: CleanupContext) -> str:
        changed_tokens = 0
        output_lines: list[str] = []

        for line in text.splitlines():
            output_fields: list[str] = []
            for field in line.split("\t"):
                if not _is_data_field(field):
                    output_fields.append(field)
                    continue

                output_tokens: list[str] = []
                for token in field.split(" "):
                    if _is_rest_token(token):
                        cleaned = token.translate(_REST_DECORATION_TRANS)
                        if cleaned != token:
                            changed_tokens += 1
                        output_tokens.append(cleaned)
                    else:
                        output_tokens.append(token)

                output_fields.append(" ".join(output_tokens))

            output_lines.append("\t".join(output_fields))

        ctx[self.name] = {"changed_tokens": changed_tokens}
        return "\n".join(output_lines)
