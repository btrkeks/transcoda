"""Remove Grandstaff-only single editorial y markers from note tokens."""

from __future__ import annotations

import re

from transcoda.core.kern_utils import is_note_token

from ..base import CleanupContext

_SINGLE_Y_RE = re.compile(r"(?<!y)y(?!y)")


def _is_data_field(field: str) -> bool:
    return bool(field) and not field.startswith(("*", "=", "!")) and field != "."


def _remove_single_y(token: str) -> str:
    return _SINGLE_Y_RE.sub("", token)


class RemoveGrandstaffSingleYFromNotes:
    """Strip isolated ``y`` editorial markers from Grandstaff note tokens."""

    name = "remove_grandstaff_single_y_from_notes"

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
                    if is_note_token(token):
                        cleaned = _remove_single_y(token)
                        if cleaned != token:
                            changed_tokens += 1
                        output_tokens.append(cleaned)
                    else:
                        output_tokens.append(token)

                output_fields.append(" ".join(output_tokens))

            output_lines.append("\t".join(output_fields))

        ctx[self.name] = {"changed_tokens": changed_tokens}
        return "\n".join(output_lines)
