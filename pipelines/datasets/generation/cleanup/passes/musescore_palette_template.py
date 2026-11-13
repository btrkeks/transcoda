"""Remove the MuseScore palette template from O'Neill dance-tune conversions.

These MuseScore sources share a template that opens with a fixed first bar and
an ornament "palette" note (trill, fermata, mordent, inverted mordent) as the
first event of bar two. The tune itself starts right after the palette note.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ..base import CleanupContext

_NOTE_HEAD = re.compile(r"^[\[({&]*\d+\.*[qQ]*([a-gA-G])\1*")
_PALETTE_SIGNIFIER_GROUPS = ("Tt", ";", "Mm", "Ww")


def _is_data(line: str) -> bool:
    return bool(line) and not line.startswith(("!", "*", "="))


def _first(lines: list[str], start: int, predicate: Callable[[str], bool]) -> int | None:
    return next((i for i in range(start, len(lines)) if predicate(lines[i])), None)


def _is_palette(token: str) -> bool:
    match = _NOTE_HEAD.match(token)
    if match is None:
        return False
    signifiers = token[match.end() :]
    return all(any(char in signifiers for char in group) for group in _PALETTE_SIGNIFIER_GROUPS)


def _find_palette(line: str) -> tuple[int, str] | None:
    for column, field in enumerate(line.split("\t")):
        for token in field.split(" "):
            if _is_palette(token):
                return column, token
    return None


def _edit_signifiers(token: str, edit: Callable[[str], str]) -> str:
    match = _NOTE_HEAD.match(token)
    if match is None:
        return token
    return token[: match.end()] + edit(token[match.end() :])


def _strip_mordents(signifiers: str) -> str:
    return re.sub("[mMwW]", "", signifiers)


def _take_over_beam(signifiers: str) -> str:
    if "J" in signifiers:
        signifiers = signifiers.replace("J", "", 1)
        if "L" not in signifiers and "J" not in signifiers:
            signifiers = re.sub("[kK]", "", signifiers)
    elif "L" in signifiers:
        signifiers = signifiers.replace("L", "LL", 1)
    else:
        signifiers += "L"
    return signifiers


def _is_beamable(token: str) -> bool:
    return _NOTE_HEAD.match(token) is not None and not re.search("[qQ]", token)


def _edit_field(lines: list[str], index: int, column: int, edit: Callable[[str], str]) -> None:
    fields = lines[index].split("\t")
    fields[column] = " ".join(_edit_signifiers(token, edit) for token in fields[column].split(" "))
    lines[index] = "\t".join(fields)


def _drop_slur_close(lines: list[str], start: int, column: int, width: int) -> None:
    depth = 1
    for index in range(start, len(lines)):
        fields = lines[index].split("\t")
        if not _is_data(lines[index]) or len(fields) != width:
            continue
        field = fields[column]
        for position, char in enumerate(field):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    fields[column] = field[:position] + field[position + 1 :]
                    lines[index] = "\t".join(fields)
                    return


class RemoveMuseScorePaletteTemplate:
    """Drop the template first bar and palette note, keeping the tune."""

    name = "remove_musescore_palette_template"

    def transform(self, text: str, ctx: CleanupContext) -> str:
        ctx[self.name] = {"removed_templates": 0}
        lines = text.split("\n")
        first_data = _first(lines, 0, _is_data)
        if first_data is None:
            return text
        barline = _first(lines, first_data + 1, lambda line: line.startswith("="))
        if barline is None:
            return text
        palette_line = _first(lines, barline + 1, _is_data)
        if palette_line is None:
            return text
        found = _find_palette(lines[palette_line])
        if found is None:
            return text

        column, palette = found
        width = len(lines[palette_line].split("\t"))

        def in_column(line: str) -> list[str] | None:
            fields = line.split("\t")
            return fields[column].split(" ") if _is_data(line) and len(fields) == width else None

        next_line = _first(lines, palette_line + 1, _is_data)
        if next_line is not None and in_column(lines[next_line]) is not None:
            _edit_field(lines, next_line, column, _strip_mordents)
        if "L" in palette:
            # Grace notes sit outside the beam, so the first regular note inherits it.
            beam_line = _first(
                lines,
                palette_line + 1,
                lambda line: any(_is_beamable(token) for token in in_column(line) or []),
            )
            if beam_line is not None:
                _edit_field(lines, beam_line, column, _take_over_beam)
        if "(" in palette:
            _drop_slur_close(lines, palette_line + 1, column, width)

        kept_interpretations = [
            line
            for line in lines[first_data : palette_line + 1]
            if not (_is_data(line) or line.startswith("="))
        ]
        ctx[self.name] = {"removed_templates": 1}
        return "\n".join(lines[:first_data] + kept_interpretations + lines[palette_line + 1 :])
