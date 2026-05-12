"""Courtesy-natural augmentation for semantically redundant naturals."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.kern_utils import extract_pitch, is_bar_line, is_note_token, is_spinemerge_line, is_spinesplit_line

from .kern_utils import append_to_token

__all__ = [
    "apply_courtesy_naturals",
    "parse_key_signature_token",
]

_KEY_SIGNATURE_RE = re.compile(r"^\*k\[([^\]]*)\]$")


@dataclass
class _SpineState:
    """Per-spine accidental tracking state."""

    key_signature: dict[str, str] = field(default_factory=dict)
    measure_accidentals: dict[str, str] = field(default_factory=dict)
    unsafe_until_barline: bool = False


def parse_key_signature_token(token: str) -> dict[str, str] | None:
    """Parse a ``*k[...]`` token into diatonic-class accidentals."""
    match = _KEY_SIGNATURE_RE.match(token)
    if not match:
        return None

    body = match.group(1)
    if not body:
        return {}

    result: dict[str, str] = {}
    i = 0
    while i < len(body):
        letter = body[i]
        if letter.lower() not in "abcdefg":
            return None
        i += 1
        accidental_start = i
        while i < len(body) and body[i] in "#-":
            i += 1
        accidental = body[accidental_start:i]
        if accidental not in {"#", "##", "-", "--"}:
            return None
        result[letter.lower()] = accidental
    return result


def apply_courtesy_naturals(
    krn: str,
    per_note_probability: float = 0.35,
) -> str:
    """Add semantically redundant natural signs to safe note tokens.

    This augmenter is intentionally conservative. It only inserts ``n`` when:
    - the note has no explicit accidental already,
    - its effective accidental is already natural under key + measure state,
    - the token is a single-note token that can be parsed confidently.

    Anything ambiguous is left unchanged.
    """
    if not krn:
        return krn

    lines = krn.splitlines()
    if any(is_spinesplit_line(line) or is_spinemerge_line(line) for line in lines):
        return krn

    states: list[_SpineState] = []
    result_lines: list[str] = []

    for line in lines:
        if not line:
            result_lines.append(line)
            continue

        columns = line.split("\t")
        _ensure_spine_states(states, len(columns))

        if is_bar_line(line):
            for state in states[: len(columns)]:
                state.measure_accidentals.clear()
                state.unsafe_until_barline = False
            result_lines.append(line)
            continue

        if line.startswith("*"):
            for col_idx, token in enumerate(columns):
                key_signature = parse_key_signature_token(token)
                if key_signature is not None:
                    states[col_idx].key_signature = key_signature
                    states[col_idx].measure_accidentals.clear()
                    states[col_idx].unsafe_until_barline = False
            result_lines.append(line)
            continue

        if line.startswith("!"):
            result_lines.append(line)
            continue

        new_columns: list[str] = []
        for col_idx, token in enumerate(columns):
            state = states[col_idx]
            new_columns.append(_process_token(token, state, per_note_probability))

        result_lines.append("\t".join(new_columns))

    return "\n".join(result_lines)


def _ensure_spine_states(states: list[_SpineState], count: int) -> None:
    while len(states) < count:
        states.append(_SpineState())


def _process_token(token: str, state: _SpineState, per_note_probability: float) -> str:
    if state.unsafe_until_barline:
        return token
    if token in {"", ".", "*"} or " " in token:
        if " " in token and any(is_note_token(part) for part in token.split(" ")):
            state.unsafe_until_barline = True
        return token
    if not is_note_token(token) or "r" in token.lower() or "q" in token:
        return token

    parsed = _parse_note_token(token)
    if parsed is None:
        state.unsafe_until_barline = True
        return token

    effective_accidental = state.measure_accidentals.get(
        parsed.pitch_letters,
        state.key_signature.get(parsed.pitch_class, ""),
    )

    new_token = token
    if (
        parsed.explicit_accidental == ""
        and effective_accidental == ""
        and per_note_probability > 0.0
    ):
        import random

        if random.random() < per_note_probability:
            new_token = append_to_token(token, "n")

    if parsed.explicit_accidental:
        state.measure_accidentals[parsed.pitch_letters] = parsed.explicit_accidental

    return new_token


@dataclass(frozen=True)
class _ParsedNote:
    pitch_letters: str
    pitch_class: str
    explicit_accidental: str


def _parse_note_token(token: str) -> _ParsedNote | None:
    pitch = extract_pitch(token)
    if pitch is None:
        return None

    idx = 0
    while idx < len(pitch) and pitch[idx].lower() in "abcdefg":
        idx += 1
    if idx == 0:
        return None

    pitch_letters = pitch[:idx]
    accidental = pitch[idx:]
    if accidental not in {"", "#", "##", "-", "--", "n"}:
        return None

    return _ParsedNote(
        pitch_letters=pitch_letters,
        pitch_class=pitch_letters[0].lower(),
        explicit_accidental=accidental,
    )
