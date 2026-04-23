"""NormalizeRScale pass - rewrites active *rscale regions into visible durations."""

from __future__ import annotations

import re
from fractions import Fraction

from src.core.spine_state import (
    InvalidSpineOperationError,
    UnsupportedSpineManipulatorError,
    is_interpretation_record,
    validate_spine_operation_record,
)

from ..base import NormalizationContext

_RSCALE_RE = re.compile(r"^\*rscale:(\d+)(?:/(\d+))?$")
_SUPPORTED_MANIPULATORS = {"*^", "*v", "*-"}
_UNSUPPORTED_MANIPULATORS = {"*x", "*+"}
_LEADING_DURATION_WRAPPERS = "[](){}_&"
_GRAMMAR_RECIPROCAL_DURS = {
    "000",
    "00",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "12",
    "13",
    "14",
    "15",
    "16",
    "18",
    "20",
    "22",
    "24",
    "25",
    "26",
    "28",
    "30",
    "32",
    "34",
    "36",
    "38",
    "40",
    "44",
    "48",
    "52",
    "56",
    "60",
    "64",
    "70",
    "72",
    "80",
    "88",
    "96",
    "108",
    "112",
    "116",
    "128",
    "144",
    "160",
    "176",
    "192",
    "256",
    "272",
    "512",
    "1024",
}


def _parse_rscale(field: str) -> Fraction | None:
    match = _RSCALE_RE.match(field)
    if not match:
        return None

    numerator = int(match.group(1))
    denominator = int(match.group(2) or "1")
    if denominator == 0:
        raise ValueError(f"invalid *rscale denominator in {field!r}")
    return Fraction(numerator, denominator)


def _parse_duration_span(token: str) -> tuple[int, int] | None:
    index = 0
    while index < len(token) and token[index] in _LEADING_DURATION_WRAPPERS:
        index += 1

    start = index
    while index < len(token) and token[index].isdigit():
        index += 1

    if index == start:
        return None

    if index < len(token) and token[index] == "%":
        index += 1
        denom_start = index
        while index < len(token) and token[index].isdigit():
            index += 1
        if index == denom_start:
            raise ValueError(f"malformed rational duration in token {token!r}")

    while index < len(token) and token[index] == ".":
        index += 1

    return start, index


def _parse_duration_text(duration_text: str) -> tuple[Fraction, int]:
    dots = 0
    while duration_text.endswith("."):
        duration_text = duration_text[:-1]
        dots += 1

    if "%" in duration_text:
        numerator_text, denominator_text = duration_text.split("%", maxsplit=1)
        denominator = int(denominator_text)
        if denominator == 0:
            raise ValueError(f"invalid rational duration denominator in {duration_text!r}")
        base = Fraction(int(numerator_text), denominator)
        return base, dots

    if set(duration_text) == {"0"}:
        base = Fraction(2 ** len(duration_text), 1)
        return base, dots

    base = Fraction(1, int(duration_text))
    return base, dots


def _format_scaled_base(base: Fraction) -> str:
    if base.denominator == 1 and base.numerator in {2, 4, 8}:
        return {2: "0", 4: "00", 8: "000"}[base.numerator]
    if base.numerator == 1 and str(base.denominator) in _GRAMMAR_RECIPROCAL_DURS:
        return str(base.denominator)
    raise ValueError(
        f"scaled duration {base.numerator}/{base.denominator} is not representable "
        "in the supported reciprocal duration set"
    )


def _scale_duration_text(duration_text: str, scale: Fraction) -> str:
    base, dots = _parse_duration_text(duration_text)
    scaled_base = base * scale
    return _format_scaled_base(scaled_base) + ("." * dots)


def _rewrite_token(token: str, scale: Fraction, *, line_no: int) -> str:
    if scale == 1:
        return token
    if token == "." or token.startswith(("=", "*", "!")):
        return token
    if "q" in token or "Q" in token:
        return token

    span = _parse_duration_span(token)
    if span is None:
        return token

    start, end = span
    try:
        scaled_duration = _scale_duration_text(token[start:end], scale)
    except ValueError as exc:
        raise ValueError(f"line {line_no}: {exc}") from exc
    return token[:start] + scaled_duration + token[end:]


def _rewrite_field(field: str, scale: Fraction, *, line_no: int) -> str:
    if scale == 1 or field in {".", ""} or field.startswith(("=", "*", "!")):
        return field
    return " ".join(_rewrite_token(token, scale, line_no=line_no) for token in field.split(" "))


def _advance_rscale_states(
    states: list[Fraction], fields: list[str], *, line_no: int
) -> list[Fraction]:
    if len(fields) != len(states):
        raise ValueError(
            f"Line {line_no}: interpretation record width mismatch: "
            f"expected {len(states)} fields, got {len(fields)}"
        )

    unsupported = sorted({field for field in fields if field in _UNSUPPORTED_MANIPULATORS})
    if unsupported:
        ops = ", ".join(unsupported)
        raise UnsupportedSpineManipulatorError(
            f"Line {line_no}: unsupported spine manipulators: {ops}"
        )

    try:
        validate_spine_operation_record(fields)
    except InvalidSpineOperationError as exc:
        raise InvalidSpineOperationError(f"Line {line_no}: {exc}") from exc

    next_states: list[Fraction] = []
    i = 0
    while i < len(fields):
        token = fields[i]
        if token == "*v":
            j = i
            merged = {states[j]}
            while j + 1 < len(fields) and fields[j + 1] == "*v":
                j += 1
                merged.add(states[j])
            if len(merged) != 1:
                values = ", ".join(str(value) for value in sorted(merged))
                raise ValueError(
                    f"Line {line_no}: cannot merge spines with different active "
                    f"*rscale values: {values}"
                )
            next_states.append(states[i])
            i = j + 1
            continue
        if token == "*^":
            next_states.extend([states[i], states[i]])
            i += 1
            continue
        if token == "*-":
            i += 1
            continue
        next_states.append(states[i])
        i += 1
    return next_states


class NormalizeRScale:
    """
    Rewrite active *rscale regions into visible note/rest durations.

    Only the observed external-data cases are supported: *rscale:2, *rscale:1/2,
    and *rscale:1 reset. When scaled durations do not map back to a plain
    reciprocal duration, the pass emits Humdrum rational durations (e.g. 1%38).
    """

    name = "normalize_rscale"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Rewrite active *rscale regions and remove the interpretation tokens."""
        if not text:
            return text

        lines = text.split("\n")
        result_lines: list[str] = []
        states: list[Fraction] | None = None

        for line_no, line in enumerate(lines, start=1):
            if not line:
                result_lines.append(line)
                continue

            fields = line.split("\t")
            if states is None:
                states = [Fraction(1, 1) for _ in fields]
            elif len(fields) != len(states):
                raise ValueError(
                    f"line {line_no}: record width mismatch: expected {len(states)} fields, "
                    f"got {len(fields)}"
                )

            if is_interpretation_record(fields):
                output_fields = list(fields)
                for i, field in enumerate(fields):
                    scale = _parse_rscale(field)
                    if scale is None:
                        continue
                    if scale not in {Fraction(2, 1), Fraction(1, 2), Fraction(1, 1)}:
                        raise ValueError(
                            f"line {line_no}: unsupported *rscale value {field!r}; "
                            "only *rscale:2, *rscale:1/2, and *rscale:1 are supported"
                        )
                    states[i] = scale
                    output_fields[i] = "*"

                if any(field in _SUPPORTED_MANIPULATORS or field in _UNSUPPORTED_MANIPULATORS for field in fields):
                    states = _advance_rscale_states(states, fields, line_no=line_no)

                if not all(field == "*" for field in output_fields):
                    result_lines.append("\t".join(output_fields))
                continue

            rewritten_fields = [
                _rewrite_field(field, states[i], line_no=line_no)
                for i, field in enumerate(fields)
            ]
            result_lines.append("\t".join(rewritten_fields))

        ctx["normalize_rscale"] = {
            "rewrote_rscale": True,
        }
        return "\n".join(result_lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass
