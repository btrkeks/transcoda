"""Native baseline notation normalization with explicit converter boundaries."""

from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SMT_HEADER = "**kern\t**kern\n"
DEFAULT_CONVERTER_TIMEOUT_S = 60.0

_SMT_FORBIDDEN_TOKENS = (
    "*staff2",
    "*staff1",
    "*Xped",
    "*tremolo",
    "*ped",
    "*Xtuplet",
    "*tuplet",
    "*Xtremolo",
    "*cue",
    "*Xcue",
    "*rscale:1/2",
    "*rscale:1",
    "*kcancel",
    "*below",
)
_VOLATILE_HUM2ABC_PREFIXES = (
    "%%abcx-conversion-date ",
    "%%humdrum-veritas ",
    "%%humdrum-veritas-data ",
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class ChordDurationRepair:
    """One derived-evaluation repair of unequal chord-subtoken durations."""

    line_number: int
    field_number: int
    first_duration: str
    old_field: str
    new_field: str
    replacement_count: int


class CommandRunner(Protocol):
    def run(
        self,
        command: list[str],
        *,
        input_bytes: bytes,
        timeout_s: float,
    ) -> CommandResult: ...


class SubprocessCommandRunner:
    def run(
        self,
        command: list[str],
        *,
        input_bytes: bytes,
        timeout_s: float,
    ) -> CommandResult:
        try:
            result = subprocess.run(
                command,
                input=input_bytes,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"converter timed out after {timeout_s:g}s: {command[0]}") from exc
        return CommandResult(result.returncode, result.stdout, result.stderr)


_KERN_DURATION_PREFIX = re.compile(
    r"^(?P<duration>(?:[1-9][0-9]*%[1-9][0-9]*|0+|[1-9][0-9]*)(?:\.{1,4})?)"
)


def repair_unequal_kern_chord_durations(
    native_text: str,
) -> tuple[str, tuple[ChordDurationRepair, ...]]:
    """Make every duration-bearing chord subtoken use its first duration.

    This is an explicit derived-evaluation repair. It preserves the source text's
    line endings and leaves non-data records, single notes, equal-duration chords,
    and compact durationless following subtokens unchanged.
    """

    repaired_lines: list[str] = []
    repairs: list[ChordDurationRepair] = []
    for line_number, line_with_ending in enumerate(
        native_text.splitlines(keepends=True),
        start=1,
    ):
        line = line_with_ending.rstrip("\r\n")
        ending = line_with_ending[len(line) :]
        if not line or line.startswith(("*", "=", "!")):
            repaired_lines.append(line_with_ending)
            continue

        fields = line.split("\t")
        repaired_fields: list[str] = []
        for field_number, field in enumerate(fields, start=1):
            subtokens = field.split(" ")
            first_match = _KERN_DURATION_PREFIX.match(subtokens[0])
            if len(subtokens) < 2 or first_match is None:
                repaired_fields.append(field)
                continue

            first_duration = first_match.group("duration")
            replacement_count = 0
            repaired_subtokens = [subtokens[0]]
            for subtoken in subtokens[1:]:
                match = _KERN_DURATION_PREFIX.match(subtoken)
                if match is None or match.group("duration") == first_duration:
                    repaired_subtokens.append(subtoken)
                    continue
                repaired_subtokens.append(
                    first_duration + subtoken[match.end("duration") :]
                )
                replacement_count += 1

            new_field = " ".join(repaired_subtokens)
            repaired_fields.append(new_field)
            if replacement_count:
                repairs.append(
                    ChordDurationRepair(
                        line_number=line_number,
                        field_number=field_number,
                        first_duration=first_duration,
                        old_field=field,
                        new_field=new_field,
                        replacement_count=replacement_count,
                    )
                )
        repaired_lines.append("\t".join(repaired_fields) + ending)

    return "".join(repaired_lines), tuple(repairs)


def smt_bekern_to_kern(native_text: str) -> str:
    """Invert SMT's headerless two-spine bekern output without consulting GT."""

    if "**kern" in native_text:
        raise ValueError("SMT native output unexpectedly contains a Kern header")
    body = (
        native_text.replace("<s>", " ")
        .replace("<t>", "\t")
        .replace("<b>", "\n")
        .replace("·", "")
        .replace("@", "")
        .strip()
    )
    if not body or not any(line.strip() for line in body.splitlines()):
        raise ValueError("SMT native output is empty after detokenization")
    return SMT_HEADER + body + "\n"


def add_kern_envelope(text: str) -> str:
    """Add missing exclusive/termination records, preserving an existing envelope.

    Counts come from the first and last non-global records, as in the rebuttal
    wrapper. This does not repair spine topology or truncate partial output.
    """
    lines = text.strip().splitlines()
    records = [line for line in lines if line.strip() and not line.startswith("!!")]
    if not records:
        raise ValueError("cannot envelope an empty Kern body")
    if not any(field.startswith("**") for field in records[0].split("\t")):
        lines.insert(0, "\t".join(["**kern"] * len(records[0].split("\t"))))
    if not all(field == "*-" for field in records[-1].split("\t")):
        lines.append("\t".join(["*-"] * len(records[-1].split("\t"))))
    return "\n".join(lines) + "\n"


def transcoda_kern_envelope(text: str) -> str:
    """Wrap the unchanged body, terminating spines active after its last record."""
    records = [line for line in text.splitlines() if line.strip() and not line.startswith("!")]
    if not records:
        raise ValueError("cannot envelope an empty Kern body")
    prefix = ""
    if not any(field.startswith("**") for field in records[0].split("\t")):
        prefix = "\t".join(["**kern"] * len(records[0].split("\t"))) + "\n"
    suffix = "" if text.endswith("\n") else "\n"
    last_fields = records[-1].split("\t")
    width = len(last_fields)
    if all(field.startswith("*") for field in last_fields):
        width = sum(
            2 if field in {"*^", "*+"} else 0 if field == "*-" else 1
            for field in last_fields
        ) - sum(
            left == right == "*v"
            for left, right in zip(last_fields, last_fields[1:], strict=False)
        )
    if width:
        suffix += "\t".join(["*-"] * width) + "\n"
    return prefix + text + suffix


def smt_kern_to_bekern(kern_text: str) -> str:
    """Mirror the frozen clone's `clean_kern`/`parse_kern(...)[4:]` route."""

    cleaned_lines = [
        line
        for line in kern_text.strip().splitlines()
        if not any(token in line for token in _SMT_FORBIDDEN_TOKENS)
        and not re.fullmatch(r"\*(\s+\*)*", line)
    ]
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"(?<==)\d+", "", cleaned)
    serialized = cleaned.replace(" ", " <s> ")
    serialized = serialized.replace("\t", " <t> ").replace("\n", " <b> ")
    serialized = serialized.replace(" /", "").replace(" \\", "")
    serialized = serialized.replace("·/", "").replace("·\\", "")
    serialized = serialized.replace("·", " ").replace("@", " ")
    tokens = serialized.strip().split(" ")
    expected_header = ["**kern", "<t>", "**kern", "<b>"]
    if tokens[:4] != expected_header:
        raise ValueError("SMT noise-floor input must start with the fixed two-spine Kern header")
    native = "".join(tokens[4:])
    if not native:
        raise ValueError("SMT noise-floor input has no body")
    return native


def convert_abc_to_musicxml(
    abc_text: str,
    *,
    python_bin: Path,
    abc2xml_script: Path,
    runner: CommandRunner | None = None,
    timeout_s: float = DEFAULT_CONVERTER_TIMEOUT_S,
) -> str:
    script = abc2xml_script.resolve(strict=True)
    output = _run_text_converter(
        [str(python_bin), str(script), "-"],
        abc_text,
        runner=runner,
        timeout_s=timeout_s,
    )
    _validate_musicxml(output)
    return output


def convert_musicxml_to_kern(
    musicxml_text: str,
    *,
    musicxml2hum_bin: Path,
    runner: CommandRunner | None = None,
    timeout_s: float = DEFAULT_CONVERTER_TIMEOUT_S,
) -> str:
    output = _run_text_converter(
        [str(musicxml2hum_bin)],
        musicxml_text,
        runner=runner,
        timeout_s=timeout_s,
    )
    _validate_kern(output)
    return _normalized_text(output)


def convert_kern_to_musicxml(
    kern_text: str,
    *,
    hum2xml_bin: Path,
    runner: CommandRunner | None = None,
    timeout_s: float = DEFAULT_CONVERTER_TIMEOUT_S,
) -> str:
    output = _run_text_converter(
        [str(hum2xml_bin)],
        kern_text,
        runner=runner,
        timeout_s=timeout_s,
    )
    _validate_musicxml(output)
    return output


def convert_kern_to_abc(
    kern_text: str,
    *,
    hum2abc_bin: Path,
    runner: CommandRunner | None = None,
    timeout_s: float = DEFAULT_CONVERTER_TIMEOUT_S,
) -> str:
    output = _run_text_converter(
        [str(hum2abc_bin)],
        kern_text,
        runner=runner,
        timeout_s=timeout_s,
    )
    stable_lines = [
        line
        for line in output.splitlines()
        if not line.startswith(_VOLATILE_HUM2ABC_PREFIXES)
    ]
    stable = _normalized_text("\n".join(stable_lines))
    if not any(line.startswith("X:") for line in stable.splitlines()) or not any(
        line.startswith("K:") for line in stable.splitlines()
    ):
        raise ValueError("hum2abc output is not valid ABC metadata")
    return stable


def _run_text_converter(
    command: list[str],
    input_text: str,
    *,
    runner: CommandRunner | None,
    timeout_s: float,
) -> str:
    active_runner = runner or SubprocessCommandRunner()
    result = active_runner.run(
        command,
        input_bytes=input_text.encode("utf-8"),
        timeout_s=timeout_s,
    )
    if result.returncode != 0:
        diagnostic = result.stderr.decode("utf-8", errors="replace").strip()[:1000]
        raise RuntimeError(
            f"converter failed with exit code {result.returncode}: {command[0]}: {diagnostic}"
        )
    try:
        output = result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"converter output is not UTF-8: {command[0]}") from exc
    if not output.strip():
        raise ValueError(f"converter produced empty output: {command[0]}")
    return output


def _validate_musicxml(text: str) -> None:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"converter produced invalid MusicXML: {exc}") from exc
    tag = root.tag.rsplit("}", maxsplit=1)[-1]
    if tag != "score-partwise":
        raise ValueError(f"converter MusicXML root must be score-partwise, got {tag}")


def _validate_kern(text: str) -> None:
    lines = [line for line in text.splitlines() if line and not line.startswith("!!!")]
    if not lines or not lines[0].startswith("**kern"):
        raise ValueError("converter output is not Humdrum **kern")
    if not any(line.startswith("*-") for line in lines):
        raise ValueError("converter Kern output has no spine terminator")


def _normalized_text(text: str) -> str:
    return text.strip() + "\n"
