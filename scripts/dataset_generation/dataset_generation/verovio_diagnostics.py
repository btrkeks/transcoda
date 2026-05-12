"""Capture and parse native Verovio diagnostics."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

from scripts.dataset_generation.dataset_generation.types_events import VerovioDiagnostic

_ERROR_PREFIX = "Error:"
_INCONSISTENT_RHYTHM_RE = re.compile(
    r"^Error: Inconsistent rhythm analysis occurring near line (?P<near_line>\d+)$"
)
_DURATION_RE = re.compile(
    r"^Expected durationFromStart to be: (?P<expected>\S+) but found it to be (?P<found>\S+)$"
)
_LINE_RE = re.compile(r"^Line: (?P<line_text>.*)$")


@contextmanager
def capture_native_stderr(enabled: bool) -> Iterator[list[str]]:
    captured: list[str] = []
    if not enabled:
        yield captured
        return

    stderr_fd = 2
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with tempfile.TemporaryFile(mode="w+b") as capture_file:
            sys.stderr.flush()
            os.dup2(capture_file.fileno(), stderr_fd)
            try:
                yield captured
            finally:
                sys.stderr.flush()
                os.dup2(saved_stderr_fd, stderr_fd)
                capture_file.seek(0)
                raw = capture_file.read()
                if raw:
                    captured.append(raw.decode("utf-8", errors="replace"))
    finally:
        os.close(saved_stderr_fd)


def parse_verovio_diagnostics(
    stderr_text: str,
    *,
    render_attempt_idx: int | None = None,
) -> tuple[VerovioDiagnostic, ...]:
    blocks = _iter_error_blocks(stderr_text)
    return tuple(_parse_error_block(block, render_attempt_idx=render_attempt_idx) for block in blocks)


def _iter_error_blocks(stderr_text: str) -> list[list[str]]:
    lines = [line.rstrip() for line in stderr_text.replace("\r\n", "\n").split("\n")]
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        if line.startswith(_ERROR_PREFIX):
            if current:
                blocks.append(current)
            current = [line]
            continue
        if current:
            current.append(line)

    if current:
        blocks.append(current)
    return blocks


def _parse_error_block(
    block: list[str],
    *,
    render_attempt_idx: int | None,
) -> VerovioDiagnostic:
    raw_message = "\n".join(block)
    first_line = block[0] if block else ""
    rhythm_match = _INCONSISTENT_RHYTHM_RE.match(first_line)
    if rhythm_match:
        expected, found = _parse_duration_line(block)
        return VerovioDiagnostic(
            diagnostic_kind="inconsistent_rhythm_analysis",
            raw_message=raw_message,
            render_attempt_idx=render_attempt_idx,
            near_line=int(rhythm_match.group("near_line")),
            expected_duration_from_start=expected,
            found_duration_from_start=found,
            line_text=_parse_source_line(block),
        )
    return VerovioDiagnostic(
        diagnostic_kind="verovio_error",
        raw_message=raw_message,
        render_attempt_idx=render_attempt_idx,
    )


def _parse_duration_line(block: list[str]) -> tuple[str | None, str | None]:
    for line in block[1:]:
        match = _DURATION_RE.match(line)
        if match:
            return match.group("expected"), match.group("found")
    return None, None


def _parse_source_line(block: list[str]) -> str | None:
    for line in block[1:]:
        match = _LINE_RE.match(line)
        if match:
            return match.group("line_text")
    return None
