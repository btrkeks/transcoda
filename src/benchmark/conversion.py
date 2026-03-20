"""Conversion helpers for benchmark raw outputs and MusicXML artifacts."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.core.kern_postprocess import append_terminator_if_missing


@dataclass
class ConversionResult:
    """Result of converting a raw transcription to MusicXML."""

    musicxml: str | None
    error: str | None = None
    stderr: str | None = None
    stdout: str | None = None

    def diagnostics(self) -> str | None:
        parts = []
        if self.stderr:
            parts.append(self.stderr)
        if self.stdout:
            parts.append(self.stdout)
        if not parts:
            return None
        return "\n".join(parts)


def build_sample_key(index: int, source: str | None) -> str:
    """Build a stable sample key for outputs and reports."""
    if source:
        return source
    return f"{index:06d}"


def safe_sample_filename(sample_key: str) -> str:
    """Convert a sample key into a filesystem-safe file name."""
    return sample_key.replace("/", "__")


def ensure_humdrum_document(text: str) -> str:
    """Ensure a raw `**kern` transcription is a complete Humdrum document."""
    stripped = text.strip()
    if not stripped:
        return text

    lines = stripped.split("\n")
    first_line = lines[0].strip()
    if not first_line.startswith("**"):
        spine_count = lines[0].count("\t") + 1
        header = "\t".join(["**kern"] * spine_count)
        stripped = f"{header}\n{stripped}"

    return append_terminator_if_missing(stripped)


def resolve_abc2xml_command(requested_path: str) -> list[str]:
    """Resolve the ABC converter command, preferring PATH then local fallback."""
    resolved = shutil.which(requested_path)
    if resolved is not None:
        return [resolved]

    requested = Path(requested_path)
    if requested.exists():
        if requested.suffix == ".py":
            return ["python", str(requested)]
        return [str(requested)]

    local_fallback = Path("tools/abc2xml/abc2xml.py")
    if local_fallback.exists():
        return ["python", str(local_fallback)]

    raise FileNotFoundError(
        f"Could not resolve abc2xml command '{requested_path}' and local fallback is missing."
    )


def convert_kern_to_musicxml(text: str, hum2xml_path: str) -> ConversionResult:
    """Convert `**kern` text to MusicXML via hum2xml."""
    normalized = ensure_humdrum_document(text)
    if not normalized.strip():
        return ConversionResult(musicxml=None, error="Empty kern input")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".krn", encoding="utf-8") as handle:
        handle.write(normalized)
        handle.flush()
        result = subprocess.run(
            [hum2xml_path, handle.name],
            capture_output=True,
            text=True,
            check=False,
        )

    musicxml = result.stdout.strip()
    if result.returncode != 0:
        return ConversionResult(
            musicxml=None,
            error=f"hum2xml failed with exit code {result.returncode}",
            stderr=result.stderr.strip() or None,
            stdout=result.stdout.strip() or None,
        )
    if not musicxml:
        return ConversionResult(musicxml=None, error="hum2xml produced no output")

    try:
        ET.fromstring(musicxml)
    except ET.ParseError as exc:
        return ConversionResult(musicxml=None, error=f"Invalid MusicXML from hum2xml: {exc}")

    return ConversionResult(
        musicxml=musicxml,
        stderr=result.stderr.strip() or None,
        stdout=result.stdout.strip() or None,
    )


def convert_abc_to_musicxml(text: str, abc2xml_command: Sequence[str]) -> ConversionResult:
    """Convert ABC text to MusicXML via abc2xml."""
    if not text.strip():
        return ConversionResult(musicxml=None, error="Empty ABC input")

    result = subprocess.run(
        [*abc2xml_command, "-"],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    musicxml = result.stdout.strip()
    if result.returncode != 0:
        return ConversionResult(
            musicxml=None,
            error=f"abc2xml failed with exit code {result.returncode}",
            stderr=result.stderr.strip() or None,
            stdout=result.stdout.strip() or None,
        )
    if not musicxml:
        return ConversionResult(musicxml=None, error="abc2xml produced no output")

    try:
        ET.fromstring(musicxml)
    except ET.ParseError as exc:
        return ConversionResult(musicxml=None, error=f"Invalid MusicXML from abc2xml: {exc}")

    return ConversionResult(
        musicxml=musicxml,
        stderr=result.stderr.strip() or None,
        stdout=result.stdout.strip() or None,
    )
