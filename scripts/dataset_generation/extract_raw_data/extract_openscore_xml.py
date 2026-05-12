#!/usr/bin/env python3
"""
Extract OpenScore XML files into pipeline-ready flat directories.

Supports:
  - OpenScore Lieder (`lc*.mxl`) by unpacking embedded MusicXML.
  - OpenScore StringQuartets (`sq*.mscx`) by converting with MuseScore CLI.

Usage:
    python -m scripts.dataset_generation.extract_raw_data.extract_openscore_xml \
        --dataset lieder \
        --output_dir data/interim/train/openscore-lieder/0_raw_xml

    python -m scripts.dataset_generation.extract_raw_data.extract_openscore_xml \
        --dataset stringquartets \
        --output_dir data/interim/train/openscore-stringquartets/0_raw_xml \
        --musescore_cmd mscore
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import fire


DEFAULT_INPUT_DIRS = {
    "lieder": Path("data/raw/Lieder/scores"),
    "stringquartets": Path("data/raw/StringQuartets/scores"),
}


def _first_musicxml_member(mxl_path: Path) -> str | None:
    with zipfile.ZipFile(mxl_path, "r") as archive:
        for member in archive.namelist():
            normalized = member.lower()
            if normalized.endswith(".xml") and not normalized.startswith("meta-inf/"):
                return member
    return None


def _resolve_cmd_available(command: str) -> bool:
    if shutil.which(command):
        return True
    cmd_path = Path(command)
    return cmd_path.exists() and cmd_path.is_file()


def extract_openscore_xml(
    dataset: str,
    output_dir: str,
    input_dir: str | None = None,
    musescore_cmd: str = "mscore",
    quiet: bool = False,
) -> None:
    """
    Extract or convert OpenScore files to flat `.xml` outputs.

    Args:
        dataset: One of `lieder` or `stringquartets`.
        output_dir: Destination directory for flat XML outputs.
        input_dir: Optional override for raw source location.
        musescore_cmd: MuseScore CLI command (used for stringquartets only).
        quiet: Suppress progress output.
    """
    dataset_key = dataset.strip().lower()
    if dataset_key not in DEFAULT_INPUT_DIRS:
        raise ValueError("dataset must be one of: lieder, stringquartets")

    source_root = Path(input_dir) if input_dir else DEFAULT_INPUT_DIRS[dataset_key]
    if not source_root.is_dir():
        raise ValueError(f"input_dir does not exist or is not a directory: {source_root}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dataset_key == "lieder":
        sources = sorted(source_root.rglob("lc*.mxl"))
    else:
        sources = sorted(source_root.rglob("sq*.mscx"))
        if not _resolve_cmd_available(musescore_cmd):
            raise ValueError(
                f"MuseScore CLI command not found: '{musescore_cmd}'. "
                "Set --musescore_cmd or ensure it is in PATH."
            )

    if not sources:
        raise ValueError(f"No source files found for {dataset_key} in {source_root}")

    if not quiet:
        print(f"Dataset: {dataset_key}", file=sys.stderr)
        print(f"Source root: {source_root}", file=sys.stderr)
        print(f"Found {len(sources)} source files", file=sys.stderr)
        print(f"Output: {output_path}", file=sys.stderr)

    seen_stems: set[str] = set()
    failures: list[tuple[Path, str]] = []
    written = 0

    for src in sources:
        stem = src.stem
        if stem in seen_stems:
            failures.append((src, f"duplicate output stem '{stem}'"))
            continue
        seen_stems.add(stem)

        out_file = output_path / f"{stem}.xml"
        try:
            if dataset_key == "lieder":
                member = _first_musicxml_member(src)
                if member is None:
                    raise ValueError("no MusicXML member found in archive")
                with zipfile.ZipFile(src, "r") as archive:
                    out_file.write_bytes(archive.read(member))
            else:
                proc = subprocess.run(
                    [musescore_cmd, "-o", str(out_file), str(src)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    stderr = proc.stderr.strip() or "unknown conversion error"
                    raise RuntimeError(stderr)

            if not out_file.exists() or out_file.stat().st_size == 0:
                raise RuntimeError("empty output XML")
            written += 1
        except Exception as exc:
            if out_file.exists():
                out_file.unlink()
            failures.append((src, str(exc)))

    if not quiet:
        print("", file=sys.stderr)
        print(f"Written XML files: {written}", file=sys.stderr)
        print(f"Failures: {len(failures)}", file=sys.stderr)

    if failures:
        for src, message in failures[:20]:
            print(f"  FAIL {src}: {message}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more failures", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    fire.Fire(extract_openscore_xml)


if __name__ == "__main__":
    main()
