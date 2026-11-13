#!/usr/bin/env python3
"""Report hum2xml conversion pass/fail counts for generated Humdrum files."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_GLOB = "data/interim/train/*/6_concatenated/**/*.krn"
DEFAULT_SUMMARY_JSON = "reports/hum2xml_conversion_summary.json"
DEFAULT_FAILURES_JSONL = "reports/hum2xml_conversion_failures.jsonl"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_DIAGNOSTIC_CHARS = 12_000

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transcoda.core.kern_postprocess import ensure_renderable_humdrum_document  # noqa: E402


@dataclass
class ConversionResult:
    """Result of converting a Humdrum file to MusicXML."""

    musicxml: str | None
    error: str | None = None
    stderr: str | None = None
    stdout: str | None = None


def _clean_process_output(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value.strip() or None


def convert_kern_to_musicxml(
    text: str,
    hum2xml_path: str,
    timeout_seconds: float | None = None,
) -> ConversionResult:
    """Convert `**kern` text to MusicXML via hum2xml."""
    normalized = ensure_renderable_humdrum_document(text)
    if not normalized.strip():
        return ConversionResult(musicxml=None, error="Empty kern input")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".krn", encoding="utf-8") as handle:
        handle.write(normalized)
        handle.flush()
        try:
            result = subprocess.run(
                [hum2xml_path, handle.name],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_label = f"{timeout_seconds:g}s" if timeout_seconds is not None else "unknown"
            return ConversionResult(
                musicxml=None,
                error=f"hum2xml timed out after {timeout_label}",
                stderr=_clean_process_output(exc.stderr),
                stdout=_clean_process_output(exc.stdout),
            )

    musicxml = result.stdout.strip()
    if result.returncode != 0:
        return ConversionResult(
            musicxml=None,
            error=f"hum2xml failed with exit code {result.returncode}",
            stderr=_clean_process_output(result.stderr),
            stdout=_clean_process_output(result.stdout),
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


@dataclass(frozen=True)
class ConversionFailure:
    path: str
    dataset: str
    error: str
    stderr: str | None = None
    stdout: str | None = None


def infer_dataset(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """Infer dataset name from data/interim/{split}/{dataset}/... paths."""
    try:
        parts = path.resolve().relative_to(project_root.resolve()).parts
    except ValueError:
        parts = path.parts

    prefix = ("data", "interim")
    for idx in range(len(parts) - len(prefix) - 1):
        if parts[idx : idx + len(prefix)] == prefix:
            dataset_idx = idx + len(prefix) + 1
            if dataset_idx < len(parts):
                return parts[dataset_idx]
    return "unknown"


def expand_input_paths(inputs: Iterable[str], project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Expand files, directories, and glob patterns into sorted unique .krn files."""
    patterns = list(inputs) or [DEFAULT_INPUT_GLOB]
    found: dict[Path, None] = {}

    for raw_pattern in patterns:
        pattern_path = Path(raw_pattern)
        candidate = pattern_path if pattern_path.is_absolute() else project_root / pattern_path

        if candidate.is_file():
            if candidate.suffix == ".krn":
                found[candidate.resolve()] = None
            continue

        if candidate.is_dir():
            for path in candidate.rglob("*.krn"):
                found[path.resolve()] = None
            continue

        glob_pattern = (
            raw_pattern if pattern_path.is_absolute() else str(project_root / raw_pattern)
        )
        for match in glob.glob(glob_pattern, recursive=True):
            path = Path(match)
            if path.is_file() and path.suffix == ".krn":
                found[path.resolve()] = None

    return sorted(found)


def truncate_diagnostic(value: str | None, max_chars: int) -> str | None:
    if value is None or len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}\n... truncated {omitted} characters ..."


def check_file(
    path: Path,
    hum2xml_path: str,
    project_root: Path = PROJECT_ROOT,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    max_diagnostic_chars: int = DEFAULT_MAX_DIAGNOSTIC_CHARS,
) -> ConversionFailure | None:
    """Convert one file with hum2xml and return failure details if conversion fails."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ConversionFailure(
            path=str(path),
            dataset=infer_dataset(path, project_root),
            error=f"Failed to read file: {exc}",
        )

    result = convert_kern_to_musicxml(text, hum2xml_path, timeout_seconds=timeout_seconds)
    if result.error is None:
        return None

    return ConversionFailure(
        path=str(path),
        dataset=infer_dataset(path, project_root),
        error=result.error,
        stderr=truncate_diagnostic(result.stderr, max_diagnostic_chars),
        stdout=truncate_diagnostic(result.stdout, max_diagnostic_chars),
    )


def _check_file_worker(args: tuple[str, str, str, float | None, int]) -> ConversionFailure | None:
    path, hum2xml_path, project_root, timeout_seconds, max_diagnostic_chars = args
    return check_file(
        Path(path),
        hum2xml_path,
        Path(project_root),
        timeout_seconds=timeout_seconds,
        max_diagnostic_chars=max_diagnostic_chars,
    )


def collect_failures(
    files: list[Path],
    hum2xml_path: str,
    jobs: int,
    project_root: Path = PROJECT_ROOT,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    max_diagnostic_chars: int = DEFAULT_MAX_DIAGNOSTIC_CHARS,
) -> list[ConversionFailure]:
    if jobs <= 1:
        failures: list[ConversionFailure] = []
        for index, path in enumerate(files, start=1):
            failure = check_file(
                path,
                hum2xml_path,
                project_root,
                timeout_seconds=timeout_seconds,
                max_diagnostic_chars=max_diagnostic_chars,
            )
            if failure is not None:
                failures.append(failure)
            if index % 1000 == 0:
                print(f"Checked {index}/{len(files)} files...", file=sys.stderr)
        return failures

    worker_args = [
        (str(path), hum2xml_path, str(project_root), timeout_seconds, max_diagnostic_chars)
        for path in files
    ]
    failures = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for index, failure in enumerate(executor.map(_check_file_worker, worker_args), start=1):
            if failure is not None:
                failures.append(failure)
            if index % 1000 == 0:
                print(f"Checked {index}/{len(files)} files...", file=sys.stderr)
    return failures


def build_summary(
    files: list[Path],
    failures: list[ConversionFailure],
    *,
    hum2xml_path: str,
    timeout_seconds: float | None,
    elapsed_seconds: float,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    failure_paths = {failure.path for failure in failures}
    by_dataset: dict[str, dict[str, int]] = {}
    for path in files:
        dataset = infer_dataset(path, project_root)
        stats = by_dataset.setdefault(dataset, {"checked": 0, "passed": 0, "failed": 0})
        stats["checked"] += 1
        if str(path) in failure_paths:
            stats["failed"] += 1
        else:
            stats["passed"] += 1

    failure_count = len(failures)
    success_count = len(files) - failure_count
    return {
        "schema_version": "1.0",
        "input_count": len(files),
        "success_count": success_count,
        "failure_count": failure_count,
        "pass_rate": 0.0 if not files else success_count / len(files),
        "duration_seconds": elapsed_seconds,
        "hum2xml_path": hum2xml_path,
        "timeout_seconds": timeout_seconds,
        "by_dataset": by_dataset,
        "failure_types": dict(Counter(failure.error for failure in failures)),
    }


def write_summary(summary: dict[str, Any], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def write_failures_jsonl(failures: list[ConversionFailure], failures_path: Path) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with failures_path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(asdict(failure), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def print_summary(summary: dict[str, Any], failures: list[ConversionFailure]) -> None:
    print("\nhum2xml conversion report")
    print("=" * 40)
    print(f"Checked: {summary['input_count']}")
    print(f"Passed:  {summary['success_count']}")
    print(f"Failed:  {summary['failure_count']}")
    print(f"Pass %:  {summary['pass_rate'] * 100:.2f}")
    print(f"Elapsed: {summary['duration_seconds']:.1f}s")

    if not failures:
        return

    print("\nFailures by dataset:")
    for dataset, stats in sorted(
        summary["by_dataset"].items(), key=lambda item: item[1]["failed"], reverse=True
    ):
        if stats["failed"]:
            print(f"  {dataset}: {stats['failed']}")

    print("\nFailures by error:")
    for error, count in Counter(failure.error for failure in failures).most_common():
        print(f"  {count:>6}  {error}")

    print("\nFirst failures:")
    for failure in failures[:10]:
        print(f"  {failure.path}: {failure.error}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report hum2xml conversion success/failure counts for .krn files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(f"Files, directories, or glob patterns to report (default: {DEFAULT_INPUT_GLOB})"),
    )
    parser.add_argument("--hum2xml-path", default="hum2xml", help="hum2xml executable path")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to check")
    parser.add_argument("--jobs", type=int, default=4, help="Number of worker processes")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Maximum seconds to allow each hum2xml subprocess "
            f"(default: {DEFAULT_TIMEOUT_SECONDS:g}; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--max-diagnostic-chars",
        type=int,
        default=DEFAULT_MAX_DIAGNOSTIC_CHARS,
        help=(
            "Maximum stderr/stdout characters stored per failed file "
            f"(default: {DEFAULT_MAX_DIAGNOSTIC_CHARS})"
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(DEFAULT_SUMMARY_JSON),
        help=f"Summary JSON path (default: {DEFAULT_SUMMARY_JSON})",
    )
    parser.add_argument(
        "--failures-jsonl",
        type=Path,
        default=Path(DEFAULT_FAILURES_JSONL),
        help=f"Failure JSONL path (default: {DEFAULT_FAILURES_JSONL})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.jobs < 1:
        print("Error: --jobs must be at least 1", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 0:
        print("Error: --limit must be non-negative", file=sys.stderr)
        return 2
    if args.timeout < 0:
        print("Error: --timeout must be non-negative", file=sys.stderr)
        return 2
    if args.max_diagnostic_chars < 0:
        print("Error: --max-diagnostic-chars must be non-negative", file=sys.stderr)
        return 2

    if shutil.which(args.hum2xml_path) is None and not Path(args.hum2xml_path).exists():
        print(f"Error: hum2xml executable not found: {args.hum2xml_path}", file=sys.stderr)
        return 2

    files = expand_input_paths(args.paths)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print("Error: no .krn files matched the requested inputs", file=sys.stderr)
        return 2

    print(f"Checking {len(files)} .krn files with {args.hum2xml_path}")
    timeout_seconds = args.timeout if args.timeout > 0 else None
    if timeout_seconds is not None:
        print(f"Per-file timeout: {timeout_seconds:g}s")
    if args.jobs > 1:
        print(f"Using {args.jobs} worker processes")

    start = time.monotonic()
    failures = collect_failures(
        files,
        args.hum2xml_path,
        args.jobs,
        timeout_seconds=timeout_seconds,
        max_diagnostic_chars=args.max_diagnostic_chars,
    )
    elapsed = time.monotonic() - start

    summary = build_summary(
        files,
        failures,
        hum2xml_path=args.hum2xml_path,
        timeout_seconds=timeout_seconds,
        elapsed_seconds=elapsed,
    )
    write_summary(summary, args.summary_json)
    write_failures_jsonl(failures, args.failures_jsonl)

    print_summary(summary, failures)
    print(f"\nSummary JSON: {args.summary_json}")
    print(f"Failures JSONL: {args.failures_jsonl}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
