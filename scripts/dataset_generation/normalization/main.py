#!/usr/bin/env python3
"""
Normalize kern files through the standard normalization pipeline.

This script normalizes kern files and saves them to an output directory,
designed for integration into a processing pipeline:
    1) convert -> 2) filter -> 3) normalize -> 4) generate into final dataset

Examples:
    # Basic usage
    python -m scripts.dataset_generation.normalization.main

    # Custom paths
    python -m scripts.dataset_generation.normalization.main \\
        --input-dir data/interim/pdmx/2_filtered \\
        --output-dir data/interim/pdmx/3_normalized_kern

    # With parallel processing
    python -m scripts.dataset_generation.normalization.main --workers 8
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fire

from .presets import normalize_kern_transcription_with_context


def _extract_pass_stats(ctx: dict) -> dict[str, dict[str, int | dict[str, int]]]:
    """Extract additive per-pass normalization stats from context."""
    canonicalize_stats = ctx.get("canonicalize_note_order", {})
    unknown_counts_raw = canonicalize_stats.get("unknown_char_drop_counts", {})
    unknown_counts = {
        str(char): int(count)
        for char, count in unknown_counts_raw.items()
        if int(count) > 0
    }
    unknown_total = int(canonicalize_stats.get("unknown_char_drops_total", sum(unknown_counts.values())))

    return {
        "canonicalize_note_order": {
            "unknown_char_drops_total": unknown_total,
            "unknown_char_drop_counts": unknown_counts,
        }
    }


def normalize_file(
    input_path: Path, output_path: Path
) -> tuple[Path, bool, str | None, str | None, dict[str, dict[str, int | dict[str, int]]] | None]:
    """
    Normalize a single kern file.

    Returns:
        Tuple of (input_path, success, error_type, error_message)
    """
    try:
        content = input_path.read_text(encoding="utf-8")
        normalized, ctx = normalize_kern_transcription_with_context(content)
        output_path.write_text(normalized, encoding="utf-8")
        return (input_path, True, None, None, _extract_pass_stats(ctx))
    except Exception as e:
        return (input_path, False, type(e).__name__, str(e), None)


def normalize_kern_files(
    input_dir: str = "data/interim/pdmx/2_filtered",
    output_dir: str = "data/interim/pdmx/3_normalized_kern",
    workers: int = 1,
    quiet: bool = False,
    continue_on_error: bool = True,
    stats_json: str | None = None,
):
    """
    Normalize all kern files in a directory.

    Args:
        input_dir: Directory containing kern files to normalize
        output_dir: Directory to save normalized files to
        workers: Number of parallel workers (default: 1)
        quiet: Suppress progress output
        continue_on_error: Continue processing if a file fails (default: True)
        stats_json: Optional path to save run statistics as JSON
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    start_time = time.monotonic()

    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Collect input files
    kern_files = sorted(input_path.glob("**/*.krn"))
    if not kern_files:
        print(f"Error: No .krn files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not quiet:
        print(f"Normalizing {len(kern_files)} files...", file=sys.stderr)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0
    errors: list[tuple[Path, str | None, str | None]] = []
    error_types: dict[str, int] = {}
    canonicalize_unknown_total = 0
    canonicalize_unknown_counts: dict[str, int] = {}

    if workers > 1:
        # Parallel processing
        tasks = [(f, output_path / f.name) for f in kern_files]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(normalize_file, inp, out): inp
                for inp, out in tasks
            }
            for future in as_completed(futures):
                inp_path, success, error_type, error, pass_stats = future.result()
                if success:
                    success_count += 1
                    if pass_stats:
                        canonicalize = pass_stats.get("canonicalize_note_order", {})
                        canonicalize_unknown_total += int(canonicalize.get("unknown_char_drops_total", 0))
                        for ch, count in canonicalize.get("unknown_char_drop_counts", {}).items():
                            canonicalize_unknown_counts[ch] = canonicalize_unknown_counts.get(ch, 0) + int(count)
                else:
                    error_count += 1
                    errors.append((inp_path, error_type, error))
                    if error_type:
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                    if not continue_on_error:
                        print(f"Error processing {inp_path}: {error}", file=sys.stderr)
                        sys.exit(1)
    else:
        # Sequential processing
        for kern_file in kern_files:
            out_file = output_path / kern_file.name
            _, success, error_type, error, pass_stats = normalize_file(kern_file, out_file)
            if success:
                success_count += 1
                if pass_stats:
                    canonicalize = pass_stats.get("canonicalize_note_order", {})
                    canonicalize_unknown_total += int(canonicalize.get("unknown_char_drops_total", 0))
                    for ch, count in canonicalize.get("unknown_char_drop_counts", {}).items():
                        canonicalize_unknown_counts[ch] = canonicalize_unknown_counts.get(ch, 0) + int(count)
            else:
                error_count += 1
                errors.append((kern_file, error_type, error))
                if error_type:
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                if not continue_on_error:
                    print(f"Error processing {kern_file}: {error}", file=sys.stderr)
                    sys.exit(1)

    if not quiet:
        print(f"\nResults:", file=sys.stderr)
        print(f"  Success: {success_count}", file=sys.stderr)
        print(f"  Errors:  {error_count}", file=sys.stderr)
        print(f"\nNormalized files saved to {output_dir}", file=sys.stderr)

        if errors:
            print(f"\nFailed files:", file=sys.stderr)
            for path, err_type, err in errors[:10]:
                prefix = f"{err_type}: " if err_type else ""
                print(f"  {path.name}: {prefix}{err}", file=sys.stderr)
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more", file=sys.stderr)

    if stats_json:
        stats_path = Path(stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "input_dir": str(input_path),
            "output_dir": str(output_path),
            "input_count": len(kern_files),
            "success_count": success_count,
            "error_count": error_count,
            "pass_rate": (success_count / len(kern_files)) if kern_files else 0.0,
            "duration_seconds": time.monotonic() - start_time,
            "error_types": {
                name: int(count)
                for name, count in sorted(error_types.items(), key=lambda item: (-item[1], item[0]))
            },
            "failed_examples": [
                {
                    "file": str(path),
                    "error_type": err_type,
                    "error": err,
                }
                for path, err_type, err in errors[:20]
            ],
            "normalization_pass_stats": {
                "canonicalize_note_order": {
                    "unknown_char_drops_total": int(canonicalize_unknown_total),
                    "unknown_char_drop_counts": {
                        name: int(count)
                        for name, count in sorted(
                            canonicalize_unknown_counts.items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                    },
                }
            },
        }
        stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    fire.Fire(normalize_kern_files)


if __name__ == "__main__":
    main()
