#!/usr/bin/env python3
"""Clean converted kern files before AST parser based normalization."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fire

from .presets import clean_kern_for_parser_with_context


def _extract_pass_stats(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "remove_musescore_palette_template": ctx.get(
            "remove_musescore_palette_template",
            {},
        ),
        "remove_rest_decorations": ctx.get("remove_rest_decorations", {}),
        "remove_exact_interpretations": ctx.get("remove_exact_interpretations", {}),
        "remove_grandstaff_single_y_from_notes": ctx.get(
            "remove_grandstaff_single_y_from_notes",
            {},
        ),
    }


def _merge_pass_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    template_stats = source.get("remove_musescore_palette_template", {})
    target["remove_musescore_palette_template"]["removed_templates"] += int(
        template_stats.get("removed_templates", 0)
    )

    rest_stats = source.get("remove_rest_decorations", {})
    target["remove_rest_decorations"]["changed_tokens"] += int(
        rest_stats.get("changed_tokens", 0)
    )

    interpretation_stats = source.get("remove_exact_interpretations", {})
    target["remove_exact_interpretations"]["removed_total"] += int(
        interpretation_stats.get("removed_total", 0)
    )
    for token, count in interpretation_stats.get("removed_counts", {}).items():
        removed_counts = target["remove_exact_interpretations"]["removed_counts"]
        removed_counts[token] = removed_counts.get(token, 0) + int(count)

    grandstaff_y_stats = source.get("remove_grandstaff_single_y_from_notes", {})
    target["remove_grandstaff_single_y_from_notes"]["changed_tokens"] += int(
        grandstaff_y_stats.get("changed_tokens", 0)
    )


def clean_file(
    input_path: Path,
    output_path: Path,
    source_dataset: str | None = None,
) -> tuple[Path, bool, str | None, str | None, dict[str, Any] | None]:
    """Clean one kern file."""
    try:
        content = input_path.read_text(encoding="utf-8")
        cleaned, ctx = clean_kern_for_parser_with_context(
            content,
            source_dataset=source_dataset,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned, encoding="utf-8")
        return (input_path, True, None, None, _extract_pass_stats(ctx))
    except Exception as e:
        return (input_path, False, type(e).__name__, str(e), None)


def clean_kern_files(
    input_dir: str = "data/interim/pdmx/2_pre_filtered",
    output_dir: str = "data/interim/pdmx/3_cleaned",
    workers: int = 1,
    quiet: bool = False,
    continue_on_error: bool = True,
    stats_json: str | None = None,
    source_dataset: str | None = None,
) -> None:
    """
    Clean all kern files in a directory.

    Args:
        input_dir: Directory containing filtered kern files.
        output_dir: Directory to save cleaned files to.
        workers: Number of parallel workers.
        quiet: Suppress progress output.
        continue_on_error: Continue processing if one file fails.
        stats_json: Optional path to write run statistics as JSON.
        source_dataset: Optional source dataset name for dataset-specific cleanup.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    start_time = time.monotonic()

    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    kern_files = sorted(input_path.glob("**/*.krn"))
    if not kern_files:
        print(f"Error: No .krn files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not quiet:
        print(f"Cleaning {len(kern_files)} files...", file=sys.stderr)

    output_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0
    errors: list[tuple[Path, str | None, str | None]] = []
    error_types: dict[str, int] = {}
    pass_stats: dict[str, Any] = {
        "remove_musescore_palette_template": {"removed_templates": 0},
        "remove_rest_decorations": {"changed_tokens": 0},
        "remove_exact_interpretations": {
            "removed_total": 0,
            "removed_counts": {},
        },
        "remove_grandstaff_single_y_from_notes": {"changed_tokens": 0},
    }

    tasks = [(path, output_path / path.relative_to(input_path)) for path in kern_files]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(clean_file, inp, out, source_dataset): inp
                for inp, out in tasks
            }
            for future in as_completed(futures):
                inp_path, success, error_type, error, file_stats = future.result()
                if success:
                    success_count += 1
                    if file_stats:
                        _merge_pass_stats(pass_stats, file_stats)
                else:
                    error_count += 1
                    errors.append((inp_path, error_type, error))
                    if error_type:
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                    if not continue_on_error:
                        print(f"Error processing {inp_path}: {error}", file=sys.stderr)
                        sys.exit(1)
    else:
        for inp, out in tasks:
            inp_path, success, error_type, error, file_stats = clean_file(
                inp,
                out,
                source_dataset,
            )
            if success:
                success_count += 1
                if file_stats:
                    _merge_pass_stats(pass_stats, file_stats)
            else:
                error_count += 1
                errors.append((inp_path, error_type, error))
                if error_type:
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                if not continue_on_error:
                    print(f"Error processing {inp_path}: {error}", file=sys.stderr)
                    sys.exit(1)

    if not quiet:
        print("\nResults:", file=sys.stderr)
        print(f"  Success: {success_count}", file=sys.stderr)
        print(f"  Errors:  {error_count}", file=sys.stderr)
        print(f"\nCleaned files saved to {output_dir}", file=sys.stderr)

        if errors:
            print("\nFailed files:", file=sys.stderr)
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
            "source_dataset": source_dataset,
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
            "cleanup_pass_stats": pass_stats,
        }
        stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    fire.Fire(clean_kern_files)


if __name__ == "__main__":
    main()
