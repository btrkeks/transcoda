#!/usr/bin/env python3
"""
Filter kern files through a composable filter pipeline.

This script filters kern files and copies passing files to an output directory,
designed for integration into a processing pipeline:
    1) convert -> 2) pre-filter -> 3) clean -> 4) normalize

Examples:
    # Basic usage - copy passing files to output directory
    uv run -m pipelines.datasets.generation.filtering.main data/interim/pdmx/1_kern_conversions/ data/interim/pdmx/2_pre_filtered/

    # With JSON stats
    uv run -m pipelines.datasets.generation.filtering.main input/ output/ --stats-json stats.json

    # Select specific filters, including rhythm if needed
    uv run -m pipelines.datasets.generation.filtering.main input/ output/ --filters staves,rational_duration,rhythm
"""

import shutil
import sys
from pathlib import Path

import fire

from . import FilterPipeline
from .filters import (
    FILTER_REGISTRY,
    RationalDurationFilter,
    RhythmFilter,
    StavesFilter,
)


def _same_file(src_path: Path, dst_path: Path) -> bool:
    try:
        return src_path.samefile(dst_path)
    except FileNotFoundError:
        return False
    except OSError:
        return False


def _replace_with_copy(src_path: Path, dst_path: Path) -> None:
    if _same_file(src_path, dst_path):
        return
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    shutil.copy2(src_path, dst_path)


def _replace_with_hardlink(src_path: Path, dst_path: Path) -> None:
    if _same_file(src_path, dst_path):
        return
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    dst_path.hardlink_to(src_path)


def _materialize_passing_file(src_path: Path, dst_path: Path, link_mode: str) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if link_mode == "copy":
        _replace_with_copy(src_path, dst_path)
        return

    if link_mode == "hardlink":
        _replace_with_hardlink(src_path, dst_path)
        return

    if link_mode != "auto":
        raise ValueError(
            f"Invalid link_mode {link_mode!r}. Expected one of: auto, copy, hardlink"
        )

    try:
        _replace_with_hardlink(src_path, dst_path)
    except OSError:
        _replace_with_copy(src_path, dst_path)


def filter_kern_files(
    input_dir: str,
    output_dir: str,
    stats_json: str | None = None,
    filters: str | None = None,
    no_rhythm: bool = False,
    rhythm_checker: str = "./binaries/rhythm_checker/target/release/rhythm_checker",
    allow_anacrusis: bool = True,
    allow_incomplete_final: bool = True,
    quiet: bool = False,
    workers: int = 1,
    link_mode: str = "auto",
):
    """
    Filter kern files and materialize passing files to output directory.

    Args:
        input_dir: Directory containing kern files to check
        output_dir: Directory to materialize passing files to
        stats_json: Path to save JSON statistics
        filters: Comma-separated list of filters to use (e.g., "staves,rational_duration,rhythm")
        no_rhythm: Deprecated compatibility flag; rhythm is not run by default
        rhythm_checker: Path to rhythm_checker binary
        allow_anacrusis: Allow incomplete first measure (default: True)
        allow_incomplete_final: Allow incomplete final measure (default: True)
        quiet: Suppress progress output
        workers: Number of parallel workers (default: 1 for sequential)
        link_mode: How to materialize passing files: auto, copy, or hardlink
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    valid_link_modes = {"auto", "copy", "hardlink"}

    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    if link_mode not in valid_link_modes:
        print(
            f"Error: Invalid link_mode '{link_mode}'. Available: {sorted(valid_link_modes)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build filter list
    filter_instances = []

    if filters:
        # Use specified filters
        # Fire may parse comma-separated values as a tuple
        if isinstance(filters, tuple):
            filter_names = list(filters)
        else:
            filter_names = [f.strip() for f in filters.split(",")]
        for name in filter_names:
            if name not in FILTER_REGISTRY:
                print(
                    f"Error: Unknown filter '{name}'. Available: {list(FILTER_REGISTRY.keys())}",
                    file=sys.stderr,
                )
                sys.exit(1)

            if name == "rhythm":
                filter_instances.append(
                    RhythmFilter(
                        binary_path=rhythm_checker,
                        allow_anacrusis=allow_anacrusis,
                        allow_incomplete_final=allow_incomplete_final,
                    )
                )
            else:
                filter_instances.append(FILTER_REGISTRY[name]())
    else:
        # Use default filter order.
        # Reject unusually large systems before later content-quality filters
        filter_instances.append(StavesFilter())
        # Always filter rational durations (mensural notation with % not in tokenizer)
        filter_instances.append(RationalDurationFilter())
        # Rhythm validation is temporarily disabled in the default dataset filter
        # pipeline. It remains available via --filters rhythm for targeted runs.

    if not filter_instances:
        print("Error: No filters specified", file=sys.stderr)
        sys.exit(1)

    # Collect input files
    kern_files = sorted(input_path.glob("**/*.krn"))
    if not kern_files:
        print(f"Error: No .krn files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    filter_names_str = ", ".join(f.name for f in filter_instances)
    if not quiet:
        print(
            f"Filtering {len(kern_files)} files through: {filter_names_str}",
            file=sys.stderr,
        )

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Run pipeline and materialize passing files
    pipeline = FilterPipeline(filter_instances)
    materialized_count = 0

    def destination_for(src_path: Path) -> Path:
        return output_path / src_path.relative_to(input_path)

    if workers > 1:
        # Parallel processing
        for src_path in pipeline.filter_files_parallel(kern_files, workers):
            dst_path = destination_for(src_path)
            _materialize_passing_file(src_path, dst_path, link_mode)
            materialized_count += 1
    else:
        # Sequential processing
        for src_path in pipeline.filter_files(kern_files):
            dst_path = destination_for(src_path)
            _materialize_passing_file(src_path, dst_path, link_mode)
            materialized_count += 1

    # Print stats
    stats = pipeline.stats
    if not quiet:
        print("", file=sys.stderr)
        print("Results:", file=sys.stderr)
        print(f"  {stats.summary()}", file=sys.stderr)
        print(
            f"\nMaterialized {materialized_count} files to {output_dir}",
            file=sys.stderr,
        )

    # Save JSON stats if requested
    if stats_json:
        stats_path = Path(stats_json)
        stats.save_json(stats_path)
        if not quiet:
            print(f"Stats saved to {stats_json}", file=sys.stderr)


def main():
    fire.Fire(filter_kern_files)


if __name__ == "__main__":
    main()
