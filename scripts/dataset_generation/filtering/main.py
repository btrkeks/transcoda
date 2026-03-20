#!/usr/bin/env python3
"""
Filter kern files through a composable filter pipeline.

This script filters kern files and copies passing files to an output directory,
designed for integration into a processing pipeline:
    1) convert -> 2) filter -> 3) normalize -> 4) generate into final dataset

Examples:
    # Basic usage - copy passing files to output directory
    uv run -m scripts.filtering.main data/interim/pdmx/1_kern_conversions/ data/interim/pdmx/2_filtered/

    # With JSON stats
    uv run -m scripts.filtering.main input/ output/ --stats-json stats.json

    # Skip rhythm check (faster)
    uv run -m scripts.filtering.main input/ output/ --no-rhythm

    # Select specific filters
    uv run -m scripts.filtering.main input/ output/ --filters utf8,termination
"""

import shutil
import sys
from pathlib import Path

import fire

from . import FilterPipeline
from .filters import (
    FILTER_REGISTRY,
    AccidentalsFilter,
    ExcessiveOctaveFilter,
    HeaderClefFilter,
    RationalDurationFilter,
    RhythmFilter,
    TerminationFilter,
    UTF8Filter,
)


def filter_kern_files(
    input_dir: str,
    output_dir: str,
    stats_json: str | None = None,
    filters: str | None = None,
    no_rhythm: bool = False,
    no_utf8: bool = False,
    no_termination: bool = False,
    rhythm_checker: str = "./binaries/rhythm_checker/target/release/rhythm_checker",
    allow_anacrusis: bool = True,
    allow_incomplete_final: bool = True,
    quiet: bool = False,
    workers: int = 1,
):
    """
    Filter kern files and copy passing files to output directory.

    Args:
        input_dir: Directory containing kern files to check
        output_dir: Directory to copy passing files to
        stats_json: Path to save JSON statistics
        filters: Comma-separated list of filters to use (e.g., "utf8,termination,rhythm")
        no_rhythm: Skip rhythm validation
        no_utf8: Skip UTF-8 validation
        no_termination: Skip termination validation
        rhythm_checker: Path to rhythm_checker binary
        allow_anacrusis: Allow incomplete first measure (default: True)
        allow_incomplete_final: Allow incomplete final measure (default: True)
        quiet: Suppress progress output
        workers: Number of parallel workers (default: 1 for sequential)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
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
        # Use default filter order with exclusions
        if not no_utf8:
            filter_instances.append(UTF8Filter())
        if not no_termination:
            filter_instances.append(TerminationFilter())
        filter_instances.append(HeaderClefFilter())
        # Always filter rational durations (mensural notation with % not in tokenizer)
        filter_instances.append(RationalDurationFilter())
        # Filter malformed accidentals from bad transposition in source files
        filter_instances.append(AccidentalsFilter())
        # Filter corrupted pitches with excessive octave repetitions
        filter_instances.append(ExcessiveOctaveFilter())
        if not no_rhythm:
            rhythm_binary = Path(rhythm_checker)
            if not rhythm_binary.exists():
                print(
                    f"Warning: rhythm_checker not found at {rhythm_checker}, skipping rhythm filter",
                    file=sys.stderr,
                )
            else:
                filter_instances.append(
                    RhythmFilter(
                        binary_path=rhythm_checker,
                        allow_anacrusis=allow_anacrusis,
                        allow_incomplete_final=allow_incomplete_final,
                    )
                )

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

    # Run pipeline and copy passing files
    pipeline = FilterPipeline(filter_instances)
    copied_count = 0

    if workers > 1:
        # Parallel processing
        for src_path in pipeline.filter_files_parallel(kern_files, workers):
            dst_path = output_path / src_path.name
            shutil.copy2(src_path, dst_path)
            copied_count += 1
    else:
        # Sequential processing
        for src_path in pipeline.filter_files(kern_files):
            dst_path = output_path / src_path.name
            shutil.copy2(src_path, dst_path)
            copied_count += 1

    # Print stats
    stats = pipeline.stats
    if not quiet:
        print("", file=sys.stderr)
        print("Results:", file=sys.stderr)
        print(f"  {stats.summary()}", file=sys.stderr)
        print(f"\nCopied {copied_count} files to {output_dir}", file=sys.stderr)

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
