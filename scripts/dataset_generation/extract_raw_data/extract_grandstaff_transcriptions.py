#!/usr/bin/env python3
"""
Extract antoniorv6/grandstaff dataset ekern transcriptions.

Downloads the HuggingFace dataset and saves each transcription as a raw .ekern file.

Usage:
    python -m scripts.dataset_generation.raw_kern_file.extract_grandstaff_transcriptions
    python -m scripts.dataset_generation.raw_kern_file.extract_grandstaff_transcriptions --split train
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
from datasets import load_dataset

OUTPUT_DIR = Path("data/interim")


def extract_grandstaff(
    split: str | None = None,
    output_dir: str = str(OUTPUT_DIR),
    quiet: bool = False,
):
    """
    Extract grandstaff dataset ekern transcriptions.

    Args:
        split: Dataset split to extract (train, val, test). If None, extracts all splits.
        output_dir: Base directory to save .ekern files to.
        quiet: Suppress progress output.
    """
    output_path = Path(output_dir)

    if not quiet:
        print("Loading antoniorv6/grandstaff dataset...", file=sys.stderr)

    ds = load_dataset("antoniorv6/grandstaff")

    splits = [split] if split else list(ds.keys())

    total_count = 0
    for split_name in splits:
        if split_name not in ds:
            print(f"Error: Split '{split_name}' not found in dataset", file=sys.stderr)
            sys.exit(1)

        split_data = ds[split_name]
        split_output = output_path / split_name / "grandstaff" / "0_raw_ekern"
        split_output.mkdir(parents=True, exist_ok=True)

        if not quiet:
            print(f"Extracting {len(split_data)} samples from '{split_name}' split...", file=sys.stderr)

        for idx, sample in enumerate(split_data):
            transcription = sample["transcription"]

            filename = f"{idx:06d}.ekern"
            output_file = split_output / filename
            output_file.write_text(transcription, encoding="utf-8")

        total_count += len(split_data)

        if not quiet:
            print(f"  Saved to {split_output}/", file=sys.stderr)

    if not quiet:
        print(f"\nDone! Extracted {total_count} samples total.", file=sys.stderr)
        print(f"Output directory: {output_path}", file=sys.stderr)


def main():
    fire.Fire(extract_grandstaff)


if __name__ == "__main__":
    main()
