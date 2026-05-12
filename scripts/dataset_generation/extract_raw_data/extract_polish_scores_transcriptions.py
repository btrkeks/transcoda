#!/usr/bin/env python3
"""
Extract PRAIG/polish-scores dataset ekern transcriptions.

Downloads the HuggingFace dataset and saves each transcription as a raw .ekern file.

Usage:
    python -m scripts.dataset_generation.raw_kern_file.extract_polish_scores_transcriptions
    python -m scripts.dataset_generation.raw_kern_file.extract_polish_scores_transcriptions --target_split val
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
from datasets import load_dataset

OUTPUT_DIR = Path("data/interim")


def extract_polish_scores(
    target_split: str | None = None,
    output_dir: str = str(OUTPUT_DIR),
    quiet: bool = False,
):
    """
    Extract polish-scores dataset ekern transcriptions.

    Args:
        target_split: When set, dumps all HF splits into this single split directory
            (e.g. "val" → data/interim/val/polish-scores/0_raw_ekern/).
            When None, each HF split goes to its own directory.
        output_dir: Base directory to save .ekern files to.
        quiet: Suppress progress output.
    """
    output_path = Path(output_dir)

    if not quiet:
        print("Loading PRAIG/polish-scores dataset...", file=sys.stderr)

    ds = load_dataset("PRAIG/polish-scores")

    total_count = 0
    for split_name in ds.keys():
        split_data = ds[split_name]

        dest_split = target_split if target_split else split_name
        split_output = output_path / dest_split / "polish-scores" / "0_raw_ekern"
        split_output.mkdir(parents=True, exist_ok=True)

        if not quiet:
            print(f"Extracting {len(split_data)} samples from '{split_name}' split...", file=sys.stderr)

        for idx, sample in enumerate(split_data):
            transcription = sample["transcription"]

            filename = f"{split_name}_{idx:06d}.ekern"
            output_file = split_output / filename
            output_file.write_text(transcription, encoding="utf-8")

        total_count += len(split_data)

        if not quiet:
            print(f"  Saved to {split_output}/", file=sys.stderr)

    if not quiet:
        print(f"\nDone! Extracted {total_count} samples total.", file=sys.stderr)
        print(f"Output directory: {output_path}", file=sys.stderr)


def main():
    fire.Fire(extract_polish_scores)


if __name__ == "__main__":
    main()
