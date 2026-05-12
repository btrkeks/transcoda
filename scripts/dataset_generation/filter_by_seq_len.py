#!/usr/bin/env python3
"""
Filter a HuggingFace dataset by token sequence length.

Removes samples whose transcription exceeds a given token-length threshold.

Length is computed with tokenizer special tokens included so the filtering
contract matches training-time sequence length checks.

Examples:
    # Filter in-place (replaces original dataset)
    python scripts/dataset_generation/filter_by_seq_len.py \
        --dataset_path data/datasets/train_full \
        --vocab_path vocab/bpe4k \
        --max_seq_len 2000

    # Save to a separate output directory
    python scripts/dataset_generation/filter_by_seq_len.py \
        --dataset_path data/datasets/train_full \
        --vocab_path vocab/bpe4k \
        --max_seq_len 2000 \
        --output_path data/datasets/train_full_filtered
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import fire
import numpy as np
from datasets import load_from_disk
from tokenizers import Tokenizer


def filter_by_seq_len(
    dataset_path: str,
    vocab_path: str,
    max_seq_len: int,
    output_path: str | None = None,
    stats_json: str | None = None,
):
    """
    Filter a HuggingFace dataset to remove samples exceeding a token-length threshold.

    Args:
        dataset_path: Path to the HuggingFace dataset on disk.
        vocab_path: Path to the tokenizer directory (containing tokenizer.json).
        max_seq_len: Maximum allowed token sequence length (inclusive).
        output_path: Where to save the filtered dataset. If None, replaces the original.
        stats_json: Optional path to save run statistics as JSON.
    """
    start_time = time.monotonic()
    dataset_dir = Path(dataset_path)
    if not dataset_dir.is_dir():
        print(f"Error: {dataset_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    tokenizer_path = Path(vocab_path) / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"Error: tokenizer not found at {tokenizer_path}", file=sys.stderr)
        sys.exit(1)

    target_output = output_path if output_path is not None else f"{dataset_path} (atomic replace)"
    print("Filter settings:", file=sys.stderr)
    print(f"  dataset_path: {dataset_path}", file=sys.stderr)
    print(f"  vocab_path: {vocab_path}", file=sys.stderr)
    print(f"  max_seq_len: {max_seq_len}", file=sys.stderr)
    print(f"  output:      {target_output}", file=sys.stderr)

    # Load dataset and tokenizer
    print(f"Loading dataset from {dataset_path}...", file=sys.stderr)
    dataset = load_from_disk(dataset_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    total = len(dataset)
    print(f"Total samples: {total}", file=sys.stderr)

    # Compute token lengths for all samples.
    # Use add_special_tokens=True to match training/validation length semantics.
    print("Tokenizing transcriptions...", file=sys.stderr)
    lengths = np.array(
        [len(tokenizer.encode(t, add_special_tokens=True).ids) for t in dataset["transcription"]]
    )

    # Print before-filter stats
    print(f"\nBefore filtering:", file=sys.stderr)
    print(f"  Max length:  {int(lengths.max())}", file=sys.stderr)
    print(f"  Mean length: {lengths.mean():.1f}", file=sys.stderr)
    print(f"  P99 length:  {int(np.percentile(lengths, 99))}", file=sys.stderr)

    # Filter
    keep_mask = lengths <= max_seq_len
    kept = int(keep_mask.sum())
    removed = total - kept

    keep_indices = np.where(keep_mask)[0].tolist()
    filtered_dataset = dataset.select(keep_indices)

    # Print after-filter stats
    kept_lengths = lengths[keep_mask]
    print(f"\nAfter filtering (max_seq_len={max_seq_len}):", file=sys.stderr)
    print(f"  Kept:    {kept} ({kept / total * 100:.3f}%)", file=sys.stderr)
    print(f"  Removed: {removed} ({removed / total * 100:.3f}%)", file=sys.stderr)
    if len(kept_lengths) > 0:
        print(f"  Max length:  {int(kept_lengths.max())}", file=sys.stderr)
        print(f"  Mean length: {kept_lengths.mean():.1f}", file=sys.stderr)
        print(f"  P99 length:  {int(np.percentile(kept_lengths, 99))}", file=sys.stderr)

    if removed == 0:
        print(
            "\nNo samples exceed max_seq_len; dataset unchanged. Skipping save/replace.",
            file=sys.stderr,
        )
        if stats_json:
            stats_path = Path(stats_json)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": "1.0",
                "dataset_path": str(dataset_dir),
                "output_path": output_path,
                "max_seq_len": int(max_seq_len),
                "total": int(total),
                "kept": int(kept),
                "removed": int(removed),
                "removed_rate": 0.0,
                "before": {
                    "max_length": int(lengths.max()),
                    "mean_length": float(lengths.mean()),
                    "p99_length": int(np.percentile(lengths, 99)),
                },
                "after": {
                    "max_length": int(lengths.max()),
                    "mean_length": float(lengths.mean()),
                    "p99_length": int(np.percentile(lengths, 99)),
                },
                "duration_seconds": time.monotonic() - start_time,
            }
            stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    # Save
    if output_path is not None:
        save_dir = Path(output_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving filtered dataset to {output_path}...", file=sys.stderr)
        filtered_dataset.save_to_disk(output_path)
    else:
        # Atomic replace: save to temp dir, then swap
        tmp_dir = dataset_dir.parent / f"{dataset_dir.name}_filtered_tmp"
        print(f"\nSaving filtered dataset to temp dir...", file=sys.stderr)
        filtered_dataset.save_to_disk(str(tmp_dir))

        backup_dir = dataset_dir.parent / f"{dataset_dir.name}_pre_filter_backup"
        print(f"Replacing original dataset (backup at {backup_dir})...", file=sys.stderr)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        dataset_dir.rename(backup_dir)
        tmp_dir.rename(dataset_dir)
        try:
            shutil.rmtree(backup_dir)
        except OSError:
            print(
                f"Warning: could not remove backup dir {backup_dir}, remove it manually.",
                file=sys.stderr,
            )

    if stats_json:
        stats_path = Path(stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "dataset_path": str(dataset_dir),
            "output_path": output_path,
            "max_seq_len": int(max_seq_len),
            "total": int(total),
            "kept": int(kept),
            "removed": int(removed),
            "removed_rate": (removed / total) if total else 0.0,
            "before": {
                "max_length": int(lengths.max()),
                "mean_length": float(lengths.mean()),
                "p99_length": int(np.percentile(lengths, 99)),
            },
            "after": {
                "max_length": int(kept_lengths.max()) if len(kept_lengths) > 0 else 0,
                "mean_length": float(kept_lengths.mean()) if len(kept_lengths) > 0 else 0.0,
                "p99_length": (
                    int(np.percentile(kept_lengths, 99))
                    if len(kept_lengths) > 0
                    else 0
                ),
            },
            "duration_seconds": time.monotonic() - start_time,
        }
        stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    fire.Fire(filter_by_seq_len)
