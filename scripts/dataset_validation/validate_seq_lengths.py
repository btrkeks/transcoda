#!/usr/bin/env python3
"""Validate that all samples in a dataset are within the max sequence length."""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset
from transformers import PreTrainedTokenizerFast

VOCAB_DIR = "vocab/bpe4k"
MAX_SEQ_LEN = 2000
HISTOGRAM_WIDTH = 50

DATA_DIR = Path("data/datasets")
DEFAULT_DATASETS = [
    DATA_DIR / "train_full",
    *sorted(p for p in DATA_DIR.glob("validation/*") if p.is_dir()),
]


@dataclass
class DatasetStats:
    label: str
    num_samples: int
    lengths: list[int]
    num_violations: int

    def percentile(self, p: float) -> int:
        idx = int(len(self.lengths) * p / 100)
        return self.lengths[min(idx, len(self.lengths) - 1)]


def print_histogram(lengths: list[int], max_seq_len: int, num_buckets: int = 10) -> None:
    bucket_size = max_seq_len // num_buckets
    # +1 for the overflow bucket (> max_seq_len)
    counts = [0] * (num_buckets + 1)
    for length in lengths:
        bucket = min(length // bucket_size, num_buckets)
        counts[bucket] += 1

    max_count = max(counts) if counts else 0
    print(f"\n  {'Range':>16s}  {'Count':>7s}  Distribution")
    for i, count in enumerate(counts):
        if i < num_buckets:
            lo = i * bucket_size
            hi = lo + bucket_size - 1
            label = f"{lo:>5d}-{hi:<5d}"
        else:
            label = f"{max_seq_len:>5d}+     "
        bar_len = int(count / max_count * HISTOGRAM_WIDTH) if max_count else 0
        bar = "\u2588" * bar_len
        print(f"  {label:>16s}  {count:>7d}  {bar}")


def print_percentiles(stats: DatasetStats) -> None:
    print(f"\n  Min: {stats.lengths[0]}  p50: {stats.percentile(50)}"
          f"  p90: {stats.percentile(90)}  p95: {stats.percentile(95)}"
          f"  p99: {stats.percentile(99)}  Max: {stats.lengths[-1]}"
          f"  Mean: {statistics.mean(stats.lengths):.0f}")


def validate_dataset(
    path: Path, tokenizer: PreTrainedTokenizerFast, max_seq_len: int
) -> DatasetStats:
    label = str(path).removeprefix(str(DATA_DIR) + "/")

    print(f"\n{'=' * 60}")
    print(f" {label}")
    print(f"{'=' * 60}")

    ds = Dataset.load_from_disk(str(path))
    print(f"Loaded {len(ds)} samples")

    # Select only transcription column to avoid decoding images
    ds = ds.select_columns(["transcription"])

    lengths: list[int] = []
    violations: list[tuple[int, int]] = []
    for i, example in enumerate(ds):
        n = len(tokenizer(example["transcription"], add_special_tokens=True)["input_ids"])
        lengths.append(n)
        if n > max_seq_len:
            violations.append((i, n))

    lengths.sort()

    stats = DatasetStats(
        label=label,
        num_samples=len(ds),
        lengths=lengths,
        num_violations=len(violations),
    )

    print_percentiles(stats)
    print_histogram(lengths, max_seq_len)

    if violations:
        print(f"\n{len(violations)} samples exceed {max_seq_len} tokens:")
        for idx, length in violations[:10]:
            print(f"  Sample {idx}: {length} tokens")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more")
    else:
        print(f"\nAll {len(ds)} samples are within {max_seq_len} tokens")

    return stats


def print_summary_table(all_stats: list[DatasetStats]) -> None:
    print(f"\n{'=' * 90}")
    print(" Summary")
    print(f"{'=' * 90}")
    header = f"{'Dataset':<30s} {'Samples':>8s} {'Min':>6s} {'p50':>6s} {'p90':>6s} {'p95':>6s} {'p99':>6s} {'Max':>6s} {'Viol.':>6s}"
    print(header)
    print("-" * len(header))
    for s in all_stats:
        print(
            f"{s.label:<30s} {s.num_samples:>8d} {s.lengths[0]:>6d} {s.percentile(50):>6d}"
            f" {s.percentile(90):>6d} {s.percentile(95):>6d} {s.percentile(99):>6d}"
            f" {s.lengths[-1]:>6d} {s.num_violations:>6d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset sequence lengths")
    parser.add_argument(
        "dataset_paths", nargs="*", type=Path, help="Paths to dataset directories (default: train_full + validation/*)"
    )
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--tokenizer", type=str, default=VOCAB_DIR, help="Path to tokenizer directory (default: %(default)s)")
    args = parser.parse_args()

    dataset_paths = args.dataset_paths or DEFAULT_DATASETS

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    all_stats: list[DatasetStats] = []
    any_failed = False
    for path in dataset_paths:
        if not path.exists():
            print(f"\nSkipping {path} (not found)")
            continue
        stats = validate_dataset(path, tokenizer, args.max_seq_len)
        all_stats.append(stats)
        if stats.num_violations > 0:
            any_failed = True

    if all_stats:
        print_summary_table(all_stats)

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
