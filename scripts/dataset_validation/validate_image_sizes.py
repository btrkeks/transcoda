#!/usr/bin/env python3
"""Validate that all images in a dataset have the expected dimensions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import Dataset

DEFAULT_WIDTH = 1050
DEFAULT_HEIGHT = 1485


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset image sizes")
    parser.add_argument("dataset_path", type=Path, help="Path to dataset directory")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}...")
    ds = Dataset.load_from_disk(str(args.dataset_path))
    print(f"Loaded {len(ds)} samples")

    # Select only image column to avoid decoding transcriptions
    ds = ds.select_columns(["image"])

    expected = (args.width, args.height)
    violations = []
    for i, example in enumerate(ds):
        size = example["image"].size  # PIL returns (width, height)
        if size != expected:
            violations.append((i, size))

    if violations:
        print(f"\n{len(violations)} samples have unexpected dimensions (expected {expected[0]}x{expected[1]}):")
        for idx, size in violations[:10]:
            print(f"  Sample {idx}: {size[0]}x{size[1]}")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more")
        sys.exit(1)
    else:
        print(f"All {len(ds)} samples are {expected[0]}x{expected[1]}")


if __name__ == "__main__":
    main()
