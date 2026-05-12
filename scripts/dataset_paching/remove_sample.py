#!/usr/bin/env python3
"""Remove a sample from the train_full dataset by index."""

import argparse
import shutil
import sys
from pathlib import Path

from datasets import load_from_disk

DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "datasets" / "train_full"


def main():
    parser = argparse.ArgumentParser(description="Remove a sample from train_full by index")
    parser.add_argument("index", type=int, help="Index of sample to remove")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from {DATASET_PATH}...")
    ds = load_from_disk(str(DATASET_PATH))
    print(f"Original dataset size: {len(ds)}")

    if args.index < 0 or args.index >= len(ds):
        print(f"Error: Index {args.index} out of range [0, {len(ds) - 1}]", file=sys.stderr)
        sys.exit(1)

    print(f"Removing sample at index {args.index}...")
    indices_to_keep = list(range(args.index)) + list(range(args.index + 1, len(ds)))
    new_ds = ds.select(indices_to_keep)
    print(f"New dataset size: {len(new_ds)}")

    temp_path = DATASET_PATH.with_name(DATASET_PATH.name + "_temp")
    print(f"Saving to temp location: {temp_path}...")
    if temp_path.exists():
        shutil.rmtree(temp_path)
    new_ds.save_to_disk(str(temp_path))

    print("Replacing original dataset...")
    backup_path = DATASET_PATH.with_name(DATASET_PATH.name + "_backup")
    if backup_path.exists():
        shutil.rmtree(backup_path)
    shutil.move(DATASET_PATH, backup_path)
    shutil.move(temp_path, DATASET_PATH)
    shutil.rmtree(backup_path)

    print("Done!")


if __name__ == "__main__":
    main()
