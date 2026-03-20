#!/usr/bin/env python3
"""Assemble split-aware PRAIG/polish-scores datasets from normalized .krn files."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import datasets
import fire
from datasets import Dataset, Features, Value, load_dataset
from PIL import Image

from scripts.dataset_generation.data_spec import (
    DEFAULT_DATA_SPEC_PATH,
    resolve_image_size_from_spec,
)

UPSTREAM_DATASET_NAME = "PRAIG/polish-scores"
VALID_SPLITS = frozenset({"train", "val", "test"})


def parse_filename(filename: str) -> tuple[str, int]:
    """Parse a filename like 'train_000003.krn' into (split, index)."""
    match = re.match(r"^([a-z]+)_(\d+)\.krn$", filename)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")
    return match.group(1), int(match.group(2))


def resize_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize image to target dimensions (width x height)."""
    return image.resize((target_width, target_height), Image.LANCZOS)


def _parse_include_splits(include_splits: str | Iterable[str] | None) -> list[str] | None:
    if include_splits is None:
        return None

    if isinstance(include_splits, str):
        raw_values = include_splits.split(",")
    else:
        raw_values = list(include_splits)

    values = [str(value).strip() for value in raw_values if str(value).strip()]
    if not values:
        return None

    invalid = sorted(set(values) - VALID_SPLITS)
    if invalid:
        raise ValueError(
            f"Unsupported split(s) {invalid}; valid splits are {sorted(VALID_SPLITS)}"
        )

    # Preserve user order while removing duplicates.
    return list(dict.fromkeys(values))


def assemble_polish_scores_dataset(
    normalized_dir: str = "data/interim/val/polish-scores/4_manual_fixes",
    output_dir: str = "data/datasets/validation/polish",
    include_splits: str | list[str] | tuple[str, ...] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    data_spec_path: str = str(DEFAULT_DATA_SPEC_PATH),
    strict_data_spec: bool = True,
    hf_cache_dir: str | None = None,
    quiet: bool = False,
) -> dict[str, int | list[str] | str]:
    """Assemble a Polish-scores HuggingFace dataset from normalized .krn files."""
    image_width, image_height = resolve_image_size_from_spec(
        image_width=image_width,
        image_height=image_height,
        data_spec_path=data_spec_path,
        strict_data_spec=strict_data_spec,
    )

    normalized_path = Path(normalized_dir)
    output_path = Path(output_dir)
    selected_splits = _parse_include_splits(include_splits)

    krn_files = sorted(normalized_path.glob("*.krn"))
    if not krn_files:
        raise ValueError(f"No .krn files found in {normalized_path}")

    filtered_krn_files = []
    for krn_path in krn_files:
        split, _ = parse_filename(krn_path.name)
        if selected_splits is None or split in selected_splits:
            filtered_krn_files.append(krn_path)

    if not filtered_krn_files:
        requested = selected_splits if selected_splits is not None else sorted(VALID_SPLITS)
        raise ValueError(
            f"No .krn files matched include_splits={requested} in {normalized_path}"
        )

    if not quiet:
        print(
            f"Found {len(filtered_krn_files)} normalized .krn files "
            f"(selected_splits={selected_splits or 'all'})"
        )
        print(f"Loading {UPSTREAM_DATASET_NAME}...")

    load_kwargs = {}
    if hf_cache_dir:
        load_kwargs["cache_dir"] = hf_cache_dir
    hf_ds = load_dataset(UPSTREAM_DATASET_NAME, **load_kwargs)

    files_by_split: dict[str, list[tuple[int, Path]]] = {}
    for krn_path in filtered_krn_files:
        split, idx = parse_filename(krn_path.name)
        files_by_split.setdefault(split, []).append((idx, krn_path))

    samples: list[dict] = []
    skipped = 0
    for split, entries in sorted(files_by_split.items()):
        if split not in hf_ds:
            if not quiet:
                print(f"Warning: split '{split}' not found upstream, skipping {len(entries)} files")
            skipped += len(entries)
            continue

        split_data = hf_ds[split]
        for idx, krn_path in entries:
            if idx >= len(split_data):
                if not quiet:
                    print(
                        f"Warning: index {idx} out of range for split '{split}' "
                        f"(size {len(split_data)}), skipping"
                    )
                skipped += 1
                continue

            image = split_data[idx]["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image)
            image = image.convert("RGB")
            image = resize_image(image, image_width, image_height)
            if image.size != (image_width, image_height):
                raise ValueError(
                    f"Unexpected resized image size {image.size}; expected "
                    f"{(image_width, image_height)}"
                )

            samples.append(
                {
                    "image": image,
                    "transcription": krn_path.read_text(encoding="utf-8"),
                    "source": krn_path.name,
                    "source_dataset": UPSTREAM_DATASET_NAME,
                    "source_split": split,
                    "sample_id": krn_path.stem,
                    "curation_stage": "manual_fix",
                    "source_domain": "real",
                }
            )

    if not samples:
        raise ValueError(
            f"No samples assembled from {normalized_path}; check split filters and upstream mappings"
        )

    features = Features(
        {
            "image": datasets.Image(mode="RGB"),
            "transcription": Value("string"),
            "source": Value("string"),
            "source_dataset": Value("string"),
            "source_split": Value("string"),
            "sample_id": Value("string"),
            "curation_stage": Value("string"),
            "source_domain": Value("string"),
        }
    )

    dataset = Dataset.from_list(samples, features=features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    result = {
        "output_dir": str(output_path),
        "num_rows": len(dataset),
        "skipped": skipped,
        "selected_splits": selected_splits or sorted(files_by_split.keys()),
    }
    if not quiet:
        print(f"Saved {len(dataset)} samples to {output_path} ({skipped} skipped)")
    return result


def main() -> None:
    fire.Fire(assemble_polish_scores_dataset)


if __name__ == "__main__":
    main()
