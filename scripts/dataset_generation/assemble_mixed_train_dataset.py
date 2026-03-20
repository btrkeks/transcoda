#!/usr/bin/env python3
"""Build a deterministic mixed finetune dataset from synth and real HF datasets."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import fire
from datasets import concatenate_datasets, load_from_disk


def _set_column(dataset, name: str, values: list[str]):
    if name in dataset.column_names:
        dataset = dataset.remove_columns(name)
    return dataset.add_column(name, values)


def _derive_stem(value: str, fallback: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        return fallback
    return Path(cleaned).stem


def _normalize_real_dataset_metadata(dataset):
    row_count = len(dataset)
    column_values = {name: dataset[name] for name in dataset.column_names}

    sample_ids = [
        _derive_stem(
            column_values.get("sample_id", [""] * row_count)[idx]
            if "sample_id" in column_values
            else column_values.get("source", [""] * row_count)[idx],
            fallback=f"real_{idx:06d}",
        )
        for idx in range(row_count)
    ]
    source_values = [
        str(column_values.get("source", [f"{sample_ids[idx]}.krn"] * row_count)[idx])
        if "source" in column_values
        else f"{sample_ids[idx]}.krn"
        for idx in range(row_count)
    ]
    source_split_values = [
        str(column_values.get("source_split", ["train"] * row_count)[idx])
        if "source_split" in column_values
        else "train"
        for idx in range(row_count)
    ]
    source_dataset_values = [
        str(column_values.get("source_dataset", ["PRAIG/polish-scores"] * row_count)[idx])
        if "source_dataset" in column_values
        else "PRAIG/polish-scores"
        for idx in range(row_count)
    ]
    curation_values = [
        str(column_values.get("curation_stage", ["manual_fix"] * row_count)[idx])
        if "curation_stage" in column_values
        else "manual_fix"
        for idx in range(row_count)
    ]
    domain_values = [
        str(column_values.get("source_domain", ["real"] * row_count)[idx])
        if "source_domain" in column_values
        else "real"
        for idx in range(row_count)
    ]

    dataset = _set_column(dataset, "source", source_values)
    dataset = _set_column(dataset, "source_dataset", source_dataset_values)
    dataset = _set_column(dataset, "source_split", source_split_values)
    dataset = _set_column(dataset, "sample_id", sample_ids)
    dataset = _set_column(dataset, "curation_stage", curation_values)
    dataset = _set_column(dataset, "source_domain", domain_values)
    return dataset


def _prepare_synth_subset(dataset, synth_indices: list[int]):
    subset = dataset.select(synth_indices)
    row_count = len(subset)
    source_values = [f"train_full_{idx:06d}" for idx in synth_indices]
    sample_ids = list(source_values)

    subset = _set_column(subset, "source", source_values)
    subset = _set_column(subset, "source_dataset", ["train_full"] * row_count)
    subset = _set_column(subset, "source_split", ["train"] * row_count)
    subset = _set_column(subset, "sample_id", sample_ids)
    subset = _set_column(subset, "curation_stage", ["synthetic"] * row_count)
    subset = _set_column(subset, "source_domain", ["synth"] * row_count)
    return subset


def assemble_mixed_train_dataset(
    synth_dataset_path: str = "data/datasets/train_full",
    real_dataset_path: str = "data/datasets/train_polish_adapt",
    output_dir: str = "data/datasets/train_polish_adapt_mixed",
    target_synth_count: int = 1000,
    target_real_count: int = 996,
    seed: int = 42,
    metadata_out: str | None = "reports/dataset_generation/polish_adapt/latest.json",
    shuffle_output: bool = True,
    quiet: bool = False,
) -> dict[str, object]:
    """Assemble a deterministic mixed synth+real HF dataset."""
    if target_synth_count < 1:
        raise ValueError("target_synth_count must be >= 1")
    if target_real_count < 1:
        raise ValueError("target_real_count must be >= 1")

    synth_path = Path(synth_dataset_path)
    real_path = Path(real_dataset_path)
    output_path = Path(output_dir)

    synth_ds = load_from_disk(str(synth_path))
    real_ds = _normalize_real_dataset_metadata(load_from_disk(str(real_path)))

    if len(synth_ds) < target_synth_count:
        raise ValueError(
            f"synth dataset only has {len(synth_ds)} rows, "
            f"cannot sample {target_synth_count}"
        )
    if len(real_ds) == 0:
        raise ValueError("real dataset is empty")

    rng = random.Random(seed)
    synth_indices = rng.sample(range(len(synth_ds)), target_synth_count)
    synth_subset = _prepare_synth_subset(synth_ds, synth_indices)

    full_repeats, remainder = divmod(target_real_count, len(real_ds))
    real_indices = list(range(len(real_ds))) * full_repeats + list(range(remainder))
    real_subset = real_ds.select(real_indices)

    mixed_ds = concatenate_datasets([synth_subset, real_subset])
    if shuffle_output:
        mixed_ds = mixed_ds.shuffle(seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixed_ds.save_to_disk(str(output_path))

    repeat_histogram = Counter(real_indices)
    metadata = {
        "output_dir": str(output_path),
        "seed": seed,
        "shuffle_output": shuffle_output,
        "synth_dataset_path": str(synth_path),
        "real_dataset_path": str(real_path),
        "target_synth_count": target_synth_count,
        "target_real_count": target_real_count,
        "total_rows": len(mixed_ds),
        "synth_rows": len(synth_subset),
        "real_rows": len(real_subset),
        "synth_sample_indices": synth_indices,
        "real_repeat_histogram": {
            str(real_ds[idx]["sample_id"]): repeat_histogram[idx] for idx in range(len(real_ds))
        },
    }
    if metadata_out:
        metadata_path = Path(metadata_out)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    if not quiet:
        print(
            f"Saved mixed dataset with {len(mixed_ds)} rows to {output_path} "
            f"({len(synth_subset)} synth + {len(real_subset)} real)"
        )
    return metadata


def main() -> None:
    fire.Fire(assemble_mixed_train_dataset)


if __name__ == "__main__":
    main()
