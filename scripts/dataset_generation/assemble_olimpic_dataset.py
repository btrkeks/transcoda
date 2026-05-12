#!/usr/bin/env python3
"""Assemble an OlimpIC real-image dataset from normalized kern + manifest mapping."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import datasets
import fire
from datasets import Dataset, Features, Value
from PIL import Image

from scripts.dataset_generation.data_spec import (
    DEFAULT_DATA_SPEC_PATH,
    resolve_image_size_from_spec,
)


def _load_manifest_rows(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"manifest line {line_no} is not an object")
            rows.append({str(k): str(v) for k, v in payload.items() if v is not None})
        return rows

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            samples = payload.get("samples")
            if not isinstance(samples, list):
                raise ValueError("JSON manifest object must include a 'samples' list")
            return [{str(k): str(v) for k, v in item.items() if v is not None} for item in samples]
        if isinstance(payload, list):
            return [{str(k): str(v) for k, v in item.items() if v is not None} for item in payload]
        raise ValueError("JSON manifest must be a list or an object with 'samples'")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [{str(k): str(v) for k, v in row.items() if v is not None} for row in csv.DictReader(handle)]

    raise ValueError(
        f"Unsupported manifest extension '{path.suffix}'. Use .jsonl, .json, or .csv"
    )


def _resolve_image_path(row: dict[str, str], manifest_path: Path) -> Path:
    image_path = row.get("image_path")
    if image_path:
        resolved = Path(image_path)
        if not resolved.is_absolute():
            resolved = (manifest_path.parent / resolved).resolve()
        if resolved.exists():
            return resolved

    image_relpath = row.get("image_relpath")
    if image_relpath:
        split_root = manifest_path.parent.parent
        resolved = (split_root / image_relpath).resolve()
        if resolved.exists():
            return resolved

    sample_id = row.get("sample_id", "<missing sample_id>")
    raise ValueError(f"Could not resolve image path for sample_id={sample_id}")


def _load_manifest_index(manifest_path: Path) -> dict[str, Path]:
    rows = _load_manifest_rows(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    index: dict[str, Path] = {}
    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            raise ValueError("Manifest row missing sample_id")
        if sample_id in index:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        index[sample_id] = _resolve_image_path(row, manifest_path)
    return index


def _resize_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    return image.resize((target_width, target_height), Image.LANCZOS)


def assemble_olimpic_dataset(
    normalized_dir: str,
    manifest_path: str,
    output_dir: str,
    source_name: str = "olimpic",
    image_width: int | None = None,
    image_height: int | None = None,
    data_spec_path: str = str(DEFAULT_DATA_SPEC_PATH),
    strict_data_spec: bool = True,
    quiet: bool = False,
):
    """Assemble an HF dataset from normalized transcriptions + source images."""
    final_width, final_height = resolve_image_size_from_spec(
        image_width=image_width,
        image_height=image_height,
        data_spec_path=data_spec_path,
        strict_data_spec=strict_data_spec,
    )

    normalized_path = Path(normalized_dir)
    if not normalized_path.exists():
        raise ValueError(f"normalized_dir does not exist: {normalized_path}")

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise ValueError(f"manifest_path does not exist: {manifest_file}")

    index = _load_manifest_index(manifest_file)

    krn_files = sorted(normalized_path.glob("*.krn"))
    if not krn_files:
        raise ValueError(f"No .krn files found in {normalized_path}")

    samples: list[dict] = []
    missing_mapping = 0
    for krn_path in krn_files:
        sample_id = krn_path.stem
        image_path = index.get(sample_id)
        if image_path is None:
            missing_mapping += 1
            if not quiet:
                print(
                    f"Warning: sample_id={sample_id} missing from manifest mapping, skipping",
                    file=sys.stderr,
                )
            continue

        transcription = krn_path.read_text(encoding="utf-8")
        image = Image.open(image_path).convert("RGB")
        image = _resize_image(image, final_width, final_height)

        if image.size != (final_width, final_height):
            raise ValueError(
                f"Unexpected resized image size {image.size}; expected {(final_width, final_height)}"
            )

        samples.append(
            {
                "image": image,
                "transcription": transcription,
                "source": source_name,
            }
        )

    if not samples:
        raise ValueError(
            f"No samples assembled from normalized_dir={normalized_path}; check manifest/sample_id alignment"
        )

    features = Features(
        {
            "image": datasets.Image(mode="RGB"),
            "transcription": Value("string"),
            "source": Value("string"),
        }
    )

    dataset = Dataset.from_list(samples, features=features)
    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    if not quiet:
        print(
            f"Saved {len(dataset)} samples to {output_path}"
            f" (missing mappings skipped: {missing_mapping})",
            file=sys.stderr,
        )


def main() -> None:
    fire.Fire(assemble_olimpic_dataset)


if __name__ == "__main__":
    main()
