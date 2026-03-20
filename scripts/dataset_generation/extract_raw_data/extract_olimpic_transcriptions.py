#!/usr/bin/env python3
"""Extract OlimpIC XML/image pairs into split-specific interim directories.

This script consumes a split manifest and copies source XML/image files from
`data/raw/olimpic-1.0-scanned` into the project interim layout:

    data/interim/{split}/olimpic/
      0_raw_xml/
      0_raw_images/
      metadata/manifest.jsonl

The manifest must include at least:
- sample_id
- doc_id
- split
- xml_path (or xml_relpath)
- image_path (or image_relpath)

Paths in the manifest are interpreted as relative to --raw_root unless already
absolute. The script validates that no document id appears in multiple splits.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import fire

OUTPUT_DIR = Path("data/interim")
RAW_ROOT = Path("data/raw/olimpic-1.0-scanned")


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    doc_id: str
    split: str
    xml_path: Path
    image_path: Path


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    aliases = {
        "validation": "val",
        "valid": "val",
        "dev": "val",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split '{split}'. Expected one of train/val/test.")
    return normalized


def _sanitize_sample_id(sample_id: str) -> str:
    value = str(sample_id).strip()
    if not value:
        raise ValueError("sample_id cannot be empty")
    # Keep filenames shell- and path-safe while preserving readability.
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _load_manifest_rows(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Manifest line {line_no} is not an object")
            rows.append(payload)
        return rows

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            samples = payload.get("samples")
            if not isinstance(samples, list):
                raise ValueError("JSON manifest object must include a 'samples' list")
            return [dict(item) for item in samples]
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        raise ValueError("JSON manifest must be a list or an object with 'samples'")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    raise ValueError(
        f"Unsupported manifest extension '{path.suffix}'. Use .jsonl, .json, or .csv"
    )


def _resolve_path(path_value: str, raw_root: Path) -> Path:
    candidate = Path(path_value)
    resolved = candidate if candidate.is_absolute() else (raw_root / candidate)
    resolved = resolved.resolve()
    if not resolved.exists():
        raise ValueError(f"Manifest path does not exist: {resolved}")
    return resolved


def _require_field(row: dict[str, object], primary: str, *aliases: str) -> str:
    for key in (primary, *aliases):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    joined = ", ".join((primary, *aliases))
    raise ValueError(f"Missing required manifest field. Expected one of: {joined}")


def load_split_manifest(
    split_manifest: str,
    raw_root: str = str(RAW_ROOT),
) -> list[ManifestRecord]:
    """Load and validate split manifest records."""
    manifest_path = Path(split_manifest)
    if not manifest_path.exists():
        raise ValueError(f"Split manifest not found: {manifest_path}")

    resolved_raw_root = Path(raw_root).resolve()
    if not resolved_raw_root.exists():
        raise ValueError(f"raw_root not found: {resolved_raw_root}")

    rows = _load_manifest_rows(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    records: list[ManifestRecord] = []
    seen_sample_ids: set[str] = set()
    doc_ids_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for idx, row in enumerate(rows, start=1):
        sample_id = _sanitize_sample_id(_require_field(row, "sample_id", "id"))
        doc_id = _require_field(row, "doc_id", "document_id")
        split = _normalize_split(_require_field(row, "split"))
        xml_raw = _require_field(row, "xml_path", "xml_relpath")
        image_raw = _require_field(row, "image_path", "image_relpath")

        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id in manifest at row {idx}: {sample_id}")
        seen_sample_ids.add(sample_id)

        xml_path = _resolve_path(xml_raw, resolved_raw_root)
        image_path = _resolve_path(image_raw, resolved_raw_root)

        records.append(
            ManifestRecord(
                sample_id=sample_id,
                doc_id=doc_id,
                split=split,
                xml_path=xml_path,
                image_path=image_path,
            )
        )
        doc_ids_by_split[split].add(doc_id)

    all_splits = sorted(doc_ids_by_split.keys())
    for i, split_a in enumerate(all_splits):
        for split_b in all_splits[i + 1 :]:
            overlap = doc_ids_by_split[split_a] & doc_ids_by_split[split_b]
            if overlap:
                preview = ", ".join(sorted(list(overlap))[:5])
                raise ValueError(
                    "doc_id overlap across splits detected "
                    f"({split_a} vs {split_b}); examples: {preview}"
                )

    records.sort(key=lambda item: (item.split, item.sample_id))
    return records


def extract_olimpic_transcriptions(
    split_manifest: str,
    raw_root: str = str(RAW_ROOT),
    output_dir: str = str(OUTPUT_DIR),
    copy_mode: str = "copy",
    quiet: bool = False,
) -> None:
    """Copy OlimpIC XML/image pairs into interim split directories."""
    mode = str(copy_mode).strip().lower()
    if mode != "copy":
        raise ValueError("copy_mode currently supports only 'copy'")

    records = load_split_manifest(split_manifest=split_manifest, raw_root=raw_root)
    output_path = Path(output_dir)

    split_manifests: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}

    for record in records:
        split_root = output_path / record.split / "olimpic"
        raw_xml_dir = split_root / "0_raw_xml"
        raw_img_dir = split_root / "0_raw_images"
        metadata_dir = split_root / "metadata"

        raw_xml_dir.mkdir(parents=True, exist_ok=True)
        raw_img_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        xml_dst = raw_xml_dir / f"{record.sample_id}.xml"
        img_suffix = record.image_path.suffix or ".png"
        image_dst = raw_img_dir / f"{record.sample_id}{img_suffix.lower()}"

        shutil.copy2(record.xml_path, xml_dst)
        shutil.copy2(record.image_path, image_dst)

        split_manifests[record.split].append(
            {
                "sample_id": record.sample_id,
                "doc_id": record.doc_id,
                "split": record.split,
                "xml_path": str(xml_dst.resolve()),
                "image_path": str(image_dst.resolve()),
                "xml_relpath": str(xml_dst.relative_to(split_root)),
                "image_relpath": str(image_dst.relative_to(split_root)),
            }
        )

    total_written = 0
    for split, rows in split_manifests.items():
        if not rows:
            continue
        split_root = output_path / split / "olimpic"
        manifest_out = split_root / "metadata" / "manifest.jsonl"
        with manifest_out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        total_written += len(rows)
        if not quiet:
            print(
                f"{split}: wrote {len(rows)} samples -> {manifest_out}",
                file=sys.stderr,
            )

    if not quiet:
        print(f"Done. Extracted {total_written} OlimpIC samples.", file=sys.stderr)


def main() -> None:
    fire.Fire(extract_olimpic_transcriptions)


if __name__ == "__main__":
    main()
