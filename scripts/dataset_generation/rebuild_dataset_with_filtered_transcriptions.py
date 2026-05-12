#!/usr/bin/env python3
"""Rebuild a HF dataset by replacing transcriptions from sibling 2_filtered files."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
from datasets import load_from_disk


@dataclass(frozen=True)
class FilteredCandidate:
    """A distinct filtered-text candidate for one normalized transcription."""

    filtered_text: str
    representative_filtered_source: str
    representative_normalized_source: str
    normalized_sources: tuple[str, ...]


@dataclass(frozen=True)
class RowResolution:
    """Resolved replacement data for one dataset row."""

    status: str
    replacement_transcription: str | None
    chosen_normalized_source: str | None
    chosen_filtered_source: str | None
    candidate_normalized_sources: tuple[str, ...]
    candidate_filtered_representatives: tuple[str, ...]
    distinct_filtered_candidate_count: int


def _parse_source_datasets(
    source_datasets: str | Iterable[str] | None,
) -> list[str]:
    if source_datasets is None:
        raise ValueError("source_datasets must not be None")

    if isinstance(source_datasets, str):
        raw_values = source_datasets.split(",")
    else:
        raw_values = list(source_datasets)

    values = [str(value).strip() for value in raw_values if str(value).strip()]
    if not values:
        raise ValueError("source_datasets must include at least one dataset name")
    return list(dict.fromkeys(values))


def _default_row_audit_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}.rows.jsonl")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_row_rng(seed: int, row_index: int, transcription: str) -> random.Random:
    digest = hashlib.sha256(
        f"{seed}:{row_index}:{transcription}".encode("utf-8")
    ).digest()
    rng_seed = int.from_bytes(digest[:8], byteorder="big")
    return random.Random(rng_seed)


def _resolve_direct_candidate(
    *,
    interim_root: Path,
    normalized_stage: str,
    filtered_stage: str,
    source_dataset: str,
    source: str,
) -> FilteredCandidate | None:
    cleaned_dataset = str(source_dataset).strip()
    cleaned_source = str(source).strip()
    if not cleaned_dataset or not cleaned_source:
        return None

    normalized_path = interim_root / cleaned_dataset / normalized_stage / cleaned_source
    filtered_path = interim_root / cleaned_dataset / filtered_stage / cleaned_source
    if not filtered_path.exists():
        return None

    filtered_text = filtered_path.read_text(encoding="utf-8")
    normalized_source = (
        str(normalized_path) if normalized_path.exists() else str(normalized_path)
    )
    return FilteredCandidate(
        filtered_text=filtered_text,
        representative_filtered_source=str(filtered_path),
        representative_normalized_source=normalized_source,
        normalized_sources=(normalized_source,),
    )


def _build_transcription_index(
    *,
    interim_root: Path,
    source_datasets: list[str],
    normalized_stage: str,
    filtered_stage: str,
) -> tuple[dict[str, tuple[FilteredCandidate, ...]], dict[str, int]]:
    grouped: dict[str, dict[str, dict[str, list[str] | str]]] = defaultdict(dict)
    scanned = 0
    paired = 0
    ignored = 0

    for dataset_name in source_datasets:
        normalized_dir = interim_root / dataset_name / normalized_stage
        filtered_dir = interim_root / dataset_name / filtered_stage
        if not normalized_dir.is_dir():
            raise ValueError(f"Normalized directory not found: {normalized_dir}")

        for normalized_path in sorted(normalized_dir.glob("*.krn")):
            scanned += 1
            filtered_path = filtered_dir / normalized_path.name
            if not filtered_path.exists():
                ignored += 1
                continue

            normalized_text = normalized_path.read_text(encoding="utf-8")
            filtered_text = filtered_path.read_text(encoding="utf-8")
            paired += 1

            text_bucket = grouped[normalized_text]
            filtered_bucket = text_bucket.get(filtered_text)
            if filtered_bucket is None:
                filtered_bucket = {
                    "normalized_sources": [],
                    "filtered_sources": [],
                }
                text_bucket[filtered_text] = filtered_bucket
            filtered_bucket["normalized_sources"].append(str(normalized_path))
            filtered_bucket["filtered_sources"].append(str(filtered_path))

    index: dict[str, tuple[FilteredCandidate, ...]] = {}
    for normalized_text, filtered_groups in grouped.items():
        candidates: list[FilteredCandidate] = []
        for filtered_text, payload in filtered_groups.items():
            normalized_sources = tuple(sorted(payload["normalized_sources"]))
            filtered_sources = tuple(sorted(payload["filtered_sources"]))
            candidates.append(
                FilteredCandidate(
                    filtered_text=filtered_text,
                    representative_filtered_source=filtered_sources[0],
                    representative_normalized_source=normalized_sources[0],
                    normalized_sources=normalized_sources,
                )
            )
        candidates.sort(key=lambda item: item.representative_filtered_source)
        index[normalized_text] = tuple(candidates)

    return index, {
        "scanned_source_files": scanned,
        "paired_source_files": paired,
        "ignored_source_files": ignored,
    }


def _resolve_from_candidates(
    *,
    candidates: tuple[FilteredCandidate, ...],
    row_index: int,
    transcription: str,
    seed: int,
) -> RowResolution:
    candidate_normalized_sources = tuple(
        sorted(
            source
            for candidate in candidates
            for source in candidate.normalized_sources
        )
    )
    candidate_filtered_representatives = tuple(
        candidate.representative_filtered_source for candidate in candidates
    )

    if len(candidates) == 1:
        candidate = candidates[0]
        return RowResolution(
            status="unique_match",
            replacement_transcription=candidate.filtered_text,
            chosen_normalized_source=candidate.representative_normalized_source,
            chosen_filtered_source=candidate.representative_filtered_source,
            candidate_normalized_sources=candidate_normalized_sources,
            candidate_filtered_representatives=candidate_filtered_representatives,
            distinct_filtered_candidate_count=1,
        )

    rng = _build_row_rng(seed=seed, row_index=row_index, transcription=transcription)
    candidate = rng.choice(list(candidates))
    return RowResolution(
        status="ambiguous_match",
        replacement_transcription=candidate.filtered_text,
        chosen_normalized_source=candidate.representative_normalized_source,
        chosen_filtered_source=candidate.representative_filtered_source,
        candidate_normalized_sources=candidate_normalized_sources,
        candidate_filtered_representatives=candidate_filtered_representatives,
        distinct_filtered_candidate_count=len(candidates),
    )


def _resolve_row(
    *,
    row_index: int,
    transcription: str,
    seed: int,
    interim_root: Path,
    normalized_stage: str,
    filtered_stage: str,
    source_dataset: str | None,
    source: str | None,
    transcription_index: dict[str, tuple[FilteredCandidate, ...]] | None,
) -> RowResolution:
    if source_dataset is not None and source is not None:
        direct_candidate = _resolve_direct_candidate(
            interim_root=interim_root,
            normalized_stage=normalized_stage,
            filtered_stage=filtered_stage,
            source_dataset=source_dataset,
            source=source,
        )
        if direct_candidate is not None:
            return _resolve_from_candidates(
                candidates=(direct_candidate,),
                row_index=row_index,
                transcription=transcription,
                seed=seed,
            )

    if transcription_index is None:
        return RowResolution(
            status="missing_match",
            replacement_transcription=None,
            chosen_normalized_source=None,
            chosen_filtered_source=None,
            candidate_normalized_sources=(),
            candidate_filtered_representatives=(),
            distinct_filtered_candidate_count=0,
        )

    candidates = transcription_index.get(transcription)
    if not candidates:
        return RowResolution(
            status="missing_match",
            replacement_transcription=None,
            chosen_normalized_source=None,
            chosen_filtered_source=None,
            candidate_normalized_sources=(),
            candidate_filtered_representatives=(),
            distinct_filtered_candidate_count=0,
        )

    return _resolve_from_candidates(
        candidates=candidates,
        row_index=row_index,
        transcription=transcription,
        seed=seed,
    )


def rebuild_dataset_with_filtered_transcriptions(
    dataset_path: str = "data/datasets/train_full",
    interim_root: str = "data/interim/train",
    source_datasets: str | list[str] | tuple[str, ...] = "grandstaff,musetrainer,pdmx",
    normalized_stage: str = "3_normalized",
    filtered_stage: str = "2_filtered",
    output_dir: str = "data/datasets/train_full_filtered_ablation",
    manifest_out: str = "reports/dataset_generation/train_full_filtered_ablation/latest.json",
    row_audit_out: str | None = None,
    seed: int = 42,
    quiet: bool = False,
) -> dict[str, Any]:
    """Rebuild a dataset using sibling 2_filtered transcriptions."""
    dataset_dir = Path(dataset_path)
    interim_dir = Path(interim_root)
    output_path = Path(output_dir)
    manifest_path = Path(manifest_out)
    row_audit_path = Path(row_audit_out) if row_audit_out else _default_row_audit_path(manifest_path)
    parsed_source_datasets = _parse_source_datasets(source_datasets)

    dataset = load_from_disk(str(dataset_dir))
    if "transcription" not in dataset.column_names:
        raise ValueError(f"Dataset at {dataset_dir} is missing a transcription column")

    original_column_names = list(dataset.column_names)
    transcriptions = dataset["transcription"]
    source_values = dataset["source"] if "source" in dataset.column_names else None
    source_dataset_values = (
        dataset["source_dataset"] if "source_dataset" in dataset.column_names else None
    )

    transcription_index: dict[str, tuple[FilteredCandidate, ...]] | None = None
    scan_stats = {
        "scanned_source_files": 0,
        "paired_source_files": 0,
        "ignored_source_files": 0,
    }

    def ensure_index() -> dict[str, tuple[FilteredCandidate, ...]]:
        nonlocal transcription_index, scan_stats
        if transcription_index is None:
            transcription_index, scan_stats = _build_transcription_index(
                interim_root=interim_dir,
                source_datasets=parsed_source_datasets,
                normalized_stage=normalized_stage,
                filtered_stage=filtered_stage,
            )
        return transcription_index

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    row_audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept_indices: list[int] = []
    replacement_transcriptions: list[str] = []
    status_counts = {
        "unique_match": 0,
        "ambiguous_match": 0,
        "missing_match": 0,
    }

    if not quiet:
        print(f"Loading dataset from {dataset_dir}")
        print(f"Writing rebuilt dataset to {output_path}")

    output_row_index = 0
    with row_audit_path.open("w", encoding="utf-8") as audit_handle:
        for row_index, transcription in enumerate(transcriptions):
            resolution = _resolve_row(
                row_index=row_index,
                transcription=transcription,
                seed=seed,
                interim_root=interim_dir,
                normalized_stage=normalized_stage,
                filtered_stage=filtered_stage,
                source_dataset=(
                    str(source_dataset_values[row_index]) if source_dataset_values is not None else None
                ),
                source=str(source_values[row_index]) if source_values is not None else None,
                transcription_index=(
                    transcription_index
                    if transcription_index is not None
                    else (None if (source_values is not None and source_dataset_values is not None) else ensure_index())
                ),
            )

            if (
                resolution.status == "missing_match"
                and source_values is not None
                and source_dataset_values is not None
            ):
                resolution = _resolve_row(
                    row_index=row_index,
                    transcription=transcription,
                    seed=seed,
                    interim_root=interim_dir,
                    normalized_stage=normalized_stage,
                    filtered_stage=filtered_stage,
                    source_dataset=None,
                    source=None,
                    transcription_index=ensure_index(),
                )

            status_counts[resolution.status] += 1

            audit_payload = {
                "original_row_index": row_index,
                "output_row_index": output_row_index if resolution.replacement_transcription is not None else None,
                "status": resolution.status,
                "chosen_normalized_source": resolution.chosen_normalized_source,
                "chosen_filtered_source": resolution.chosen_filtered_source,
                "candidate_normalized_sources": list(resolution.candidate_normalized_sources),
                "candidate_filtered_representatives": list(
                    resolution.candidate_filtered_representatives
                ),
                "distinct_filtered_candidate_count": resolution.distinct_filtered_candidate_count,
                "input_transcription_sha256": _sha256_text(transcription),
            }

            if resolution.replacement_transcription is not None:
                kept_indices.append(row_index)
                replacement_transcriptions.append(resolution.replacement_transcription)
                audit_payload["chosen_filtered_transcription_sha256"] = _sha256_text(
                    resolution.replacement_transcription
                )
                output_row_index += 1

            audit_handle.write(json.dumps(audit_payload, sort_keys=True) + "\n")

    rebuilt_dataset = dataset.select(kept_indices)
    rebuilt_dataset = rebuilt_dataset.remove_columns(["transcription"])
    rebuilt_dataset = rebuilt_dataset.add_column("transcription", replacement_transcriptions)
    if hasattr(rebuilt_dataset, "select_columns"):
        rebuilt_dataset = rebuilt_dataset.select_columns(original_column_names)
    rebuilt_dataset.save_to_disk(str(output_path))

    summary = {
        "dataset_path": str(dataset_dir),
        "output_dir": str(output_path),
        "row_audit_out": str(row_audit_path),
        "source_datasets": parsed_source_datasets,
        "normalized_stage": normalized_stage,
        "filtered_stage": filtered_stage,
        "seed": int(seed),
        "input_row_count": len(dataset),
        "output_row_count": len(rebuilt_dataset),
        "skipped_row_count": status_counts["missing_match"],
        "ambiguous_row_count": status_counts["ambiguous_match"],
        "unique_match_row_count": status_counts["unique_match"],
        **scan_stats,
    }
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if not quiet:
        print(
            f"Saved {len(rebuilt_dataset)} rows to {output_path} "
            f"({status_counts['missing_match']} skipped, {status_counts['ambiguous_match']} ambiguous)"
        )

    return summary


def main() -> None:
    fire.Fire(rebuild_dataset_with_filtered_transcriptions)


if __name__ == "__main__":
    main()
