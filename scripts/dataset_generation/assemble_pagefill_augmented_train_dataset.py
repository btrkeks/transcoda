#!/usr/bin/env python3
"""Build a train_full_2 dataset augmented with curated page-filling samples."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import fire
import numpy as np
from datasets import Features, Image, Value, concatenate_datasets, load_from_disk
from tokenizers import Tokenizer


FINAL_FEATURES = Features(
    {
        "image": Image(mode="RGB"),
        "transcription": Value("string"),
        "source": Value("string"),
        "source_dataset": Value("string"),
        "source_split": Value("string"),
        "sample_id": Value("string"),
        "curation_stage": Value("string"),
        "source_domain": Value("string"),
        "mix_component": Value("string"),
        "original_dataset_path": Value("string"),
        "original_index": Value("int64"),
        "token_length": Value("int32"),
        "bottom_whitespace_ratio": Value("float32"),
        "vertical_fill_ratio": Value("float32"),
        "bottom_whitespace_px": Value("int32"),
        "content_height_px": Value("int32"),
    }
)

FINAL_COLUMNS = list(FINAL_FEATURES)
LAYOUT_COLUMNS = (
    "bottom_whitespace_ratio",
    "vertical_fill_ratio",
    "bottom_whitespace_px",
    "content_height_px",
)


def _hash_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=16).hexdigest()


def _derive_source_group(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        value = value[0] if value else "unknown"
    text = str(value)
    lowered = text.lower()
    for marker in ("grandstaff", "musetrainer", "openscore", "pdmx", "polish", "olimpic"):
        if marker in lowered:
            return marker
    return text[:120] or "unknown"


def _first_source_id(row: dict[str, Any], fallback: str) -> str:
    if "source" in row and row["source"]:
        return str(row["source"])
    source_ids = row.get("source_ids")
    if source_ids:
        return str(source_ids[0])
    return fallback


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _percentile(values: list[float] | list[int], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def _line_stats(text: str) -> tuple[int, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    max_spines = max((line.count("\t") + 1 for line in lines), default=0)
    first_line = lines[0][:120] if lines else ""
    return max_spines, first_line


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    token_lengths = [int(row["token_length"]) for row in rows]
    bottom = [
        float(row["bottom_whitespace_ratio"])
        for row in rows
        if row.get("bottom_whitespace_ratio") is not None
    ]
    fill = [
        float(row["vertical_fill_ratio"])
        for row in rows
        if row.get("vertical_fill_ratio") is not None
    ]
    max_spines: list[int] = []
    first_lines: Counter[str] = Counter()
    for row in rows:
        spines, first_line = _line_stats(str(row["transcription"]))
        max_spines.append(spines)
        first_lines[first_line] += 1
    return {
        "rows": len(rows),
        "source_dataset_counts": dict(Counter(str(row["source_dataset"]) for row in rows).most_common()),
        "mix_component_counts": dict(Counter(str(row["mix_component"]) for row in rows).most_common()),
        "token_length": {
            "mean": float(mean(token_lengths)),
            "p50": _percentile(token_lengths, 50),
            "p90": _percentile(token_lengths, 90),
            "p95": _percentile(token_lengths, 95),
            "p99": _percentile(token_lengths, 99),
            "max": max(token_lengths),
        },
        "bottom_whitespace_ratio": {
            "mean": float(mean(bottom)) if bottom else None,
            "p50": _percentile(bottom, 50),
            "p90": _percentile(bottom, 90),
            "p95": _percentile(bottom, 95),
        },
        "vertical_fill_ratio": {
            "mean": float(mean(fill)) if fill else None,
            "p50": _percentile(fill, 50),
            "p90": _percentile(fill, 90),
            "p95": _percentile(fill, 95),
        },
        "max_spines": {
            "mean": float(mean(max_spines)),
            "p50": _percentile(max_spines, 50),
            "p90": _percentile(max_spines, 90),
            "p95": _percentile(max_spines, 95),
            "p99": _percentile(max_spines, 99),
            "max": max(max_spines),
        },
        "top_first_lines": dict(first_lines.most_common(20)),
    }


def _token_lengths(dataset, tokenizer: Tokenizer) -> list[int]:
    return [
        len(tokenizer.encode(text or "", add_special_tokens=True).ids)
        for text in dataset["transcription"]
    ]


def _base_row(dataset, idx: int, token_length: int, base_dataset_path: str) -> dict[str, Any]:
    row = dataset[idx]
    sample_id = str(row.get("sample_id") or f"base_{idx:06d}")
    source = str(row.get("source") or f"{sample_id}.krn")
    source_dataset = str(row.get("source_dataset") or _derive_source_group(source))
    return {
        "transcription": row["transcription"],
        "source": source,
        "source_dataset": source_dataset,
        "source_split": str(row.get("source_split") or "train"),
        "sample_id": sample_id,
        "curation_stage": str(row.get("curation_stage") or "synthetic"),
        "source_domain": str(row.get("source_domain") or "synth"),
        "mix_component": "base",
        "original_dataset_path": base_dataset_path,
        "original_index": idx,
        "token_length": int(token_length),
        "bottom_whitespace_ratio": _coerce_float(row.get("bottom_whitespace_ratio")),
        "vertical_fill_ratio": _coerce_float(row.get("vertical_fill_ratio")),
        "bottom_whitespace_px": _coerce_int(row.get("bottom_whitespace_px")),
        "content_height_px": _coerce_int(row.get("content_height_px")),
    }


def _candidate_row(dataset, idx: int, token_length: int, candidate_dataset_path: str) -> dict[str, Any]:
    row = dataset[idx]
    sample_id = str(row.get("sample_id") or f"pagefill_{idx:06d}")
    source = _first_source_id(row, fallback=f"{sample_id}.krn")
    return {
        "transcription": row["transcription"],
        "source": source,
        "source_dataset": _derive_source_group(row.get("source_ids") or source),
        "source_split": str(row.get("source_split") or "train"),
        "sample_id": f"pagefill_{sample_id}",
        "curation_stage": str(row.get("curation_stage") or "synthetic"),
        "source_domain": str(row.get("source_domain") or "synth"),
        "mix_component": "pagefill",
        "original_dataset_path": candidate_dataset_path,
        "original_index": idx,
        "token_length": int(token_length),
        "bottom_whitespace_ratio": _coerce_float(row.get("bottom_whitespace_ratio")),
        "vertical_fill_ratio": _coerce_float(row.get("vertical_fill_ratio")),
        "bottom_whitespace_px": _coerce_int(row.get("bottom_whitespace_px")),
        "content_height_px": _coerce_int(row.get("content_height_px")),
    }


def _normalize_selected_dataset(dataset, indices: list[int], rows: list[dict[str, Any]]):
    selected = dataset.select(indices)
    selected = selected.select_columns(["image", "transcription"])
    for column in FINAL_COLUMNS:
        if column in ("image", "transcription"):
            continue
        selected = selected.add_column(column, [row[column] for row in rows])
    return selected.cast(FINAL_FEATURES)


def _select_pagefill_rows(
    *,
    base_hashes: set[str],
    candidate_dataset,
    candidate_token_lengths: list[int],
    candidate_dataset_path: str,
    max_seq_len: int,
    target_add_count: int,
) -> tuple[list[int], list[dict[str, Any]], dict[str, int]]:
    ranked: list[tuple[float, float, int, int]] = []
    counters: Counter[str] = Counter()
    seen_candidate_hashes: set[str] = set()

    for idx, token_length in enumerate(candidate_token_lengths):
        row = candidate_dataset[idx]
        text = row["transcription"] or ""
        text_hash = _hash_text(text)
        counters["candidate_rows"] += 1
        if token_length > max_seq_len:
            counters["dropped_over_max_seq_len"] += 1
            continue
        if text_hash in base_hashes:
            counters["dropped_duplicate_with_base"] += 1
            continue
        if text_hash in seen_candidate_hashes:
            counters["dropped_duplicate_with_candidates"] += 1
            continue
        bottom = row.get("bottom_whitespace_ratio")
        fill = row.get("vertical_fill_ratio")
        if bottom is None or fill is None:
            counters["dropped_missing_layout"] += 1
            continue
        seen_candidate_hashes.add(text_hash)
        ranked.append((float(bottom), -float(fill), int(token_length), idx))

    ranked.sort()
    if len(ranked) < target_add_count:
        raise ValueError(
            f"Only {len(ranked)} eligible pagefill candidates after filtering; "
            f"cannot select {target_add_count}."
        )

    selected_indices = [idx for _, _, _, idx in ranked[:target_add_count]]
    selected = [
        _candidate_row(
            candidate_dataset,
            idx=idx,
            token_length=candidate_token_lengths[idx],
            candidate_dataset_path=candidate_dataset_path,
        )
        for idx in selected_indices
    ]
    counters["eligible_candidates"] = len(ranked)
    counters["selected_pagefill_rows"] = len(selected)
    return selected_indices, selected, dict(counters)


def assemble_pagefill_augmented_train_dataset(
    base_dataset_path: str = "data/datasets/train/train_full_2",
    candidate_dataset_path: str = "data/datasets/train/train_200k",
    output_dir: str = "data/datasets/train/train_full_2_plus_pagefill_30k_len2048",
    target_add_count: int = 30000,
    max_seq_len: int = 2048,
    tokenizer: str = "vocab/bpe3k-splitspaces/tokenizer.json",
    seed: int = 42,
    metadata_out: str | None = None,
    overwrite: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Create a separate page-fill augmented train dataset."""
    del seed  # Selection is rank-deterministic; retained in the CLI for provenance/stability.
    if target_add_count < 1:
        raise ValueError("target_add_count must be >= 1")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be >= 1")

    output_path = Path(output_dir)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass overwrite=True to replace it")
        shutil.rmtree(output_path)

    tokenizer_path = Path(tokenizer)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    tok = Tokenizer.from_file(str(tokenizer_path))

    base_ds = load_from_disk(base_dataset_path)
    candidate_ds = load_from_disk(candidate_dataset_path)

    base_meta = base_ds.select_columns([column for column in base_ds.column_names if column != "image"])
    candidate_meta = candidate_ds.select_columns(
        [column for column in candidate_ds.column_names if column != "image"]
    )

    base_token_lengths = _token_lengths(base_meta, tok)
    candidate_token_lengths = _token_lengths(candidate_meta, tok)
    base_hashes = {_hash_text(text or "") for text in base_meta["transcription"]}

    base_rows = [
        _base_row(base_meta, idx=idx, token_length=token_length, base_dataset_path=base_dataset_path)
        for idx, token_length in enumerate(base_token_lengths)
    ]
    pagefill_indices, pagefill_rows, selection_counts = _select_pagefill_rows(
        base_hashes=base_hashes,
        candidate_dataset=candidate_meta,
        candidate_token_lengths=candidate_token_lengths,
        candidate_dataset_path=candidate_dataset_path,
        max_seq_len=max_seq_len,
        target_add_count=target_add_count,
    )

    mixed_rows = base_rows + pagefill_rows
    base_normalized = _normalize_selected_dataset(
        base_ds,
        indices=list(range(len(base_ds))),
        rows=base_rows,
    )
    pagefill_normalized = _normalize_selected_dataset(
        candidate_ds,
        indices=pagefill_indices,
        rows=pagefill_rows,
    )
    mixed_ds = concatenate_datasets([base_normalized, pagefill_normalized])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixed_ds.save_to_disk(str(output_path))

    metadata_path = (
        Path(metadata_out)
        if metadata_out is not None
        else output_path.parent / f"{output_path.name}.metadata.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "1.0",
        "output_dir": str(output_path),
        "base_dataset_path": base_dataset_path,
        "candidate_dataset_path": candidate_dataset_path,
        "target_add_count": int(target_add_count),
        "max_seq_len": int(max_seq_len),
        "tokenizer": str(tokenizer_path),
        "total_rows": len(mixed_rows),
        "base_rows": len(base_rows),
        "pagefill_rows": len(pagefill_rows),
        "selection_counts": selection_counts,
        "base_summary": _summarize_rows(base_rows),
        "pagefill_summary": _summarize_rows(pagefill_rows),
        "mixed_summary": _summarize_rows(mixed_rows),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    if not quiet:
        print(
            f"Saved {len(mixed_rows):,} rows to {output_path} "
            f"({len(base_rows):,} base + {len(pagefill_rows):,} pagefill)."
        )
        print(f"Wrote metadata to {metadata_path}")
    return metadata


def main() -> None:
    fire.Fire(assemble_pagefill_augmented_train_dataset)


if __name__ == "__main__":
    main()
