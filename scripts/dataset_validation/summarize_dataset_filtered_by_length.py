#!/usr/bin/env python3
"""Summarize a HF dataset before/after a token-length filter."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_from_disk
from tokenizers import Tokenizer


def source_group(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        value = value[0] if value else "unknown"
    text = str(value).lower()
    for marker in ("grandstaff", "musetrainer", "openscore", "pdmx", "polish", "olimpic"):
        if marker in text:
            return marker
    return str(value)[:120]


def source_for_row(row: dict[str, Any]) -> str:
    for col in ("source_dataset", "source_domain", "source", "source_ids", "sample_id"):
        if col in row:
            return source_group(row[col])
    return "unknown"


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    token_lengths = [row["tokens"] for row in rows]
    return {
        "rows": len(rows),
        "source_counts": dict(Counter(row["source"] for row in rows).most_common()),
        "token_length": {
            "mean": float(np.mean(token_lengths)),
            "p50": pct(token_lengths, 50),
            "p90": pct(token_lengths, 90),
            "p95": pct(token_lengths, 95),
            "p99": pct(token_lengths, 99),
            "max": max(token_lengths),
        },
        "max_spines": {
            "mean": float(np.mean([row["max_spines"] for row in rows])),
            "p50": pct([row["max_spines"] for row in rows], 50),
            "p90": pct([row["max_spines"] for row in rows], 90),
            "p95": pct([row["max_spines"] for row in rows], 95),
            "p99": pct([row["max_spines"] for row in rows], 99),
            "max": max(row["max_spines"] for row in rows),
        },
        "measure_count": {
            "mean": float(np.mean([row["measures"] for row in rows])),
            "p95": pct([row["measures"] for row in rows], 95),
            "p99": pct([row["measures"] for row in rows], 99),
            "max": max(row["measures"] for row in rows),
        },
        "vertical_fill_ratio": {
            "mean": float(np.mean([row["vertical_fill_ratio"] for row in rows if row["vertical_fill_ratio"] is not None])),
            "p50": pct([row["vertical_fill_ratio"] for row in rows if row["vertical_fill_ratio"] is not None], 50),
            "p95": pct([row["vertical_fill_ratio"] for row in rows if row["vertical_fill_ratio"] is not None], 95),
        },
        "bottom_whitespace_ratio": {
            "mean": float(np.mean([row["bottom_whitespace_ratio"] for row in rows if row["bottom_whitespace_ratio"] is not None])),
            "p50": pct([row["bottom_whitespace_ratio"] for row in rows if row["bottom_whitespace_ratio"] is not None], 50),
            "p95": pct([row["bottom_whitespace_ratio"] for row in rows if row["bottom_whitespace_ratio"] is not None], 95),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ds = load_from_disk(args.dataset)
    keep_columns = [
        col
        for col in (
            "transcription",
            "source_dataset",
            "source_domain",
            "source",
            "source_ids",
            "sample_id",
            "vertical_fill_ratio",
            "bottom_whitespace_ratio",
        )
        if col in ds.column_names
    ]
    ds = ds.select_columns(keep_columns)

    rows: list[dict[str, Any]] = []
    for row in ds:
        text = row["transcription"] or ""
        lines = [line for line in text.splitlines() if line.strip()]
        rows.append(
            {
                "tokens": len(tokenizer.encode(text, add_special_tokens=True).ids),
                "source": source_for_row(row),
                "max_spines": max((line.count("\t") + 1 for line in lines), default=0),
                "measures": sum(1 for line in lines if line.startswith("=")),
                "vertical_fill_ratio": row.get("vertical_fill_ratio"),
                "bottom_whitespace_ratio": row.get("bottom_whitespace_ratio"),
            }
        )

    kept = [row for row in rows if row["tokens"] <= args.max_len]
    removed = [row for row in rows if row["tokens"] > args.max_len]
    payload = {
        "dataset": args.dataset,
        "max_len": args.max_len,
        "all": summarize_rows(rows),
        "kept": summarize_rows(kept),
        "removed": summarize_rows(removed),
        "removed_rate": len(removed) / len(rows) if rows else 0.0,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
