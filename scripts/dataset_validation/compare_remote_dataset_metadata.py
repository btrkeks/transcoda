#!/usr/bin/env python3
"""Summarize non-image metadata columns for two HF datasets."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_from_disk


def source_group(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        value = value[0] if value else "unknown"
    text = str(value)
    low = text.lower()
    for marker in ("grandstaff", "musetrainer", "pdmx", "polish", "olimpic", "openscore"):
        if marker in low:
            return marker
    return text[:120]


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def summarize_column(values: list[Any]) -> dict[str, Any]:
    non_null = [v for v in values if v is not None]
    result: dict[str, Any] = {"non_null": len(non_null)}
    if not non_null:
        return result
    if all(isinstance(v, bool) for v in non_null):
        true_count = sum(bool(v) for v in non_null)
        result.update({"true": true_count, "false": len(non_null) - true_count, "true_rate": true_count / len(non_null)})
    elif all(is_number(v) for v in non_null):
        nums = [float(v) for v in non_null if math.isfinite(float(v))]
        result.update(
            {
                "min": min(nums) if nums else None,
                "mean": float(np.mean(nums)) if nums else None,
                "p50": pct(nums, 50),
                "p90": pct(nums, 90),
                "p95": pct(nums, 95),
                "p99": pct(nums, 99),
                "max": max(nums) if nums else None,
            }
        )
    else:
        counts = Counter(str(v) for v in non_null)
        result["top"] = dict(counts.most_common(30))
        result["unique"] = len(counts)
    return result


def summarize(path: str) -> dict[str, Any]:
    ds = load_from_disk(path)
    columns = [c for c in ds.column_names if c != "image"]
    meta = ds.select_columns(columns)
    out: dict[str, Any] = {"path": path, "num_rows": len(ds), "columns": ds.column_names, "columns_summary": {}}
    for col in columns:
        if col == "transcription":
            continue
        values = meta[col]
        out["columns_summary"][col] = summarize_column(values)

    source_candidates = [c for c in ("source_dataset", "source_domain", "source", "source_ids", "sample_id") if c in columns]
    out["source_candidates"] = {}
    for col in source_candidates:
        out["source_candidates"][col] = dict(Counter(source_group(v) for v in meta[col]).most_common(50))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-label", required=True)
    parser.add_argument("--a-path", required=True)
    parser.add_argument("--b-label", required=True)
    parser.add_argument("--b-path", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    payload = {
        args.a_label: summarize(args.a_path),
        args.b_label: summarize(args.b_path),
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
