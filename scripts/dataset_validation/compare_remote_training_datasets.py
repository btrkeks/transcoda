#!/usr/bin/env python3
"""Compare two HF OMR training datasets using text/provenance-only stats."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from datasets import load_from_disk
from tokenizers import Tokenizer


THRESHOLDS = (512, 1024, 1536, 1800, 1900, 2000, 2048)
CONTROL_PREFIXES = ("*", "!", "=")
SPINE_OP_PATTERNS = {
    "split": "*^",
    "merge": "*v",
    "exchange": "*x",
    "add": "*+",
    "terminate": "*-",
}


def percentile(values: list[int] | np.ndarray, q: float) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.percentile(values, q))


def compact_counter(counter: Counter[str], limit: int) -> dict[str, int]:
    return dict(counter.most_common(limit))


def source_key(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        if not value:
            return "unknown"
        value = value[0]
    text = str(value)
    for marker in ("grandstaff", "musetrainer", "pdmx", "polish", "olimpic", "openscore"):
        if marker in text.lower():
            return marker
    parts = Path(text).parts
    for part in parts:
        low = part.lower()
        if low in {"grandstaff", "musetrainer", "pdmx", "polish", "olimpic", "openscore"}:
            return low
    return parts[0] if parts else text[:80]


def infer_source(row: dict[str, Any]) -> str:
    for col in ("source", "dataset", "dataset_name", "source_dataset"):
        if col in row:
            return source_key(row[col])
    if "source_ids" in row:
        return source_key(row["source_ids"])
    if "sample_id" in row:
        return source_key(row["sample_id"])
    return "unknown"


def longest_identical_run(lines: list[str]) -> int:
    best = 0
    current = 0
    previous = None
    for line in lines:
        if line == previous:
            current += 1
        else:
            previous = line
            current = 1
        if current > best:
            best = current
    return best


def token_ngram_repeat_stats(token_ids: list[int], n: int = 10) -> tuple[int, int]:
    if len(token_ids) < n:
        return 0, 0
    counts: Counter[tuple[int, ...]] = Counter(
        tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)
    )
    repeated = sum(1 for count in counts.values() if count > 1)
    max_occurrences = max(counts.values(), default=0)
    return repeated, max_occurrences


def analyze_dataset(
    label: str,
    path: str,
    tokenizer: Tokenizer,
    sample_size: int,
    top_examples: int,
) -> dict[str, Any]:
    ds = load_from_disk(path)
    original_columns = list(ds.column_names)
    keep_columns = [
        col
        for col in (
            "transcription",
            "source",
            "source_ids",
            "sample_id",
            "dataset",
            "dataset_name",
            "source_dataset",
        )
        if col in ds.column_names
    ]
    ds = ds.select_columns(keep_columns)

    n = len(ds)
    rng = random.Random(0)
    sample_indices = set(rng.sample(range(n), min(sample_size, n))) if n else set()

    token_lengths: list[int] = []
    char_lengths: list[int] = []
    line_counts: list[int] = []
    measure_counts: list[int] = []
    max_spines: list[int] = []
    line_repeat_ratios: list[float] = []
    longest_runs: list[int] = []
    control_line_rates: list[float] = []
    top_sources: Counter[str] = Counter()
    terminal_lines: Counter[str] = Counter()
    first_lines: Counter[str] = Counter()
    spine_ops: Counter[str] = Counter()
    hashes: set[str] = set()

    bad_examples: dict[str, list[dict[str, Any]]] = {
        "longest_tokens": [],
        "missing_termination": [],
        "high_identical_line_run": [],
        "high_10gram_repeat": [],
        "empty_or_tiny": [],
    }

    counts = Counter()

    for idx, row in enumerate(ds):
        text = row["transcription"] or ""
        token_ids = tokenizer.encode(text, add_special_tokens=True).ids
        token_len = len(token_ids)
        lines = [line.rstrip("\n") for line in text.splitlines() if line.strip()]
        line_count = len(lines)
        unique_lines = len(set(lines))
        measure_count = sum(1 for line in lines if line.startswith("="))
        max_spine = max((line.count("\t") + 1 for line in lines), default=0)
        control_lines = sum(1 for line in lines if line.startswith(CONTROL_PREFIXES))
        last = lines[-1] if lines else ""
        first = lines[0] if lines else ""
        identical_run = longest_identical_run(lines)

        token_lengths.append(token_len)
        char_lengths.append(len(text))
        line_counts.append(line_count)
        measure_counts.append(measure_count)
        max_spines.append(max_spine)
        line_repeat_ratios.append(1.0 - unique_lines / line_count if line_count else 0.0)
        longest_runs.append(identical_run)
        control_line_rates.append(control_lines / line_count if line_count else 0.0)
        top_sources[infer_source(row)] += 1
        terminal_lines[last[:120]] += 1
        first_lines[first[:120]] += 1
        hashes.add(hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=12).hexdigest())

        if not lines:
            counts["empty"] += 1
        if token_len <= 32:
            counts["token_len_le_32"] += 1
        if not first.startswith("**kern"):
            counts["first_line_not_kern"] += 1
        if "*-" not in last:
            counts["last_line_not_termination"] += 1
        if not text.endswith("\n"):
            counts["no_trailing_newline"] += 1
        if "\ufffd" in text:
            counts["replacement_char"] += 1
        if any(ord(ch) > 127 for ch in text):
            counts["non_ascii"] += 1
        if token_len > 2048:
            counts["token_len_gt_2048"] += 1
        if token_len >= 2000:
            counts["token_len_ge_2000"] += 1
        if identical_run >= 5:
            counts["identical_line_run_ge_5"] += 1
        if identical_run >= 10:
            counts["identical_line_run_ge_10"] += 1
        if line_count and control_lines / line_count >= 0.5:
            counts["control_line_rate_ge_50pct"] += 1

        for name, pattern in SPINE_OP_PATTERNS.items():
            hits = text.count(pattern)
            if hits:
                spine_ops[name] += hits

        example = {
            "idx": idx,
            "source": infer_source(row),
            "tokens": token_len,
            "chars": len(text),
            "lines": line_count,
            "measures": measure_count,
            "max_spines": max_spine,
            "last": last[:120],
            "first": first[:120],
            "hash": hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=12).hexdigest(),
        }

        def keep_top(bucket: str, score: int) -> None:
            items = bad_examples[bucket]
            item = dict(example)
            item["score"] = score
            items.append(item)
            items.sort(key=lambda x: x["score"], reverse=True)
            del items[top_examples:]

        keep_top("longest_tokens", token_len)
        if "*-" not in last:
            bad_examples["missing_termination"].append(example)
            del bad_examples["missing_termination"][top_examples:]
        if identical_run >= 5:
            keep_top("high_identical_line_run", identical_run)
        if token_len <= 32 or not lines:
            bad_examples["empty_or_tiny"].append(example)
            del bad_examples["empty_or_tiny"][top_examples:]

        if idx in sample_indices:
            repeated_10grams, max_10gram_occurrences = token_ngram_repeat_stats(token_ids, 10)
            if repeated_10grams:
                counts["sample_has_repeated_10gram"] += 1
            if max_10gram_occurrences >= 5:
                counts["sample_10gram_occurs_ge_5"] += 1
            if max_10gram_occurrences >= 10:
                counts["sample_10gram_occurs_ge_10"] += 1
            keep_top("high_10gram_repeat", max_10gram_occurrences)

    arr = np.array(token_lengths, dtype=np.int32)
    summary = {
        "label": label,
        "path": path,
        "num_rows": n,
        "columns": original_columns,
        "text_columns_used": keep_columns,
        "unique_transcription_hashes": len(hashes),
        "duplicate_transcription_rows": n - len(hashes),
        "source_counts": compact_counter(top_sources, 30),
        "source_rates": {k: v / n for k, v in top_sources.most_common(30)} if n else {},
        "token_length": {
            "min": int(arr.min()) if n else None,
            "mean": float(arr.mean()) if n else None,
            "p50": percentile(arr, 50),
            "p90": percentile(arr, 90),
            "p95": percentile(arr, 95),
            "p99": percentile(arr, 99),
            "p999": percentile(arr, 99.9),
            "max": int(arr.max()) if n else None,
            "threshold_counts": {f">={t}": int((arr >= t).sum()) for t in THRESHOLDS},
            "threshold_rates": {f">={t}": float((arr >= t).mean()) for t in THRESHOLDS} if n else {},
        },
        "char_length": {
            "mean": mean(char_lengths) if char_lengths else None,
            "p95": percentile(char_lengths, 95),
            "p99": percentile(char_lengths, 99),
            "max": max(char_lengths) if char_lengths else None,
        },
        "line_count": {
            "mean": mean(line_counts) if line_counts else None,
            "p95": percentile(line_counts, 95),
            "p99": percentile(line_counts, 99),
            "max": max(line_counts) if line_counts else None,
        },
        "measure_count": {
            "mean": mean(measure_counts) if measure_counts else None,
            "p95": percentile(measure_counts, 95),
            "p99": percentile(measure_counts, 99),
            "max": max(measure_counts) if measure_counts else None,
        },
        "max_spines": {
            "mean": mean(max_spines) if max_spines else None,
            "p95": percentile(max_spines, 95),
            "p99": percentile(max_spines, 99),
            "max": max(max_spines) if max_spines else None,
        },
        "line_repetition": {
            "mean_unique_line_deficit": mean(line_repeat_ratios) if line_repeat_ratios else None,
            "p95_longest_identical_run": percentile(longest_runs, 95),
            "p99_longest_identical_run": percentile(longest_runs, 99),
            "max_longest_identical_run": max(longest_runs) if longest_runs else None,
        },
        "control_line_rate": {
            "mean": mean(control_line_rates) if control_line_rates else None,
            "p95": percentile(control_line_rates, 95),
            "p99": percentile(control_line_rates, 99),
        },
        "counts": dict(counts),
        "count_rates": {k: v / n for k, v in counts.items()} if n else {},
        "spine_op_totals": dict(spine_ops),
        "spine_op_per_1k_samples": {k: v / n * 1000 for k, v in spine_ops.items()} if n else {},
        "top_terminal_lines": compact_counter(terminal_lines, 20),
        "top_first_lines": compact_counter(first_lines, 20),
        "sample_size_for_ngram_stats": len(sample_indices),
        "examples": bad_examples,
        "_hashes": hashes,
    }
    return summary


def diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    hashes_a = a.pop("_hashes")
    hashes_b = b.pop("_hashes")
    return {
        "a_minus_b_unique_transcriptions": len(hashes_a - hashes_b),
        "b_minus_a_unique_transcriptions": len(hashes_b - hashes_a),
        "exact_transcription_overlap": len(hashes_a & hashes_b),
        "exact_overlap_rate_of_a": len(hashes_a & hashes_b) / a["unique_transcription_hashes"],
        "exact_overlap_rate_of_b": len(hashes_a & hashes_b) / b["unique_transcription_hashes"],
        "source_count_delta_a_minus_b": {
            k: a["source_counts"].get(k, 0) - b["source_counts"].get(k, 0)
            for k in sorted(set(a["source_counts"]) | set(b["source_counts"]))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-label", default="new")
    parser.add_argument("--a-path", required=True)
    parser.add_argument("--b-label", default="old")
    parser.add_argument("--b-path", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--top-examples", type=int, default=10)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    a = analyze_dataset(args.a_label, args.a_path, tokenizer, args.sample_size, args.top_examples)
    b = analyze_dataset(args.b_label, args.b_path, tokenizer, args.sample_size, args.top_examples)
    payload = {"datasets": {args.a_label: a, args.b_label: b}}
    payload["diff"] = diff(a, b)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
