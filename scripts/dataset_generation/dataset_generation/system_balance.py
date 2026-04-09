"""Token-length calibration and soft balancing helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from tokenizers import Tokenizer

from scripts.dataset_generation.dataset_generation.composer import plan_sample
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.source_index import SourceIndex
from scripts.dataset_generation.dataset_generation.types import SamplePlan

DEFAULT_TOKENIZER_DIR = Path("vocab/bpe3k-splitspaces")
TARGET_SYSTEM_BUCKETS = (1, 2, 3, 4, 5, 6)
DEFAULT_CANDIDATE_PLAN_COUNT = 8
DEFAULT_BUNDLED_SYSTEM_BALANCE_SPEC_PATH = (
    Path(__file__).resolve().parent / "default_system_balance_spec.json"
)


@dataclass(frozen=True)
class TokenLengthBucket:
    min_length: int
    max_length: int
    center_length: float

    def contains(self, token_length: int) -> bool:
        return self.min_length <= int(token_length) <= self.max_length


@dataclass(frozen=True)
class SystemBalanceSpec:
    path: Path
    mode: str
    tokenizer_path: Path
    tokenizer_fingerprint: str
    recipe_fingerprint: str
    candidate_plan_count: int
    token_length_ranges: dict[int, TokenLengthBucket]


@dataclass(frozen=True)
class CandidatePlanScore:
    candidate_idx: int
    plan: SamplePlan
    token_length: int
    target_bucket: int
    target_center_length: float
    in_target_range: bool
    distance_to_bucket: float


def resolve_tokenizer_dir(tokenizer_path: str | Path | None = None) -> Path:
    candidate = DEFAULT_TOKENIZER_DIR if tokenizer_path is None else Path(tokenizer_path)
    resolved = candidate.expanduser().resolve()
    tokenizer_json_path = resolved / "tokenizer.json"
    if not tokenizer_json_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_json_path}")
    return resolved


def load_tokenizer(tokenizer_path: str | Path | None = None) -> tuple[Path, Tokenizer]:
    tokenizer_dir = resolve_tokenizer_dir(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    return tokenizer_dir, tokenizer


def compute_recipe_fingerprint(recipe: ProductionRecipe) -> str:
    serialized = json.dumps(asdict(recipe), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_tokenizer_fingerprint(tokenizer_dir: str | Path) -> str:
    tokenizer_json_path = resolve_tokenizer_dir(tokenizer_dir) / "tokenizer.json"
    return hashlib.sha256(tokenizer_json_path.read_bytes()).hexdigest()


def compute_token_length(transcription: str, tokenizer: Tokenizer) -> int:
    return len(tokenizer.encode(transcription, add_special_tokens=True).ids)


def load_system_balance_spec(path: str | Path) -> SystemBalanceSpec:
    spec_path = Path(path).expanduser().resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    ranges_payload = payload.get("recommended_token_length_ranges")
    if not isinstance(ranges_payload, dict) or not ranges_payload:
        raise ValueError("System balance spec is missing recommended_token_length_ranges")
    token_length_ranges: dict[int, TokenLengthBucket] = {}
    for raw_bucket, raw_range in ranges_payload.items():
        bucket = int(raw_bucket)
        if bucket not in TARGET_SYSTEM_BUCKETS:
            continue
        if not isinstance(raw_range, dict):
            raise ValueError(f"Invalid token-length range for bucket {bucket}")
        token_length_ranges[bucket] = TokenLengthBucket(
            min_length=int(raw_range["min"]),
            max_length=int(raw_range["max"]),
            center_length=float(raw_range.get("center", (raw_range["min"] + raw_range["max"]) / 2.0)),
        )
    if not token_length_ranges:
        raise ValueError("System balance spec contains no supported target buckets")
    tokenizer_meta = payload.get("tokenizer", {})
    return SystemBalanceSpec(
        path=spec_path,
        mode=str(payload.get("mode", "length_proxy")),
        tokenizer_path=Path(tokenizer_meta.get("path", DEFAULT_TOKENIZER_DIR)).expanduser().resolve(),
        tokenizer_fingerprint=str(tokenizer_meta.get("fingerprint", "")),
        recipe_fingerprint=str(payload.get("recipe_fingerprint", "")),
        candidate_plan_count=int(payload.get("candidate_plan_count", DEFAULT_CANDIDATE_PLAN_COUNT)),
        token_length_ranges=token_length_ranges,
    )


def load_bundled_system_balance_spec() -> SystemBalanceSpec:
    return load_system_balance_spec(DEFAULT_BUNDLED_SYSTEM_BALANCE_SPEC_PATH)


def choose_target_bucket(accepted_system_histogram: Mapping[int | str, int]) -> int:
    bucket_counts = {
        bucket: int(
            accepted_system_histogram.get(bucket, accepted_system_histogram.get(str(bucket), 0))  # type: ignore[arg-type]
        )
        for bucket in TARGET_SYSTEM_BUCKETS
    }
    min_count = min(bucket_counts.values())
    for bucket in TARGET_SYSTEM_BUCKETS:
        if bucket_counts[bucket] == min_count:
            return bucket
    return TARGET_SYSTEM_BUCKETS[0]


def choose_balanced_plan(
    *,
    source_index: SourceIndex,
    recipe: ProductionRecipe,
    sample_idx: int,
    base_seed: int,
    excluded_paths: set[Path] | None,
    tokenizer: Tokenizer,
    spec: SystemBalanceSpec,
    accepted_system_histogram: Mapping[int | str, int],
    candidate_plan_count: int | None = None,
) -> CandidatePlanScore:
    available_buckets = tuple(
        bucket for bucket in TARGET_SYSTEM_BUCKETS if bucket in spec.token_length_ranges
    )
    if not available_buckets:
        raise ValueError("System balance spec does not define any target buckets")
    bucket_counts = {
        bucket: int(
            accepted_system_histogram.get(bucket, accepted_system_histogram.get(str(bucket), 0))  # type: ignore[arg-type]
        )
        for bucket in available_buckets
    }
    min_count = min(bucket_counts.values())
    target_bucket = next(bucket for bucket in available_buckets if bucket_counts[bucket] == min_count)
    bucket = spec.token_length_ranges[target_bucket]

    total_candidates = max(1, int(candidate_plan_count or spec.candidate_plan_count or 1))
    best_score: CandidatePlanScore | None = None
    last_error: ValueError | None = None
    for candidate_idx in range(total_candidates):
        candidate_base_seed = derive_candidate_base_seed(
            base_seed=base_seed,
            sample_idx=sample_idx,
            candidate_idx=candidate_idx,
        )
        try:
            plan = plan_sample(
                source_index,
                recipe,
                sample_idx=sample_idx,
                base_seed=candidate_base_seed,
                excluded_paths=excluded_paths,
            )
        except ValueError as exc:
            last_error = exc
            continue
        token_length = compute_token_length(plan.label_transcription, tokenizer)
        distance = distance_to_bucket(token_length, bucket)
        score = CandidatePlanScore(
            candidate_idx=candidate_idx,
            plan=plan,
            token_length=token_length,
            target_bucket=target_bucket,
            target_center_length=bucket.center_length,
            in_target_range=bucket.contains(token_length),
            distance_to_bucket=distance,
        )
        if best_score is None or _candidate_sort_key(score) < _candidate_sort_key(best_score):
            best_score = score

    if best_score is None:
        reason = (
            str(last_error)
            if last_error is not None
            else "no candidate plans could be generated"
        )
        raise ValueError(f"All candidate plans were invalid or exhausted: {reason}")
    return best_score


def derive_candidate_base_seed(*, base_seed: int, sample_idx: int, candidate_idx: int) -> int:
    mixed = (
        (int(base_seed) & 0xFFFFFFFF) * 1_000_003
        + (int(sample_idx) & 0xFFFFFFFF) * 97_409
        + (int(candidate_idx) & 0xFFFFFFFF) * 65_537
    ) & 0xFFFFFFFF
    return mixed


def distance_to_bucket(token_length: int, bucket: TokenLengthBucket) -> float:
    token_length = int(token_length)
    if bucket.contains(token_length):
        return abs(token_length - bucket.center_length) / max(1.0, bucket.max_length - bucket.min_length + 1.0)
    if token_length < bucket.min_length:
        return float(bucket.min_length - token_length)
    return float(token_length - bucket.max_length)


def build_calibration_artifact(
    *,
    records: list[dict[str, object]],
    tokenizer_dir: Path,
    recipe: ProductionRecipe,
    input_dirs: tuple[Path, ...],
    sample_budget: int,
    candidate_plan_count: int = DEFAULT_CANDIDATE_PLAN_COUNT,
) -> dict[str, object]:
    def _accepted_system_count(record: dict[str, object]) -> int | None:
        value = record.get("accepted_render_system_count", record.get("svg_system_count"))
        return int(value) if value is not None else None

    def _full_render_system_count(record: dict[str, object]) -> int | None:
        value = record.get("full_render_system_count", record.get("svg_system_count"))
        return int(value) if value is not None else None

    def _calibration_bucket_system_count(record: dict[str, object]) -> int | None:
        full_value = _full_render_system_count(record)
        if full_value is not None:
            return full_value
        return _accepted_system_count(record)

    accepted_records = [
        record
        for record in records
        if record.get("accepted") is True
        and (_accepted_system_count(record) or 0) in TARGET_SYSTEM_BUCKETS
    ]
    calibration_records = [
        record
        for record in records
        if (_calibration_bucket_system_count(record) or 0) in TARGET_SYSTEM_BUCKETS
    ]
    recommended_ranges: dict[str, dict[str, int | float]] = {}
    per_system_stats: dict[str, dict[str, object]] = {}

    for bucket in TARGET_SYSTEM_BUCKETS:
        bucket_records = [
            int(record["token_length"])
            for record in calibration_records
            if _calibration_bucket_system_count(record) == bucket
        ]
        if not bucket_records:
            continue
        stats = summarize_lengths(bucket_records)
        recommended_ranges[str(bucket)] = {
            "min": int(stats["q20"]),
            "max": int(stats["q80"]),
            "center": float(stats["median"]),
        }
        per_system_stats[str(bucket)] = {
            "calibration_count": len(bucket_records),
            "accepted_count": sum(
                1
                for record in accepted_records
                if _accepted_system_count(record) == bucket
            ),
            "full_render_count": sum(
                1
                for record in records
                if _full_render_system_count(record) == bucket
            ),
            "token_length": stats,
        }

    return {
        "schema_version": "1.0",
        "mode": "length_proxy",
        "calibration_target_metric": "full_render_system_count_preferred",
        "created_at": time_now(),
        "input_dirs": [str(path) for path in input_dirs],
        "sample_budget": int(sample_budget),
        "attempted_samples": len(records),
        "accepted_samples": len(accepted_records),
        "candidate_plan_count": int(candidate_plan_count),
        "tokenizer": {
            "path": str(tokenizer_dir),
            "fingerprint": compute_tokenizer_fingerprint(tokenizer_dir),
        },
        "recipe_version": recipe.version,
        "recipe_fingerprint": compute_recipe_fingerprint(recipe),
        "systems": per_system_stats,
        "recommended_token_length_ranges": recommended_ranges,
        "diagnostics": build_overlap_diagnostics(recommended_ranges),
        "coverage": {
            "accepted_render_system_histogram": histogram_by_key(
                accepted_records,
                key=lambda record: (
                    str(_accepted_system_count(record))
                    if _accepted_system_count(record) is not None
                    else None
                ),
            ),
            "full_render_system_histogram": histogram_by_key(
                [record for record in records if _full_render_system_count(record) is not None],
                key=lambda record: (
                    str(_full_render_system_count(record))
                    if _full_render_system_count(record) is not None
                    else None
                ),
            ),
            "truncated_accepted_output_system_histogram": histogram_by_key(
                [
                    record
                    for record in accepted_records
                    if record.get("truncation_applied") is True
                    and _accepted_system_count(record) is not None
                ],
                key=lambda record: (
                    str(_accepted_system_count(record))
                    if _accepted_system_count(record) is not None
                    else None
                ),
            ),
            "preferred_5_6_counts": histogram_by_key(
                [record for record in records if record.get("preferred_5_6_status") is not None],
                key=lambda record: record.get("preferred_5_6_status"),
            ),
            "accepted_segment_histogram": histogram_by_key(
                accepted_records,
                key=lambda record: str(int(record["segment_count"])),
            ),
            "accepted_source_root_histogram": histogram_by_key(
                accepted_records,
                key=lambda record: ",".join(record.get("source_root_labels", [])),  # type: ignore[arg-type]
            ),
        },
        "records": records,
    }


def summarize_lengths(lengths: list[int]) -> dict[str, int | float]:
    values = np.asarray(lengths, dtype=np.int32)
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.percentile(values, 50)),
        "q10": int(np.percentile(values, 10)),
        "q20": int(np.percentile(values, 20)),
        "q25": int(np.percentile(values, 25)),
        "q75": int(np.percentile(values, 75)),
        "q80": int(np.percentile(values, 80)),
        "q90": int(np.percentile(values, 90)),
    }


def build_overlap_diagnostics(
    recommended_ranges: Mapping[str, Mapping[str, int | float]]
) -> dict[str, object]:
    overlaps: list[dict[str, int | list[str]]] = []
    ordered = sorted((int(bucket), bucket_range) for bucket, bucket_range in recommended_ranges.items())
    for idx, (left_bucket, left_range) in enumerate(ordered):
        for right_bucket, right_range in ordered[idx + 1 :]:
            overlap_min = max(int(left_range["min"]), int(right_range["min"]))
            overlap_max = min(int(left_range["max"]), int(right_range["max"]))
            if overlap_min > overlap_max:
                continue
            overlaps.append(
                {
                    "systems": [str(left_bucket), str(right_bucket)],
                    "overlap_min": overlap_min,
                    "overlap_max": overlap_max,
                    "overlap_width": overlap_max - overlap_min + 1,
                }
            )
    return {
        "bucket_overlaps": overlaps,
    }


def histogram_by_key(
    records: list[dict[str, object]],
    *,
    key,
) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for record in records:
        raw_key = key(record)
        if raw_key is None or raw_key == "":
            continue
        histogram[str(raw_key)] = histogram.get(str(raw_key), 0) + 1
    return histogram


def time_now() -> float:
    import time

    return time.time()


def _candidate_sort_key(score: CandidatePlanScore) -> tuple[float, float, int]:
    bucket = score.distance_to_bucket
    center_distance = abs(score.token_length - score.target_center_length)
    return (bucket, center_distance, score.candidate_idx)
