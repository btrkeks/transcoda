"""Spine-aware line-count calibration and soft balancing helpers."""

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
SPINE_CLASS_ORDER = ("all", "1", "2", "3_plus")
UNSAFE_VERTICAL_FIT_REJECTION_REASONS = {
    "multi_page",
    "top_clearance",
    "bottom_clearance",
    "crop_risk",
}


@dataclass(frozen=True)
class LineCountBucket:
    min_line_count: int
    max_line_count: int
    center_line_count: float

    def contains(self, line_count: int) -> bool:
        return self.min_line_count <= int(line_count) <= self.max_line_count


@dataclass(frozen=True)
class VerticalFitBucket:
    safe_max_line_count: int
    median_content_height_px: float
    safe_sample_count: int


@dataclass(frozen=True)
class SystemBalanceSpec:
    path: Path
    mode: str
    tokenizer_path: Path
    tokenizer_fingerprint: str
    recipe_fingerprint: str
    candidate_plan_count: int
    line_count_ranges: dict[str, dict[int, LineCountBucket]]
    vertical_fit_model: dict[str, dict[int, VerticalFitBucket]]


@dataclass(frozen=True)
class CandidatePlanScore:
    candidate_idx: int
    plan: SamplePlan
    line_count: int
    source_max_initial_spine_count: int
    spine_class: str
    target_bucket: int
    target_center_line_count: float
    in_target_range: bool
    distance_to_bucket: float
    vertical_fit_penalty: float


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


def spine_class_for_count(initial_spine_count: int) -> str:
    count = max(1, int(initial_spine_count))
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3_plus"


def load_system_balance_spec(path: str | Path) -> SystemBalanceSpec:
    spec_path = Path(path).expanduser().resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    ranges_payload = payload.get("recommended_line_count_ranges")
    if not isinstance(ranges_payload, dict) or not ranges_payload:
        raise ValueError("System balance spec is missing recommended_line_count_ranges")

    line_count_ranges: dict[str, dict[int, LineCountBucket]] = {}
    for raw_class, raw_class_ranges in ranges_payload.items():
        class_name = str(raw_class)
        if class_name not in SPINE_CLASS_ORDER:
            continue
        if not isinstance(raw_class_ranges, dict):
            raise ValueError(f"Invalid line-count ranges for spine class {class_name!r}")
        parsed_class_ranges: dict[int, LineCountBucket] = {}
        for raw_bucket, raw_range in raw_class_ranges.items():
            bucket = int(raw_bucket)
            if bucket not in TARGET_SYSTEM_BUCKETS:
                continue
            if not isinstance(raw_range, dict):
                raise ValueError(f"Invalid line-count range for bucket {bucket}")
            parsed_class_ranges[bucket] = LineCountBucket(
                min_line_count=int(raw_range["min"]),
                max_line_count=int(raw_range["max"]),
                center_line_count=float(
                    raw_range.get("center", (raw_range["min"] + raw_range["max"]) / 2.0)
                ),
            )
        if parsed_class_ranges:
            line_count_ranges[class_name] = parsed_class_ranges
    if not line_count_ranges:
        raise ValueError("System balance spec contains no supported line-count buckets")

    vertical_fit_payload = payload.get("vertical_fit_model", {})
    if not isinstance(vertical_fit_payload, dict):
        raise ValueError("System balance spec has invalid vertical_fit_model")
    vertical_fit_model: dict[str, dict[int, VerticalFitBucket]] = {}
    for raw_class, raw_class_buckets in vertical_fit_payload.items():
        class_name = str(raw_class)
        if class_name not in SPINE_CLASS_ORDER:
            continue
        if not isinstance(raw_class_buckets, dict):
            raise ValueError(f"Invalid vertical-fit entries for spine class {class_name!r}")
        parsed_fit_buckets: dict[int, VerticalFitBucket] = {}
        for raw_bucket, raw_entry in raw_class_buckets.items():
            bucket = int(raw_bucket)
            if bucket not in TARGET_SYSTEM_BUCKETS:
                continue
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Invalid vertical-fit entry for bucket {bucket}")
            parsed_fit_buckets[bucket] = VerticalFitBucket(
                safe_max_line_count=int(raw_entry["safe_max_line_count"]),
                median_content_height_px=float(raw_entry["median_content_height_px"]),
                safe_sample_count=int(raw_entry["safe_sample_count"]),
            )
        if parsed_fit_buckets:
            vertical_fit_model[class_name] = parsed_fit_buckets

    tokenizer_meta = payload.get("tokenizer", {})
    return SystemBalanceSpec(
        path=spec_path,
        mode=str(payload.get("mode", "spine_aware_line_proxy")),
        tokenizer_path=Path(tokenizer_meta.get("path", DEFAULT_TOKENIZER_DIR)).expanduser().resolve(),
        tokenizer_fingerprint=str(tokenizer_meta.get("fingerprint", "")),
        recipe_fingerprint=str(payload.get("recipe_fingerprint", "")),
        candidate_plan_count=int(payload.get("candidate_plan_count", DEFAULT_CANDIDATE_PLAN_COUNT)),
        line_count_ranges=line_count_ranges,
        vertical_fit_model=vertical_fit_model,
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
    spec: SystemBalanceSpec,
    accepted_system_histogram: Mapping[int | str, int],
    candidate_plan_count: int | None = None,
) -> CandidatePlanScore:
    available_buckets = tuple(
        bucket for bucket in TARGET_SYSTEM_BUCKETS if _has_line_count_bucket(spec, bucket)
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
        line_count = int(plan.source_non_empty_line_count)
        source_max_initial_spine_count = int(plan.source_max_initial_spine_count)
        spine_class = spine_class_for_count(source_max_initial_spine_count)
        bucket = _resolve_line_count_bucket(spec=spec, spine_class=spine_class, target_bucket=target_bucket)
        vertical_fit_bucket = _resolve_vertical_fit_bucket(
            spec=spec,
            spine_class=spine_class,
            target_bucket=target_bucket,
        )
        distance = distance_to_bucket(line_count, bucket)
        score = CandidatePlanScore(
            candidate_idx=candidate_idx,
            plan=plan,
            line_count=line_count,
            source_max_initial_spine_count=source_max_initial_spine_count,
            spine_class=spine_class,
            target_bucket=target_bucket,
            target_center_line_count=bucket.center_line_count,
            in_target_range=bucket.contains(line_count),
            distance_to_bucket=distance,
            vertical_fit_penalty=compute_vertical_fit_penalty(
                line_count=line_count,
                vertical_fit_bucket=vertical_fit_bucket,
            ),
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


def distance_to_bucket(line_count: int, bucket: LineCountBucket) -> float:
    line_count = int(line_count)
    if bucket.contains(line_count):
        return abs(line_count - bucket.center_line_count) / max(
            1.0, bucket.max_line_count - bucket.min_line_count + 1.0
        )
    if line_count < bucket.min_line_count:
        return float(bucket.min_line_count - line_count)
    return float(line_count - bucket.max_line_count)


def compute_vertical_fit_penalty(
    *,
    line_count: int,
    vertical_fit_bucket: VerticalFitBucket | None,
) -> float:
    if vertical_fit_bucket is None:
        return 0.0
    safe_max_line_count = max(1, int(vertical_fit_bucket.safe_max_line_count))
    if int(line_count) <= safe_max_line_count:
        return 0.0
    return float(int(line_count) - safe_max_line_count) / float(safe_max_line_count)


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

    recommended_ranges: dict[str, dict[str, dict[str, int | float]]] = {}
    vertical_fit_model: dict[str, dict[str, dict[str, int | float]]] = {}
    systems_by_spine_class: dict[str, dict[str, dict[str, object]]] = {}

    for spine_class in SPINE_CLASS_ORDER:
        class_recommended_ranges: dict[str, dict[str, int | float]] = {}
        class_vertical_fit_model: dict[str, dict[str, int | float]] = {}
        per_system_stats: dict[str, dict[str, object]] = {}
        class_accepted_records = [
            record for record in accepted_records if _record_matches_spine_class(record, spine_class)
        ]
        class_calibration_records = [
            record for record in calibration_records if _record_matches_spine_class(record, spine_class)
        ]

        for bucket in TARGET_SYSTEM_BUCKETS:
            bucket_records = [
                int(record["source_non_empty_line_count"])
                for record in class_calibration_records
                if _calibration_bucket_system_count(record) == bucket
                and record.get("source_non_empty_line_count") is not None
            ]
            if not bucket_records:
                continue
            stats = summarize_lengths(bucket_records)
            class_recommended_ranges[str(bucket)] = {
                "min": int(stats["q20"]),
                "max": int(stats["q80"]),
                "center": float(stats["median"]),
            }
            per_system_stats[str(bucket)] = {
                "calibration_count": len(bucket_records),
                "accepted_count": sum(
                    1
                    for record in class_accepted_records
                    if _accepted_system_count(record) == bucket
                ),
                "full_render_count": sum(
                    1
                    for record in class_calibration_records
                    if _full_render_system_count(record) == bucket
                ),
                "line_count": stats,
            }

            safe_records = [
                record
                for record in class_calibration_records
                if _calibration_bucket_system_count(record) == bucket
                and _record_is_vertical_fit_safe(record)
            ]
            if not safe_records:
                continue
            safe_line_counts = [int(record["source_non_empty_line_count"]) for record in safe_records]
            safe_content_heights = [int(record["full_render_content_height_px"]) for record in safe_records]
            class_vertical_fit_model[str(bucket)] = {
                "safe_max_line_count": int(np.percentile(np.asarray(safe_line_counts, dtype=np.int32), 80)),
                "median_content_height_px": float(
                    np.percentile(np.asarray(safe_content_heights, dtype=np.int32), 50)
                ),
                "safe_sample_count": len(safe_records),
            }

        if class_recommended_ranges:
            recommended_ranges[spine_class] = class_recommended_ranges
        if class_vertical_fit_model:
            vertical_fit_model[spine_class] = class_vertical_fit_model
        if per_system_stats:
            systems_by_spine_class[spine_class] = per_system_stats

    return {
        "schema_version": "2.0",
        "mode": "spine_aware_line_proxy",
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
        "systems": systems_by_spine_class,
        "recommended_line_count_ranges": recommended_ranges,
        "vertical_fit_model": vertical_fit_model,
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
            "accepted_spine_class_histogram": histogram_by_key(
                accepted_records,
                key=lambda record: record.get("spine_class"),
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
    recommended_ranges: Mapping[str, Mapping[str, Mapping[str, int | float]]]
) -> dict[str, object]:
    overlaps_by_class: dict[str, list[dict[str, int | list[str]]]] = {}
    for spine_class, class_ranges in recommended_ranges.items():
        overlaps: list[dict[str, int | list[str]]] = []
        ordered = sorted((int(bucket), bucket_range) for bucket, bucket_range in class_ranges.items())
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
        overlaps_by_class[spine_class] = overlaps
    return {
        "bucket_overlaps_by_spine_class": overlaps_by_class,
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


def _candidate_sort_key(score: CandidatePlanScore) -> tuple[float, float, float, int]:
    center_distance = abs(score.line_count - score.target_center_line_count)
    return (
        score.distance_to_bucket,
        score.vertical_fit_penalty,
        center_distance,
        score.candidate_idx,
    )


def _has_line_count_bucket(spec: SystemBalanceSpec, target_bucket: int) -> bool:
    for spine_class in SPINE_CLASS_ORDER:
        class_ranges = spec.line_count_ranges.get(spine_class, {})
        if int(target_bucket) in class_ranges:
            return True
    return False


def _resolve_line_count_bucket(
    *,
    spec: SystemBalanceSpec,
    spine_class: str,
    target_bucket: int,
) -> LineCountBucket:
    class_ranges = spec.line_count_ranges.get(spine_class, {})
    if int(target_bucket) in class_ranges:
        return class_ranges[int(target_bucket)]
    fallback_ranges = spec.line_count_ranges.get("all", {})
    if int(target_bucket) in fallback_ranges:
        return fallback_ranges[int(target_bucket)]
    raise ValueError(
        f"System balance spec does not define line-count ranges for bucket {target_bucket} "
        f"under spine class {spine_class!r} or 'all'"
    )


def _resolve_vertical_fit_bucket(
    *,
    spec: SystemBalanceSpec,
    spine_class: str,
    target_bucket: int,
) -> VerticalFitBucket | None:
    class_fit_buckets = spec.vertical_fit_model.get(spine_class, {})
    if int(target_bucket) in class_fit_buckets:
        return class_fit_buckets[int(target_bucket)]
    fallback_fit_buckets = spec.vertical_fit_model.get("all", {})
    return fallback_fit_buckets.get(int(target_bucket))


def _record_spine_class(record: dict[str, object]) -> str:
    return spine_class_for_count(int(record.get("source_max_initial_spine_count", 1)))


def _record_matches_spine_class(record: dict[str, object], spine_class: str) -> bool:
    if spine_class == "all":
        return True
    return _record_spine_class(record) == spine_class


def _record_is_vertical_fit_safe(record: dict[str, object]) -> bool:
    if record.get("full_render_system_count") is None:
        return False
    if record.get("full_render_content_height_px") is None:
        return False
    rejection_reason = record.get("full_render_rejection_reason")
    return rejection_reason not in UNSAFE_VERTICAL_FIT_REJECTION_REASONS
