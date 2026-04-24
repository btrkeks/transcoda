"""Progress counters, snapshots, and histogram summarization.

Houses the pure-data helpers that turn the live ``counters`` dict into
resume snapshots, progress payloads, and layout statistics. Previously
these lived alongside scheduling and queueing in ``executor.py``; splitting
them out keeps the orchestrator focused on pool/future/retry bookkeeping.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from scripts.dataset_generation.dataset_generation.io import write_json
from scripts.dataset_generation.dataset_generation.resume_store import (
    ResumableShardStore,
    RuntimeSnapshot,
)
from scripts.dataset_generation.dataset_generation.run_context import RunContext
from scripts.dataset_generation.dataset_generation.types_snapshots import ResumeSnapshot


def build_runtime_counters(snapshot: ResumeSnapshot | None) -> dict[str, object]:
    if snapshot is None:
        return {
            "next_sample_idx": 0,
            "accepted_samples": 0,
            "run_start_accepted_samples": 0,
            "rejected_samples": 0,
            "failure_reason_counts": Counter(),
            "truncation_counts": Counter({"attempted": 0, "rescued": 0, "failed": 0}),
            "full_render_system_histogram": Counter(),
            "accepted_system_histogram": defaultdict(Counter),
            "truncated_output_system_histogram": Counter(),
            "preferred_5_6_counts": Counter(
                {
                    "preferred_5_6_accepted_full": 0,
                    "preferred_5_6_rescued": 0,
                    "preferred_5_6_truncated": 0,
                    "preferred_5_6_failed": 0,
                }
            ),
            "bottom_whitespace_px_histogram": Counter(),
            "top_whitespace_px_histogram": Counter(),
            "content_height_px_histogram": Counter(),
            "terminal_timeout_crash_artifacts": 0,
            "terminal_process_expired_crash_artifacts": 0,
            "requested_target_bucket_histogram": Counter(),
            "target_full_render_system_histogram": defaultdict(Counter),
            "target_accepted_system_histogram": defaultdict(Counter),
            "target_failure_reason_counts": defaultdict(Counter),
            "candidate_hit_counts": Counter(),
            "retry_counts": Counter(),
            "quarantined_sources": set(),
            "quarantined_entry_ids": set(),
            "augmentation_outcome_counts": Counter(),
            "augmentation_band_counts": Counter(),
            "augmentation_branch_counts": Counter(),
            "final_geometry_counts": Counter(),
            "oob_failure_reason_counts": Counter(),
            "outer_gate_failure_reason_counts": Counter(),
            "augmentation_geom_ms_histogram": Counter(),
            "augmentation_gates_ms_histogram": Counter(),
            "augmentation_augraphy_ms_histogram": Counter(),
            "augmentation_texture_ms_histogram": Counter(),
            "augmentation_preview_geometry_discarded": 0,
            "augmentation_preview_outer_gate_rejected": 0,
        }
    return {
        "next_sample_idx": snapshot.next_sample_idx,
        "accepted_samples": snapshot.accepted_samples,
        "run_start_accepted_samples": snapshot.accepted_samples,
        "rejected_samples": snapshot.rejected_samples,
        "failure_reason_counts": Counter(snapshot.failure_reason_counts),
        "truncation_counts": Counter(snapshot.truncation_counts),
        "full_render_system_histogram": Counter(
            {int(key): int(value) for key, value in snapshot.full_render_system_histogram.items()}
        ),
        "accepted_system_histogram": defaultdict(
            Counter,
            {
                str(spine_cls): Counter(
                    {int(bucket): int(count) for bucket, count in bucket_counts.items()}
                )
                for spine_cls, bucket_counts in snapshot.accepted_system_histogram.items()
            },
        ),
        "truncated_output_system_histogram": Counter(
            {
                int(key): int(value)
                for key, value in snapshot.truncated_output_system_histogram.items()
            }
        ),
        "preferred_5_6_counts": Counter(snapshot.preferred_5_6_counts),
        "bottom_whitespace_px_histogram": Counter(
            {
                int(key): int(value)
                for key, value in snapshot.bottom_whitespace_px_histogram.items()
            }
        ),
        "top_whitespace_px_histogram": Counter(
            {
                int(key): int(value)
                for key, value in snapshot.top_whitespace_px_histogram.items()
            }
        ),
        "content_height_px_histogram": Counter(
            {
                int(key): int(value)
                for key, value in snapshot.content_height_px_histogram.items()
            }
        ),
        "terminal_timeout_crash_artifacts": int(snapshot.terminal_timeout_crash_artifacts),
        "terminal_process_expired_crash_artifacts": int(
            snapshot.terminal_process_expired_crash_artifacts
        ),
        "requested_target_bucket_histogram": Counter(
            {
                int(key): int(value)
                for key, value in snapshot.requested_target_bucket_histogram.items()
            }
        ),
        "target_full_render_system_histogram": defaultdict(
            Counter,
            {
                int(target_bucket): Counter(
                    {int(system_count): int(count) for system_count, count in bucket_counts.items()}
                )
                for target_bucket, bucket_counts in snapshot.target_full_render_system_histogram.items()
            },
        ),
        "target_accepted_system_histogram": defaultdict(
            Counter,
            {
                int(target_bucket): Counter(
                    {int(system_count): int(count) for system_count, count in bucket_counts.items()}
                )
                for target_bucket, bucket_counts in snapshot.target_accepted_system_histogram.items()
            },
        ),
        "target_failure_reason_counts": defaultdict(
            Counter,
            {
                int(target_bucket): Counter(
                    {str(reason): int(count) for reason, count in bucket_counts.items()}
                )
                for target_bucket, bucket_counts in snapshot.target_failure_reason_counts.items()
            },
        ),
        "candidate_hit_counts": Counter(snapshot.candidate_hit_counts),
        "retry_counts": Counter(snapshot.retry_counts),
        "quarantined_sources": {Path(path).expanduser().resolve() for path in snapshot.quarantined_sources},
        "quarantined_entry_ids": set(),
        "augmentation_outcome_counts": Counter(snapshot.augmentation_outcome_counts),
        "augmentation_band_counts": Counter(snapshot.augmentation_band_counts),
        "augmentation_branch_counts": Counter(snapshot.augmentation_branch_counts),
        "final_geometry_counts": Counter(snapshot.final_geometry_counts),
        "oob_failure_reason_counts": Counter(snapshot.oob_failure_reason_counts),
        "outer_gate_failure_reason_counts": Counter(snapshot.outer_gate_failure_reason_counts),
        "augmentation_geom_ms_histogram": Counter(snapshot.augmentation_geom_ms_histogram),
        "augmentation_gates_ms_histogram": Counter(snapshot.augmentation_gates_ms_histogram),
        "augmentation_augraphy_ms_histogram": Counter(snapshot.augmentation_augraphy_ms_histogram),
        "augmentation_texture_ms_histogram": Counter(snapshot.augmentation_texture_ms_histogram),
        "augmentation_preview_geometry_discarded": 0,
        "augmentation_preview_outer_gate_rejected": 0,
    }


def snapshot_from_counters(counters: dict[str, object]) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        next_sample_idx=int(counters["next_sample_idx"]),
        accepted_samples=int(counters["accepted_samples"]),
        rejected_samples=int(counters["rejected_samples"]),
        failure_reason_counts=dict(counters["failure_reason_counts"]),
        truncation_counts=dict(counters["truncation_counts"]),
        full_render_system_histogram={
            str(key): int(value)
            for key, value in dict(counters["full_render_system_histogram"]).items()
        },
        accepted_system_histogram={
            str(spine_cls): {
                str(bucket): int(count)
                for bucket, count in dict(bucket_counts).items()
            }
            for spine_cls, bucket_counts in dict(counters["accepted_system_histogram"]).items()
        },
        truncated_output_system_histogram={
            str(key): int(value)
            for key, value in dict(counters["truncated_output_system_histogram"]).items()
        },
        preferred_5_6_counts=dict(counters["preferred_5_6_counts"]),
        bottom_whitespace_px_histogram={
            str(key): int(value)
            for key, value in dict(counters["bottom_whitespace_px_histogram"]).items()
        },
        top_whitespace_px_histogram={
            str(key): int(value)
            for key, value in dict(counters["top_whitespace_px_histogram"]).items()
        },
        content_height_px_histogram={
            str(key): int(value)
            for key, value in dict(counters["content_height_px_histogram"]).items()
        },
        terminal_timeout_crash_artifacts=int(counters["terminal_timeout_crash_artifacts"]),
        terminal_process_expired_crash_artifacts=int(
            counters["terminal_process_expired_crash_artifacts"]
        ),
        requested_target_bucket_histogram={
            str(key): int(value)
            for key, value in dict(counters["requested_target_bucket_histogram"]).items()
        },
        target_full_render_system_histogram={
            str(target_bucket): {
                str(system_count): int(count)
                for system_count, count in dict(system_counts).items()
            }
            for target_bucket, system_counts in dict(counters["target_full_render_system_histogram"]).items()
        },
        target_accepted_system_histogram={
            str(target_bucket): {
                str(system_count): int(count)
                for system_count, count in dict(system_counts).items()
            }
            for target_bucket, system_counts in dict(counters["target_accepted_system_histogram"]).items()
        },
        target_failure_reason_counts={
            str(target_bucket): {
                str(reason): int(count)
                for reason, count in dict(reason_counts).items()
            }
            for target_bucket, reason_counts in dict(counters["target_failure_reason_counts"]).items()
        },
        candidate_hit_counts=dict(counters["candidate_hit_counts"]),
        retry_counts=dict(counters["retry_counts"]),
        quarantined_sources=tuple(
            sorted(str(path) for path in counters["quarantined_sources"])  # type: ignore[arg-type]
        ),
        augmentation_outcome_counts=dict(counters["augmentation_outcome_counts"]),
        augmentation_band_counts=dict(counters["augmentation_band_counts"]),
        augmentation_branch_counts=dict(counters["augmentation_branch_counts"]),
        final_geometry_counts=dict(counters["final_geometry_counts"]),
        oob_failure_reason_counts=dict(counters["oob_failure_reason_counts"]),
        outer_gate_failure_reason_counts=dict(counters["outer_gate_failure_reason_counts"]),
        augmentation_geom_ms_histogram={
            str(key): int(value)
            for key, value in dict(counters["augmentation_geom_ms_histogram"]).items()
        },
        augmentation_gates_ms_histogram={
            str(key): int(value)
            for key, value in dict(counters["augmentation_gates_ms_histogram"]).items()
        },
        augmentation_augraphy_ms_histogram={
            str(key): int(value)
            for key, value in dict(counters["augmentation_augraphy_ms_histogram"]).items()
        },
        augmentation_texture_ms_histogram={
            str(key): int(value)
            for key, value in dict(counters["augmentation_texture_ms_histogram"]).items()
        },
    )


def build_layout_summary(counters: dict[str, object]) -> dict[str, object]:
    bottom_histogram: Counter = counters["bottom_whitespace_px_histogram"]  # type: ignore[assignment]
    top_histogram: Counter = counters["top_whitespace_px_histogram"]  # type: ignore[assignment]
    content_histogram: Counter = counters["content_height_px_histogram"]  # type: ignore[assignment]
    page_height = 1485.0
    return {
        "layout_summary_version": 1,
        "layout_summary_population": "accepted_samples",
        "accepted_samples_total": int(counters["accepted_samples"]),
        "bottom_whitespace_px_stats": _summarize_histogram(bottom_histogram),
        "top_whitespace_px_stats": _summarize_histogram(top_histogram),
        "content_height_px_stats": _summarize_histogram(content_histogram),
        "bottom_whitespace_ratio_stats": _summarize_scaled_histogram(
            bottom_histogram,
            scale=page_height,
        ),
        "vertical_fill_ratio_stats": _summarize_scaled_histogram(
            content_histogram,
            scale=page_height,
        ),
    }


def maybe_flush_and_report(
    *,
    resume_store: ResumableShardStore,
    run_context: RunContext,
    counters: dict[str, object],
    pending_rows: list[dict[str, object]],
    active_workers: int,
    target_samples: int,
    last_progress_at_ref: list[float],
    progress_interval_seconds: float,
    generation_start_perf_counter: float,
    quiet: bool,
    quarantine_out_path: Path,
    write_quarantined_sources: Callable,
) -> None:
    now = time.time()
    elapsed_seconds = max(0.0, time.perf_counter() - generation_start_perf_counter)
    if pending_rows:
        rows_to_commit = list(pending_rows)
        pending_rows.clear()
        resume_store.commit(
            snapshot=snapshot_from_counters(counters), sample_rows=rows_to_commit
        )
    if (
        now - last_progress_at_ref[0] >= progress_interval_seconds
        or int(counters["accepted_samples"]) >= target_samples
    ):
        progress_payload = {
            "attempted_samples": int(counters["next_sample_idx"]),
            "accepted_samples": int(counters["accepted_samples"]),
            "rejected_samples": int(counters["rejected_samples"]),
            "active_workers": active_workers,
            "failure_reason_counts": dict(counters["failure_reason_counts"]),
            "truncation_counts": dict(counters["truncation_counts"]),
            "full_render_system_histogram": {
                str(key): int(value)
                for key, value in dict(counters["full_render_system_histogram"]).items()
            },
            "accepted_system_histogram": {
                str(spine_cls): {
                    str(bucket): int(count)
                    for bucket, count in dict(bucket_counts).items()
                }
                for spine_cls, bucket_counts in dict(counters["accepted_system_histogram"]).items()
            },
            "truncated_output_system_histogram": {
                str(key): int(value)
                for key, value in dict(counters["truncated_output_system_histogram"]).items()
            },
            "preferred_5_6_counts": dict(counters["preferred_5_6_counts"]),
            "requested_target_bucket_histogram": {
                str(key): int(value)
                for key, value in dict(counters["requested_target_bucket_histogram"]).items()
            },
            "target_full_render_system_histogram": {
                str(target_bucket): {
                    str(system_count): int(count)
                    for system_count, count in dict(system_counts).items()
                }
                for target_bucket, system_counts in dict(
                    counters["target_full_render_system_histogram"]
                ).items()
            },
            "target_accepted_system_histogram": {
                str(target_bucket): {
                    str(system_count): int(count)
                    for system_count, count in dict(system_counts).items()
                }
                for target_bucket, system_counts in dict(
                    counters["target_accepted_system_histogram"]
                ).items()
            },
            "target_failure_reason_counts": {
                str(target_bucket): {
                    str(reason): int(count)
                    for reason, count in dict(reason_counts).items()
                }
                for target_bucket, reason_counts in dict(
                    counters["target_failure_reason_counts"]
                ).items()
            },
            "candidate_hit_counts": dict(counters["candidate_hit_counts"]),
            "retry_counts": dict(counters["retry_counts"]),
            "quarantined_source_count": len(counters["quarantined_sources"]),
            "terminal_timeout_crash_artifacts": int(counters["terminal_timeout_crash_artifacts"]),
            "terminal_process_expired_crash_artifacts": int(
                counters["terminal_process_expired_crash_artifacts"]
            ),
            "augmentation_outcome_counts": dict(counters["augmentation_outcome_counts"]),
            "augmentation_band_counts": dict(counters["augmentation_band_counts"]),
            "augmentation_branch_counts": dict(counters["augmentation_branch_counts"]),
            "final_geometry_counts": dict(counters["final_geometry_counts"]),
            "oob_failure_reason_counts": dict(counters["oob_failure_reason_counts"]),
            "outer_gate_failure_reason_counts": dict(counters["outer_gate_failure_reason_counts"]),
            "augmentation_geom_ms_histogram": {
                str(key): int(value)
                for key, value in dict(counters["augmentation_geom_ms_histogram"]).items()
            },
            "augmentation_gates_ms_histogram": {
                str(key): int(value)
                for key, value in dict(counters["augmentation_gates_ms_histogram"]).items()
            },
            "augmentation_augraphy_ms_histogram": {
                str(key): int(value)
                for key, value in dict(counters["augmentation_augraphy_ms_histogram"]).items()
            },
            "augmentation_texture_ms_histogram": {
                str(key): int(value)
                for key, value in dict(counters["augmentation_texture_ms_histogram"]).items()
            },
        }
        progress_payload.update(build_layout_summary(counters))
        progress_payload.update(
            _build_timing_summary(
                counters=counters,
                target_samples=target_samples,
                elapsed_seconds=elapsed_seconds,
                now=now,
            )
        )
        write_json(run_context.progress_path, progress_payload)
        write_quarantined_sources(
            quarantined_sources=counters["quarantined_sources"],  # type: ignore[arg-type]
            destination=quarantine_out_path,
        )
        last_progress_at_ref[0] = now
        if not quiet:
            print(
                "Progress: "
                f"{progress_payload['accepted_samples']}/{target_samples} accepted, "
                f"{progress_payload['attempted_samples']} attempted, "
                f"elapsed {_format_duration(progress_payload['elapsed_seconds'])}, "
                f"ETA {_format_optional_duration(progress_payload['eta_seconds'])}"
            )


def _build_timing_summary(
    *,
    counters: dict[str, object],
    target_samples: int,
    elapsed_seconds: float,
    now: float,
) -> dict[str, float | int | None]:
    accepted_samples = int(counters["accepted_samples"])
    run_start_accepted_samples = int(counters.get("run_start_accepted_samples", 0))
    accepted_samples_this_run = max(0, accepted_samples - run_start_accepted_samples)
    remaining_samples = max(0, int(target_samples) - accepted_samples)
    accepted_samples_per_second = (
        float(accepted_samples_this_run) / elapsed_seconds
        if accepted_samples_this_run > 0 and elapsed_seconds > 0
        else 0.0
    )
    eta_seconds: float | None = None
    estimated_completion_at: float | None = None
    if accepted_samples_per_second > 0.0:
        eta_seconds = float(remaining_samples) / accepted_samples_per_second
        estimated_completion_at = now + eta_seconds
    return {
        "elapsed_seconds": elapsed_seconds,
        "accepted_samples_per_second": accepted_samples_per_second,
        "accepted_samples_at_run_start": run_start_accepted_samples,
        "accepted_samples_this_run": accepted_samples_this_run,
        "accepted_samples_total": accepted_samples,
        "remaining_samples": remaining_samples,
        "eta_seconds": eta_seconds,
        "estimated_completion_at": estimated_completion_at,
    }


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_optional_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    return _format_duration(seconds)


def _summarize_histogram(histogram: Counter) -> dict[str, float | int]:
    if not histogram:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    total = int(sum(int(count) for count in histogram.values()))
    weighted_sum = float(sum(int(value) * int(count) for value, count in histogram.items()))
    sorted_items = sorted((int(value), int(count)) for value, count in histogram.items())
    return {
        "count": total,
        "mean": weighted_sum / total,
        "min": float(sorted_items[0][0]),
        "max": float(sorted_items[-1][0]),
        "p50": _percentile_from_histogram(sorted_items, total, 50.0),
        "p95": _percentile_from_histogram(sorted_items, total, 95.0),
        "p99": _percentile_from_histogram(sorted_items, total, 99.0),
    }


def _summarize_scaled_histogram(histogram: Counter, *, scale: float) -> dict[str, float | int]:
    raw = _summarize_histogram(histogram)
    if raw["count"] == 0:
        return raw
    return {
        "count": raw["count"],
        "mean": float(raw["mean"]) / scale,
        "min": float(raw["min"]) / scale,
        "max": float(raw["max"]) / scale,
        "p50": float(raw["p50"]) / scale,
        "p95": float(raw["p95"]) / scale,
        "p99": float(raw["p99"]) / scale,
    }


def _percentile_from_histogram(
    sorted_items: list[tuple[int, int]],
    total_count: int,
    percentile: float,
) -> float:
    if total_count <= 0:
        return 0.0
    rank = max(1, int(round((percentile / 100.0) * total_count)))
    cumulative = 0
    for value, count in sorted_items:
        cumulative += count
        if cumulative >= rank:
            return float(value)
    return float(sorted_items[-1][0])
