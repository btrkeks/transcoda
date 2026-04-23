"""Augmentation summary counters and preview artifacts.

Consolidates all augmentation-specific bookkeeping that was previously
scattered between ``executor._commit_contiguous_results`` (inline counter
updates), ``executor._maybe_write_augmentation_preview`` (preview I/O),
and ``outcome_events.update_augmentation_summary_counters``
(final-geometry / OOB / outer-gate breakdowns).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from scripts.dataset_generation.dataset_generation.io import write_json
from scripts.dataset_generation.dataset_generation.run_context import RunContext
from scripts.dataset_generation.dataset_generation.types_events import AugmentationTraceEvent
from scripts.dataset_generation.dataset_generation.types_outcomes import (
    WorkerFailure,
    WorkerSuccess,
)

_PREVIEW_CAP = 12


def update_summary_counters(
    *,
    counters: dict[str, object],
    outcome: WorkerSuccess,
) -> None:
    trace = outcome.augmentation_trace
    if trace is None:
        return

    aug_outcome_counts: Counter = counters["augmentation_outcome_counts"]  # type: ignore[assignment]
    aug_band_counts: Counter = counters["augmentation_band_counts"]  # type: ignore[assignment]
    aug_branch_counts: Counter = counters["augmentation_branch_counts"]  # type: ignore[assignment]
    aug_outcome_counts[trace.final_outcome] += 1
    aug_band_counts[trace.band] += 1
    aug_branch_counts[trace.branch] += 1
    _record_timing_histogram(counters, "augmentation_geom_ms_histogram", trace.offline_geom_ms)
    _record_timing_histogram(counters, "augmentation_gates_ms_histogram", trace.offline_gates_ms)
    _record_timing_histogram(
        counters,
        "augmentation_augraphy_ms_histogram",
        trace.offline_augraphy_ms,
    )
    _record_timing_histogram(
        counters,
        "augmentation_texture_ms_histogram",
        trace.offline_texture_ms,
    )

    _update_geometry_and_gate_counters(counters=counters, trace=trace)


def _record_timing_histogram(
    counters: dict[str, object],
    key: str,
    value_ms: float,
) -> None:
    histogram: Counter = counters[key]  # type: ignore[assignment]
    histogram[max(0, int(round(float(value_ms))))] += 1


def _update_geometry_and_gate_counters(
    *,
    counters: dict[str, object],
    trace: AugmentationTraceEvent,
) -> None:
    final_geometry_counts: Counter = counters["final_geometry_counts"]  # type: ignore[assignment]
    oob_failure_reason_counts: Counter = counters["oob_failure_reason_counts"]  # type: ignore[assignment]
    outer_gate_failure_reason_counts: Counter = counters[  # type: ignore[assignment]
        "outer_gate_failure_reason_counts"
    ]

    if not trace.outer_gate.passed:
        final_geometry_counts["base_image_returned"] += 1
    elif trace.final_geometry_applied:
        final_geometry_counts["geometry_survived"] += 1
    else:
        final_geometry_counts["geometry_discarded"] += 1

    if trace.initial_oob_gate.failure_reason is not None:
        oob_failure_reason_counts[trace.initial_oob_gate.failure_reason] += 1
    if trace.retry_oob_gate is not None and trace.retry_oob_gate.failure_reason is not None:
        oob_failure_reason_counts[trace.retry_oob_gate.failure_reason] += 1
    if trace.outer_gate.failure_reason is not None:
        outer_gate_failure_reason_counts[trace.outer_gate.failure_reason] += 1


def maybe_write_preview(
    *,
    run_context: RunContext,
    outcome: WorkerSuccess | WorkerFailure,
    counters: dict[str, object],
) -> None:
    if not isinstance(outcome, WorkerSuccess):
        return
    if outcome.augmentation_trace is None or outcome.augmentation_preview is None:
        return

    geometry_discarded_selected = False
    outer_gate_rejected_selected = False
    if (
        outcome.augmentation_trace.retry_geometry is not None
        and not outcome.augmentation_trace.final_geometry_applied
        and int(counters["augmentation_preview_geometry_discarded"]) < _PREVIEW_CAP
    ):
        counters["augmentation_preview_geometry_discarded"] = (
            int(counters["augmentation_preview_geometry_discarded"]) + 1
        )
        geometry_discarded_selected = True
    if (
        not outcome.augmentation_trace.outer_gate.passed
        and int(counters["augmentation_preview_outer_gate_rejected"]) < _PREVIEW_CAP
    ):
        counters["augmentation_preview_outer_gate_rejected"] = (
            int(counters["augmentation_preview_outer_gate_rejected"]) + 1
        )
        outer_gate_rejected_selected = True
    if not geometry_discarded_selected and not outer_gate_rejected_selected:
        return

    preview_dir = run_context.augmentation_previews_dir / outcome.sample.sample_id
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.joinpath("base.jpg").write_bytes(outcome.augmentation_preview.base_image_jpeg)
    preview_dir.joinpath("pre_augraphy.jpg").write_bytes(
        outcome.augmentation_preview.pre_augraphy_image_jpeg
    )
    preview_dir.joinpath("final.jpg").write_bytes(outcome.augmentation_preview.final_image_jpeg)
    write_json(preview_dir / "trace.json", asdict(outcome.augmentation_trace))
