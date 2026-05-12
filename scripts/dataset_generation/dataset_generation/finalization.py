"""Run finalization helpers for the dataset-generation executor.

Splits the success-path and failure-path finalization ("commit trailing rows,
write ``info.json``, update the latest-run pointer, hand off to
``ResumableShardStore.finalize()``") out of ``executor.run_dataset_generation``
so the orchestration loop stays focused on scheduling and result collection.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.dataset_generation.dataset_generation.io import write_json
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.resume_store import (
    ResumableShardStore,
    write_run_info,
)
from scripts.dataset_generation.dataset_generation.run_context import RunContext


@dataclass(frozen=True)
class RunInfoBundle:
    """Parameters passed through to ``write_run_info`` in both finalization paths."""

    active_recipe: ProductionRecipe
    normalized_input_dirs: Sequence[Path]
    output_dir: str
    resolved_num_workers: int
    target_samples: int
    attempt_budget: int
    base_seed: int
    resume_mode: str
    failure_policy: dict[str, Any]
    capture_verovio_diagnostics: bool
    quarantine_in: str | Path | None
    quarantine_out_path: Path
    system_balance: dict[str, object]
    auto_quarantined_source_count: int
    invalid_source_examples: Sequence[dict[str, object]]


def finalize_run_success(
    *,
    run_context: RunContext,
    resume_store: ResumableShardStore,
    counters: dict[str, object],
    pending_rows: list[dict[str, object]],
    generation_seconds: float,
    layout_summary: dict[str, object],
    run_info: RunInfoBundle,
    snapshot_from_counters,
    write_quarantined_sources,
) -> None:
    if pending_rows:
        resume_store.commit(
            snapshot=snapshot_from_counters(counters), sample_rows=pending_rows
        )
        pending_rows.clear()
    resume_store.commit(snapshot=snapshot_from_counters(counters), sample_rows=[])
    finalization = resume_store.finalize()
    runtime_seconds = {
        "generation": generation_seconds,
        "total": generation_seconds,
    }
    write_quarantined_sources(
        quarantined_sources=counters["quarantined_sources"],  # type: ignore[arg-type]
        destination=run_info.quarantine_out_path,
    )
    write_run_info(
        run_context=run_context,
        recipe=run_info.active_recipe,
        input_dirs=tuple(str(path) for path in run_info.normalized_input_dirs),
        output_dir=run_info.output_dir,
        num_workers=run_info.resolved_num_workers,
        target_samples=run_info.target_samples,
        max_attempts=run_info.attempt_budget,
        base_seed=run_info.base_seed,
        resume_mode=run_info.resume_mode,
        failure_policy=run_info.failure_policy,
        capture_verovio_diagnostics=run_info.capture_verovio_diagnostics,
        quarantine_in=_resolve_quarantine_in(run_info.quarantine_in),
        quarantine_out=str(run_info.quarantine_out_path),
        system_balance=run_info.system_balance,
        runtime_seconds=runtime_seconds,
        layout_summary=layout_summary,
        snapshot=finalization["snapshot"],
        finalization=finalization,
        auto_quarantined_source_count=run_info.auto_quarantined_source_count,
        invalid_source_examples=run_info.invalid_source_examples,
    )
    write_json(
        run_context.latest_run_path,
        {
            "run_id": run_context.run_id,
            "run_artifacts_dir": str(run_context.run_artifacts_dir),
            "info_path": str(run_context.info_path),
            "updated_at": time.time(),
        },
    )


def finalize_run_failure(
    *,
    run_context: RunContext,
    resume_store: ResumableShardStore,
    counters: dict[str, object],
    pending_rows: list[dict[str, object]],
    layout_summary: dict[str, object],
    run_info: RunInfoBundle,
    snapshot_from_counters,
    write_quarantined_sources,
) -> None:
    if pending_rows:
        resume_store.commit(
            snapshot=snapshot_from_counters(counters), sample_rows=pending_rows
        )
    write_quarantined_sources(
        quarantined_sources=counters["quarantined_sources"],  # type: ignore[arg-type]
        destination=run_info.quarantine_out_path,
    )
    resume_store.mark_terminal_status(status="failed")
    write_run_info(
        run_context=run_context,
        recipe=run_info.active_recipe,
        input_dirs=tuple(str(path) for path in run_info.normalized_input_dirs),
        output_dir=run_info.output_dir,
        num_workers=run_info.resolved_num_workers,
        target_samples=run_info.target_samples,
        max_attempts=run_info.attempt_budget,
        base_seed=run_info.base_seed,
        resume_mode=run_info.resume_mode,
        failure_policy=run_info.failure_policy,
        capture_verovio_diagnostics=run_info.capture_verovio_diagnostics,
        quarantine_in=_resolve_quarantine_in(run_info.quarantine_in),
        quarantine_out=str(run_info.quarantine_out_path),
        system_balance=run_info.system_balance,
        layout_summary=layout_summary,
        snapshot=snapshot_from_counters(counters),
        auto_quarantined_source_count=run_info.auto_quarantined_source_count,
        invalid_source_examples=run_info.invalid_source_examples,
    )


def _resolve_quarantine_in(quarantine_in: str | Path | None) -> str | None:
    if quarantine_in is None:
        return None
    return str(Path(quarantine_in).expanduser().resolve())
