"""Crash-artifact helpers for terminal worker failures."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.render_transcription import (
    build_render_transcription,
)
from scripts.dataset_generation.dataset_generation.run_context import RunContext
from scripts.dataset_generation.dataset_generation.truncation import build_prefix_candidates
from scripts.dataset_generation.dataset_generation.types import SamplePlan


def write_crash_artifact(
    *,
    run_context: RunContext,
    recipe: ProductionRecipe,
    plan: SamplePlan,
    sample_idx: int,
    event_type: str,
    retry_count: int,
    will_retry: bool,
    queue_wait_ms: float,
    dropped_pending_tasks: int,
    target_bucket: int | None,
    planned_token_length: int | None,
    candidate_in_target_range: bool | None,
    exception_payload: dict[str, object] | None,
) -> tuple[Path, int]:
    crash_dir = run_context.crash_samples_dir / plan.sample_id
    crash_dir.mkdir(parents=True, exist_ok=True)

    repro_entries: list[dict[str, object]] = []

    full_render_path = crash_dir / f"{sample_idx:08d}_full.krn"
    full_render_path.write_text(
        build_render_transcription(plan.label_transcription, recipe, seed=plan.seed) + "\n",
        encoding="utf-8",
    )
    repro_entries.append(
        {
            "stage": "full",
            "seed": plan.seed,
            "render_transcription_path": str(full_render_path),
        }
    )

    for idx, candidate in enumerate(build_prefix_candidates(plan.label_transcription, recipe), start=1):
        candidate_seed = (plan.seed + candidate.chunk_count * 17) & 0xFFFFFFFF
        candidate_path = crash_dir / f"{sample_idx:08d}_truncation_{idx:02d}.krn"
        candidate_path.write_text(
            build_render_transcription(candidate.transcription, recipe, seed=candidate_seed) + "\n",
            encoding="utf-8",
        )
        repro_entries.append(
            {
                "stage": "truncation_candidate",
                "seed": candidate_seed,
                "chunk_count": candidate.chunk_count,
                "total_chunks": candidate.total_chunks,
                "ratio": candidate.ratio,
                "render_transcription_path": str(candidate_path),
            }
        )

    artifact = {
        "event_type": event_type,
        "sample_id": plan.sample_id,
        "sample_idx": sample_idx,
        "retry_count": retry_count,
        "will_retry": will_retry,
        "queue_wait_ms": queue_wait_ms,
        "dropped_pending_tasks": dropped_pending_tasks,
        "source_paths": [str(segment.path.resolve()) for segment in plan.segments],
        "target_bucket": target_bucket,
        "planned_token_length": planned_token_length,
        "candidate_in_target_range": candidate_in_target_range,
        "stage_unknown_due_to_timeout": event_type == "timeout",
        "exception": exception_payload,
        "repro_entries": repro_entries,
    }
    artifact_path = crash_dir / f"{sample_idx:08d}_{event_type}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact_path, len(repro_entries)
