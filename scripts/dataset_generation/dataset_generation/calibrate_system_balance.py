"""CLI for token-length system-balance calibration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, TimeoutError, wait
from pathlib import Path
from typing import Callable

from pebble import ProcessExpired, ProcessPool

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
)
from scripts.dataset_generation.dataset_generation.failure_policy import resolve_failure_policy
from scripts.dataset_generation.dataset_generation.io import write_json
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.source_index import build_source_index
from scripts.dataset_generation.dataset_generation.system_balance import (
    DEFAULT_CANDIDATE_PLAN_COUNT,
    build_calibration_artifact,
    compute_token_length,
    load_tokenizer,
)
from scripts.dataset_generation.dataset_generation.types import SamplePlan, WorkerFailure, WorkerSuccess
from scripts.dataset_generation.dataset_generation.worker import (
    evaluate_sample_plan,
    init_generation_worker,
    process_sample_plan,
)


def run_calibration(
    input_dirs: str | Path | tuple[str | Path, ...] | list[str | Path],
    *,
    tokenizer_path: str | Path = "vocab/bpe3k-splitspaces",
    sample_budget: int = 3000,
    output_json: str | Path | None = None,
    base_seed: int = 0,
    num_workers: int = 1,
    quiet: bool = False,
    recipe: ProductionRecipe | None = None,
    renderer: VerovioRenderer | None = None,
    render_fn: Callable[..., object] | None = None,
    augment_fn: Callable[..., object] | None = None,
) -> dict[str, object]:
    normalized_input_dirs = _normalize_input_dirs(input_dirs)
    active_recipe = recipe or ProductionRecipe()
    source_index = build_source_index(*normalized_input_dirs)
    entry_by_path = {entry.path.resolve(): entry for entry in source_index.entries}
    tokenizer_dir, tokenizer = load_tokenizer(tokenizer_path)
    destination = _resolve_output_json(output_json)
    failure_policy = resolve_failure_policy("balanced")

    records: list[dict[str, object]] = []
    use_process_pool = (
        int(num_workers) > 1
        and render_fn is None
        and augment_fn is None
        and renderer is None
    )

    if not quiet:
        print(
            f"Calibrating system balance from {len(source_index.entries)} sources "
            f"across {len(normalized_input_dirs)} root(s)"
        )

    if use_process_pool:
        max_in_flight = max(int(num_workers) * 4, int(num_workers))
        next_sample_idx = 0
        futures_by_handle: dict[object, tuple[int, SamplePlan, int, list[str]]] = {}
        with ProcessPool(
            max_workers=max(1, int(num_workers)),
            max_tasks=200,
            initializer=init_generation_worker,
            initargs=(active_recipe,),
        ) as pool:
            while next_sample_idx < sample_budget or futures_by_handle:
                while next_sample_idx < sample_budget and len(futures_by_handle) < max_in_flight:
                    plan = _plan_sample(source_index, active_recipe, next_sample_idx, base_seed)
                    token_length = compute_token_length(plan.label_transcription, tokenizer)
                    root_labels = _source_root_labels(plan, entry_by_path)
                    future = pool.schedule(
                        process_sample_plan,
                        args=(plan,),
                        timeout=failure_policy.task_timeout_seconds,
                    )
                    futures_by_handle[future] = (next_sample_idx, plan, token_length, root_labels)
                    next_sample_idx += 1

                done, _ = wait(tuple(futures_by_handle), timeout=0.5, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    sample_idx, plan, token_length, root_labels = futures_by_handle.pop(future)
                    try:
                        outcome = future.result()
                    except TimeoutError:
                        records.append(
                            _build_record(
                                sample_idx=sample_idx,
                                plan=plan,
                                token_length=token_length,
                                source_root_labels=root_labels,
                                accepted=False,
                                failure_reason="timeout",
                                full_render_system_count=None,
                                accepted_render_system_count=None,
                                truncation_applied=False,
                                preferred_5_6_status=None,
                            )
                        )
                    except ProcessExpired:
                        records.append(
                            _build_record(
                                sample_idx=sample_idx,
                                plan=plan,
                                token_length=token_length,
                                source_root_labels=root_labels,
                                accepted=False,
                                failure_reason="process_expired",
                                full_render_system_count=None,
                                accepted_render_system_count=None,
                                truncation_applied=False,
                                preferred_5_6_status=None,
                            )
                        )
                    except BaseException as exc:
                        records.append(
                            _build_record(
                                sample_idx=sample_idx,
                                plan=plan,
                                token_length=token_length,
                                source_root_labels=root_labels,
                                accepted=False,
                                failure_reason=f"task_error:{type(exc).__name__}",
                                full_render_system_count=None,
                                accepted_render_system_count=None,
                                truncation_applied=False,
                                preferred_5_6_status=None,
                            )
                        )
                    else:
                        records.append(
                            _record_from_outcome(
                                sample_idx=sample_idx,
                                plan=plan,
                                token_length=token_length,
                                source_root_labels=root_labels,
                                outcome=outcome,
                            )
                        )
    else:
        sync_renderer = renderer or VerovioRenderer()
        for sample_idx in range(sample_budget):
            plan = _plan_sample(source_index, active_recipe, sample_idx, base_seed)
            token_length = compute_token_length(plan.label_transcription, tokenizer)
            root_labels = _source_root_labels(plan, entry_by_path)
            outcome = evaluate_sample_plan(
                plan,
                recipe=active_recipe,
                renderer=sync_renderer,
                **{
                    key: value
                    for key, value in (
                        ("render_fn", render_fn),
                        ("augment_fn", augment_fn),
                    )
                    if value is not None
                },
            )
            records.append(
                _record_from_outcome(
                    sample_idx=sample_idx,
                    plan=plan,
                    token_length=token_length,
                    source_root_labels=root_labels,
                    outcome=outcome,
                )
            )

    artifact = build_calibration_artifact(
        records=records,
        tokenizer_dir=tokenizer_dir,
        recipe=active_recipe,
        input_dirs=normalized_input_dirs,
        sample_budget=sample_budget,
        candidate_plan_count=DEFAULT_CANDIDATE_PLAN_COUNT,
    )
    write_json(destination, artifact)
    summary = {
        "output_json": destination,
        "attempted_samples": len(records),
        "accepted_samples": sum(1 for record in records if record["accepted"]),
        "tokenizer_path": tokenizer_dir,
    }
    if not quiet:
        print(json.dumps(summary, default=_json_default, sort_keys=True))
    return summary


def main(
    *input_dirs: str,
    tokenizer_path: str = "vocab/bpe3k-splitspaces",
    sample_budget: int = 3000,
    output_json: str | None = None,
    base_seed: int = 0,
    num_workers: int = 1,
    quiet: bool = False,
) -> dict[str, object]:
    return run_calibration(
        input_dirs=input_dirs,
        tokenizer_path=tokenizer_path,
        sample_budget=sample_budget,
        output_json=output_json,
        base_seed=base_seed,
        num_workers=num_workers,
        quiet=quiet,
    )


def _plan_sample(source_index, recipe, sample_idx: int, base_seed: int) -> SamplePlan:
    from scripts.dataset_generation.dataset_generation.composer import plan_sample

    return plan_sample(source_index, recipe, sample_idx=sample_idx, base_seed=base_seed)


def _source_root_labels(plan: SamplePlan, entry_by_path: dict[Path, object]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for segment in plan.segments:
        entry = entry_by_path[segment.path.resolve()]
        root_label = getattr(entry, "root_label")
        if root_label in seen:
            continue
        seen.add(root_label)
        labels.append(root_label)
    return labels


def _record_from_outcome(
    *,
    sample_idx: int,
    plan: SamplePlan,
    token_length: int,
    source_root_labels: list[str],
    outcome,
) -> dict[str, object]:
    if isinstance(outcome, WorkerSuccess):
        return _build_record(
            sample_idx=sample_idx,
            plan=plan,
            token_length=token_length,
            source_root_labels=source_root_labels,
            accepted=True,
            failure_reason=None,
            full_render_system_count=outcome.full_render_system_count,
            accepted_render_system_count=outcome.accepted_render_system_count,
            truncation_applied=outcome.sample.truncation_applied,
            preferred_5_6_status=outcome.preferred_5_6_status,
        )
    assert isinstance(outcome, WorkerFailure)
    return _build_record(
        sample_idx=sample_idx,
        plan=plan,
        token_length=token_length,
        source_root_labels=source_root_labels,
        accepted=False,
        failure_reason=outcome.failure_reason,
        full_render_system_count=outcome.full_render_system_count,
        accepted_render_system_count=outcome.accepted_render_system_count,
        truncation_applied=False,
        preferred_5_6_status=outcome.preferred_5_6_status,
    )


def _build_record(
    *,
    sample_idx: int,
    plan: SamplePlan,
    token_length: int,
    source_root_labels: list[str],
    accepted: bool,
    failure_reason: str | None,
    full_render_system_count: int | None,
    accepted_render_system_count: int | None,
    truncation_applied: bool,
    preferred_5_6_status: str | None,
) -> dict[str, object]:
    return {
        "sample_idx": int(sample_idx),
        "sample_id": plan.sample_id,
        "token_length": int(token_length),
        "full_render_system_count": (
            int(full_render_system_count) if full_render_system_count is not None else None
        ),
        "accepted_render_system_count": (
            int(accepted_render_system_count) if accepted_render_system_count is not None else None
        ),
        "accepted": bool(accepted),
        "failure_reason": failure_reason,
        "truncation_applied": bool(truncation_applied),
        "preferred_5_6_status": preferred_5_6_status,
        "segment_count": int(plan.segment_count),
        "source_root_labels": list(source_root_labels),
    }


def _normalize_input_dirs(
    input_dirs: str | Path | tuple[str | Path, ...] | list[str | Path],
) -> tuple[Path, ...]:
    if isinstance(input_dirs, (str, Path)):
        values = (input_dirs,)
    else:
        values = tuple(input_dirs)
    if not values:
        raise ValueError("At least one input directory is required")
    return tuple(Path(value).expanduser().resolve() for value in values)


def _resolve_output_json(output_json: str | Path | None) -> Path:
    if output_json is not None:
        return Path(output_json).expanduser().resolve()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return (
        Path("reports/dataset_generation/system_balance").expanduser().resolve()
        / f"{timestamp}.json"
    )


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate token-length buckets for soft system balancing.",
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="One or more normalized input directories containing .krn files.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="vocab/bpe3k-splitspaces",
        help="Tokenizer directory used for token-length measurement.",
    )
    parser.add_argument(
        "--sample_budget",
        type=int,
        default=3000,
        help="Number of planned samples to render for calibration.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional destination JSON path for the calibration artifact.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Base RNG seed used for deterministic planning.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes to use during calibration.",
    )
    parser.add_argument(
        "--quiet",
        type=_parse_bool,
        default=False,
        help="Whether to suppress calibration progress logging (true/false).",
    )
    args = parser.parse_args(argv)
    summary = main(
        *args.input_dirs,
        tokenizer_path=args.tokenizer_path,
        sample_budget=args.sample_budget,
        output_json=args.output_json,
        base_seed=args.base_seed,
        num_workers=args.num_workers,
        quiet=args.quiet,
    )
    print(json.dumps(summary, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
