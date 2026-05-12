"""Profiling helpers for benchmark inference runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _safe_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    index = min(len(sorted_values) - 1, round(0.95 * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def _summarize_numeric(values: list[float]) -> dict[str, float]:
    normalized = [float(value) for value in values]
    return {
        "mean": _safe_mean(normalized),
        "median": _safe_median(normalized),
        "p95": _safe_p95(normalized),
        "total": float(sum(normalized)),
    }


def _share(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    return float((part / whole) * 100.0)


@dataclass(frozen=True)
class BenchmarkProfileConfig:
    enabled: bool = False
    warmup_batches: int = 2
    max_batches: int | None = None
    trace_enabled: bool = False


class InferenceProfileRecorder:
    """Collect per-batch profiling records and write aggregate artifacts."""

    def __init__(
        self,
        *,
        enabled: bool,
        run_dir: Path,
        output_root: Path,
        dataset_name: str,
        model_name: str,
        config: BenchmarkProfileConfig,
        metadata: dict[str, Any],
    ) -> None:
        self.enabled = bool(enabled)
        self.run_dir = run_dir
        self.output_root = output_root
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.config = config
        self.metadata = dict(metadata)
        self.profile_dir = self.run_dir / dataset_name / model_name / "profile"
        self._rows: list[dict[str, Any]] = []
        self._collected_batches = 0
        self._trace_exported = False

    def plan_batch(self, batch_index: int) -> dict[str, Any]:
        if not self.enabled:
            return {"collect": False, "trace_path": None}

        after_warmup = batch_index >= max(0, int(self.config.warmup_batches))
        under_limit = self.config.max_batches is None or self._collected_batches < int(self.config.max_batches)
        collect = bool(after_warmup and under_limit)
        trace_path: Path | None = None
        if collect and self.config.trace_enabled and not self._trace_exported:
            trace_path = self.profile_dir / f"trace_batch_{batch_index:04d}.json"
            self._trace_exported = True
        return {
            "collect": collect,
            "trace_path": trace_path,
        }

    def record_batch(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._rows.append(row)
        self._collected_batches += 1

    def write_outputs(self) -> dict[str, Any] | None:
        if not self.enabled or not self._rows:
            return None

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        batches_path = self.profile_dir / "profile_batches.jsonl"
        with batches_path.open("w", encoding="utf-8") as handle:
            for row in self._rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = self._build_summary()
        counterpart = self._load_counterpart_summary()
        if counterpart is not None:
            summary["comparison"] = self._build_comparison(summary, counterpart)
        else:
            summary["comparison"] = None
        summary["conclusion"] = self._build_conclusion(summary)

        summary_path = self.profile_dir / "profile_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _build_summary(self) -> dict[str, Any]:
        rows = list(self._rows)
        samples = [int(row["num_samples"]) for row in rows]
        batch_latency_ms = [float(row["batch_wall_ms"]) for row in rows]
        generated_tokens = [int(row["generated_tokens"]) for row in rows]
        generated_tokens_per_sample = [
            (float(row["generated_tokens"]) / max(1, int(row["num_samples"]))) for row in rows
        ]

        preprocess_ms = [float(row["adapter_profile"]["preprocess_ms"]) for row in rows]
        setup_ms = [float(row["adapter_profile"]["tensor_setup_ms"]) for row in rows]
        bundle_ms = [float(row["adapter_profile"]["constraint_bundle_ms"]) for row in rows]
        generate_ms = [float(row["adapter_profile"]["generate_ms"]) for row in rows]
        finalize_ms = [float(row["adapter_profile"]["finalize_ms"]) for row in rows]

        grammar_total_ms = [
            float(row["adapter_profile"].get("constraint_stats", {}).get("grammar", {}).get("total_ms", 0.0))
            for row in rows
        ]
        semantic_total_ms = [
            float(row["adapter_profile"].get("constraint_stats", {}).get("semantic", {}).get("total_ms", 0.0))
            for row in rows
        ]

        total_latency_ms = float(sum(batch_latency_ms))
        total_generate_ms = float(sum(generate_ms))
        total_samples = int(sum(samples))
        total_tokens = int(sum(generated_tokens))

        trace_files = [
            row.get("trace_path")
            for row in rows
            if row.get("trace_path")
        ]

        summary = {
            "metadata": {
                **self.metadata,
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "warmup_batches": int(self.config.warmup_batches),
                "profile_max_batches": self.config.max_batches,
                "profile_trace_enabled": bool(self.config.trace_enabled),
                "profiled_batches": len(rows),
                "trace_files": trace_files,
            },
            "totals": {
                "samples": total_samples,
                "generated_tokens": total_tokens,
                "batch_wall_ms": total_latency_ms,
                "generate_ms": total_generate_ms,
            },
            "throughput": {
                "samples_per_second": (total_samples / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0,
                "tokens_per_second": (total_tokens / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0,
                "generate_tokens_per_second": (
                    total_tokens / (total_generate_ms / 1000.0)
                ) if total_generate_ms > 0 else 0.0,
            },
            "batch_latency_ms": _summarize_numeric(batch_latency_ms),
            "generated_tokens_per_sample": _summarize_numeric(generated_tokens_per_sample),
            "stages_ms": {
                "preprocess": _summarize_numeric(preprocess_ms),
                "tensor_setup": _summarize_numeric(setup_ms),
                "constraint_bundle": _summarize_numeric(bundle_ms),
                "generate": _summarize_numeric(generate_ms),
                "finalize": _summarize_numeric(finalize_ms),
            },
            "time_shares_pct": {
                "preprocess": _share(sum(preprocess_ms), total_latency_ms),
                "tensor_setup": _share(sum(setup_ms), total_latency_ms),
                "constraint_bundle": _share(sum(bundle_ms), total_latency_ms),
                "generate": _share(sum(generate_ms), total_latency_ms),
                "finalize": _share(sum(finalize_ms), total_latency_ms),
                "grammar_within_generate": _share(sum(grammar_total_ms), total_generate_ms),
                "semantic_within_generate": _share(sum(semantic_total_ms), total_generate_ms),
            },
            "constraint_processors": self._summarize_constraint_processors(rows),
        }
        return summary

    def _summarize_constraint_processors(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        def gather(path: tuple[str, ...]) -> list[float]:
            values: list[float] = []
            for row in rows:
                node: Any = row.get("adapter_profile", {}).get("constraint_stats", {})
                for key in path:
                    if not isinstance(node, dict):
                        node = None
                        break
                    node = node.get(key)
                if node is None:
                    continue
                values.append(float(node))
            return values

        grammar = {
            "total_ms": _summarize_numeric(gather(("grammar", "total_ms"))),
            "matcher_state_advance_ms": _summarize_numeric(gather(("grammar", "matcher_state_advance_ms"))),
            "bitmask_fill_ms": _summarize_numeric(gather(("grammar", "bitmask_fill_ms"))),
            "bitmask_apply_ms": _summarize_numeric(gather(("grammar", "bitmask_apply_ms"))),
            "calls": _summarize_numeric(gather(("grammar", "calls"))),
            "rows_processed": _summarize_numeric(gather(("grammar", "rows_processed"))),
            "externally_finished_rows": _summarize_numeric(gather(("grammar", "externally_finished_rows"))),
        }
        semantic = {
            "total_ms": _summarize_numeric(gather(("semantic", "total_ms"))),
            "advance_row_ms": _summarize_numeric(gather(("semantic", "advance_row_ms"))),
            "mask_row_ms": _summarize_numeric(gather(("semantic", "mask_row_ms"))),
            "calls": _summarize_numeric(gather(("semantic", "calls"))),
            "rows_processed": _summarize_numeric(gather(("semantic", "rows_processed"))),
            "inactive_rows": _summarize_numeric(gather(("semantic", "inactive_rows"))),
            "terminated_rows": _summarize_numeric(gather(("semantic", "terminated_rows"))),
        }
        return {
            "grammar": grammar,
            "semantic": semantic,
        }

    def _comparison_root(self) -> Path:
        configured_root = self.output_root
        if configured_root == self.run_dir:
            return configured_root.parent
        return configured_root

    def _load_counterpart_summary(self) -> dict[str, Any] | None:
        root = self._comparison_root()
        if not root.exists():
            return None

        desired_constraints = bool(self.metadata.get("constraints_enabled"))
        candidates: list[tuple[str, dict[str, Any]]] = []
        pattern = f"*/{self.dataset_name}/{self.model_name}/profile/profile_summary.json"
        for summary_path in root.glob(pattern):
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            if bool(metadata.get("constraints_enabled")) == desired_constraints:
                continue
            if metadata.get("dataset_name") != self.dataset_name:
                continue
            if metadata.get("model_name") != self.model_name:
                continue
            if metadata.get("checkpoint_path") != self.metadata.get("checkpoint_path"):
                continue
            if metadata.get("resolved_batch_size") != self.metadata.get("resolved_batch_size"):
                continue
            if metadata.get("limit") != self.metadata.get("limit"):
                continue
            created_at = metadata.get("created_at")
            score = created_at if isinstance(created_at, str) else ""
            candidates.append((score, payload))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _build_comparison(
        self,
        current: dict[str, Any],
        counterpart: dict[str, Any],
    ) -> dict[str, Any]:
        current_constraints = bool(current["metadata"]["constraints_enabled"])
        constrained = current if current_constraints else counterpart
        unconstrained = counterpart if current_constraints else current

        constrained_batch_mean = float(constrained["batch_latency_ms"]["mean"])
        unconstrained_batch_mean = float(unconstrained["batch_latency_ms"]["mean"])
        constrained_generate_mean = float(constrained["stages_ms"]["generate"]["mean"])
        unconstrained_generate_mean = float(unconstrained["stages_ms"]["generate"]["mean"])

        slowdown_ratio = (
            constrained_batch_mean / unconstrained_batch_mean
            if unconstrained_batch_mean > 0
            else 0.0
        )
        generate_delta_ms = constrained_generate_mean - unconstrained_generate_mean
        batch_delta_ms = constrained_batch_mean - unconstrained_batch_mean

        return {
            "counterpart_constraints_enabled": bool(counterpart["metadata"]["constraints_enabled"]),
            "counterpart_run_created_at": counterpart["metadata"].get("created_at"),
            "batch_latency_ms_delta": batch_delta_ms,
            "batch_latency_slowdown_ratio": slowdown_ratio,
            "generate_ms_delta": generate_delta_ms,
            "samples_per_second_delta": (
                float(constrained["throughput"]["samples_per_second"])
                - float(unconstrained["throughput"]["samples_per_second"])
            ),
            "tokens_per_second_delta": (
                float(constrained["throughput"]["tokens_per_second"])
                - float(unconstrained["throughput"]["tokens_per_second"])
            ),
            "constrained_constraints": constrained["constraint_processors"],
            "unconstrained_constraints": unconstrained["constraint_processors"],
        }

    def _build_conclusion(self, summary: dict[str, Any]) -> str:
        comparison = summary.get("comparison")
        grammar_mean = float(summary["constraint_processors"]["grammar"]["total_ms"]["mean"])
        semantic_mean = float(summary["constraint_processors"]["semantic"]["total_ms"]["mean"])

        if comparison:
            generate_delta_ms = float(comparison.get("generate_ms_delta", 0.0))
            threshold = max(1.0, generate_delta_ms * 0.5)
            if semantic_mean >= grammar_mean and semantic_mean >= threshold:
                return "constraint overhead dominated by semantic processor"
            if grammar_mean > semantic_mean and grammar_mean >= threshold:
                return "constraint overhead dominated by grammar mask application"
            if generate_delta_ms > 0:
                return "slowdown mostly in model/gpu compute"
        if semantic_mean > grammar_mean and semantic_mean > 0.0:
            return "constraint overhead dominated by semantic processor"
        if grammar_mean >= semantic_mean and grammar_mean > 0.0:
            return "constraint overhead dominated by grammar mask application"
        if float(summary["stages_ms"]["generate"]["mean"]) > 0.0:
            return "slowdown mostly in model/gpu compute"
        return "profiling inconclusive, inspect Chrome trace"
