"""Orchestration layer for benchmark runs."""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from datasets import load_from_disk
from rich.console import Console
from rich.table import Table

from src.benchmark.adapters import (
    BenchmarkModelAdapter,
    LegatoAdapter,
    OurCheckpointAdapter,
    SMTPlusPlusAdapter,
    resolve_device,
)
from src.benchmark.conversion import (
    build_sample_key,
    convert_abc_to_musicxml,
    convert_kern_to_musicxml,
    resolve_abc2xml_command,
    safe_sample_filename,
)
from src.data.preprocessing import LayoutNormalizationConfig
from src.benchmark.metrics import compute_omr_ned_xml, compute_tedn_from_musicxml
from src.benchmark.profiling import BenchmarkProfileConfig, InferenceProfileRecorder
from src.benchmark.results import DatasetModelSummary, SampleMetric, summarize_overall, summarize_samples
from src.evaluation.omr_ned import compute_omr_ned
from src.evaluation.omr_ned_aggregation import resolve_omr_ned_score
from src.evaluation.string_metrics import compute_cer

console = Console()
SUPPORTED_METRICS = frozenset({"cer", "omr_ned", "tedn"})


def resolve_batch_size(requested: int | str, device) -> int:
    """Resolve the effective inference batch size."""
    if isinstance(requested, int):
        return max(1, requested)
    if requested != "auto":
        raise ValueError(f"Unsupported batch size value: {requested!r}")
    if device.type == "cuda":
        return 4
    return 1


def resolve_metric_workers(requested: int | str) -> int:
    """Resolve the effective metric worker count."""
    if isinstance(requested, int):
        return max(1, requested)
    if requested != "auto":
        raise ValueError(f"Unsupported metric worker value: {requested!r}")

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, min(8, int(slurm_cpus)))
        except ValueError:
            pass

    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


@dataclass(frozen=True)
class BenchmarkRunnerConfig:
    dataset_root: str = "data/datasets/benchmark"
    datasets: tuple[str, ...] = ()
    models: tuple[str, ...] = ("ours", "smtpp", "legato")
    metrics: tuple[str, ...] = ("omr_ned", "tedn", "cer")
    output_root: str = "outputs/benchmark"
    ours_checkpoint: str = "weights/GrandStaff/smt-model.ckpt"
    smtpp_model_id: str = "PRAIG/smt-fp-grandstaff"
    smtpp_max_length: int | None = None
    legato_model_id: str = "guangyangmusic/legato"
    legato_encoder_path: str | None = None
    legato_max_length: int = 2048
    legato_num_beams: int = 10
    device: str = "auto"
    limit: int | None = None
    batch_size: int | str = "auto"
    metric_workers: int | str = "auto"
    hum2xml_path: str = "hum2xml"
    abc2xml_path: str = "abc2xml"
    resume: bool = False
    skip_inference: bool = False
    skip_invalid_gold: bool = False
    ours_normalize_layout: bool = False
    ours_strategy: str | None = None
    ours_num_beams: int | None = None
    ours_repetition_penalty: float | None = None
    ours_loop_recovery: bool = False
    ours_loop_recovery_repetition_penalty: float = 1.35
    disable_constraints: bool = False
    profile: bool = False
    profile_warmup_batches: int = 2
    profile_max_batches: int | None = None
    profile_trace: bool = False


@dataclass(frozen=True)
class BenchmarkSample:
    sample_index: int
    sample_key: str
    source: str
    ground_truth: str


@dataclass(frozen=True)
class InvalidGoldSample:
    dataset_name: str
    sample_index: int
    sample_key: str
    source: str
    error: str
    diagnostics: str | None


@dataclass(frozen=True)
class MetricJob:
    sample_index: int
    sample_key: str
    source: str
    prediction: str
    ground_truth: str
    raw_format: str
    gold_xml_text: str
    hum2xml_path: str
    abc2xml_command: tuple[str, ...] | None
    metrics: tuple[str, ...]
    compute_cer: bool = False


@dataclass(frozen=True)
class MetricJobResult:
    sample_metric: SampleMetric
    pred_xml_text: str | None


def discover_datasets(dataset_root: str, requested: tuple[str, ...]) -> list[str]:
    """Discover benchmark datasets under the root directory."""
    root = Path(dataset_root)
    if requested:
        return list(requested)
    datasets = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name != "_runs":
            datasets.append(child.name)
    return datasets


def resolve_run_dir(output_root: str, *, resume: bool, skip_inference: bool) -> Path:
    """Resolve the effective run directory for fresh or resumed runs."""
    root = Path(output_root)
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        root.mkdir(parents=True, exist_ok=True)
        return root

    if resume or skip_inference:
        candidates = sorted(
            [child for child in root.iterdir() if child.is_dir()],
            reverse=True,
        ) if root.exists() else []
        if candidates:
            return candidates[0]
        return root

    run_dir = root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def metric_job_requires_conversion(raw_format: str, metrics: tuple[str, ...]) -> bool:
    """Return whether a metric job must convert predictions before scoring."""
    metric_set = set(metrics)
    if raw_format == "kern":
        return "tedn" in metric_set
    if raw_format == "abc":
        return "tedn" in metric_set or "omr_ned" in metric_set
    return False


def build_metric_worker_crash_result(job: MetricJob, message: str) -> MetricJobResult:
    """Map unexpected metric worker crashes to a sample result without faking conversion failures."""
    conversion_failed = metric_job_requires_conversion(job.raw_format, job.metrics)
    conversion_error = message if conversion_failed else None
    return MetricJobResult(
        sample_metric=SampleMetric(
            sample_index=job.sample_index,
            sample_key=job.sample_key,
            source=job.source,
            metric_worker_failed=True,
            conversion_failed=conversion_failed,
            conversion_error=conversion_error,
            tedn=100.0 if "tedn" in set(job.metrics) else None,
            tedn_error=message if "tedn" in set(job.metrics) else None,
            omr_ned=100.0 if "omr_ned" in set(job.metrics) else None,
            omr_ned_error=message if "omr_ned" in set(job.metrics) else None,
        ),
        pred_xml_text=None,
    )


def compute_metric_job(job: MetricJob) -> MetricJobResult:
    """Compute benchmark metrics for a single sample."""
    try:
        conversion_failed = False
        conversion_error: str | None = None
        tedn_error: str | None = None
        omr_error: str | None = None
        tedn_score: float | None = None
        omr_score: float | None = None
        cer_score: float | None = None
        pred_xml_text: str | None = None
        metrics = set(job.metrics)

        if job.compute_cer and job.raw_format == "kern" and "cer" in metrics:
            cer_score = compute_cer(job.prediction, job.ground_truth)

        if job.raw_format == "kern":
            if "omr_ned" in metrics:
                try:
                    omr_result = compute_omr_ned(job.prediction, job.ground_truth)
                except Exception as exc:
                    omr_score = 100.0
                    omr_error = str(exc)
                else:
                    omr_score, failed = resolve_omr_ned_score(omr_result)
                    if failed:
                        omr_error = omr_result.parse_error

            if "tedn" in metrics:
                conversion = convert_kern_to_musicxml(job.prediction, job.hum2xml_path)
                if conversion.musicxml is None:
                    conversion_failed = True
                    conversion_error = conversion.error or "Prediction conversion failed"
                    tedn_score = 100.0
                    tedn_error = conversion_error
                else:
                    pred_xml_text = conversion.musicxml
                    try:
                        tedn_score = compute_tedn_from_musicxml(pred_xml_text, job.gold_xml_text)
                    except Exception as exc:
                        tedn_score = 100.0
                        tedn_error = str(exc)
        else:
            if "tedn" in metrics or "omr_ned" in metrics:
                if job.abc2xml_command is None:
                    raise RuntimeError("ABC conversion command is missing for ABC prediction.")
                conversion = convert_abc_to_musicxml(job.prediction, job.abc2xml_command)
                if conversion.musicxml is None:
                    conversion_failed = True
                    conversion_error = conversion.error or "Prediction conversion failed"
                    if "tedn" in metrics:
                        tedn_score = 100.0
                        tedn_error = conversion_error
                    if "omr_ned" in metrics:
                        omr_score = 100.0
                        omr_error = conversion_error
                else:
                    pred_xml_text = conversion.musicxml
                    if "tedn" in metrics:
                        try:
                            tedn_score = compute_tedn_from_musicxml(pred_xml_text, job.gold_xml_text)
                        except Exception as exc:
                            tedn_score = 100.0
                            tedn_error = str(exc)

                    if "omr_ned" in metrics:
                        try:
                            omr_result = compute_omr_ned_xml(pred_xml_text, job.gold_xml_text)
                        except Exception as exc:
                            omr_score = 100.0
                            omr_error = str(exc)
                        else:
                            omr_score, failed = resolve_omr_ned_score(omr_result)
                            if failed:
                                omr_error = omr_result.parse_error

        return MetricJobResult(
            sample_metric=SampleMetric(
                sample_index=job.sample_index,
                sample_key=job.sample_key,
                source=job.source,
                metric_worker_failed=False,
                conversion_failed=conversion_failed,
                conversion_error=conversion_error,
                tedn=tedn_score,
                tedn_error=tedn_error,
                omr_ned=omr_score,
                omr_ned_error=omr_error,
                cer=cer_score,
            ),
            pred_xml_text=pred_xml_text,
        )
    except Exception as exc:
        message = f"Metric worker crashed: {exc}"
        return build_metric_worker_crash_result(job, message)


class BenchmarkRunner:
    """Run benchmark inference, conversion, and XML metrics."""

    def __init__(self, config: BenchmarkRunnerConfig) -> None:
        self.config = config
        self.created_at = datetime.now(UTC).isoformat()
        self.dataset_names = discover_datasets(config.dataset_root, config.datasets)
        self.run_dir = resolve_run_dir(
            config.output_root,
            resume=config.resume,
            skip_inference=config.skip_inference,
        )
        self.device = resolve_device(config.device)
        self.resolved_batch_size = resolve_batch_size(config.batch_size, self.device)
        self.resolved_metric_workers = resolve_metric_workers(config.metric_workers)
        self.selected_metrics = self._resolve_metrics(config.metrics)
        self.profile_config = BenchmarkProfileConfig(
            enabled=config.profile,
            warmup_batches=max(0, int(config.profile_warmup_batches)),
            max_batches=(
                None
                if config.profile_max_batches is None
                else max(1, int(config.profile_max_batches))
            ),
            trace_enabled=bool(config.profile_trace),
        )
        self.abc2xml_command: list[str] | None = None
        self._adapters: dict[str, BenchmarkModelAdapter] = {}
        self.invalid_gold_samples: list[InvalidGoldSample] = []

    @staticmethod
    def _resolve_metrics(metrics: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(metric.strip().lower() for metric in metrics if metric.strip())
        if not normalized:
            raise ValueError("At least one benchmark metric must be selected.")
        unsupported = sorted(set(normalized) - SUPPORTED_METRICS)
        if unsupported:
            raise ValueError(
                f"Unsupported benchmark metrics: {', '.join(unsupported)}. "
                f"Supported metrics: {', '.join(sorted(SUPPORTED_METRICS))}."
            )
        return normalized

    def run(self) -> dict:
        """Execute the full benchmark run."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        summaries: list[DatasetModelSummary] = []
        console.log(
            f"[benchmark] starting run_dir={self.run_dir} device={self.device} "
            f"datasets={','.join(self.dataset_names)} models={','.join(self.config.models)}"
        )

        our_decode_policy = None
        if "ours" in self.config.models:
            adapter = self._get_adapter("ours")
            our_decode_policy = getattr(adapter, "decode_policy", None)

        manifest = {
            "created_at": self.created_at,
            "run_dir": str(self.run_dir),
            "dataset_root": self.config.dataset_root,
            "datasets": self.dataset_names,
            "models": list(self.config.models),
            "metrics": list(self.selected_metrics),
            "model_ids": {
                "ours": self.config.ours_checkpoint,
                "smtpp": self.config.smtpp_model_id,
                "legato": self.config.legato_model_id,
                "legato_encoder": self.config.legato_encoder_path,
            },
            "tool_paths": {
                "hum2xml": self.config.hum2xml_path,
                "abc2xml": self._get_abc2xml_command() if "legato" in self.config.models else None,
            },
            "our_decode_policy": our_decode_policy,
            "requested_batch_size": self.config.batch_size,
            "resolved_batch_size": self.resolved_batch_size,
            "requested_metric_workers": self.config.metric_workers,
            "resolved_metric_workers": self.resolved_metric_workers,
            "limit": self.config.limit,
            "resume": self.config.resume,
            "skip_inference": self.config.skip_inference,
            "skip_invalid_gold": self.config.skip_invalid_gold,
            "ours_normalize_layout": self.config.ours_normalize_layout,
            "ours_loop_recovery": self.config.ours_loop_recovery,
            "ours_loop_recovery_repetition_penalty": self.config.ours_loop_recovery_repetition_penalty,
            "profile": {
                "enabled": self.profile_config.enabled,
                "warmup_batches": self.profile_config.warmup_batches,
                "max_batches": self.profile_config.max_batches,
                "trace_enabled": self.profile_config.trace_enabled,
            },
        }

        for dataset_name in self.dataset_names:
            dataset_dir = Path(self.config.dataset_root) / dataset_name
            dataset = load_from_disk(str(dataset_dir))
            if self.config.limit is not None and self.config.limit > 0:
                dataset = dataset.select(range(min(self.config.limit, len(dataset))))
            samples = self._build_samples(dataset)
            console.log(f"[benchmark] dataset={dataset_name} samples={len(samples)}")
            dataset_summaries: list[DatasetModelSummary] = []

            for model_name in self.config.models:
                model_summary = self._evaluate_model(
                    dataset_name=dataset_name,
                    dataset=dataset,
                    samples=samples,
                    model_name=model_name,
                )
                summaries.append(model_summary)
                dataset_summaries.append(model_summary)

            self._print_summary_table(dataset_name, dataset_summaries)

        overall = summarize_overall(summaries)
        self._print_summary_table("overall", overall)

        payload = {
            "run_dir": str(self.run_dir),
            "datasets": self.dataset_names,
            "rows": [row.to_dict() for row in summaries],
            "overall": [row.to_dict() for row in overall],
        }
        manifest["failure_counts"] = {
            f"{row.dataset_name}:{row.model_name}": row.num_conversion_failures for row in summaries
        }
        manifest["loop_recovery_counts"] = {
            f"{row.dataset_name}:{row.model_name}": {
                "primary_flagged": row.num_loop_recovery_primary_flagged,
                "rerun_attempted": row.num_loop_recovery_rerun_attempted,
                "recovered": row.num_loop_recovery_recovered,
                "unrecovered": row.num_loop_recovery_unrecovered,
            }
            for row in summaries
            if (
                row.num_loop_recovery_primary_flagged
                or row.num_loop_recovery_rerun_attempted
                or row.num_loop_recovery_recovered
                or row.num_loop_recovery_unrecovered
            )
        }
        manifest["invalid_gold_samples"] = [
            {
                "dataset_name": row.dataset_name,
                "sample_index": row.sample_index,
                "sample_key": row.sample_key,
                "source": row.source,
                "error": row.error,
                "diagnostics": row.diagnostics,
            }
            for row in self.invalid_gold_samples
        ]
        self._write_summary_files(summaries, overall)
        (self.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        console.log(f"[benchmark] completed run_dir={self.run_dir}")
        return payload

    def _get_abc2xml_command(self) -> list[str]:
        if self.abc2xml_command is None:
            self.abc2xml_command = resolve_abc2xml_command(self.config.abc2xml_path)
        return self.abc2xml_command

    def _requires_gold_xml(self, raw_format: str) -> bool:
        metrics = set(self.selected_metrics)
        return "tedn" in metrics or (raw_format == "abc" and "omr_ned" in metrics)

    def _get_adapter(self, model_name: str) -> BenchmarkModelAdapter:
        if model_name in self._adapters:
            return self._adapters[model_name]

        if model_name == "ours":
            layout_normalization = None
            if self.config.ours_normalize_layout:
                layout_normalization = LayoutNormalizationConfig(enabled=True)
            adapter = OurCheckpointAdapter(
                self.config.ours_checkpoint,
                self.device,
                layout_normalization=layout_normalization,
                strategy=self.config.ours_strategy,
                num_beams=self.config.ours_num_beams,
                repetition_penalty=self.config.ours_repetition_penalty,
                use_constraints=not self.config.disable_constraints,
                profiling_enabled=self.profile_config.enabled,
                loop_recovery_enabled=self.config.ours_loop_recovery,
                loop_recovery_repetition_penalty=self.config.ours_loop_recovery_repetition_penalty,
            )
        elif model_name == "smtpp":
            adapter = SMTPlusPlusAdapter(
                self.config.smtpp_model_id,
                self.device,
                max_length=self.config.smtpp_max_length,
            )
        elif model_name == "legato":
            adapter = LegatoAdapter(
                self.config.legato_model_id,
                self.device,
                encoder_path=self.config.legato_encoder_path,
                max_length=self.config.legato_max_length,
                num_beams=self.config.legato_num_beams,
            )
        else:
            raise ValueError(f"Unsupported benchmark model '{model_name}'")

        self._adapters[model_name] = adapter
        return adapter

    def _build_samples(self, dataset) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []
        for idx, row in enumerate(dataset):
            source = row.get("source")
            sample_key = build_sample_key(idx, source)
            samples.append(
                BenchmarkSample(
                    sample_index=idx,
                    sample_key=sample_key,
                    source=source or "",
                    ground_truth=row["transcription"],
                )
            )
        return samples

    def _convert_gold(
        self,
        dataset_name: str,
        samples: list[BenchmarkSample],
    ) -> tuple[dict[str, str], list[BenchmarkSample], Path]:
        gold_xml: dict[str, str] = {}
        valid_samples: list[BenchmarkSample] = []
        gold_xml_dir = self.run_dir / dataset_name / "_gold_xml"
        gold_xml_dir.mkdir(parents=True, exist_ok=True)
        start_time = perf_counter()
        total = len(samples)
        console.log(f"[benchmark] gold-conversion dataset={dataset_name} total={total}")
        for idx, sample in enumerate(samples, start=1):
            conversion = convert_kern_to_musicxml(sample.ground_truth, self.config.hum2xml_path)
            if conversion.musicxml is None:
                details = conversion.diagnostics()
                invalid = InvalidGoldSample(
                    dataset_name=dataset_name,
                    sample_index=sample.sample_index,
                    sample_key=sample.sample_key,
                    source=sample.source,
                    error=conversion.error or "Gold conversion failed",
                    diagnostics=details,
                )
                self.invalid_gold_samples.append(invalid)
                if not self.config.skip_invalid_gold:
                    message = (
                        f"Gold conversion failed for dataset={dataset_name} sample={sample.sample_key}: "
                        f"{conversion.error}"
                    )
                    if details:
                        message = f"{message}\n{details}"
                    raise RuntimeError(message)
                console.log(
                    f"[benchmark] skipping invalid gold dataset={dataset_name} sample={sample.sample_key} "
                    f"error={invalid.error}"
                )
                continue
            gold_xml[sample.sample_key] = conversion.musicxml
            filename = safe_sample_filename(sample.sample_key) + ".xml"
            (gold_xml_dir / filename).write_text(conversion.musicxml, encoding="utf-8")
            valid_samples.append(sample)
            if self._should_log_progress(idx, total):
                self._log_progress(
                    stage="gold-conversion",
                    dataset_name=dataset_name,
                    completed=idx,
                    total=total,
                    started_at=start_time,
                )
        return gold_xml, valid_samples, gold_xml_dir

    def _load_raw_predictions(self, path: Path) -> list[dict]:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
        return rows

    def _write_raw_predictions(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _run_inference(
        self,
        dataset,
        samples: list[BenchmarkSample],
        adapter: BenchmarkModelAdapter,
        *,
        dataset_name: str,
        model_name: str,
    ) -> list[dict]:
        predictions: list[dict] = []
        start_time = perf_counter()
        total = len(samples)
        total_batches = max(1, (total + self.resolved_batch_size - 1) // self.resolved_batch_size)
        profile_recorder = self._build_profile_recorder(dataset_name=dataset_name, model_name=model_name)
        console.log(
            f"[benchmark] inference-start dataset={dataset_name} model={model_name} "
            f"total={total} batch_size={self.resolved_batch_size} total_batches={total_batches}"
        )
        for batch_index, start in enumerate(range(0, len(samples), self.resolved_batch_size)):
            batch_samples = samples[start : start + self.resolved_batch_size]
            batch_images = [dataset[sample.sample_index]["image"] for sample in batch_samples]
            profile_plan = {"collect": False, "trace_path": None}
            if profile_recorder is not None:
                profile_plan = profile_recorder.plan_batch(batch_index)
            prepare_profile = getattr(adapter, "prepare_profile", None)
            if callable(prepare_profile):
                prepare_profile(
                    collect_profile=bool(profile_plan["collect"]),
                    trace_path=profile_plan["trace_path"],
                )
            batch_started_at = perf_counter()
            try:
                predict_batch_rows = getattr(adapter, "predict_batch_rows", None)
                if callable(predict_batch_rows):
                    batch_rows = predict_batch_rows(batch_images)
                else:
                    batch_rows = [
                        {"prediction": prediction}
                        for prediction in adapter.predict_batch(batch_images)
                    ]
            except Exception as exc:
                console.log(
                    f"[benchmark] inference-crash dataset={dataset_name} model={model_name} "
                    f"batch_start={start} batch_size={len(batch_samples)} "
                    f"sample_keys={[sample.sample_key for sample in batch_samples]} "
                    f"sample_indices={[sample.sample_index for sample in batch_samples]} "
                    f"error={exc!r}"
                )
                raise
            batch_wall_ms = (perf_counter() - batch_started_at) * 1000.0
            if len(batch_rows) != len(batch_samples):
                raise RuntimeError(
                    f"Adapter returned {len(batch_rows)} rows for {len(batch_samples)} samples"
                )
            for sample, batch_row in zip(batch_samples, batch_rows, strict=False):
                prediction = batch_row["prediction"]
                raw_format = batch_row.get("raw_format", adapter.raw_format)
                extra_fields = {
                    key: value
                    for key, value in batch_row.items()
                    if key not in {"prediction", "raw_format"}
                }
                predictions.append(
                    {
                        "sample_index": sample.sample_index,
                        "sample_key": sample.sample_key,
                        "source": sample.source,
                        "raw_format": raw_format,
                        "prediction": prediction,
                        **extra_fields,
                    }
                )
            completed = min(start + len(batch_samples), total)
            if profile_recorder is not None and bool(profile_plan["collect"]):
                adapter_profile = self._consume_adapter_profile(adapter)
                profile_recorder.record_batch(
                    {
                        "batch_index": batch_index,
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "num_samples": len(batch_samples),
                        "sample_keys": [sample.sample_key for sample in batch_samples],
                        "batch_wall_ms": batch_wall_ms,
                        "generated_tokens": int(adapter_profile.get("generated_tokens", 0)),
                        "constraints_enabled": not self.config.disable_constraints,
                        "cumulative_samples_per_second": (
                            completed / max(perf_counter() - start_time, 1e-6)
                        ),
                        "adapter_profile": adapter_profile,
                        "trace_path": None
                        if profile_plan["trace_path"] is None
                        else str(profile_plan["trace_path"]),
                    }
                )
            self._log_progress(
                stage="inference",
                dataset_name=dataset_name,
                model_name=model_name,
                completed=completed,
                total=total,
                started_at=start_time,
                batch_index=batch_index + 1,
                total_batches=total_batches,
                batch_size=len(batch_samples),
                batch_wall_ms=batch_wall_ms,
            )
        if profile_recorder is not None:
            summary = profile_recorder.write_outputs()
            if summary is not None:
                console.log(
                    f"[benchmark] profile-written dataset={dataset_name} model={model_name} "
                    f"path={profile_recorder.profile_dir / 'profile_summary.json'}"
                )
        return predictions

    def _build_profile_recorder(
        self,
        *,
        dataset_name: str,
        model_name: str,
    ) -> InferenceProfileRecorder | None:
        if not self.profile_config.enabled:
            return None
        checkpoint_path = None
        if model_name == "ours":
            checkpoint_path = self.config.ours_checkpoint
        elif model_name == "smtpp":
            checkpoint_path = self.config.smtpp_model_id
        elif model_name == "legato":
            checkpoint_path = self.config.legato_model_id

        metadata = {
            "created_at": self.created_at,
            "run_dir": str(self.run_dir),
            "output_root": self.config.output_root,
            "device": str(self.device),
            "constraints_enabled": bool(model_name == "ours" and not self.config.disable_constraints),
            "checkpoint_path": checkpoint_path,
            "legato_encoder_path": self.config.legato_encoder_path if model_name == "legato" else None,
            "limit": self.config.limit,
            "resolved_batch_size": self.resolved_batch_size,
            "metrics": list(self.selected_metrics),
        }
        return InferenceProfileRecorder(
            enabled=True,
            run_dir=self.run_dir,
            output_root=Path(self.config.output_root),
            dataset_name=dataset_name,
            model_name=model_name,
            config=self.profile_config,
            metadata=metadata,
        )

    @staticmethod
    def _consume_adapter_profile(adapter: BenchmarkModelAdapter) -> dict:
        consume_profile = getattr(adapter, "consume_last_profile", None)
        if callable(consume_profile):
            profile = consume_profile()
            if isinstance(profile, dict):
                return profile
        return {
            "preprocess_ms": 0.0,
            "tensor_setup_ms": 0.0,
            "constraint_bundle_ms": 0.0,
            "generate_ms": 0.0,
            "finalize_ms": 0.0,
            "generated_tokens": 0,
            "generated_tokens_per_sample": [],
            "generate_tokens_per_second": 0.0,
            "constraint_stats": {},
            "trace_path": None,
        }

    def _evaluate_model(
        self,
        *,
        dataset_name: str,
        dataset,
        samples: list[BenchmarkSample],
        model_name: str,
    ) -> DatasetModelSummary:
        model_dir = self.run_dir / dataset_name / model_name
        summary_path = model_dir / "summary.json"
        raw_predictions_path = model_dir / "raw_predictions.jsonl"

        if self.config.resume and not self.config.skip_inference and summary_path.exists():
            return DatasetModelSummary(**json.loads(summary_path.read_text()))

        raw_format = "abc" if model_name == "legato" else "kern"
        if self.config.skip_inference:
            if not raw_predictions_path.exists():
                raise FileNotFoundError(
                    f"--skip-inference was requested but cached predictions are missing at {raw_predictions_path}"
                )
            predictions = self._load_raw_predictions(raw_predictions_path)
        elif self.config.resume and raw_predictions_path.exists():
            predictions = self._load_raw_predictions(raw_predictions_path)
        else:
            adapter = self._get_adapter(model_name)
            predictions = self._run_inference(
                dataset,
                samples,
                adapter,
                dataset_name=dataset_name,
                model_name=model_name,
            )
            self._write_raw_predictions(raw_predictions_path, predictions)
            raw_format = adapter.raw_format
        if predictions:
            raw_format = predictions[0].get("raw_format", raw_format)
        loop_recovery_counts = self._summarize_loop_recovery(predictions)

        eval_samples = samples
        gold_xml: dict[str, str] = {}
        if self._requires_gold_xml(raw_format):
            gold_xml, eval_samples, _gold_xml_dir = self._convert_gold(dataset_name, samples)
            if not eval_samples:
                console.log(
                    f"[benchmark] dataset={dataset_name} model={model_name} "
                    "has no valid gold samples after filtering"
                )
                sample_rows = []
                self._write_sample_metrics(model_dir / "per_sample_metrics.csv", sample_rows)
                summary = summarize_samples(dataset_name, model_name, raw_format, sample_rows)
                self._apply_loop_recovery_counts(summary, loop_recovery_counts)
                summary_path.write_text(json.dumps(summary.to_dict(), indent=2))
                return summary

        sample_rows: list[SampleMetric] = []
        pred_xml_dir = model_dir / "pred_xml"
        pred_xml_dir.mkdir(parents=True, exist_ok=True)
        metrics_start = perf_counter()
        total = len(eval_samples)
        console.log(
            f"[benchmark] metrics-start dataset={dataset_name} model={model_name} total={total} "
            f"workers={self.resolved_metric_workers}"
        )

        prediction_by_key = {row["sample_key"]: row for row in predictions}
        sample_rows = self._compute_metrics(
            dataset_name=dataset_name,
            model_name=model_name,
            samples=eval_samples,
            gold_xml=gold_xml,
            prediction_by_key=prediction_by_key,
            raw_format=raw_format,
            pred_xml_dir=pred_xml_dir,
            metrics_start=metrics_start,
        )

        self._write_sample_metrics(model_dir / "per_sample_metrics.csv", sample_rows)
        summary = summarize_samples(dataset_name, model_name, raw_format, sample_rows)
        self._apply_loop_recovery_counts(summary, loop_recovery_counts)
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2))
        tedn_text = "N/A" if summary.tedn is None else f"{summary.tedn:.3f}"
        omr_text = "N/A" if summary.omr_ned is None else f"{summary.omr_ned:.3f}"
        console.log(
            f"[benchmark] model-complete dataset={dataset_name} model={model_name} "
            f"tedn={tedn_text} omr_ned={omr_text} "
            f"conversion_failures={summary.num_conversion_failures}/{summary.num_samples}"
        )
        return summary

    @staticmethod
    def _summarize_loop_recovery(predictions: list[dict]) -> dict[str, int]:
        counts = {
            "primary_flagged": 0,
            "rerun_attempted": 0,
            "recovered": 0,
            "unrecovered": 0,
        }
        for row in predictions:
            loop_recovery = row.get("loop_recovery")
            if not isinstance(loop_recovery, dict):
                continue
            primary_flagged = bool(loop_recovery.get("primary_detected"))
            rerun_attempted = bool(loop_recovery.get("rerun_attempted"))
            recovered = bool(loop_recovery.get("replaced_prediction"))
            if primary_flagged:
                counts["primary_flagged"] += 1
            if rerun_attempted:
                counts["rerun_attempted"] += 1
            if recovered:
                counts["recovered"] += 1
            if primary_flagged and rerun_attempted and not recovered:
                counts["unrecovered"] += 1
        return counts

    @staticmethod
    def _apply_loop_recovery_counts(summary: DatasetModelSummary, counts: dict[str, int]) -> None:
        summary.num_loop_recovery_primary_flagged = int(counts["primary_flagged"])
        summary.num_loop_recovery_rerun_attempted = int(counts["rerun_attempted"])
        summary.num_loop_recovery_recovered = int(counts["recovered"])
        summary.num_loop_recovery_unrecovered = int(counts["unrecovered"])

    def _compute_metrics(
        self,
        *,
        dataset_name: str,
        model_name: str,
        samples: list[BenchmarkSample],
        gold_xml: dict[str, str],
        prediction_by_key: dict[str, dict],
        raw_format: str,
        pred_xml_dir: Path,
        metrics_start: float,
    ) -> list[SampleMetric]:
        pred_xml_dir.mkdir(parents=True, exist_ok=True)
        abc2xml_command = None
        needs_gold_xml = self._requires_gold_xml(raw_format)
        if any(
            prediction_by_key[sample.sample_key].get("raw_format", raw_format) == "abc"
            for sample in samples
        ):
            abc2xml_command = tuple(self._get_abc2xml_command())
        if needs_gold_xml:
            missing_keys = [sample.sample_key for sample in samples if sample.sample_key not in gold_xml]
            if missing_keys:
                preview = ", ".join(missing_keys[:3])
                raise RuntimeError(
                    f"Missing converted gold MusicXML for {len(missing_keys)} samples: {preview}"
                )
        jobs: list[MetricJob] = []
        for sample in samples:
            sample_prediction = prediction_by_key[sample.sample_key]
            sample_raw_format = sample_prediction.get("raw_format", raw_format)
            jobs.append(
                MetricJob(
                    sample_index=sample.sample_index,
                    sample_key=sample.sample_key,
                    source=sample.source,
                    prediction=sample_prediction["prediction"],
                    ground_truth=sample.ground_truth,
                    raw_format=sample_raw_format,
                    gold_xml_text=gold_xml.get(sample.sample_key, ""),
                    hum2xml_path=self.config.hum2xml_path,
                    abc2xml_command=abc2xml_command,
                    metrics=self.selected_metrics,
                    compute_cer=sample_raw_format == "kern" and "cer" in self.selected_metrics,
                )
            )

        results: list[MetricJobResult] = []
        total = len(jobs)
        completed = 0

        if self.resolved_metric_workers == 1:
            for job in jobs:
                results.append(compute_metric_job(job))
                completed += 1
                if self._should_log_progress(completed, total):
                    self._log_progress(
                        stage="metrics",
                        dataset_name=dataset_name,
                        model_name=model_name,
                        completed=completed,
                        total=total,
                        started_at=metrics_start,
                    )
        else:
            try:
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=self.resolved_metric_workers,
                    mp_context=ctx,
                ) as executor:
                    futures = {executor.submit(compute_metric_job, job): job for job in jobs}
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            message = f"Metric worker crashed: {exc}"
                            result = build_metric_worker_crash_result(job, message)
                        results.append(result)
                        completed += 1
                        if self._should_log_progress(completed, total):
                            self._log_progress(
                                stage="metrics",
                                dataset_name=dataset_name,
                                model_name=model_name,
                                completed=completed,
                                total=total,
                                started_at=metrics_start,
                            )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to start metric worker pool. Retry with --metric-workers 1."
                ) from exc

        results.sort(key=lambda row: row.sample_metric.sample_index)
        for result in results:
            if result.pred_xml_text is not None:
                filename = safe_sample_filename(result.sample_metric.sample_key) + ".xml"
                (pred_xml_dir / filename).write_text(result.pred_xml_text, encoding="utf-8")
        return [result.sample_metric for result in results]

    def _write_sample_metrics(self, path: Path, rows: list[SampleMetric]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_index",
                    "sample_key",
                    "source",
                    "metric_worker_failed",
                    "conversion_failed",
                    "conversion_error",
                    "tedn",
                    "tedn_error",
                    "omr_ned",
                    "omr_ned_error",
                    "cer",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_dict())

    def _write_summary_files(
        self,
        rows: list[DatasetModelSummary],
        overall: list[DatasetModelSummary],
    ) -> None:
        payload = {
            "rows": [row.to_dict() for row in rows],
            "overall": [row.to_dict() for row in overall],
        }
        (self.run_dir / "summary.json").write_text(json.dumps(payload, indent=2))
        with (self.run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "dataset_name",
                    "model_name",
                    "raw_format",
                    "num_samples",
                    "num_metric_worker_failures",
                    "num_conversion_failures",
                    "conversion_success_rate",
                    "cer",
                    "tedn",
                    "omr_ned",
                    "num_loop_recovery_primary_flagged",
                    "num_loop_recovery_rerun_attempted",
                    "num_loop_recovery_recovered",
                    "num_loop_recovery_unrecovered",
                ],
            )
            writer.writeheader()
            for row in [*rows, *overall]:
                writer.writerow(row.to_dict())

    def _should_log_progress(self, completed: int, total: int) -> bool:
        if total <= 0:
            return False
        if completed >= total:
            return True
        interval = max(1, min(250, total // 20 or 1))
        return completed == 1 or completed % interval == 0

    def _log_progress(
        self,
        *,
        stage: str,
        dataset_name: str,
        completed: int,
        total: int,
        started_at: float,
        model_name: str | None = None,
        batch_index: int | None = None,
        total_batches: int | None = None,
        batch_size: int | None = None,
        batch_wall_ms: float | None = None,
    ) -> None:
        elapsed = max(perf_counter() - started_at, 1e-6)
        rate = completed / elapsed
        remaining = max(total - completed, 0)
        eta = remaining / rate if rate > 0 else 0.0
        target = dataset_name if model_name is None else f"{dataset_name}/{model_name}"
        batch_info = ""
        if batch_index is not None and total_batches is not None:
            batch_info = f" batch={batch_index}/{total_batches}"
        if batch_size is not None:
            batch_info = f"{batch_info} batch_size={batch_size}"
        if batch_wall_ms is not None:
            batch_info = f"{batch_info} batch_ms={batch_wall_ms:.1f}"
        console.log(
            f"[benchmark] {stage} target={target} progress={completed}/{total} "
            f"({completed / total:.1%}){batch_info} elapsed={self._format_seconds(elapsed)} "
            f"eta={self._format_seconds(eta)} rate={rate:.2f}/s"
        )

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m{secs:02d}s"
        if minutes:
            return f"{minutes:d}m{secs:02d}s"
        return f"{secs:d}s"

    def _print_summary_table(self, title: str, rows: list[DatasetModelSummary]) -> None:
        table = Table(title=f"Benchmark Summary: {title}")
        has_cer = any(row.cer is not None for row in rows)
        table.add_column("Model", style="cyan")
        table.add_column("Format")
        table.add_column("Samples", justify="right")
        table.add_column("Metric Fail", justify="right")
        table.add_column("Conv Fail", justify="right")
        table.add_column("Conv OK", justify="right")
        if has_cer:
            table.add_column("CER", justify="right")
        table.add_column("TEDn", justify="right")
        table.add_column("OMR-NED", justify="right")
        for row in rows:
            display_row = [
                row.model_name,
                row.raw_format,
                str(row.num_samples),
                str(row.num_metric_worker_failures),
                str(row.num_conversion_failures),
                f"{row.conversion_success_rate:.2f}%",
            ]
            if has_cer:
                display_row.append("N/A" if row.cer is None else f"{row.cer:.2f}%")
            display_row.extend(
                [
                    "N/A" if row.tedn is None else f"{row.tedn:.2f}%",
                    "N/A" if row.omr_ned is None else f"{row.omr_ned:.2f}%",
                ]
            )
            table.add_row(*display_row)
        console.print(table)
