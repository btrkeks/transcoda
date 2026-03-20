"""Evaluation harness for orchestrating model comparison."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from datasets import load_from_disk
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from torch.utils.data import DataLoader

from .collator import EvalCollator, EvalDatasetWrapper
from .omr_ned_aggregation import OMRNEDAggregator
from .results import EvaluationResults, ModelResults, SampleResult, SourceMetrics
from .string_metrics import compute_cer, compute_ler, compute_ser
from .wrappers import ModelWrapper

console = Console()


class EvaluationHarness:
    """Orchestrates model evaluation on validation dataset.

    Handles dataset loading, model inference, metric computation,
    and result aggregation across multiple models and data sources.
    """

    def __init__(
        self,
        dataset_path: str,
        models: list[ModelWrapper],
        device: torch.device,
        compute_omr_ned: bool = False,
        limit: int | None = None,
        batch_size: int = 1,
    ):
        """Initialize evaluation harness.

        Args:
            dataset_path: Path to HuggingFace dataset directory
            models: List of model wrappers to evaluate
            device: Device for inference (used for logging only, models handle their own devices)
            compute_omr_ned: If True, compute OMR-NED metric (slower, requires musicdiff)
            limit: If set, only evaluate on the first N samples (useful for quick testing)
            batch_size: Batch size for inference. Higher values speed up evaluation for models
                that support batching. Models without batch_predict() fall back to sequential.
        """
        self.dataset_path = dataset_path
        self.dataset = load_from_disk(dataset_path)
        self.models = models
        self.device = device
        self.compute_omr_ned = compute_omr_ned
        self.batch_size = batch_size

        # Apply limit if specified
        if limit is not None and limit > 0:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

        # Check OMR-NED availability if requested
        if compute_omr_ned:
            from .omr_ned import is_musicdiff_available

            if not is_musicdiff_available():
                console.print(
                    "[yellow]Warning: musicdiff not installed. "
                    "OMR-NED will not be computed. "
                    "Install with: uv sync --group omr-ned[/yellow]"
                )
                self.compute_omr_ned = False
            else:
                console.print("[cyan]OMR-NED computation enabled (this may be slow)[/cyan]")

        console.print(
            f"[green]Loaded dataset:[/green] {len(self.dataset)} samples from {dataset_path}"
        )
        if batch_size > 1:
            console.print(f"[cyan]Batch size:[/cyan] {batch_size}")

    def run(self) -> EvaluationResults:
        """Run evaluation on all models.

        Returns:
            EvaluationResults containing metrics for all models
        """
        results = []

        for model in self.models:
            console.print(f"\n[bold cyan]Evaluating {model.name}...[/bold cyan]")
            model_results = self._evaluate_model(model)
            results.append(model_results)

        return EvaluationResults(
            models=results,
            dataset_path=self.dataset_path,
            total_samples=len(self.dataset),
        )

    def _evaluate_model(self, model: ModelWrapper) -> ModelResults:
        """Evaluate a single model on all samples.

        Dispatches to batched or sequential evaluation based on model support
        and batch_size setting.

        Args:
            model: Model wrapper to evaluate

        Returns:
            ModelResults with per-sample and aggregated metrics
        """
        # Check if model supports batched inference
        supports_batching = hasattr(model, "batch_predict") and self.batch_size > 1

        if supports_batching:
            return self._evaluate_model_batched(model)
        else:
            return self._evaluate_model_sequential(model)

    @staticmethod
    def _new_source_metrics_bucket() -> dict[str, list[float]]:
        return {
            "cer": [],
            "ser": [],
            "ler": [],
        }

    def _resolve_omr_ned_compute_fn(self) -> Callable[[str, str], Any] | None:
        if not self.compute_omr_ned:
            return None

        from .omr_ned import compute_omr_ned as _compute_omr_ned

        return _compute_omr_ned

    def _record_prediction_result(
        self,
        *,
        sample_id: int,
        source: str,
        prediction: str,
        ground_truth: str,
        samples: list[SampleResult],
        metrics_by_source: dict[str, dict[str, list[float]]],
        omr_aggregator: OMRNEDAggregator | None,
        compute_omr_ned_func: Callable[[str, str], Any] | None,
    ) -> None:
        cer = compute_cer(prediction, ground_truth)
        ser = compute_ser(prediction, ground_truth)
        ler = compute_ler(prediction, ground_truth)

        omr_ned_value: float | None = None
        omr_ned_error: str | None = None
        if compute_omr_ned_func is not None:
            ned_result = compute_omr_ned_func(prediction, ground_truth)
            omr_ned_value = ned_result.omr_ned
            omr_ned_error = ned_result.parse_error
            if omr_aggregator is not None:
                omr_aggregator.add_result(ned_result, source)

        samples.append(
            SampleResult(
                sample_id=sample_id,
                source=source,
                prediction=prediction,
                ground_truth=ground_truth,
                cer=cer,
                ser=ser,
                ler=ler,
                omr_ned=omr_ned_value,
                omr_ned_parse_error=omr_ned_error,
            )
        )

        source_metrics = metrics_by_source[source]
        source_metrics["cer"].append(cer)
        source_metrics["ser"].append(ser)
        source_metrics["ler"].append(ler)

    def _evaluate_model_sequential(self, model: ModelWrapper) -> ModelResults:
        """Evaluate model using sequential single-sample inference.

        Args:
            model: Model wrapper to evaluate

        Returns:
            ModelResults with per-sample and aggregated metrics
        """
        compute_omr_ned_func = self._resolve_omr_ned_compute_fn()

        # Accumulators for metrics by source
        metrics_by_source: dict[str, dict[str, list[float]]] = defaultdict(
            self._new_source_metrics_bucket
        )
        samples: list[SampleResult] = []
        omr_aggregator = OMRNEDAggregator() if compute_omr_ned_func is not None else None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]{model.name}", total=len(self.dataset))

            for idx, sample in enumerate(self.dataset):
                # Extract sample data
                image: Image.Image = sample["image"]
                ground_truth: str = sample["transcription"]
                source: str = sample.get("source", "unknown")

                # Preprocess and predict
                image_tensor = model.preprocess(image)
                prediction = model.predict(image_tensor)

                self._record_prediction_result(
                    sample_id=idx,
                    source=source,
                    prediction=prediction,
                    ground_truth=ground_truth,
                    samples=samples,
                    metrics_by_source=metrics_by_source,
                    omr_aggregator=omr_aggregator,
                    compute_omr_ned_func=compute_omr_ned_func,
                )

                progress.update(task, advance=1)

        return self._aggregate_results(model.name, samples, metrics_by_source, omr_aggregator)

    def _evaluate_model_batched(self, model: ModelWrapper) -> ModelResults:
        """Evaluate model using batched inference with DataLoader.

        Args:
            model: Model wrapper with batch_predict() support

        Returns:
            ModelResults with per-sample and aggregated metrics
        """
        compute_omr_ned_func = self._resolve_omr_ned_compute_fn()

        # Accumulators for metrics by source
        metrics_by_source: dict[str, dict[str, list[float]]] = defaultdict(
            self._new_source_metrics_bucket
        )
        samples: list[SampleResult] = []
        omr_aggregator = OMRNEDAggregator() if compute_omr_ned_func is not None else None

        # Wrap dataset for DataLoader
        eval_dataset = EvalDatasetWrapper(self.dataset, model.preprocess)
        collator = EvalCollator()
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,  # Keep simple - preprocessing is fast
            pin_memory=str(self.device).startswith("cuda"),
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]{model.name}", total=len(self.dataset))

            for batch in dataloader:
                # Run batched inference
                predictions = model.batch_predict(
                    pixel_values=batch["pixel_values"],
                    image_sizes=batch["image_sizes"],
                )

                # Process each sample in batch
                for i, prediction in enumerate(predictions):
                    ground_truth = batch["transcription"][i]
                    source = batch["source"][i]
                    sample_id = batch["sample_id"][i]

                    self._record_prediction_result(
                        sample_id=sample_id,
                        source=source,
                        prediction=prediction,
                        ground_truth=ground_truth,
                        samples=samples,
                        metrics_by_source=metrics_by_source,
                        omr_aggregator=omr_aggregator,
                        compute_omr_ned_func=compute_omr_ned_func,
                    )

                progress.update(task, advance=len(predictions))

        return self._aggregate_results(model.name, samples, metrics_by_source, omr_aggregator)

    def _aggregate_results(
        self,
        model_name: str,
        samples: list[SampleResult],
        metrics_by_source: dict[str, dict[str, list[float]]],
        omr_aggregator: OMRNEDAggregator | None,
    ) -> ModelResults:
        """Aggregate per-sample metrics into ModelResults.

        Args:
            model_name: Name of the model
            samples: List of per-sample results
            metrics_by_source: Metrics accumulated by source

        Returns:
            Aggregated ModelResults
        """
        # Aggregate metrics
        all_cer = [s.cer for s in samples]
        all_ser = [s.ser for s in samples]
        all_ler = [s.ler for s in samples]
        omr_summary = omr_aggregator.compute() if omr_aggregator is not None else None

        # Build per-source aggregations
        per_source: dict[str, SourceMetrics] = {}
        for source_name, metrics in metrics_by_source.items():
            if metrics["cer"]:  # Only if we have samples for this source
                source_omr = omr_summary.by_source.get(source_name) if omr_summary is not None else None
                per_source[source_name] = SourceMetrics(
                    cer=sum(metrics["cer"]) / len(metrics["cer"]),
                    ser=sum(metrics["ser"]) / len(metrics["ser"]),
                    ler=sum(metrics["ler"]) / len(metrics["ler"]),
                    count=len(metrics["cer"]),
                    omr_ned=source_omr.score if source_omr is not None else None,
                    omr_ned_valid_count=(
                        source_omr.samples - source_omr.failures if source_omr is not None else 0
                    ),
                    omr_ned_parse_failures=source_omr.failures if source_omr is not None else 0,
                )

        return ModelResults(
            model_name=model_name,
            cer=sum(all_cer) / len(all_cer) if all_cer else 0.0,
            ser=sum(all_ser) / len(all_ser) if all_ser else 0.0,
            ler=sum(all_ler) / len(all_ler) if all_ler else 0.0,
            num_samples=len(samples),
            per_source=per_source,
            samples=samples,
            omr_ned=omr_summary.overall.score if omr_summary is not None else None,
            omr_ned_valid_count=(
                omr_summary.overall.samples - omr_summary.overall.failures
                if omr_summary is not None
                else 0
            ),
            omr_ned_parse_failures=omr_summary.overall.failures if omr_summary is not None else 0,
        )
