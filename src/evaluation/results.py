"""Result dataclasses and output formatting for evaluation harness."""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table


@dataclass
class SampleResult:
    """Result for a single evaluation sample."""

    sample_id: int
    source: str
    prediction: str
    ground_truth: str
    cer: float
    ser: float
    ler: float
    # OMR-NED fields (optional - None if not computed or parse failed)
    omr_ned: float | None = None
    omr_ned_parse_error: str | None = None


@dataclass
class SourceMetrics:
    """Aggregated metrics for a single data source."""

    cer: float
    ser: float
    ler: float
    count: int
    # OMR-NED fields (optional - None if not computed)
    omr_ned: float | None = None
    omr_ned_valid_count: int = 0
    omr_ned_parse_failures: int = 0


@dataclass
class ModelResults:
    """Results for a single model across all samples."""

    model_name: str
    cer: float
    ser: float
    ler: float
    num_samples: int
    per_source: dict[str, SourceMetrics] = field(default_factory=dict)
    samples: list[SampleResult] = field(default_factory=list)
    # OMR-NED fields (optional - None if not computed)
    omr_ned: float | None = None
    omr_ned_valid_count: int = 0
    omr_ned_parse_failures: int = 0


@dataclass
class EvaluationResults:
    """Complete evaluation results for all models."""

    models: list[ModelResults]
    dataset_path: str
    total_samples: int

    def to_json(self) -> str:
        """Serialize results to JSON string."""

        def serialize(obj):
            if hasattr(obj, "__dict__"):
                return {k: serialize(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            else:
                return obj

        return json.dumps(serialize(self), indent=2)

    def to_table(self, console: Console | None = None) -> str:
        """Format results as rich console tables.

        Args:
            console: Optional Rich console for printing. If None, returns string.

        Returns:
            Formatted table string
        """
        if console is None:
            console = Console(force_terminal=False, record=True)
            should_export = True
        else:
            should_export = False

        # Check if OMR-NED was computed for any model
        has_omr_ned = any(model.omr_ned is not None for model in self.models)

        # Main comparison table
        main_table = Table(title="Model Evaluation Results", show_header=True)
        main_table.add_column("Model", style="cyan", no_wrap=True)
        main_table.add_column("CER", justify="right")
        main_table.add_column("SER", justify="right")
        main_table.add_column("LER", justify="right")
        if has_omr_ned:
            main_table.add_column("OMR-NED", justify="right")
        main_table.add_column("Samples", justify="right")

        for model in self.models:
            row = [
                model.model_name,
                f"{model.cer:.2f}%",
                f"{model.ser:.2f}%",
                f"{model.ler:.2f}%",
            ]
            if has_omr_ned:
                if model.omr_ned is not None:
                    row.append(f"{model.omr_ned:.2f}%")
                else:
                    row.append("N/A")
            row.append(str(model.num_samples))
            main_table.add_row(*row)

        console.print(main_table)
        console.print()

        # Per-source breakdown for each model
        for model in self.models:
            if model.per_source:
                # Check if this model has OMR-NED data
                model_has_omr_ned = any(m.omr_ned is not None for m in model.per_source.values())

                source_table = Table(
                    title=f"Per-Source Breakdown ({model.model_name})", show_header=True
                )
                source_table.add_column("Source", style="cyan", no_wrap=True)
                source_table.add_column("CER", justify="right")
                source_table.add_column("SER", justify="right")
                source_table.add_column("LER", justify="right")
                if model_has_omr_ned:
                    source_table.add_column("OMR-NED", justify="right")
                source_table.add_column("Samples", justify="right")

                for source_name, metrics in model.per_source.items():
                    row = [
                        source_name,
                        f"{metrics.cer:.2f}%",
                        f"{metrics.ser:.2f}%",
                        f"{metrics.ler:.2f}%",
                    ]
                    if model_has_omr_ned:
                        if metrics.omr_ned is not None:
                            row.append(f"{metrics.omr_ned:.2f}%")
                        else:
                            row.append("N/A")
                    row.append(str(metrics.count))
                    source_table.add_row(*row)

                console.print(source_table)
                console.print()

        if should_export:
            return console.export_text()
        return ""

    def to_csv(self, path: str | Path) -> None:
        """Export per-sample results to CSV for detailed analysis.

        Args:
            path: Output CSV file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

            # Header
            writer.writerow(
                [
                    "model",
                    "sample_id",
                    "source",
                    "cer",
                    "ser",
                    "ler",
                    "omr_ned",
                    "omr_ned_parse_error",
                    "prediction",
                    "ground_truth",
                ]
            )

            # Data rows
            for model in self.models:
                for sample in model.samples:
                    writer.writerow(
                        [
                            model.model_name,
                            sample.sample_id,
                            sample.source,
                            f"{sample.cer:.4f}",
                            f"{sample.ser:.4f}",
                            f"{sample.ler:.4f}",
                            f"{sample.omr_ned:.4f}" if sample.omr_ned is not None else "",
                            sample.omr_ned_parse_error or "",
                            sample.prediction,
                            sample.ground_truth,
                        ]
                    )
