"""Dataclasses for benchmark outputs and aggregations."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SampleMetric:
    sample_index: int
    sample_key: str
    source: str
    metric_worker_failed: bool = False
    conversion_failed: bool = False
    conversion_error: str | None = None
    tedn: float | None = None
    tedn_error: str | None = None
    omr_ned: float | None = None
    omr_ned_error: str | None = None
    cer: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetModelSummary:
    dataset_name: str
    model_name: str
    raw_format: str
    num_samples: int
    num_metric_worker_failures: int = 0
    num_conversion_failures: int = 0
    conversion_success_rate: float = 0.0
    tedn: float | None = None
    omr_ned: float | None = None
    num_loop_recovery_primary_flagged: int = 0
    num_loop_recovery_rerun_attempted: int = 0
    num_loop_recovery_recovered: int = 0
    num_loop_recovery_unrecovered: int = 0
    cer: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_samples(
    dataset_name: str,
    model_name: str,
    raw_format: str,
    sample_metrics: list[SampleMetric],
) -> DatasetModelSummary:
    """Aggregate sample-level metrics into dataset-level summary rows."""
    num_samples = len(sample_metrics)
    num_metric_worker_failures = sum(1 for row in sample_metrics if row.metric_worker_failed)
    num_conversion_failures = sum(1 for row in sample_metrics if row.conversion_failed)
    conversion_success_rate = (
        100.0 * (num_samples - num_conversion_failures) / num_samples if num_samples else 0.0
    )
    cer_values = [row.cer for row in sample_metrics if row.cer is not None]
    tedn_values = [row.tedn for row in sample_metrics if row.tedn is not None]
    omr_values = [row.omr_ned for row in sample_metrics if row.omr_ned is not None]
    cer = sum(cer_values) / len(cer_values) if cer_values else None
    tedn = sum(tedn_values) / len(tedn_values) if tedn_values else None
    omr_ned = sum(omr_values) / len(omr_values) if omr_values else None
    return DatasetModelSummary(
        dataset_name=dataset_name,
        model_name=model_name,
        raw_format=raw_format,
        num_samples=num_samples,
        num_metric_worker_failures=num_metric_worker_failures,
        num_conversion_failures=num_conversion_failures,
        conversion_success_rate=conversion_success_rate,
        tedn=tedn,
        omr_ned=omr_ned,
        cer=cer,
    )


def summarize_overall(rows: list[DatasetModelSummary]) -> list[DatasetModelSummary]:
    """Build weighted overall summaries across all datasets per model."""
    grouped: dict[tuple[str, str], list[DatasetModelSummary]] = {}
    for row in rows:
        grouped.setdefault((row.model_name, row.raw_format), []).append(row)

    overall: list[DatasetModelSummary] = []
    for (model_name, raw_format), group in sorted(grouped.items()):
        total = sum(row.num_samples for row in group)
        total_metric_worker_failures = sum(row.num_metric_worker_failures for row in group)
        total_failures = sum(row.num_conversion_failures for row in group)
        cer_rows = [row for row in group if row.cer is not None]
        tedn_rows = [row for row in group if row.tedn is not None]
        omr_rows = [row for row in group if row.omr_ned is not None]
        cer_total = sum(row.num_samples for row in cer_rows)
        tedn_total = sum(row.num_samples for row in tedn_rows)
        omr_total = sum(row.num_samples for row in omr_rows)
        weighted_cer = (
            sum(row.cer * row.num_samples for row in cer_rows if row.cer is not None) / cer_total
            if cer_total
            else None
        )
        weighted_tedn = (
            sum(row.tedn * row.num_samples for row in tedn_rows if row.tedn is not None) / tedn_total
            if tedn_total
            else None
        )
        weighted_omr = (
            sum(row.omr_ned * row.num_samples for row in omr_rows if row.omr_ned is not None) / omr_total
            if omr_total
            else None
        )
        overall.append(
            DatasetModelSummary(
                dataset_name="__overall__",
                model_name=model_name,
                raw_format=raw_format,
                num_samples=total,
                num_metric_worker_failures=total_metric_worker_failures,
                num_conversion_failures=total_failures,
                conversion_success_rate=100.0 * (total - total_failures) / total if total else 0.0,
                tedn=weighted_tedn,
                omr_ned=weighted_omr,
                num_loop_recovery_primary_flagged=sum(
                    row.num_loop_recovery_primary_flagged for row in group
                ),
                num_loop_recovery_rerun_attempted=sum(
                    row.num_loop_recovery_rerun_attempted for row in group
                ),
                num_loop_recovery_recovered=sum(
                    row.num_loop_recovery_recovered for row in group
                ),
                num_loop_recovery_unrecovered=sum(
                    row.num_loop_recovery_unrecovered for row in group
                ),
                cer=weighted_cer,
            )
        )
    return overall
