"""Post-validation VeroSim OMR-NED integration compatibility facade."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from transcoda.validation.metrics.stored_targets import target_lookup_for
from transcoda.validation.metrics.verosim import (
    KERN_PARSER_ENV_VAR as _KERN_PARSER_ENV_VAR,
)
from transcoda.validation.metrics.verosim import (
    VEROSIM_ENV_VAR as _VEROSIM_ENV_VAR,
)
from transcoda.validation.metrics.verosim import (
    VEROSIM_LAYOUT_SURFACE as _VEROSIM_LAYOUT_SURFACE,
)
from transcoda.validation.metrics.verosim import (
    BinaryResolution,
    VeroSimBinaryResolver,
    VeroSimOMRNEDBackend,
    VeroSimOMRNEDResult,
    diagnostics_summary_metric_payload,
    metric_payload_from_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KERN_PARSER_ENV_VAR = _KERN_PARSER_ENV_VAR
VEROSIM_ENV_VAR = _VEROSIM_ENV_VAR
VEROSIM_LAYOUT_SURFACE = _VEROSIM_LAYOUT_SURFACE


def resolve_kern_parser_binary(explicit: str | Path | None = None) -> BinaryResolution:
    return _binary_resolver().resolve_kern_parser_binary(explicit)


def resolve_verosim_binary(explicit: str | Path | None = None) -> BinaryResolution:
    return _binary_resolver().resolve_verosim_binary(explicit)


def run_verosim_omr_ned(
    *,
    predictions_dir: str | Path,
    kern_parser_binary: str | Path | None = None,
    verosim_binary: str | Path | None = None,
    mode: str | None = None,
    detail: str | None = None,
    jobs: int = 0,
    failure_score: float | None = None,
    target_text: str | None = None,
    validation_paths: Mapping[str, str] | None = None,
    exclude_val_sets: Collection[str] = (),
) -> VeroSimOMRNEDResult:
    """Convert validation prediction JSONL and score it with VeroSim.

    ``target_text: stored`` scores against the datasets' stored labels, read
    from ``validation_paths``; unset or ``round_trip`` keeps the records' targets.
    Records of ``exclude_val_sets`` (no-music sets) are left out of scoring.
    """
    backend = VeroSimOMRNEDBackend(
        binary_resolver=_binary_resolver(),
        kern_parser_binary=kern_parser_binary,
        verosim_binary=verosim_binary,
        mode=mode,
        detail=detail,
        jobs=jobs,
        failure_score=failure_score,
        target_lookup=target_lookup_for(target_text, validation_paths or {}),
        exclude_val_sets=exclude_val_sets,
    )
    return backend.run(predictions_dir=predictions_dir)


def log_verosim_omr_ned_summary(
    wandb_logger: Any,
    summary: dict[str, Any],
    diagnostics_summary: dict[str, Any] | None = None,
) -> None:
    """Best-effort logging of VeroSim summary metrics to the active W&B logger."""
    experiment = getattr(wandb_logger, "experiment", None)
    log_fn = getattr(experiment, "log", None)
    if not callable(log_fn):
        return

    payload = metric_payload_from_summary(summary)
    payload.update(diagnostics_summary_metric_payload(diagnostics_summary))
    if payload:
        log_fn(payload)


def _binary_resolver() -> VeroSimBinaryResolver:
    return VeroSimBinaryResolver(repo_root=REPO_ROOT)
