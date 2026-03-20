"""Lazy exports for evaluation utilities.

This package is imported by the benchmark path for OMR-NED helpers.
Keep imports lazy so a metric-only caller does not eagerly pull in the
full training/inference stack.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "EvaluationHarness": ("src.evaluation.harness", "EvaluationHarness"),
    "EvaluationResults": ("src.evaluation.results", "EvaluationResults"),
    "ModelResults": ("src.evaluation.results", "ModelResults"),
    "SampleResult": ("src.evaluation.results", "SampleResult"),
    "ModelWrapper": ("src.evaluation.wrappers", "ModelWrapper"),
    "UserSMTWrapper": ("src.evaluation.wrappers", "UserSMTWrapper"),
    "PRAIGModelWrapper": ("src.evaluation.wrappers", "PRAIGModelWrapper"),
    "EvalCollator": ("src.evaluation.collator", "EvalCollator"),
    "EvalDatasetWrapper": ("src.evaluation.collator", "EvalDatasetWrapper"),
    "compute_cer": ("src.evaluation.string_metrics", "compute_cer"),
    "compute_ser": ("src.evaluation.string_metrics", "compute_ser"),
    "compute_ler": ("src.evaluation.string_metrics", "compute_ler"),
    "compute_omr_ned": ("src.evaluation.omr_ned", "compute_omr_ned"),
    "compute_omr_ned_from_musicxml": ("src.evaluation.omr_ned", "compute_omr_ned_from_musicxml"),
    "is_musicdiff_available": ("src.evaluation.omr_ned", "is_musicdiff_available"),
    "OMRNEDResult": ("src.evaluation.omr_ned", "OMRNEDResult"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
