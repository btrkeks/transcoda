"""Lazy exports for evaluation utilities.

Keep imports lazy so a metric-only caller does not eagerly pull in the
full training/inference stack.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "compute_cer": ("transcoda.evaluation.string_metrics", "compute_cer"),
    "compute_ser": ("transcoda.evaluation.string_metrics", "compute_ser"),
    "compute_ler": ("transcoda.evaluation.string_metrics", "compute_ler"),
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
