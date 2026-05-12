"""Benchmark pipeline for cross-model OMR evaluation."""

from __future__ import annotations

from importlib import import_module

__all__ = ["BenchmarkRunner", "BenchmarkRunnerConfig"]

_EXPORTS = {
    "BenchmarkRunner": ("src.benchmark.runner", "BenchmarkRunner"),
    "BenchmarkRunnerConfig": ("src.benchmark.runner", "BenchmarkRunnerConfig"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
