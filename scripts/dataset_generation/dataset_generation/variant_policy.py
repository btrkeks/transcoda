"""Variant policy helpers for dataset generation."""

from __future__ import annotations

from pathlib import Path


FIXED_VARIANTS_PER_FILE = 3


def build_fixed_variant_plan(
    file_paths: list[Path],
    variants_per_file: int = FIXED_VARIANTS_PER_FILE,
) -> tuple[dict[Path, int], dict[str, object]]:
    """Build per-file variant counts and summary metadata for fixed scheduling."""
    variant_count_by_file = {file_path: int(variants_per_file) for file_path in file_paths}
    total_available_tasks = int(len(file_paths) * int(variants_per_file))
    summary = {
        "enabled": False,
        "policy": "fixed",
        "file_count": len(file_paths),
        "variant_count_distribution": {str(int(variants_per_file)): len(file_paths)},
        "total_available_tasks": total_available_tasks,
        "mean_variants_per_file": float(variants_per_file) if file_paths else 0.0,
    }
    return variant_count_by_file, summary
