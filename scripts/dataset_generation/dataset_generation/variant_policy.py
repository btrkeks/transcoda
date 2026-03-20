"""Adaptive variant policy helpers for dataset generation."""

from __future__ import annotations

from pathlib import Path

LINE_COUNT_VARIANT_BINS: tuple[tuple[int | None, int], ...] = (
    (256, 1),
    (512, 2),
    (1024, 4),
    (None, 5),
)


def count_non_empty_lines(path: Path) -> int:
    """Count non-empty lines in a kern file."""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def variants_for_line_count(line_count: int) -> int:
    """Map line count to variant count using fixed adaptive bins."""
    for max_lines, variants in LINE_COUNT_VARIANT_BINS:
        if max_lines is None or line_count <= max_lines:
            return variants
    return LINE_COUNT_VARIANT_BINS[-1][1]


def build_adaptive_variant_plan(
    file_paths: list[Path],
) -> tuple[dict[Path, int], dict[str, object]]:
    """Build per-file variant counts and summary metadata for adaptive scheduling."""
    variant_count_by_file: dict[Path, int] = {}
    line_bin_counts = {"le_256": 0, "257_512": 0, "513_1024": 0, "gt_1024": 0}

    for file_path in file_paths:
        line_count = count_non_empty_lines(file_path)
        if line_count <= 256:
            line_bin_counts["le_256"] += 1
        elif line_count <= 512:
            line_bin_counts["257_512"] += 1
        elif line_count <= 1024:
            line_bin_counts["513_1024"] += 1
        else:
            line_bin_counts["gt_1024"] += 1
        variant_count_by_file[file_path] = variants_for_line_count(line_count)

    total_available_tasks = int(sum(variant_count_by_file.values()))
    file_count = len(file_paths)
    mean_variants_per_file = (total_available_tasks / file_count) if file_count else 0.0
    variant_count_distribution: dict[str, int] = {}
    for variants in variant_count_by_file.values():
        key = str(variants)
        variant_count_distribution[key] = variant_count_distribution.get(key, 0) + 1

    summary = {
        "enabled": True,
        "policy": "line_count_v1",
        "bins": [
            {"line_count_max": 256, "variants": 1},
            {"line_count_max": 512, "variants": 2},
            {"line_count_max": 1024, "variants": 4},
            {"line_count_max": None, "variants": 5},
        ],
        "file_count": file_count,
        "line_count_bin_counts": line_bin_counts,
        "variant_count_distribution": variant_count_distribution,
        "total_available_tasks": total_available_tasks,
        "mean_variants_per_file": mean_variants_per_file,
    }
    return variant_count_by_file, summary


def build_fixed_variant_plan(
    file_paths: list[Path],
    variants_per_file: int,
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
