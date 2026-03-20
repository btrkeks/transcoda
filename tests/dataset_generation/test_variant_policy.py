from pathlib import Path

from scripts.dataset_generation.dataset_generation.variant_policy import (
    build_adaptive_variant_plan,
    build_fixed_variant_plan,
    variants_for_line_count,
)


def _write_lines(path: Path, count: int) -> None:
    path.write_text("".join("4c\n" for _ in range(count)), encoding="utf-8")


def test_variants_for_line_count_bin_boundaries():
    assert variants_for_line_count(0) == 1
    assert variants_for_line_count(256) == 1
    assert variants_for_line_count(257) == 2
    assert variants_for_line_count(512) == 2
    assert variants_for_line_count(513) == 4
    assert variants_for_line_count(1024) == 4
    assert variants_for_line_count(1025) == 5


def test_build_fixed_variant_plan(tmp_path):
    paths = []
    for idx in range(3):
        path = tmp_path / f"{idx:03}.krn"
        _write_lines(path, 10)
        paths.append(path)

    plan, summary = build_fixed_variant_plan(paths, variants_per_file=3)

    assert all(plan[path] == 3 for path in paths)
    assert summary["enabled"] is False
    assert summary["policy"] == "fixed"
    assert summary["file_count"] == 3
    assert summary["total_available_tasks"] == 9
    assert summary["variant_count_distribution"] == {"3": 3}
    assert summary["mean_variants_per_file"] == 3.0


def test_build_adaptive_variant_plan(tmp_path):
    path_a = tmp_path / "a.krn"
    path_b = tmp_path / "b.krn"
    path_c = tmp_path / "c.krn"
    path_d = tmp_path / "d.krn"
    _write_lines(path_a, 100)
    _write_lines(path_b, 300)
    _write_lines(path_c, 900)
    _write_lines(path_d, 1300)

    plan, summary = build_adaptive_variant_plan([path_a, path_b, path_c, path_d])

    assert plan[path_a] == 1
    assert plan[path_b] == 2
    assert plan[path_c] == 4
    assert plan[path_d] == 5
    assert summary["enabled"] is True
    assert summary["policy"] == "line_count_v1"
    assert summary["file_count"] == 4
    assert summary["line_count_bin_counts"] == {
        "le_256": 1,
        "257_512": 1,
        "513_1024": 1,
        "gt_1024": 1,
    }
    assert summary["variant_count_distribution"] == {"1": 1, "2": 1, "4": 1, "5": 1}
    assert summary["total_available_tasks"] == 12
    assert summary["mean_variants_per_file"] == 3.0
