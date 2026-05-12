from pathlib import Path

from scripts.dataset_generation.dataset_generation.source_stats import compute_kern_source_stats


def test_compute_kern_source_stats_counts_non_empty_lines_and_barlines(tmp_path):
    path = Path(tmp_path) / "sample.krn"
    path.write_text(
        "\n".join(
            [
                "*clefG2\t*clefF4",
                "*M4/4\t*M4/4",
                "",
                "=1\t=1",
                "4c\t4e",
                "=2\t=2",
                "4d\t4f",
                "==\t==",
                "*-\t*-",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = compute_kern_source_stats(path)

    assert stats.non_empty_line_count == 8
    assert stats.measure_count == 3
