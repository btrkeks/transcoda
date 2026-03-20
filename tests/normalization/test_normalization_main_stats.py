"""Tests for normalization.main stats JSON pass-level metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization.main import normalize_kern_files


def test_stats_json_includes_unknown_char_drop_counts(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    stats_path = tmp_path / "stats.json"
    input_dir.mkdir()

    (input_dir / "sample.krn").write_text(
        "*clefG2\t*clefF4\n"
        "*k[]\t*k[]\n"
        "*M4/4\t*M4/4\n"
        "*\t*tuplet\n"
        "8czL\t8ffSLs\n",
        encoding="utf-8",
    )

    normalize_kern_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        workers=1,
        quiet=True,
        stats_json=str(stats_path),
    )

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    pass_stats = payload["normalization_pass_stats"]["canonicalize_note_order"]
    assert pass_stats["unknown_char_drops_total"] == 1
    assert pass_stats["unknown_char_drop_counts"] == {"z": 1}

    normalized = (output_dir / "sample.krn").read_text(encoding="utf-8")
    assert "*tuplet" not in normalized
    assert "8ffL" in normalized
