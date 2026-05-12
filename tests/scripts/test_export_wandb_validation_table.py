from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_wandb_validation_table import export_validation_table


def _write_table(path: Path, rows: list[list[object]], columns: list[str] | None = None) -> None:
    payload = {
        "_type": "table",
        "columns": columns
        or [
            "ID",
            "Category",
            "Ground Truth Image",
            "Ground Truth",
            "Prediction",
            "SER",
            "CER",
            "Diff",
        ],
        "data": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_export_validation_table_writes_review_folders(tmp_path: Path) -> None:
    bundle = tmp_path / "run"
    table_path = bundle / "qualitative" / "Validation Table.table.json"
    image_path = bundle / "qualitative" / "media" / "images" / "sample.png"
    diff_path = bundle / "qualitative" / "media" / "html" / "sample.html"
    image_path.parent.mkdir(parents=True)
    diff_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    diff_path.write_text("<html></html>", encoding="utf-8")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(
        table_path,
        [
            [
                "train_000001.krn",
                "worst",
                {"path": "media/images/sample.png", "_type": "image-file"},
                "*clefG2\n4c",
                "*clefG2\n4d",
                "12.5",
                "3.25",
                {"path": "media/html/sample.html", "_type": "html-file"},
            ]
        ],
    )

    count = export_validation_table(table_path, bundle / "table")

    assert count == 1
    sample_dir = bundle / "table" / "worst" / "polish" / "000_train_000001"
    assert (sample_dir / "image.png").read_bytes() == b"png"
    assert (sample_dir / "diff.html").read_text(encoding="utf-8") == "<html></html>"
    assert (sample_dir / "ground_truth.krn").read_text(encoding="utf-8") == "*clefG2\n4c\n"
    assert (sample_dir / "prediction.krn").read_text(encoding="utf-8") == "*clefG2\n4d\n"
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["metrics"] == {"CER": 3.25, "SER": 12.5}
    assert metadata["validation_set"] == "polish"
    index = json.loads((bundle / "table" / "index.json").read_text(encoding="utf-8"))
    assert index["categories"] == {"worst": 1}
    assert index["validation_sets"] == {"polish": 1}


def test_export_validation_table_prefixes_duplicate_ids(tmp_path: Path) -> None:
    table_path = tmp_path / "Validation Table.table.json"
    image_path = tmp_path / "media" / "images" / "sample.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    _write_table(
        table_path,
        [
            ["unknown", "best", {"path": "media/images/sample.png"}, "gt 1", "pred 1", None, None, None],
            ["unknown", "best", {"path": "media/images/sample.png"}, "gt 2", "pred 2", None, None, None],
        ],
    )

    export_validation_table(table_path, tmp_path / "table")

    assert (tmp_path / "table" / "best" / "synth" / "000_unknown" / "ground_truth.krn").exists()
    assert (tmp_path / "table" / "best" / "synth" / "001_unknown" / "ground_truth.krn").exists()


def test_export_validation_table_uses_explicit_validation_set_column(tmp_path: Path) -> None:
    table_path = tmp_path / "Validation Table.table.json"
    image_path = tmp_path / "media" / "images" / "sample.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    _write_table(
        table_path,
        [["sample.krn", "best", "synth", {"path": "media/images/sample.png"}, "gt", "pred"]],
        columns=[
            "ID",
            "Category",
            "Validation Set",
            "Ground Truth Image",
            "Ground Truth",
            "Prediction",
        ],
    )

    export_validation_table(table_path, tmp_path / "table")

    assert (tmp_path / "table" / "best" / "synth" / "000_sample" / "ground_truth.krn").exists()


def test_export_validation_table_requires_expected_columns(tmp_path: Path) -> None:
    table_path = tmp_path / "Validation Table.table.json"
    _write_table(table_path, [], columns=["ID", "Category"])

    with pytest.raises(ValueError, match="missing required columns"):
        export_validation_table(table_path, tmp_path / "table")
