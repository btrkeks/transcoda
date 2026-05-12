import json

import pytest

from scripts.dataset_generation.extract_raw_data.extract_olimpic_transcriptions import (
    extract_olimpic_transcriptions,
    load_split_manifest,
)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_extract_olimpic_transcriptions_writes_split_layout(tmp_path):
    raw_root = tmp_path / "raw"
    (raw_root / "xml").mkdir(parents=True)
    (raw_root / "images").mkdir(parents=True)

    (raw_root / "xml" / "score_a.xml").write_text("<score/>", encoding="utf-8")
    (raw_root / "xml" / "score_b.xml").write_text("<score/>", encoding="utf-8")
    (raw_root / "images" / "score_a.png").write_bytes(b"PNG")
    (raw_root / "images" / "score_b.jpg").write_bytes(b"JPG")

    manifest_path = tmp_path / "split_manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "sample_id": "olimpic-a",
                "doc_id": "doc-train-001",
                "split": "train",
                "xml_relpath": "xml/score_a.xml",
                "image_relpath": "images/score_a.png",
            },
            {
                "sample_id": "olimpic-b",
                "doc_id": "doc-val-001",
                "split": "val",
                "xml_relpath": "xml/score_b.xml",
                "image_relpath": "images/score_b.jpg",
            },
        ],
    )

    output_dir = tmp_path / "interim"
    extract_olimpic_transcriptions(
        split_manifest=str(manifest_path),
        raw_root=str(raw_root),
        output_dir=str(output_dir),
        quiet=True,
    )

    assert (output_dir / "train" / "olimpic" / "0_raw_xml" / "olimpic-a.xml").exists()
    assert (output_dir / "train" / "olimpic" / "0_raw_images" / "olimpic-a.png").exists()
    assert (output_dir / "val" / "olimpic" / "0_raw_xml" / "olimpic-b.xml").exists()
    assert (output_dir / "val" / "olimpic" / "0_raw_images" / "olimpic-b.jpg").exists()

    train_manifest = output_dir / "train" / "olimpic" / "metadata" / "manifest.jsonl"
    val_manifest = output_dir / "val" / "olimpic" / "metadata" / "manifest.jsonl"
    assert train_manifest.exists()
    assert val_manifest.exists()

    train_rows = [json.loads(line) for line in train_manifest.read_text(encoding="utf-8").splitlines()]
    assert len(train_rows) == 1
    assert train_rows[0]["sample_id"] == "olimpic-a"


def test_load_split_manifest_rejects_doc_overlap_across_splits(tmp_path):
    raw_root = tmp_path / "raw"
    (raw_root / "xml").mkdir(parents=True)
    (raw_root / "images").mkdir(parents=True)

    (raw_root / "xml" / "score_a.xml").write_text("<score/>", encoding="utf-8")
    (raw_root / "xml" / "score_b.xml").write_text("<score/>", encoding="utf-8")
    (raw_root / "images" / "score_a.png").write_bytes(b"PNG")
    (raw_root / "images" / "score_b.png").write_bytes(b"PNG")

    manifest_path = tmp_path / "split_manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "sample_id": "olimpic-a",
                "doc_id": "doc-shared",
                "split": "train",
                "xml_path": str(raw_root / "xml" / "score_a.xml"),
                "image_path": str(raw_root / "images" / "score_a.png"),
            },
            {
                "sample_id": "olimpic-b",
                "doc_id": "doc-shared",
                "split": "val",
                "xml_path": str(raw_root / "xml" / "score_b.xml"),
                "image_path": str(raw_root / "images" / "score_b.png"),
            },
        ],
    )

    with pytest.raises(ValueError, match="doc_id overlap across splits"):
        load_split_manifest(split_manifest=str(manifest_path), raw_root=str(raw_root))
