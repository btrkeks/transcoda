import json

from datasets import load_from_disk
from PIL import Image

from scripts.dataset_generation.assemble_olimpic_dataset import assemble_olimpic_dataset


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_assemble_olimpic_dataset_builds_fixed_size_dataset(tmp_path):
    normalized_dir = tmp_path / "interim" / "train" / "olimpic" / "3_normalized"
    normalized_dir.mkdir(parents=True)
    (normalized_dir / "sample_001.krn").write_text("**kern\n*-\n", encoding="utf-8")

    image_path = tmp_path / "raw" / "sample_001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (120, 80), color=(255, 255, 255)).save(image_path)

    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "sample_001",
                "doc_id": "doc-001",
                "split": "train",
                "image_path": str(image_path),
            }
        ],
    )

    output_dir = tmp_path / "datasets" / "train_olimpic_scanned"
    assemble_olimpic_dataset(
        normalized_dir=str(normalized_dir),
        manifest_path=str(manifest),
        output_dir=str(output_dir),
        source_name="olimpic",
        image_width=1050,
        image_height=1485,
        strict_data_spec=False,
        quiet=True,
    )

    ds = load_from_disk(str(output_dir))
    assert len(ds) == 1
    assert ds[0]["source"] == "olimpic"
    assert ds[0]["image"].size == (1050, 1485)


def test_assemble_olimpic_dataset_skips_samples_without_manifest_mapping(tmp_path):
    normalized_dir = tmp_path / "interim" / "train" / "olimpic" / "3_normalized"
    normalized_dir.mkdir(parents=True)
    (normalized_dir / "sample_001.krn").write_text("**kern\n*-\n", encoding="utf-8")
    (normalized_dir / "sample_002.krn").write_text("**kern\n*-\n", encoding="utf-8")

    image_path = tmp_path / "raw" / "sample_001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(image_path)

    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "sample_001",
                "doc_id": "doc-001",
                "split": "train",
                "image_path": str(image_path),
            }
        ],
    )

    output_dir = tmp_path / "datasets" / "train_olimpic_scanned"
    assemble_olimpic_dataset(
        normalized_dir=str(normalized_dir),
        manifest_path=str(manifest),
        output_dir=str(output_dir),
        image_width=1050,
        image_height=1485,
        strict_data_spec=False,
        quiet=True,
    )

    ds = load_from_disk(str(output_dir))
    assert len(ds) == 1
    assert ds[0]["transcription"].startswith("**kern")
