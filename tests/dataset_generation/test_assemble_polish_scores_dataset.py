from datasets import load_from_disk
from PIL import Image

from scripts.dataset_generation import assemble_polish_scores_dataset as apsd


def test_assemble_polish_scores_dataset_filters_splits_and_writes_metadata(tmp_path, monkeypatch):
    normalized_dir = tmp_path / "interim" / "val" / "polish-scores" / "4_manual_fixes"
    normalized_dir.mkdir(parents=True)
    (normalized_dir / "train_000001.krn").write_text("**kern\n4c\n*-\n", encoding="utf-8")
    (normalized_dir / "val_000000.krn").write_text("**kern\n4d\n*-\n", encoding="utf-8")

    upstream = {
        "train": [
            {"image": Image.new("RGB", (100, 80), color=(255, 255, 255))}
            for _ in range(2)
        ],
        "val": [{"image": Image.new("RGB", (100, 80), color=(240, 240, 240))}],
    }
    monkeypatch.setattr(apsd, "load_dataset", lambda *args, **kwargs: upstream)

    output_dir = tmp_path / "datasets" / "train_polish_adapt"
    apsd.assemble_polish_scores_dataset(
        normalized_dir=str(normalized_dir),
        output_dir=str(output_dir),
        include_splits="train",
        image_width=1050,
        image_height=1485,
        strict_data_spec=False,
        quiet=True,
    )

    ds = load_from_disk(str(output_dir))
    assert len(ds) == 1
    assert ds.column_names == [
        "image",
        "transcription",
        "source",
        "source_dataset",
        "source_split",
        "sample_id",
        "curation_stage",
        "source_domain",
    ]
    assert ds[0]["source"] == "train_000001.krn"
    assert ds[0]["source_split"] == "train"
    assert ds[0]["sample_id"] == "train_000001"
    assert ds[0]["curation_stage"] == "manual_fix"
    assert ds[0]["source_domain"] == "real"
    assert ds[0]["image"].size == (1050, 1485)


def test_assemble_polish_scores_dataset_passes_hf_cache_dir(tmp_path, monkeypatch):
    normalized_dir = tmp_path / "interim" / "val" / "polish-scores" / "4_manual_fixes"
    normalized_dir.mkdir(parents=True)
    (normalized_dir / "train_000000.krn").write_text("**kern\n4c\n*-\n", encoding="utf-8")

    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {"train": [{"image": Image.new("RGB", (20, 20), color=(255, 255, 255))}]}

    monkeypatch.setattr(apsd, "load_dataset", fake_load_dataset)

    apsd.assemble_polish_scores_dataset(
        normalized_dir=str(normalized_dir),
        output_dir=str(tmp_path / "datasets" / "train_polish_adapt"),
        include_splits="train",
        image_width=1050,
        image_height=1485,
        strict_data_spec=False,
        hf_cache_dir=str(tmp_path / "hf-cache"),
        quiet=True,
    )

    assert len(calls) == 1
    assert calls[0]["args"] == (apsd.UPSTREAM_DATASET_NAME,)
    assert calls[0]["kwargs"]["cache_dir"] == str(tmp_path / "hf-cache")


def test_assemble_polish_scores_dataset_rejects_unknown_splits(tmp_path):
    normalized_dir = tmp_path / "interim" / "val" / "polish-scores" / "4_manual_fixes"
    normalized_dir.mkdir(parents=True)
    (normalized_dir / "train_000000.krn").write_text("**kern\n4c\n*-\n", encoding="utf-8")

    try:
        apsd.assemble_polish_scores_dataset(
            normalized_dir=str(normalized_dir),
            output_dir=str(tmp_path / "datasets" / "bad"),
            include_splits="train,dev",
            image_width=1050,
            image_height=1485,
            strict_data_spec=False,
            quiet=True,
        )
    except ValueError as exc:
        assert "Unsupported split" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported split")
