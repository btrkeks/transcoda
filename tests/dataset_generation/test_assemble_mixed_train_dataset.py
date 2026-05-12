import json

from datasets import Dataset, load_from_disk
from PIL import Image

from scripts.dataset_generation.assemble_mixed_train_dataset import (
    assemble_mixed_train_dataset,
)


def _build_dataset(rows):
    return Dataset.from_list(rows)


def test_assemble_mixed_train_dataset_is_deterministic_and_adds_metadata(tmp_path):
    synth_dir = tmp_path / "datasets" / "train_full"
    real_dir = tmp_path / "datasets" / "train_polish_adapt"
    out_a = tmp_path / "datasets" / "mixed_a"
    out_b = tmp_path / "datasets" / "mixed_b"
    meta_a = tmp_path / "reports" / "a.json"
    meta_b = tmp_path / "reports" / "b.json"

    synth_ds = _build_dataset(
        [
            {
                "image": Image.new("RGB", (20, 20), color=(255, 255, 255)),
                "transcription": f"**kern\n4c{i}\n*-\n",
                "sample_id": f"sample_{i:08d}",
                "source_ids": [f"train/full_{i:03d}"],
                "segment_count": 1,
                "source_measure_count": 2,
                "source_non_empty_line_count": 4,
                "svg_system_count": 4,
                "truncation_applied": False,
                "truncation_reason": None,
                "truncation_ratio": None,
                "vertical_fill_ratio": 0.5,
                "bottom_whitespace_ratio": 0.2,
                "recipe_version": "v1",
            }
            for i in range(5)
        ]
    )
    synth_ds.save_to_disk(str(synth_dir))

    real_ds = _build_dataset(
        [
            {
                "image": Image.new("RGB", (20, 20), color=(240, 240, 240)),
                "transcription": f"**kern\n4d{i}\n*-\n",
                "source": f"train_00000{i}.krn",
                "source_dataset": "PRAIG/polish-scores",
                "source_split": "train",
                "sample_id": f"train_00000{i}",
                "curation_stage": "manual_fix",
                "source_domain": "real",
            }
            for i in range(2)
        ]
    )
    real_ds.save_to_disk(str(real_dir))

    assemble_mixed_train_dataset(
        synth_dataset_path=str(synth_dir),
        real_dataset_path=str(real_dir),
        output_dir=str(out_a),
        target_synth_count=3,
        target_real_count=4,
        seed=7,
        metadata_out=str(meta_a),
        quiet=True,
    )
    assemble_mixed_train_dataset(
        synth_dataset_path=str(synth_dir),
        real_dataset_path=str(real_dir),
        output_dir=str(out_b),
        target_synth_count=3,
        target_real_count=4,
        seed=7,
        metadata_out=str(meta_b),
        quiet=True,
    )

    ds_a = load_from_disk(str(out_a))
    ds_b = load_from_disk(str(out_b))
    assert len(ds_a) == 7
    assert len(ds_b) == 7
    assert ds_a["source"] == ds_b["source"]
    assert ds_a["source_domain"].count("synth") == 3
    assert ds_a["source_domain"].count("real") == 4
    assert all(name in ds_a.column_names for name in [
        "source",
        "source_dataset",
        "source_split",
        "sample_id",
        "curation_stage",
        "source_domain",
    ])

    meta_payload = json.loads(meta_a.read_text(encoding="utf-8"))
    assert meta_payload["target_synth_count"] == 3
    assert meta_payload["target_real_count"] == 4
    assert len(meta_payload["synth_sample_indices"]) == 3
    assert sorted(meta_payload["real_repeat_histogram"].values()) == [2, 2]
