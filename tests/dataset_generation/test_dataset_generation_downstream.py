from pathlib import Path

from datasets import Dataset
from PIL import Image

from scripts.dataset_generation.assemble_mixed_train_dataset import (
    assemble_mixed_train_dataset,
)
from src.data.datasets import load_dataset_direct
from src.evaluation.collator import EvalDatasetWrapper


class _FakeTokenizer:
    def __call__(self, values, add_special_tokens=True):
        del add_special_tokens
        return {"input_ids": [[1, 2, 3] for _ in values]}


def test_load_dataset_direct_derives_source_from_source_ids(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    ds = Dataset.from_list(
        [
            {
                "image": Image.new("RGB", (20, 20), color=(255, 255, 255)),
                "transcription": "**kern\n4c\n*-\n",
                "sample_id": "sample_00000000",
                "source_ids": ["train/a_piece"],
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
        ]
    )
    ds.save_to_disk(str(dataset_dir))

    wrapped = load_dataset_direct(str(dataset_dir), tokenizer=_FakeTokenizer())
    sample = wrapped[0]

    assert sample["source"] == "train/a_piece"


def test_eval_dataset_wrapper_derives_source_from_source_ids():
    dataset = [
        {
            "image": Image.new("RGB", (20, 20), color=(255, 255, 255)),
            "transcription": "**kern\n4c\n*-\n",
            "source_ids": ["train/a_piece"],
            "sample_id": "sample_00000000",
        }
    ]
    wrapped = EvalDatasetWrapper(dataset, preprocess_fn=lambda image: image)

    sample = wrapped[0]

    assert sample["source"] == "train/a_piece"


def test_assemble_mixed_train_dataset_accepts_rewrite_schema(tmp_path: Path):
    synth_dir = tmp_path / "datasets" / "train_full"
    real_dir = tmp_path / "datasets" / "train_polish_adapt"
    out_dir = tmp_path / "datasets" / "mixed"
    meta_out = tmp_path / "reports" / "mixed.json"

    synth_ds = Dataset.from_list(
        [
            {
                "image": Image.new("RGB", (20, 20), color=(255, 255, 255)),
                "transcription": f"**kern\n4c{i}\n*-\n",
                "sample_id": f"sample_{i:08d}",
                "source_ids": [f"train/piece_{i:03d}"],
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
            for i in range(3)
        ]
    )
    synth_ds.save_to_disk(str(synth_dir))

    real_ds = Dataset.from_list(
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
        output_dir=str(out_dir),
        target_synth_count=2,
        target_real_count=2,
        metadata_out=str(meta_out),
        quiet=True,
    )

    mixed = Dataset.load_from_disk(str(out_dir))
    assert "source" in mixed.column_names
    assert mixed["source_domain"].count("synth") == 2
