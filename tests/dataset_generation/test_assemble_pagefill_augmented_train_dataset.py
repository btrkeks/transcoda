import json
from pathlib import Path

import pytest
from datasets import Dataset, load_from_disk
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers

from scripts.dataset_generation.assemble_pagefill_augmented_train_dataset import (
    assemble_pagefill_augmented_train_dataset,
)


def _write_tokenizer(path: Path) -> Path:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return path


def _text(token_count: int, suffix: str) -> str:
    return " ".join(["tok"] * token_count + [suffix])


def _image(color: int) -> Image.Image:
    return Image.new("RGB", (12, 12), color=(color, color, color))


def _build_base_dataset(path: Path) -> None:
    Dataset.from_list(
        [
            {
                "image": _image(255),
                "transcription": _text(2, "base-a"),
                "source": "base_a.krn",
                "source_dataset": "grandstaff",
                "source_split": "train",
                "sample_id": "base_a",
                "curation_stage": "synthetic",
                "source_domain": "synth",
                "actual_system_count": 2,
                "bottom_whitespace_ratio": 0.5,
                "vertical_fill_ratio": 0.4,
            },
            {
                "image": _image(245),
                "transcription": _text(2, "base-b"),
                "source": "base_b.krn",
                "source_dataset": "pdmx",
                "source_split": "train",
                "sample_id": "base_b",
                "curation_stage": "synthetic",
                "source_domain": "synth",
                "actual_system_count": 2,
                "bottom_whitespace_ratio": 0.6,
                "vertical_fill_ratio": 0.35,
            },
        ]
    ).save_to_disk(str(path))


def _build_candidate_dataset(path: Path, duplicate_text: str) -> None:
    Dataset.from_list(
        [
            {
                "image": _image(200),
                "transcription": _text(3, "low-bottom"),
                "sample_id": "cand_low_bottom",
                "source_ids": ["pdmx/piece_low"],
                "bottom_whitespace_ratio": 0.20,
                "vertical_fill_ratio": 0.50,
                "bottom_whitespace_px": 300,
                "content_height_px": 800,
            },
            {
                "image": _image(190),
                "transcription": duplicate_text,
                "sample_id": "cand_base_duplicate",
                "source_ids": ["pdmx/piece_duplicate"],
                "bottom_whitespace_ratio": 0.01,
                "vertical_fill_ratio": 0.99,
                "bottom_whitespace_px": 15,
                "content_height_px": 1300,
            },
            {
                "image": _image(180),
                "transcription": _text(5, "too-long"),
                "sample_id": "cand_too_long",
                "source_ids": ["openscore/piece_long"],
                "bottom_whitespace_ratio": 0.02,
                "vertical_fill_ratio": 0.98,
                "bottom_whitespace_px": 20,
                "content_height_px": 1280,
            },
            {
                "image": _image(170),
                "transcription": _text(3, "low-bottom"),
                "sample_id": "cand_candidate_duplicate",
                "source_ids": ["pdmx/piece_duplicate_candidate"],
                "bottom_whitespace_ratio": 0.05,
                "vertical_fill_ratio": 0.95,
                "bottom_whitespace_px": 75,
                "content_height_px": 1250,
            },
            {
                "image": _image(160),
                "transcription": _text(3, "second"),
                "sample_id": "cand_second",
                "source_ids": ["grandstaff/piece_second"],
                "bottom_whitespace_ratio": 0.25,
                "vertical_fill_ratio": 0.95,
                "bottom_whitespace_px": 375,
                "content_height_px": 850,
            },
            {
                "image": _image(150),
                "transcription": _text(3, "missing-layout"),
                "sample_id": "cand_missing_layout",
                "source_ids": ["musetrainer/piece_missing"],
                "bottom_whitespace_ratio": None,
                "vertical_fill_ratio": 0.99,
                "bottom_whitespace_px": None,
                "content_height_px": None,
            },
        ]
    ).save_to_disk(str(path))


def test_assemble_pagefill_augmented_train_dataset_filters_ranks_and_writes_metadata(tmp_path: Path):
    base_dir = tmp_path / "train_full_2"
    candidate_dir = tmp_path / "train_200k"
    output_dir = tmp_path / "train_full_2_plus_pagefill"
    metadata_out = tmp_path / "metadata.json"
    tokenizer_path = _write_tokenizer(tmp_path / "tokenizer.json")

    _build_base_dataset(base_dir)
    base_duplicate_text = load_from_disk(str(base_dir))[0]["transcription"]
    _build_candidate_dataset(candidate_dir, duplicate_text=base_duplicate_text)

    metadata = assemble_pagefill_augmented_train_dataset(
        base_dataset_path=str(base_dir),
        candidate_dataset_path=str(candidate_dir),
        output_dir=str(output_dir),
        target_add_count=2,
        max_seq_len=4,
        tokenizer=str(tokenizer_path),
        metadata_out=str(metadata_out),
        quiet=True,
    )

    mixed = load_from_disk(str(output_dir))
    assert len(mixed) == 4
    assert mixed["mix_component"] == ["base", "base", "pagefill", "pagefill"]
    assert mixed["sample_id"][2:] == ["pagefill_cand_low_bottom", "pagefill_cand_second"]
    assert mixed["source_dataset"][2:] == ["pdmx", "grandstaff"]
    assert mixed["original_index"][2:] == [0, 4]
    assert max(mixed["token_length"]) <= 4
    assert "bottom_whitespace_px" in mixed.column_names
    assert "content_height_px" in mixed.column_names

    assert metadata["selection_counts"]["dropped_over_max_seq_len"] == 1
    assert metadata["selection_counts"]["dropped_duplicate_with_base"] == 1
    assert metadata["selection_counts"]["dropped_duplicate_with_candidates"] == 1
    assert metadata["selection_counts"]["dropped_missing_layout"] == 1
    assert metadata["selection_counts"]["selected_pagefill_rows"] == 2

    written = json.loads(metadata_out.read_text(encoding="utf-8"))
    assert written["total_rows"] == 4
    assert written["pagefill_summary"]["source_dataset_counts"] == {
        "grandstaff": 1,
        "pdmx": 1,
    }


def test_assemble_pagefill_augmented_train_dataset_refuses_existing_output(tmp_path: Path):
    base_dir = tmp_path / "train_full_2"
    candidate_dir = tmp_path / "train_200k"
    output_dir = tmp_path / "existing"
    tokenizer_path = _write_tokenizer(tmp_path / "tokenizer.json")
    _build_base_dataset(base_dir)
    base_duplicate_text = load_from_disk(str(base_dir))[0]["transcription"]
    _build_candidate_dataset(candidate_dir, duplicate_text=base_duplicate_text)
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        assemble_pagefill_augmented_train_dataset(
            base_dataset_path=str(base_dir),
            candidate_dataset_path=str(candidate_dir),
            output_dir=str(output_dir),
            target_add_count=1,
            max_seq_len=4,
            tokenizer=str(tokenizer_path),
            quiet=True,
        )
