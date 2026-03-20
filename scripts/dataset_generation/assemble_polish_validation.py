#!/usr/bin/env python3
"""Compatibility wrapper for legacy all-splits Polish validation assembly."""

from __future__ import annotations

import fire
from scripts.dataset_generation.assemble_polish_scores_dataset import (
    DEFAULT_DATA_SPEC_PATH,
    assemble_polish_scores_dataset,
)


def main(
    normalized_dir: str = "data/interim/val/polish-scores/3_normalized",
    output_dir: str = "data/datasets/validation/polish",
    image_width: int | None = None,
    image_height: int | None = None,
    data_spec_path: str = str(DEFAULT_DATA_SPEC_PATH),
    strict_data_spec: bool = True,
    hf_cache_dir: str | None = None,
    quiet: bool = False,
):
    """Assemble the legacy all-splits Polish validation dataset."""
    return assemble_polish_scores_dataset(
        normalized_dir=normalized_dir,
        output_dir=output_dir,
        include_splits=["train", "val", "test"],
        image_width=image_width,
        image_height=image_height,
        data_spec_path=data_spec_path,
        strict_data_spec=strict_data_spec,
        hf_cache_dir=hf_cache_dir,
        quiet=quiet,
    )


if __name__ == "__main__":
    fire.Fire(main)
