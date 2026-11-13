from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from transcoda.core.sample_identity import (
    sample_id_as_string,
    validate_sample_revision_digest,
)

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class SequenceLengthFilterStats:
    max_target_len: int
    before_count: int
    after_count: int
    removed_count: int

    @property
    def removed_rate(self) -> float:
        return self.removed_count / self.before_count if self.before_count else 0.0


def tok_targets(
    batch_or_texts,
    indices: list[int] | None = None,
    *,
    tokenizer: PreTrainedTokenizer,
    text_col: str = "transcription",
):
    if isinstance(batch_or_texts, dict):
        texts = batch_or_texts[text_col]
    else:
        texts = batch_or_texts
    enc = tokenizer(
        texts,
        add_special_tokens=True,
    )
    result = {"labels": enc["input_ids"]}
    if indices is not None:
        result["source_row_index"] = list(indices)
    return result


def _target_lengths_within_limit(
    labels_batch: list[list[int]],
    *,
    max_target_len: int,
) -> list[bool]:
    return [len(labels) <= max_target_len for labels in labels_batch]


def _hf_transform(ex, image_col: str, custom_transform: Callable | None = None):
    """
    Transform function for HuggingFace dataset.

    Args:
        ex: Batch from dataset
        image_col: Name of the image column
        custom_transform: Optional custom transform to apply instead of default normalization

    Returns:
        Dict with pixel_values and labels
    """
    from transcoda.data.preprocessing import normalize_image

    images = ex[image_col]
    pixel_values = []
    for img in images:
        if hasattr(img, "convert"):
            img = np.array(img.convert("RGB"))
        else:
            img = np.array(img)

        # Use custom transform if provided, otherwise apply standard normalization
        if custom_transform is not None:
            t = custom_transform(img)
        else:
            # Normalize: uint8 [0,255] -> float32 [-1,1]
            t = normalize_image(img)

        pixel_values.append(t)
    result = {"pixel_values": pixel_values, "labels": ex["labels"]}
    result["source"] = _derive_source_batch(ex)
    if "source_row_index" in ex:
        result["source_row_index"] = ex["source_row_index"]
    if "sample_id" in ex:
        result["dataset_sample_id"] = [
            sample_id_as_string(value) for value in ex["sample_id"]
        ]
    if "sample_revision_digest" in ex:
        result["sample_revision_digest"] = [
            validate_sample_revision_digest(value, allow_none=False)
            for value in ex["sample_revision_digest"]
        ]
    return result


def _derive_source_batch(ex) -> list[str]:
    if "source" in ex:
        return [str(value) for value in ex["source"]]
    if "source_ids" in ex:
        derived: list[str] = []
        source_ids = ex["source_ids"]
        sample_ids = ex.get("sample_id", [f"sample_{idx}" for idx in range(len(source_ids))])
        for idx, value in enumerate(source_ids):
            if value:
                derived.append(str(value[0]))
            else:
                derived.append(str(sample_ids[idx]))
        return derived
    if "sample_id" in ex:
        return [str(value) for value in ex["sample_id"]]
    row_count = len(ex.get("labels", []))
    return ["unknown"] * row_count


def load_dataset_direct(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    custom_transform: Callable | None = None,
    text_col: str = "transcription",
    image_col: str = "image",
    max_target_len: int | None = None,
    sample_limit: int | None = None,
) -> "HFDatasetWrapper":
    """
    Load HuggingFace dataset directly from path (no split subdirectory).

    Args:
        dataset_path: Direct path to the HuggingFace dataset on disk
        tokenizer: Tokenizer for encoding text
        custom_transform: Optional custom transform to apply to images
        text_col: Name of the text column in the dataset
        image_col: Name of the image column in the dataset
        max_target_len: Optional maximum tokenized target length, inclusive
        sample_limit: Optional prefix row limit applied before tokenization

    Returns:
        Wrapped dataset compatible with torch DataLoader
    """
    ds = load_from_disk(dataset_path)
    if sample_limit is not None:
        sample_limit = int(sample_limit)
        if sample_limit < 1:
            raise ValueError("sample_limit must be >= 1")
        ds = ds.select(range(min(sample_limit, len(ds))))
    ds = ds.map(
        partial(tok_targets, tokenizer=tokenizer, text_col=text_col),
        input_columns=[text_col],
        batched=True,
        with_indices=True,
    )
    filter_stats = None
    if max_target_len is not None:
        max_target_len = int(max_target_len)
        before_count = len(ds)
        ds = ds.filter(
            partial(_target_lengths_within_limit, max_target_len=max_target_len),
            input_columns=["labels"],
            batched=True,
        )
        after_count = len(ds)
        filter_stats = SequenceLengthFilterStats(
            max_target_len=max_target_len,
            before_count=before_count,
            after_count=after_count,
            removed_count=before_count - after_count,
        )
    ds.set_transform(partial(_hf_transform, image_col=image_col, custom_transform=custom_transform))
    return HFDatasetWrapper(ds, sequence_length_filter=filter_stats)


class HFDatasetWrapper(Dataset):
    """Thin wrapper around HuggingFace dataset for torch DataLoader compatibility."""

    def __init__(
        self,
        dataset,
        sequence_length_filter: SequenceLengthFilterStats | None = None,
    ) -> None:
        self.dataset = dataset
        self.sequence_length_filter = sequence_length_filter

    def __getitem__(self, i):
        sample = self.dataset[i]
        source_row_index = int(sample.get("source_row_index", i))
        if "dataset_sample_id" not in sample and "sample_id" in sample:
            sample["dataset_sample_id"] = str(sample["sample_id"])
        sample["source_row_index"] = source_row_index
        sample["sample_id"] = source_row_index
        return sample

    def __len__(self):
        return len(self.dataset)
