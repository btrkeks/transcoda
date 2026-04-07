from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

if TYPE_CHECKING:
    pass


def tok_targets(batch, tokenizer: PreTrainedTokenizer, text_col: str = "transcription"):
    enc = tokenizer(
        batch[text_col],
        add_special_tokens=True,
    )
    labels = enc["input_ids"]
    return {"labels": labels}


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
    from src.data.preprocessing import normalize_image

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
    # Preserve source field for per-source metric tracking (if present)
    if "source" in ex:
        result["source"] = ex["source"]
    return result


def load_dataset_direct(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    custom_transform: Callable | None = None,
    text_col: str = "transcription",
    image_col: str = "image",
) -> "HFDatasetWrapper":
    """
    Load HuggingFace dataset directly from path (no split subdirectory).

    Args:
        dataset_path: Direct path to the HuggingFace dataset on disk
        tokenizer: Tokenizer for encoding text
        custom_transform: Optional custom transform to apply to images
        text_col: Name of the text column in the dataset
        image_col: Name of the image column in the dataset

    Returns:
        Wrapped dataset compatible with torch DataLoader
    """
    ds = load_from_disk(dataset_path)
    ds = ds.map(
        partial(tok_targets, tokenizer=tokenizer, text_col=text_col),
        batched=True,
    )
    ds.set_transform(
        partial(_hf_transform, image_col=image_col, custom_transform=custom_transform)
    )
    return HFDatasetWrapper(ds)


class HFDatasetWrapper(Dataset):
    """Thin wrapper around HuggingFace dataset for torch DataLoader compatibility."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __getitem__(self, i):
        sample = self.dataset[i]
        sample["sample_id"] = i
        return sample

    def __len__(self):
        return len(self.dataset)
