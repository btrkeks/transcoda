"""Collator and dataset wrapper for batched evaluation."""

from __future__ import annotations

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.preprocessing import NORMALIZED_PAD_VALUE


class EvalCollator:
    """Collates preprocessed samples for batched evaluation.

    Unlike the training collator, this does NOT handle:
    - Tokenization (no labels needed for generation)
    - Decoder attention masks

    It DOES handle:
    - Padding images to uniform H/W within batch
    - Tracking image_sizes for encoder masking
    - Passing through transcription and source strings
    """

    def __init__(self, image_pad_value: float = NORMALIZED_PAD_VALUE):
        """Initialize collator.

        Args:
            image_pad_value: Value to use for image padding (default -1.0 = black after normalization)
        """
        self.image_pad_value = image_pad_value

    def __call__(self, batch: list[dict]) -> dict:
        """Collate a batch of preprocessed samples.

        Args:
            batch: List of dicts with keys:
                - pixel_values: Tensor (C, H, W) - preprocessed image
                - transcription: str - ground truth text
                - source: str - data source identifier
                - sample_id: int - original dataset index

        Returns:
            Dict with keys:
                - pixel_values: Tensor (B, C, H, W) - padded batch
                - image_sizes: Tensor (B, 2) - original (H, W) per sample
                - transcription: list[str] - ground truth texts
                - source: list[str] - data sources
                - sample_id: list[int] - original indices
        """
        # Extract image tensors
        imgs = [b["pixel_values"] for b in batch]

        # Find max dimensions
        Hmax = max(t.shape[1] for t in imgs)
        Wmax = max(t.shape[2] for t in imgs)

        # Pad images and track original sizes
        padded_imgs = []
        image_sizes = []
        for t in imgs:
            orig_h, orig_w = t.shape[1], t.shape[2]
            image_sizes.append((orig_h, orig_w))

            pad_h = Hmax - orig_h
            pad_w = Wmax - orig_w
            # Pad (left, right, top, bottom) = (0, pad_w, 0, pad_h)
            padded = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=self.image_pad_value)
            padded_imgs.append(padded)

        pixel_values = torch.stack(padded_imgs, dim=0)  # (B, C, H, W)
        image_sizes_tensor = torch.tensor(image_sizes, dtype=torch.long)  # (B, 2)

        return {
            "pixel_values": pixel_values,
            "image_sizes": image_sizes_tensor,
            "transcription": [b["transcription"] for b in batch],
            "source": [b["source"] for b in batch],
            "sample_id": [b["sample_id"] for b in batch],
        }


class EvalDatasetWrapper(Dataset):
    """Wraps HuggingFace dataset for evaluation with preprocessing.

    This wrapper applies model-specific preprocessing and tracks sample metadata
    for the EvalCollator.
    """

    def __init__(self, hf_dataset, preprocess_fn):
        """Initialize wrapper.

        Args:
            hf_dataset: HuggingFace dataset with 'image', 'transcription', 'source' columns
            preprocess_fn: Function to preprocess PIL Image -> Tensor
        """
        self.dataset = hf_dataset
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get a preprocessed sample.

        Returns:
            Dict with pixel_values, transcription, source, sample_id
        """
        sample = self.dataset[idx]
        image: Image.Image = sample["image"]

        return {
            "pixel_values": self.preprocess_fn(image),
            "transcription": sample["transcription"],
            "source": sample.get("source", "unknown"),
            "sample_id": idx,
        }
