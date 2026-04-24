"""Shared helpers for model diagnostics."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.preprocessing import LayoutNormalizationConfig, preprocess_pil_image
from src.model.checkpoint_loader import LoadedCheckpoint, load_model_from_checkpoint


def load(weights: str, device: str = "auto") -> tuple[LoadedCheckpoint, torch.device]:
    """Load a checkpoint for diagnostic inference."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    loaded = load_model_from_checkpoint(weights, dev)
    loaded.model.eval()
    return loaded, dev


def encode_pil_image(
    loaded: LoadedCheckpoint,
    image: Image.Image,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a PIL image into the pixel_values tensor expected by the model."""
    tensor, model_input_size = preprocess_pil_image(
        image=image.convert("RGB"),
        image_width=loaded.image_width,
        fixed_size=loaded.fixed_size,
        layout_normalization=LayoutNormalizationConfig(enabled=False),
    )
    return tensor.unsqueeze(0).to(device), torch.tensor([model_input_size], device=device)


def encode_image_path(
    loaded: LoadedCheckpoint,
    image_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and preprocess an image path."""
    with Image.open(image_path) as image:
        return encode_pil_image(loaded, image, device)
