from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import normalize_image
from src.pretraining.fcmae.config import FCMAEDataConfig

log = logging.getLogger(__name__)
NORMALIZED_WHITE_VALUE = 1.0


def _drop_empty_files(paths: list[Path]) -> list[Path]:
    kept = []
    dropped = []
    for path in paths:
        try:
            is_empty = path.is_file() and path.stat().st_size == 0
        except OSError:
            is_empty = False
        if is_empty:
            dropped.append(path)
        else:
            kept.append(path)
    if dropped:
        log.warning("Skipping %d empty FCMAE image file(s), e.g. %s", len(dropped), dropped[0])
    return kept


def _collect_image_paths(config: FCMAEDataConfig) -> list[Path]:
    extensions = {extension.lower() for extension in config.extensions}
    if config.image_dir is not None:
        image_dir = Path(config.image_dir).expanduser()
        paths = [
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ]
        return _drop_empty_files(sorted(paths))

    if config.manifest_path is None:
        raise ValueError("manifest_path is required when image_dir is not set")
    manifest_path = Path(config.manifest_path).expanduser()
    base_dir = manifest_path.parent
    paths = []
    for raw_line in manifest_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        paths.append(path)
    return _drop_empty_files(paths)


def fit_image_to_canvas(
    image: Image.Image,
    *,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a full RGB image into a white canvas and return valid pixels.

    FCMAE pretraining deliberately keeps the supervised OMR `[-1, 1]` pixel
    convention. The resize is full-image fit with right/bottom white padding,
    so no systems are silently cropped from arbitrary scan folders.
    """
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be positive")
    if image.mode != "RGB":
        image = image.convert("RGB")

    scale = min(image_width / image.width, image_height / image.height)
    resized_width = max(1, int(round(image.width * scale)))
    resized_height = max(1, int(round(image.height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

    canvas = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
    canvas[:resized_height, :resized_width, :] = np.asarray(resized, dtype=np.uint8)
    valid_pixel_mask = torch.zeros(image_height, image_width, dtype=torch.bool)
    valid_pixel_mask[:resized_height, :resized_width] = True
    return normalize_image(canvas), valid_pixel_mask


def valid_pixel_mask_to_patch_mask(valid_pixel_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """A patch is valid iff all pixels in that patch are valid content."""
    if valid_pixel_mask.ndim != 2:
        raise ValueError("valid_pixel_mask must have shape (H, W)")
    height, width = valid_pixel_mask.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("valid_pixel_mask dimensions must be divisible by patch_size")
    invalid = ~valid_pixel_mask
    invalid = invalid.reshape(height // patch_size, patch_size, width // patch_size, patch_size)
    invalid_patch = invalid.any(dim=(1, 3))
    return ~invalid_patch


def pad_to_patch_multiple(
    pixel_values: torch.Tensor,
    valid_pixel_mask: torch.Tensor,
    patch_size: int,
    alignment_multiple: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right/bottom pad pixels and validity mask to a patch/stride-aligned canvas."""
    if pixel_values.ndim != 3:
        raise ValueError("pixel_values must have shape (C, H, W)")
    if valid_pixel_mask.ndim != 2:
        raise ValueError("valid_pixel_mask must have shape (H, W)")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if alignment_multiple is None:
        alignment_multiple = patch_size
    if alignment_multiple <= 0:
        raise ValueError("alignment_multiple must be positive")
    if alignment_multiple % patch_size != 0:
        raise ValueError("alignment_multiple must be divisible by patch_size")

    height, width = pixel_values.shape[-2:]
    if valid_pixel_mask.shape != (height, width):
        raise ValueError("valid_pixel_mask shape must match pixel_values spatial dimensions")

    padded_height = ((height + alignment_multiple - 1) // alignment_multiple) * alignment_multiple
    padded_width = ((width + alignment_multiple - 1) // alignment_multiple) * alignment_multiple
    pad_h = padded_height - height
    pad_w = padded_width - width
    if pad_h == 0 and pad_w == 0:
        return pixel_values, valid_pixel_mask

    pixel_values = F.pad(
        pixel_values,
        (0, pad_w, 0, pad_h),
        value=NORMALIZED_WHITE_VALUE,
    )
    valid_pixel_mask = F.pad(valid_pixel_mask, (0, pad_w, 0, pad_h), value=False)
    return pixel_values, valid_pixel_mask


def compute_patch_ink_density(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Mean per-patch ink density in [0, 1], with 1 = fully black.

    ``pixel_values`` is the normalized CHW tensor in ``[-1, 1]`` produced by
    ``normalize_image``; intensity is recovered as ``(value + 1) / 2``.
    """
    if pixel_values.ndim != 3:
        raise ValueError("pixel_values must have shape (C, H, W)")
    height, width = pixel_values.shape[-2:]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("pixel_values dimensions must be divisible by patch_size")
    intensity = (pixel_values.float().mean(dim=0) + 1.0) / 2.0
    ink = (1.0 - intensity).clamp_(0.0, 1.0)
    grid_h = height // patch_size
    grid_w = width // patch_size
    return ink.reshape(grid_h, patch_size, grid_w, patch_size).mean(dim=(1, 3))


class FCMAEImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        config: FCMAEDataConfig,
        *,
        patch_size: int,
        encoder_stride: int | None = None,
    ) -> None:
        self.config = config
        self.patch_size = patch_size
        self.encoder_stride = encoder_stride or patch_size
        self.alignment_multiple = math.lcm(self.patch_size, self.encoder_stride)
        self.paths = _collect_image_paths(config)
        if not self.paths:
            source = config.image_dir if config.image_dir is not None else config.manifest_path
            raise ValueError(f"no images found for FCMAE pretraining source: {source}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        with Image.open(path) as image:
            pixel_values, valid_pixel_mask = fit_image_to_canvas(
                image,
                image_height=self.config.image_height,
                image_width=self.config.image_width,
            )
        pixel_values, valid_pixel_mask = pad_to_patch_multiple(
            pixel_values,
            valid_pixel_mask,
            self.patch_size,
            self.alignment_multiple,
        )
        valid_patch_mask = valid_pixel_mask_to_patch_mask(valid_pixel_mask, self.patch_size)
        return {
            "pixel_values": pixel_values,
            "valid_patch_mask": valid_patch_mask,
            "path": str(path),
        }


class FCMAEDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_config: FCMAEDataConfig,
        *,
        patch_size: int,
        encoder_stride: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.patch_size = patch_size
        self.encoder_stride = encoder_stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set: FCMAEImageDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_set is None:
            self.train_set = FCMAEImageDataset(
                self.data_config,
                patch_size=self.patch_size,
                encoder_stride=self.encoder_stride,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            self.setup("fit")
        assert self.train_set is not None
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
