"""Image preprocessing utilities for OMR pipeline.

This module provides standardized image preprocessing that should be used
consistently across all code paths (training, validation, test, inference).

The normalization maps pixel values from [0, 255] uint8 to [-1, 1] float32,
which is the expected input range for the ConvNeXtV2 encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

# Normalization constants (maps [0,1] -> [-1,1])
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
NORM_MEAN_TENSOR = torch.tensor(NORM_MEAN, dtype=torch.float32).view(3, 1, 1)
NORM_STD_TENSOR = torch.tensor(NORM_STD, dtype=torch.float32).view(3, 1, 1)

# Padding value after normalization (black = 0 uint8 -> -1.0 after normalization)
NORMALIZED_PAD_VALUE = -1.0


@dataclass(frozen=True)
class LayoutNormalizationConfig:
    """Optional content-aware reframing onto a synthetic page canvas."""

    enabled: bool = False
    top_margin_px: int = 36
    side_margin_px: int = 36
    threshold: int = 100
    min_component_area_fraction: float = 0.00002


def normalize_image(img: np.ndarray) -> torch.Tensor:
    """Normalize an image from uint8 [0,255] to float32 [-1,1].

    Args:
        img: Image as numpy array (H, W, C) in uint8 format [0, 255]

    Returns:
        Normalized tensor (C, H, W) as float32 in range [-1, 1]
    """
    # Convert HWC numpy array to CHW tensor and normalize
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return (t - NORM_MEAN_TENSOR) / NORM_STD_TENSOR


def _remove_small_components(mask: np.ndarray, *, min_component_area_fraction: float) -> np.ndarray:
    if mask.size == 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    min_area = max(4, int(round(mask.size * max(0.0, min_component_area_fraction))))
    filtered = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == label] = True
    return filtered


def _visible_notation_mask(
    image_array: np.ndarray,
    *,
    threshold: int,
    min_component_area_fraction: float,
) -> np.ndarray:
    rgb = image_array[:, :, :3]
    mask = np.min(rgb, axis=2) <= threshold
    return _remove_small_components(
        mask,
        min_component_area_fraction=min_component_area_fraction,
    )


def _content_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    top = int(coords[:, 0].min())
    bottom = int(coords[:, 0].max())
    left = int(coords[:, 1].min())
    right = int(coords[:, 1].max())
    return top, bottom, left, right


def _resolve_page_aspect_ratio(
    *,
    image_array: np.ndarray,
    image_width: int,
    fixed_size: tuple[int, int] | None,
) -> float:
    if fixed_size is not None:
        fixed_height = int(fixed_size[0])
        fixed_width = int(fixed_size[1])
        if fixed_height > 0 and fixed_width > 0:
            return fixed_height / fixed_width
    return image_array.shape[0] / max(1, int(image_width))


def _normalize_layout_to_canvas(
    image_array: np.ndarray,
    *,
    config: LayoutNormalizationConfig,
    image_width: int,
    fixed_size: tuple[int, int] | None,
) -> np.ndarray:
    if not config.enabled:
        return image_array

    mask = _visible_notation_mask(
        image_array,
        threshold=config.threshold,
        min_component_area_fraction=config.min_component_area_fraction,
    )
    bbox = _content_bbox_from_mask(mask)
    if bbox is None:
        return image_array

    top, bottom, left, right = bbox
    crop = image_array[top : bottom + 1, left : right + 1]
    content_height, content_width = crop.shape[:2]
    input_height, input_width = image_array.shape[:2]
    top_margin = max(0, int(config.top_margin_px))
    side_margin = max(0, int(config.side_margin_px))
    page_aspect_ratio = _resolve_page_aspect_ratio(
        image_array=image_array,
        image_width=image_width,
        fixed_size=fixed_size,
    )

    required_width = max(input_width, content_width + (2 * side_margin))
    required_height = max(input_height, content_height + (2 * top_margin))
    min_width_for_height = int(np.ceil(required_height / max(page_aspect_ratio, 1e-6)))
    canvas_width = max(required_width, min_width_for_height)
    canvas_height = int(np.ceil(canvas_width * page_aspect_ratio))
    if canvas_height < required_height:
        canvas_height = required_height

    canvas = np.full((canvas_height, canvas_width, image_array.shape[2]), 255, dtype=image_array.dtype)

    dst_top = min(top_margin, max(0, canvas_height - content_height))
    centered_left = max(0, (canvas_width - content_width) // 2)
    min_left = side_margin
    max_left = max(side_margin, canvas_width - content_width - side_margin)
    dst_left = min(max(centered_left, min_left), max_left)
    dst_left = max(0, min(dst_left, canvas_width - content_width))

    canvas[
        dst_top : dst_top + content_height,
        dst_left : dst_left + content_width,
    ] = crop
    return canvas


def preprocess_pil_image(
    image: Image.Image,
    image_width: int,
    fixed_size: tuple[int, int] | None = None,
    layout_normalization: LayoutNormalizationConfig | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Preprocess a PIL image for SMT inference/evaluation.

    Steps:
    1. Convert to RGB
    2. Resize to target width (from checkpoint/data config), preserving aspect ratio
    3. If fixed_size=(H, W) is provided, enforce exact geometry by bottom crop/pad
       after width resize (matching synthetic data generation behavior)
    4. Normalize to float32 CHW in [-1, 1]

    Args:
        image: Input PIL image.
        image_width: Target width used during training.
        fixed_size: Optional fixed model input size as (height, width).
        layout_normalization: Optional synthetic-page reframing before resize/pad.

    Returns:
        Tuple of:
            - Preprocessed tensor (C, H, W)
            - Model input size as (H, W) after all preprocessing
    """
    if image_width <= 0:
        raise ValueError(f"image_width must be > 0, got {image_width}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    image_array = np.array(image)
    if layout_normalization is not None and layout_normalization.enabled:
        image_array = _normalize_layout_to_canvas(
            image_array,
            config=layout_normalization,
            image_width=image_width,
            fixed_size=fixed_size,
        )
        image = Image.fromarray(image_array)

    target_width = int(image_width)
    target_height: int | None = None
    if fixed_size is not None:
        target_height = int(fixed_size[0])
        fixed_width = int(fixed_size[1])
        if target_height <= 0 or fixed_width <= 0:
            raise ValueError(f"fixed_size must be positive (H, W), got {fixed_size}")
        target_width = fixed_width

    scale_factor = target_width / image.width
    new_height = max(1, int(image.height * scale_factor))
    image = image.resize((target_width, new_height), Image.Resampling.BILINEAR)
    image_array = np.array(image)

    if target_height is not None:
        current_height = image_array.shape[0]
        if current_height > target_height:
            image_array = image_array[:target_height, :, :]
        elif current_height < target_height:
            pad_height = target_height - current_height
            padding = np.full((pad_height, image_array.shape[1], 3), 255, dtype=image_array.dtype)
            image_array = np.concatenate([image_array, padding], axis=0)

    model_input_size = (int(image_array.shape[0]), int(image_array.shape[1]))
    return normalize_image(image_array), model_input_size
