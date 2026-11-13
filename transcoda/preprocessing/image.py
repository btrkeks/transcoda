"""Model-image preprocessing utilities for Transcoda.

This module owns the runtime image contract for inference and evaluation:
PIL image in, model-ready normalized tensor out. Dataset code may re-use the
same normalization primitives, but callers should prefer ``ImagePreprocessor``
when preparing images for model generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

# Normalization constants (maps [0,1] -> [-1,1])
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
NORM_MEAN_TENSOR = torch.tensor(NORM_MEAN, dtype=torch.float32).view(3, 1, 1)
NORM_STD_TENSOR = torch.tensor(NORM_STD, dtype=torch.float32).view(3, 1, 1)

# Padding value after normalization (white = 255 uint8 -> 1.0 after normalization).
NORMALIZED_PAD_VALUE = 1.0


@dataclass(frozen=True)
class LayoutNormalizationConfig:
    """Optional content-aware reframing onto a synthetic page canvas."""

    enabled: bool = False
    top_margin_px: int = 36
    side_margin_px: int = 36
    threshold: int = 100
    min_component_area_fraction: float = 0.00002


@dataclass(frozen=True)
class WhitespaceTrimConfig:
    """Content-aware whitespace trimming before model resize/pad."""

    enabled: bool = True
    threshold: int = 235
    margin_px: int = 36
    min_component_area_fraction: float = 0.00002


@dataclass(frozen=True)
class ImagePreprocessingConfig:
    """Configuration for model-image preprocessing."""

    target_width: int
    fixed_size: tuple[int, int] | None = None
    layout_normalization: LayoutNormalizationConfig = field(
        default_factory=LayoutNormalizationConfig
    )
    whitespace_trim: WhitespaceTrimConfig = field(default_factory=WhitespaceTrimConfig)


@dataclass(frozen=True)
class PreprocessedImage:
    """A model-ready image plus preprocessing metadata."""

    pixel_values: torch.Tensor
    model_input_size: tuple[int, int]
    original_size: tuple[int, int]
    debug_metadata: dict[str, Any] = field(default_factory=dict)


def normalize_image(img: np.ndarray) -> torch.Tensor:
    """Normalize an image from uint8 [0,255] to float32 CHW [-1,1]."""
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return (t - NORM_MEAN_TENSOR) / NORM_STD_TENSOR


def image_to_uint8_chw(img: np.ndarray) -> torch.Tensor:
    """Convert a uint8 HWC image to a uint8 CHW tensor without normalizing.

    Counterpart of :func:`normalize_image` for pipelines that defer
    normalization to the GPU via :func:`normalize_uint8_pixel_values`.
    """
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def normalize_uint8_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
    """Normalize uint8 [0,255] CHW or BCHW pixels to float32 [-1,1].

    Same mapping as :func:`normalize_image`, but applicable on any device.
    """
    mean = NORM_MEAN_TENSOR.to(pixel_values.device)
    std = NORM_STD_TENSOR.to(pixel_values.device)
    return (pixel_values.float() / 255.0 - mean) / std


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


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    *,
    margin_px: int,
    image_height: int,
    image_width: int,
) -> tuple[int, int, int, int]:
    top, bottom, left, right = bbox
    margin = max(0, int(margin_px))
    return (
        max(0, top - margin),
        min(image_height - 1, bottom + margin),
        max(0, left - margin),
        min(image_width - 1, right + margin),
    )


def _trim_whitespace(
    image_array: np.ndarray,
    *,
    config: WhitespaceTrimConfig,
    target_width: int,
    target_height: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "enabled": bool(config.enabled),
        "applied": False,
        "threshold": int(config.threshold),
        "margin_px": int(config.margin_px),
        "min_component_area_fraction": float(config.min_component_area_fraction),
    }
    if not config.enabled:
        return image_array, metadata

    mask = _visible_notation_mask(
        image_array,
        threshold=int(config.threshold),
        min_component_area_fraction=float(config.min_component_area_fraction),
    )
    bbox = _content_bbox_from_mask(mask)
    if bbox is None:
        metadata["reason"] = "empty_content"
        return image_array, metadata

    input_height, input_width = image_array.shape[:2]
    expanded_bbox = _expand_bbox(
        bbox,
        margin_px=int(config.margin_px),
        image_height=input_height,
        image_width=input_width,
    )
    top, bottom, left, right = expanded_bbox
    crop = image_array[top : bottom + 1, left : right + 1]
    crop_height, crop_width = crop.shape[:2]
    metadata.update(
        {
            "applied": True,
            "content_bbox": bbox,
            "expanded_bbox": expanded_bbox,
            "crop_size": (int(crop_height), int(crop_width)),
        }
    )

    if target_height is None:
        return crop, metadata

    scale = min(
        int(target_width) / max(1, int(crop_width)),
        int(target_height) / max(1, int(crop_height)),
    )
    resized_width = max(1, min(int(target_width), int(round(crop_width * scale))))
    resized_height = max(1, min(int(target_height), int(round(crop_height * scale))))
    resized = cv2.resize(
        crop,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (int(target_height), int(target_width), image_array.shape[2]),
        255,
        dtype=image_array.dtype,
    )
    paste_x = max(0, (int(target_width) - resized_width) // 2)
    paste_y = 0
    canvas[paste_y : paste_y + resized_height, paste_x : paste_x + resized_width] = resized
    metadata.update(
        {
            "scale": float(scale),
            "resized_size": (int(resized_height), int(resized_width)),
            "paste_offset": (int(paste_y), int(paste_x)),
            "canvas_size": (int(target_height), int(target_width)),
        }
    )
    return canvas, metadata


class ImagePreprocessor:
    """Black-box PIL image to model tensor preprocessor."""

    def __init__(self, config: ImagePreprocessingConfig):
        if config.target_width <= 0:
            raise ValueError(f"target_width must be > 0, got {config.target_width}")
        if config.fixed_size is not None:
            fixed_h, fixed_w = config.fixed_size
            if fixed_h <= 0 or fixed_w <= 0:
                raise ValueError(f"fixed_size must be positive (H, W), got {config.fixed_size}")
        if config.whitespace_trim.margin_px < 0:
            raise ValueError(f"margin_px must be >= 0, got {config.whitespace_trim.margin_px}")
        self.config = config

    @classmethod
    def from_checkpoint_metadata(
        cls,
        *,
        image_width: int,
        fixed_size: tuple[int, int] | None,
        layout_normalization: LayoutNormalizationConfig | None = None,
        whitespace_trim: WhitespaceTrimConfig | None = None,
    ) -> ImagePreprocessor:
        return cls(
            ImagePreprocessingConfig(
                target_width=int(image_width),
                fixed_size=fixed_size,
                layout_normalization=layout_normalization or LayoutNormalizationConfig(),
                whitespace_trim=whitespace_trim or WhitespaceTrimConfig(),
            )
        )

    def __call__(self, image: Image.Image) -> PreprocessedImage:
        """Preprocess a PIL image for model inference/evaluation."""
        original_size = (int(image.height), int(image.width))
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_array = np.array(image)
        target_width = int(self.config.target_width)
        target_height: int | None = None
        if self.config.fixed_size is not None:
            target_height = int(self.config.fixed_size[0])
            target_width = int(self.config.fixed_size[1])

        layout_normalized = False
        if self.config.layout_normalization.enabled:
            image_array = _normalize_layout_to_canvas(
                image_array,
                config=self.config.layout_normalization,
                image_width=self.config.target_width,
                fixed_size=self.config.fixed_size,
            )
            layout_normalized = True
            image = Image.fromarray(image_array)

        whitespace_trimmed = False
        trim_metadata: dict[str, Any] = {"enabled": False, "applied": False}
        if not layout_normalized:
            image_array, trim_metadata = _trim_whitespace(
                image_array,
                config=self.config.whitespace_trim,
                target_width=target_width,
                target_height=target_height,
            )
            whitespace_trimmed = bool(trim_metadata.get("applied", False))
            image = Image.fromarray(image_array)

        if target_height is None or not whitespace_trimmed:
            scale_factor = target_width / image.width
            new_height = max(1, int(image.height * scale_factor))
            image = image.resize((target_width, new_height), Image.Resampling.BILINEAR)
            image_array = np.array(image)

        if target_height is not None and not whitespace_trimmed:
            current_height = image_array.shape[0]
            if current_height > target_height:
                image_array = image_array[:target_height, :, :]
            elif current_height < target_height:
                pad_height = target_height - current_height
                padding = np.full((pad_height, image_array.shape[1], 3), 255, dtype=image_array.dtype)
                image_array = np.concatenate([image_array, padding], axis=0)

        model_input_size = (int(image_array.shape[0]), int(image_array.shape[1]))
        return PreprocessedImage(
            pixel_values=normalize_image(image_array),
            model_input_size=model_input_size,
            original_size=original_size,
            debug_metadata={
                "layout_normalized": layout_normalized,
                "whitespace_trim": trim_metadata,
                "resized_to_width": target_width,
                "fixed_size": self.config.fixed_size,
            },
        )


def preprocess_pil_image(
    image: Image.Image,
    image_width: int,
    fixed_size: tuple[int, int] | None = None,
    layout_normalization: LayoutNormalizationConfig | None = None,
    whitespace_trim: WhitespaceTrimConfig | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Compatibility wrapper returning the historical tuple shape."""
    preprocessor = ImagePreprocessor.from_checkpoint_metadata(
        image_width=image_width,
        fixed_size=fixed_size,
        layout_normalization=layout_normalization,
        whitespace_trim=whitespace_trim,
    )
    preprocessed = preprocessor(image)
    return preprocessed.pixel_values, preprocessed.model_input_size
