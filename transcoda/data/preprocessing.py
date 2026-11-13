"""Backward-compatible imports for image preprocessing.

Runtime model preprocessing lives in :mod:`transcoda.preprocessing`. This module is
kept as a thin shim for older dataset/training imports.
"""

from transcoda.preprocessing import (
    NORM_MEAN,
    NORM_STD,
    NORMALIZED_PAD_VALUE,
    ImagePreprocessingConfig,
    ImagePreprocessor,
    LayoutNormalizationConfig,
    PreprocessedImage,
    WhitespaceTrimConfig,
    normalize_image,
    preprocess_pil_image,
)

__all__ = [
    "ImagePreprocessingConfig",
    "ImagePreprocessor",
    "LayoutNormalizationConfig",
    "NORMALIZED_PAD_VALUE",
    "NORM_MEAN",
    "NORM_STD",
    "PreprocessedImage",
    "WhitespaceTrimConfig",
    "normalize_image",
    "preprocess_pil_image",
]
