from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

ImageU8 = npt.NDArray[np.uint8]

_WHITE_RGB = (255, 255, 255)


@dataclass(frozen=True)
class GeometricTransform:
    """Sampled geometric transform shared across page and foreground branches."""

    affine: np.ndarray
    perspective: np.ndarray | None = None


def transform_matrix(transform: GeometricTransform | None) -> np.ndarray | None:
    """Return the composed 3x3 forward transform matrix."""
    if transform is None:
        return None
    affine3 = np.vstack([transform.affine, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    if transform.perspective is None:
        return affine3.astype(np.float32)
    return (transform.perspective @ affine3).astype(np.float32)


def transform_points(points: np.ndarray, transform: GeometricTransform | None) -> np.ndarray:
    """Apply the forward transform to 2D points."""
    if transform is None:
        return points.astype(np.float32, copy=True)
    matrix = transform_matrix(transform)
    assert matrix is not None
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    warped_h = (matrix @ points_h.T).T
    denom = np.clip(warped_h[:, 2:3], 1e-6, None)
    return (warped_h[:, :2] / denom).astype(np.float32)


def inverse_transform_points(points: np.ndarray, transform: GeometricTransform | None) -> np.ndarray:
    """Map output-space 2D points back into input-space coordinates."""
    if transform is None:
        return points.astype(np.float32, copy=True)
    matrix = transform_matrix(transform)
    assert matrix is not None
    inv_matrix = np.linalg.inv(matrix.astype(np.float64)).astype(np.float32)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    warped_h = (inv_matrix @ points_h.T).T
    denom = np.clip(warped_h[:, 2:3], 1e-6, None)
    return (warped_h[:, :2] / denom).astype(np.float32)


def sample_geometric_transform(
    image_shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    conservative: bool = False,
    x_squeeze_prob: float = 0.45,
    x_squeeze_min_scale: float = 0.70,
    x_squeeze_max_scale: float = 0.95,
    x_squeeze_apply_in_conservative: bool = True,
    x_squeeze_force_scale: float | None = None,
) -> GeometricTransform | None:
    """Sample a scanner-like transform while preserving the output size."""
    height, width = image_shape
    if width < 2 or height < 2:
        return None

    if conservative:
        scale_range = (0.95, 1.08)
        translation_frac = 0.025
        rotation_deg = 1.2
        perspective_prob = 0.0
        perspective_frac = 0.0
        stretch_prob = 0.0
        stretch_frac = 0.0
    else:
        scale_range = (0.90, 1.15)
        translation_frac = 0.04
        rotation_deg = 1.8
        perspective_prob = 0.30
        perspective_frac = 0.015
        stretch_prob = 0.20
        stretch_frac = 0.03

    if x_squeeze_force_scale is not None:
        scale_range = (1.0, 1.0)
        translation_frac = 0.0
        rotation_deg = 0.0
        perspective_prob = 0.0
        perspective_frac = 0.0
        stretch_prob = 0.0
        stretch_frac = 0.0

    if x_squeeze_force_scale is None and rng.random() >= 0.85:
        return None

    scale = float(np.exp(rng.uniform(np.log(scale_range[0]), np.log(scale_range[1]))))
    tx = float(rng.uniform(-translation_frac, translation_frac) * width)
    ty = float(rng.uniform(-translation_frac, translation_frac) * height)
    angle = float(rng.uniform(-rotation_deg, rotation_deg))

    sx = 1.0
    sy = 1.0
    squeeze_applied = False
    squeeze_enabled = (not conservative) or x_squeeze_apply_in_conservative
    if x_squeeze_force_scale is not None:
        sx *= float(x_squeeze_force_scale)
        squeeze_applied = True
    elif (
        squeeze_enabled
        and x_squeeze_prob > 0.0
        and x_squeeze_min_scale <= x_squeeze_max_scale
        and rng.random() < x_squeeze_prob
    ):
        sx *= float(rng.uniform(x_squeeze_min_scale, x_squeeze_max_scale))
        squeeze_applied = True

    if stretch_prob > 0 and rng.random() < stretch_prob:
        if not squeeze_applied:
            sx *= float(rng.uniform(1.0 - stretch_frac, 1.0 + stretch_frac))
        sy *= float(rng.uniform(1.0 - stretch_frac, 1.0 + stretch_frac))

    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, angle, scale)
    affine[0, 0] *= sx
    affine[0, 1] *= sx
    affine[1, 0] *= sy
    affine[1, 1] *= sy
    affine[0, 2] += tx
    affine[1, 2] += ty

    perspective = None
    if perspective_prob > 0 and rng.random() < perspective_prob:
        src = np.array(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            dtype=np.float32,
        )
        max_dx = perspective_frac * width
        max_dy = perspective_frac * height
        dst = src + rng.uniform(
            low=[-max_dx, -max_dy],
            high=[max_dx, max_dy],
            size=src.shape,
        ).astype(np.float32)
        perspective = cv2.getPerspectiveTransform(src, dst)

    return GeometricTransform(affine=affine.astype(np.float32), perspective=perspective)


def apply_geometric_transform(
    image: ImageU8,
    transform: GeometricTransform | None,
    *,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int | tuple[int, int, int] = _WHITE_RGB,
    interpolation: int = cv2.INTER_LINEAR,
) -> ImageU8:
    """Apply a sampled transform to an image while preserving output shape."""
    if image.ndim not in (2, 3) or image.dtype != np.uint8:
        return image
    if transform is None:
        return np.ascontiguousarray(image)

    height, width = image.shape[:2]
    warped = cv2.warpAffine(
        image,
        transform.affine,
        (width, height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )
    if transform.perspective is not None:
        warped = cv2.warpPerspective(
            warped,
            transform.perspective,
            (width, height),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
    return np.ascontiguousarray(warped)


def geometric_augment(
    image: ImageU8,
    rng: np.random.Generator,
    *,
    conservative: bool = False,
    x_squeeze_prob: float = 0.45,
    x_squeeze_min_scale: float = 0.70,
    x_squeeze_max_scale: float = 0.95,
    x_squeeze_apply_in_conservative: bool = True,
    x_squeeze_force_scale: float | None = None,
) -> ImageU8:
    """Apply scanner-centric transforms to a white-backed score image."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return image

    transform = sample_geometric_transform(
        image.shape[:2],
        rng,
        conservative=conservative,
        x_squeeze_prob=x_squeeze_prob,
        x_squeeze_min_scale=x_squeeze_min_scale,
        x_squeeze_max_scale=x_squeeze_max_scale,
        x_squeeze_apply_in_conservative=x_squeeze_apply_in_conservative,
        x_squeeze_force_scale=x_squeeze_force_scale,
    )
    scale = np.sqrt(float(np.linalg.det(transform.affine[:, :2]))) if transform is not None else 1.0
    interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    return apply_geometric_transform(
        image,
        transform,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=_WHITE_RGB,
        interpolation=interp,
    )
