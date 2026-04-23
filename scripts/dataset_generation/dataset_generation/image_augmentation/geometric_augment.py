from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

ImageU8 = npt.NDArray[np.uint8]

_WHITE_RGB = (255, 255, 255)
_DEFAULT_MIN_MARGIN_PX = 2
_DEFAULT_MAX_CENTROID_SHIFT_FRACTION = 0.08


@dataclass(frozen=True)
class GeometricTransform:
    """Sampled geometric transform shared across page and foreground branches."""

    affine: np.ndarray
    perspective: np.ndarray | None = None
    angle_deg: float = 0.0
    scale: float = 1.0
    tx_px: float = 0.0
    ty_px: float = 0.0
    x_scale: float = 1.0
    y_scale: float = 1.0
    conservative: bool = False


def _content_bbox_from_mask(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    if mask is None:
        return None
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    top = int(coords[:, 0].min())
    bottom = int(coords[:, 0].max())
    left = int(coords[:, 1].min())
    right = int(coords[:, 1].max())
    return top, bottom, left, right


def _content_mask_from_image(image: ImageU8, *, threshold: int = 250) -> np.ndarray:
    if image.ndim == 2:
        return image < threshold
    return np.min(image[:, :, :3], axis=2) < threshold


def _translation_interval_from_bbox(
    bbox: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int],
    *,
    min_margin_px: int,
) -> tuple[float, float, float, float]:
    if bbox is None:
        return 0.0, 0.0, 0.0, 0.0

    top, bottom, left, right = bbox
    height, width = image_shape
    left_room = float(max(0, left - min_margin_px))
    right_room = float(max(0, (width - 1 - right) - min_margin_px))
    top_room = float(max(0, top - min_margin_px))
    bottom_room = float(max(0, (height - 1 - bottom) - min_margin_px))
    return -left_room, right_room, -top_room, bottom_room


def _bbox_corners(bbox: tuple[int, int, int, int] | None) -> np.ndarray | None:
    if bbox is None:
        return None
    top, bottom, left, right = bbox
    return np.array(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=np.float32,
    )


def _bbox_from_points(points: np.ndarray | None) -> tuple[float, float, float, float] | None:
    if points is None or points.size == 0:
        return None
    return (
        float(points[:, 1].min()),
        float(points[:, 1].max()),
        float(points[:, 0].min()),
        float(points[:, 0].max()),
    )


def _interval_intersection(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float] | None:
    lo = max(left[0], right[0])
    hi = min(left[1], right[1])
    if lo > hi:
        return None
    return float(lo), float(hi)


def _translation_interval_from_transformed_bbox(
    base_bbox: tuple[int, int, int, int] | None,
    transformed_bbox: tuple[float, float, float, float] | None,
    image_shape: tuple[int, int],
    *,
    min_margin_px: int,
    max_centroid_shift_fraction: float,
) -> tuple[float, float, float, float] | None:
    if base_bbox is None or transformed_bbox is None:
        return 0.0, 0.0, 0.0, 0.0

    trans_top, trans_bottom, trans_left, trans_right = transformed_bbox
    height, width = image_shape

    tx_margin = (
        float(min_margin_px) - trans_left,
        float(width - 1 - min_margin_px) - trans_right,
    )
    ty_margin = (
        float(min_margin_px) - trans_top,
        float(height - 1 - min_margin_px) - trans_bottom,
    )

    base_top, base_bottom, base_left, base_right = base_bbox
    base_cx = (base_left + base_right) / 2.0
    base_cy = (base_top + base_bottom) / 2.0
    trans_cx = (trans_left + trans_right) / 2.0
    trans_cy = (trans_top + trans_bottom) / 2.0
    max_dx = float(max_centroid_shift_fraction) * float(width)
    max_dy = float(max_centroid_shift_fraction) * float(height)
    tx_centroid = (base_cx - max_dx - trans_cx, base_cx + max_dx - trans_cx)
    ty_centroid = (base_cy - max_dy - trans_cy, base_cy + max_dy - trans_cy)

    tx_interval = _interval_intersection(tx_margin, tx_centroid)
    ty_interval = _interval_intersection(ty_margin, ty_centroid)
    if tx_interval is None or ty_interval is None:
        return None
    return tx_interval[0], tx_interval[1], ty_interval[0], ty_interval[1]


def _translation_matrix(tx_px: float, ty_px: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, tx_px],
            [0.0, 1.0, ty_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _compose_transform_matrix(
    affine: np.ndarray,
    perspective: np.ndarray | None,
    *,
    tx_px: float,
    ty_px: float,
) -> np.ndarray:
    affine3 = np.vstack([affine, np.array([0.0, 0.0, 1.0], dtype=np.float32)]).astype(np.float32)
    matrix = affine3
    if perspective is not None:
        matrix = perspective.astype(np.float32) @ matrix
    return _translation_matrix(tx_px, ty_px) @ matrix


def transform_matrix(transform: GeometricTransform | None) -> np.ndarray | None:
    """Return the composed 3x3 forward transform matrix."""
    if transform is None:
        return None
    return _compose_transform_matrix(
        transform.affine,
        transform.perspective,
        tx_px=float(transform.tx_px),
        ty_px=float(transform.ty_px),
    )


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
    content_mask: np.ndarray | None = None,
    min_margin_px: int = _DEFAULT_MIN_MARGIN_PX,
    max_centroid_shift_fraction: float = _DEFAULT_MAX_CENTROID_SHIFT_FRACTION,
) -> GeometricTransform | None:
    """Sample a scanner-like transform while preserving the output size."""
    height, width = image_shape
    if width < 2 or height < 2:
        return None

    if conservative:
        scale_range = (0.95, 1.08)
        rotation_deg = 1.2
        perspective_prob = 0.0
        perspective_frac = 0.0
        stretch_prob = 0.0
        stretch_frac = 0.0
    else:
        scale_range = (0.90, 1.15)
        rotation_deg = 1.8
        perspective_prob = 0.30
        perspective_frac = 0.015
        stretch_prob = 0.20
        stretch_frac = 0.03

    if x_squeeze_force_scale is not None:
        scale_range = (1.0, 1.0)
        rotation_deg = 0.0
        perspective_prob = 0.0
        perspective_frac = 0.0
        stretch_prob = 0.0
        stretch_frac = 0.0

    if x_squeeze_force_scale is None and rng.random() >= 0.85:
        return None

    base_bbox = _content_bbox_from_mask(content_mask)
    scale = float(np.exp(rng.uniform(np.log(scale_range[0]), np.log(scale_range[1]))))
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

    tx = 0.0
    ty = 0.0
    bbox_corners = _bbox_corners(base_bbox)
    transformed_bbox = _bbox_from_points(
        transform_points(
            bbox_corners,
            GeometricTransform(
                affine=affine.astype(np.float32),
                perspective=perspective,
                tx_px=0.0,
                ty_px=0.0,
                angle_deg=angle,
                scale=scale,
                x_scale=sx,
                y_scale=sy,
                conservative=conservative,
            ),
        )
        if bbox_corners is not None
        else None
    )
    translation_interval = _translation_interval_from_transformed_bbox(
        base_bbox,
        transformed_bbox,
        image_shape,
        min_margin_px=min_margin_px,
        max_centroid_shift_fraction=max_centroid_shift_fraction,
    )
    if translation_interval is None:
        return None
    tx_min, tx_max, ty_min, ty_max = translation_interval
    if x_squeeze_force_scale is None:
        tx = float(rng.uniform(tx_min, tx_max))
        ty = float(rng.uniform(ty_min, ty_max))
    elif not (tx_min <= 0.0 <= tx_max and ty_min <= 0.0 <= ty_max):
        return None

    return GeometricTransform(
        affine=affine.astype(np.float32),
        perspective=perspective,
        angle_deg=angle,
        scale=scale,
        tx_px=tx,
        ty_px=ty,
        x_scale=sx,
        y_scale=sy,
        conservative=conservative,
    )


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
    if transform.perspective is None:
        affine = transform.affine.astype(np.float32).copy()
        affine[0, 2] += float(transform.tx_px)
        affine[1, 2] += float(transform.ty_px)
        warped = cv2.warpAffine(
            image,
            affine,
            (width, height),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        return np.ascontiguousarray(warped)

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
    translation_affine = np.array(
        [
            [1.0, 0.0, float(transform.tx_px)],
            [0.0, 1.0, float(transform.ty_px)],
        ],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(
        warped,
        translation_affine,
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
    min_margin_px: int = _DEFAULT_MIN_MARGIN_PX,
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
        content_mask=_content_mask_from_image(image),
        min_margin_px=min_margin_px,
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
