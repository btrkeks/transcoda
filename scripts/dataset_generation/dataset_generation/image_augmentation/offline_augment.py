from __future__ import annotations

import hashlib
import time

import cv2
import numpy as np
import numpy.typing as npt

from ..image_generation.image_post import alpha_composite, load_paper_textures, synthesize_background
from ..image_generation.types import RenderedPage
from .augraphy_augment import augraphy_augment
from .geometric_augment import (
    GeometricTransform,
    apply_geometric_transform,
    geometric_augment,
    inverse_transform_points,
    sample_geometric_transform,
)

ImageU8 = npt.NDArray[np.uint8]
_MIN_MARGIN_PX = 12
_MAX_CENTROID_SHIFT_FRACTION = 0.08
_MIN_AREA_RETENTION = 0.70
_REALISTIC_BRANCH_PROB = 0.85
_VISIBLE_MASK_THRESHOLD = 160
_PAD_SAFETY_PX = 8
_BORDER_FILL_BAND_PX = 4

OfflineAugmentTimings = dict[str, float]


def offline_augment(
    image: ImageU8,
    *,
    render_layers: RenderedPage | None = None,
    texturize_image: bool = True,
    filename: str,
    variant_idx: int,
    augment_seed: int | None,
    geom_x_squeeze_prob: float = 0.45,
    geom_x_squeeze_min_scale: float = 0.70,
    geom_x_squeeze_max_scale: float = 0.95,
    geom_x_squeeze_apply_in_conservative: bool = True,
    geom_x_squeeze_preview_force_scale: float | None = None,
    return_timings: bool = False,
) -> ImageU8 | tuple[ImageU8, OfflineAugmentTimings]:
    """Apply hybrid geometric + background + document artifact augmentation."""
    if not isinstance(image, np.ndarray):
        return (image, _empty_timings()) if return_timings else image

    base_image = np.ascontiguousarray(image)
    if base_image.dtype != np.uint8 or base_image.ndim != 3 or base_image.shape[2] not in (3, 4):
        return (base_image, _empty_timings()) if return_timings else base_image
    if base_image.shape[2] == 4:
        base_image = base_image[:, :, :3]

    if not 0.0 <= geom_x_squeeze_prob <= 1.0:
        raise ValueError(f"geom_x_squeeze_prob must be in [0.0, 1.0], got {geom_x_squeeze_prob}")
    if not 0.0 < geom_x_squeeze_min_scale <= geom_x_squeeze_max_scale <= 1.0:
        raise ValueError(
            "geom_x_squeeze_min_scale/geom_x_squeeze_max_scale must satisfy "
            f"0 < min <= max <= 1.0, got min={geom_x_squeeze_min_scale}, "
            f"max={geom_x_squeeze_max_scale}"
        )
    if geom_x_squeeze_preview_force_scale is not None and not (
        0.0 < geom_x_squeeze_preview_force_scale <= 1.0
    ):
        raise ValueError(
            "geom_x_squeeze_preview_force_scale must be in (0.0, 1.0], "
            f"got {geom_x_squeeze_preview_force_scale}"
        )

    base_layers = _normalize_render_layers(base_image, render_layers)
    base_mask = _merge_visible_and_hint_masks(
        _visible_notation_mask(base_image),
        base_layers.alpha >= 8,
    )
    rng = _build_rng(filename=filename, variant_idx=variant_idx, augment_seed=augment_seed)
    seed_base = int(rng.integers(0, 2**31 - 1))
    timings: OfflineAugmentTimings = _empty_timings()
    textures = load_paper_textures() if texturize_image else []

    geom_start_ns = time.perf_counter_ns()
    branch_choice = _sample_branch_choice(rng)
    candidate, candidate_mask = _build_augmented_candidate(
        base_image=base_image,
        base_layers=base_layers,
        base_mask=base_mask,
        textures=textures,
        rng=rng,
        conservative=False,
        branch_choice=branch_choice,
        geom_x_squeeze_prob=geom_x_squeeze_prob,
        geom_x_squeeze_min_scale=geom_x_squeeze_min_scale,
        geom_x_squeeze_max_scale=geom_x_squeeze_max_scale,
        geom_x_squeeze_apply_in_conservative=geom_x_squeeze_apply_in_conservative,
        geom_x_squeeze_preview_force_scale=geom_x_squeeze_preview_force_scale,
    )
    timings["offline_geom_ms"] = _elapsed_ms(geom_start_ns)

    gates_start_ns = time.perf_counter_ns()
    if not _passes_out_of_bounds_gate_from_masks(base_mask, candidate_mask):
        retry_rng = np.random.default_rng(seed_base + 1)
        retry_branch = _sample_branch_choice(retry_rng)
        geom_retry, retry_mask = _build_augmented_candidate(
            base_image=base_image,
            base_layers=base_layers,
            base_mask=base_mask,
            textures=textures,
            rng=retry_rng,
            conservative=True,
            branch_choice=retry_branch,
            geom_x_squeeze_prob=geom_x_squeeze_prob,
            geom_x_squeeze_min_scale=geom_x_squeeze_min_scale,
            geom_x_squeeze_max_scale=geom_x_squeeze_max_scale,
            geom_x_squeeze_apply_in_conservative=geom_x_squeeze_apply_in_conservative,
            geom_x_squeeze_preview_force_scale=geom_x_squeeze_preview_force_scale,
        )
        if _passes_out_of_bounds_gate_from_masks(base_mask, retry_mask):
            candidate = geom_retry
            candidate_mask = retry_mask
        else:
            candidate = base_image
            candidate_mask = base_mask
    timings["offline_gates_ms"] = _elapsed_ms(gates_start_ns)

    texture_start_ns = time.perf_counter_ns()
    result = np.ascontiguousarray(candidate)
    timings["offline_texture_ms"] += _elapsed_ms(texture_start_ns)

    artifact_seed = seed_base + 2 if augment_seed is not None else None
    augraphy_start_ns = time.perf_counter_ns()
    artifact_candidate = augraphy_augment(result, seed=artifact_seed)
    timings["offline_augraphy_ms"] += _elapsed_ms(augraphy_start_ns)
    normalized_artifact = _normalize_aug_output(base_image, artifact_candidate)
    if normalized_artifact is not None:
        return (normalized_artifact, timings) if return_timings else normalized_artifact

    fallback_seed = seed_base + 3 if augment_seed is not None else None
    augraphy_fallback_start_ns = time.perf_counter_ns()
    fallback_candidate = augraphy_augment(base_image, seed=fallback_seed)
    timings["offline_augraphy_ms"] += _elapsed_ms(augraphy_fallback_start_ns)
    normalized_fallback = _normalize_aug_output(base_image, fallback_candidate)
    if normalized_fallback is not None:
        return (normalized_fallback, timings) if return_timings else normalized_fallback

    result = np.ascontiguousarray(base_image)
    return (result, timings) if return_timings else result


def passes_quality_gate(image: ImageU8, *, min_margin_px: int = _MIN_MARGIN_PX) -> bool:
    """Balanced quality gate to reject obvious corruption failures."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        return False

    rgb = image[:, :, :3]
    gray = rgb.mean(axis=2)
    mean_luma = float(gray.mean())
    if mean_luma < 120.0 or mean_luma > 252.0:
        return False

    black_mask = gray <= 120.0
    black_ratio = float(black_mask.mean())
    if black_ratio < 0.005 or black_ratio > 0.35:
        return False

    bbox = _content_bbox_from_mask(black_mask)
    if bbox is None:
        return False
    if not _has_minimum_margins(bbox, black_mask.shape, min_margin_px=min_margin_px):
        return False
    if _touches_too_many_borders(black_mask):
        return False
    return True


def passes_transform_consistency(
    base_image: ImageU8,
    candidate: ImageU8,
    *,
    max_centroid_shift_fraction: float = _MAX_CENTROID_SHIFT_FRACTION,
    min_area_retention: float = _MIN_AREA_RETENTION,
) -> bool:
    """Ensure augmentation does not move/crop notation unrealistically."""
    if (
        base_image.dtype != np.uint8
        or candidate.dtype != np.uint8
        or base_image.ndim != 3
        or candidate.ndim != 3
        or base_image.shape[2] not in (3, 4)
        or candidate.shape[2] not in (3, 4)
    ):
        return False
    if base_image.shape[:2] != candidate.shape[:2]:
        return False

    base_mask = _visible_notation_mask(base_image)
    candidate_mask = _visible_notation_mask(candidate)
    return _passes_out_of_bounds_gate_from_masks(
        base_mask.astype(np.uint8) * 255,
        candidate_mask.astype(np.uint8) * 255,
        max_centroid_shift_fraction=max_centroid_shift_fraction,
        min_area_retention=min_area_retention,
    )


def _normalize_render_layers(base_image: ImageU8, render_layers: RenderedPage | None) -> RenderedPage:
    if render_layers is not None:
        foreground = np.ascontiguousarray(render_layers.foreground[:, :, :3])
        alpha = np.ascontiguousarray(render_layers.alpha.astype(np.uint8))
        if foreground.shape[:2] == base_image.shape[:2] and alpha.shape[:2] == base_image.shape[:2]:
            return RenderedPage(image=base_image, foreground=foreground, alpha=alpha)

    alpha = np.ascontiguousarray((255 - np.min(base_image[:, :, :3], axis=2)).astype(np.uint8))
    foreground = np.full_like(base_image[:, :, :3], 255)
    mask = alpha > 0
    foreground[mask] = base_image[:, :, :3][mask]
    return RenderedPage(image=base_image, foreground=foreground, alpha=alpha)


def _build_augmented_candidate(
    *,
    base_image: ImageU8,
    base_layers: RenderedPage,
    base_mask: np.ndarray,
    textures: list[str],
    rng: np.random.Generator,
    conservative: bool,
    branch_choice: str,
    geom_x_squeeze_prob: float,
    geom_x_squeeze_min_scale: float,
    geom_x_squeeze_max_scale: float,
    geom_x_squeeze_apply_in_conservative: bool,
    geom_x_squeeze_preview_force_scale: float | None,
) -> tuple[ImageU8, np.ndarray]:
    transform = sample_geometric_transform(
        base_image.shape[:2],
        rng,
        conservative=conservative,
        x_squeeze_prob=geom_x_squeeze_prob,
        x_squeeze_min_scale=geom_x_squeeze_min_scale,
        x_squeeze_max_scale=geom_x_squeeze_max_scale,
        x_squeeze_apply_in_conservative=geom_x_squeeze_apply_in_conservative,
        x_squeeze_force_scale=geom_x_squeeze_preview_force_scale,
    )
    if branch_choice == "foreground":
        return _foreground_branch(base_image, base_layers, transform, textures, rng)
    return _realistic_branch(base_image, base_layers, transform, textures, rng)


def _realistic_branch(
    base_image: ImageU8,
    base_layers: RenderedPage,
    transform: GeometricTransform | None,
    textures: list[str],
    rng: np.random.Generator,
) -> tuple[ImageU8, np.ndarray]:
    height, width = base_image.shape[:2]
    pad_y, pad_x = required_padding_for_safe_crop(transform, (height, width))
    canvas_h = height + (2 * pad_y)
    canvas_w = width + (2 * pad_x)

    background = synthesize_background(
        canvas_w,
        canvas_h,
        rng=rng,
        textures=textures,
        allow_textures=True,
    )
    foreground_canvas = _embed_in_canvas(base_layers.foreground, (canvas_h, canvas_w, 3), (pad_y, pad_x), 255)
    alpha_canvas = _embed_in_canvas(base_layers.alpha, (canvas_h, canvas_w), (pad_y, pad_x), 0)
    page_canvas = alpha_composite(background, foreground_canvas, alpha_canvas)

    adjusted_transform = _translate_transform(transform, offset_x=pad_x, offset_y=pad_y)
    border_value = _background_border_value(background)
    warped_page = apply_geometric_transform(
        page_canvas,
        adjusted_transform,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=border_value,
        interpolation=cv2.INTER_LINEAR,
    )
    warped_alpha = apply_geometric_transform(
        alpha_canvas,
        adjusted_transform,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
        interpolation=cv2.INTER_LINEAR,
    )
    cropped_page = _center_crop(warped_page, (height, width), (pad_y, pad_x))
    cropped_alpha = _center_crop(warped_alpha, (height, width), (pad_y, pad_x))
    visible_mask = _merge_visible_and_hint_masks(
        _visible_notation_mask(cropped_page),
        cropped_alpha >= 8,
    )
    if _detect_border_fill(cropped_page, border_value, band_px=_BORDER_FILL_BAND_PX):
        return cropped_page, np.zeros((height, width), dtype=np.uint8)
    return cropped_page, visible_mask.astype(np.uint8) * 255


def _foreground_branch(
    base_image: ImageU8,
    base_layers: RenderedPage,
    transform: GeometricTransform | None,
    textures: list[str],
    rng: np.random.Generator,
) -> tuple[ImageU8, np.ndarray]:
    height, width = base_image.shape[:2]
    warped_foreground = apply_geometric_transform(
        base_layers.foreground,
        transform,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=255,
        interpolation=cv2.INTER_LINEAR,
    )
    warped_alpha = apply_geometric_transform(
        base_layers.alpha,
        transform,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
        interpolation=cv2.INTER_LINEAR,
    )
    background = synthesize_background(width, height, rng=rng, textures=textures, allow_textures=True)
    candidate = alpha_composite(background, warped_foreground, warped_alpha)
    visible_mask = _merge_visible_and_hint_masks(
        _visible_notation_mask(candidate),
        warped_alpha >= 8,
    )
    return candidate, visible_mask.astype(np.uint8) * 255


def _sample_branch_choice(rng: np.random.Generator) -> str:
    return "realistic" if float(rng.random()) < _REALISTIC_BRANCH_PROB else "foreground"


def required_padding_for_safe_crop(
    transform: GeometricTransform | None,
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    if transform is None:
        return _PAD_SAFETY_PX, _PAD_SAFETY_PX

    height, width = image_shape
    crop_boundary = _sample_crop_boundary_points((height, width))
    inverse_points = inverse_transform_points(crop_boundary, transform)
    min_x = float(inverse_points[:, 0].min())
    max_x = float(inverse_points[:, 0].max())
    min_y = float(inverse_points[:, 1].min())
    max_y = float(inverse_points[:, 1].max())
    pad_x = int(np.ceil(max(-min_x, max_x - (width - 1.0), 0.0))) + _PAD_SAFETY_PX
    pad_y = int(np.ceil(max(-min_y, max_y - (height - 1.0), 0.0))) + _PAD_SAFETY_PX
    return max(pad_y, _PAD_SAFETY_PX), max(pad_x, _PAD_SAFETY_PX)


def _translate_transform(
    transform: GeometricTransform | None,
    *,
    offset_x: int,
    offset_y: int,
) -> GeometricTransform | None:
    if transform is None:
        return None

    translate = np.array(
        [[1.0, 0.0, float(offset_x)], [0.0, 1.0, float(offset_y)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    untranslate = np.array(
        [[1.0, 0.0, -float(offset_x)], [0.0, 1.0, -float(offset_y)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    affine3 = np.vstack([transform.affine, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    adjusted_affine = translate @ affine3 @ untranslate
    adjusted_perspective = None
    if transform.perspective is not None:
        adjusted_perspective = translate @ transform.perspective @ untranslate
    return GeometricTransform(
        affine=adjusted_affine[:2, :].astype(np.float32),
        perspective=None if adjusted_perspective is None else adjusted_perspective.astype(np.float32),
    )


def _embed_in_canvas(
    image: np.ndarray,
    canvas_shape: tuple[int, ...],
    offset: tuple[int, int],
    fill_value: int,
) -> np.ndarray:
    canvas = np.full(canvas_shape, fill_value, dtype=np.uint8)
    top, left = offset
    bottom = top + image.shape[0]
    right = left + image.shape[1]
    canvas[top:bottom, left:right] = image
    return np.ascontiguousarray(canvas)


def _center_crop(
    image: np.ndarray,
    target_shape: tuple[int, int],
    offset: tuple[int, int],
) -> np.ndarray:
    top, left = offset
    height, width = target_shape
    cropped = image[top : top + height, left : left + width]
    return np.ascontiguousarray(cropped)


def _background_border_value(background: np.ndarray) -> tuple[int, int, int]:
    sample = background.reshape(-1, background.shape[-1]).mean(axis=0)
    return tuple(int(np.clip(round(v), 0, 255)) for v in sample[:3])


def _sample_crop_boundary_points(image_shape: tuple[int, int], *, steps_per_edge: int = 5) -> np.ndarray:
    height, width = image_shape
    xs = np.linspace(0.0, max(width - 1.0, 0.0), num=max(steps_per_edge, 2), dtype=np.float32)
    ys = np.linspace(0.0, max(height - 1.0, 0.0), num=max(steps_per_edge, 2), dtype=np.float32)
    points: list[tuple[float, float]] = []
    for x in xs:
        points.append((float(x), 0.0))
        points.append((float(x), float(max(height - 1.0, 0.0))))
    for y in ys[1:-1]:
        points.append((0.0, float(y)))
        points.append((float(max(width - 1.0, 0.0)), float(y)))
    return np.array(points, dtype=np.float32)


def _detect_border_fill(
    image: np.ndarray,
    border_value: tuple[int, int, int],
    *,
    band_px: int,
) -> bool:
    if band_px <= 0 or image.size == 0:
        return False
    height, width = image.shape[:2]
    band = min(band_px, height, width)
    edge_mask = np.zeros((height, width), dtype=bool)
    edge_mask[:band, :] = True
    edge_mask[-band:, :] = True
    edge_mask[:, :band] = True
    edge_mask[:, -band:] = True
    border_arr = np.array(border_value, dtype=np.uint8)
    matches = np.all(image[:, :, :3] == border_arr[None, None, :], axis=2) & edge_mask
    match_count = int(matches.sum())
    threshold = max(8, int(edge_mask.sum() * 0.02))
    return match_count >= threshold


def _visible_notation_mask(image: ImageU8, *, threshold: int = _VISIBLE_MASK_THRESHOLD) -> np.ndarray:
    rgb = image[:, :, :3]
    mask = np.min(rgb, axis=2) <= threshold
    return _remove_small_components(mask)


def _remove_small_components(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    min_area = max(4, int(round(mask.size * 0.00002)))
    filtered = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == label] = True
    return filtered


def _merge_visible_and_hint_masks(visible_mask: np.ndarray, hint_mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.logical_or(visible_mask, hint_mask))


def _touches_too_many_borders(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return True
    top = int(coords[:, 0].min()) == 0
    bottom = int(coords[:, 0].max()) == (mask.shape[0] - 1)
    left = int(coords[:, 1].min()) == 0
    right = int(coords[:, 1].max()) == (mask.shape[1] - 1)
    return sum((top, bottom, left, right)) >= 3


def _content_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    top = int(coords[:, 0].min())
    bottom = int(coords[:, 0].max())
    left = int(coords[:, 1].min())
    right = int(coords[:, 1].max())
    return top, bottom, left, right


def _has_minimum_margins(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    *,
    min_margin_px: int,
) -> bool:
    top, bottom, left, right = bbox
    height, width = image_shape
    return all(
        margin >= min_margin_px
        for margin in (top, height - 1 - bottom, left, width - 1 - right)
    )


def _bbox_center_and_area(bbox: tuple[int, int, int, int]) -> tuple[float, float, float]:
    top, bottom, left, right = bbox
    return (
        (left + right) / 2.0,
        (top + bottom) / 2.0,
        float((right - left + 1) * (bottom - top + 1)),
    )


def _passes_pre_augraphy_gate(base_image: ImageU8, candidate: ImageU8) -> bool:
    return passes_quality_gate(candidate) and passes_transform_consistency(base_image, candidate)


def _passes_out_of_bounds_gate(
    base_image: ImageU8,
    candidate: ImageU8,
    *,
    min_margin_px: int = _MIN_MARGIN_PX,
    max_centroid_shift_fraction: float = _MAX_CENTROID_SHIFT_FRACTION,
    min_area_retention: float = _MIN_AREA_RETENTION,
) -> bool:
    base_mask = _visible_notation_mask(base_image).astype(np.uint8) * 255
    candidate_mask = _visible_notation_mask(candidate).astype(np.uint8) * 255
    return _passes_out_of_bounds_gate_from_masks(
        base_mask,
        candidate_mask,
        min_margin_px=min_margin_px,
        max_centroid_shift_fraction=max_centroid_shift_fraction,
        min_area_retention=min_area_retention,
    )


def _passes_out_of_bounds_gate_from_masks(
    base_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    min_margin_px: int = _MIN_MARGIN_PX,
    max_centroid_shift_fraction: float = _MAX_CENTROID_SHIFT_FRACTION,
    min_area_retention: float = _MIN_AREA_RETENTION,
) -> bool:
    if base_mask.shape != candidate_mask.shape:
        return False
    base_bbox = _content_bbox_from_mask(base_mask > 0)
    cand_bbox = _content_bbox_from_mask(candidate_mask > 0)
    if base_bbox is None or cand_bbox is None:
        return False
    if not _has_minimum_margins(cand_bbox, candidate_mask.shape, min_margin_px=min_margin_px):
        return False
    if _touches_too_many_borders(candidate_mask > 0):
        return False

    base_cx, base_cy, base_area = _bbox_center_and_area(base_bbox)
    cand_cx, cand_cy, cand_area = _bbox_center_and_area(cand_bbox)
    if base_area <= 0:
        return False
    height, width = base_mask.shape[:2]
    dx_frac = abs(cand_cx - base_cx) / max(width, 1)
    dy_frac = abs(cand_cy - base_cy) / max(height, 1)
    if dx_frac > max_centroid_shift_fraction or dy_frac > max_centroid_shift_fraction:
        return False
    if (cand_area / base_area) < min_area_retention:
        return False
    return True


def _content_mask_fast(image: ImageU8, *, threshold: int = 120) -> np.ndarray:
    rgb = image[:, :, :3]
    return np.min(rgb, axis=2) <= threshold


def _normalize_aug_output(base_image: ImageU8, candidate: ImageU8) -> ImageU8 | None:
    if not isinstance(candidate, np.ndarray):
        return None
    if candidate.dtype != np.uint8:
        return None
    if candidate.ndim == 2:
        candidate = np.stack([candidate, candidate, candidate], axis=-1)
    elif candidate.ndim == 3 and candidate.shape[2] == 1:
        candidate = np.repeat(candidate, 3, axis=2)
    elif candidate.ndim != 3 or candidate.shape[2] not in (3, 4):
        return None
    if candidate.shape[:2] != base_image.shape[:2]:
        return None
    if candidate.shape[2] == 4:
        candidate = candidate[:, :, :3]
    return np.ascontiguousarray(candidate)


def _build_rng(*, filename: str, variant_idx: int, augment_seed: int | None) -> np.random.Generator:
    if augment_seed is None:
        return np.random.default_rng()
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(str(augment_seed).encode("utf-8"))
    hasher.update(filename.encode("utf-8"))
    hasher.update(str(variant_idx).encode("utf-8"))
    seed = int.from_bytes(hasher.digest(), byteorder="little", signed=False)
    return np.random.default_rng(seed)


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _empty_timings() -> OfflineAugmentTimings:
    return {
        "offline_geom_ms": 0.0,
        "offline_gates_ms": 0.0,
        "offline_augraphy_ms": 0.0,
        "offline_texture_ms": 0.0,
    }
