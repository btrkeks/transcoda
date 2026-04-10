from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from ..image_generation.image_post import alpha_composite, load_paper_textures, synthesize_background
from ..image_generation.types import RenderedPage
from .augraphy_augment import augraphy_augment
from .geometric_augment import (
    apply_geometric_transform,
    sample_geometric_transform,
)

ImageU8 = npt.NDArray[np.uint8]
_MIN_MARGIN_PX = 12
_MAX_CENTROID_SHIFT_FRACTION = 0.08
_MIN_AREA_RETENTION = 0.70
_VISIBLE_MASK_THRESHOLD = 160

OfflineAugmentTimings = dict[str, float]


@dataclass(frozen=True)
class OfflineAugmentTrace:
    """Structured trace of every decision made inside ``offline_augment``."""

    branch: str  # "geometric" | "none"
    geom_transform_applied: bool
    geom_conservative_retry: bool
    geom_oob_outcome: str  # "pass" | "retry_pass" | "retry_fail"
    augraphy_outcome: str  # "applied" | "noop" | "error" | "invalid_input"
    augraphy_normalize_accepted: bool
    augraphy_fallback_attempted: bool
    augraphy_fallback_outcome: str | None
    augraphy_fallback_normalize_accepted: bool | None
    offline_geom_ms: float
    offline_gates_ms: float
    offline_augraphy_ms: float
    offline_texture_ms: float


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
) -> tuple[ImageU8, ImageU8, OfflineAugmentTrace]:
    """Apply hybrid geometric + background + document artifact augmentation.

    Returns ``(image, pre_augraphy_image, trace)`` where *pre_augraphy_image*
    is the geometric + textured candidate before Augraphy artifacts were applied,
    and *trace* captures every decision made during this call, including timings.
    """
    if not isinstance(image, np.ndarray):
        return image, image, _early_exit_trace()

    base_image = np.ascontiguousarray(image)
    if base_image.dtype != np.uint8 or base_image.ndim != 3 or base_image.shape[2] not in (3, 4):
        return base_image, base_image, _early_exit_trace()
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
    textures = load_paper_textures() if texturize_image else []

    # --- Geometric transform (foreground + alpha only) ---
    geom_start_ns = time.perf_counter_ns()
    warped_fg, warped_alpha, candidate_mask, geom_transform_applied = _build_augmented_candidate(
        base_layers=base_layers,
        rng=rng,
        conservative=False,
        geom_x_squeeze_prob=geom_x_squeeze_prob,
        geom_x_squeeze_min_scale=geom_x_squeeze_min_scale,
        geom_x_squeeze_max_scale=geom_x_squeeze_max_scale,
        geom_x_squeeze_apply_in_conservative=geom_x_squeeze_apply_in_conservative,
        geom_x_squeeze_preview_force_scale=geom_x_squeeze_preview_force_scale,
    )
    offline_geom_ms = _elapsed_ms(geom_start_ns)

    # --- OOB gate check ---
    gates_start_ns = time.perf_counter_ns()
    geom_conservative_retry = False
    geom_oob_outcome = "pass"

    if not _passes_out_of_bounds_gate_from_masks(base_mask, candidate_mask):
        geom_conservative_retry = True
        retry_rng = np.random.default_rng(seed_base + 1)
        retry_fg, retry_alpha, retry_mask, _ = _build_augmented_candidate(
            base_layers=base_layers,
            rng=retry_rng,
            conservative=True,
            geom_x_squeeze_prob=geom_x_squeeze_prob,
            geom_x_squeeze_min_scale=geom_x_squeeze_min_scale,
            geom_x_squeeze_max_scale=geom_x_squeeze_max_scale,
            geom_x_squeeze_apply_in_conservative=geom_x_squeeze_apply_in_conservative,
            geom_x_squeeze_preview_force_scale=geom_x_squeeze_preview_force_scale,
        )
        if _passes_out_of_bounds_gate_from_masks(base_mask, retry_mask):
            warped_fg = retry_fg
            warped_alpha = retry_alpha
            candidate_mask = retry_mask
            geom_oob_outcome = "retry_pass"
        else:
            warped_fg = base_layers.foreground
            warped_alpha = base_layers.alpha
            candidate_mask = base_mask
            geom_oob_outcome = "retry_fail"
    offline_gates_ms = _elapsed_ms(gates_start_ns)

    # --- Background synthesis + compositing (always runs) ---
    texture_start_ns = time.perf_counter_ns()
    height, width = base_image.shape[:2]
    texture_rng = np.random.default_rng(seed_base + 10)
    background = synthesize_background(
        width, height, rng=texture_rng, textures=textures, allow_textures=bool(textures),
    )
    result = np.ascontiguousarray(alpha_composite(background, warped_fg, warped_alpha))
    offline_texture_ms = _elapsed_ms(texture_start_ns)

    pre_augraphy_candidate = np.ascontiguousarray(result)

    # --- Primary Augraphy pass ---
    branch_choice = "geometric"
    artifact_seed = seed_base + 2 if augment_seed is not None else None
    augraphy_start_ns = time.perf_counter_ns()
    artifact_candidate, augraphy_outcome = augraphy_augment(result, seed=artifact_seed)
    offline_augraphy_ms = _elapsed_ms(augraphy_start_ns)

    normalized_artifact = _normalize_aug_output(base_image, artifact_candidate)
    augraphy_normalize_accepted = normalized_artifact is not None

    if augraphy_normalize_accepted:
        trace = OfflineAugmentTrace(
            branch=branch_choice,
            geom_transform_applied=geom_transform_applied,
            geom_conservative_retry=geom_conservative_retry,
            geom_oob_outcome=geom_oob_outcome,
            augraphy_outcome=augraphy_outcome,
            augraphy_normalize_accepted=True,
            augraphy_fallback_attempted=False,
            augraphy_fallback_outcome=None,
            augraphy_fallback_normalize_accepted=None,
            offline_geom_ms=offline_geom_ms,
            offline_gates_ms=offline_gates_ms,
            offline_augraphy_ms=offline_augraphy_ms,
            offline_texture_ms=offline_texture_ms,
        )
        return normalized_artifact, pre_augraphy_candidate, trace

    # --- Fallback Augraphy pass (on base image) ---
    fallback_seed = seed_base + 3 if augment_seed is not None else None
    augraphy_fallback_start_ns = time.perf_counter_ns()
    fallback_candidate, augraphy_fallback_outcome = augraphy_augment(base_image, seed=fallback_seed)
    offline_augraphy_ms += _elapsed_ms(augraphy_fallback_start_ns)

    normalized_fallback = _normalize_aug_output(base_image, fallback_candidate)
    augraphy_fallback_normalize_accepted = normalized_fallback is not None

    if augraphy_fallback_normalize_accepted:
        trace = OfflineAugmentTrace(
            branch=branch_choice,
            geom_transform_applied=geom_transform_applied,
            geom_conservative_retry=geom_conservative_retry,
            geom_oob_outcome=geom_oob_outcome,
            augraphy_outcome=augraphy_outcome,
            augraphy_normalize_accepted=False,
            augraphy_fallback_attempted=True,
            augraphy_fallback_outcome=augraphy_fallback_outcome,
            augraphy_fallback_normalize_accepted=True,
            offline_geom_ms=offline_geom_ms,
            offline_gates_ms=offline_gates_ms,
            offline_augraphy_ms=offline_augraphy_ms,
            offline_texture_ms=offline_texture_ms,
        )
        return normalized_fallback, pre_augraphy_candidate, trace

    # --- Both Augraphy passes failed; return textured candidate ---
    trace = OfflineAugmentTrace(
        branch=branch_choice,
        geom_transform_applied=geom_transform_applied,
        geom_conservative_retry=geom_conservative_retry,
        geom_oob_outcome=geom_oob_outcome,
        augraphy_outcome=augraphy_outcome,
        augraphy_normalize_accepted=False,
        augraphy_fallback_attempted=True,
        augraphy_fallback_outcome=augraphy_fallback_outcome,
        augraphy_fallback_normalize_accepted=False,
        offline_geom_ms=offline_geom_ms,
        offline_gates_ms=offline_gates_ms,
        offline_augraphy_ms=offline_augraphy_ms,
        offline_texture_ms=offline_texture_ms,
    )
    return np.ascontiguousarray(result), pre_augraphy_candidate, trace


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
    base_layers: RenderedPage,
    rng: np.random.Generator,
    conservative: bool,
    geom_x_squeeze_prob: float,
    geom_x_squeeze_min_scale: float,
    geom_x_squeeze_max_scale: float,
    geom_x_squeeze_apply_in_conservative: bool,
    geom_x_squeeze_preview_force_scale: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Warp foreground and alpha layers geometrically.

    Returns ``(warped_foreground, warped_alpha, mask, transform_applied)``.
    Background synthesis is handled separately after the OOB gate.
    """
    height, width = base_layers.foreground.shape[:2]
    transform = sample_geometric_transform(
        (height, width),
        rng,
        conservative=conservative,
        x_squeeze_prob=geom_x_squeeze_prob,
        x_squeeze_min_scale=geom_x_squeeze_min_scale,
        x_squeeze_max_scale=geom_x_squeeze_max_scale,
        x_squeeze_apply_in_conservative=geom_x_squeeze_apply_in_conservative,
        x_squeeze_force_scale=geom_x_squeeze_preview_force_scale,
    )
    transform_applied = transform is not None

    warped_fg = apply_geometric_transform(
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

    visible_mask = _merge_visible_and_hint_masks(
        _visible_notation_mask(warped_fg),
        warped_alpha >= 8,
    )
    return warped_fg, warped_alpha, visible_mask.astype(np.uint8) * 255, transform_applied


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


def _early_exit_trace() -> OfflineAugmentTrace:
    """Trace for early-exit paths (invalid input)."""
    return OfflineAugmentTrace(
        branch="none",
        geom_transform_applied=False,
        geom_conservative_retry=False,
        geom_oob_outcome="pass",
        augraphy_outcome="invalid_input",
        augraphy_normalize_accepted=False,
        augraphy_fallback_attempted=False,
        augraphy_fallback_outcome=None,
        augraphy_fallback_normalize_accepted=None,
        offline_geom_ms=0.0,
        offline_gates_ms=0.0,
        offline_augraphy_ms=0.0,
        offline_texture_ms=0.0,
    )
