from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from ..image_generation.image_post import (
    alpha_composite,
    load_paper_textures,
    synthesize_background,
)
from ..image_generation.types import RenderedPage
from ..types_events import (
    BoundsGateTrace,
    GeometryTrace,
    MarginTrace,
    OfflineAugmentTrace,
    OuterGateTrace,
    QualityGateTrace,
)
from .augraphy_augment import augraphy_augment
from .geometric_augment import (
    GeometricTransform,
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
class AugmentedCandidate:
    foreground: np.ndarray
    alpha: np.ndarray
    mask: np.ndarray
    geometry: GeometryTrace


def _geometry_trace_from_transform(
    transform: GeometricTransform | None,
    *,
    conservative: bool,
) -> GeometryTrace:
    if transform is None:
        return GeometryTrace(
            sampled=False,
            conservative=conservative,
            angle_deg=None,
            scale=None,
            tx_px=None,
            ty_px=None,
            x_scale=None,
            y_scale=None,
            perspective_applied=False,
        )
    return GeometryTrace(
        sampled=True,
        conservative=conservative,
        angle_deg=float(transform.angle_deg),
        scale=float(transform.scale),
        tx_px=float(transform.tx_px),
        ty_px=float(transform.ty_px),
        x_scale=float(transform.x_scale),
        y_scale=float(transform.y_scale),
        perspective_applied=transform.perspective is not None,
    )


def _coerce_augmented_candidate(
    candidate: AugmentedCandidate | tuple[np.ndarray, np.ndarray, np.ndarray, bool],
    *,
    conservative: bool,
) -> AugmentedCandidate:
    if isinstance(candidate, AugmentedCandidate):
        return candidate
    foreground, alpha, mask, transform_applied = candidate
    return AugmentedCandidate(
        foreground=foreground,
        alpha=alpha,
        mask=mask,
        geometry=(
            GeometryTrace(
                sampled=True,
                conservative=conservative,
                angle_deg=None,
                scale=None,
                tx_px=None,
                ty_px=None,
                x_scale=None,
                y_scale=None,
                perspective_applied=False,
            )
            if transform_applied
            else _geometry_trace_from_transform(None, conservative=conservative)
        ),
    )


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
    initial_candidate = _coerce_augmented_candidate(
        _build_augmented_candidate(
        base_layers=base_layers,
        rng=rng,
        conservative=False,
        geom_x_squeeze_prob=geom_x_squeeze_prob,
        geom_x_squeeze_min_scale=geom_x_squeeze_min_scale,
        geom_x_squeeze_max_scale=geom_x_squeeze_max_scale,
        geom_x_squeeze_apply_in_conservative=geom_x_squeeze_apply_in_conservative,
        geom_x_squeeze_preview_force_scale=geom_x_squeeze_preview_force_scale,
        ),
        conservative=False,
    )
    offline_geom_ms = _elapsed_ms(geom_start_ns)

    # --- OOB gate check ---
    gates_start_ns = time.perf_counter_ns()
    initial_oob_gate = evaluate_oob_gate_from_masks(base_mask, initial_candidate.mask)
    retry_candidate: AugmentedCandidate | None = None
    retry_oob_gate: BoundsGateTrace | None = None
    selected_candidate = initial_candidate
    selected_geometry = initial_candidate.geometry

    if not initial_oob_gate.passed:
        retry_rng = np.random.default_rng(seed_base + 1)
        retry_candidate = _coerce_augmented_candidate(
            _build_augmented_candidate(
            base_layers=base_layers,
            rng=retry_rng,
            conservative=True,
            geom_x_squeeze_prob=geom_x_squeeze_prob,
            geom_x_squeeze_min_scale=geom_x_squeeze_min_scale,
            geom_x_squeeze_max_scale=geom_x_squeeze_max_scale,
            geom_x_squeeze_apply_in_conservative=geom_x_squeeze_apply_in_conservative,
            geom_x_squeeze_preview_force_scale=geom_x_squeeze_preview_force_scale,
            ),
            conservative=True,
        )
        retry_oob_gate = evaluate_oob_gate_from_masks(base_mask, retry_candidate.mask)
        if retry_oob_gate.passed:
            selected_candidate = retry_candidate
            selected_geometry = retry_candidate.geometry
        else:
            selected_candidate = AugmentedCandidate(
                foreground=base_layers.foreground,
                alpha=base_layers.alpha,
                mask=base_mask.astype(np.uint8),
                geometry=_geometry_trace_from_transform(None, conservative=False),
            )
            selected_geometry = selected_candidate.geometry
    offline_gates_ms = _elapsed_ms(gates_start_ns)

    # --- Background synthesis + compositing (always runs) ---
    texture_start_ns = time.perf_counter_ns()
    height, width = base_image.shape[:2]
    texture_rng = np.random.default_rng(seed_base + 10)
    background = synthesize_background(
        width, height, rng=texture_rng, textures=textures, allow_textures=bool(textures),
    )
    result = np.ascontiguousarray(
        alpha_composite(background, selected_candidate.foreground, selected_candidate.alpha)
    )
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
            initial_geometry=initial_candidate.geometry,
            retry_geometry=retry_candidate.geometry if retry_candidate is not None else None,
            selected_geometry=selected_geometry,
            final_geometry_applied=selected_geometry.sampled,
            initial_oob_gate=initial_oob_gate,
            retry_oob_gate=retry_oob_gate,
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
            initial_geometry=initial_candidate.geometry,
            retry_geometry=retry_candidate.geometry if retry_candidate is not None else None,
            selected_geometry=selected_geometry,
            final_geometry_applied=selected_geometry.sampled,
            initial_oob_gate=initial_oob_gate,
            retry_oob_gate=retry_oob_gate,
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
        initial_geometry=initial_candidate.geometry,
        retry_geometry=retry_candidate.geometry if retry_candidate is not None else None,
        selected_geometry=selected_geometry,
        final_geometry_applied=selected_geometry.sampled,
        initial_oob_gate=initial_oob_gate,
        retry_oob_gate=retry_oob_gate,
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
    return evaluate_quality_gate(image, min_margin_px=min_margin_px).passed


def passes_transform_consistency(
    base_image: ImageU8,
    candidate: ImageU8,
    *,
    max_centroid_shift_fraction: float = _MAX_CENTROID_SHIFT_FRACTION,
    min_area_retention: float = _MIN_AREA_RETENTION,
) -> bool:
    """Ensure augmentation does not move/crop notation unrealistically."""
    return evaluate_transform_consistency(
        base_image,
        candidate,
        max_centroid_shift_fraction=max_centroid_shift_fraction,
        min_area_retention=min_area_retention,
    ).passed


def evaluate_quality_gate(
    image: ImageU8,
    *,
    min_margin_px: int = _MIN_MARGIN_PX,
) -> QualityGateTrace:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        return QualityGateTrace(
            passed=False,
            failure_reason="invalid_input",
            mean_luma=None,
            black_ratio=None,
            margins_px=None,
            border_touch_count=None,
        )

    rgb = image[:, :, :3]
    gray = rgb.mean(axis=2)
    mean_luma = float(gray.mean())
    black_mask = gray <= 120.0
    black_ratio = float(black_mask.mean())
    bbox = _content_bbox_from_mask(black_mask)
    margins = _margins_from_bbox(bbox, black_mask.shape) if bbox is not None else None
    border_touch_count = _border_touch_count(black_mask) if bbox is not None else None

    if mean_luma < 120.0 or mean_luma > 252.0:
        return QualityGateTrace(
            passed=False,
            failure_reason="mean_luma",
            mean_luma=mean_luma,
            black_ratio=black_ratio,
            margins_px=margins,
            border_touch_count=border_touch_count,
        )
    if black_ratio < 0.005 or black_ratio > 0.35:
        return QualityGateTrace(
            passed=False,
            failure_reason="black_ratio",
            mean_luma=mean_luma,
            black_ratio=black_ratio,
            margins_px=margins,
            border_touch_count=border_touch_count,
        )
    if bbox is None:
        return QualityGateTrace(
            passed=False,
            failure_reason="empty_content",
            mean_luma=mean_luma,
            black_ratio=black_ratio,
            margins_px=None,
            border_touch_count=None,
        )
    if not _has_minimum_margins(bbox, black_mask.shape, min_margin_px=min_margin_px):
        return QualityGateTrace(
            passed=False,
            failure_reason="min_margin",
            mean_luma=mean_luma,
            black_ratio=black_ratio,
            margins_px=margins,
            border_touch_count=border_touch_count,
        )
    if border_touch_count is not None and border_touch_count >= 3:
        return QualityGateTrace(
            passed=False,
            failure_reason="border_touches",
            mean_luma=mean_luma,
            black_ratio=black_ratio,
            margins_px=margins,
            border_touch_count=border_touch_count,
        )
    return QualityGateTrace(
        passed=True,
        failure_reason=None,
        mean_luma=mean_luma,
        black_ratio=black_ratio,
        margins_px=margins,
        border_touch_count=border_touch_count,
    )


def evaluate_transform_consistency(
    base_image: ImageU8,
    candidate: ImageU8,
    *,
    max_centroid_shift_fraction: float = _MAX_CENTROID_SHIFT_FRACTION,
    min_area_retention: float = _MIN_AREA_RETENTION,
) -> BoundsGateTrace:
    if (
        base_image.dtype != np.uint8
        or candidate.dtype != np.uint8
        or base_image.ndim != 3
        or candidate.ndim != 3
        or base_image.shape[2] not in (3, 4)
        or candidate.shape[2] not in (3, 4)
    ):
        return BoundsGateTrace(
            passed=False,
            failure_reason="invalid_input",
            margins_px=None,
            border_touch_count=None,
            dx_frac=None,
            dy_frac=None,
            area_retention=None,
        )
    if base_image.shape[:2] != candidate.shape[:2]:
        return BoundsGateTrace(
            passed=False,
            failure_reason="shape_mismatch",
            margins_px=None,
            border_touch_count=None,
            dx_frac=None,
            dy_frac=None,
            area_retention=None,
        )
    base_mask = _visible_notation_mask(base_image)
    candidate_mask = _visible_notation_mask(candidate)
    return evaluate_oob_gate_from_masks(
        base_mask.astype(np.uint8) * 255,
        candidate_mask.astype(np.uint8) * 255,
        max_centroid_shift_fraction=max_centroid_shift_fraction,
        min_area_retention=min_area_retention,
    )


def evaluate_outer_gate(
    base_image: ImageU8,
    candidate: ImageU8,
) -> OuterGateTrace:
    quality_gate = evaluate_quality_gate(candidate)
    transform_consistency = evaluate_transform_consistency(base_image, candidate)
    failure_reason = None
    if not quality_gate.passed:
        failure_reason = f"quality:{quality_gate.failure_reason}"
    elif not transform_consistency.passed:
        failure_reason = f"transform_consistency:{transform_consistency.failure_reason}"
    return OuterGateTrace(
        passed=quality_gate.passed and transform_consistency.passed,
        failure_reason=failure_reason,
        quality_gate=quality_gate,
        transform_consistency=transform_consistency,
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
) -> AugmentedCandidate:
    """Warp foreground and alpha layers geometrically.

    Returns the warped layers, visible mask, and sampled geometry metadata.
    Background synthesis is handled separately after the OOB gate.
    """
    height, width = base_layers.foreground.shape[:2]
    content_mask = _merge_visible_and_hint_masks(
        _visible_notation_mask(base_layers.foreground),
        base_layers.alpha >= 8,
    )
    transform = sample_geometric_transform(
        (height, width),
        rng,
        conservative=conservative,
        x_squeeze_prob=geom_x_squeeze_prob,
        x_squeeze_min_scale=geom_x_squeeze_min_scale,
        x_squeeze_max_scale=geom_x_squeeze_max_scale,
        x_squeeze_apply_in_conservative=geom_x_squeeze_apply_in_conservative,
        x_squeeze_force_scale=geom_x_squeeze_preview_force_scale,
        content_mask=content_mask,
        min_margin_px=_MIN_MARGIN_PX,
    )
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
    return AugmentedCandidate(
        foreground=warped_fg,
        alpha=warped_alpha,
        mask=visible_mask.astype(np.uint8) * 255,
        geometry=_geometry_trace_from_transform(transform, conservative=conservative),
    )


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
    return _border_touch_count(mask) >= 3


def _border_touch_count(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 4
    top = int(coords[:, 0].min()) == 0
    bottom = int(coords[:, 0].max()) == (mask.shape[0] - 1)
    left = int(coords[:, 1].min()) == 0
    right = int(coords[:, 1].max()) == (mask.shape[1] - 1)
    return sum((top, bottom, left, right))


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


def _margins_from_bbox(
    bbox: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int],
) -> MarginTrace | None:
    if bbox is None:
        return None
    top, bottom, left, right = bbox
    height, width = image_shape
    return MarginTrace(
        top_px=int(top),
        bottom_px=int(height - 1 - bottom),
        left_px=int(left),
        right_px=int(width - 1 - right),
    )


def _bbox_center_and_area(bbox: tuple[int, int, int, int]) -> tuple[float, float, float]:
    top, bottom, left, right = bbox
    return (
        (left + right) / 2.0,
        (top + bottom) / 2.0,
        float((right - left + 1) * (bottom - top + 1)),
    )


def evaluate_oob_gate_from_masks(
    base_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    min_margin_px: int = _MIN_MARGIN_PX,
    max_centroid_shift_fraction: float = _MAX_CENTROID_SHIFT_FRACTION,
    min_area_retention: float = _MIN_AREA_RETENTION,
) -> BoundsGateTrace:
    if base_mask.shape != candidate_mask.shape:
        return BoundsGateTrace(
            passed=False,
            failure_reason="shape_mismatch",
            margins_px=None,
            border_touch_count=None,
            dx_frac=None,
            dy_frac=None,
            area_retention=None,
        )
    base_bbox = _content_bbox_from_mask(base_mask > 0)
    cand_bbox = _content_bbox_from_mask(candidate_mask > 0)
    if base_bbox is None:
        return BoundsGateTrace(
            passed=False,
            failure_reason="empty_base",
            margins_px=None,
            border_touch_count=None,
            dx_frac=None,
            dy_frac=None,
            area_retention=None,
        )
    if cand_bbox is None:
        return BoundsGateTrace(
            passed=False,
            failure_reason="empty_candidate",
            margins_px=None,
            border_touch_count=None,
            dx_frac=None,
            dy_frac=None,
            area_retention=None,
        )
    margins = _margins_from_bbox(cand_bbox, candidate_mask.shape)
    border_touch_count = _border_touch_count(candidate_mask > 0)
    if not _has_minimum_margins(cand_bbox, candidate_mask.shape, min_margin_px=min_margin_px):
        return BoundsGateTrace(
            passed=False,
            failure_reason="min_margin",
            margins_px=margins,
            border_touch_count=border_touch_count,
            dx_frac=None,
            dy_frac=None,
            area_retention=None,
        )
    if border_touch_count >= 3:
        return BoundsGateTrace(
            passed=False,
            failure_reason="border_touches",
            margins_px=margins,
            border_touch_count=border_touch_count,
            dx_frac=None,
            dy_frac=None,
            area_retention=None,
        )

    base_cx, base_cy, base_area = _bbox_center_and_area(base_bbox)
    cand_cx, cand_cy, cand_area = _bbox_center_and_area(cand_bbox)
    if base_area <= 0:
        return BoundsGateTrace(
            passed=False,
            failure_reason="empty_base",
            margins_px=margins,
            border_touch_count=border_touch_count,
            dx_frac=None,
            dy_frac=None,
            area_retention=None,
        )
    height, width = base_mask.shape[:2]
    dx_frac = abs(cand_cx - base_cx) / max(width, 1)
    dy_frac = abs(cand_cy - base_cy) / max(height, 1)
    area_retention = cand_area / base_area
    if dx_frac > max_centroid_shift_fraction:
        return BoundsGateTrace(
            passed=False,
            failure_reason="centroid_shift_x",
            margins_px=margins,
            border_touch_count=border_touch_count,
            dx_frac=dx_frac,
            dy_frac=dy_frac,
            area_retention=area_retention,
        )
    if dy_frac > max_centroid_shift_fraction:
        return BoundsGateTrace(
            passed=False,
            failure_reason="centroid_shift_y",
            margins_px=margins,
            border_touch_count=border_touch_count,
            dx_frac=dx_frac,
            dy_frac=dy_frac,
            area_retention=area_retention,
        )
    if area_retention < min_area_retention:
        return BoundsGateTrace(
            passed=False,
            failure_reason="area_retention",
            margins_px=margins,
            border_touch_count=border_touch_count,
            dx_frac=dx_frac,
            dy_frac=dy_frac,
            area_retention=area_retention,
        )
    return BoundsGateTrace(
        passed=True,
        failure_reason=None,
        margins_px=margins,
        border_touch_count=border_touch_count,
        dx_frac=dx_frac,
        dy_frac=dy_frac,
        area_retention=area_retention,
    )


def _passes_out_of_bounds_gate_from_masks(
    base_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    min_margin_px: int = _MIN_MARGIN_PX,
    max_centroid_shift_fraction: float = _MAX_CENTROID_SHIFT_FRACTION,
    min_area_retention: float = _MIN_AREA_RETENTION,
) -> bool:
    return evaluate_oob_gate_from_masks(
        base_mask,
        candidate_mask,
        min_margin_px=min_margin_px,
        max_centroid_shift_fraction=max_centroid_shift_fraction,
        min_area_retention=min_area_retention,
    ).passed


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
        initial_geometry=_geometry_trace_from_transform(None, conservative=False),
        retry_geometry=None,
        selected_geometry=_geometry_trace_from_transform(None, conservative=False),
        final_geometry_applied=False,
        initial_oob_gate=BoundsGateTrace(
            passed=True,
            failure_reason=None,
            margins_px=None,
            border_touch_count=None,
            dx_frac=None,
            dy_frac=None,
            area_retention=None,
        ),
        retry_oob_gate=None,
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
