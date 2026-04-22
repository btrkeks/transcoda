"""Single-pass score generation."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .image_post import (
    pad_or_crop_alpha_to_height,
    pad_or_crop_to_height,
    resize_to_width,
)
from .metadata_generator import generate_metadata_prefix
from .rendering.verovio_backend import (
    VEROVIO_FONTS,
    VerovioRenderer,
    VerovioRenderOptions,
)
from .types import GeneratedScore, GenerationConfig, RenderedPage

logger = logging.getLogger(__name__)

_MAX_RENDER_ATTEMPTS = 3
_MIN_FRAME_MARGIN_PX = 2
_TARGET_FRAME_MARGIN_PX = 12

RenderRejectionReason = Literal[
    "multi_page",
    "top_clearance",
    "bottom_clearance",
    "left_clearance",
    "right_clearance",
    "crop_risk",
    "no_content_detected",
]
_RETRYABLE_REJECTION_REASONS: set[RenderRejectionReason] = {
    "top_clearance",
    "bottom_clearance",
    "left_clearance",
    "right_clearance",
    "crop_risk",
    "no_content_detected",
}
_HORIZONTAL_RETRY_REASONS: set[RenderRejectionReason] = {"left_clearance", "right_clearance"}


@dataclass(frozen=True)
class _RenderAttemptResult:
    """Internal state for a single render attempt."""

    image: np.ndarray | None
    render_layers: RenderedPage | None
    system_count: int
    page_count: int
    render_options: VerovioRenderOptions
    rejection_reason: RenderRejectionReason | None = None
    quality_metrics: dict[str, int | None] | None = None


def _find_content_bbox(
    image: np.ndarray,
    *,
    threshold: int = 100,
) -> tuple[int, int, int, int] | None:
    """Return content bbox as (top, bottom, left, right), or None when empty."""
    if image.ndim == 3:
        black = np.all(image[:, :, :3] <= threshold, axis=2)
    else:
        black = image <= threshold

    if not black.any():
        return None

    coords = np.argwhere(black)
    top = int(coords[:, 0].min())
    bottom = int(coords[:, 0].max())
    left = int(coords[:, 1].min())
    right = int(coords[:, 1].max())
    return top, bottom, left, right


def _clamp_float(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def _lerp(min_value: float, max_value: float, t: float) -> float:
    return min_value + (max_value - min_value) * t


def _bernoulli(probability: float) -> bool:
    return random.random() < probability


def _sample_page_width_factor() -> float:
    roll = random.random()
    if roll < 0.20:
        return random.uniform(1.18, 1.38)
    if roll < 0.48:
        return random.uniform(1.65, 1.95)
    if roll < 0.84:
        return random.uniform(1.95, 2.25)
    return random.uniform(2.25, 2.70)


def _apply_render_option_guardrails(
    options: VerovioRenderOptions,
    *,
    page_width_factor: float,
) -> None:
    """Cap extreme combinations that tend to produce unrealistic layouts."""
    if page_width_factor < 1.85:
        if "spacingSystem" in options:
            options["spacingSystem"] = min(options["spacingSystem"], 18)
        if "spacingStaff" in options:
            options["spacingStaff"] = min(options["spacingStaff"], 20)

    if page_width_factor > 2.4 and "measureMinWidth" in options:
        # Avoid overly sparse pages when page width is large.
        options["measureMinWidth"] = min(options["measureMinWidth"], 26)

    if options.get("staffLineWidth", 0.0) > 0.26:
        if "stemWidth" in options:
            options["stemWidth"] = min(options["stemWidth"], 0.40)
        if "ledgerLineThickness" in options:
            options["ledgerLineThickness"] = min(options["ledgerLineThickness"], 0.42)


def generate(
    transcription: str,
    renderer: VerovioRenderer,
    config: GenerationConfig,
) -> GeneratedScore | None:
    """Generate a single score.

    Returns None on any rejected rendering outcome.
    """
    score, _ = generate_with_diagnostics(transcription, renderer, config)
    return score


def generate_with_diagnostics(
    transcription: str,
    renderer: VerovioRenderer,
    config: GenerationConfig,
) -> tuple[GeneratedScore | None, RenderRejectionReason | None]:
    """Generate a single score with one render pass.

    Args:
        transcription: The kern transcription to render.
        renderer: Verovio renderer instance.
        config: Generation configuration.

    Returns:
        Tuple of (GeneratedScore or None, rejection reason).
    """
    # Generate metadata
    metadata_prefix = generate_metadata_prefix(config.include_title, config.include_author)

    # Prepare renderable content
    renderable = metadata_prefix + transcription

    render_options = _sample_render_options(
        config.image_width,
        layout_profile=config.render_layout_profile,
    )

    for attempt_idx in range(1, _MAX_RENDER_ATTEMPTS + 1):
        attempt = _render_once_with_options(
            renderable=renderable,
            renderer=renderer,
            config=config,
            render_options=render_options,
        )

        if attempt.rejection_reason is not None:
            if attempt.rejection_reason == "multi_page":
                logger.debug(
                    "Discarding multi-page render (attempt=%d, pages=%d)",
                    attempt_idx,
                    attempt.page_count,
                )
                return None, attempt.rejection_reason

            logger.debug(
                "Rejecting render attempt pre-gate (attempt=%d, reason=%s, metrics=%s)",
                attempt_idx,
                attempt.rejection_reason,
                attempt.quality_metrics,
            )
            if (
                attempt_idx < _MAX_RENDER_ATTEMPTS
                and attempt.rejection_reason in _RETRYABLE_REJECTION_REASONS
            ):
                render_options = _next_retry_render_options(
                    attempt_idx=attempt_idx,
                    options=attempt.render_options,
                    image_width=config.image_width,
                    rejection_reason=attempt.rejection_reason,
                )
                logger.debug(
                    "Retrying with compact vertical fallback (attempt=%d -> %d)",
                    attempt_idx,
                    attempt_idx + 1,
                )
                continue
            return None, attempt.rejection_reason

        assert attempt.image is not None, "Single-page attempt is missing image data"
        is_ok, rejection_reason, metrics = _assess_frame_fit(
            attempt.image,
            target_height=config.image_height,
            min_frame_margin_px=_MIN_FRAME_MARGIN_PX,
            target_frame_margin_px=_TARGET_FRAME_MARGIN_PX,
        )
        if is_ok:
            final_image = _finalize_image_height(attempt.image, config)
            final_layers = None
            if attempt.render_layers is not None:
                final_layers = RenderedPage(
                    image=final_image,
                    foreground=_finalize_image_height(attempt.render_layers.foreground, config),
                    alpha=pad_or_crop_alpha_to_height(
                        attempt.render_layers.alpha,
                        config.image_height,
                    ),
                )
            if attempt_idx > 1:
                logger.debug("Render retry succeeded (attempt=%d)", attempt_idx)
            return (
                GeneratedScore(
                    image=final_image,
                    render_layers=final_layers,
                    transcription=transcription,
                    actual_system_count=attempt.system_count,
                    metadata_prefix=metadata_prefix,
                    bottom_whitespace_ratio=(
                        float(metrics["bottom_whitespace_ratio"])
                        if metrics.get("bottom_whitespace_ratio") is not None
                        else None
                    ),
                    vertical_fill_ratio=(
                        float(metrics["vertical_fill_ratio"])
                        if metrics.get("vertical_fill_ratio") is not None
                        else None
                    ),
                ),
                None,
            )

        assert rejection_reason is not None
        logger.debug(
            "Rejecting render attempt (attempt=%d, reason=%s, metrics=%s)",
            attempt_idx,
            rejection_reason,
            metrics,
        )

        if (
            attempt_idx < _MAX_RENDER_ATTEMPTS
            and rejection_reason in _RETRYABLE_REJECTION_REASONS
        ):
            render_options = _next_retry_render_options(
                attempt_idx=attempt_idx,
                options=attempt.render_options,
                image_width=config.image_width,
                rejection_reason=rejection_reason,
            )
            logger.debug(
                "Retrying with compact vertical fallback (attempt=%d -> %d)",
                attempt_idx,
                attempt_idx + 1,
            )
            continue

        return None, rejection_reason

    return None, "no_content_detected"


def _render_once_with_options(
    *,
    renderable: str,
    renderer: VerovioRenderer,
    config: GenerationConfig,
    render_options: VerovioRenderOptions,
) -> _RenderAttemptResult:
    """Render once and apply pre-gate image post-processing."""
    rendered_page, system_count, page_count = renderer.render_with_counts(renderable, render_options)

    if page_count > 1:
        return _RenderAttemptResult(
            image=None,
            render_layers=None,
            system_count=system_count,
            page_count=page_count,
            render_options=render_options,
            rejection_reason="multi_page",
        )

    if isinstance(rendered_page, np.ndarray):
        image = resize_to_width(rendered_page, config.image_width)
        render_layers = None
    else:
        image = resize_to_width(rendered_page.image, config.image_width)
        foreground = resize_to_width(rendered_page.foreground, config.image_width)
        alpha = resize_to_width(rendered_page.alpha, config.image_width)
        if alpha.ndim == 3:
            alpha = alpha[:, :, 0]
        render_layers = RenderedPage(
            image=np.ascontiguousarray(image),
            foreground=np.ascontiguousarray(foreground),
            alpha=np.ascontiguousarray(alpha.astype(np.uint8)),
        )

    return _RenderAttemptResult(
        image=np.ascontiguousarray(image),
        render_layers=render_layers,
        system_count=system_count,
        page_count=page_count,
        render_options=render_options,
    )


def _assess_frame_fit(
    image: np.ndarray,
    *,
    target_height: int | None,
    min_frame_margin_px: int,
    target_frame_margin_px: int = _TARGET_FRAME_MARGIN_PX,
) -> tuple[bool, RenderRejectionReason | None, dict[str, int | float | None]]:
    """Reject images whose content is too close to current or crop edges."""
    current_height = int(image.shape[0])
    current_width = int(image.shape[1])
    bbox = _find_content_bbox(image)

    metrics: dict[str, int | float | None] = {
        "current_height": current_height,
        "current_width": current_width,
        "target_height": target_height,
        "frame_height": target_height if target_height is not None else current_height,
        "top_row": None,
        "bottom_row": None,
        "left_col": None,
        "right_col": None,
        "content_height_px": None,
        "content_width_px": None,
        "top_clearance_px": None,
        "bottom_clearance_px": None,
        "left_clearance_px": None,
        "right_clearance_px": None,
        "cropped_clearance_px": None,
        "final_bottom_whitespace_px": None,
        "bottom_whitespace_ratio": None,
        "vertical_fill_ratio": None,
        "hard_margin_px": None,
        "target_margin_px": None,
        "borderline_clearance_px": None,
    }

    if bbox is None:
        return False, "no_content_detected", metrics

    top_row, bottom_row, left_col, right_col = bbox
    top_clearance_px = top_row
    bottom_clearance_px = current_height - 1 - bottom_row
    left_clearance_px = left_col
    right_clearance_px = current_width - 1 - right_col

    metrics["top_row"] = top_row
    metrics["bottom_row"] = bottom_row
    metrics["left_col"] = left_col
    metrics["right_col"] = right_col
    metrics["content_height_px"] = bottom_row - top_row + 1
    metrics["content_width_px"] = right_col - left_col + 1
    metrics["top_clearance_px"] = top_clearance_px
    metrics["bottom_clearance_px"] = bottom_clearance_px
    metrics["left_clearance_px"] = left_clearance_px
    metrics["right_clearance_px"] = right_clearance_px
    frame_height = int(target_height) if target_height is not None else current_height
    final_bottom_whitespace_px = frame_height - 1 - bottom_row
    metrics["final_bottom_whitespace_px"] = int(final_bottom_whitespace_px)
    if frame_height > 0:
        metrics["bottom_whitespace_ratio"] = max(0.0, float(final_bottom_whitespace_px) / frame_height)
        metrics["vertical_fill_ratio"] = float(bottom_row - top_row + 1) / frame_height
    hard_margin_px = max(0, int(min_frame_margin_px))
    target_margin_px = max(hard_margin_px, int(target_frame_margin_px))
    metrics["hard_margin_px"] = hard_margin_px
    metrics["target_margin_px"] = target_margin_px

    if top_clearance_px < hard_margin_px:
        return False, "top_clearance", metrics
    if bottom_clearance_px < hard_margin_px:
        return False, "bottom_clearance", metrics
    if left_clearance_px < hard_margin_px:
        return False, "left_clearance", metrics
    if right_clearance_px < hard_margin_px:
        return False, "right_clearance", metrics

    if target_height is not None and current_height > target_height:
        cropped_clearance_px = target_height - 1 - bottom_row
        metrics["cropped_clearance_px"] = int(cropped_clearance_px)
        if cropped_clearance_px < hard_margin_px:
            return False, "crop_risk", metrics
        if cropped_clearance_px < target_margin_px:
            metrics["borderline_clearance_px"] = int(cropped_clearance_px)

    min_clearance = min(top_clearance_px, bottom_clearance_px, left_clearance_px, right_clearance_px)
    if metrics["borderline_clearance_px"] is None and min_clearance < target_margin_px:
        metrics["borderline_clearance_px"] = int(min_clearance)

    return True, None, metrics


def _finalize_image_height(image: np.ndarray, config: GenerationConfig) -> np.ndarray:
    """Apply final height normalization after quality checks pass."""
    if config.image_height is not None:
        return pad_or_crop_to_height(image, config.image_height)
    return image


def _make_vertical_compact_fallback(
    options: VerovioRenderOptions,
    *,
    image_width: int,
) -> VerovioRenderOptions:
    """Make a deterministic fallback with reduced vertical layout pressure."""
    fallback: VerovioRenderOptions = dict(options)

    if "spacingSystem" in fallback:
        fallback["spacingSystem"] = _clamp_int(min(int(fallback["spacingSystem"]), 12), 4, 20)
    if "spacingStaff" in fallback:
        fallback["spacingStaff"] = _clamp_int(int(fallback["spacingStaff"]) - 2, 4, 48)
    if "scale" in fallback:
        fallback["scale"] = _clamp_int(int(fallback["scale"]) - 10, 40, 1000)
    if "pageMarginTop" in fallback:
        fallback["pageMarginTop"] = _clamp_int(int(fallback["pageMarginTop"]) - 20, 20, 500)
    if "pageMarginBottom" in fallback:
        fallback["pageMarginBottom"] = _clamp_int(int(fallback["pageMarginBottom"]) - 20, 20, 500)

    fallback["justifyVertically"] = False

    page_width = int(fallback.get("pageWidth", image_width * 2))
    page_width_factor = max(page_width / image_width, 0.1)
    _apply_render_option_guardrails(fallback, page_width_factor=page_width_factor)
    return fallback


def _make_horizontal_compact_fallback(
    options: VerovioRenderOptions,
    *,
    image_width: int,
    rejection_reason: RenderRejectionReason | None,
) -> VerovioRenderOptions:
    """Reduce horizontal pressure when left/right boundary failures occur."""
    fallback: VerovioRenderOptions = dict(options)

    if "scale" in fallback:
        fallback["scale"] = _clamp_int(int(fallback["scale"]) - 8, 40, 1000)
    if "spacingLinear" in fallback:
        fallback["spacingLinear"] = _clamp_float(float(fallback["spacingLinear"]) - 0.04, 0.12, 1.0)
    if "spacingNonLinear" in fallback:
        fallback["spacingNonLinear"] = _clamp_float(float(fallback["spacingNonLinear"]) - 0.08, 0.20, 1.5)
    if "measureMinWidth" in fallback:
        fallback["measureMinWidth"] = _clamp_int(int(fallback["measureMinWidth"]) - 4, 1, 30)
    if "pageWidth" in fallback:
        page_width = int(fallback["pageWidth"])
        fallback["pageWidth"] = _clamp_int(page_width + max(80, int(round(page_width * 0.08))), image_width, 5000)

    left_bump = 16
    right_bump = 16
    if rejection_reason == "left_clearance":
        left_bump = 28
        right_bump = 8
    elif rejection_reason == "right_clearance":
        left_bump = 8
        right_bump = 28
    if "pageMarginLeft" in fallback:
        fallback["pageMarginLeft"] = _clamp_int(int(fallback["pageMarginLeft"]) + left_bump, 0, 500)
    if "pageMarginRight" in fallback:
        fallback["pageMarginRight"] = _clamp_int(int(fallback["pageMarginRight"]) + right_bump, 0, 500)

    page_width = int(fallback.get("pageWidth", image_width * 2))
    page_width_factor = max(page_width / image_width, 0.1)
    _apply_render_option_guardrails(fallback, page_width_factor=page_width_factor)
    return fallback


def _make_verovio_fit_fallback(
    options: VerovioRenderOptions,
    *,
    image_width: int,
    rejection_reason: RenderRejectionReason | None = None,
) -> VerovioRenderOptions:
    """Enable Verovio fit options for retry-only recovery."""
    if rejection_reason in _HORIZONTAL_RETRY_REASONS:
        base = _make_horizontal_compact_fallback(
            options,
            image_width=image_width,
            rejection_reason=rejection_reason,
        )
        fallback: VerovioRenderOptions = _make_vertical_compact_fallback(base, image_width=image_width)
    else:
        fallback = _make_vertical_compact_fallback(options, image_width=image_width)
    fallback["adjustPageHeight"] = True
    fallback["shrinkToFit"] = True
    fallback["adjustPageWidth"] = False
    fallback["justifyVertically"] = False
    return fallback


def _next_retry_render_options(
    *,
    attempt_idx: int,
    options: VerovioRenderOptions,
    image_width: int,
    rejection_reason: RenderRejectionReason | None,
) -> VerovioRenderOptions:
    """Choose deterministic retry fallback by attempt number."""
    if attempt_idx == 1:
        if rejection_reason in _HORIZONTAL_RETRY_REASONS:
            return _make_horizontal_compact_fallback(
                options,
                image_width=image_width,
                rejection_reason=rejection_reason,
            )
        return _make_vertical_compact_fallback(options, image_width=image_width)
    return _make_verovio_fit_fallback(
        options,
        image_width=image_width,
        rejection_reason=rejection_reason,
    )


def _sample_render_options(
    image_width: int,
    *,
    layout_profile: str = "default",
) -> VerovioRenderOptions:
    """Build randomized Verovio options for visual variety.

    Args:
        image_width: Target output width in pixels. pageWidth is set to 2x this
            value to render at higher resolution before downscaling for quality.
    """
    if layout_profile == "target_5_6_systems":
        options = _sample_render_options_target_5_6_systems(image_width)
        options["font"] = random.choice(VEROVIO_FONTS)
        return options
    if layout_profile != "default":
        raise ValueError(
            f"Unsupported render_layout_profile: {layout_profile}"
        )

    layout_density = random.random()
    vertical_openness = random.random()
    stroke_heaviness = random.random()
    page_width_factor = _sample_page_width_factor()
    if page_width_factor < 1.40:
        page_width = int(round(image_width * page_width_factor))
        margin_ratio = lambda: random.uniform(0.010, 0.022)
        options: VerovioRenderOptions = {
            "scale": random.randint(76, 90),
            "barLineWidth": _clamp_float(random.uniform(0.18, 0.72), 0.10, 0.80),
            "beamMaxSlope": random.randint(4, 16),
            "staffLineWidth": _clamp_float(random.uniform(0.10, 0.24), 0.10, 0.30),
            "stemWidth": _clamp_float(random.uniform(0.10, 0.36), 0.10, 0.45),
            "ledgerLineThickness": _clamp_float(random.uniform(0.10, 0.38), 0.10, 0.50),
            "thickBarlineThickness": _clamp_float(random.uniform(0.70, 1.80), 0.50, 2.00),
            "spacingLinear": _clamp_float(random.uniform(0.24, 0.36), 0.12, 1.0),
            "spacingNonLinear": _clamp_float(random.uniform(0.44, 0.68), 0.20, 1.5),
            "spacingStaff": random.randint(6, 11),
            "spacingSystem": random.randint(7, 13),
            "measureMinWidth": random.randint(12, 22),
            "pageMarginLeft": _clamp_int(int(round(page_width * margin_ratio())), 0, 500),
            "pageMarginRight": _clamp_int(int(round(page_width * margin_ratio())), 0, 500),
            "pageMarginTop": _clamp_int(int(round(page_width * margin_ratio())), 0, 500),
            "pageMarginBottom": _clamp_int(int(round(page_width * margin_ratio())), 0, 500),
            "pageWidth": page_width,
            "font": random.choice(VEROVIO_FONTS),
            "breaksNoWidow": True,
            "justifyVertically": True,
            "noJustification": False,
            "footer": "none",
            "breaks": "auto",
        }
        _apply_render_option_guardrails(options, page_width_factor=page_width_factor)
        return options

    page_width_band_openness = _clamp_float((page_width_factor - 1.65) / (2.70 - 1.65), 0.0, 1.0)
    layout_openness = 1.0 - layout_density

    page_width = int(round(image_width * page_width_factor))

    scale_driver = _clamp_float(
        0.65 * page_width_band_openness + 0.35 * layout_openness,
        0.0,
        1.0,
    )
    # Verovio scale is percent (100 = normal size). Keep diversity while allowing a lower-size tail.
    scale = _clamp_int(
        int(round(50 + 90 * scale_driver + random.uniform(-12.0, 12.0))),
        40,
        145,
    )

    margin_openness = _clamp_float(0.7 * layout_openness + 0.3 * page_width_band_openness, 0.0, 1.0)
    lr_base = _lerp(0.015, 0.085, margin_openness)
    tb_base = _lerp(0.010, 0.070, margin_openness)
    lr_jitter = 0.015
    tb_jitter = 0.012
    page_margin_left = _clamp_int(
        int(
            round(
                page_width
                * _clamp_float(lr_base + random.uniform(-lr_jitter, lr_jitter), 0.015, 0.085)
            )
        ),
        0,
        500,
    )
    page_margin_right = _clamp_int(
        int(
            round(
                page_width
                * _clamp_float(lr_base + random.uniform(-lr_jitter, lr_jitter), 0.015, 0.085)
            )
        ),
        0,
        500,
    )
    page_margin_top = _clamp_int(
        int(
            round(
                page_width
                * _clamp_float(tb_base + random.uniform(-tb_jitter, tb_jitter), 0.010, 0.070)
            )
        ),
        0,
        500,
    )
    page_margin_bottom = _clamp_int(
        int(
            round(
                page_width
                * _clamp_float(tb_base + random.uniform(-tb_jitter, tb_jitter), 0.010, 0.070)
            )
        ),
        0,
        500,
    )

    horizontal_openness = _clamp_float(0.6 * layout_openness + 0.4 * page_width_band_openness, 0.0, 1.0)
    spacing_linear = _clamp_float(
        _lerp(0.18, 0.42, horizontal_openness) + random.uniform(-0.04, 0.04),
        0.18,
        0.42,
    )
    spacing_non_linear = _clamp_float(
        _lerp(0.35, 0.85, horizontal_openness) + random.uniform(-0.08, 0.08),
        0.35,
        0.85,
    )

    scale_openness = _clamp_float((scale - 65) / (145 - 65), 0.0, 1.0)
    measure_driver = _clamp_float(
        0.55 * horizontal_openness + 0.25 * layout_openness + 0.20 * scale_openness,
        0.0,
        1.0,
    )
    measure_min_width = _clamp_int(
        int(round(_lerp(8.0, 28.0, measure_driver) + random.uniform(-3.0, 3.0))),
        1,
        30,
    )

    compact_page_penalty = _clamp_float((1.85 - page_width_factor) / 0.20, 0.0, 1.0)
    vertical_driver = _clamp_float(vertical_openness - 0.2 * compact_page_penalty, 0.0, 1.0)
    spacing_staff = _clamp_int(
        int(round(4 + 22 * vertical_driver + random.uniform(-2.0, 2.0))),
        4,
        26,
    )
    spacing_system = _clamp_int(
        int(round(4 + 18 * vertical_driver + random.uniform(-2.0, 2.0))),
        4,
        20,
    )

    staff_line_width = _clamp_float(
        _lerp(0.10, 0.30, stroke_heaviness) + random.uniform(-0.02, 0.02),
        0.10,
        0.30,
    )
    bar_line_width = _clamp_float(
        _lerp(0.16, 0.80, stroke_heaviness) + random.uniform(-0.06, 0.06),
        0.10,
        0.80,
    )
    stem_width = _clamp_float(
        _lerp(0.10, 0.45, stroke_heaviness) + random.uniform(-0.04, 0.04),
        0.10,
        0.45,
    )
    ledger_line_thickness = _clamp_float(
        _lerp(0.10, 0.50, stroke_heaviness) + random.uniform(-0.05, 0.05),
        0.10,
        0.50,
    )
    thick_barline_thickness = _clamp_float(
        _lerp(0.60, 2.00, stroke_heaviness) + random.uniform(-0.16, 0.16),
        0.50,
        2.00,
    )

    beam_max_slope = _clamp_int(
        int(round(13 + random.uniform(-7.0, 7.0) + random.uniform(-1.0, 1.0))),
        0,
        20,
    )

    options: VerovioRenderOptions = {
        "scale": scale,
        "barLineWidth": bar_line_width,
        "beamMaxSlope": beam_max_slope,
        "staffLineWidth": staff_line_width,
        "stemWidth": stem_width,
        "ledgerLineThickness": ledger_line_thickness,
        "thickBarlineThickness": thick_barline_thickness,
        "spacingLinear": spacing_linear,
        "spacingNonLinear": spacing_non_linear,
        "spacingStaff": spacing_staff,
        "spacingSystem": spacing_system,
        "measureMinWidth": measure_min_width,
        "pageMarginLeft": page_margin_left,
        "pageMarginRight": page_margin_right,
        "pageMarginTop": page_margin_top,
        "pageMarginBottom": page_margin_bottom,
        "pageWidth": page_width,
        "font": random.choice(VEROVIO_FONTS),
        "breaksNoWidow": _bernoulli(0.30),
        "justifyVertically": _bernoulli(0.15),
        "noJustification": _bernoulli(0.08),
        "footer": "none",
        "breaks": "auto",
    }
    _apply_render_option_guardrails(options, page_width_factor=page_width_factor)
    return options


def _sample_render_options_target_5_6_systems(image_width: int) -> VerovioRenderOptions:
    """Sample mildly compact render options for 5-6-system targets."""
    page_width_factor = random.uniform(2.08, 2.34)
    page_width = int(round(image_width * page_width_factor))

    margin_ratio = lambda: random.uniform(0.015, 0.040)
    options: VerovioRenderOptions = {
        "scale": random.randint(56, 74),
        "barLineWidth": _clamp_float(random.uniform(0.16, 0.70), 0.10, 0.80),
        "beamMaxSlope": random.randint(4, 16),
        "staffLineWidth": _clamp_float(random.uniform(0.10, 0.24), 0.10, 0.30),
        "stemWidth": _clamp_float(random.uniform(0.10, 0.36), 0.10, 0.45),
        "ledgerLineThickness": _clamp_float(random.uniform(0.10, 0.38), 0.10, 0.50),
        "thickBarlineThickness": _clamp_float(random.uniform(0.60, 1.60), 0.50, 2.00),
        "spacingLinear": _clamp_float(random.uniform(0.15, 0.27), 0.12, 1.0),
        "spacingNonLinear": _clamp_float(random.uniform(0.24, 0.46), 0.20, 1.5),
        "spacingStaff": random.randint(5, 18),
        "spacingSystem": random.randint(4, 11),
        "measureMinWidth": random.randint(5, 14),
        "pageMarginLeft": _clamp_int(int(round(page_width * margin_ratio())), 0, 500),
        "pageMarginRight": _clamp_int(int(round(page_width * margin_ratio())), 0, 500),
        "pageMarginTop": _clamp_int(int(round(page_width * margin_ratio())), 0, 500),
        "pageMarginBottom": _clamp_int(int(round(page_width * margin_ratio())), 0, 500),
        "pageWidth": page_width,
        "breaksNoWidow": False,
        "justifyVertically": False,
        "noJustification": False,
        "footer": "none",
        "breaks": "auto",
    }
    _apply_render_option_guardrails(options, page_width_factor=page_width_factor)
    return options
