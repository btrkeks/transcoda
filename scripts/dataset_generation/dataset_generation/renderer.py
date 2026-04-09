"""Rendering and SVG-derived diagnostics for the rewrite pipeline."""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np

from scripts.dataset_generation.dataset_generation.image_generation.image_post import (
    pad_or_crop_to_height,
    resize_to_width,
)
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderOptions,
    VerovioRenderer,
    count_nr_of_systems_in_svg,
)
from scripts.dataset_generation.dataset_generation.image_generation.types import RenderedPage
from scripts.dataset_generation.dataset_generation_new.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation_new.types import RenderResult, SvgLayoutDiagnostics

_RETRYABLE_REJECTION_REASONS = {
    "top_clearance",
    "bottom_clearance",
    "left_clearance",
    "right_clearance",
    "crop_risk",
    "no_content_detected",
}


def count_systems_in_svg(svg: str) -> int:
    return count_nr_of_systems_in_svg(svg)


def render_sample(
    render_transcription: str,
    recipe: ProductionRecipe,
    *,
    seed: int,
    renderer: VerovioRenderer | None = None,
) -> RenderResult:
    return _render_sample_impl(
        render_transcription,
        recipe,
        seed=seed,
        renderer=renderer,
        mode="default",
    )


def render_sample_with_layout_rescue(
    render_transcription: str,
    recipe: ProductionRecipe,
    *,
    seed: int,
    renderer: VerovioRenderer | None = None,
) -> RenderResult:
    return _render_sample_impl(
        render_transcription,
        recipe,
        seed=seed,
        renderer=renderer,
        mode="preferred_5_6_rescue",
    )


def prepare_render_attempt(
    render_transcription: str,
    recipe: ProductionRecipe,
    *,
    seed: int,
    mode: str = "default",
) -> dict[str, object]:
    with _seeded_random(seed):
        metadata_prefix = _generate_metadata_prefix(recipe)
        render_options = _sample_render_options(recipe)
        max_attempts = recipe.render_only_aug.max_render_attempts
        if mode == "preferred_5_6_rescue":
            render_options = _tighten_layout_for_preferred_5_6_rescue(render_options)
            max_attempts = max(1, min(2, recipe.render_only_aug.max_render_attempts))
        return {
            "mode": mode,
            "seed": int(seed),
            "metadata_prefix": metadata_prefix,
            "render_transcription": render_transcription,
            "renderable": metadata_prefix + render_transcription,
            "render_options": dict(render_options),
            "max_attempts": int(max_attempts),
        }


def _render_sample_impl(
    render_transcription: str,
    recipe: ProductionRecipe,
    *,
    seed: int,
    renderer: VerovioRenderer | None = None,
    mode: str = "default",
) -> RenderResult:
    active_renderer = renderer or VerovioRenderer()
    prepared_attempt = prepare_render_attempt(
        render_transcription,
        recipe,
        seed=seed,
        mode=mode,
    )
    metadata_prefix = str(prepared_attempt["metadata_prefix"])
    renderable = str(prepared_attempt["renderable"])
    render_options = dict(prepared_attempt["render_options"])
    max_attempts = int(prepared_attempt["max_attempts"])

    for attempt_idx in range(1, max_attempts + 1):
        try:
            rendered_page, system_count, page_count = active_renderer.render_with_counts(
                renderable,
                render_options,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            return RenderResult(
                image=None,
                render_layers=None,
                svg_diagnostics=SvgLayoutDiagnostics(system_count=0, page_count=0),
                bottom_whitespace_ratio=None,
                vertical_fill_ratio=None,
                bottom_whitespace_px=None,
                top_whitespace_px=None,
                content_height_px=None,
                rejection_reason=f"render_error:{exc}",
                metadata_prefix=metadata_prefix,
            )

        diagnostics = SvgLayoutDiagnostics(system_count=system_count, page_count=page_count)
        if page_count > 1:
            return RenderResult(
                image=None,
                render_layers=None,
                svg_diagnostics=diagnostics,
                bottom_whitespace_ratio=None,
                vertical_fill_ratio=None,
                bottom_whitespace_px=None,
                top_whitespace_px=None,
                content_height_px=None,
                rejection_reason="multi_page",
                metadata_prefix=metadata_prefix,
            )

        image, render_layers = _resize_rendered_page(rendered_page, recipe.image_width)
        passed, rejection_reason, metrics = _assess_frame_fit(
            image,
            target_height=recipe.image_height,
            min_frame_margin_px=recipe.render_only_aug.min_frame_margin_px,
            target_frame_margin_px=recipe.render_only_aug.target_frame_margin_px,
        )
        if passed:
            final_image = pad_or_crop_to_height(image, recipe.image_height)
            final_layers = None
            if render_layers is not None:
                final_layers = RenderedPage(
                    image=final_image,
                    foreground=pad_or_crop_to_height(render_layers.foreground, recipe.image_height),
                    alpha=pad_or_crop_to_height(render_layers.alpha, recipe.image_height),
                )
            return RenderResult(
                image=final_image,
                render_layers=final_layers,
                svg_diagnostics=diagnostics,
                bottom_whitespace_ratio=_coerce_float(metrics.get("bottom_whitespace_ratio")),
                vertical_fill_ratio=_coerce_float(metrics.get("vertical_fill_ratio")),
                bottom_whitespace_px=_coerce_int(metrics.get("bottom_whitespace_px")),
                top_whitespace_px=_coerce_int(metrics.get("top_whitespace_px")),
                content_height_px=_coerce_int(metrics.get("content_height_px")),
                rejection_reason=None,
                metadata_prefix=metadata_prefix,
            )

        if (
            attempt_idx < max_attempts
            and rejection_reason in _RETRYABLE_REJECTION_REASONS
        ):
            if mode == "preferred_5_6_rescue":
                render_options = _next_preferred_5_6_rescue_options(
                    render_options,
                    rejection_reason=rejection_reason,
                )
            else:
                render_options = _next_retry_render_options(
                    render_options,
                    recipe,
                    rejection_reason=rejection_reason,
                )
            continue

        return RenderResult(
            image=None,
            render_layers=None,
            svg_diagnostics=diagnostics,
            bottom_whitespace_ratio=_coerce_float(metrics.get("bottom_whitespace_ratio")),
            vertical_fill_ratio=_coerce_float(metrics.get("vertical_fill_ratio")),
            bottom_whitespace_px=_coerce_int(metrics.get("bottom_whitespace_px")),
            top_whitespace_px=_coerce_int(metrics.get("top_whitespace_px")),
            content_height_px=_coerce_int(metrics.get("content_height_px")),
            rejection_reason=rejection_reason,
            metadata_prefix=metadata_prefix,
        )
    return RenderResult(
        image=None,
        render_layers=None,
        svg_diagnostics=SvgLayoutDiagnostics(system_count=0, page_count=0),
        bottom_whitespace_ratio=None,
        vertical_fill_ratio=None,
        bottom_whitespace_px=None,
        top_whitespace_px=None,
        content_height_px=None,
        rejection_reason="no_content_detected",
        metadata_prefix=metadata_prefix,
    )


def _generate_metadata_prefix(recipe: ProductionRecipe) -> str:
    try:
        from scripts.dataset_generation.dataset_generation.image_generation.metadata_generator import (
            generate_metadata_prefix,
        )
    except ImportError:
        return ""

    include_title = random.random() < recipe.render_only_aug.include_title_probability
    include_author = random.random() < recipe.render_only_aug.include_author_probability
    return generate_metadata_prefix(include_title=include_title, include_author=include_author)


def _resize_rendered_page(
    rendered_page: RenderedPage,
    image_width: int,
) -> tuple[np.ndarray, RenderedPage | None]:
    image = np.ascontiguousarray(resize_to_width(rendered_page.image, image_width))
    foreground = np.ascontiguousarray(resize_to_width(rendered_page.foreground, image_width))
    alpha = resize_to_width(rendered_page.alpha, image_width)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    return image, RenderedPage(
        image=image,
        foreground=foreground,
        alpha=np.ascontiguousarray(alpha.astype(np.uint8)),
    )


def _sample_render_options(recipe: ProductionRecipe) -> VerovioRenderOptions:
    image_width = recipe.image_width
    layout_profile = recipe.render_only_aug.render_layout_profile

    if layout_profile == "target_5_6_systems":
        page_width_factor = random.uniform(2.42, 2.54)
        page_width = int(round(image_width * page_width_factor))
        margin = lambda: max(0, min(500, int(round(page_width * random.uniform(0.01, 0.025)))))
        return {
            "scale": random.randint(46, 58),
            "barLineWidth": random.uniform(0.16, 0.70),
            "beamMaxSlope": random.randint(4, 16),
            "staffLineWidth": random.uniform(0.10, 0.24),
            "stemWidth": random.uniform(0.10, 0.36),
            "ledgerLineThickness": random.uniform(0.10, 0.38),
            "thickBarlineThickness": random.uniform(0.60, 1.60),
            "spacingLinear": random.uniform(0.13, 0.20),
            "spacingNonLinear": random.uniform(0.20, 0.36),
            "spacingStaff": random.randint(4, 8),
            "spacingSystem": random.randint(3, 6),
            "measureMinWidth": random.randint(4, 10),
            "pageMarginLeft": margin(),
            "pageMarginRight": margin(),
            "pageMarginTop": margin(),
            "pageMarginBottom": margin(),
            "pageWidth": page_width,
            "breaksNoWidow": False,
            "justifyVertically": False,
            "noJustification": False,
            "footer": "none",
            "breaks": "auto",
        }

    page_width_factor = random.uniform(1.80, 2.40)
    page_width = int(round(image_width * page_width_factor))
    margin = lambda: max(0, min(500, int(round(page_width * random.uniform(0.015, 0.05)))))
    return {
        "scale": random.randint(52, 84),
        "barLineWidth": random.uniform(0.16, 0.72),
        "beamMaxSlope": random.randint(4, 16),
        "staffLineWidth": random.uniform(0.10, 0.26),
        "stemWidth": random.uniform(0.10, 0.36),
        "ledgerLineThickness": random.uniform(0.10, 0.40),
        "thickBarlineThickness": random.uniform(0.60, 1.80),
        "spacingLinear": random.uniform(0.16, 0.32),
        "spacingNonLinear": random.uniform(0.24, 0.56),
        "spacingStaff": random.randint(4, 12),
        "spacingSystem": random.randint(4, 10),
        "measureMinWidth": random.randint(6, 18),
        "pageMarginLeft": margin(),
        "pageMarginRight": margin(),
        "pageMarginTop": margin(),
        "pageMarginBottom": margin(),
        "pageWidth": page_width,
        "breaksNoWidow": False,
        "justifyVertically": False,
        "noJustification": False,
        "footer": "none",
        "breaks": "auto",
    }


def _next_retry_render_options(
    options: VerovioRenderOptions,
    recipe: ProductionRecipe,
    *,
    rejection_reason: str | None,
) -> VerovioRenderOptions:
    retry = dict(options)
    retry["scale"] = max(40, int(retry.get("scale", 52)) - 8)
    retry["spacingStaff"] = max(4, int(retry.get("spacingStaff", 8)) - 1)
    retry["spacingSystem"] = max(3, int(retry.get("spacingSystem", 6)) - 1)
    retry["measureMinWidth"] = max(1, int(retry.get("measureMinWidth", 8)) - 2)
    retry["spacingLinear"] = max(0.12, float(retry.get("spacingLinear", 0.18)) - 0.03)
    retry["spacingNonLinear"] = max(0.20, float(retry.get("spacingNonLinear", 0.30)) - 0.04)
    if rejection_reason in {"left_clearance", "right_clearance"}:
        retry["pageWidth"] = int(retry.get("pageWidth", recipe.image_width * 2)) + 120
    return retry


def _tighten_layout_for_preferred_5_6_rescue(
    options: VerovioRenderOptions,
) -> VerovioRenderOptions:
    rescue = dict(options)
    rescue["scale"] = max(40, int(rescue.get("scale", 52)) - 4)
    rescue["pageWidth"] = int(rescue.get("pageWidth", 0)) + 80
    rescue["spacingStaff"] = max(4, int(rescue.get("spacingStaff", 8)) - 1)
    rescue["spacingSystem"] = max(3, int(rescue.get("spacingSystem", 6)) - 1)
    rescue["measureMinWidth"] = max(1, int(rescue.get("measureMinWidth", 8)) - 1)
    rescue["spacingLinear"] = max(0.12, float(rescue.get("spacingLinear", 0.18)) - 0.02)
    rescue["spacingNonLinear"] = max(0.20, float(rescue.get("spacingNonLinear", 0.30)) - 0.03)
    for key in ("pageMarginLeft", "pageMarginRight", "pageMarginTop", "pageMarginBottom"):
        rescue[key] = max(0, int(rescue.get(key, 0)) - 6)
    return rescue


def _next_preferred_5_6_rescue_options(
    options: VerovioRenderOptions,
    *,
    rejection_reason: str | None,
) -> VerovioRenderOptions:
    retry = dict(options)
    retry["scale"] = max(38, int(retry.get("scale", 48)) - 3)
    retry["pageWidth"] = int(retry.get("pageWidth", 0)) + 80
    retry["spacingStaff"] = max(4, int(retry.get("spacingStaff", 7)) - 1)
    retry["spacingSystem"] = max(3, int(retry.get("spacingSystem", 5)) - 1)
    retry["measureMinWidth"] = max(1, int(retry.get("measureMinWidth", 7)) - 1)
    retry["spacingLinear"] = max(0.12, float(retry.get("spacingLinear", 0.16)) - 0.02)
    retry["spacingNonLinear"] = max(0.20, float(retry.get("spacingNonLinear", 0.28)) - 0.03)
    if rejection_reason in {"left_clearance", "right_clearance"}:
        retry["pageWidth"] = int(retry.get("pageWidth", 0)) + 80
    for key in ("pageMarginLeft", "pageMarginRight", "pageMarginTop", "pageMarginBottom"):
        retry[key] = max(0, int(retry.get(key, 0)) - 4)
    return retry


def _assess_frame_fit(
    image: np.ndarray,
    *,
    target_height: int | None,
    min_frame_margin_px: int,
    target_frame_margin_px: int,
) -> tuple[bool, str | None, dict[str, int | float | None]]:
    current_height = int(image.shape[0])
    current_width = int(image.shape[1])
    bbox = _find_content_bbox(image)

    metrics: dict[str, int | float | None] = {
        "current_height": current_height,
        "current_width": current_width,
        "target_height": target_height,
        "top_row": None,
        "bottom_row": None,
        "left_col": None,
        "right_col": None,
        "top_whitespace_px": None,
        "bottom_whitespace_px": None,
        "content_height_px": None,
        "bottom_whitespace_ratio": None,
        "vertical_fill_ratio": None,
    }
    if bbox is None:
        return False, "no_content_detected", metrics

    top_row, bottom_row, left_col, right_col = bbox
    top_clearance = top_row
    bottom_clearance = current_height - 1 - bottom_row
    left_clearance = left_col
    right_clearance = current_width - 1 - right_col
    frame_height = int(target_height) if target_height is not None else current_height
    final_bottom_whitespace = frame_height - 1 - bottom_row
    content_height = bottom_row - top_row + 1

    metrics.update(
        {
            "top_row": top_row,
            "bottom_row": bottom_row,
            "left_col": left_col,
            "right_col": right_col,
            "top_whitespace_px": int(top_row),
            "bottom_whitespace_px": int(max(0, final_bottom_whitespace)),
            "content_height_px": int(content_height),
            "bottom_whitespace_ratio": (
                max(0.0, float(final_bottom_whitespace) / frame_height) if frame_height > 0 else None
            ),
            "vertical_fill_ratio": (
                float(content_height) / frame_height if frame_height > 0 else None
            ),
        }
    )

    if top_clearance < min_frame_margin_px:
        return False, "top_clearance", metrics
    if bottom_clearance < min_frame_margin_px:
        return False, "bottom_clearance", metrics
    if left_clearance < min_frame_margin_px:
        return False, "left_clearance", metrics
    if right_clearance < min_frame_margin_px:
        return False, "right_clearance", metrics

    if target_height is not None and current_height > target_height:
        cropped_clearance = target_height - 1 - bottom_row
        if cropped_clearance < min_frame_margin_px:
            return False, "crop_risk", metrics

    if min(top_clearance, bottom_clearance, left_clearance, right_clearance) < target_frame_margin_px:
        return True, None, metrics
    return True, None, metrics


def _find_content_bbox(image: np.ndarray, *, threshold: int = 100) -> tuple[int, int, int, int] | None:
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


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


@contextmanager
def _seeded_random(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)
