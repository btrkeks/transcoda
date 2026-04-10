from __future__ import annotations

import logging
import multiprocessing as mp
import re

import numpy as np
from typing_extensions import TypedDict

from ..exceptions import LayoutOverflowError
from ..types import RenderedPage
from .svg_augmentation import augment_svg, normalize_svg_colors

logger = logging.getLogger(__name__)

_SYSTEM_CLASS_PATTERN = re.compile(r'<g\b[^>]*\bclass="[^"]*\bsystem\b[^"]*"')

VEROVIO_FONTS: list[str] = ["Leipzig", "Bravura", "Gootville", "Leland", "Petaluma"]

__all__ = [
    "VEROVIO_FONTS",
    "VerovioRenderer",
    "VerovioRenderOptions",
    "count_nr_of_systems_in_svg",
    "svg_to_rendered_page",
    "svg_to_rgb",
]


class VerovioRenderOptions(TypedDict, total=False):
    """Type for Verovio rendering options."""

    scale: int
    barLineWidth: float
    beamMaxSlope: int
    staffLineWidth: float
    spacingStaff: int
    spacingSystem: int
    spacingLinear: float
    spacingNonLinear: float
    pageMarginBottom: int
    pageMarginLeft: int
    pageMarginRight: int
    pageMarginTop: int
    measureMinWidth: int
    stemWidth: float
    ledgerLineThickness: float
    thickBarlineThickness: float
    breaksNoWidow: bool
    justifyVertically: bool
    noJustification: bool
    adjustPageHeight: bool
    adjustPageWidth: bool
    shrinkToFit: bool
    pageWidth: int
    font: str
    footer: str
    breaks: str


class VerovioRenderer:
    """Encapsulates Verovio toolkit for rendering **kern sequences to images."""

    def __init__(self) -> None:
        """Initialize the Verovio toolkit."""
        import verovio

        verovio.enableLog(verovio.LOG_ERROR)
        self.tk = verovio.toolkit()

    def render_to_svg(
        self,
        music_sequence: str,
        render_options: VerovioRenderOptions,
    ) -> tuple[str, int]:
        """Render a **kern sequence to SVG images using Verovio.

        Args:
            music_sequence: Humdrum kern data to render
            render_options: Verovio rendering options
            validate: Whether to validate the input (currently unused)

        Returns:
            Tuple of (SVG string, page count)
        """
        self.tk.loadData(music_sequence)
        self.tk.setOptions(render_options)
        page_count = self.tk.getPageCount()
        svg = self.tk.renderToSVG()
        return svg, page_count

    def render_to_rgb(
        self,
        transcription: str,
        render_options: VerovioRenderOptions,
        max_systems: int | None = None,
    ) -> np.ndarray:
        """Render a transcription to RGB image with validation.

        Args:
            transcription: The kern transcription to render
            metadata_prefix: Metadata prefix (title/author) to prepend
            render_options: Verovio rendering options
            max_systems: Maximum allowed systems (None to skip check)

        Returns:
            raw RGBA as a NumPy array.

        Raises:
            LayoutOverflowError: If rendering produces multiple pages or too many systems
        """
        # Render to SVG
        svg, page_count = self.render_to_svg(
            transcription,
            render_options=render_options,
        )

        svg = normalize_svg_colors(svg)
        svg = augment_svg(svg)

        # Enforce page limit
        if page_count != 1:
            logger.error("Layout overflow: produced %d pages (allowed=1)", page_count)
            raise LayoutOverflowError(
                "Rendered output spans multiple pages.",
                page_count=page_count,
                allowed_pages=1,
            )

        # Enforce systems limit if specified
        if max_systems is not None:
            system_count = count_nr_of_systems_in_svg(svg)
            if system_count > max_systems:
                logger.error("Layout overflow: %d systems (max=%d)", system_count, max_systems)
                raise LayoutOverflowError(
                    "Rendered SVG contains more systems than allowed.",
                    system_count=system_count,
                    max_systems=max_systems,
                )

        npy_image = svg_to_rgb(svg)
        return npy_image

    def render_to_rgb_isolated(self, transcription: str, render_options: VerovioRenderOptions):
        """Render a transcription to an RGB image isolated in a separate process."""
        raise NotImplementedError("Isolated rendering not implemented yet.")

    def render_with_counts(
        self,
        transcription: str,
        render_options: VerovioRenderOptions,
    ) -> tuple[RenderedPage, int, int]:
        """Render transcription and return image with layout counts.

        This is the primary rendering method for single-pass generation.
        Unlike render_to_rgb, it does not validate or raise on overflow.

        Args:
            transcription: The **kern transcription to render.
            render_options: Verovio rendering options.

        Returns:
            Tuple of (image, system_count, page_count) where:
            - image: RenderedPage payload with clean page and notation layers
            - system_count: Number of systems in the rendered output
            - page_count: Number of pages (typically 1)
        """
        svg, page_count = self.render_to_svg(transcription, render_options)
        svg = normalize_svg_colors(svg)
        svg = augment_svg(svg)
        system_count = count_nr_of_systems_in_svg(svg)
        rendered_page = svg_to_rendered_page(svg)
        return rendered_page, system_count, page_count

    def count_systems_and_pages(
        self,
        transcription: str,
        metadata_prefix: str,
        render_options: VerovioRenderOptions,
    ) -> tuple[int, int]:
        """Count the number of systems and pages in a transcription.
        Note: Transcription needs to be renderable! (With **kern header etc)

        Args:
            transcription: The kern transcription
            metadata_prefix: Metadata prefix to prepend
            render_options: Verovio rendering options

        Returns:
            Tuple of (system_count, page_count)
        """
        svg, page_count = self.render_to_svg(
            transcription,
            render_options=render_options,
        )
        system_count = count_nr_of_systems_in_svg(svg)
        return system_count, page_count


def count_nr_of_systems_in_svg(svg: str) -> int:
    """Counts the number of systems in the rendered SVG."""
    modern_count = len(_SYSTEM_CLASS_PATTERN.findall(svg))
    if modern_count > 0:
        return modern_count
    return svg.count('class="grpSym"')


def svg_to_rgb(svg: str) -> np.ndarray:
    """Convert a SVG string into raw RGB as a NumPy array."""
    return svg_to_rendered_page(svg).image


def svg_to_rendered_page(svg: str) -> RenderedPage:
    """Convert a SVG string into clean page and separated notation layers."""
    import pyvips

    img = pyvips.Image.new_from_buffer(svg.encode("utf-8"), "scale=1")
    if img.bands < 4:
        img = img.bandjoin(255)

    rgba_raw = img.write_to_memory()
    rgba = np.frombuffer(rgba_raw, dtype=np.uint8).reshape((img.height, img.width, img.bands))
    alpha = np.ascontiguousarray(rgba[:, :, 3].copy())

    rgb_img = img.flatten(background=[255, 255, 255])
    raw = rgb_img.write_to_memory()
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape((rgb_img.height, rgb_img.width, rgb_img.bands))
    clean_page = np.ascontiguousarray(rgb.copy())

    foreground = np.full_like(clean_page, 255)
    if np.any(alpha):
        alpha_rgb = (alpha.astype(np.float32) / 255.0)[:, :, None]
        clean_f = clean_page.astype(np.float32)
        recovered = 255.0 - ((255.0 - clean_f) / np.maximum(alpha_rgb, 1e-6))
        foreground = np.where(alpha_rgb > 0.0, np.clip(recovered, 0.0, 255.0), 255.0).astype(
            np.uint8
        )

    return RenderedPage(
        image=clean_page,
        foreground=np.ascontiguousarray(foreground),
        alpha=alpha,
    )


def _render_to_rgb_worker(
    transcription: str,
    render_options: VerovioRenderOptions,
    max_systems: int | None,
    result_queue: mp.Queue,
) -> None:
    """Worker function run in a separate process.

    It must be at module top level so it is picklable for the 'spawn' start method.
    """
    import traceback

    try:
        renderer = VerovioRenderer()
        image = renderer.render_to_rgb(
            transcription=transcription,
            render_options=render_options,
            max_systems=max_systems,
        )
        result_queue.put(("ok", image))
    except Exception as exc:  # noqa: BLE001 - we want to catch everything here
        result_queue.put(
            (
                "error",
                (
                    exc.__class__.__name__,
                    str(exc),
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                ),
            )
        )
