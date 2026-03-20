"""SVG-level augmentations for synthetic music rendering.

This module provides augmentation functions that operate on SVG strings
before they are converted to raster images. These augmentations include
color shifts to increase training data diversity.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "augment_svg",
    "normalize_svg_colors",
    "random_color_shift",
    "change_color_svg",
]

_SAFE_COLOR_VALUES = frozenset({"none", "currentcolor", "inherit", "#ffffff", "white"})
_COLOR_ATTR_RE = re.compile(r'\b(fill|stroke|color)="([^"]*)"')


def normalize_svg_colors(svg: str) -> str:
    """Replace all explicit color values in SVG attributes with black.

    Targets ``fill``, ``stroke``, and ``color`` attributes, replacing their
    values with ``#000000`` unless the value is a safe keyword (``none``,
    ``currentColor``, ``inherit``, ``white``, ``#ffffff``).

    This strips Verovio's red error-highlighting before the color-shift
    augmentation runs, so the model never sees non-black notation.
    """

    def _replace(match: re.Match) -> str:
        attr = match.group(1)
        value = match.group(2).strip().lower()
        if value in _SAFE_COLOR_VALUES:
            return match.group(0)
        return f'{attr}="#000000"'

    return _COLOR_ATTR_RE.sub(_replace, svg)


def augment_svg(svg: str) -> str:
    """Apply random augmentations to an SVG string.

    Args:
        svg: The input SVG string to augment

    Returns:
        The augmented SVG string
    """
    svg = random_color_shift(svg)
    return svg


def random_color_shift(svg: str) -> str:
    """Apply a random color shift to darken the SVG elements.

    The color is randomly selected from a grayscale range to simulate
    variations in ink darkness or print quality.

    Args:
        svg: The input SVG string

    Returns:
        The color-shifted SVG string
    """
    COLOR_RANGE = (0, 60)
    color = np.random.randint(COLOR_RANGE[0], COLOR_RANGE[1] + 1)
    color_hex = f"#{color:02x}{color:02x}{color:02x}"
    svg = change_color_svg(svg, color_hex)
    return svg


def change_color_svg(svg: str, color: str = "#2a2a2a") -> str:
    """Change the color of SVG elements.

    Modifies both the color attribute and the style rules to apply
    the specified color to paths, ellipses, and polygons.

    Args:
        svg: The input SVG string
        color: Hex color string (e.g., "#2a2a2a")

    Returns:
        The color-modified SVG string
    """
    # Update the color attribute
    svg = re.sub(r'(<svg[^>]*class="definition-scale"[^>]*color=")[^"]*(")', rf"\1{color}\2", svg)

    # Update the style rule to also set fill
    svg = svg.replace(
        "path {stroke:currentColor}",
        "path, ellipse, polygon {stroke:currentColor; fill:currentColor}",
    )
    return svg


