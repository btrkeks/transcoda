"""Image normalization and synthetic-page raster helpers for dataset materialization."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from pipelines.datasets.materialization.augraphy import augraphy_augment
from pipelines.datasets.materialization.rendering.types import RenderedPage

_MAX_SIGNED_C_INT = 2_147_483_647
_SAFE_SVG_COLOR_VALUES = frozenset({"none", "currentcolor", "inherit", "#ffffff", "white"})
_SVG_INK_ATTR_RE = re.compile(r'\b(fill|stroke|color)="([^"]*)"')
_PAPER_TEXTURES: list[str] | None = None
_TEXTURE_RGB_CACHE: dict[str, Image.Image] = {}
_TEXTURE_TILED_CACHE: dict[tuple[str, int, int], Image.Image] = {}
SYNTHETIC_BACKGROUND_MODES = frozenset({"current", "simple", "white"})


def stable_seed(value: str) -> int:
    return (
        int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest(), "big")
        % _MAX_SIGNED_C_INT
    )


def normalize_pil_image(image: Image.Image, *, width: int, height: int) -> Image.Image:
    """Resize to target width, then bottom-pad or top-crop to target height."""
    rgb = image.convert("RGB")
    scaled_height = max(1, round(rgb.height * (width / rgb.width)))
    resized = rgb.resize((width, scaled_height), Image.LANCZOS)

    if scaled_height == height:
        return resized
    if scaled_height > height:
        return resized.crop((0, scaled_height - height, width, scaled_height))

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    canvas.paste(resized, (0, 0))
    return canvas


def normalize_rendered_page(page: RenderedPage, *, width: int, height: int) -> RenderedPage:
    image = normalize_pil_image(Image.fromarray(page.image), width=width, height=height)
    foreground = normalize_pil_image(Image.fromarray(page.foreground), width=width, height=height)
    alpha = _normalize_alpha(page.alpha, width=width, height=height)
    return RenderedPage(
        image=np.ascontiguousarray(np.array(image, dtype=np.uint8)),
        foreground=np.ascontiguousarray(np.array(foreground, dtype=np.uint8)),
        alpha=alpha,
    )


def choose_synthetic_ink_color(seed: int) -> str:
    """Choose a deterministic page-wide synthetic ink color."""
    rng = np.random.default_rng(_bounded_seed(seed))
    if rng.random() < 0.5:
        return "#000000"

    value = int(rng.integers(0, 61))
    return f"#{value:02x}{value:02x}{value:02x}"


def normalize_svg_ink(svg_text: str, ink_color: str = "#000000") -> str:
    """Replace explicit notation colors in SVG attributes with one ink color."""

    def _replace(match: re.Match[str]) -> str:
        attr = match.group(1)
        value = match.group(2).strip().lower()
        if value in _SAFE_SVG_COLOR_VALUES:
            return match.group(0)
        return f'{attr}="{ink_color}"'

    return _SVG_INK_ATTR_RE.sub(_replace, svg_text)


def rasterize_synthetic_svg(
    svg_text: str,
    *,
    width: int,
    height: int,
    seed: int,
    apply_augraphy: bool = True,
    synthetic_background: str = "current",
) -> tuple[Image.Image, str]:
    """Rasterize SVG, composite onto paper, then apply Augraphy in memory."""
    _validate_synthetic_background(synthetic_background)
    svg_text = normalize_svg_ink(svg_text, ink_color=choose_synthetic_ink_color(seed))
    rendered = normalize_rendered_page(_svg_to_rendered_page(svg_text), width=width, height=height)
    if synthetic_background == "white":
        paper_page = rendered.image
    else:
        paper_page = _compose_rendered_page_on_background(
            rendered,
            width=width,
            height=height,
            seed=seed,
            synthetic_background=synthetic_background,
        )
    if not apply_augraphy:
        normalized = normalize_pil_image(Image.fromarray(paper_page), width=width, height=height)
        return normalized, "disabled"

    augmented, outcome = augraphy_augment(paper_page, seed=_bounded_seed(seed + 20))
    normalized = normalize_pil_image(Image.fromarray(augmented), width=width, height=height)
    return normalized, outcome


def _validate_synthetic_background(synthetic_background: str) -> None:
    if synthetic_background not in SYNTHETIC_BACKGROUND_MODES:
        choices = ", ".join(sorted(SYNTHETIC_BACKGROUND_MODES))
        raise ValueError(
            f"synthetic_background must be one of: {choices}; got {synthetic_background!r}"
        )


def _compose_rendered_page_on_background(
    rendered: RenderedPage,
    *,
    width: int,
    height: int,
    seed: int,
    synthetic_background: str,
) -> np.ndarray:
    rng = np.random.default_rng(_bounded_seed(seed + 10))
    if synthetic_background == "current":
        textures = _load_paper_textures()
        background = _synthesize_background(
            width,
            height,
            rng=rng,
            textures=textures,
            allow_textures=bool(textures),
        )
    elif synthetic_background == "simple":
        background = _synthesize_simple_background(width, height, rng=rng)
    else:
        raise AssertionError(f"unhandled synthetic background mode: {synthetic_background}")
    return _alpha_composite(background, rendered.foreground, rendered.alpha)


def _synthesize_simple_background(
    width: int,
    height: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    base = np.float32(rng.integers(240, 252))
    xs, ys, distance = _background_coordinate_grids(width, height)
    gx = np.float32(rng.uniform(-10.0, 10.0))
    gy = np.float32(rng.uniform(-10.0, 10.0))
    radial = np.float32(rng.uniform(-6.0, 6.0))
    plane = np.full((height, width), base, dtype=np.float32)
    plane += (xs - 0.5) * gx + (ys - 0.5) * gy + distance * radial

    contrast = np.float32(rng.uniform(0.985, 1.015))
    brightness = np.float32(rng.uniform(-6.0, 6.0))
    plane = (plane - 128.0) * contrast + 128.0 + brightness

    low_res_noise = rng.standard_normal(size=(128, 128), dtype=np.float32)
    noise_image = Image.fromarray(low_res_noise).resize((width, height), Image.BICUBIC)
    noise = np.asarray(noise_image, dtype=np.float32)
    plane += noise * np.float32(rng.uniform(0.3, 1.2))
    plane = np.clip(plane, 225.0, 255.0)

    color_cast = np.array(
        [
            rng.uniform(-4.0, 4.0),
            rng.uniform(-3.0, 3.0),
            rng.uniform(-6.0, 6.0),
        ],
        dtype=np.float32,
    )
    background = plane[:, :, None] + color_cast[None, None, :]
    return np.ascontiguousarray(np.clip(background, 225.0, 255.0).astype(np.uint8))


def _normalize_alpha(alpha: np.ndarray, *, width: int, height: int) -> np.ndarray:
    alpha_image = Image.fromarray(alpha).convert("L")
    scaled_height = max(1, round(alpha_image.height * (width / alpha_image.width)))
    resized = alpha_image.resize((width, scaled_height), Image.LANCZOS)

    if scaled_height == height:
        result = resized
    elif scaled_height > height:
        result = resized.crop((0, scaled_height - height, width, scaled_height))
    else:
        canvas = Image.new("L", (width, height), color=0)
        canvas.paste(resized, (0, 0))
        result = canvas
    return np.ascontiguousarray(np.array(result, dtype=np.uint8))


def _bounded_seed(seed: int) -> int:
    return int(seed) % _MAX_SIGNED_C_INT


def _svg_to_rendered_page(svg_text: str) -> RenderedPage:
    from pipelines.datasets.materialization.rendering.verovio_backend import (
        svg_to_rendered_page,
    )

    return svg_to_rendered_page(svg_text)


def _load_paper_textures() -> list[str]:
    global _PAPER_TEXTURES
    if _PAPER_TEXTURES is not None:
        return _PAPER_TEXTURES

    textures_dir = (
        Path(__file__).resolve().parent
        / "assets"
        / "paper_textures"
    )
    if not textures_dir.is_dir():
        _PAPER_TEXTURES = []
        return _PAPER_TEXTURES

    _PAPER_TEXTURES = [str(path) for path in sorted(textures_dir.iterdir()) if path.is_file()]
    return _PAPER_TEXTURES


def _synthesize_background(
    width: int,
    height: int,
    *,
    rng: np.random.Generator,
    textures: list[str] | None = None,
    allow_textures: bool = True,
) -> np.ndarray:
    clean_roll = float(rng.random())
    if (not allow_textures) or clean_roll < 0.10:
        base = np.full((height, width, 3), rng.integers(240, 252), dtype=np.uint8)
    else:
        base = _sample_texture_canvas(width, height, textures=textures, rng=rng)

    background = base.astype(np.float32)
    xs, ys, distance = _background_coordinate_grids(width, height)
    gx = float(rng.uniform(-10.0, 10.0))
    gy = float(rng.uniform(-10.0, 10.0))
    radial = float(rng.uniform(-6.0, 6.0))
    gradient = (xs - 0.5) * gx + (ys - 0.5) * gy + distance * radial
    background += gradient[:, :, None]

    contrast = float(rng.uniform(0.985, 1.015))
    brightness = float(rng.uniform(-6.0, 6.0))
    background = (background - 128.0) * contrast + 128.0 + brightness
    color_cast = np.array(
        [
            rng.uniform(-4.0, 4.0),
            rng.uniform(-3.0, 3.0),
            rng.uniform(-6.0, 6.0),
        ],
        dtype=np.float32,
    )
    background += color_cast[None, None, :]
    noise_scale = np.float32(rng.uniform(0.3, 1.2))
    background += rng.standard_normal(size=background.shape, dtype=np.float32) * noise_scale
    return np.ascontiguousarray(np.clip(background, 225.0, 255.0).astype(np.uint8))


@lru_cache(maxsize=8)
def _background_coordinate_grids(
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    distance = ((xs - 0.5) ** 2 + (ys - 0.5) ** 2) / 0.5
    return xs, ys, distance


def _alpha_composite(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    alpha_f = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)
    alpha_rgb = alpha_f[:, :, None]
    composed = foreground.astype(np.float32) * alpha_rgb + background.astype(np.float32) * (
        1.0 - alpha_rgb
    )
    return np.ascontiguousarray(np.clip(composed, 0.0, 255.0).astype(np.uint8))


def _sample_texture_canvas(
    width: int,
    height: int,
    *,
    textures: list[str] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    available = _load_paper_textures() if textures is None else textures
    if not available:
        return np.full((height, width, 3), 245, dtype=np.uint8)

    texture_path = str(available[int(rng.integers(0, len(available)))])
    texture = _get_cached_texture_rgb(texture_path, min_width=width, min_height=height)
    max_left = max(texture.width - width, 0)
    max_top = max(texture.height - height, 0)
    left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
    top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
    crop = texture.crop((left, top, left + width, top + height))
    return np.ascontiguousarray(np.array(crop, dtype=np.uint8))


def _get_cached_texture_rgb(
    texture_path: str,
    *,
    min_width: int,
    min_height: int,
) -> Image.Image:
    texture_key = str(Path(texture_path))
    texture = _TEXTURE_RGB_CACHE.get(texture_key)
    if texture is None:
        with Image.open(texture_key) as texture_image:
            texture = texture_image.convert("RGB")
        _TEXTURE_RGB_CACHE[texture_key] = texture

    if texture.width >= min_width and texture.height >= min_height:
        return texture

    repeat_x = (min_width + texture.width - 1) // texture.width
    repeat_y = (min_height + texture.height - 1) // texture.height
    tiled_key = (texture_key, repeat_x, repeat_y)
    tiled = _TEXTURE_TILED_CACHE.get(tiled_key)
    if tiled is None:
        tiled = _tile_texture(texture, repeat_x=repeat_x, repeat_y=repeat_y)
        _TEXTURE_TILED_CACHE[tiled_key] = tiled
    return tiled


def _tile_texture(texture: Image.Image, *, repeat_x: int, repeat_y: int) -> Image.Image:
    tiled = Image.new("RGB", (texture.width * repeat_x, texture.height * repeat_y))
    for y in range(repeat_y):
        for x in range(repeat_x):
            tiled.paste(texture, (x * texture.width, y * texture.height))
    return tiled
