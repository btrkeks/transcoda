from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

__all__ = [
    "resize_to_width",
    "pad_or_crop_to_height",
    "pad_or_crop_alpha_to_height",
    "trim_bottom_white",
    "find_content_bottom_row",
    "apply_random_texture",
    "inkify_image",
    "load_paper_textures",
    "sample_texture_canvas",
    "synthesize_background",
    "alpha_composite",
]

_TEXTURE_RGB_CACHE: dict[str, Image.Image] = {}
_TEXTURE_TILED_CACHE: dict[tuple[str, int, int], Image.Image] = {}
_PAPER_TEXTURES: list[str] | None = None


def pad_or_crop_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Pad with white or crop image to exact target height.

    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        target_height: Desired output height in pixels

    Returns:
        Image with height equal to target_height
    """
    current_height = image.shape[0]
    if current_height == target_height:
        return image

    if current_height > target_height:
        return image[:target_height]

    # Pad with white (255) at the bottom
    pad_height = target_height - current_height
    if image.ndim == 3:
        padding = np.full((pad_height, image.shape[1], image.shape[2]), 255, dtype=image.dtype)
    else:
        padding = np.full((pad_height, image.shape[1]), 255, dtype=image.dtype)
    return np.concatenate([image, padding], axis=0)


def pad_or_crop_alpha_to_height(alpha: np.ndarray, target_height: int) -> np.ndarray:
    """Pad alpha with transparency or crop to exact target height."""
    current_height = alpha.shape[0]
    if current_height == target_height:
        return alpha

    if current_height > target_height:
        return alpha[:target_height]

    pad_height = target_height - current_height
    if alpha.ndim == 3:
        padding = np.zeros((pad_height, alpha.shape[1], alpha.shape[2]), dtype=alpha.dtype)
    else:
        padding = np.zeros((pad_height, alpha.shape[1]), dtype=alpha.dtype)
    return np.concatenate([alpha, padding], axis=0)


def resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize image to target width, preserving aspect ratio.

    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        target_width: Desired output width in pixels

    Returns:
        Resized image with width equal to target_width

    Raises:
        ValueError: If target_width is not positive
    """
    if target_width <= 0:
        raise ValueError("target_width must be positive")

    current_width = image.shape[1]
    if current_width == target_width:
        return image

    scale = target_width / current_width
    new_height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (target_width, new_height))


def find_content_bottom_row(sample: np.ndarray, threshold: int = 100) -> int | None:
    """
    Return the last row index (from top) that contains a black pixel.
    If none found, return None.

    threshold: treat values <= threshold as black (helps with JPEG noise).
    """
    if sample.ndim == 3:  # H x W x C
        # True where pixel is black across all channels
        black = np.all(sample <= threshold, axis=-1)
    else:  # grayscale H x W
        black = sample <= threshold

    # Rows that have at least one black pixel
    rows_with_black = np.any(black, axis=1)
    if not rows_with_black.any():
        return None

    # Index of the last such row (i.e., closest to bottom)
    return int(np.flatnonzero(rows_with_black)[-1])


def trim_bottom_white(img):
    if (h := find_content_bottom_row(img)) is not None:
        img = img[: h + 10]
    return img


def inkify_image(sample: np.ndarray) -> Image.Image:
    """Apply optional ink-style post-processing (currently a no-op placeholder)."""

    return Image.fromarray(sample.astype(np.uint8))


def apply_random_texture(image: np.ndarray, textures: list[str]) -> np.ndarray:
    """Overlay the rendered image on a randomly cropped paper texture.

    Args:
        image: Input image as numpy array
        textures: List of texture file paths to choose from

    Returns:
        Image with texture applied as numpy array
    """

    if not textures:
        return image

    music_image = inkify_image(image)
    img_width, img_height = music_image.size

    texture_path = random.choice(textures)
    texture = _get_cached_texture_rgb(texture_path, min_width=img_width, min_height=img_height)

    max_left = max(texture.width - img_width, 0)
    max_top = max(texture.height - img_height, 0)
    left = random.randint(0, max_left)
    top = random.randint(0, max_top)
    texture_crop = texture.crop((left, top, left + img_width, top + img_height))

    inverted_music = ImageOps.invert(music_image.convert("RGB"))
    mask = inverted_music.convert("L")
    texture_crop.paste(music_image, mask=mask)
    return np.array(texture_crop)


def load_paper_textures() -> list[str]:
    """Load paper texture paths from the assets directory once per process."""
    global _PAPER_TEXTURES
    if _PAPER_TEXTURES is not None:
        return _PAPER_TEXTURES

    textures_dir = Path(__file__).parent / "assets" / "paper_textures"
    if not textures_dir.is_dir():
        _PAPER_TEXTURES = []
        return _PAPER_TEXTURES

    _PAPER_TEXTURES = [
        str(path)
        for path in sorted(textures_dir.iterdir())
        if path.is_file()
    ]
    return _PAPER_TEXTURES


def sample_texture_canvas(
    width: int,
    height: int,
    *,
    textures: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a full-page texture crop sized to the requested canvas."""
    if rng is None:
        rng = np.random.default_rng()
    available = load_paper_textures() if textures is None else textures
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


def synthesize_background(
    width: int,
    height: int,
    *,
    rng: np.random.Generator,
    textures: list[str] | None = None,
    allow_textures: bool = True,
) -> np.ndarray:
    """Create a coherent paper-like background with lightweight variation."""
    clean_roll = float(rng.random())
    if (not allow_textures) or clean_roll < 0.10:
        base = np.full((height, width, 3), rng.integers(240, 252), dtype=np.uint8)
    else:
        base = sample_texture_canvas(width, height, textures=textures, rng=rng)

    background = base.astype(np.float32)

    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    gx = float(rng.uniform(-10.0, 10.0))
    gy = float(rng.uniform(-10.0, 10.0))
    radial = float(rng.uniform(-6.0, 6.0))
    distance = ((xs - 0.5) ** 2 + (ys - 0.5) ** 2) / 0.5
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

    noise = rng.normal(loc=0.0, scale=rng.uniform(0.3, 1.2), size=background.shape).astype(
        np.float32
    )
    background += noise
    return np.ascontiguousarray(np.clip(background, 225.0, 255.0).astype(np.uint8))


def alpha_composite(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Composite foreground onto background using an antialiased alpha mask."""
    alpha_f = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)
    alpha_rgb = alpha_f[:, :, None]
    composed = foreground.astype(np.float32) * alpha_rgb + background.astype(np.float32) * (
        1.0 - alpha_rgb
    )
    return np.ascontiguousarray(np.clip(composed, 0.0, 255.0).astype(np.uint8))


def _get_cached_texture_rgb(
    texture_path: str,
    *,
    min_width: int,
    min_height: int,
) -> Image.Image:
    """Load/resize texture once per process and reuse for subsequent samples."""
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
    """Build a larger texture by tiling instead of interpolation resize."""
    tiled = Image.new("RGB", (texture.width * repeat_x, texture.height * repeat_y))
    for y in range(repeat_y):
        for x in range(repeat_x):
            tiled.paste(texture, (x * texture.width, y * texture.height))
    return tiled
