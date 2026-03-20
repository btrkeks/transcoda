#!/usr/bin/env python3
"""Generate a 2-up preview for strongest horizontal squeeze augmentation."""

from __future__ import annotations

import random
import time
from pathlib import Path

import fire
import numpy as np
from PIL import Image

from scripts.dataset_generation.data_spec import DEFAULT_DATA_SPEC_PATH, resolve_image_size_from_spec
from scripts.dataset_generation.dataset_generation.image_augmentation.geometric_augment import (
    geometric_augment,
)
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
)
from scripts.dataset_generation.dataset_generation.image_generation.score_generator import (
    generate_with_diagnostics as generate_score_with_diagnostics,
)
from scripts.dataset_generation.dataset_generation.image_generation.types import GenerationConfig
from scripts.dataset_generation.dataset_generation.worker import ensure_kern_header, is_valid_kern


def _save_png(image: np.ndarray, out_path: Path) -> None:
    Image.fromarray(np.ascontiguousarray(image[:, :, :3]).astype(np.uint8)).save(out_path)


def _make_two_up(left: np.ndarray, right: np.ndarray, *, gutter_px: int = 24) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError(f"Expected equal shapes for 2-up image, got {left.shape} vs {right.shape}")
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("2-up images must be uint8")

    h, _, c = left.shape
    if c != 3:
        raise ValueError(f"Expected RGB arrays with 3 channels, got {c}")
    gutter = np.full((h, gutter_px, c), 255, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate([left, gutter, right], axis=1))


def _pick_default_kern_file(normalized_dir: Path) -> Path:
    files = sorted(normalized_dir.glob("*.krn"))
    if not files:
        raise ValueError(f"No .krn files found in {normalized_dir}")
    return files[0]


def main(
    normalized_dir: str = "data/interim/val/polish-scores/3_normalized",
    kern_file: str | None = None,
    output_dir: str | None = None,
    squeeze_scale: float = 0.70,
    seed: int = 7,
    image_width: int | None = None,
    image_height: int | None = None,
    data_spec_path: str = str(DEFAULT_DATA_SPEC_PATH),
    strict_data_spec: bool = True,
) -> None:
    """Render one synthetic sample and save baseline vs strongest x-squeeze.

    The squeezed preview uses geometric_augment with a forced squeeze scale and
    no additional geometric effects.
    """
    if not 0.0 < squeeze_scale <= 1.0:
        raise ValueError(f"squeeze_scale must be in (0.0, 1.0], got {squeeze_scale}")

    image_width, image_height = resolve_image_size_from_spec(
        image_width=image_width,
        image_height=image_height,
        data_spec_path=data_spec_path,
        strict_data_spec=strict_data_spec,
    )

    norm_dir = Path(normalized_dir)
    kern_path = Path(kern_file) if kern_file else _pick_default_kern_file(norm_dir)
    if not kern_path.exists():
        raise FileNotFoundError(f"kern_file not found: {kern_path}")

    kern_content = kern_path.read_text(encoding="utf-8")
    is_valid, error_msg = is_valid_kern(kern_content)
    if not is_valid:
        raise ValueError(f"Invalid kern source ({kern_path.name}): {error_msg}")

    random.seed(seed)
    np.random.seed(seed)

    renderer = VerovioRenderer()
    config = GenerationConfig(
        num_systems_hint=1,
        include_author=False,
        include_title=False,
        image_width=image_width,
        image_height=image_height,
        texturize_image=False,
    )
    result, rejection_reason = generate_score_with_diagnostics(
        ensure_kern_header(kern_content),
        renderer,
        config,
    )
    if result is None:
        raise RuntimeError(
            f"Failed to render preview for {kern_path.name}; rejection_reason={rejection_reason}"
        )

    baseline = np.ascontiguousarray(result.image[:, :, :3])
    forced_rng = np.random.default_rng(seed)
    strongest = geometric_augment(
        baseline,
        forced_rng,
        conservative=False,
        x_squeeze_prob=1.0,
        x_squeeze_min_scale=squeeze_scale,
        x_squeeze_max_scale=squeeze_scale,
        x_squeeze_apply_in_conservative=True,
        x_squeeze_force_scale=squeeze_scale,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) if output_dir else Path(f"/tmp/x_squeeze_preview_{ts}")
    out_path.mkdir(parents=True, exist_ok=True)

    baseline_path = out_path / "baseline.png"
    squeeze_token = f"{squeeze_scale:.2f}".replace(".", "p")
    squeeze_name = f"squeezed_{squeeze_token}x.png"
    squeezed_path = out_path / squeeze_name
    comparison_path = out_path / "comparison_2up.png"

    _save_png(baseline, baseline_path)
    _save_png(strongest, squeezed_path)
    _save_png(_make_two_up(baseline, strongest), comparison_path)

    print(f"Source kern: {kern_path}")
    print(f"Baseline: {baseline_path}")
    print(f"Squeezed: {squeezed_path}")
    print(f"Comparison: {comparison_path}")


if __name__ == "__main__":
    fire.Fire(main)
