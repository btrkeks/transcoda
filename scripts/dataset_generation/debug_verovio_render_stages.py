#!/usr/bin/env python3
"""Debug Verovio rendering by saving SVGs and PNGs at each pipeline stage.

This script helps isolate whether broken synthetic renders originate from:
1) Verovio SVG generation (raw.svg),
2) SVG color normalization (normalized.svg),
3) SVG augmentation (augmented.svg), or
4) SVG->RGB rasterization (raw/normalized/augmented PNGs).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import numpy as np
from PIL import Image

from scripts.dataset_generation.dataset_generation.image_generation.metadata_generator import (
    generate_metadata_prefix,
)
from scripts.dataset_generation.dataset_generation.image_generation.rendering.svg_augmentation import (
    augment_svg,
    normalize_svg_colors,
)
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
    count_nr_of_systems_in_svg,
    svg_to_rgb,
)
from scripts.dataset_generation.dataset_generation.image_generation.score_generator import (
    _sample_render_options,
)


def ensure_kern_header(content: str) -> str:
    """Prepend **kern header if the content lacks an exclusive interpretation line."""
    first_line = content.split("\n", 1)[0]
    if first_line.startswith("**"):
        return content
    num_spines = first_line.count("\t") + 1
    header = "\t".join(["**kern"] * num_spines)
    return header + "\n" + content


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    return value


def _svg_element_counts(svg: str) -> dict[str, int]:
    """Cheap SVG shape counts to compare stage-to-stage changes."""
    return {
        "path": svg.count("<path"),
        "use": svg.count("<use"),
        "line": svg.count("<line"),
        "polygon": svg.count("<polygon"),
        "ellipse": svg.count("<ellipse"),
        "rect": svg.count("<rect"),
        "grpSym_count": svg.count('class="grpSym"'),
    }


def _write_png_from_svg(svg: str, out_path: Path) -> dict[str, Any]:
    rgb = svg_to_rgb(svg)
    Image.fromarray(rgb.astype(np.uint8)).save(out_path)
    return {
        "path": str(out_path),
        "shape": [int(x) for x in rgb.shape],
        "black_ratio_le_120": float((rgb.mean(axis=2) <= 120).mean()),
    }


def main(
    kern_path: str,
    output_dir: str | None = None,
    image_width: int = 1050,
    seed: int = 20260226,
    include_title: bool = False,
    include_author: bool = False,
    randomize_metadata_flags: bool = True,
):
    """Render one .krn file and save debug artifacts for each stage.

    Args:
        kern_path: Path to a normalized .krn file (with or without **kern header).
        output_dir: Directory to write debug artifacts. Defaults to /tmp.
        image_width: Target width used by _sample_render_options().
        seed: Random seed for Python and NumPy (affects sampler + SVG augmentation).
        include_title: Metadata title flag if randomize_metadata_flags is False.
        include_author: Metadata author flag if randomize_metadata_flags is False.
        randomize_metadata_flags: Match pipeline behavior by randomizing title/author booleans.
    """
    src_path = Path(kern_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"File not found: {src_path}")

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path("/tmp") / f"verovio_stage_debug_{src_path.stem}_{ts}"
    else:
        output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    if randomize_metadata_flags:
        include_title = bool(random.random() > 0.5)
        include_author = bool(random.random() > 0.5)

    metadata_prefix = generate_metadata_prefix(include_title, include_author)
    original_kern = src_path.read_text(encoding="utf-8")
    renderable = metadata_prefix + ensure_kern_header(original_kern)

    render_options = _sample_render_options(image_width)
    renderer = VerovioRenderer()
    raw_svg, page_count = renderer.render_to_svg(renderable, render_options)

    normalized_svg = normalize_svg_colors(raw_svg)
    augmented_svg = augment_svg(normalized_svg)

    # Write text artifacts first so they remain even if rasterization fails.
    (output_root / "source.krn").write_text(original_kern, encoding="utf-8")
    (output_root / "renderable.krn").write_text(renderable, encoding="utf-8")
    (output_root / "metadata_prefix.txt").write_text(metadata_prefix, encoding="utf-8")
    (output_root / "raw.svg").write_text(raw_svg, encoding="utf-8")
    (output_root / "normalized.svg").write_text(normalized_svg, encoding="utf-8")
    (output_root / "augmented.svg").write_text(augmented_svg, encoding="utf-8")
    (output_root / "render_options.json").write_text(
        json.dumps(render_options, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    raster_outputs: dict[str, dict[str, Any]] = {}
    raster_errors: dict[str, str] = {}
    for stage_name, stage_svg in (
        ("raw", raw_svg),
        ("normalized", normalized_svg),
        ("augmented", augmented_svg),
    ):
        try:
            raster_outputs[stage_name] = _write_png_from_svg(stage_svg, output_root / f"{stage_name}.png")
        except Exception as exc:  # noqa: BLE001 - debugging script should capture failures
            raster_errors[stage_name] = f"{exc.__class__.__name__}: {exc}"

    line_field_counts = Counter(line.count("\t") + 1 for line in original_kern.splitlines())

    summary = {
        "kern_path": str(src_path),
        "output_dir": str(output_root),
        "seed": seed,
        "image_width": image_width,
        "metadata_flags": {
            "include_title": include_title,
            "include_author": include_author,
            "randomize_metadata_flags": randomize_metadata_flags,
        },
        "page_count": int(page_count),
        "render_options": render_options,
        "raw_svg_stats": {
            "chars": len(raw_svg),
            "system_count_proxy": count_nr_of_systems_in_svg(raw_svg),
            "elements": _svg_element_counts(raw_svg),
        },
        "normalized_svg_stats": {
            "chars": len(normalized_svg),
            "system_count_proxy": count_nr_of_systems_in_svg(normalized_svg),
            "elements": _svg_element_counts(normalized_svg),
        },
        "augmented_svg_stats": {
            "chars": len(augmented_svg),
            "system_count_proxy": count_nr_of_systems_in_svg(augmented_svg),
            "elements": _svg_element_counts(augmented_svg),
        },
        "raster_outputs": raster_outputs,
        "raster_errors": raster_errors,
        "source_kern_field_count_histogram": dict(sorted(line_field_counts.items())),
    }
    (output_root / "summary.json").write_text(
        json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    fire.Fire(main)
