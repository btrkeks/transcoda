"""Probe synthetic generation in worker processes and capture degenerate renders.

This script exercises the same child-process rendering path as dataset generation
workers (VerovioRenderer + score_generator.generate), computes lightweight image
metrics, and saves raw SVGs for suspicious outputs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fire
import numpy as np
from pebble import ProcessExpired, ProcessPool
from PIL import Image

from scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment import (
    offline_augment,
)
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
    svg_to_rgb,
)
from scripts.dataset_generation.dataset_generation.image_generation.score_generator import (
    generate as generate_score,
)
from scripts.dataset_generation.dataset_generation.image_generation.types import GenerationConfig
from scripts.dataset_generation.dataset_generation.worker import (
    _capture_stderr_fd,
    ensure_kern_header,
    is_valid_kern,
)

TASK_TIMEOUT_SECONDS = 45

_probe_renderer: VerovioRenderer | None = None
_image_width: int | None = None
_image_height: int | None = None
_augment_seed: int | None = None
_capture_root: Path | None = None


@dataclass(frozen=True)
class ImageMetrics:
    height: int
    width: int
    mean_luma: float
    black_ratio_le_120: float
    black_ratio_le_160: float


def _compute_metrics(image: np.ndarray) -> ImageMetrics:
    rgb = image[:, :, :3] if image.ndim == 3 else image
    gray = rgb.mean(axis=2) if rgb.ndim == 3 else rgb.astype(np.float32)
    return ImageMetrics(
        height=int(rgb.shape[0]),
        width=int(rgb.shape[1]),
        mean_luma=float(gray.mean()),
        black_ratio_le_120=float((gray <= 120.0).mean()),
        black_ratio_le_160=float((gray <= 160.0).mean()),
    )


def _save_png(image: np.ndarray, path: Path) -> None:
    arr = np.ascontiguousarray(image[:, :, :3] if image.ndim == 3 else image)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def init_probe_worker(
    image_width: int,
    image_height: int | None,
    augment_seed: int | None,
    capture_root: str,
) -> None:
    global _probe_renderer, _image_width, _image_height, _augment_seed, _capture_root
    _probe_renderer = VerovioRenderer()
    _image_width = image_width
    _image_height = image_height
    _augment_seed = augment_seed
    _capture_root = Path(capture_root)
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))


def _is_suspicious(pre: ImageMetrics, post: ImageMetrics, actual_system_count: int) -> bool:
    # Barline-only failures are extremely sparse / pale and often report 0 systems with
    # the current proxy. Use broad thresholds for capture (false positives are acceptable).
    if pre.black_ratio_le_120 < 0.005:
        return True
    if post.black_ratio_le_120 < 0.005:
        return True
    if actual_system_count <= 0 and pre.black_ratio_le_120 < 0.03:
        return True
    return False


def _probe_one(file_path: str, variant_idx: int) -> dict[str, Any]:
    global _probe_renderer, _image_width, _image_height, _augment_seed, _capture_root
    assert _probe_renderer is not None
    assert _image_width is not None
    assert _capture_root is not None

    filename = Path(file_path).name
    kern_path = Path(file_path)

    try:
        kern_content = kern_path.read_text()
        is_valid, error_msg = is_valid_kern(kern_content)
        if not is_valid:
            return {"status": "invalid", "filename": filename, "error": error_msg}

        kern_for_verovio = ensure_kern_header(kern_content)
        config = GenerationConfig(
            num_systems_hint=1,
            include_author=bool(np.random.random() > 0.5),
            include_title=bool(np.random.random() > 0.5),
            image_width=_image_width,
            image_height=_image_height,
            texturize_image=False,
        )

        captured_svg_calls: list[dict[str, Any]] = []
        sampled_options: list[dict[str, Any]] = []

        import scripts.dataset_generation.dataset_generation.image_generation.score_generator as score_gen_mod

        orig_render_to_svg = _probe_renderer.render_to_svg
        orig_sampler = score_gen_mod._sample_render_options

        def wrapped_render_to_svg(music_sequence: str, render_options: dict[str, Any]):
            svg, page_count = orig_render_to_svg(music_sequence, render_options)
            captured_svg_calls.append(
                {
                    "page_count": int(page_count),
                    "render_options": dict(render_options),
                    "svg": svg,
                }
            )
            return svg, page_count

        def wrapped_sampler(image_width: int):
            opts = orig_sampler(image_width)
            sampled_options.append(dict(opts))
            return opts

        _probe_renderer.render_to_svg = wrapped_render_to_svg  # type: ignore[method-assign]
        score_gen_mod._sample_render_options = wrapped_sampler  # type: ignore[assignment]

        try:
            with _capture_stderr_fd() as captured_stderr:
                result = generate_score(kern_for_verovio, _probe_renderer, config)
        finally:
            _probe_renderer.render_to_svg = orig_render_to_svg  # type: ignore[method-assign]
            score_gen_mod._sample_render_options = orig_sampler  # type: ignore[assignment]

        stderr_text = "".join(captured_stderr).strip()
        if result is None:
            return {
                "status": "rejected",
                "filename": filename,
                "stderr": stderr_text,
                "sampled_options": sampled_options,
                "svg_call_count": len(captured_svg_calls),
            }

        pre_metrics = _compute_metrics(result.image)
        post_image = offline_augment(
            result.image,
            filename=filename,
            variant_idx=variant_idx,
            augment_seed=_augment_seed,
        )
        post_metrics = _compute_metrics(post_image)
        suspicious = _is_suspicious(pre_metrics, post_metrics, result.actual_system_count)

        capture_dir: str | None = None
        if suspicious:
            ts = int(time.time() * 1000)
            capture_path = _capture_root / f"{kern_path.stem}_v{variant_idx}_{os.getpid()}_{ts}"
            capture_path.mkdir(parents=True, exist_ok=True)
            capture_dir = str(capture_path)

            (capture_path / "source.krn").write_text(kern_content)
            (capture_path / "renderable.krn").write_text(result.metadata_prefix + kern_for_verovio)
            _save_png(result.image, capture_path / "pre_offline.png")
            _save_png(post_image, capture_path / "post_offline.png")

            for idx, call in enumerate(captured_svg_calls):
                svg_path = capture_path / f"raw_attempt_{idx}.svg"
                svg_path.write_text(call["svg"])
                try:
                    raw_rgb = svg_to_rgb(call["svg"])
                    _save_png(raw_rgb, capture_path / f"raw_attempt_{idx}.png")
                except Exception as exc:  # noqa: BLE001
                    (capture_path / f"raw_attempt_{idx}.raster_error.txt").write_text(
                        f"{type(exc).__name__}: {exc}"
                    )

            meta = {
                "filename": filename,
                "variant_idx": variant_idx,
                "pid": os.getpid(),
                "status": "success_suspicious",
                "actual_system_count": int(result.actual_system_count),
                "metadata_prefix": result.metadata_prefix,
                "config": asdict(config),
                "pre_metrics": asdict(pre_metrics),
                "post_metrics": asdict(post_metrics),
                "stderr": stderr_text,
                "sampled_options": sampled_options,
                "captured_svg_calls": [
                    {
                        "page_count": c["page_count"],
                        "render_options": c["render_options"],
                        "svg_len": len(c["svg"]),
                    }
                    for c in captured_svg_calls
                ],
            }
            (capture_path / "meta.json").write_text(json.dumps(meta, indent=2))

        return {
            "status": "success",
            "filename": filename,
            "suspicious": suspicious,
            "capture_dir": capture_dir,
            "actual_system_count": int(result.actual_system_count),
            "pre_metrics": asdict(pre_metrics),
            "post_metrics": asdict(post_metrics),
            "stderr": stderr_text,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "filename": filename,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main(
    normalized_dir: str,
    *,
    num_samples: int = 120,
    num_workers: int = 4,
    image_width: int = 1050,
    image_height: int | None = 1485,
    augment_seed: int | None = None,
    out_dir: str | None = None,
) -> None:
    src_dir = Path(normalized_dir)
    files = sorted(src_dir.glob("*.krn"))
    files = files[: min(num_samples, len(files))]

    if not files:
        raise ValueError(f"No .krn files found in {src_dir}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) if out_dir else Path(f"/tmp/processpool_render_probe_{src_dir.name}_{ts}")
    out_path.mkdir(parents=True, exist_ok=True)
    capture_root = out_path / "captures"
    capture_root.mkdir(exist_ok=True)

    manifest_path = out_path / "manifest.jsonl"
    summary_path = out_path / "summary.json"

    results: list[dict[str, Any]] = []
    counters = {
        "submitted": len(files),
        "success": 0,
        "suspicious": 0,
        "rejected": 0,
        "invalid": 0,
        "error": 0,
        "expired": 0,
        "timeout": 0,
    }

    with ProcessPool(
        max_workers=num_workers,
        max_tasks=200,
        initializer=init_probe_worker,
        initargs=(image_width, image_height, augment_seed, str(capture_root)),
    ) as pool, manifest_path.open("w") as manifest_f:
        futures = [
            pool.schedule(_probe_one, args=(str(path), 0), timeout=TASK_TIMEOUT_SECONDS)
            for path in files
        ]
        for fut in futures:
            try:
                result = fut.result()
            except TimeoutError:
                counters["timeout"] += 1
                continue
            except ProcessExpired:
                counters["expired"] += 1
                continue

            status = result.get("status")
            if status == "success":
                counters["success"] += 1
                if result.get("suspicious"):
                    counters["suspicious"] += 1
            elif status == "rejected":
                counters["rejected"] += 1
            elif status == "invalid":
                counters["invalid"] += 1
            else:
                counters["error"] += 1

            results.append(result)
            manifest_f.write(json.dumps(result) + "\n")

    summary = {
        "source_dir": str(src_dir),
        "num_workers": num_workers,
        "image_width": image_width,
        "image_height": image_height,
        "augment_seed": augment_seed,
        "counters": counters,
        "capture_root": str(capture_root),
        "suspicious_capture_dirs": [
            r["capture_dir"] for r in results if r.get("status") == "success" and r.get("suspicious")
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
