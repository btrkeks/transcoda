"""Shared I/O helpers for the production rewrite."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def encode_jpeg_image(image: np.ndarray, *, quality: int = 80) -> bytes:
    if image.dtype != np.uint8:
        raise ValueError("encode_jpeg_image expects uint8 image data")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    buffer = BytesIO()
    Image.fromarray(image).save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=False,
        progressive=True,
    )
    return buffer.getvalue()


def write_json(path: str | Path, payload: object) -> None:
    import json

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: str | Path, rows: list[object]) -> None:
    import json

    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
