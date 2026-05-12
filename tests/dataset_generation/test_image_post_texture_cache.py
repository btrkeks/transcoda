from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.dataset_generation.dataset_generation.image_generation import image_post


def test_apply_random_texture_reuses_decoded_texture(monkeypatch, tmp_path: Path):
    texture_path = tmp_path / "texture.png"
    texture = np.full((300, 300, 3), 245, dtype=np.uint8)
    texture[30:80, 30:80] = [230, 248, 236]
    Image.fromarray(texture, mode="RGB").save(texture_path)

    sample = np.full((120, 140, 3), 255, dtype=np.uint8)
    sample[20:40, 20:100] = 0

    image_post._TEXTURE_RGB_CACHE.clear()
    image_post._TEXTURE_TILED_CACHE.clear()

    open_calls = {"count": 0}
    original_open = image_post.Image.open

    def _counting_open(path, *args, **kwargs):
        open_calls["count"] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(image_post.Image, "open", _counting_open)

    for _ in range(3):
        out = image_post.apply_random_texture(sample, [str(texture_path)])
        assert out.shape == sample.shape
        assert out.dtype == np.uint8

    assert open_calls["count"] == 1
