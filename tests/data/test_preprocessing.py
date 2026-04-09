import numpy as np
import torch
from PIL import Image

from src.data.preprocessing import (
    LayoutNormalizationConfig,
    normalize_image,
    preprocess_pil_image,
)


def _tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image_hwc = tensor.clamp(-1.0, 1.0).permute(1, 2, 0)
    return ((image_hwc + 1.0) * 127.5).round().to(torch.uint8).numpy()


def _find_bbox(image: np.ndarray, threshold: int = 100) -> tuple[int, int, int, int] | None:
    black = np.min(image[:, :, :3], axis=2) <= threshold
    coords = np.argwhere(black)
    if coords.size == 0:
        return None
    return (
        int(coords[:, 0].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].min()),
        int(coords[:, 1].max()),
    )


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    top, bottom, left, right = bbox
    return bottom - top + 1, right - left + 1


def test_preprocess_pil_image_resizes_to_target_width():
    image = Image.fromarray(np.full((100, 200, 3), 128, dtype=np.uint8))

    tensor, model_size = preprocess_pil_image(image=image, image_width=100, fixed_size=None)

    assert model_size == (50, 100)
    assert tuple(tensor.shape) == (3, 50, 100)


def test_preprocess_pil_image_applies_fixed_size_bottom_padding():
    image = Image.fromarray(np.full((100, 200, 3), 0, dtype=np.uint8))

    tensor, model_size = preprocess_pil_image(image=image, image_width=100, fixed_size=(80, 100))

    assert model_size == (80, 100)
    assert tuple(tensor.shape) == (3, 80, 100)
    assert torch.allclose(tensor[:, -1, :], torch.ones(3, 100))


def test_preprocess_pil_image_applies_fixed_size_bottom_crop():
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)
    image_np[59, :, :] = 255
    image_np[99, :, :] = 127
    image = Image.fromarray(image_np)

    tensor, model_size = preprocess_pil_image(image=image, image_width=100, fixed_size=(60, 100))

    assert model_size == (60, 100)
    assert tuple(tensor.shape) == (3, 60, 100)
    assert torch.allclose(tensor[:, -1, :], torch.ones(3, 100))


def test_normalize_image_uses_cached_norm_tensors(monkeypatch):
    image = np.full((10, 10, 3), 127, dtype=np.uint8)

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("normalize_image should not allocate norm tensors per call")

    monkeypatch.setattr(torch, "tensor", _raise_if_called)

    out = normalize_image(image)
    assert out.shape == (3, 10, 10)


def test_preprocess_pil_image_normalizes_layout_to_reframed_page():
    image_np = np.full((40, 100, 3), 255, dtype=np.uint8)
    image_np[12:32, 20:80, :] = 0
    image = Image.fromarray(image_np)

    tensor, model_size = preprocess_pil_image(
        image=image,
        image_width=100,
        fixed_size=(150, 100),
        layout_normalization=LayoutNormalizationConfig(
            enabled=True,
            top_margin_px=10,
            side_margin_px=10,
        ),
    )

    assert model_size == (150, 100)
    bbox = _find_bbox(_tensor_to_uint8_image(tensor))
    assert bbox == (10, 29, 20, 79)


def test_preprocess_pil_image_layout_normalization_preserves_content_scale():
    image_np = np.full((40, 100, 3), 255, dtype=np.uint8)
    image_np[14:30, 28:74, :] = 0
    image = Image.fromarray(image_np)

    tensor, _ = preprocess_pil_image(
        image=image,
        image_width=100,
        fixed_size=(150, 100),
        layout_normalization=LayoutNormalizationConfig(enabled=True, top_margin_px=8, side_margin_px=12),
    )

    bbox = _find_bbox(_tensor_to_uint8_image(tensor))
    assert bbox is not None
    assert _bbox_size(bbox) == (16, 46)


def test_preprocess_pil_image_layout_normalization_ignores_tiny_speckles():
    image_np = np.full((40, 100, 3), 255, dtype=np.uint8)
    image_np[2, 2, :] = 0
    image_np[12:28, 30:70, :] = 0
    image = Image.fromarray(image_np)

    tensor, _ = preprocess_pil_image(
        image=image,
        image_width=100,
        fixed_size=(150, 100),
        layout_normalization=LayoutNormalizationConfig(enabled=True, top_margin_px=10, side_margin_px=10),
    )

    bbox = _find_bbox(_tensor_to_uint8_image(tensor))
    assert bbox == (10, 25, 30, 69)


def test_preprocess_pil_image_layout_normalization_falls_back_for_empty_image():
    image = Image.fromarray(np.full((80, 120, 3), 255, dtype=np.uint8))

    tensor, model_size = preprocess_pil_image(
        image=image,
        image_width=60,
        fixed_size=(80, 60),
        layout_normalization=LayoutNormalizationConfig(enabled=True),
    )

    assert model_size == (80, 60)
    assert tuple(tensor.shape) == (3, 80, 60)
    assert _find_bbox(_tensor_to_uint8_image(tensor)) is None


def test_preprocess_pil_image_layout_normalization_expands_width_for_side_margins():
    image_np = np.full((60, 100, 3), 255, dtype=np.uint8)
    image_np[20:40, 2:98, :] = 0
    image = Image.fromarray(image_np)

    tensor, _ = preprocess_pil_image(
        image=image,
        image_width=100,
        fixed_size=(150, 100),
        layout_normalization=LayoutNormalizationConfig(enabled=True, top_margin_px=10, side_margin_px=20),
    )

    bbox = _find_bbox(_tensor_to_uint8_image(tensor))
    assert bbox is not None
    _, _, left, right = bbox
    assert left >= 15
    assert (99 - right) >= 15


def test_preprocess_pil_image_layout_normalization_centers_content_horizontally():
    image_np = np.full((40, 100, 3), 255, dtype=np.uint8)
    image_np[12:28, 8:68, :] = 0
    image = Image.fromarray(image_np)

    tensor, _ = preprocess_pil_image(
        image=image,
        image_width=100,
        fixed_size=(150, 100),
        layout_normalization=LayoutNormalizationConfig(enabled=True, top_margin_px=10, side_margin_px=10),
    )

    bbox = _find_bbox(_tensor_to_uint8_image(tensor))
    assert bbox is not None
    _, _, left, right = bbox
    left_margin = left
    right_margin = 99 - right
    assert abs(left_margin - right_margin) <= 1


def test_preprocess_pil_image_layout_normalization_preserves_page_like_inputs():
    image_np = np.full((150, 100, 3), 255, dtype=np.uint8)
    image_np[12:32, 20:80, :] = 0
    image = Image.fromarray(image_np)

    tensor, model_size = preprocess_pil_image(
        image=image,
        image_width=100,
        fixed_size=(150, 100),
        layout_normalization=LayoutNormalizationConfig(enabled=True, top_margin_px=12, side_margin_px=10),
    )

    assert model_size == (150, 100)
    bbox = _find_bbox(_tensor_to_uint8_image(tensor))
    assert bbox == (12, 31, 20, 79)
