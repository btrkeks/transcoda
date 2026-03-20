import numpy as np
import cv2

from scripts.dataset_generation.dataset_generation.image_augmentation.geometric_augment import (
    GeometricTransform,
    apply_geometric_transform,
    geometric_augment,
)
from scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment import (
    _background_border_value,
    _detect_border_fill,
    _passes_out_of_bounds_gate_from_masks,
    required_padding_for_safe_crop,
    offline_augment,
    passes_quality_gate,
    passes_transform_consistency,
)
from scripts.dataset_generation.dataset_generation.image_generation.types import RenderedPage


def test_geometric_augment_preserves_shape_dtype():
    img = np.full((64, 128, 3), 255, dtype=np.uint8)
    img[20:44, 30:98] = 0
    out = geometric_augment(img, np.random.default_rng(4))
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]


def test_geometric_zoom_out_keeps_white_fill_for_clean_page_warp():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = geometric_augment(img, np.random.default_rng(0))
    assert np.any(np.all(out == 255, axis=2))


def test_geometric_zoom_in_no_black_border_fill():
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    img[16:48, 16:48] = 0
    out = geometric_augment(img, np.random.default_rng(1))
    borders = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]], axis=0)
    assert np.any(np.any(borders > 0, axis=1))


def test_geometric_augment_deterministic_seed():
    img = np.full((80, 120, 3), 255, dtype=np.uint8)
    img[30:50, 40:80] = 0
    out1 = geometric_augment(img, np.random.default_rng(123))
    out2 = geometric_augment(img, np.random.default_rng(123))
    assert np.array_equal(out1, out2)


def test_passes_quality_gate_rejects_extremes():
    dark = np.zeros((64, 64, 3), dtype=np.uint8)
    bright = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert not passes_quality_gate(dark)
    assert not passes_quality_gate(bright)


def test_passes_quality_gate_rejects_three_border_touch():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[0, :, :] = 0
    img[-1, :, :] = 0
    img[:, 0, :] = 0
    assert not passes_quality_gate(img)


def test_passes_quality_gate_rejects_small_margin_translation():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[40:60, 2:22] = 0
    assert not passes_quality_gate(img)


def test_passes_transform_consistency_rejects_large_shift():
    base = np.full((120, 120, 3), 255, dtype=np.uint8)
    base[40:80, 40:80] = 0
    shifted = np.full((120, 120, 3), 255, dtype=np.uint8)
    shifted[40:80, 90:119] = 0
    assert not passes_transform_consistency(base, shifted)


def test_passes_transform_consistency_accepts_small_shift():
    base = np.full((120, 120, 3), 255, dtype=np.uint8)
    base[40:80, 40:80] = 0
    shifted = np.full((120, 120, 3), 255, dtype=np.uint8)
    shifted[42:82, 44:84] = 0
    assert passes_transform_consistency(base, shifted)


def test_offline_augment_falls_back_to_base_when_all_fail(monkeypatch):
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0
    bad = np.zeros((64, 64, 3), dtype=np.uint8)
    invalid = np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._build_augmented_candidate",
        lambda **kwargs: (bad, np.zeros((64, 64), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: invalid,
    )

    out = offline_augment(base, filename="x.krn", variant_idx=0, augment_seed=7)
    assert np.array_equal(out, base)


def test_offline_augment_accepts_post_augraphy_output_without_quality_gate(monkeypatch):
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0
    dark = np.zeros((64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._build_augmented_candidate",
        lambda **kwargs: (base, np.where(base[:, :, 0] < 255, 255, 0).astype(np.uint8)),
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: dark,
    )

    out = offline_augment(base, filename="x.krn", variant_idx=0, augment_seed=7)
    assert np.array_equal(out, dark)


def test_offline_augment_retries_conservative_when_oob_gate_fails(monkeypatch):
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0
    bad = np.zeros((64, 64, 3), dtype=np.uint8)
    calls = []

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._build_augmented_candidate",
        lambda conservative, **kwargs: (
            calls.append(conservative) or (
                (base, np.where(base[:, :, 0] < 255, 255, 0).astype(np.uint8))
                if conservative
                else (bad, np.zeros((64, 64), dtype=np.uint8))
            )
        ),
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: image,
    )

    out = offline_augment(base, filename="x.krn", variant_idx=1, augment_seed=7)
    assert np.array_equal(out, base)
    assert calls == [False, True]


def test_offline_augment_returns_substage_timings(monkeypatch):
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._build_augmented_candidate",
        lambda **kwargs: (base, np.where(base[:, :, 0] < 255, 255, 0).astype(np.uint8)),
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: image,
    )

    out, timings = offline_augment(
        base,
        filename="x.krn",
        variant_idx=0,
        augment_seed=11,
        return_timings=True,
    )
    assert np.array_equal(out, base)
    assert set(timings.keys()) == {
        "offline_geom_ms",
        "offline_gates_ms",
        "offline_augraphy_ms",
        "offline_texture_ms",
    }
    assert timings["offline_geom_ms"] >= 0.0
    assert timings["offline_gates_ms"] >= 0.0
    assert timings["offline_augraphy_ms"] >= 0.0
    assert timings["offline_texture_ms"] >= 0.0


def test_offline_augment_realistic_branch_avoids_white_wedges(monkeypatch):
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0
    layers = RenderedPage(image=base, foreground=base.copy(), alpha=(255 - base[:, :, 0]).astype(np.uint8))

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._sample_branch_choice",
        lambda rng: "realistic",
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: image,
    )

    out = offline_augment(
        base,
        render_layers=layers,
        filename="x.krn",
        variant_idx=0,
        augment_seed=5,
    )
    assert out.shape == base.shape
    assert not np.any(np.all(out == 255, axis=2))


def test_gate_rejects_visible_extra_notation_even_if_alpha_mask_misses_it():
    base_mask = np.zeros((96, 96), dtype=np.uint8)
    base_mask[28:68, 20:56] = 255

    candidate_mask = np.zeros((96, 96), dtype=np.uint8)
    candidate_mask[30:70, 22:58] = 255

    candidate_visible = candidate_mask.copy()
    candidate_visible[30:70, 76:92] = 255

    assert not _passes_out_of_bounds_gate_from_masks(base_mask, candidate_visible)
    assert _passes_out_of_bounds_gate_from_masks(base_mask, candidate_mask)


def test_required_padding_for_safe_crop_identity_returns_safety_pad():
    assert required_padding_for_safe_crop(None, (128, 96)) == (8, 8)


def test_required_padding_for_safe_crop_grows_for_translation():
    transform = GeometricTransform(
        affine=np.array([[1.0, 0.0, 18.0], [0.0, 1.0, -11.0]], dtype=np.float32),
        perspective=None,
    )
    pad_y, pad_x = required_padding_for_safe_crop(transform, (128, 96))
    assert pad_x >= 26
    assert pad_y >= 19


def test_required_padding_for_safe_crop_handles_perspective():
    transform = GeometricTransform(
        affine=np.array([[1.0, 0.0, 6.0], [0.0, 1.0, 4.0]], dtype=np.float32),
        perspective=np.array(
            [[1.0, 0.0, 3.0], [0.0, 1.0, 2.0], [0.0008, 0.0004, 1.0]],
            dtype=np.float32,
        ),
    )
    pad_y, pad_x = required_padding_for_safe_crop(transform, (128, 96))
    assert pad_x >= 8
    assert pad_y >= 8
    assert pad_x < 200
    assert pad_y < 200


def test_realistic_branch_fixed_transform_does_not_create_reflected_right_strip():
    from scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment import (
        _center_crop,
        _embed_in_canvas,
        _normalize_render_layers,
        _realistic_branch,
        _translate_transform,
    )

    base = np.full((128, 128, 3), 255, dtype=np.uint8)
    base[22:104, 12:108] = 0
    layers = _normalize_render_layers(base, None)
    transform = GeometricTransform(
        affine=np.array(
            [[0.75544864, -0.01353208, 50.862587], [0.01676752, 0.93607205, 22.991432]],
            dtype=np.float32,
        ),
        perspective=None,
    )

    out, visible_mask = _realistic_branch(
        base,
        layers,
        transform,
        textures=[],
        rng=np.random.default_rng(17),
    )

    pad_y, pad_x = required_padding_for_safe_crop(transform, base.shape[:2])
    alpha_canvas = _embed_in_canvas(layers.alpha, (128 + (2 * pad_y), 128 + (2 * pad_x)), (pad_y, pad_x), 0)
    adjusted_transform = _translate_transform(transform, offset_x=pad_x, offset_y=pad_y)
    warped_alpha = apply_geometric_transform(
        alpha_canvas,
        adjusted_transform,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
        interpolation=cv2.INTER_LINEAR,
    )
    cropped_alpha = _center_crop(warped_alpha, (128, 128), (pad_y, pad_x))
    expected_support = cv2.dilate((cropped_alpha >= 8).astype(np.uint8), np.ones((3, 3), np.uint8))
    border_value = _background_border_value(np.full((128 + (2 * pad_y), 128 + (2 * pad_x), 3), 245, dtype=np.uint8))

    assert out.shape == base.shape
    assert visible_mask.shape == base.shape[:2]
    assert not np.any((visible_mask > 0) & (expected_support == 0))
    assert not _detect_border_fill(out, border_value, band_px=4)


def test_offline_augment_foreground_branch_composites_without_hard_seams(monkeypatch):
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[20:44, 16:48] = 180
    foreground = np.full((64, 64, 3), 255, dtype=np.uint8)
    foreground[20:44, 16:48] = 0
    layers = RenderedPage(image=base, foreground=foreground, alpha=alpha)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._sample_branch_choice",
        lambda rng: "foreground",
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: image,
    )

    out = offline_augment(
        base,
        render_layers=layers,
        filename="x.krn",
        variant_idx=0,
        augment_seed=9,
    )
    assert out.shape == base.shape
    assert np.any((out[:, :, 0] > 0) & (out[:, :, 0] < 255))


def test_offline_augment_can_be_bypassed_in_worker_postprocess(monkeypatch):
    import scripts.dataset_generation.dataset_generation.worker as worker
    from scripts.dataset_generation.dataset_generation.worker_models import WorkerInitConfig

    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0

    config = WorkerInitConfig(
        image_width=64,
        image_height=64,
        disable_offline_image_augmentations=True,
    )

    called = {"offline": False}

    def _offline_should_not_run(*args, **kwargs):
        called["offline"] = True
        return base

    monkeypatch.setitem(
        __import__("sys").modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        type("_OfflineModule", (), {"offline_augment": _offline_should_not_run})(),
    )

    jpeg_bytes, offline_ms, jpeg_ms, stage_timings = worker._postprocess_rendered_image(
        base,
        filename="x.krn",
        variant_idx=0,
        config=config,
        profile_enabled=True,
    )

    assert called["offline"] is False
    assert isinstance(jpeg_bytes, bytes)
    assert offline_ms == 0.0
    assert jpeg_ms >= 0.0
    assert stage_timings is None
