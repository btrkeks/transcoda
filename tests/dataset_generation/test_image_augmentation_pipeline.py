import numpy as np

from scripts.dataset_generation.dataset_generation.image_augmentation.geometric_augment import (
    geometric_augment,
    sample_geometric_transform,
)
from scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment import (
    OfflineAugmentTrace,
    _passes_out_of_bounds_gate_from_masks,
    offline_augment,
    passes_quality_gate,
    passes_transform_consistency,
)
from scripts.dataset_generation.dataset_generation.image_generation.types import RenderedPage


class _StubRng:
    def __init__(self, *, random_values, uniform_values):
        self._random_values = iter(random_values)
        self._uniform_values = iter(uniform_values)

    def random(self):
        return next(self._random_values)

    def uniform(self, low=0.0, high=1.0, size=None):
        value = next(self._uniform_values)
        if size is None:
            return value
        array = np.asarray(value, dtype=np.float32)
        return np.broadcast_to(array, size).copy()


def _make_good_candidate(base):
    """Build a (fg, alpha, mask, True) tuple that passes the OOB gate."""
    fg = base.copy()
    alpha = np.where(np.min(base, axis=2) < 255, 255, 0).astype(np.uint8)
    mask = alpha.copy()
    return fg, alpha, mask, True


def _make_bad_candidate(shape):
    """Build a (fg, alpha, mask, True) tuple that fails the OOB gate."""
    h, w = shape[:2]
    return (
        np.zeros((h, w, 3), dtype=np.uint8),
        np.zeros((h, w), dtype=np.uint8),
        np.zeros((h, w), dtype=np.uint8),
        True,
    )


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


def test_sample_geometric_transform_uses_available_bottom_whitespace():
    mask = np.zeros((200, 200), dtype=bool)
    mask[20:40, 50:150] = True
    rng = _StubRng(
        random_values=[0.0],
        uniform_values=[0.0, 0.0, 148.0, 0.0],
    )

    transform = sample_geometric_transform(
        (200, 200),
        rng,
        conservative=True,
        x_squeeze_prob=0.0,
        content_mask=mask,
        min_margin_px=12,
    )

    assert transform is not None
    assert np.isclose(transform.ty_px, 148.0)
    assert transform.ty_px > 0.025 * 200


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


def test_offline_augment_falls_back_to_textured_candidate_when_augraphy_fails(monkeypatch):
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0
    invalid = np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._build_augmented_candidate",
        lambda **kwargs: _make_bad_candidate(base.shape),
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: (invalid, "applied"),
    )

    out, _, trace = offline_augment(base, filename="x.krn", variant_idx=0, augment_seed=7)
    assert out.shape == base.shape
    assert out.dtype == np.uint8
    assert isinstance(trace, OfflineAugmentTrace)
    assert trace.augraphy_fallback_attempted
    assert not trace.augraphy_normalize_accepted
    assert trace.augraphy_fallback_normalize_accepted is False


def test_offline_augment_accepts_post_augraphy_output_without_quality_gate(monkeypatch):
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0
    dark = np.zeros((64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._build_augmented_candidate",
        lambda **kwargs: _make_good_candidate(base),
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: (dark, "applied"),
    )

    out, _, trace = offline_augment(base, filename="x.krn", variant_idx=0, augment_seed=7)
    assert np.array_equal(out, dark)
    assert trace.augraphy_outcome == "applied"
    assert trace.augraphy_normalize_accepted


def test_offline_augment_retries_conservative_when_oob_gate_fails(monkeypatch):
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0
    calls = []

    good = _make_good_candidate(base)
    bad = _make_bad_candidate(base.shape)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._build_augmented_candidate",
        lambda conservative, **kwargs: (
            calls.append(conservative) or (good if conservative else bad)
        ),
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: (image, "applied"),
    )

    out, _, trace = offline_augment(base, filename="x.krn", variant_idx=1, augment_seed=7)
    assert out.shape == base.shape
    assert out.dtype == np.uint8
    assert calls == [False, True]
    assert trace.retry_geometry is not None
    assert trace.retry_oob_gate is not None
    assert trace.retry_oob_gate.passed
    assert trace.final_geometry_applied


def test_offline_augment_returns_trace_with_timings(monkeypatch):
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._build_augmented_candidate",
        lambda **kwargs: _make_good_candidate(base),
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: (image, "applied"),
    )

    out, _, trace = offline_augment(
        base,
        filename="x.krn",
        variant_idx=0,
        augment_seed=11,
    )
    assert out.shape == base.shape
    assert isinstance(trace, OfflineAugmentTrace)
    assert trace.offline_geom_ms >= 0.0
    assert trace.offline_gates_ms >= 0.0
    assert trace.offline_augraphy_ms >= 0.0
    assert trace.offline_texture_ms >= 0.0
    assert trace.branch == "geometric"


def test_offline_augment_trace_records_augraphy_fallback(monkeypatch):
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0
    bad_shape = np.zeros((10, 10, 3), dtype=np.uint8)

    call_count = [0]

    def mock_augraphy(image, seed=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return bad_shape, "applied"  # will fail normalize
        return image, "applied"  # fallback succeeds

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._build_augmented_candidate",
        lambda **kwargs: _make_good_candidate(base),
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        mock_augraphy,
    )

    out, _, trace = offline_augment(base, filename="x.krn", variant_idx=0, augment_seed=5)
    assert not trace.augraphy_normalize_accepted
    assert trace.augraphy_fallback_attempted
    assert trace.augraphy_fallback_outcome == "applied"
    assert trace.augraphy_fallback_normalize_accepted


def test_gate_rejects_visible_extra_notation_even_if_alpha_mask_misses_it():
    base_mask = np.zeros((96, 96), dtype=np.uint8)
    base_mask[28:68, 20:56] = 255

    candidate_mask = np.zeros((96, 96), dtype=np.uint8)
    candidate_mask[30:70, 22:58] = 255

    candidate_visible = candidate_mask.copy()
    candidate_visible[30:70, 76:92] = 255

    assert not _passes_out_of_bounds_gate_from_masks(base_mask, candidate_visible)
    assert _passes_out_of_bounds_gate_from_masks(base_mask, candidate_mask)


def test_offline_augment_composites_with_blended_alpha(monkeypatch):
    """With partial alpha, output should have blended (intermediate) pixel values."""
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[20:44, 16:48] = 180
    foreground = np.full((64, 64, 3), 255, dtype=np.uint8)
    foreground[20:44, 16:48] = 0
    layers = RenderedPage(image=base, foreground=foreground, alpha=alpha)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: (image, "applied"),
    )

    out, _, trace = offline_augment(
        base,
        render_layers=layers,
        filename="x.krn",
        variant_idx=0,
        augment_seed=9,
    )
    assert out.shape == base.shape
    assert np.any((out[:, :, 0] > 0) & (out[:, :, 0] < 255))
    assert trace.branch == "geometric"


def test_retry_fail_still_gets_textured_background(monkeypatch):
    """Even when both OOB gate attempts fail, the output should have a textured
    (non-white) background rather than the original white Verovio render."""
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[20:44, 16:48] = 0

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment._build_augmented_candidate",
        lambda **kwargs: _make_bad_candidate(base.shape),
    )
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment.augraphy_augment",
        lambda image, seed=None: (image, "applied"),
    )

    out, _, trace = offline_augment(base, filename="x.krn", variant_idx=0, augment_seed=42)
    assert out.shape == base.shape
    assert trace.retry_geometry is not None
    assert trace.retry_oob_gate is not None
    assert trace.retry_oob_gate.failure_reason == "empty_candidate"
    assert not trace.final_geometry_applied
    # The non-notation region should have textured background (not all 255)
    bg_region = out[0:19, 0:15]  # top-left corner, well away from notation
    assert not np.all(bg_region == 255), "Background should be textured, not plain white"
