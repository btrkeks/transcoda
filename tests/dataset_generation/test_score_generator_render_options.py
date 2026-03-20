# ruff: noqa: E402, I001

import random
import sys
import types

import numpy as np
import pytest

sys.modules.setdefault(
    "pebble",
    types.SimpleNamespace(ProcessExpired=RuntimeError, ProcessPool=object),
)
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules.setdefault("cv2", types.SimpleNamespace(resize=lambda image, size: image))
sys.modules.setdefault("names", types.SimpleNamespace(get_full_name=lambda: "Test Author"))


class _DummyRandomSentence:
    def sentence(self):
        return "Test Title"


sys.modules.setdefault("wonderwords", types.SimpleNamespace(RandomSentence=_DummyRandomSentence))

from scripts.dataset_generation.dataset_generation.image_generation.score_generator import (
    _assess_frame_fit,
    _apply_render_option_guardrails,
    _make_horizontal_compact_fallback,
    _make_verovio_fit_fallback,
    _make_vertical_compact_fallback,
    _sample_render_options,
    generate,
    generate_with_diagnostics,
)
from scripts.dataset_generation.dataset_generation.image_generation.types import GenerationConfig


def test_sample_render_options_bounds_and_types():
    random.seed(12345)

    for _ in range(2000):
        opts = _sample_render_options(image_width=1050)

        assert opts["footer"] == "none"
        assert opts["breaks"] == "auto"

        assert isinstance(opts["scale"], int)
        assert 40 <= opts["scale"] <= 145
        assert isinstance(opts["beamMaxSlope"], int)
        assert 0 <= opts["beamMaxSlope"] <= 20
        assert isinstance(opts["spacingStaff"], int)
        assert 4 <= opts["spacingStaff"] <= 26
        assert isinstance(opts["spacingSystem"], int)
        assert 4 <= opts["spacingSystem"] <= 20
        assert isinstance(opts["pageWidth"], int)
        assert int(round(1050 * 1.18)) <= opts["pageWidth"] <= int(round(1050 * 2.70))

        assert 0.10 <= opts["barLineWidth"] <= 0.80
        assert 0.10 <= opts["staffLineWidth"] <= 0.30
        assert 0.10 <= opts["stemWidth"] <= 0.45
        assert 0.10 <= opts["ledgerLineThickness"] <= 0.50
        assert 0.50 <= opts["thickBarlineThickness"] <= 2.00
        assert 0.18 <= opts["spacingLinear"] <= 0.42
        assert 0.35 <= opts["spacingNonLinear"] <= 0.85
        assert 1 <= opts["measureMinWidth"] <= 30

        for key in (
            "pageMarginLeft",
            "pageMarginRight",
            "pageMarginTop",
            "pageMarginBottom",
        ):
            assert isinstance(opts[key], int)
            assert opts[key] >= 0

        for key in ("breaksNoWidow", "justifyVertically", "noJustification"):
            assert isinstance(opts[key], bool)


def test_sample_render_options_distribution_smoke():
    random.seed(9876)
    samples = [_sample_render_options(image_width=1050) for _ in range(2500)]

    scales = {opts["scale"] for opts in samples}
    assert len(scales) >= 12

    polish_like = sum(opts["pageWidth"] < int(1050 * 1.40) for opts in samples)
    compact = sum(
        int(1050 * 1.65) <= opts["pageWidth"] < int(1050 * 1.95) for opts in samples
    )
    mid = sum(int(1050 * 1.95) <= opts["pageWidth"] < int(1050 * 2.25) for opts in samples)
    open_ = sum(opts["pageWidth"] >= int(1050 * 2.25) for opts in samples)
    assert polish_like > 300
    assert compact > 450
    assert mid > 650
    assert open_ > 250

    no_just = sum(opts["noJustification"] for opts in samples)
    justify_vertical = sum(opts["justifyVertically"] for opts in samples)
    no_widow = sum(opts["breaksNoWidow"] for opts in samples)
    assert 40 <= no_just <= 260
    assert 550 <= justify_vertical <= 1200
    assert 650 <= no_widow <= 1400


def test_default_profile_can_emit_polish_like_dense_pages():
    random.seed(13579)

    compact_hits = []
    open_hits = []
    for _ in range(3000):
        opts = _sample_render_options(image_width=1050)
        if opts["pageWidth"] < int(round(1050 * 1.40)):
            compact_hits.append(opts)
        if opts["pageWidth"] >= int(round(1050 * 2.25)):
            open_hits.append(opts)

    assert compact_hits
    assert open_hits

    for opts in compact_hits[:25]:
        assert 76 <= opts["scale"] <= 90
        assert 6 <= opts["spacingStaff"] <= 11
        assert 7 <= opts["spacingSystem"] <= 13
        assert 12 <= opts["measureMinWidth"] <= 22
        assert 0.24 <= opts["spacingLinear"] <= 0.36
        assert 0.44 <= opts["spacingNonLinear"] <= 0.68
        assert opts["justifyVertically"] is True
        assert opts["breaksNoWidow"] is True


def test_sample_render_options_target_5_6_systems_bounds():
    random.seed(2468)

    for _ in range(1000):
        opts = _sample_render_options(image_width=1050, layout_profile="target_5_6_systems")
        assert 46 <= opts["scale"] <= 58
        assert 4 <= opts["spacingStaff"] <= 8
        assert 3 <= opts["spacingSystem"] <= 6
        assert 4 <= opts["measureMinWidth"] <= 10
        assert int(round(1050 * 2.42)) <= opts["pageWidth"] <= int(round(1050 * 2.54))
        assert opts["justifyVertically"] is False
        assert opts["noJustification"] is False
        assert opts["breaksNoWidow"] is False


def test_sample_render_options_polish_5_6_systems_bounds():
    random.seed(8642)

    for _ in range(1000):
        opts = _sample_render_options(image_width=1050, layout_profile="polish_5_6_systems")
        assert 76 <= opts["scale"] <= 90
        assert 6 <= opts["spacingStaff"] <= 11
        assert 7 <= opts["spacingSystem"] <= 13
        assert 12 <= opts["measureMinWidth"] <= 22
        assert int(round(1050 * 1.18)) <= opts["pageWidth"] <= int(round(1050 * 1.38))
        assert 0.24 <= opts["spacingLinear"] <= 0.36
        assert 0.44 <= opts["spacingNonLinear"] <= 0.68
        assert opts["justifyVertically"] is True
        assert opts["noJustification"] is False
        assert opts["breaksNoWidow"] is True


def test_compact_page_guardrail_caps_vertical_spacing():
    opts = {
        "spacingStaff": 26,
        "spacingSystem": 30,
        "measureMinWidth": 72,
        "staffLineWidth": 0.20,
        "stemWidth": 0.40,
        "ledgerLineThickness": 0.45,
    }

    _apply_render_option_guardrails(opts, page_width_factor=1.80)

    assert opts["spacingStaff"] == 20
    assert opts["spacingSystem"] == 18


def test_open_page_guardrail_caps_measure_min_width():
    opts = {
        "measureMinWidth": 72,
        "staffLineWidth": 0.20,
        "stemWidth": 0.40,
        "ledgerLineThickness": 0.45,
    }

    _apply_render_option_guardrails(opts, page_width_factor=2.55)

    assert opts["measureMinWidth"] == 26


def test_thick_staff_guardrail_caps_stem_and_ledger():
    opts = {
        "staffLineWidth": 0.28,
        "stemWidth": 0.44,
        "ledgerLineThickness": 0.49,
    }

    _apply_render_option_guardrails(opts, page_width_factor=2.10)

    assert opts["stemWidth"] == 0.40
    assert opts["ledgerLineThickness"] == 0.42


def _make_image(*, height=120, width=1050, row=None, col=20):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    if row is not None:
        row = int(row)
        col = int(col)
        col_end = min(width, col + 20)
        img[row, col:col_end] = 0
    return img


def test_assess_frame_fit_no_content_detected():
    ok, reason, metrics = _assess_frame_fit(
        _make_image(row=None),
        target_height=None,
        min_frame_margin_px=12,
    )

    assert not ok
    assert reason == "no_content_detected"
    assert metrics["bottom_row"] is None


def test_assess_frame_fit_passes_with_safe_margins():
    ok, reason, metrics = _assess_frame_fit(
        _make_image(height=120, width=200, row=60, col=90),
        target_height=None,
        min_frame_margin_px=12,
    )

    assert ok
    assert reason is None
    assert metrics["top_clearance_px"] == 60
    assert metrics["left_clearance_px"] == 90


def test_assess_frame_fit_rejects_bottom_clearance():
    ok, reason, metrics = _assess_frame_fit(
        _make_image(height=120, width=200, row=112, col=90),
        target_height=None,
        min_frame_margin_px=12,
    )

    assert not ok
    assert reason == "bottom_clearance"
    assert metrics["bottom_clearance_px"] == 7


def test_assess_frame_fit_rejects_top_and_side_clearance():
    ok_top, reason_top, _ = _assess_frame_fit(
        _make_image(height=120, width=200, row=5, col=90),
        target_height=None,
        min_frame_margin_px=12,
    )
    ok_left, reason_left, _ = _assess_frame_fit(
        _make_image(height=120, width=200, row=60, col=5),
        target_height=None,
        min_frame_margin_px=12,
    )
    ok_right, reason_right, _ = _assess_frame_fit(
        _make_image(height=120, width=200, row=60, col=185),
        target_height=None,
        min_frame_margin_px=12,
    )

    assert not ok_top
    assert reason_top == "top_clearance"
    assert not ok_left
    assert reason_left == "left_clearance"
    assert not ok_right
    assert reason_right == "right_clearance"


def test_assess_frame_fit_rejects_crop_risk():
    ok, reason, metrics = _assess_frame_fit(
        _make_image(height=160, width=200, row=120, col=90),
        target_height=100,
        min_frame_margin_px=12,
    )

    assert not ok
    assert reason == "crop_risk"
    assert metrics["bottom_clearance_px"] == 39
    assert metrics["cropped_clearance_px"] == -21


def test_assess_frame_fit_passes_crop_case_with_clearance():
    ok, reason, metrics = _assess_frame_fit(
        _make_image(height=160, width=200, row=60, col=90),
        target_height=100,
        min_frame_margin_px=12,
    )

    assert ok
    assert reason is None
    assert metrics["cropped_clearance_px"] == 39


def test_assess_frame_fit_accepts_borderline_with_hard_margin_and_records_metric():
    ok, reason, metrics = _assess_frame_fit(
        _make_image(height=120, width=200, row=9, col=90),
        target_height=None,
        min_frame_margin_px=8,
        target_frame_margin_px=12,
    )

    assert ok
    assert reason is None
    assert metrics["hard_margin_px"] == 8
    assert metrics["target_margin_px"] == 12
    assert metrics["borderline_clearance_px"] == 9


def test_assess_frame_fit_records_bottom_whitespace_and_vertical_fill_ratios():
    ok, reason, metrics = _assess_frame_fit(
        _make_image(height=120, width=200, row=70, col=90),
        target_height=160,
        min_frame_margin_px=12,
    )

    assert ok
    assert reason is None
    assert metrics["content_height_px"] == 1
    assert metrics["final_bottom_whitespace_px"] == 89
    assert metrics["bottom_whitespace_ratio"] == pytest.approx(89 / 160)
    assert metrics["vertical_fill_ratio"] == pytest.approx(1 / 160)


def test_make_vertical_compact_fallback_reduces_vertical_budget_and_preserves_horizontal():
    original = {
        "pageWidth": 2300,
        "spacingSystem": 19,
        "spacingStaff": 20,
        "scale": 110,
        "pageMarginTop": 120,
        "pageMarginBottom": 100,
        "pageMarginLeft": 90,
        "pageMarginRight": 90,
        "spacingLinear": 0.31,
        "spacingNonLinear": 0.70,
        "measureMinWidth": 22,
        "justifyVertically": True,
        "staffLineWidth": 0.20,
        "stemWidth": 0.30,
        "ledgerLineThickness": 0.30,
    }

    fallback = _make_vertical_compact_fallback(original, image_width=1050)

    assert original["spacingSystem"] == 19  # original not mutated
    assert fallback["spacingSystem"] <= 12
    assert fallback["spacingStaff"] == 18
    assert fallback["scale"] == 100
    assert fallback["pageMarginTop"] == 100
    assert fallback["pageMarginBottom"] == 80
    assert fallback["justifyVertically"] is False

    assert fallback["pageWidth"] == original["pageWidth"]
    assert fallback["spacingLinear"] == original["spacingLinear"]
    assert fallback["spacingNonLinear"] == original["spacingNonLinear"]
    assert fallback["measureMinWidth"] == original["measureMinWidth"]


def test_make_horizontal_compact_fallback_reduces_horizontal_pressure():
    original = _base_options()
    fallback = _make_horizontal_compact_fallback(
        original,
        image_width=1050,
        rejection_reason="left_clearance",
    )

    assert fallback["pageWidth"] > original["pageWidth"]
    assert fallback["pageMarginLeft"] > original["pageMarginLeft"]
    assert fallback["pageMarginRight"] >= original["pageMarginRight"]
    assert fallback["spacingLinear"] < original["spacingLinear"]
    assert fallback["spacingNonLinear"] < original["spacingNonLinear"]
    assert fallback["measureMinWidth"] < original["measureMinWidth"]


def test_make_verovio_fit_fallback_enables_retry_only_fit_options():
    original = _base_options()
    fallback = _make_verovio_fit_fallback(original, image_width=1050)

    assert fallback["adjustPageHeight"] is True
    assert fallback["adjustPageWidth"] is False
    assert fallback["shrinkToFit"] is True
    assert fallback["justifyVertically"] is False


def test_make_verovio_fit_fallback_keeps_horizontal_recovery_when_reason_is_horizontal():
    original = _base_options()
    fallback = _make_verovio_fit_fallback(
        original,
        image_width=1050,
        rejection_reason="right_clearance",
    )

    assert fallback["adjustPageHeight"] is True
    assert fallback["shrinkToFit"] is True
    assert fallback["pageWidth"] > original["pageWidth"]
    assert fallback["pageMarginRight"] > original["pageMarginRight"]


class _FakeRenderer:
    def __init__(self, scripted_results):
        self._scripted_results = list(scripted_results)
        self.calls = []

    def render_with_counts(self, renderable, render_options):
        self.calls.append((renderable, dict(render_options)))
        if not self._scripted_results:
            raise AssertionError("No scripted render results left")
        image, system_count, page_count = self._scripted_results.pop(0)
        return image, system_count, page_count


def _base_options():
    return {
        "pageWidth": 2300,
        "spacingSystem": 19,
        "spacingStaff": 20,
        "scale": 110,
        "pageMarginTop": 120,
        "pageMarginBottom": 100,
        "pageMarginLeft": 90,
        "pageMarginRight": 90,
        "spacingLinear": 0.31,
        "spacingNonLinear": 0.70,
        "measureMinWidth": 22,
        "justifyVertically": True,
        "footer": "none",
        "breaks": "auto",
        "staffLineWidth": 0.2,
        "stemWidth": 0.3,
        "ledgerLineThickness": 0.3,
    }


def test_generate_retries_on_bottom_clearance_and_succeeds(monkeypatch):
    import scripts.dataset_generation.dataset_generation.image_generation.score_generator as sg

    clipped = _make_image(height=140, width=220, row=139, col=90)
    safe = _make_image(height=140, width=220, row=90, col=90)
    renderer = _FakeRenderer([(clipped, 4, 1), (safe, 4, 1)])

    monkeypatch.setattr(sg, "generate_metadata_prefix", lambda include_title, include_author: "")
    monkeypatch.setattr(sg, "resize_to_width", lambda image, width: image)
    monkeypatch.setattr(
        sg,
        "_sample_render_options",
        lambda image_width, layout_profile="default": _base_options(),
    )

    result = generate("**kern\n4c\n*-\n", renderer, GenerationConfig(texturize_image=False, image_width=1050))

    assert result is not None
    assert result.actual_system_count == 4
    assert len(renderer.calls) == 2
    assert renderer.calls[0][1]["spacingSystem"] == 19
    assert renderer.calls[1][1]["spacingSystem"] <= 12


def test_generate_retries_with_horizontal_fallback_on_left_clearance(monkeypatch):
    import scripts.dataset_generation.dataset_generation.image_generation.score_generator as sg

    left_clipped = _make_image(height=140, width=220, row=90, col=0)
    safe = _make_image(height=140, width=220, row=90, col=60)
    renderer = _FakeRenderer([(left_clipped, 4, 1), (safe, 4, 1)])

    monkeypatch.setattr(sg, "generate_metadata_prefix", lambda include_title, include_author: "")
    monkeypatch.setattr(sg, "resize_to_width", lambda image, width: image)
    monkeypatch.setattr(
        sg,
        "_sample_render_options",
        lambda image_width, layout_profile="default": _base_options(),
    )

    result = generate("**kern\n4c\n*-\n", renderer, GenerationConfig(texturize_image=False, image_width=1050))

    assert result is not None
    assert len(renderer.calls) == 2
    assert renderer.calls[1][1]["pageWidth"] > renderer.calls[0][1]["pageWidth"]
    assert renderer.calls[1][1]["pageMarginLeft"] > renderer.calls[0][1]["pageMarginLeft"]
    assert renderer.calls[1][1]["spacingLinear"] < renderer.calls[0][1]["spacingLinear"]


def test_generate_uses_verovio_retry_fallback_on_third_attempt(monkeypatch):
    import scripts.dataset_generation.dataset_generation.image_generation.score_generator as sg

    clipped1 = _make_image(height=140, width=220, row=139, col=90)
    clipped2 = _make_image(height=140, width=220, row=138, col=90)
    safe3 = _make_image(height=140, width=220, row=90, col=90)
    renderer = _FakeRenderer([(clipped1, 4, 1), (clipped2, 4, 1), (safe3, 4, 1)])

    monkeypatch.setattr(sg, "generate_metadata_prefix", lambda include_title, include_author: "")
    monkeypatch.setattr(sg, "resize_to_width", lambda image, width: image)
    monkeypatch.setattr(
        sg,
        "_sample_render_options",
        lambda image_width, layout_profile="default": _base_options(),
    )

    result = generate("**kern\n4c\n*-\n", renderer, GenerationConfig(texturize_image=False, image_width=1050))

    assert result is not None
    assert len(renderer.calls) == 3
    assert renderer.calls[2][1]["adjustPageHeight"] is True
    assert renderer.calls[2][1]["shrinkToFit"] is True


def test_generate_does_not_retry_on_multi_page(monkeypatch):
    import scripts.dataset_generation.dataset_generation.image_generation.score_generator as sg

    renderer = _FakeRenderer([(_make_image(height=140, width=220, row=90, col=90), 4, 2)])

    monkeypatch.setattr(sg, "generate_metadata_prefix", lambda include_title, include_author: "")
    monkeypatch.setattr(sg, "resize_to_width", lambda image, width: image)
    monkeypatch.setattr(
        sg,
        "_sample_render_options",
        lambda image_width, layout_profile="default": _base_options(),
    )

    result = generate("**kern\n4c\n*-\n", renderer, GenerationConfig(texturize_image=False, image_width=1050))

    assert result is None
    assert len(renderer.calls) == 1


def test_generate_retries_on_crop_risk(monkeypatch):
    import scripts.dataset_generation.dataset_generation.image_generation.score_generator as sg

    crop_risk = _make_image(height=160, width=220, row=120, col=90)
    safe_for_crop = _make_image(height=160, width=220, row=60, col=90)
    renderer = _FakeRenderer([(crop_risk, 3, 1), (safe_for_crop, 3, 1)])

    monkeypatch.setattr(sg, "generate_metadata_prefix", lambda include_title, include_author: "")
    monkeypatch.setattr(sg, "resize_to_width", lambda image, width: image)
    monkeypatch.setattr(
        sg,
        "_sample_render_options",
        lambda image_width, layout_profile="default": _base_options(),
    )

    config = GenerationConfig(texturize_image=False, image_width=1050, image_height=100)
    result = generate("**kern\n4c\n*-\n", renderer, config)

    assert result is not None
    assert result.image.shape[0] == 100
    assert len(renderer.calls) == 2


def test_generate_with_diagnostics_returns_reason(monkeypatch):
    import scripts.dataset_generation.dataset_generation.image_generation.score_generator as sg

    renderer = _FakeRenderer([(_make_image(height=140, width=220, row=1, col=90), 4, 1)] * 3)

    monkeypatch.setattr(sg, "generate_metadata_prefix", lambda include_title, include_author: "")
    monkeypatch.setattr(sg, "resize_to_width", lambda image, width: image)
    monkeypatch.setattr(
        sg,
        "_sample_render_options",
        lambda image_width, layout_profile="default": _base_options(),
    )

    score, reason = generate_with_diagnostics(
        "**kern\n4c\n*-\n", renderer, GenerationConfig(texturize_image=False, image_width=1050)
    )
    assert score is None
    assert reason == "top_clearance"
