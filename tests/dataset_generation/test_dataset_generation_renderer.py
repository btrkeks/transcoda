import os

import numpy as np
import pytest

from scripts.dataset_generation.dataset_generation.image_generation.types import RenderedPage
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.renderer import _assess_frame_fit, render_sample


class _NoisyRenderer:
    def render_with_counts(self, transcription, render_options):
        del transcription, render_options
        os.write(
            2,
            b"Error: Inconsistent rhythm analysis occurring near line 12\n"
            b"Expected durationFromStart to be: 64 but found it to be 62\n"
            b"Line: 4G\t.\t.\t4c 4e\n",
        )
        image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
        image[30:130, 40:700] = 0
        return RenderedPage(image=image, foreground=image, alpha=np.full((1485, 1050), 255, dtype=np.uint8)), 1, 1


def test_assess_frame_fit_reports_pixel_metrics_consistent_with_ratios():
    image = np.full((120, 80, 3), 255, dtype=np.uint8)
    image[15:55, 10:40] = 0

    passed, reason, metrics = _assess_frame_fit(
        image,
        target_height=160,
        min_frame_margin_px=5,
        target_frame_margin_px=12,
    )

    assert passed is True
    assert reason is None
    assert metrics["top_whitespace_px"] == 15
    assert metrics["bottom_whitespace_px"] == 160 - 1 - 54
    assert metrics["content_height_px"] == 40
    assert metrics["bottom_whitespace_ratio"] == pytest.approx((160 - 1 - 54) / 160)
    assert metrics["vertical_fill_ratio"] == pytest.approx(40 / 160)


def test_render_sample_captures_verovio_diagnostics_without_changing_success():
    result = render_sample(
        "**kern\n=1\n4c\n*-\n",
        ProductionRecipe(),
        seed=123,
        renderer=_NoisyRenderer(),
    )

    assert result.succeeded is True
    assert result.rejection_reason is None
    assert len(result.verovio_diagnostics) == 1
    diagnostic = result.verovio_diagnostics[0]
    assert diagnostic.diagnostic_kind == "inconsistent_rhythm_analysis"
    assert diagnostic.render_attempt_idx == 1
    assert diagnostic.near_line == 12
    assert diagnostic.expected_duration_from_start == "64"
    assert diagnostic.found_duration_from_start == "62"


def test_render_sample_can_disable_verovio_diagnostic_capture():
    result = render_sample(
        "**kern\n=1\n4c\n*-\n",
        ProductionRecipe(),
        seed=123,
        renderer=_NoisyRenderer(),
        capture_verovio_diagnostics=False,
    )

    assert result.succeeded is True
    assert result.verovio_diagnostics == ()
