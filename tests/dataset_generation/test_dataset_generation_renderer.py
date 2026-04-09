import numpy as np
import pytest

from scripts.dataset_generation.dataset_generation.renderer import _assess_frame_fit


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
