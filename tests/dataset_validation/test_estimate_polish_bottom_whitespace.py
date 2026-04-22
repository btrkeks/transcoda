from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from scripts.dataset_validation import estimate_polish_bottom_whitespace as estimator


def _draw_rect(mask: np.ndarray, top: int, bottom: int, left: int, right: int) -> None:
    mask[top : bottom + 1, left : right + 1] = 0


def test_extract_bands_merges_small_gaps() -> None:
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[10 : 21, 5:111] = 255
    mask[24 : 35, 5:111] = 255

    bands = estimator.extract_bands(mask)

    assert len(bands) == 1
    assert bands[0].start_row == 10
    assert bands[0].end_row == 34


def test_analyze_image_estimates_clear_final_system(tmp_path: Path) -> None:
    image = np.full((1485, 1050), 255, dtype=np.uint8)
    _draw_rect(image, 1000, 1120, 40, 1000)
    _draw_rect(image, 1180, 1320, 30, 1015)
    path = tmp_path / "clear.png"
    cv2.imwrite(str(path), image)

    result = estimator.analyze_image(path)

    assert result.estimated_bottom_whitespace_px == 1485 - 1 - 1320
    assert result.music_band_start_row == 1180
    assert result.ambiguous is False
    assert result.confidence >= estimator.LOW_CONFIDENCE_THRESHOLD


def test_analyze_image_ignores_tiny_footer_text(tmp_path: Path) -> None:
    image = np.full((1485, 1050), 255, dtype=np.uint8)
    _draw_rect(image, 1150, 1325, 40, 1000)
    _draw_rect(image, 1410, 1416, 470, 560)
    path = tmp_path / "tiny-footer.png"
    cv2.imwrite(str(path), image)

    result = estimator.analyze_image(path)

    assert result.estimated_bottom_whitespace_px == 1485 - 1 - 1325
    assert result.ambiguous is False


def test_analyze_image_marks_near_footer_band_ambiguous(tmp_path: Path) -> None:
    image = np.full((1485, 1050), 255, dtype=np.uint8)
    _draw_rect(image, 1160, 1310, 60, 995)
    _draw_rect(image, 1360, 1378, 350, 820)
    path = tmp_path / "near-footer.png"
    cv2.imwrite(str(path), image)

    result = estimator.analyze_image(path)

    assert result.estimated_bottom_whitespace_px == 1485 - 1 - 1310
    assert result.ambiguous is True
    assert result.ambiguity_reason == "near_footer_band"


def test_analyze_image_ignores_bottom_speckles(tmp_path: Path) -> None:
    image = np.full((1485, 1050), 255, dtype=np.uint8)
    _draw_rect(image, 1175, 1330, 50, 1005)
    image[1479:1481, 200:210] = 0
    image[1482:1484, 800:808] = 0
    path = tmp_path / "speckles.png"
    cv2.imwrite(str(path), image)

    result = estimator.analyze_image(path)

    assert result.estimated_bottom_whitespace_px == 1485 - 1 - 1330
    assert result.ambiguous is False


def test_analyze_image_reports_no_candidate(tmp_path: Path) -> None:
    image = np.full((400, 400), 255, dtype=np.uint8)
    _draw_rect(image, 300, 315, 180, 225)
    path = tmp_path / "no-candidate.png"
    cv2.imwrite(str(path), image)

    result = estimator.analyze_image(path)

    assert result.estimated_bottom_whitespace_px is None
    assert result.ambiguous is True
    assert result.ambiguity_reason == "no_candidate"


def test_analyze_image_marks_multiple_low_candidates(tmp_path: Path) -> None:
    image = np.full((1485, 1050), 255, dtype=np.uint8)
    _draw_rect(image, 1000, 1135, 60, 960)
    _draw_rect(image, 1165, 1235, 55, 945)
    _draw_rect(image, 1260, 1325, 80, 760)
    path = tmp_path / "multiple-low.png"
    cv2.imwrite(str(path), image)

    result = estimator.analyze_image(path)

    assert result.estimated_bottom_whitespace_px == 1485 - 1 - 1325
    assert result.ambiguous is True
    assert result.ambiguity_reason == "multiple_low_candidates"


def test_estimate_directory_writes_csv_and_overlay(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    image = np.full((1485, 1050), 255, dtype=np.uint8)
    _draw_rect(image, 1175, 1335, 50, 1005)
    image_path = input_dir / "sample.png"
    cv2.imwrite(str(image_path), image)

    output_csv = tmp_path / "out" / "estimates.csv"
    overlay_dir = tmp_path / "overlays"
    results = estimator.estimate_directory(
        input_dir,
        output_csv,
        write_overlays=True,
        overlay_dir=overlay_dir,
    )

    assert len(results) == 1
    assert output_csv.exists()
    assert (overlay_dir / "sample.png").exists()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["filename"] == "sample.png"


def test_real_polish_pages_produce_stable_non_null_estimates() -> None:
    root = Path(__file__).resolve().parents[2]
    sample_dir = root / "data" / "analysis" / "polish_validation_images"

    result_a = estimator.analyze_image(sample_dir / "0004.png")
    result_b = estimator.analyze_image(sample_dir / "0058.png")
    result_c = estimator.analyze_image(sample_dir / "0005.png")

    assert result_a.estimated_bottom_whitespace_px is not None
    assert result_b.estimated_bottom_whitespace_px is not None
    assert result_c.estimated_bottom_whitespace_px is not None
    assert result_a.ambiguous is False
    assert result_b.ambiguous is False
    assert result_c.ambiguous is True
    assert result_c.ambiguity_reason == "multiple_low_candidates"
