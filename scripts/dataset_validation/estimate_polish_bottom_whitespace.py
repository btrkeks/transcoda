#!/usr/bin/env python3
"""Estimate bottom whitespace for Polish validation scan images.

This is an analysis-grade heuristic intended for batch diagnostics, not
ground-truth page annotation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import median

import cv2
import numpy as np

DEFAULT_INPUT_DIR = Path("data/analysis/polish_validation_images")
DEFAULT_OUTPUT_CSV = Path("data/analysis/polish_validation_bottom_whitespace.csv")
DEFAULT_OVERLAY_DIR = Path("data/analysis/polish_validation_bottom_whitespace_overlays")

MIN_COMPONENT_AREA = 20
MAX_SPECKLE_HEIGHT = 2
MAX_SPECKLE_WIDTH = 20
MIN_ACTIVE_ROW_FOREGROUND = 25
MAX_INTERNAL_GAP_ROWS = 6

MUSIC_MIN_HEIGHT_PX = 45
MUSIC_MIN_FOREGROUND_PIXELS = 5000
MUSIC_MIN_SPAN_RATIO = 0.60

FOOTER_NEAR_GAP_PX = 120
FOOTER_MEANINGFUL_PIXELS = 1500
FOOTER_MEANINGFUL_SPAN_RATIO = 0.25
LOW_CANDIDATE_LOOKBACK_PX = 260
LOW_CONFIDENCE_THRESHOLD = 0.55


@dataclass(frozen=True)
class Band:
    start_row: int
    end_row: int
    height_px: int
    foreground_pixels: int
    horizontal_span_px: int
    max_row_foreground: int


@dataclass(frozen=True)
class ImageEstimate:
    filename: str
    estimated_bottom_whitespace_px: int | None
    music_band_start_row: int | None
    music_band_end_row: int | None
    music_band_height_px: int | None
    music_band_span_px: int | None
    music_band_foreground_pixels: int | None
    num_total_bands: int
    num_music_candidates: int
    confidence: float
    ambiguous: bool
    ambiguity_reason: str


CSV_FIELDNAMES = [field.name for field in fields(ImageEstimate)]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _scale_term(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 1.0 if value >= upper else 0.0
    return _clip01((value - lower) / (upper - lower))


def clean_foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Return a binary foreground mask with obvious speckles removed."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    cleaned = np.zeros_like(binary_inv)
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if int(area) < MIN_COMPONENT_AREA:
            continue
        if int(height) <= MAX_SPECKLE_HEIGHT and int(width) <= MAX_SPECKLE_WIDTH:
            continue
        cleaned[labels == label] = 255
    return cleaned


def extract_bands(mask: np.ndarray) -> list[Band]:
    """Collapse the foreground mask into horizontal activity bands."""
    row_foreground = (mask > 0).sum(axis=1)
    active_rows = row_foreground >= MIN_ACTIVE_ROW_FOREGROUND
    bands: list[Band] = []
    row_count = int(active_rows.shape[0])
    index = 0
    while index < row_count:
        if not bool(active_rows[index]):
            index += 1
            continue
        start = index
        end = index
        gap_rows = 0
        index += 1
        while index < row_count:
            if bool(active_rows[index]):
                end = index
                gap_rows = 0
            else:
                gap_rows += 1
                if gap_rows > MAX_INTERNAL_GAP_ROWS:
                    break
            index += 1
        band_slice = mask[start : end + 1] > 0
        cols = np.where(band_slice.any(axis=0))[0]
        horizontal_span = int(cols[-1] - cols[0] + 1) if cols.size else 0
        band_rows = row_foreground[start : end + 1]
        bands.append(
            Band(
                start_row=int(start),
                end_row=int(end),
                height_px=int(end - start + 1),
                foreground_pixels=int(band_slice.sum()),
                horizontal_span_px=horizontal_span,
                max_row_foreground=int(band_rows.max()) if band_rows.size else 0,
            )
        )
    return bands


def is_music_candidate(band: Band, image_width: int) -> bool:
    return (
        band.height_px >= MUSIC_MIN_HEIGHT_PX
        and band.foreground_pixels >= MUSIC_MIN_FOREGROUND_PIXELS
        and band.horizontal_span_px >= int(round(MUSIC_MIN_SPAN_RATIO * image_width))
    )


def candidate_strength(band: Band, image_width: int) -> float:
    span_target = 0.90 * image_width
    height_term = _scale_term(band.height_px, MUSIC_MIN_HEIGHT_PX, 180.0)
    pixels_term = _scale_term(band.foreground_pixels, MUSIC_MIN_FOREGROUND_PIXELS, 25000.0)
    span_term = _scale_term(band.horizontal_span_px, MUSIC_MIN_SPAN_RATIO * image_width, span_target)
    return _clip01((height_term + pixels_term + span_term) / 3.0)


def footer_band_strength(band: Band, image_width: int) -> float:
    pixels_term = _scale_term(band.foreground_pixels, 300.0, 2500.0)
    span_term = _scale_term(
        band.horizontal_span_px,
        FOOTER_MEANINGFUL_SPAN_RATIO * image_width,
        0.60 * image_width,
    )
    return max(pixels_term, span_term)


def analyze_image(image_path: str | Path) -> ImageEstimate:
    image_path = Path(image_path)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"failed to load image: {image_path}")

    image_height, image_width = gray.shape
    cleaned_mask = clean_foreground_mask(gray)
    bands = extract_bands(cleaned_mask)
    candidates = [band for band in bands if is_music_candidate(band, image_width)]
    chosen_band = candidates[-1] if candidates else None

    if chosen_band is None:
        return ImageEstimate(
            filename=image_path.name,
            estimated_bottom_whitespace_px=None,
            music_band_start_row=None,
            music_band_end_row=None,
            music_band_height_px=None,
            music_band_span_px=None,
            music_band_foreground_pixels=None,
            num_total_bands=len(bands),
            num_music_candidates=len(candidates),
            confidence=0.0,
            ambiguous=True,
            ambiguity_reason="no_candidate",
        )

    chosen_strength = candidate_strength(chosen_band, image_width)
    lower_bands = [band for band in bands if band.start_row > chosen_band.end_row]
    lower_non_candidates = [band for band in lower_bands if not is_music_candidate(band, image_width)]

    if lower_non_candidates:
        nearest_lower = min(lower_non_candidates, key=lambda band: band.start_row)
        nearest_gap = max(0, nearest_lower.start_row - chosen_band.end_row - 1)
        lower_band_distance_term = _scale_term(float(nearest_gap), 0.0, float(FOOTER_NEAR_GAP_PX))
    else:
        lower_band_distance_term = 1.0

    if lower_bands:
        strongest_footer_band = max(footer_band_strength(band, image_width) for band in lower_bands)
        footer_absence_term = 1.0 - strongest_footer_band
    else:
        footer_absence_term = 1.0

    confidence = _clip01(
        0.5 * chosen_strength + 0.2 * lower_band_distance_term + 0.3 * footer_absence_term
    )

    alternative_low_candidates = [
        band
        for band in candidates[:-1]
        if band.start_row >= image_height // 2
        and 0 <= chosen_band.start_row - band.end_row - 1 <= FOOTER_NEAR_GAP_PX
        and band.end_row >= chosen_band.start_row - LOW_CANDIDATE_LOOKBACK_PX
        and candidate_strength(band, image_width) >= max(0.55, chosen_strength - 0.15)
    ]
    nearby_footer_bands = [
        band
        for band in lower_bands
        if max(0, band.start_row - chosen_band.end_row - 1) <= FOOTER_NEAR_GAP_PX
        and (
            band.foreground_pixels >= FOOTER_MEANINGFUL_PIXELS
            or band.horizontal_span_px >= int(round(FOOTER_MEANINGFUL_SPAN_RATIO * image_width))
        )
    ]

    ambiguous = False
    ambiguity_reason = ""
    if alternative_low_candidates and chosen_strength < 0.70:
        ambiguous = True
        ambiguity_reason = "multiple_low_candidates"
    elif nearby_footer_bands:
        ambiguous = True
        ambiguity_reason = "near_footer_band"
    elif confidence < LOW_CONFIDENCE_THRESHOLD:
        ambiguous = True
        ambiguity_reason = "low_confidence"

    return ImageEstimate(
        filename=image_path.name,
        estimated_bottom_whitespace_px=int(image_height - 1 - chosen_band.end_row),
        music_band_start_row=chosen_band.start_row,
        music_band_end_row=chosen_band.end_row,
        music_band_height_px=chosen_band.height_px,
        music_band_span_px=chosen_band.horizontal_span_px,
        music_band_foreground_pixels=chosen_band.foreground_pixels,
        num_total_bands=len(bands),
        num_music_candidates=len(candidates),
        confidence=round(confidence, 4),
        ambiguous=ambiguous,
        ambiguity_reason=ambiguity_reason,
    )


def write_overlay(image_path: str | Path, estimate: ImageEstimate, output_path: str | Path) -> None:
    image_path = Path(image_path)
    output_path = Path(output_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to load image for overlay: {image_path}")

    if estimate.music_band_start_row is not None and estimate.music_band_end_row is not None:
        cv2.rectangle(
            image,
            (0, estimate.music_band_start_row),
            (image.shape[1] - 1, estimate.music_band_end_row),
            (0, 0, 255),
            3,
        )
        cv2.line(
            image,
            (0, estimate.music_band_end_row),
            (image.shape[1] - 1, estimate.music_band_end_row),
            (0, 255, 0),
            2,
        )

    whitespace_text = (
        "ws=none"
        if estimate.estimated_bottom_whitespace_px is None
        else f"ws={estimate.estimated_bottom_whitespace_px}px"
    )
    confidence_text = f"conf={estimate.confidence:.2f}"
    status_text = "ambiguous" if estimate.ambiguous else "clear"
    cv2.putText(
        image,
        f"{whitespace_text} {confidence_text} {status_text}",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255) if estimate.ambiguous else (0, 160, 0),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"failed to write overlay: {output_path}")


def estimate_directory(
    input_dir: str | Path,
    output_csv: str | Path,
    *,
    write_overlays: bool = False,
    overlay_dir: str | Path = DEFAULT_OVERLAY_DIR,
    limit: int | None = None,
) -> list[ImageEstimate]:
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    overlay_dir = Path(overlay_dir)
    image_paths = sorted(input_dir.glob("*.png"))
    if limit is not None:
        image_paths = image_paths[:limit]

    results: list[ImageEstimate] = []
    for image_path in image_paths:
        estimate = analyze_image(image_path)
        results.append(estimate)
        if write_overlays:
            write_overlay(image_path, estimate, overlay_dir / image_path.name)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    return results


def print_summary(results: list[ImageEstimate]) -> None:
    processed = len(results)
    ambiguous_results = [result for result in results if result.ambiguous]
    whitespace_values = [
        int(result.estimated_bottom_whitespace_px)
        for result in results
        if result.estimated_bottom_whitespace_px is not None
    ]

    print(f"Processed {processed} image(s)")
    ambiguous_pct = (100.0 * len(ambiguous_results) / processed) if processed else 0.0
    print(f"Ambiguous: {len(ambiguous_results)} ({ambiguous_pct:.1f}%)")
    if whitespace_values:
        print(
            "Whitespace px min/median/max: "
            f"{min(whitespace_values)}/{int(median(whitespace_values))}/{max(whitespace_values)}"
        )
    else:
        print("Whitespace px min/median/max: n/a")

    if ambiguous_results:
        print("Lowest-confidence ambiguous cases:")
        for result in sorted(ambiguous_results, key=lambda item: (item.confidence, item.filename))[:10]:
            whitespace = (
                "none" if result.estimated_bottom_whitespace_px is None else result.estimated_bottom_whitespace_px
            )
            reason = result.ambiguity_reason or "unspecified"
            print(f"  {result.filename}: conf={result.confidence:.2f}, ws={whitespace}, reason={reason}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing Polish validation PNG images.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Destination CSV path for per-image estimates.",
    )
    parser.add_argument(
        "--write-overlays",
        action="store_true",
        help="Write annotated preview overlays alongside the CSV output.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=DEFAULT_OVERLAY_DIR,
        help="Directory for overlay PNGs when --write-overlays is set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick spot-check runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = estimate_directory(
        args.input_dir,
        args.output_csv,
        write_overlays=args.write_overlays,
        overlay_dir=args.overlay_dir,
        limit=args.limit,
    )
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
