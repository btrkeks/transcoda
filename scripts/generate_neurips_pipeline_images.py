from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_generation.augmentation.articulations import ACCENT, SFORZANDO
from scripts.dataset_generation.augmentation.kern_utils import (
    find_barline_indices,
    find_note_tokens,
)
from scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment import (
    offline_augment,
)
from scripts.dataset_generation.dataset_generation.image_generation.image_post import (
    pad_or_crop_alpha_to_height,
    resize_to_width,
)
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
    VerovioRenderOptions,
)
from scripts.dataset_generation.dataset_generation.image_generation.types import RenderedPage
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.render_transcription import (
    DynamicMarkPlan,
    HairpinPlan,
    NoteSuffixPlan,
    PedalSpanPlan,
    RenderAugmentationPlan,
    materialize_render_transcription,
)

INPUT_KRN = ROOT / "duetto_I_bmv_802.krn"
OUTPUT_DIR = ROOT / "figures" / "neurips_pipeline"
TARGET_WIDTH = 1498
TARGET_HEIGHT = 364
SEED = 814


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transcription = INPUT_KRN.read_text(encoding="utf-8")
    recipe = ProductionRecipe(image_width=TARGET_WIDTH, image_height=TARGET_HEIGHT)
    renderer = VerovioRenderer()

    clean_image, _ = render_fixed_strip(renderer, transcription, render_options=clean_render_options())
    clean_image, _ = fit_rendered_page_to_centered_canvas(clean_image, None)
    render_plan = build_visible_render_augmentation_plan(transcription)
    injected_transcription = materialize_render_transcription(
        transcription,
        recipe,
        augmentation_plan=render_plan,
        source_line_indices=tuple(range(len(transcription.splitlines()))),
    )
    injected_image, injected_layers = render_fixed_strip(
        renderer,
        injected_transcription,
        render_options=clean_render_options(),
    )
    injected_image, injected_layers = fit_rendered_page_to_centered_canvas(
        injected_image,
        injected_layers,
    )

    final_augmented, geometric_image, trace = offline_augment(
        injected_image,
        render_layers=injected_layers,
        texturize_image=False,
        filename="duetto_I_bmv_802_neurips",
        variant_idx=0,
        augment_seed=SEED,
        geom_x_squeeze_prob=1.0,
        geom_x_squeeze_min_scale=0.80,
        geom_x_squeeze_max_scale=0.90,
        geom_x_squeeze_apply_in_conservative=True,
    )
    outputs = {
        "01_clean_render.png": clean_image,
        "02_notation_injections.png": injected_image,
        "03_geometric_augmentation.png": geometric_image,
        "04_augraphy_degradation.png": final_augmented,
    }
    outputs = apply_shared_content_crop(outputs)
    for filename, image in outputs.items():
        write_rgb(OUTPUT_DIR / filename, image)

    print(f"Wrote {OUTPUT_DIR}")
    first_image = next(iter(outputs.values()))
    print(f"Final image size: {first_image.shape[1]} x {first_image.shape[0]}")
    print(f"Geometry applied: {trace.final_geometry_applied}")
    print(f"Augraphy outcome: {trace.augraphy_outcome}")


def render_fixed_strip(
    renderer: VerovioRenderer,
    transcription: str,
    *,
    render_options: VerovioRenderOptions | None = None,
) -> tuple[np.ndarray, RenderedPage]:
    rendered_page, _system_count, page_count = renderer.render_with_counts(
        transcription,
        render_options=render_options or figure_render_options(),
    )
    if page_count != 1:
        raise RuntimeError(f"Expected one rendered page, got {page_count}")

    image = normalize_canvas(resize_to_width(rendered_page.image, TARGET_WIDTH), fill=255)
    foreground = normalize_canvas(resize_to_width(rendered_page.foreground, TARGET_WIDTH), fill=255)
    alpha = resize_to_width(rendered_page.alpha, TARGET_WIDTH)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    alpha = normalize_alpha_canvas(alpha.astype(np.uint8))

    return (
        np.ascontiguousarray(image),
        RenderedPage(
            image=np.ascontiguousarray(image),
            foreground=np.ascontiguousarray(foreground),
            alpha=np.ascontiguousarray(alpha),
        ),
    )


def clean_render_options() -> VerovioRenderOptions:
    options = figure_render_options()
    options.update(
        {
            "scale": 42,
            "pageWidth": 3000,
            "spacingLinear": 0.20,
            "spacingNonLinear": 0.30,
            "measureMinWidth": 6,
        }
    )
    return options


def figure_render_options() -> VerovioRenderOptions:
    return {
        "scale": 70,
        "pageWidth": 3200,
        "pageMarginLeft": 40,
        "pageMarginRight": 40,
        "pageMarginTop": 40,
        "pageMarginBottom": 40,
        "spacingStaff": 8,
        "spacingSystem": 8,
        "spacingLinear": 0.45,
        "spacingNonLinear": 0.80,
        "measureMinWidth": 20,
        "barLineWidth": 0.32,
        "beamMaxSlope": 10,
        "staffLineWidth": 0.16,
        "stemWidth": 0.24,
        "ledgerLineThickness": 0.24,
        "thickBarlineThickness": 1.1,
        "font": "Leipzig",
        "breaksNoWidow": False,
        "justifyVertically": False,
        "noJustification": False,
        "adjustPageHeight": True,
        "footer": "none",
        "breaks": "auto",
    }


def build_visible_render_augmentation_plan(transcription: str) -> RenderAugmentationPlan:
    note_positions = find_note_tokens(transcription)
    if len(note_positions) < 8:
        raise RuntimeError("Expected enough note tokens for visible augmentation")
    barline_indices = find_barline_indices(transcription)
    if len(barline_indices) < 3:
        raise RuntimeError("Expected enough barlines for pedal augmentation")

    eligible_note_line_indices = tuple(sorted({position.line_idx for position in note_positions}))
    if len(eligible_note_line_indices) < 5:
        raise RuntimeError("Expected enough note lines for hairpin augmentation")

    note_suffixes = (
        NoteSuffixPlan(note_positions[3].line_idx, note_positions[3].col_idx, ACCENT),
        NoteSuffixPlan(note_positions[7].line_idx, note_positions[7].col_idx, SFORZANDO),
        NoteSuffixPlan(note_positions[-6].line_idx, note_positions[-6].col_idx, ACCENT),
    )
    pedal_spans = (
        PedalSpanPlan(barline_indices[0], barline_indices[1]),
        PedalSpanPlan(barline_indices[2], barline_indices[3]),
    )
    hairpins = (
        HairpinPlan(
            origin_line_indices=eligible_note_line_indices[1:5],
            is_crescendo=True,
        ),
    )
    dynamic_marks = (
        DynamicMarkPlan(
            origin_line_idx=eligible_note_line_indices[-3],
            token="f",
        ),
    )

    return RenderAugmentationPlan(
        note_suffixes=note_suffixes,
        pedal_spans=pedal_spans,
        include_instrument_label=True,
        tempo=None,
        hairpins=hairpins,
        dynamic_marks=dynamic_marks,
    )


def normalize_canvas(image: np.ndarray, *, fill: int) -> np.ndarray:
    current_height = image.shape[0]
    if current_height == TARGET_HEIGHT:
        return image
    if current_height > TARGET_HEIGHT:
        top = max(0, (current_height - TARGET_HEIGHT) // 2)
        return image[top : top + TARGET_HEIGHT]

    pad_top = (TARGET_HEIGHT - current_height) // 2
    pad_bottom = TARGET_HEIGHT - current_height - pad_top
    padding_top = np.full((pad_top, image.shape[1], image.shape[2]), fill, dtype=image.dtype)
    padding_bottom = np.full((pad_bottom, image.shape[1], image.shape[2]), fill, dtype=image.dtype)
    return np.concatenate([padding_top, image, padding_bottom], axis=0)


def normalize_alpha_canvas(alpha: np.ndarray) -> np.ndarray:
    current_height = alpha.shape[0]
    if current_height == TARGET_HEIGHT:
        return alpha
    if current_height > TARGET_HEIGHT:
        top = max(0, (current_height - TARGET_HEIGHT) // 2)
        return alpha[top : top + TARGET_HEIGHT]
    pad_top = (TARGET_HEIGHT - current_height) // 2
    shifted = np.pad(alpha, ((pad_top, 0), (0, 0)), mode="constant", constant_values=0)
    return pad_or_crop_alpha_to_height(shifted, TARGET_HEIGHT)


def fit_rendered_page_to_centered_canvas(
    image: np.ndarray,
    render_layers: RenderedPage | None,
    *,
    content_margin_px: int = 14,
    canvas_margin_x_px: int = 56,
    canvas_margin_y_px: int = 42,
) -> tuple[np.ndarray, RenderedPage | None]:
    mask = np.min(image[:, :, :3], axis=2) < 250
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image, render_layers

    top = max(0, int(coords[:, 0].min()) - content_margin_px)
    bottom = min(image.shape[0] - 1, int(coords[:, 0].max()) + content_margin_px)
    left = max(0, int(coords[:, 1].min()) - content_margin_px)
    right = min(image.shape[1] - 1, int(coords[:, 1].max()) + content_margin_px)

    available_width = TARGET_WIDTH - 2 * canvas_margin_x_px
    available_height = TARGET_HEIGHT - 2 * canvas_margin_y_px
    crop_width = right - left + 1
    crop_height = bottom - top + 1
    scale = min(available_width / crop_width, available_height / crop_height)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    paste_x = (TARGET_WIDTH - resized_width) // 2
    paste_y = (TARGET_HEIGHT - resized_height) // 2

    fitted_image = _fit_image_with_box(
        image,
        box=(top, bottom, left, right),
        size=(resized_width, resized_height),
        offset=(paste_x, paste_y),
        fill=255,
    )
    if render_layers is None:
        return fitted_image, None

    fitted_foreground = _fit_image_with_box(
        render_layers.foreground,
        box=(top, bottom, left, right),
        size=(resized_width, resized_height),
        offset=(paste_x, paste_y),
        fill=255,
    )
    fitted_alpha = _fit_alpha_with_box(
        render_layers.alpha,
        box=(top, bottom, left, right),
        size=(resized_width, resized_height),
        offset=(paste_x, paste_y),
    )
    return (
        fitted_image,
        RenderedPage(
            image=fitted_image,
            foreground=fitted_foreground,
            alpha=fitted_alpha,
        ),
    )


def _fit_image_with_box(
    image: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    offset: tuple[int, int],
    fill: int,
) -> np.ndarray:
    top, bottom, left, right = box
    resized_width, resized_height = size
    paste_x, paste_y = offset
    crop = image[top : bottom + 1, left : right + 1]
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((TARGET_HEIGHT, TARGET_WIDTH, 3), fill, dtype=image.dtype)
    canvas[paste_y : paste_y + resized_height, paste_x : paste_x + resized_width] = resized
    return np.ascontiguousarray(canvas)


def _fit_alpha_with_box(
    alpha: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    offset: tuple[int, int],
) -> np.ndarray:
    top, bottom, left, right = box
    resized_width, resized_height = size
    paste_x, paste_y = offset
    crop = alpha[top : bottom + 1, left : right + 1]
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=alpha.dtype)
    canvas[paste_y : paste_y + resized_height, paste_x : paste_x + resized_width] = resized
    return np.ascontiguousarray(canvas)


def apply_shared_content_crop(
    images: dict[str, np.ndarray],
    *,
    margin_px: int = 42,
) -> dict[str, np.ndarray]:
    crop_sources = [
        images["01_clean_render.png"],
        images["02_notation_injections.png"],
        images["03_geometric_augmentation.png"],
    ]
    boxes = [_content_bbox(image) for image in crop_sources]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return images

    top = max(0, min(box[0] for box in boxes) - margin_px)
    bottom = min(TARGET_HEIGHT - 1, max(box[1] for box in boxes) + margin_px)
    left = max(0, min(box[2] for box in boxes) - margin_px)
    right = min(TARGET_WIDTH - 1, max(box[3] for box in boxes) + margin_px)

    return {
        filename: np.ascontiguousarray(image[top : bottom + 1, left : right + 1])
        for filename, image in images.items()
    }


def _content_bbox(image: np.ndarray, *, threshold: int = 235) -> tuple[int, int, int, int] | None:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    coords = np.argwhere(gray < threshold)
    if coords.size == 0:
        return None
    return (
        int(coords[:, 0].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].min()),
        int(coords[:, 1].max()),
    )


def write_rgb(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image for {path}, got {image.shape}")
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write {path}")


if __name__ == "__main__":
    main()
