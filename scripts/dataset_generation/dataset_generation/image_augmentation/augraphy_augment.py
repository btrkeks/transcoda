import os
import random

import numpy as np
import numpy.typing as npt
from augraphy import (
    AugraphyPipeline,
    BleedThrough,
    Brightness,
    BrightnessTexturize,
    DirtyDrum,
    DirtyRollers,
    DirtyScreen,
    InkBleed,
    InkMottling,
    Jpeg,
    LightingGradient,
    LowInkPeriodicLines,
    LowInkRandomLines,
    NoiseTexturize,
    OneOf,
    ShadowCast,
    SubtleNoise,
)

ImageU8 = npt.NDArray[np.uint8]


def augraphy_augment(image: ImageU8, seed: int | None = None) -> ImageU8:
    """
    Apply Augraphy-based, document-style degradations to a rendered music page.

    This function is called inside each worker process right after Verovio has
    produced the RGB image and before we hand it to the Hugging Face dataset
    writer. The input is an RGB NumPy array (H, W, 3) with dtype=uint8 and the
    output MUST be the same shape/dtype. Keep this function fast and robust—
    failures should fall back to the original image rather than failing the
    whole sample. (Workers stream results directly into the HF dataset.)

    Responsibilities
    ----------------
    1) No-op rate: With 5% probability, return the image unchanged.
    2) Lazy, per-process pipeline: Build an AugraphyPipeline once per worker and cache.
    3) Pipeline design: light-to-moderate artifacts typical of scanned music.
    4) Type & shape guarantees: return uint8, (H,W,3), C-contiguous.
    5) Determinism: optional AUGRAPHY_SEED.
    6) Robustness: try/except around augmentation; fallback to original.
    7) Performance: reuse cached pipeline; conservative probabilities.
    """
    # 1) Validate and normalize input
    if not isinstance(image, np.ndarray):
        print("[WARNING] augraphy_augment: Input is not a NumPy array, returning original")
        return image

    if image.dtype != np.uint8:
        print("[WARNING] augraphy_augment: Input dtype is not uint8, returning original")
        return image

    # Convert grayscale to RGB if needed
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim != 3 or image.shape[2] not in (3, 4):
        print(f"[WARNING] augraphy_augment: Unexpected shape {image.shape}, returning original")
        return image

    # Drop alpha if RGBA
    if image.shape[2] == 4:
        image = image[:, :, :3]

    if seed is not None:
        np_state = np.random.get_state()
        py_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)
    else:
        np_state = None
        py_state = None

    try:
        # 2) 5% no-op rate
        if np.random.random() < 0.05:
            return np.ascontiguousarray(image)

        # 3) Apply augmentation pipeline (robustness wrapper)
        pipeline = _get_pipeline(seed=seed)
        result = pipeline.augment(image)
        output = result["output"]

        # 4) Normalize output
        if output.dtype != np.uint8:
            output = output.astype(np.uint8)
        if output.ndim == 3 and output.shape[2] == 4:
            output = output[:, :, :3]
        if output.ndim == 2:
            output = np.stack([output, output, output], axis=-1)
        elif output.ndim == 3 and output.shape[2] == 1:
            output = np.repeat(output, 3, axis=2)

        return np.ascontiguousarray(output)

    except Exception as e:
        print(f"[WARNING] augraphy_augment failed: {e}. Returning original image.")
        return np.ascontiguousarray(image)
    finally:
        if seed is not None and np_state is not None and py_state is not None:
            np.random.set_state(np_state)
            random.setstate(py_state)


# --- Lazy pipeline cache (one per worker process) ---
_PIPELINE = None


def _build_pipeline(*, seed: int | None) -> AugraphyPipeline:
    """Build a configured AugraphyPipeline."""
    ink_phase, paper_phase, post_phase = _build_phases()
    return AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        random_seed=seed,
    )


def _get_pipeline(*, seed: int | None = None):
    """Build and cache the AugraphyPipeline once per process when unseeded."""
    global _PIPELINE
    if seed is not None:
        return _build_pipeline(seed=seed)

    if _PIPELINE is not None:
        return _PIPELINE

    # Optional determinism
    seed_env = os.getenv("AUGRAPHY_SEED")
    env_seed = int(seed_env) if seed_env is not None and seed_env.isdigit() else None
    _PIPELINE = _build_pipeline(seed=env_seed)
    return _PIPELINE


def _build_phases():
    # --- INK PHASE ---
    ink_phase = [
        InkBleed(intensity_range=(0.2, 0.5), p=0.20),
        InkMottling(p=0.10),
        OneOf(
            [
                LowInkRandomLines(use_consistent_lines=False),
                LowInkRandomLines(use_consistent_lines=True),
                LowInkPeriodicLines(use_consistent_lines=False),
                LowInkPeriodicLines(use_consistent_lines=True),
            ],
            p=0.5,
        ),
    ]

    # --- PAPER PHASE ---
    paper_phase = [
        BrightnessTexturize(p=0.10),
        Brightness(brightness_range=(0.5, 1.5), p=1.0),
        BleedThrough(intensity_range=(0.5, 1.0), alpha=0.2, p=0.1),
        LightingGradient(
            light_position=None,  # random position
            direction=None,  # random orientation (0–360)
            max_brightness=235,  # keep subtle highlights
            min_brightness=10,  # avoid crushing blacks
            mode="gaussian",  # smooth falloff
            linear_decay_rate=None,  # unused for gaussian
            transparency=0.92,  # light blend; preserves content
            numba_jit=1,
            p=0.25,
        ),
    ]

    # --- POST PHASE ---
    post_phase = [
        NoiseTexturize(sigma_range=(1, 2), p=0.60),
        SubtleNoise(subtle_range=8, p=0.33),
        DirtyScreen(n_clusters=(1, 100), value_range=(20, 250), p=0.9),
        Jpeg(quality_range=(60, 95), p=0.50),
        OneOf(
            [
                DirtyRollers(line_width_range=(2, 16), scanline_type=0),
                DirtyDrum(
                    line_width_range=(1, 6),
                    line_concentration=0.08,
                    direction=0,
                    noise_intensity=0.35,
                    ksize=(3, 3),
                    sigmaX=0,
                ),
            ],
            p=0.20,
        ),
        # Soft cast shadow from page curl/hand
        ShadowCast(p=0.15),
        LowInkRandomLines(
            count_range=(1, 3),  # very sparse to avoid erasing staff lines
            use_consistent_lines=True,  # keep width/alpha stable (thin, faint streaks)
            noise_probability=0.05,  # tiny noise so lines look natural but not blotchy
            p=0.05,
        ),
    ]

    return ink_phase, paper_phase, post_phase
