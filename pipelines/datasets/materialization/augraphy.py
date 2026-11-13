"""Augraphy-based document degradation for final dataset assembly."""

from __future__ import annotations

import os
import random

import numpy as np
import numpy.typing as npt

ImageU8 = npt.NDArray[np.uint8]


def augraphy_augment(image: ImageU8, seed: int | None = None) -> tuple[ImageU8, str]:
    """
    Apply Augraphy-based, document-style degradations to a rendered music page.

    The input is an RGB NumPy array (H, W, 3) with dtype=uint8. The output is
    normalized back to the same channel convention. Failures fall back to the
    original image so one bad Augraphy sample does not fail a dataset build.
    """
    if not isinstance(image, np.ndarray):
        print("[WARNING] augraphy_augment: Input is not a NumPy array, returning original")
        return image, "invalid_input"

    if image.dtype != np.uint8:
        print("[WARNING] augraphy_augment: Input dtype is not uint8, returning original")
        return image, "invalid_input"

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim != 3 or image.shape[2] not in (3, 4):
        print(f"[WARNING] augraphy_augment: Unexpected shape {image.shape}, returning original")
        return image, "invalid_input"

    if image.shape[2] == 4:
        image = image[:, :, :3]

    if seed is not None:
        np_state = np.random.get_state()
        py_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)
        _set_cv2_seed(seed)
    else:
        np_state = None
        py_state = None

    try:
        if np.random.random() < 0.05:
            return np.ascontiguousarray(image), "noop"

        pipeline = _get_pipeline(seed=seed)
        result = pipeline.augment(image)
        output = result["output"]

        if output.dtype != np.uint8:
            output = output.astype(np.uint8)
        if output.ndim == 3 and output.shape[2] == 4:
            output = output[:, :, :3]
        if output.ndim == 2:
            output = np.stack([output, output, output], axis=-1)
        elif output.ndim == 3 and output.shape[2] == 1:
            output = np.repeat(output, 3, axis=2)

        return np.ascontiguousarray(output), "applied"

    except Exception as e:
        print(f"[WARNING] augraphy_augment failed: {e}. Returning original image.")
        return np.ascontiguousarray(image), "error"
    finally:
        if seed is not None and np_state is not None and py_state is not None:
            np.random.set_state(np_state)
            random.setstate(py_state)


_PIPELINE = None


def _build_pipeline(*, seed: int | None):
    from augraphy import AugraphyPipeline

    ink_phase, paper_phase, post_phase = _build_phases()
    return AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        random_seed=seed,
    )


def _get_pipeline(*, seed: int | None = None):
    global _PIPELINE
    if seed is not None:
        return _build_pipeline(seed=seed)

    if _PIPELINE is not None:
        return _PIPELINE

    seed_env = os.getenv("AUGRAPHY_SEED")
    env_seed = int(seed_env) if seed_env is not None and seed_env.isdigit() else None
    _PIPELINE = _build_pipeline(seed=env_seed)
    return _PIPELINE


def _set_cv2_seed(seed: int) -> None:
    try:
        import cv2
    except Exception:
        return
    cv2.setRNGSeed(int(seed))


def _build_phases():
    from augraphy import (
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

    paper_phase = [
        BrightnessTexturize(p=0.10),
        Brightness(brightness_range=(0.5, 1.5), p=1.0),
        BleedThrough(intensity_range=(0.5, 1.0), alpha=0.2, p=0.1),
        LightingGradient(
            light_position=None,
            direction=None,
            max_brightness=235,
            min_brightness=10,
            mode="gaussian",
            linear_decay_rate=None,
            transparency=0.92,
            numba_jit=1,
            p=0.25,
        ),
    ]

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
        ShadowCast(p=0.15),
        LowInkRandomLines(
            count_range=(1, 3),
            use_consistent_lines=True,
            noise_probability=0.05,
            p=0.05,
        ),
    ]

    return ink_phase, paper_phase, post_phase
