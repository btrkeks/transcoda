"""
Worker process functions for parallel synthetic data generation.

This module contains functions executed by multiprocessing worker processes.
It is imported by FileDataGenerator and should not be used directly.
"""

import os
import re
import sys
import time
import math
from io import BytesIO
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from PIL import Image

from scripts.dataset_generation.augmentation.courtesy_naturals import apply_courtesy_naturals
from scripts.dataset_generation.augmentation.articulations import apply_accents, apply_sforzandos
from scripts.dataset_generation.augmentation.hairpins import apply_render_hairpins
from scripts.dataset_generation.augmentation.instrument_label import apply_instrument_label_piano
from scripts.dataset_generation.augmentation.pedaling import apply_pedaling
from scripts.dataset_generation.augmentation.render_dynamic_marks import apply_render_dynamic_marks
from scripts.dataset_generation.augmentation.tempo_markings import apply_tempo_markings
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
)
from scripts.dataset_generation.dataset_generation.image_generation.score_generator import (
    generate_with_diagnostics as generate_score_with_diagnostics,
)
from scripts.dataset_generation.dataset_generation.image_generation.types import (
    GenerationConfig,
    RenderedPage,
)
from scripts.dataset_generation.dataset_generation.resumable_dataset import derive_sample_seed
from scripts.dataset_generation.dataset_generation.truncation import (
    PrefixTruncationCandidate,
    build_prefix_truncation_space,
)
from scripts.dataset_generation.dataset_generation.worker_models import (
    PROFILE_STAGE_NAMES,
    SampleFailure,
    SampleOutcome,
    SampleProfile,
    SampleSuccess,
    WorkerInitConfig,
    outcome_to_legacy_tuple,
)

# --- Module-level globals for worker processes ---
_image_width: int | None = None
_image_height: int | None = None
_file_renderer: VerovioRenderer | None = None
_augment_seed: int | None = None
_render_pedals_enabled: bool = True
_render_pedals_probability: float = 0.20
_render_pedals_measures_probability: float = 0.3
_render_instrument_piano_enabled: bool = True
_render_instrument_piano_probability: float = 0.15
_render_sforzando_enabled: bool = True
_render_sforzando_probability: float = 0.20
_render_sforzando_per_note_probability: float = 0.03
_render_accent_enabled: bool = True
_render_accent_probability: float = 0.10
_render_accent_per_note_probability: float = 0.015
_render_tempo_enabled: bool = True
_render_tempo_probability: float = 0.12
_render_tempo_include_mm_probability: float = 0.35
_render_hairpins_enabled: bool = True
_render_hairpins_probability: float = 0.25
_render_hairpins_max_spans: int = 2
_render_dynamic_marks_enabled: bool = True
_render_dynamic_marks_probability: float = 0.15
_render_dynamic_marks_min_count: int = 1
_render_dynamic_marks_max_count: int = 2
_courtesy_naturals_probability: float = 0.15
_disable_offline_image_augmentations: bool = False
_geom_x_squeeze_prob: float = 0.45
_geom_x_squeeze_min_scale: float = 0.70
_geom_x_squeeze_max_scale: float = 0.95
_geom_x_squeeze_apply_in_conservative: bool = True
_geom_x_squeeze_preview_force_scale: float | None = None
_target_min_systems: int | None = None
_target_max_systems: int | None = None
_render_layout_profile: str = "default"
_overflow_truncation_enabled: bool = True
_overflow_truncation_max_trials: int = 24
_profile_enabled: bool = False
_SPARSE_RENDER_BLACK_RATIO_MIN = 0.005
_worker_config: WorkerInitConfig | None = None


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def encode_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    """
    Encode a HxWxC NumPy RGB or grayscale image into JPEG bytes.
    """
    if img.dtype != np.uint8:
        raise ValueError("encode_jpeg expects uint8 image data")

    # Handle grayscale vs RGB automatically
    if img.ndim == 2:
        pil_img = Image.fromarray(img, mode="L")
    elif img.ndim == 3 and img.shape[2] == 3:
        pil_img = Image.fromarray(img, mode="RGB")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    buf = BytesIO()
    pil_img.save(
        buf,
        format="JPEG",
        quality=quality,
        optimize=False,  # Faster encoding, marginal size increase
        progressive=True,
    )
    return buf.getvalue()


def _black_ratio_le_threshold(image: np.ndarray, threshold: float = 120.0) -> float:
    """Return fraction of pixels whose grayscale luma is <= threshold."""
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    gray = image[:, :, :3].mean(axis=2)
    return float((gray <= threshold).mean())


def _is_sparse_render(
    image: np.ndarray,
    *,
    min_black_ratio: float = _SPARSE_RENDER_BLACK_RATIO_MIN,
) -> tuple[bool, float]:
    """Detect obviously degenerate renders (e.g. barline-only outputs)."""
    black_ratio = _black_ratio_le_threshold(image, threshold=120.0)
    return black_ratio < min_black_ratio, black_ratio


# Pattern to detect rational durations (e.g., "1%143" or "24%-1")
# These indicate malformed MusicXML source data with corrupted durations
_RATIONAL_DURATION_PATTERN = re.compile(r"\d%-?\d")


def is_valid_kern(content: str) -> tuple[bool, str | None]:
    """Validate kern content for known malformation patterns.

    Args:
        content: The kern file content.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if _RATIONAL_DURATION_PATTERN.search(content):
        return False, "rational duration (corrupted source)"
    return True, None


def ensure_render_header(
    content: str,
    *,
    last_spine_type: str | None = None,
) -> str:
    """Ensure render content has an exclusive interpretation header.

    The normalization pipeline strips **kern headers, but Verovio requires them.
    This function prepends the header if missing.

    Args:
        content: The kern file content.
        last_spine_type: Optional type for an appended trailing spine.
            Currently supports ``"dynam"``.

    Returns:
        Content with the required header prepended if it was missing.
    """
    first_data_like_line: str | None = None
    for line in content.splitlines():
        if not line.strip():
            continue
        if line.startswith("!!"):
            continue
        first_data_like_line = line
        break

    if first_data_like_line is None:
        return content
    if first_data_like_line.startswith("**"):
        return content

    # Count spines from the first line (tabs + 1)
    num_spines = first_data_like_line.count("\t") + 1

    if last_spine_type == "dynam":
        if num_spines < 2:
            raise ValueError("Cannot build mixed header with fewer than 2 spines")
        header_tokens = ["**kern"] * (num_spines - 1) + ["**dynam"]
    else:
        header_tokens = ["**kern"] * num_spines
    header = "\t".join(header_tokens)
    return header + "\n" + content


def ensure_kern_header(content: str) -> str:
    """Backward-compatible wrapper for existing call sites."""
    return ensure_render_header(content)


def init_file_worker(
    image_width: int | WorkerInitConfig,
    image_height: int | None = None,
    augment_seed: int | None = None,
    render_pedals_enabled: bool = True,
    render_pedals_probability: float = 0.20,
    render_pedals_measures_probability: float = 0.3,
    render_instrument_piano_enabled: bool = True,
    render_instrument_piano_probability: float = 0.15,
    render_sforzando_enabled: bool = True,
    render_sforzando_probability: float = 0.20,
    render_sforzando_per_note_probability: float = 0.03,
    render_accent_enabled: bool = True,
    render_accent_probability: float = 0.10,
    render_accent_per_note_probability: float = 0.015,
    render_tempo_enabled: bool = True,
    render_tempo_probability: float = 0.12,
    render_tempo_include_mm_probability: float = 0.35,
    render_hairpins_enabled: bool = True,
    render_hairpins_probability: float = 0.25,
    render_hairpins_max_spans: int = 2,
    render_dynamic_marks_enabled: bool = True,
    render_dynamic_marks_probability: float = 0.15,
    render_dynamic_marks_min_count: int = 1,
    render_dynamic_marks_max_count: int = 2,
    courtesy_naturals_probability: float = 0.15,
    disable_offline_image_augmentations: bool = False,
    geom_x_squeeze_prob: float = 0.45,
    geom_x_squeeze_min_scale: float = 0.70,
    geom_x_squeeze_max_scale: float = 0.95,
    geom_x_squeeze_apply_in_conservative: bool = True,
    geom_x_squeeze_preview_force_scale: float | None = None,
    target_min_systems: int | None = None,
    target_max_systems: int | None = None,
    render_layout_profile: str = "default",
    overflow_truncation_enabled: bool = True,
    overflow_truncation_max_trials: int = 24,
) -> None:
    """Initialize worker for file-based generation.

    Pre-allocates the VerovioRenderer once per worker to reduce memory pressure.
    Each task still gets a different kern file but reuses this expensive object.

    Args:
        image_width: Target image width in pixels.
        image_height: Target image height in pixels, or None for content-determined height.
        augment_seed: Optional seed for deterministic augmentation.
        render_pedals_enabled: Whether to inject pedal markings for rendering only.
        render_pedals_probability: Per-sample probability of applying
            pedal augmentation before the per-measure pedal probability.
        render_pedals_measures_probability: Per-measure probability of inserting pedal
            markings for rendering only.
        render_instrument_piano_enabled: Whether to inject *I"Piano labels for
            rendering only.
        render_instrument_piano_probability: Per-sample probability of inserting
            *I"Piano labels for rendering only.
        render_sforzando_enabled: Whether to inject note-level z articulations
            for rendering only.
        render_sforzando_probability: Per-sample probability of inserting
            note-level z articulations.
        render_sforzando_per_note_probability: Per-note probability for z
            articulation insertion when enabled for a sample.
        render_accent_enabled: Whether to inject note-level ^ accents for
            rendering only.
        render_accent_probability: Per-sample probability of inserting
            note-level ^ accents.
        render_accent_per_note_probability: Per-note probability for ^
            accent insertion when enabled for a sample.
        render_tempo_enabled: Whether to inject OMD/*MM tempo metadata for
            rendering only.
        render_tempo_probability: Per-sample probability of applying
            render-only tempo augmentation.
        render_tempo_include_mm_probability: Conditional probability of adding
            numeric *MM after OMD is selected.
        render_hairpins_enabled: Whether to inject temporary trailing ``**dynam``
            hairpins for rendering only.
        render_hairpins_probability: Per-sample probability of inserting
            render-only hairpin spans.
        render_hairpins_max_spans: Maximum number of non-overlapping hairpin spans
            to sample per score.
        render_dynamic_marks_enabled: Whether to inject canonical terraced
            dynamics in a trailing ``**dynam`` spine for rendering only.
        render_dynamic_marks_probability: Per-sample probability of inserting
            render-only terraced dynamic marks.
        render_dynamic_marks_min_count: Minimum dynamic marks to inject when
            augmentation is applied.
        render_dynamic_marks_max_count: Maximum dynamic marks to inject when
            augmentation is applied.
        disable_offline_image_augmentations: Skip raster-stage geometric OpenCV
            augmentation and Augraphy document artifacts.
        geom_x_squeeze_prob: Probability of applying explicit horizontal squeeze.
        geom_x_squeeze_min_scale: Minimum horizontal squeeze scale.
        geom_x_squeeze_max_scale: Maximum horizontal squeeze scale.
        geom_x_squeeze_apply_in_conservative: Whether squeeze may apply on conservative retry.
        geom_x_squeeze_preview_force_scale: Optional forced squeeze scale for debugging.
        overflow_truncation_enabled: Whether to attempt truncation rescue for multi-page renders.
        overflow_truncation_max_trials: Max prefix truncation candidates to try per sample.
    """
    if isinstance(image_width, WorkerInitConfig):
        worker_config = image_width
    else:
        worker_config = WorkerInitConfig(
            image_width=image_width,
            image_height=image_height,
            augment_seed=augment_seed,
            deterministic_seed_salt=None,
            render_pedals_enabled=render_pedals_enabled,
            render_pedals_probability=render_pedals_probability,
            render_pedals_measures_probability=render_pedals_measures_probability,
            render_instrument_piano_enabled=render_instrument_piano_enabled,
            render_instrument_piano_probability=render_instrument_piano_probability,
            render_sforzando_enabled=render_sforzando_enabled,
            render_sforzando_probability=render_sforzando_probability,
            render_sforzando_per_note_probability=render_sforzando_per_note_probability,
            render_accent_enabled=render_accent_enabled,
            render_accent_probability=render_accent_probability,
            render_accent_per_note_probability=render_accent_per_note_probability,
            render_tempo_enabled=render_tempo_enabled,
            render_tempo_probability=render_tempo_probability,
            render_tempo_include_mm_probability=render_tempo_include_mm_probability,
            render_hairpins_enabled=render_hairpins_enabled,
            render_hairpins_probability=render_hairpins_probability,
            render_hairpins_max_spans=render_hairpins_max_spans,
            render_dynamic_marks_enabled=render_dynamic_marks_enabled,
            render_dynamic_marks_probability=render_dynamic_marks_probability,
            render_dynamic_marks_min_count=render_dynamic_marks_min_count,
            render_dynamic_marks_max_count=render_dynamic_marks_max_count,
            courtesy_naturals_probability=courtesy_naturals_probability,
            disable_offline_image_augmentations=disable_offline_image_augmentations,
            geom_x_squeeze_prob=geom_x_squeeze_prob,
            geom_x_squeeze_min_scale=geom_x_squeeze_min_scale,
            geom_x_squeeze_max_scale=geom_x_squeeze_max_scale,
            geom_x_squeeze_apply_in_conservative=geom_x_squeeze_apply_in_conservative,
            geom_x_squeeze_preview_force_scale=geom_x_squeeze_preview_force_scale,
            target_min_systems=target_min_systems,
            target_max_systems=target_max_systems,
            render_layout_profile=render_layout_profile,
            overflow_truncation_enabled=overflow_truncation_enabled,
            overflow_truncation_max_trials=overflow_truncation_max_trials,
        )
    worker_config.validate()

    global _image_width, _image_height, _file_renderer, _augment_seed
    global _worker_config
    global _render_pedals_enabled, _render_pedals_probability, _render_pedals_measures_probability
    global _render_instrument_piano_enabled, _render_instrument_piano_probability
    global _render_sforzando_enabled, _render_sforzando_probability
    global _render_sforzando_per_note_probability
    global _render_accent_enabled, _render_accent_probability
    global _render_accent_per_note_probability
    global _render_tempo_enabled, _render_tempo_probability, _render_tempo_include_mm_probability
    global _render_hairpins_enabled, _render_hairpins_probability, _render_hairpins_max_spans
    global _render_dynamic_marks_enabled, _render_dynamic_marks_probability
    global _render_dynamic_marks_min_count, _render_dynamic_marks_max_count
    global _courtesy_naturals_probability
    global _disable_offline_image_augmentations
    global _geom_x_squeeze_prob, _geom_x_squeeze_min_scale, _geom_x_squeeze_max_scale
    global _geom_x_squeeze_apply_in_conservative, _geom_x_squeeze_preview_force_scale
    global _target_min_systems, _target_max_systems, _render_layout_profile
    global _overflow_truncation_enabled, _overflow_truncation_max_trials
    global _profile_enabled
    _worker_config = worker_config
    _image_width = worker_config.image_width
    _image_height = worker_config.image_height
    _augment_seed = worker_config.augment_seed
    _render_pedals_enabled = worker_config.render_pedals_enabled
    _render_pedals_probability = worker_config.render_pedals_probability
    _render_pedals_measures_probability = worker_config.render_pedals_measures_probability
    _render_instrument_piano_enabled = worker_config.render_instrument_piano_enabled
    _render_instrument_piano_probability = worker_config.render_instrument_piano_probability
    _render_sforzando_enabled = worker_config.render_sforzando_enabled
    _render_sforzando_probability = worker_config.render_sforzando_probability
    _render_sforzando_per_note_probability = worker_config.render_sforzando_per_note_probability
    _render_accent_enabled = worker_config.render_accent_enabled
    _render_accent_probability = worker_config.render_accent_probability
    _render_accent_per_note_probability = worker_config.render_accent_per_note_probability
    _render_tempo_enabled = worker_config.render_tempo_enabled
    _render_tempo_probability = worker_config.render_tempo_probability
    _render_tempo_include_mm_probability = worker_config.render_tempo_include_mm_probability
    _render_hairpins_enabled = worker_config.render_hairpins_enabled
    _render_hairpins_probability = worker_config.render_hairpins_probability
    _render_hairpins_max_spans = worker_config.render_hairpins_max_spans
    _render_dynamic_marks_enabled = worker_config.render_dynamic_marks_enabled
    _render_dynamic_marks_probability = worker_config.render_dynamic_marks_probability
    _render_dynamic_marks_min_count = worker_config.render_dynamic_marks_min_count
    _render_dynamic_marks_max_count = worker_config.render_dynamic_marks_max_count
    _courtesy_naturals_probability = worker_config.courtesy_naturals_probability
    _disable_offline_image_augmentations = worker_config.disable_offline_image_augmentations
    _geom_x_squeeze_prob = worker_config.geom_x_squeeze_prob
    _geom_x_squeeze_min_scale = worker_config.geom_x_squeeze_min_scale
    _geom_x_squeeze_max_scale = worker_config.geom_x_squeeze_max_scale
    _geom_x_squeeze_apply_in_conservative = worker_config.geom_x_squeeze_apply_in_conservative
    _geom_x_squeeze_preview_force_scale = worker_config.geom_x_squeeze_preview_force_scale
    _target_min_systems = worker_config.target_min_systems
    _target_max_systems = worker_config.target_max_systems
    _render_layout_profile = worker_config.render_layout_profile
    _overflow_truncation_enabled = worker_config.overflow_truncation_enabled
    _overflow_truncation_max_trials = worker_config.overflow_truncation_max_trials
    _profile_enabled = worker_config.profile_enabled
    _file_renderer = VerovioRenderer()
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))


def _capture_stderr_fd():
    """Context manager to capture stderr at the file descriptor level.

    This captures stderr from C/C++ libraries (like Verovio) that write
    directly to the file descriptor, not through Python's sys.stderr.
    """
    import os
    from contextlib import contextmanager

    @contextmanager
    def capture():
        # Save original stderr fd
        stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fd)

        # Create a pipe to capture stderr
        read_fd, write_fd = os.pipe()

        try:
            # Redirect stderr to our pipe
            os.dup2(write_fd, stderr_fd)
            os.close(write_fd)

            captured = []

            yield captured

            # Flush stderr to ensure all output is captured
            sys.stderr.flush()

            # Restore original stderr
            os.dup2(saved_stderr_fd, stderr_fd)

            # Read captured output
            os.set_blocking(read_fd, False)
            try:
                while True:
                    data = os.read(read_fd, 4096)
                    if not data:
                        break
                    captured.append(data.decode("utf-8", errors="replace"))
            except BlockingIOError:
                pass

        finally:
            os.close(saved_stderr_fd)
            os.close(read_fd)

    return capture()


_RENDER_FIT_REJECTION_REASONS = {
    "top_clearance",
    "bottom_clearance",
    "left_clearance",
    "right_clearance",
    "crop_risk",
    "no_content_detected",
}


def _current_worker_config() -> WorkerInitConfig:
    global _image_width, _image_height, _augment_seed
    global _render_pedals_enabled, _render_pedals_probability, _render_pedals_measures_probability
    global _render_instrument_piano_enabled, _render_instrument_piano_probability
    global _render_sforzando_enabled, _render_sforzando_probability
    global _render_sforzando_per_note_probability
    global _render_accent_enabled, _render_accent_probability
    global _render_accent_per_note_probability
    global _render_tempo_enabled, _render_tempo_probability, _render_tempo_include_mm_probability
    global _render_hairpins_enabled, _render_hairpins_probability, _render_hairpins_max_spans
    global _render_dynamic_marks_enabled, _render_dynamic_marks_probability
    global _render_dynamic_marks_min_count, _render_dynamic_marks_max_count
    global _courtesy_naturals_probability
    global _disable_offline_image_augmentations
    global _geom_x_squeeze_prob, _geom_x_squeeze_min_scale, _geom_x_squeeze_max_scale
    global _geom_x_squeeze_apply_in_conservative, _geom_x_squeeze_preview_force_scale
    global _target_min_systems, _target_max_systems, _render_layout_profile
    global _overflow_truncation_enabled, _overflow_truncation_max_trials
    global _profile_enabled
    assert _image_width is not None, "Worker not initialized"

    config = WorkerInitConfig(
        image_width=_image_width,
        image_height=_image_height,
        augment_seed=_augment_seed,
        deterministic_seed_salt=(
            _worker_config.deterministic_seed_salt if _worker_config is not None else None
        ),
        render_pedals_enabled=_render_pedals_enabled,
        render_pedals_probability=_render_pedals_probability,
        render_pedals_measures_probability=_render_pedals_measures_probability,
        render_instrument_piano_enabled=_render_instrument_piano_enabled,
        render_instrument_piano_probability=_render_instrument_piano_probability,
        render_sforzando_enabled=_render_sforzando_enabled,
        render_sforzando_probability=_render_sforzando_probability,
        render_sforzando_per_note_probability=_render_sforzando_per_note_probability,
        render_accent_enabled=_render_accent_enabled,
        render_accent_probability=_render_accent_probability,
        render_accent_per_note_probability=_render_accent_per_note_probability,
        render_tempo_enabled=_render_tempo_enabled,
        render_tempo_probability=_render_tempo_probability,
        render_tempo_include_mm_probability=_render_tempo_include_mm_probability,
        render_hairpins_enabled=_render_hairpins_enabled,
        render_hairpins_probability=_render_hairpins_probability,
        render_hairpins_max_spans=_render_hairpins_max_spans,
        render_dynamic_marks_enabled=_render_dynamic_marks_enabled,
        render_dynamic_marks_probability=_render_dynamic_marks_probability,
        render_dynamic_marks_min_count=_render_dynamic_marks_min_count,
        render_dynamic_marks_max_count=_render_dynamic_marks_max_count,
        courtesy_naturals_probability=_courtesy_naturals_probability,
        disable_offline_image_augmentations=_disable_offline_image_augmentations,
        geom_x_squeeze_prob=_geom_x_squeeze_prob,
        geom_x_squeeze_min_scale=_geom_x_squeeze_min_scale,
        geom_x_squeeze_max_scale=_geom_x_squeeze_max_scale,
        geom_x_squeeze_apply_in_conservative=_geom_x_squeeze_apply_in_conservative,
        geom_x_squeeze_preview_force_scale=_geom_x_squeeze_preview_force_scale,
        target_min_systems=_target_min_systems,
        target_max_systems=_target_max_systems,
        render_layout_profile=_render_layout_profile,
        overflow_truncation_enabled=_overflow_truncation_enabled,
        overflow_truncation_max_trials=_overflow_truncation_max_trials,
        profile_enabled=_profile_enabled,
    )
    config.validate()
    return config


def _build_render_transcription(kern_content: str, config: WorkerInitConfig) -> str:
    """Build render-only transcription from source kern text."""
    render_transcription = kern_content
    if (
        config.render_pedals_enabled
        and config.render_pedals_probability > 0.0
        and config.render_pedals_measures_probability > 0.0
        and float(np.random.random()) < config.render_pedals_probability
    ):
        render_transcription = apply_pedaling(
            render_transcription,
            measures_probability=config.render_pedals_measures_probability,
        )
    if (
        config.render_instrument_piano_enabled
        and config.render_instrument_piano_probability > 0.0
        and float(np.random.random()) < config.render_instrument_piano_probability
    ):
        render_transcription = apply_instrument_label_piano(render_transcription)
    if (
        config.render_sforzando_enabled
        and config.render_sforzando_probability > 0.0
        and float(np.random.random()) < config.render_sforzando_probability
    ):
        render_transcription = apply_sforzandos(
            render_transcription,
            per_note_probability=config.render_sforzando_per_note_probability,
        )
    if (
        config.render_accent_enabled
        and config.render_accent_probability > 0.0
        and float(np.random.random()) < config.render_accent_probability
    ):
        render_transcription = apply_accents(
            render_transcription,
            per_note_probability=config.render_accent_per_note_probability,
        )
    if (
        config.render_tempo_enabled
        and config.render_tempo_probability > 0.0
        and float(np.random.random()) < config.render_tempo_probability
    ):
        render_transcription = apply_tempo_markings(
            render_transcription,
            include_mm_probability=config.render_tempo_include_mm_probability,
        )

    hairpins_added = False
    if config.render_hairpins_enabled and config.render_hairpins_probability > 0.0:
        hairpin_augmented = apply_render_hairpins(
            render_transcription,
            sample_probability=config.render_hairpins_probability,
            max_spans=config.render_hairpins_max_spans,
        )
        hairpins_added = hairpin_augmented != render_transcription
        render_transcription = hairpin_augmented

    dynamic_marks_added = False
    if config.render_dynamic_marks_enabled and config.render_dynamic_marks_probability > 0.0:
        dynamic_marks_augmented = apply_render_dynamic_marks(
            render_transcription,
            sample_probability=config.render_dynamic_marks_probability,
            min_marks=config.render_dynamic_marks_min_count,
            max_marks=config.render_dynamic_marks_max_count,
            assume_trailing_dynam=hairpins_added,
        )
        dynamic_marks_added = dynamic_marks_augmented != render_transcription
        render_transcription = dynamic_marks_augmented

    return ensure_render_header(
        render_transcription,
        last_spine_type="dynam" if (hairpins_added or dynamic_marks_added) else None,
    )


def _build_label_transcription(kern_content: str, config: WorkerInitConfig) -> str:
    """Build paired label transcription before render-only augmentation."""
    if config.courtesy_naturals_probability <= 0.0:
        return kern_content
    if float(np.random.random()) >= config.courtesy_naturals_probability:
        return kern_content
    return apply_courtesy_naturals(kern_content)


def _render_kern_for_sample(
    kern_for_verovio: str,
    *,
    filename: str,
    config: WorkerInitConfig,
) -> tuple[
    np.ndarray | None,
    RenderedPage | None,
    int | None,
    float | None,
    float | None,
    SampleFailure | None,
]:
    """Render kern into image and map diagnostics to structured failures."""
    assert _file_renderer is not None, "Worker not initialized"

    render_config = GenerationConfig(
        num_systems_hint=1,
        include_author=bool(np.random.random() > 0.5),
        include_title=bool(np.random.random() > 0.5),
        image_width=config.image_width,
        image_height=config.image_height,
        render_layout_profile=config.render_layout_profile,
    )

    with _capture_stderr_fd() as captured_stderr:
        result, rejection_reason = generate_score_with_diagnostics(
            kern_for_verovio, _file_renderer, render_config
        )

    stderr_output = "".join(captured_stderr).strip()
    if stderr_output and "Error:" in stderr_output:
        print(f"[Verovio error in {filename}]\n{stderr_output}", file=sys.stderr)

    if result is None:
        if rejection_reason == "multi_page":
            return None, None, None, None, None, SampleFailure(code="multi_page", filename=filename)
        if rejection_reason in _RENDER_FIT_REJECTION_REASONS:
            return None, None, None, None, None, SampleFailure(
                code="render_fit",
                filename=filename,
                detail=str(rejection_reason),
            )
        return None, None, None, None, None, SampleFailure(
            code="render_rejected",
            filename=filename,
            detail=str(rejection_reason) if rejection_reason is not None else None,
        )

    return (
        result.image,
        result.render_layers,
        int(result.actual_system_count),
        result.bottom_whitespace_ratio,
        result.vertical_fill_ratio,
        None,
    )


def _has_target_system_band(config: WorkerInitConfig) -> bool:
    return config.target_min_systems is not None and config.target_max_systems is not None


def _system_count_band_status(
    system_count: int | None,
    config: WorkerInitConfig,
) -> str:
    if not _has_target_system_band(config):
        return "disabled"
    if system_count is None:
        return "unknown"
    assert config.target_min_systems is not None
    assert config.target_max_systems is not None
    if system_count < config.target_min_systems:
        return "below"
    if system_count > config.target_max_systems:
        return "above"
    return "in_band"


def _failure_code_for_band_status(
    band_status: str,
    *,
    truncation_attempted: bool = False,
) -> str:
    if band_status == "below":
        return "system_band_below_min"
    if band_status == "above":
        return "system_band_above_max"
    if truncation_attempted:
        return "system_band_truncation_exhausted"
    return "system_band_rejected"


def _classify_truncation_candidate_result(
    *,
    candidate_failure: SampleFailure | None,
    actual_system_count: int | None,
    config: WorkerInitConfig,
) -> str:
    """Classify a truncation attempt for bracketed search.

    The rescue path is only entered after the full sample was overfull or multi-page.
    For bracket updates, explicit render-overflow failures and over-band counts are
    treated as "too large". Under-band counts are treated as "too small".
    Less-informative failures like timeouts are conservatively treated as too large so
    the search still converges toward shorter prefixes.
    """
    if candidate_failure is not None:
        return "render_failure"
    candidate_band_status = _system_count_band_status(actual_system_count, config)
    if candidate_band_status in {"disabled", "in_band"}:
        return "in_band"
    if candidate_band_status == "below":
        return "too_small"
    return "too_large"


def _search_prefix_truncation_candidate(
    kern_text: str,
    *,
    config: WorkerInitConfig,
    render_candidate,
) -> tuple[
    PrefixTruncationCandidate | None,
    tuple[np.ndarray, RenderedPage | None, int | None, float | None, float | None] | None,
    int,
    str,
]:
    """Search for the longest accepted truncation candidate using bracket + refinement."""
    space = build_prefix_truncation_space(kern_text)
    total_chunks = space.total_chunks
    max_trials = config.overflow_truncation_max_trials
    if total_chunks <= 1 or max_trials < 1:
        return None, None, 0, "unknown"

    attempted_trials = 0
    tried_counts: set[int] = set()
    best_candidate: PrefixTruncationCandidate | None = None
    best_render_result: (
        tuple[np.ndarray, RenderedPage | None, int | None, float | None, float | None] | None
    ) = None
    observed_too_small = False
    observed_too_large = False
    observed_render_failure = False

    def note_classification(classification: str) -> None:
        nonlocal observed_too_small, observed_too_large, observed_render_failure
        if classification == "too_small":
            observed_too_small = True
        elif classification == "render_failure":
            observed_render_failure = True
        elif classification == "too_large":
            observed_too_large = True

    lower_bound = 0
    upper_bound = total_chunks
    binary_budget = min(max_trials, max(1, math.ceil(math.log2(total_chunks)) + 1))

    while attempted_trials < binary_budget and upper_bound - lower_bound > 1:
        chunk_count = (lower_bound + upper_bound + 1) // 2
        if chunk_count >= total_chunks or chunk_count <= 0 or chunk_count in tried_counts:
            break
        candidate = space.candidate_for_chunk_count(chunk_count)
        tried_counts.add(chunk_count)
        if candidate is None:
            upper_bound = min(upper_bound, chunk_count)
            continue
        attempted_trials += 1
        (
            image,
            render_layers,
            actual_system_count,
            bottom_whitespace_ratio,
            vertical_fill_ratio,
            candidate_failure,
        ) = render_candidate(candidate.transcription)
        classification = _classify_truncation_candidate_result(
            candidate_failure=candidate_failure,
            actual_system_count=actual_system_count,
            config=config,
        )
        note_classification(classification)
        if classification == "in_band" and image is not None:
            best_candidate = candidate
            best_render_result = (
                image,
                render_layers,
                actual_system_count,
                bottom_whitespace_ratio,
                vertical_fill_ratio,
            )
            lower_bound = max(lower_bound, chunk_count)
        elif classification == "too_small":
            lower_bound = max(lower_bound, chunk_count)
        else:
            upper_bound = min(upper_bound, chunk_count)

    for chunk_count in range(min(total_chunks - 1, upper_bound - 1), lower_bound, -1):
        if attempted_trials >= max_trials:
            break
        if chunk_count in tried_counts:
            continue
        candidate = space.candidate_for_chunk_count(chunk_count)
        tried_counts.add(chunk_count)
        if candidate is None:
            continue
        attempted_trials += 1
        (
            image,
            render_layers,
            actual_system_count,
            bottom_whitespace_ratio,
            vertical_fill_ratio,
            candidate_failure,
        ) = render_candidate(candidate.transcription)
        classification = _classify_truncation_candidate_result(
            candidate_failure=candidate_failure,
            actual_system_count=actual_system_count,
            config=config,
        )
        note_classification(classification)
        if classification == "in_band" and image is not None:
            return (
                candidate,
                (
                    image,
                    render_layers,
                    actual_system_count,
                    bottom_whitespace_ratio,
                    vertical_fill_ratio,
                ),
                attempted_trials,
                "in_band",
            )
        if classification == "too_small":
            break

    if observed_too_small and (observed_too_large or observed_render_failure):
        diagnostic = "mixed_gap"
    elif observed_too_small:
        diagnostic = "below_min"
    elif observed_too_large:
        diagnostic = "too_large"
    elif observed_render_failure:
        diagnostic = "render_failure"
    else:
        diagnostic = "unknown"

    return best_candidate, best_render_result, attempted_trials, diagnostic


def _postprocess_rendered_image(
    image: np.ndarray,
    *,
    render_layers: RenderedPage | None = None,
    filename: str,
    variant_idx: int,
    config: WorkerInitConfig,
    profile_enabled: bool = False,
) -> tuple[bytes, float, float, dict[str, float] | None]:
    """Apply offline image augmentation and JPEG encoding."""
    if config.disable_offline_image_augmentations:
        encode_start_ns = time.perf_counter_ns() if profile_enabled else 0
        jpeg_img = encode_jpeg(image)
        jpeg_encode_ms = _elapsed_ms(encode_start_ns) if profile_enabled else 0.0
        return jpeg_img, 0.0, jpeg_encode_ms, None

    from scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment import (
        offline_augment,
    )

    augment_start_ns = time.perf_counter_ns() if profile_enabled else 0
    augment_stage_timings: dict[str, float] | None = None
    augment_output = offline_augment(
        image,
        render_layers=render_layers,
        texturize_image=True,
        filename=filename,
        variant_idx=variant_idx,
        augment_seed=config.augment_seed,
        geom_x_squeeze_prob=config.geom_x_squeeze_prob,
        geom_x_squeeze_min_scale=config.geom_x_squeeze_min_scale,
        geom_x_squeeze_max_scale=config.geom_x_squeeze_max_scale,
        geom_x_squeeze_apply_in_conservative=config.geom_x_squeeze_apply_in_conservative,
        geom_x_squeeze_preview_force_scale=config.geom_x_squeeze_preview_force_scale,
        return_timings=profile_enabled,
    )
    if profile_enabled:
        if isinstance(augment_output, tuple):
            augmented_img, augment_stage_timings = augment_output
        else:
            # Backward-compatible path for tests/mocks that return image only.
            augmented_img = augment_output
            augment_stage_timings = None
    else:
        if isinstance(augment_output, tuple):
            augmented_img = augment_output[0]
        else:
            augmented_img = augment_output
    offline_augment_ms = _elapsed_ms(augment_start_ns) if profile_enabled else 0.0
    encode_start_ns = time.perf_counter_ns() if profile_enabled else 0
    jpeg_img = encode_jpeg(augmented_img)
    jpeg_encode_ms = _elapsed_ms(encode_start_ns) if profile_enabled else 0.0
    del augmented_img
    return jpeg_img, offline_augment_ms, jpeg_encode_ms, augment_stage_timings


def generate_sample_from_path_outcome(
    file_path: Path | str,
    variant_idx: int = 0,
    sample_id: str | None = None,
) -> SampleOutcome:
    """Generate sample from path and return structured success/failure outcome."""
    assert _file_renderer is not None, "Worker not initialized"
    config = _current_worker_config()
    if config.deterministic_seed_salt and sample_id:
        np.random.seed(
            derive_sample_seed(sample_id=sample_id, salt=config.deterministic_seed_salt)
        )
    filename = Path(file_path).name
    profile_enabled = bool(config.profile_enabled)
    total_start_ns = time.perf_counter_ns() if profile_enabled else 0
    stage_ms: dict[str, float] | None = {} if profile_enabled else None

    def add_stage_time(stage_name: str, start_ns: int) -> None:
        if not profile_enabled or stage_ms is None:
            return
        stage_ms[stage_name] = float(stage_ms.get(stage_name, 0.0)) + _elapsed_ms(start_ns)

    def finalize_profile(*, failure_stage: str | None = None) -> SampleProfile | None:
        if not profile_enabled or stage_ms is None:
            return None
        stage_ms["worker_total_ms"] = _elapsed_ms(total_start_ns)
        for stage_name in PROFILE_STAGE_NAMES:
            stage_ms.setdefault(stage_name, 0.0)
        return SampleProfile(stages_ms=dict(stage_ms), failure_stage=failure_stage)

    def build_and_render_once(
        source_transcription: str,
    ) -> tuple[
        np.ndarray | None,
        RenderedPage | None,
        int | None,
        float | None,
        float | None,
        SampleFailure | None,
    ]:
        build_start_ns = time.perf_counter_ns() if profile_enabled else 0
        kern_for_verovio = _build_render_transcription(source_transcription, config)
        add_stage_time("build_render_transcription_ms", build_start_ns)

        render_start_ns = time.perf_counter_ns() if profile_enabled else 0
        render_result = _render_kern_for_sample(
            kern_for_verovio,
            filename=filename,
            config=config,
        )
        if len(render_result) == 6:
            (
                image,
                render_layers,
                actual_system_count,
                bottom_whitespace_ratio,
                vertical_fill_ratio,
                failure,
            ) = render_result
        elif len(render_result) == 4:
            image, render_layers, actual_system_count, failure = render_result
            bottom_whitespace_ratio = None
            vertical_fill_ratio = None
        else:
            image, actual_system_count, failure = render_result
            render_layers = None
            bottom_whitespace_ratio = None
            vertical_fill_ratio = None
        add_stage_time("render_ms", render_start_ns)
        return (
            image,
            render_layers,
            actual_system_count,
            bottom_whitespace_ratio,
            vertical_fill_ratio,
            failure,
        )

    def finalize_success(
        *,
        image: np.ndarray,
        render_layers: RenderedPage | None,
        transcription: str,
        actual_system_count: int | None,
        truncation_applied: bool,
        truncation_ratio: float | None,
        bottom_whitespace_ratio: float | None,
        vertical_fill_ratio: float | None,
    ) -> SampleSuccess:
        jpeg_img, offline_augment_ms, jpeg_encode_ms, augment_stage_timings = (
            _postprocess_rendered_image(
                image,
                render_layers=render_layers,
                filename=filename,
                variant_idx=variant_idx,
                config=config,
                profile_enabled=profile_enabled,
            )
        )
        if profile_enabled and stage_ms is not None:
            stage_ms["offline_augment_ms"] = (
                float(stage_ms.get("offline_augment_ms", 0.0)) + float(offline_augment_ms)
            )
        if profile_enabled and stage_ms is not None and augment_stage_timings is not None:
            stage_ms["offline_geom_ms"] = float(stage_ms.get("offline_geom_ms", 0.0)) + float(
                augment_stage_timings.get("offline_geom_ms", 0.0)
            )
            stage_ms["offline_gates_ms"] = float(stage_ms.get("offline_gates_ms", 0.0)) + float(
                augment_stage_timings.get("offline_gates_ms", 0.0)
            )
            stage_ms["offline_augraphy_ms"] = float(
                stage_ms.get("offline_augraphy_ms", 0.0)
            ) + float(augment_stage_timings.get("offline_augraphy_ms", 0.0))
            stage_ms["offline_texture_ms"] = float(
                stage_ms.get("offline_texture_ms", 0.0)
            ) + float(augment_stage_timings.get("offline_texture_ms", 0.0))
        if profile_enabled and stage_ms is not None:
            stage_ms["jpeg_encode_ms"] = float(stage_ms.get("jpeg_encode_ms", 0.0)) + float(
                jpeg_encode_ms
            )
        return SampleSuccess(
            image=jpeg_img,
            transcription=transcription,
            filename=filename,
            actual_system_count=actual_system_count,
            truncation_applied=truncation_applied,
            truncation_ratio=truncation_ratio,
            bottom_whitespace_ratio=bottom_whitespace_ratio,
            vertical_fill_ratio=vertical_fill_ratio,
            profile=finalize_profile(),
        )

    try:
        read_start_ns = time.perf_counter_ns() if profile_enabled else 0
        kern_content = Path(file_path).read_text()
        add_stage_time("read_kern_ms", read_start_ns)

        validate_start_ns = time.perf_counter_ns() if profile_enabled else 0
        is_valid, error_msg = is_valid_kern(kern_content)
        add_stage_time("validate_kern_ms", validate_start_ns)
        if not is_valid:
            return SampleFailure(
                code="invalid_kern",
                filename=filename,
                detail=error_msg,
                profile=finalize_profile(failure_stage="validate"),
            )

        label_transcription = _build_label_transcription(kern_content, config)

        (
            image,
            render_layers,
            actual_system_count,
            bottom_whitespace_ratio,
            vertical_fill_ratio,
            failure,
        ) = build_and_render_once(label_transcription)
        band_status = _system_count_band_status(actual_system_count, config)
        if failure is None and image is not None:
            if band_status in {"disabled", "in_band"}:
                return finalize_success(
                    image=image,
                    render_layers=render_layers,
                    transcription=label_transcription,
                    actual_system_count=actual_system_count,
                    truncation_applied=False,
                    truncation_ratio=None,
                    bottom_whitespace_ratio=bottom_whitespace_ratio,
                    vertical_fill_ratio=vertical_fill_ratio,
                )
            if band_status == "below":
                return SampleFailure(
                    code="system_band_below_min",
                    filename=filename,
                    detail=f"system_count={actual_system_count}",
                    profile=finalize_profile(failure_stage="render"),
                )

        truncation_attempted = False
        should_try_truncation = False
        failure_code = failure.code if failure is not None else _failure_code_for_band_status(
            band_status
        )
        failure_detail = failure.detail if failure is not None else f"system_count={actual_system_count}"
        if config.overflow_truncation_enabled:
            if failure is not None and failure.code == "multi_page":
                should_try_truncation = True
            elif failure is None and band_status == "above":
                should_try_truncation = True

        if should_try_truncation:
            (
                rescued_candidate,
                rescued_render_result,
                truncation_attempt_count,
                truncation_diagnostic,
            ) = _search_prefix_truncation_candidate(
                label_transcription,
                config=config,
                render_candidate=build_and_render_once,
            )
            truncation_attempted = truncation_attempt_count > 0
            if rescued_candidate is not None and rescued_render_result is not None:
                (
                    image,
                    render_layers,
                    actual_system_count,
                    bottom_whitespace_ratio,
                    vertical_fill_ratio,
                ) = rescued_render_result
                return finalize_success(
                    image=image,
                    render_layers=render_layers,
                    transcription=rescued_candidate.transcription,
                    actual_system_count=actual_system_count,
                    truncation_applied=True,
                    truncation_ratio=rescued_candidate.ratio,
                    bottom_whitespace_ratio=bottom_whitespace_ratio,
                    vertical_fill_ratio=vertical_fill_ratio,
                )
            if _has_target_system_band(config) and truncation_attempted:
                failure_code = "system_band_truncation_exhausted"
                failure_detail = (
                    f"target_band={config.target_min_systems}-{config.target_max_systems};"
                    f"diagnostic={truncation_diagnostic}"
                )

        if failure is not None:
            return SampleFailure(
                code=failure_code,
                filename=failure.filename,
                detail=failure_detail,
                truncation_attempted=truncation_attempted,
                profile=finalize_profile(failure_stage="render"),
            )
        return SampleFailure(
            code=failure_code,
            filename=filename,
            detail=failure_detail,
            truncation_attempted=truncation_attempted,
            profile=finalize_profile(failure_stage="render"),
        )
    except Exception as e:
        return SampleFailure(
            code="processing_error",
            filename=filename,
            detail=f"Error processing {file_path}: {e}",
            profile=finalize_profile(failure_stage="exception"),
        )


def generate_sample_from_path(
    file_path: Path | str,
    variant_idx: int = 0,
) -> tuple[bytes, str, str] | tuple[None, str, str]:
    """Generate sample from path with legacy tuple return compatibility."""
    outcome = generate_sample_from_path_outcome(file_path=file_path, variant_idx=variant_idx)
    return outcome_to_legacy_tuple(outcome)
