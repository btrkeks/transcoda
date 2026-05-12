"""Pure data structures for synthetic score generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RenderedPage:
    """Rendered notation page with separated clean page and mask layers."""

    image: np.ndarray
    """White-backed clean page as RGB numpy array (H, W, 3)."""

    foreground: np.ndarray
    """Notation foreground as RGB numpy array (H, W, 3)."""

    alpha: np.ndarray
    """Antialiased notation opacity mask as uint8 numpy array (H, W)."""


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for score generation (pure data, no logic)."""

    num_systems_hint: int = 2
    """Target number of systems to sample. Actual count may vary."""

    include_title: bool = False
    """Whether to include a random title in the score."""

    include_author: bool = False
    """Whether to include a random author name in the score."""

    include_time_signature: bool = True
    """Whether to include time signature in the transcription."""

    texturize_image: bool = True
    """Whether to apply paper texture to the rendered image."""

    image_width: int = 1050
    """Target image width in pixels."""

    image_height: int | None = None
    """Target image height in pixels. If None, height is determined by content."""

    render_layout_profile: str = "default"
    """Render layout sampling profile."""

    def __post_init__(self) -> None:
        if self.num_systems_hint < 1:
            raise ValueError("num_systems_hint must be >= 1")
        if self.image_width <= 0:
            raise ValueError("image_width must be positive")
        if self.image_height is not None and self.image_height <= 0:
            raise ValueError("image_height must be positive")
        if self.render_layout_profile not in {"default", "target_5_6_systems"}:
            raise ValueError(
                "render_layout_profile must be one of: default, target_5_6_systems"
            )


@dataclass(frozen=True)
class GeneratedScore:
    """Output of score generation (pure data, no logic)."""

    image: np.ndarray
    """Rendered score as RGB numpy array (H, W, 3)."""

    transcription: str
    """The **kern transcription tokens."""

    actual_system_count: int
    """Number of systems in the rendered image."""

    metadata_prefix: str
    """Metadata (title/author) prepended during rendering."""

    render_layers: RenderedPage | None = None
    """Optional separated render layers used by offline augmentation."""

    bottom_whitespace_ratio: float | None = None
    """Fraction of final frame height that remains blank below content."""

    vertical_fill_ratio: float | None = None
    """Fraction of final frame height occupied by detected notation content."""
