"""Typed worker configuration and outcome models for file generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

FailureCode = Literal[
    "invalid_kern",
    "multi_page",
    "render_fit",
    "render_rejected",
    "sparse_render",
    "system_band_below_min",
    "system_band_above_max",
    "system_band_truncation_exhausted",
    "system_band_rejected",
    "processing_error",
]

ProfileStageName = Literal[
    "read_kern_ms",
    "validate_kern_ms",
    "build_render_transcription_ms",
    "render_ms",
    "offline_geom_ms",
    "offline_gates_ms",
    "offline_augraphy_ms",
    "offline_texture_ms",
    "offline_augment_ms",
    "jpeg_encode_ms",
    "worker_total_ms",
]

PROFILE_STAGE_NAMES: tuple[ProfileStageName, ...] = (
    "read_kern_ms",
    "validate_kern_ms",
    "build_render_transcription_ms",
    "render_ms",
    "offline_geom_ms",
    "offline_gates_ms",
    "offline_augraphy_ms",
    "offline_texture_ms",
    "offline_augment_ms",
    "jpeg_encode_ms",
    "worker_total_ms",
)


@dataclass(frozen=True)
class SampleProfile:
    """Optional per-sample profiling payload emitted by workers."""

    stages_ms: dict[str, float]
    failure_stage: str | None = None


@dataclass(frozen=True)
class WorkerInitConfig:
    """Configuration injected into each file-generation worker process."""

    image_width: int
    image_height: int | None = None
    augment_seed: int | None = None
    deterministic_seed_salt: str | None = None
    render_pedals_enabled: bool = True
    render_pedals_probability: float = 0.20
    render_pedals_measures_probability: float = 0.3
    render_instrument_piano_enabled: bool = True
    render_instrument_piano_probability: float = 0.15
    render_sforzando_enabled: bool = True
    render_sforzando_probability: float = 0.20
    render_sforzando_per_note_probability: float = 0.03
    render_accent_enabled: bool = True
    render_accent_probability: float = 0.10
    render_accent_per_note_probability: float = 0.015
    render_tempo_enabled: bool = True
    render_tempo_probability: float = 0.12
    render_tempo_include_mm_probability: float = 0.35
    render_hairpins_enabled: bool = True
    render_hairpins_probability: float = 0.25
    render_hairpins_max_spans: int = 2
    render_dynamic_marks_enabled: bool = True
    render_dynamic_marks_probability: float = 0.15
    render_dynamic_marks_min_count: int = 1
    render_dynamic_marks_max_count: int = 2
    courtesy_naturals_probability: float = 0.15
    disable_offline_image_augmentations: bool = False
    geom_x_squeeze_prob: float = 0.45
    geom_x_squeeze_min_scale: float = 0.70
    geom_x_squeeze_max_scale: float = 0.95
    geom_x_squeeze_apply_in_conservative: bool = True
    geom_x_squeeze_preview_force_scale: float | None = None
    target_min_systems: int | None = None
    target_max_systems: int | None = None
    render_layout_profile: str = "default"
    overflow_truncation_enabled: bool = True
    overflow_truncation_max_trials: int = 24
    profile_enabled: bool = False

    def validate(self) -> None:
        """Raise ValueError when config contains invalid values."""
        if not 0.0 <= self.render_pedals_probability <= 1.0:
            raise ValueError(
                "render_pedals_probability must be in [0.0, 1.0], "
                f"got {self.render_pedals_probability}"
            )
        if not 0.0 <= self.render_pedals_measures_probability <= 1.0:
            raise ValueError(
                "render_pedals_measures_probability must be in [0.0, 1.0], "
                f"got {self.render_pedals_measures_probability}"
            )
        if not 0.0 <= self.render_instrument_piano_probability <= 1.0:
            raise ValueError(
                "render_instrument_piano_probability must be in [0.0, 1.0], "
                f"got {self.render_instrument_piano_probability}"
            )
        if not 0.0 <= self.render_sforzando_probability <= 1.0:
            raise ValueError(
                "render_sforzando_probability must be in [0.0, 1.0], "
                f"got {self.render_sforzando_probability}"
            )
        if not 0.0 <= self.render_sforzando_per_note_probability <= 1.0:
            raise ValueError(
                "render_sforzando_per_note_probability must be in [0.0, 1.0], "
                f"got {self.render_sforzando_per_note_probability}"
            )
        if not 0.0 <= self.render_accent_probability <= 1.0:
            raise ValueError(
                "render_accent_probability must be in [0.0, 1.0], "
                f"got {self.render_accent_probability}"
            )
        if not 0.0 <= self.render_accent_per_note_probability <= 1.0:
            raise ValueError(
                "render_accent_per_note_probability must be in [0.0, 1.0], "
                f"got {self.render_accent_per_note_probability}"
            )
        if not 0.0 <= self.render_tempo_probability <= 1.0:
            raise ValueError(
                "render_tempo_probability must be in [0.0, 1.0], "
                f"got {self.render_tempo_probability}"
            )
        if not 0.0 <= self.render_tempo_include_mm_probability <= 1.0:
            raise ValueError(
                "render_tempo_include_mm_probability must be in [0.0, 1.0], "
                f"got {self.render_tempo_include_mm_probability}"
            )
        if not 0.0 <= self.render_hairpins_probability <= 1.0:
            raise ValueError(
                "render_hairpins_probability must be in [0.0, 1.0], "
                f"got {self.render_hairpins_probability}"
            )
        if self.render_hairpins_max_spans < 1:
            raise ValueError(
                "render_hairpins_max_spans must be >= 1, "
                f"got {self.render_hairpins_max_spans}"
            )
        if not 0.0 <= self.render_dynamic_marks_probability <= 1.0:
            raise ValueError(
                "render_dynamic_marks_probability must be in [0.0, 1.0], "
                f"got {self.render_dynamic_marks_probability}"
            )
        if self.render_dynamic_marks_min_count < 1:
            raise ValueError(
                "render_dynamic_marks_min_count must be >= 1, "
                f"got {self.render_dynamic_marks_min_count}"
            )
        if self.render_dynamic_marks_max_count < self.render_dynamic_marks_min_count:
            raise ValueError(
                "render_dynamic_marks_max_count must be >= render_dynamic_marks_min_count, "
                f"got min={self.render_dynamic_marks_min_count}, "
                f"max={self.render_dynamic_marks_max_count}"
            )
        if not 0.0 <= self.courtesy_naturals_probability <= 1.0:
            raise ValueError(
                "courtesy_naturals_probability must be in [0.0, 1.0], "
                f"got {self.courtesy_naturals_probability}"
            )
        if not 0.0 <= self.geom_x_squeeze_prob <= 1.0:
            raise ValueError(
                "geom_x_squeeze_prob must be in [0.0, 1.0], "
                f"got {self.geom_x_squeeze_prob}"
            )
        if not 0.0 < self.geom_x_squeeze_min_scale <= self.geom_x_squeeze_max_scale <= 1.0:
            raise ValueError(
                "geom_x_squeeze_min_scale/geom_x_squeeze_max_scale must satisfy "
                f"0 < min <= max <= 1.0, got min={self.geom_x_squeeze_min_scale}, "
                f"max={self.geom_x_squeeze_max_scale}"
            )
        if self.geom_x_squeeze_preview_force_scale is not None and not (
            0.0 < self.geom_x_squeeze_preview_force_scale <= 1.0
        ):
            raise ValueError(
                "geom_x_squeeze_preview_force_scale must be in (0.0, 1.0], "
                f"got {self.geom_x_squeeze_preview_force_scale}"
            )
        if self.target_min_systems is not None and self.target_min_systems < 1:
            raise ValueError(
                "target_min_systems must be >= 1 when provided, "
                f"got {self.target_min_systems}"
            )
        if self.target_max_systems is not None and self.target_max_systems < 1:
            raise ValueError(
                "target_max_systems must be >= 1 when provided, "
                f"got {self.target_max_systems}"
            )
        if (self.target_min_systems is None) != (self.target_max_systems is None):
            raise ValueError(
                "target_min_systems and target_max_systems must both be set or both be None"
            )
        if (
            self.target_min_systems is not None
            and self.target_max_systems is not None
            and self.target_min_systems > self.target_max_systems
        ):
            raise ValueError(
                "target_max_systems must be >= target_min_systems, "
                f"got min={self.target_min_systems}, max={self.target_max_systems}"
            )
        if self.render_layout_profile not in {
            "default",
            "target_5_6_systems",
            "polish_5_6_systems",
        }:
            raise ValueError(
                "render_layout_profile must be one of: default, target_5_6_systems, polish_5_6_systems"
            )
        if self.overflow_truncation_max_trials < 1:
            raise ValueError(
                "overflow_truncation_max_trials must be >= 1, "
                f"got {self.overflow_truncation_max_trials}"
            )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable dict for metadata logging."""
        return asdict(self)


@dataclass(frozen=True)
class SampleSuccess:
    """Successful generated sample payload."""

    image: bytes
    transcription: str
    filename: str
    actual_system_count: int | None = None
    truncation_applied: bool = False
    truncation_ratio: float | None = None
    bottom_whitespace_ratio: float | None = None
    vertical_fill_ratio: float | None = None
    profile: SampleProfile | None = None


@dataclass(frozen=True)
class SampleFailure:
    """Structured sample generation failure."""

    code: FailureCode
    filename: str
    detail: str | None = None
    truncation_attempted: bool = False
    profile: SampleProfile | None = None

    def to_legacy_message(self) -> str:
        """Map structured failures to existing string error format."""
        if self.code == "invalid_kern":
            return f"Reject:invalid_kern:{self.detail or 'unknown'}"
        if self.code == "multi_page":
            return "Reject:multi_page"
        if self.code == "render_fit":
            return f"Reject:render_fit:{self.detail or 'unknown'}"
        if self.code == "render_rejected":
            return "Reject:render_rejected"
        if self.code == "system_band_below_min":
            return f"Reject:system_band_below_min:{self.detail or 'unknown'}"
        if self.code == "system_band_above_max":
            return f"Reject:system_band_above_max:{self.detail or 'unknown'}"
        if self.code == "system_band_truncation_exhausted":
            return f"Reject:system_band_truncation_exhausted:{self.detail or 'unknown'}"
        if self.code == "system_band_rejected":
            return f"Reject:system_band_rejected:{self.detail or 'unknown'}"
        if self.code == "sparse_render":
            return f"Reject:sparse_render:{self.detail or 'unknown'}"
        if self.code == "processing_error":
            return self.detail or "Error processing sample"
        return self.detail or f"Error:{self.code}"


SampleOutcome = SampleSuccess | SampleFailure


def outcome_to_legacy_tuple(
    outcome: SampleOutcome,
) -> tuple[bytes, str, str] | tuple[None, str, str]:
    """Bridge typed outcomes to legacy tuple return shape."""
    if isinstance(outcome, SampleSuccess):
        return (outcome.image, outcome.transcription, outcome.filename)
    return (None, outcome.to_legacy_message(), outcome.filename)
