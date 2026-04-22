"""Single hard-coded recipe for the dataset-generation rewrite."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompositionPolicy:
    segment_count_weights: tuple[tuple[int, float], ...] = (
        (1, 0.62),
        (2, 0.28),
        (3, 0.10),
    )
    min_total_measures: int = 10
    max_total_measures: int = 32
    max_selection_attempts: int = 48


@dataclass(frozen=True)
class RenderOnlyAugmentationPolicy:
    include_title_probability: float = 0.50
    include_author_probability: float = 0.50
    render_layout_profile: str = "target_5_6_systems"
    render_pedals_probability: float = 0.20
    render_pedals_measures_probability: float = 0.30
    render_instrument_piano_probability: float = 0.15
    render_sforzando_probability: float = 0.20
    render_sforzando_per_note_probability: float = 0.03
    render_accent_probability: float = 0.10
    render_accent_per_note_probability: float = 0.015
    render_tempo_probability: float = 0.12
    render_tempo_include_mm_probability: float = 0.35
    render_hairpins_probability: float = 0.25
    render_hairpins_max_spans: int = 2
    render_dynamic_marks_probability: float = 0.15
    render_dynamic_marks_min_count: int = 1
    render_dynamic_marks_max_count: int = 2
    max_render_attempts: int = 3
    min_frame_margin_px: int = 2
    target_frame_margin_px: int = 12


@dataclass(frozen=True)
class TruncationPolicy:
    accept_without_truncation_max_systems: int = 4
    preferred_min_systems: int = 5
    preferred_max_systems: int = 7
    required_over_systems: int = 7
    max_candidate_trials: int = 24


@dataclass(frozen=True)
class OfflineAugmentationBandPolicy:
    geom_x_squeeze_prob: float
    geom_x_squeeze_min_scale: float
    geom_x_squeeze_max_scale: float
    geom_x_squeeze_apply_in_conservative: bool = True


@dataclass(frozen=True)
class OfflineAugmentationPolicy:
    roomy: OfflineAugmentationBandPolicy = field(
        default_factory=lambda: OfflineAugmentationBandPolicy(
            geom_x_squeeze_prob=0.55,
            geom_x_squeeze_min_scale=0.72,
            geom_x_squeeze_max_scale=0.93,
        )
    )
    balanced: OfflineAugmentationBandPolicy = field(
        default_factory=lambda: OfflineAugmentationBandPolicy(
            geom_x_squeeze_prob=0.35,
            geom_x_squeeze_min_scale=0.80,
            geom_x_squeeze_max_scale=0.95,
        )
    )
    tight: OfflineAugmentationBandPolicy = field(
        default_factory=lambda: OfflineAugmentationBandPolicy(
            geom_x_squeeze_prob=0.15,
            geom_x_squeeze_min_scale=0.88,
            geom_x_squeeze_max_scale=0.98,
        )
    )


@dataclass(frozen=True)
class ProductionRecipe:
    version: str = "dataset_generation_v1"
    image_width: int = 1050
    image_height: int = 1485
    composition: CompositionPolicy = field(default_factory=CompositionPolicy)
    render_only_aug: RenderOnlyAugmentationPolicy = field(
        default_factory=RenderOnlyAugmentationPolicy
    )
    truncation: TruncationPolicy = field(default_factory=TruncationPolicy)
    offline_aug: OfflineAugmentationPolicy = field(default_factory=OfflineAugmentationPolicy)
    max_attempt_multiplier: int = 8
