use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ParseError;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct RenderAugmentationConfig {
    /// Page gates for supervised marks, applied to pages with eligible sites.
    pub fermata_page_probability: f64,
    pub arpeggio_page_probability: f64,
    pub articulations: NoteMarkAugmentationConfig,
    pub fermatas: NoteMarkAugmentationConfig,
    pub arpeggios: NoteMarkAugmentationConfig,
    pub mordents: NoteMarkAugmentationConfig,
    pub dynamics: PointAugmentationConfig,
    pub hairpins: SpanAugmentationConfig,
    pub pedal: SpanAugmentationConfig,
    pub ottavas: SpanAugmentationConfig,
    pub tuplet_visibility: TupletVisibilityAugmentationConfig,
    pub reference_metadata: PointAugmentationConfig,
    pub staff_names: PointAugmentationConfig,
    pub metronome_tempos: PointAugmentationConfig,
    pub text_directions: PointAugmentationConfig,
}

impl Default for RenderAugmentationConfig {
    fn default() -> Self {
        Self {
            fermata_page_probability: 0.20,
            arpeggio_page_probability: 0.28,
            articulations: NoteMarkAugmentationConfig {
                enabled: true,
                probability_per_candidate: 0.08,
                max_count: 8,
            },
            fermatas: NoteMarkAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 1,
            },
            arpeggios: NoteMarkAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 1,
            },
            mordents: NoteMarkAugmentationConfig {
                enabled: true,
                probability_per_candidate: 0.015,
                max_count: 2,
            },
            dynamics: PointAugmentationConfig {
                enabled: true,
                probability_per_candidate: 0.04,
                max_count: 4,
            },
            hairpins: SpanAugmentationConfig {
                enabled: true,
                probability_per_candidate: 0.03,
                min_line_span: 1,
                max_line_span: 8,
                max_count: 3,
            },
            pedal: SpanAugmentationConfig {
                enabled: true,
                probability_per_candidate: 0.02,
                min_line_span: 1,
                max_line_span: 12,
                max_count: 2,
            },
            // Keep render-only semantic augmentations disabled unless targets
            // are projected from the same augmented document. Source
            // occurrences of these symbols may be supervised in transcoda-v2;
            // adding synthetic render-only occurrences creates image-visible
            // notation with no matching label.
            ottavas: SpanAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.01,
                min_line_span: 2,
                max_line_span: 10,
                max_count: 1,
            },
            tuplet_visibility: TupletVisibilityAugmentationConfig {
                enabled: true,
                probability_per_candidate: 0.30,
                max_count: 2,
            },
            reference_metadata: PointAugmentationConfig {
                enabled: true,
                probability_per_candidate: 0.25,
                max_count: 3,
            },
            staff_names: PointAugmentationConfig {
                enabled: true,
                probability_per_candidate: 0.20,
                max_count: 1,
            },
            metronome_tempos: PointAugmentationConfig {
                enabled: true,
                probability_per_candidate: 0.20,
                max_count: 1,
            },
            text_directions: PointAugmentationConfig {
                enabled: true,
                probability_per_candidate: 0.03,
                max_count: 2,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct NoteMarkAugmentationConfig {
    pub enabled: bool,
    pub probability_per_candidate: f64,
    pub max_count: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PointAugmentationConfig {
    pub enabled: bool,
    pub probability_per_candidate: f64,
    pub max_count: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SpanAugmentationConfig {
    pub enabled: bool,
    pub probability_per_candidate: f64,
    pub min_line_span: usize,
    pub max_line_span: usize,
    pub max_count: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct TupletVisibilityAugmentationConfig {
    pub enabled: bool,
    pub probability_per_candidate: f64,
    pub max_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderAugmentationSeed(pub u64);

impl Default for RenderAugmentationSeed {
    fn default() -> Self {
        Self(0x9e37_79b9_7f4a_7c15)
    }
}

#[derive(Debug, Error)]
pub enum AugmentationError {
    #[error("invalid augmentation config: {0}")]
    InvalidConfig(String),
    #[error("failed to build augmented line: {0}")]
    InvalidLine(#[from] ParseError),
    #[error("line {line_index} is no longer a data line")]
    ExpectedDataLine { line_index: usize },
    #[error("line {line_index} has no dynamic spine slot for staff {staff_index}")]
    MissingDynamicSlot {
        line_index: usize,
        staff_index: usize,
    },
    #[error("the score does not resolve: {0}")]
    State(crate::score::StateErrors),
}

pub(super) fn validate_config(config: &RenderAugmentationConfig) -> Result<(), AugmentationError> {
    validate_probability("fermata page", config.fermata_page_probability)?;
    validate_probability("arpeggio page", config.arpeggio_page_probability)?;
    validate_probability(
        "articulations",
        config.articulations.probability_per_candidate,
    )?;
    validate_probability("fermatas", config.fermatas.probability_per_candidate)?;
    validate_probability("arpeggios", config.arpeggios.probability_per_candidate)?;
    validate_probability("mordents", config.mordents.probability_per_candidate)?;
    validate_probability("dynamics", config.dynamics.probability_per_candidate)?;
    validate_span_config("hairpins", &config.hairpins)?;
    validate_span_config("pedal", &config.pedal)?;
    validate_span_config("ottavas", &config.ottavas)?;
    validate_probability(
        "tuplet_visibility",
        config.tuplet_visibility.probability_per_candidate,
    )?;
    validate_probability(
        "reference_metadata",
        config.reference_metadata.probability_per_candidate,
    )?;
    validate_probability("staff_names", config.staff_names.probability_per_candidate)?;
    validate_probability(
        "metronome_tempos",
        config.metronome_tempos.probability_per_candidate,
    )?;
    validate_probability(
        "text_directions",
        config.text_directions.probability_per_candidate,
    )?;
    Ok(())
}

fn validate_probability(name: &str, value: f64) -> Result<(), AugmentationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(AugmentationError::InvalidConfig(format!(
            "{name} probability must be finite and between 0 and 1"
        )))
    }
}

fn validate_span_config(
    name: &str,
    config: &SpanAugmentationConfig,
) -> Result<(), AugmentationError> {
    validate_probability(name, config.probability_per_candidate)?;
    if config.min_line_span == 0 {
        return Err(AugmentationError::InvalidConfig(format!(
            "{name} min_line_span must be greater than 0"
        )));
    }
    if config.min_line_span > config.max_line_span {
        return Err(AugmentationError::InvalidConfig(format!(
            "{name} min_line_span must not exceed max_line_span"
        )));
    }
    Ok(())
}
