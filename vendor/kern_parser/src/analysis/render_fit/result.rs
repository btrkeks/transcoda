use thiserror::Error;

use crate::analysis::{SvgOverlapSummary, SvgPageWhitespace, VerovioRenderSettings};

/// Dataset-level seed metadata for deterministic stratified candidate sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderFitSearchSeed {
    pub index: usize,
    pub count: usize,
}

/// One attempted render setting candidate and its measured outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderFitAttempt {
    pub settings: VerovioRenderSettings,
    pub page_count: u32,
    pub whitespace: Option<SvgPageWhitespace>,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderFitAttemptOrigin {
    Primary,
    SvgOverlapFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderFitAttemptOutcome {
    Accepted,
    PageCountMismatch,
    Whitespace,
    SvgOverlap,
    RenderError,
    EvaluateError,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderFitAttemptTiming {
    pub attempt_index: usize,
    pub origin: RenderFitAttemptOrigin,
    pub render_millis: f64,
    pub evaluate_millis: f64,
    pub page_count: Option<u32>,
    pub outcome: RenderFitAttemptOutcome,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderFitTimingReport {
    pub candidate_count: usize,
    pub attempt_budget: usize,
    pub attempts: Vec<RenderFitAttemptTiming>,
}

/// Best render settings found by the search.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderFitResult {
    pub settings: VerovioRenderSettings,
    pub page_count: u32,
    pub whitespace: SvgPageWhitespace,
    pub score: f64,
    pub attempts: Vec<RenderFitAttempt>,
    pub selected_pages: Vec<RenderedPageArtifact>,
    pub selected_render_source_lines: Vec<Option<usize>>,
    pub quality_report: RenderFitQualityReport,
}

/// Rendered SVG page artifact selected by the render-fit search.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedPageArtifact {
    pub page_number: u32,
    pub svg: String,
    pub whitespace: SvgPageWhitespace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvgOverlapGateMode {
    Off,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvgOverlapFallbackMode {
    Off,
    HorizontalRelief,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderFitSearchOptions {
    pub svg_overlap_gate: SvgOverlapGateMode,
    pub svg_overlap_min_ratio: f64,
    pub svg_overlap_fallback: SvgOverlapFallbackMode,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RenderFitQualityReport {
    pub svg_overlap_gate_enabled: bool,
    pub svg_overlap_gate_hit: bool,
    pub svg_overlap_rejected_attempt_count: usize,
    pub svg_overlap_fallback_attempt_count: usize,
    pub worst_svg_overlap: Option<SvgOverlapSummary>,
}

#[derive(Debug, Error, Clone, PartialEq)]
#[error("{error}")]
pub struct RenderFitSearchFailure {
    pub error: RenderFitError,
    pub quality_report: RenderFitQualityReport,
}

impl Default for RenderFitSearchOptions {
    fn default() -> Self {
        Self {
            svg_overlap_gate: SvgOverlapGateMode::Off,
            svg_overlap_min_ratio: 0.15,
            svg_overlap_fallback: SvgOverlapFallbackMode::HorizontalRelief,
        }
    }
}

impl RenderFitSearchOptions {
    pub fn svg_overlap_enabled(self) -> bool {
        self.svg_overlap_gate == SvgOverlapGateMode::Reject
    }
}

impl RenderFitQualityReport {
    pub fn for_options(options: RenderFitSearchOptions) -> Self {
        Self {
            svg_overlap_gate_enabled: options.svg_overlap_enabled(),
            ..Self::default()
        }
    }

    pub fn record_svg_overlap_rejection(&mut self, summary: SvgOverlapSummary) {
        self.svg_overlap_gate_hit = true;
        self.svg_overlap_rejected_attempt_count += 1;
        if self
            .worst_svg_overlap
            .as_ref()
            .is_none_or(|current| svg_overlap_sort_key(&summary) > svg_overlap_sort_key(current))
        {
            self.worst_svg_overlap = Some(summary);
        }
    }

    pub fn record_svg_overlap_fallback_attempt(&mut self) {
        self.svg_overlap_fallback_attempt_count += 1;
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RenderFitError {
    #[error("native engraving is unavailable; enable the `engraving` Cargo feature")]
    EngravingUnavailable,
    #[error("render fit search currently supports target_pages=1, got {target_pages}")]
    UnsupportedTargetPages { target_pages: u32 },
    #[error("render fit search requires max_attempts greater than zero")]
    InvalidMaxAttempts,
    #[error("render fit search seed count must be greater than zero")]
    InvalidSearchSeed,
    #[error("render fit config weight {setting} must be finite and non-negative")]
    InvalidWeight { setting: &'static str },
    #[error("invalid render fit bounds: {message}")]
    InvalidBounds { message: String },
    #[error("invalid initial render settings: {message}")]
    InvalidInitialSettings { message: String },
    #[error("Verovio render failed: {message}")]
    RenderFailed { message: String },
    #[error("SVG whitespace measurement failed on page {page_number}: {message}")]
    WhitespaceMeasurementFailed { page_number: u32, message: String },
    #[error("SVG overlap detection failed on page {page_number}: {message}")]
    SvgOverlapDetectionFailed { page_number: u32, message: String },
    #[error(
        "rendered page {page_number} has size {actual_width}x{actual_height}, expected 1050x1485"
    )]
    UnexpectedPageSize {
        page_number: u32,
        actual_width: u32,
        actual_height: u32,
    },
    #[error("no render settings fit {target_pages} page(s) within {attempt_count} attempt(s)")]
    NoFittingCandidate {
        target_pages: u32,
        attempt_count: usize,
    },
    #[error(transparent)]
    Dialect(#[from] crate::serialize::DialectError),
}

fn svg_overlap_sort_key(summary: &SvgOverlapSummary) -> (usize, u64, u64) {
    let worst_ratio = summary
        .worst_pairs
        .first()
        .map(|pair| pair.ratio)
        .unwrap_or(0.0)
        .to_bits();
    let worst_area = summary
        .worst_pairs
        .first()
        .map(|pair| pair.area)
        .unwrap_or(0.0)
        .to_bits();
    (summary.overlap_count, worst_ratio, worst_area)
}
