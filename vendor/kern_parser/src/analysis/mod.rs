//! Derived analyses for typed Humdrum `**kern` documents.

pub mod accidental_display;
pub(crate) mod document_regions;
pub(crate) mod explicit_accidentals;
pub mod initial_time_signature;
pub mod line_durations;
mod loose_end;
pub mod measure_spans;
pub mod render_fit;
pub mod rendered_metrics;
pub mod semantic_events;
pub mod slur_spans;
pub mod svg_geometric_aug;
pub mod svg_overlap;
pub mod svg_whitespace;
pub mod tie_chains;
pub mod verovio_settings;

pub(crate) use document_regions::leading_header_bounds;
pub use initial_time_signature::{
    resolve_initial_time_signature, InitialTimeSignatureResolutionError,
};
pub use line_durations::{
    data_line_durations, first_timeline_break, score_line_durations, DataLineDuration,
    DataLineDurationError, TimelineBreak,
};
pub use loose_end::{LooseEnd, LooseEndSide, Spanner};
pub use measure_spans::{measure_renders_empty, measure_spans, MeasureSpan};
pub use render_fit::{
    find_render_settings, find_seeded_render_settings, find_seeded_render_settings_with_options,
    find_seeded_render_settings_with_options_and_timing, RenderFitAttempt, RenderFitAttemptOrigin,
    RenderFitAttemptOutcome, RenderFitAttemptTiming, RenderFitConfig, RenderFitError,
    RenderFitQualityReport, RenderFitResult, RenderFitSearchFailure, RenderFitSearchOptions,
    RenderFitSearchSeed, RenderFitTimingReport, RenderedPageArtifact, SvgOverlapFallbackMode,
    SvgOverlapGateMode,
};
pub use rendered_metrics::{
    medium_verovio_render_metrics, verovio_render_metrics, RenderedDocumentMetrics,
    RenderedMetricsError, RenderedPageMetrics,
};
pub use semantic_events::{
    semantic_events, SemanticAccidental, SemanticEvent, SemanticEventError, SemanticPitch,
};
pub use slur_spans::{
    resolve_slur_spans, SlurKey, SlurLocation, SlurSide, SlurSpan, SlurSpanAnalysis, UnpairedSlur,
};
pub use svg_geometric_aug::{
    apply_geometric_transform_to_svg, augment_svg_geometry, transformed_svg_content_box,
    transformed_svg_content_fits_page, GeometricAugmentationConfig, GeometricAugmentationError,
    GeometricAugmentationResult, GeometricAugmentationSeed, GeometricAugmentationStatus,
    GeometricTransform,
};
pub use svg_overlap::{
    detect_bbox_overlaps, detect_svg_bbox_overlaps, extract_verovio_svg_bboxes, SvgBBox,
    SvgOverlapError, SvgOverlapOptions, SvgOverlapPair, SvgOverlapSummary,
};
pub use svg_whitespace::{
    count_svg_systems, detect_svg_content_box, detect_svg_page_size, measure_svg_whitespace,
    strip_svg_bounding_boxes, SvgContentBox, SvgContentBoxError, SvgPageSize, SvgPageSizeError,
    SvgPageWhitespace, SvgStripBoundingBoxesError, SvgSystemCountError, SvgWhitespace,
    SvgWhitespaceError,
};
pub use tie_chains::{
    analyze_tie_chains, TieChain, TieChainAnalysis, TieChainAnalysisError, TieChainFinding,
    TieChainMember, TieChainPhase, TieChainResolution, TieChainStatus,
};
pub use verovio_settings::{
    FloatRange, FontSet, IntRange, RenderAdjustment, VerovioMusicFont, VerovioRenderSettings,
    VerovioRenderSettingsBounds, VerovioRenderSettingsError, SINGLE_PAGE_RENDER_FONTS,
    SINGLE_PAGE_RENDER_SETTINGS_BOUNDS, TARGET_RENDER_HEIGHT, TARGET_RENDER_WIDTH,
};

pub(crate) mod written_measures;

pub(crate) mod diagnostics_index;
pub(crate) mod unison_heads;
