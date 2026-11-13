use std::path::Path;

use crate::analysis::{
    GeometricAugmentationStatus, GeometricTransform, RenderFitQualityReport, RenderFitResult,
    RenderFitSearchOptions, SvgContentBox, SvgOverlapPair, SvgPageWhitespace,
    VerovioRenderSettings,
};
use crate::RenderAugmentationReport;
use serde::{Deserialize, Serialize};

pub(super) const RENDER_SETTINGS_SEARCH_SCHEMA_VERSION: &str = "1.7";

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum RenderSettingsSearchStatus {
    Success,
    ParseError,
    SearchError,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct VerovioRenderSettingsPayload {
    pub(super) page_width: u32,
    pub(super) page_margin_left: u32,
    pub(super) page_margin_right: u32,
    pub(super) page_margin_top: u32,
    pub(super) page_margin_bottom: u32,
    pub(super) bar_line_width: f64,
    pub(super) beam_max_slope: u32,
    pub(super) staff_line_width: f64,
    pub(super) stem_width: f64,
    pub(super) ledger_line_thickness: f64,
    pub(super) thick_barline_thickness: f64,
    pub(super) font: String,
    pub(super) unit: f64,
    pub(super) spacing_staff: f64,
    pub(super) spacing_system: f64,
    pub(super) measure_min_width: f64,
    pub(super) spacing_linear: f64,
    pub(super) spacing_non_linear: f64,
    // no_justification: bool,
    pub(super) svg_bounding_boxes: bool,
    pub(super) svg_content_bounding_boxes: bool,
}

impl From<VerovioRenderSettings> for VerovioRenderSettingsPayload {
    fn from(settings: VerovioRenderSettings) -> Self {
        Self {
            page_width: settings.page_width,
            page_margin_left: settings.page_margin_left,
            page_margin_right: settings.page_margin_right,
            page_margin_top: settings.page_margin_top,
            page_margin_bottom: settings.page_margin_bottom,
            bar_line_width: round_two_decimals(settings.bar_line_width),
            beam_max_slope: settings.beam_max_slope,
            staff_line_width: round_two_decimals(settings.staff_line_width),
            stem_width: round_two_decimals(settings.stem_width),
            ledger_line_thickness: round_two_decimals(settings.ledger_line_thickness),
            thick_barline_thickness: round_two_decimals(settings.thick_barline_thickness),
            font: settings.font.as_str().to_owned(),
            unit: round_two_decimals(settings.unit),
            spacing_staff: round_two_decimals(settings.spacing_staff),
            spacing_system: round_two_decimals(settings.spacing_system),
            measure_min_width: round_two_decimals(settings.measure_min_width),
            spacing_linear: round_two_decimals(settings.spacing_linear),
            spacing_non_linear: round_two_decimals(settings.spacing_non_linear),
            // no_justification: settings.no_justification,
            svg_bounding_boxes: settings.svg_bounding_boxes,
            svg_content_bounding_boxes: settings.svg_content_bounding_boxes,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct SvgWhitespacePayload {
    pub(super) left: f64,
    pub(super) right: f64,
    pub(super) top: f64,
    pub(super) bottom: f64,
}

impl From<SvgPageWhitespace> for SvgWhitespacePayload {
    fn from(page: SvgPageWhitespace) -> Self {
        Self {
            left: round_two_decimals(page.whitespace.left),
            right: round_two_decimals(page.whitespace.right),
            top: round_two_decimals(page.whitespace.top),
            bottom: round_two_decimals(page.whitespace.bottom),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct SvgContentBoxPayload {
    pub(super) x: f64,
    pub(super) y: f64,
    pub(super) width: f64,
    pub(super) height: f64,
}

impl From<SvgContentBox> for SvgContentBoxPayload {
    fn from(content_box: SvgContentBox) -> Self {
        Self {
            x: round_two_decimals(content_box.x),
            y: round_two_decimals(content_box.y),
            width: round_two_decimals(content_box.width),
            height: round_two_decimals(content_box.height),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct GeometricTransformPayload {
    angle_deg: f64,
    translate_x: f64,
    translate_y: f64,
    scale_x: f64,
    scale_y: f64,
    center_x: f64,
    center_y: f64,
}

impl From<GeometricTransform> for GeometricTransformPayload {
    fn from(transform: GeometricTransform) -> Self {
        Self {
            angle_deg: round_four_decimals(transform.angle_deg),
            translate_x: round_two_decimals(transform.translate_x),
            translate_y: round_two_decimals(transform.translate_y),
            scale_x: round_six_decimals(transform.scale_x),
            scale_y: round_six_decimals(transform.scale_y),
            center_x: round_two_decimals(transform.center_x),
            center_y: round_two_decimals(transform.center_y),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum GeometricAugmentationStatusPayload {
    Applied,
    IdentityFallback,
    Disabled,
}

impl From<GeometricAugmentationStatus> for GeometricAugmentationStatusPayload {
    fn from(status: GeometricAugmentationStatus) -> Self {
        match status {
            GeometricAugmentationStatus::Applied => Self::Applied,
            GeometricAugmentationStatus::IdentityFallback => Self::IdentityFallback,
            GeometricAugmentationStatus::Disabled => Self::Disabled,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct GeometricAugmentationPayload {
    pub(super) status: GeometricAugmentationStatusPayload,
    pub(super) attempt_count: usize,
    pub(super) transform: Option<GeometricTransformPayload>,
    pub(super) transformed_content_box: SvgContentBoxPayload,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct SvgOverlapPairPayload {
    pub(super) first_id: String,
    pub(super) second_id: String,
    pub(super) first_kind: String,
    pub(super) second_kind: String,
    pub(super) area: f64,
    pub(super) ratio: f64,
}

impl From<&SvgOverlapPair> for SvgOverlapPairPayload {
    fn from(pair: &SvgOverlapPair) -> Self {
        Self {
            first_id: pair.first_id.clone(),
            second_id: pair.second_id.clone(),
            first_kind: pair.first_kind.clone(),
            second_kind: pair.second_kind.clone(),
            area: round_two_decimals(pair.area),
            ratio: round_six_decimals(pair.ratio),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct SvgOverlapGatePayload {
    pub(super) enabled: bool,
    pub(super) hit: bool,
    pub(super) rejected_attempt_count: usize,
    pub(super) fallback_attempt_count: usize,
    pub(super) worst_overlap_count: usize,
    pub(super) worst_overlap_ratio: Option<f64>,
    pub(super) worst_pairs: Vec<SvgOverlapPairPayload>,
}

impl SvgOverlapGatePayload {
    pub(super) fn from_report(report: &RenderFitQualityReport) -> Option<Self> {
        if !report.svg_overlap_gate_enabled {
            return None;
        }

        let worst_overlap = report.worst_svg_overlap.as_ref();
        Some(Self {
            enabled: true,
            hit: report.svg_overlap_gate_hit,
            rejected_attempt_count: report.svg_overlap_rejected_attempt_count,
            fallback_attempt_count: report.svg_overlap_fallback_attempt_count,
            worst_overlap_count: worst_overlap
                .map(|summary| summary.overlap_count)
                .unwrap_or(0),
            worst_overlap_ratio: worst_overlap
                .and_then(|summary| summary.worst_pairs.first())
                .map(|pair| round_six_decimals(pair.ratio)),
            worst_pairs: worst_overlap
                .map(|summary| {
                    summary
                        .worst_pairs
                        .iter()
                        .map(SvgOverlapPairPayload::from)
                        .collect()
                })
                .unwrap_or_default(),
        })
    }

    pub(super) fn from_options(options: RenderFitSearchOptions) -> Option<Self> {
        Self::from_report(&RenderFitQualityReport::for_options(options))
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FileStats {
    pub(super) line_count: usize,
    pub(super) character_count: usize,
    pub(super) initial_spine_count: Option<u32>,
    pub(super) staff_count: Option<usize>,
    pub(super) barline_count: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RenderSettingsSearchRow {
    pub(super) schema_version: String,
    pub(super) file: String,
    pub(super) line_count: usize,
    pub(super) character_count: usize,
    pub(super) initial_spine_count: Option<u32>,
    #[serde(default)]
    pub(super) staff_count: Option<usize>,
    pub(super) barline_count: Option<usize>,
    pub(super) status: RenderSettingsSearchStatus,
    pub(super) selected_settings: Option<VerovioRenderSettingsPayload>,
    pub(super) selected_score: Option<f64>,
    pub(super) selected_page_count: Option<u32>,
    #[serde(default)]
    pub(super) selected_system_count: Option<usize>,
    pub(super) selected_whitespace: Option<SvgWhitespacePayload>,
    #[serde(default)]
    pub(super) selected_content_box: Option<SvgContentBoxPayload>,
    #[serde(default)]
    pub(super) geometric_augmentation: Option<GeometricAugmentationPayload>,
    pub(super) selected_svg_file: Option<String>,
    #[serde(default)]
    pub(super) selected_svg_has_red_color: bool,
    #[serde(default)]
    pub(super) selected_render_source_lines: Option<Vec<Option<usize>>>,
    #[serde(default)]
    pub(super) svg_overlap_gate: Option<SvgOverlapGatePayload>,
    #[serde(default)]
    pub(super) render_augmentation: Option<RenderAugmentationReport>,
    pub(super) attempt_count: usize,
    pub(super) accepted_attempt_count: usize,
    pub(super) parse_error: Option<String>,
    pub(super) search_error: Option<String>,
    pub(super) verovio_stderr: Option<String>,
}

impl RenderSettingsSearchRow {
    pub(super) fn search_success(&self) -> bool {
        matches!(self.status, RenderSettingsSearchStatus::Success)
    }

    pub(super) fn parse_failed(&self) -> bool {
        matches!(self.status, RenderSettingsSearchStatus::ParseError)
    }

    pub(super) fn search_failed(&self) -> bool {
        matches!(self.status, RenderSettingsSearchStatus::SearchError)
    }

    pub(super) fn status_label(&self) -> &'static str {
        match self.status {
            RenderSettingsSearchStatus::Success => "success",
            RenderSettingsSearchStatus::ParseError => "parse_error",
            RenderSettingsSearchStatus::SearchError => "search_error",
        }
    }

    pub(super) fn parse_error(path: &Path, stats: FileStats, error: String) -> Self {
        Self {
            schema_version: RENDER_SETTINGS_SEARCH_SCHEMA_VERSION.to_owned(),
            file: path.display().to_string(),
            line_count: stats.line_count,
            character_count: stats.character_count,
            initial_spine_count: stats.initial_spine_count,
            staff_count: stats.staff_count,
            barline_count: stats.barline_count,
            status: RenderSettingsSearchStatus::ParseError,
            selected_settings: None,
            selected_score: None,
            selected_page_count: None,
            selected_system_count: None,
            selected_whitespace: None,
            selected_content_box: None,
            geometric_augmentation: None,
            selected_svg_file: None,
            selected_svg_has_red_color: false,
            selected_render_source_lines: None,
            svg_overlap_gate: None,
            render_augmentation: None,
            attempt_count: 0,
            accepted_attempt_count: 0,
            parse_error: Some(error),
            search_error: None,
            verovio_stderr: None,
        }
    }

    pub(super) fn search_error(path: &Path, stats: FileStats, search_error: String) -> Self {
        Self {
            schema_version: RENDER_SETTINGS_SEARCH_SCHEMA_VERSION.to_owned(),
            file: path.display().to_string(),
            line_count: stats.line_count,
            character_count: stats.character_count,
            initial_spine_count: stats.initial_spine_count,
            staff_count: stats.staff_count,
            barline_count: stats.barline_count,
            status: RenderSettingsSearchStatus::SearchError,
            selected_settings: None,
            selected_score: None,
            selected_page_count: None,
            selected_system_count: None,
            selected_whitespace: None,
            selected_content_box: None,
            geometric_augmentation: None,
            selected_svg_file: None,
            selected_svg_has_red_color: false,
            selected_render_source_lines: None,
            svg_overlap_gate: None,
            render_augmentation: None,
            attempt_count: 0,
            accepted_attempt_count: 0,
            parse_error: None,
            search_error: Some(search_error),
            verovio_stderr: None,
        }
    }

    pub(super) fn worker_failure(
        path: &Path,
        search_error: String,
        verovio_stderr: Option<String>,
    ) -> Self {
        Self {
            schema_version: RENDER_SETTINGS_SEARCH_SCHEMA_VERSION.to_owned(),
            file: path.display().to_string(),
            line_count: 0,
            character_count: 0,
            initial_spine_count: None,
            staff_count: None,
            barline_count: None,
            status: RenderSettingsSearchStatus::SearchError,
            selected_settings: None,
            selected_score: None,
            selected_page_count: None,
            selected_system_count: None,
            selected_whitespace: None,
            selected_content_box: None,
            geometric_augmentation: None,
            selected_svg_file: None,
            selected_svg_has_red_color: false,
            selected_render_source_lines: None,
            svg_overlap_gate: None,
            render_augmentation: None,
            attempt_count: 0,
            accepted_attempt_count: 0,
            parse_error: None,
            search_error: Some(search_error),
            verovio_stderr,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn success(
        path: &Path,
        stats: FileStats,
        result: RenderFitResult,
        selected_system_count: usize,
        selected_content_box: SvgContentBox,
        geometric_augmentation: GeometricAugmentationPayload,
        selected_svg_path: &Path,
        selected_svg_has_red_color: bool,
        render_augmentation: RenderAugmentationReport,
    ) -> Self {
        let selected_page = result
            .selected_pages
            .first()
            .expect("caller validated exactly one selected render page");
        Self {
            schema_version: RENDER_SETTINGS_SEARCH_SCHEMA_VERSION.to_owned(),
            file: path.display().to_string(),
            line_count: stats.line_count,
            character_count: stats.character_count,
            initial_spine_count: stats.initial_spine_count,
            staff_count: stats.staff_count,
            barline_count: stats.barline_count,
            status: RenderSettingsSearchStatus::Success,
            selected_settings: Some(result.settings.into()),
            selected_score: Some(result.score),
            selected_page_count: Some(result.page_count),
            selected_system_count: Some(selected_system_count),
            selected_whitespace: Some(selected_page.whitespace.into()),
            selected_content_box: Some(selected_content_box.into()),
            geometric_augmentation: Some(geometric_augmentation),
            selected_svg_file: Some(selected_svg_path.display().to_string()),
            selected_svg_has_red_color,
            selected_render_source_lines: Some(result.selected_render_source_lines),
            svg_overlap_gate: SvgOverlapGatePayload::from_report(&result.quality_report),
            render_augmentation: Some(render_augmentation),
            attempt_count: result.attempts.len(),
            accepted_attempt_count: result
                .attempts
                .iter()
                .filter(|attempt| attempt.page_count == 1 && attempt.score.is_some())
                .count(),
            parse_error: None,
            search_error: None,
            verovio_stderr: None,
        }
    }

    pub(super) fn with_svg_overlap_gate(
        mut self,
        svg_overlap_gate: Option<SvgOverlapGatePayload>,
    ) -> Self {
        self.svg_overlap_gate = svg_overlap_gate;
        self
    }
}

pub(super) fn blank_file_stats() -> FileStats {
    FileStats {
        line_count: 0,
        character_count: 0,
        initial_spine_count: None,
        staff_count: None,
        barline_count: None,
    }
}

fn round_four_decimals(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn round_six_decimals(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn round_two_decimals(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}
