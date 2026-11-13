use crate::analysis::{
    RenderFitAttemptOrigin, RenderFitAttemptOutcome, RenderFitAttemptTiming, RenderFitTimingReport,
};
use serde::{Deserialize, Serialize};

use super::row::RenderSettingsSearchRow;

pub(super) const RENDER_SETTINGS_SEARCH_PROFILE_SCHEMA_VERSION: &str = "1.0";

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(super) struct RenderSettingsSearchPhaseTimings {
    pub(super) total: f64,
    pub(super) read_input: f64,
    pub(super) parse_document: f64,
    pub(super) original_file_stats: f64,
    pub(super) derive_svg_path: f64,
    pub(super) augment_kern: f64,
    pub(super) render_fit_search: f64,
    pub(super) selected_svg_postprocess: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) worker_process: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct RenderSettingsSearchProfileRow {
    pub(super) schema_version: String,
    pub(super) file: String,
    pub(super) status: String,
    pub(super) line_count: usize,
    pub(super) character_count: usize,
    pub(super) initial_spine_count: Option<u32>,
    pub(super) staff_count: Option<usize>,
    pub(super) barline_count: Option<usize>,
    pub(super) attempt_count: usize,
    pub(super) accepted_attempt_count: usize,
    pub(super) timings_ms: RenderSettingsSearchPhaseTimings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) render_fit: Option<RenderFitTimingPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) parse_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) search_error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct RenderFitTimingPayload {
    pub(super) candidate_count: usize,
    pub(super) attempt_budget: usize,
    pub(super) attempts: Vec<RenderFitAttemptTimingPayload>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct RenderFitAttemptTimingPayload {
    pub(super) attempt_index: usize,
    pub(super) origin: String,
    pub(super) render_ms: f64,
    pub(super) evaluate_ms: f64,
    pub(super) page_count: Option<u32>,
    pub(super) outcome: String,
}

#[derive(Debug)]
pub(super) struct ProfiledRenderSettingsSearchRow {
    pub(super) row: RenderSettingsSearchRow,
    pub(super) profile: Option<RenderSettingsSearchProfileRow>,
}

impl From<RenderSettingsSearchRow> for ProfiledRenderSettingsSearchRow {
    fn from(row: RenderSettingsSearchRow) -> Self {
        Self { row, profile: None }
    }
}

impl RenderSettingsSearchProfileRow {
    pub(super) fn from_row(
        row: &RenderSettingsSearchRow,
        timings_ms: RenderSettingsSearchPhaseTimings,
        render_fit: Option<RenderFitTimingReport>,
    ) -> Self {
        Self {
            schema_version: RENDER_SETTINGS_SEARCH_PROFILE_SCHEMA_VERSION.to_owned(),
            file: row.file.clone(),
            status: row.status_label().to_owned(),
            line_count: row.line_count,
            character_count: row.character_count,
            initial_spine_count: row.initial_spine_count,
            staff_count: row.staff_count,
            barline_count: row.barline_count,
            attempt_count: row.attempt_count,
            accepted_attempt_count: row.accepted_attempt_count,
            timings_ms,
            render_fit: render_fit.map(RenderFitTimingPayload::from),
            parse_error: row.parse_error.clone(),
            search_error: row.search_error.clone(),
        }
    }

    pub(super) fn with_worker_process_millis(mut self, worker_process_millis: f64) -> Self {
        self.timings_ms.worker_process = Some(round_millis(worker_process_millis));
        self
    }
}

impl From<RenderFitTimingReport> for RenderFitTimingPayload {
    fn from(report: RenderFitTimingReport) -> Self {
        Self {
            candidate_count: report.candidate_count,
            attempt_budget: report.attempt_budget,
            attempts: report
                .attempts
                .into_iter()
                .map(RenderFitAttemptTimingPayload::from)
                .collect(),
        }
    }
}

impl From<RenderFitAttemptTiming> for RenderFitAttemptTimingPayload {
    fn from(attempt: RenderFitAttemptTiming) -> Self {
        Self {
            attempt_index: attempt.attempt_index,
            origin: attempt_origin_label(attempt.origin).to_owned(),
            render_ms: round_millis(attempt.render_millis),
            evaluate_ms: round_millis(attempt.evaluate_millis),
            page_count: attempt.page_count,
            outcome: attempt_outcome_label(attempt.outcome).to_owned(),
        }
    }
}

pub(super) fn round_millis(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn attempt_origin_label(origin: RenderFitAttemptOrigin) -> &'static str {
    match origin {
        RenderFitAttemptOrigin::Primary => "primary",
        RenderFitAttemptOrigin::SvgOverlapFallback => "svg_overlap_fallback",
    }
}

fn attempt_outcome_label(outcome: RenderFitAttemptOutcome) -> &'static str {
    match outcome {
        RenderFitAttemptOutcome::Accepted => "accepted",
        RenderFitAttemptOutcome::PageCountMismatch => "page_count_mismatch",
        RenderFitAttemptOutcome::Whitespace => "whitespace",
        RenderFitAttemptOutcome::SvgOverlap => "svg_overlap",
        RenderFitAttemptOutcome::RenderError => "render_error",
        RenderFitAttemptOutcome::EvaluateError => "evaluate_error",
    }
}
