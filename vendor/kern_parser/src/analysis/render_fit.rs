//! Search scaffolding for finding Verovio render settings that fit a document well.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::{
    analysis::{
        detect_svg_bbox_overlaps, measure_svg_whitespace, SvgOverlapOptions, SvgOverlapSummary,
        SvgPageWhitespace, VerovioRenderSettings,
    },
    KernDocument,
};

mod config;
mod rendering;
mod result;
mod sampling;
mod scoring;
mod sizing;

#[cfg(test)]
use crate::analysis::VerovioRenderSettingsBounds;
use config::validate_config;
#[cfg(test)]
use config::{
    preferred_bounds_for_profile, HIGH_STAFF_MIN_PAGE_WIDTH, MULTI_SPINE_MIN_PAGE_MARGIN_LEFT,
};
pub use config::{RenderFitConfig, RenderFitDocumentProfile};
use rendering::{render_candidate, RenderedCandidatePages};
pub use result::{
    RenderFitAttempt, RenderFitAttemptOrigin, RenderFitAttemptOutcome, RenderFitAttemptTiming,
    RenderFitError, RenderFitQualityReport, RenderFitResult, RenderFitSearchFailure,
    RenderFitSearchOptions, RenderFitSearchSeed, RenderFitTimingReport, RenderedPageArtifact,
    SvgOverlapFallbackMode, SvgOverlapGateMode,
};
#[cfg(test)]
use sampling::valid_page_widths;
#[cfg(test)]
use sampling::MAX_HORIZONTAL_SIDE_MARGIN_RATIO;
#[cfg(test)]
use sampling::{
    drawable_width_ratio, max_total_horizontal_margin, sample_horizontal_margins,
    seeded_low_discrepancy_candidate, MIN_DRAWABLE_WIDTH_RATIO,
};
use sampling::{
    fast_seeded_render_setting_candidates, overlap_horizontal_relief_candidates,
    render_setting_candidates,
};
use scoring::{candidate_fits_page, score_candidate};
pub use sizing::RenderedUnitBounds;

/// Find Verovio render settings for `document` by sampling within configured bounds.
///
/// The primary goal of this search is dataset diversity: selected settings
/// should be drawn from across the product space defined by
/// [`RenderFitConfig::bounds`], so outputs generated for many scores cover a
/// broad range of valid Verovio render styles rather than clustering around one
/// hand-tuned default. The exact sampling strategy is an implementation detail,
/// but it should be deterministic and should avoid obvious bias toward the
/// center of each range.
///
/// The secondary goal is fit: returned settings must render `document` to
/// `config.target_pages` pages, which is expected to be one page for OMR
/// generation. If an initially sampled candidate does not fit, the search may
/// probe additional candidates, including candidates biased toward using less
/// vertical or horizontal space. This fallback strategy is also an
/// implementation detail, but every attempted setting must remain within
/// `config.bounds`.
///
/// Return [`RenderFitError`] if no in-bounds candidate satisfying the target
/// page count is found within the configured attempt budget. Successful results
/// should include the selected settings, measured whitespace, all attempted
/// candidates, and selected rendered SVG page artifacts so downstream pipeline
/// stages can reuse the render without invoking Verovio again.
pub fn find_render_settings(
    document: &KernDocument,
    initial: VerovioRenderSettings,
    config: RenderFitConfig,
) -> Result<RenderFitResult, RenderFitError> {
    validate_config(config, initial)?;
    let candidates = render_setting_candidates(initial, config);
    find_render_settings_with_candidates(
        document,
        initial,
        config,
        candidates,
        RenderFitSearchOptions::default(),
    )
    .map_err(|failure| failure.error)
}

pub fn find_seeded_render_settings(
    document: &KernDocument,
    initial: VerovioRenderSettings,
    config: RenderFitConfig,
    seed: RenderFitSearchSeed,
) -> Result<RenderFitResult, RenderFitError> {
    find_seeded_render_settings_with_options(
        document,
        initial,
        config,
        seed,
        RenderFitSearchOptions::default(),
    )
    .map_err(|failure| failure.error)
}

pub fn find_seeded_render_settings_with_options(
    document: &KernDocument,
    initial: VerovioRenderSettings,
    config: RenderFitConfig,
    seed: RenderFitSearchSeed,
    options: RenderFitSearchOptions,
) -> Result<RenderFitResult, RenderFitSearchFailure> {
    let quality_report = RenderFitQualityReport::for_options(options);
    if seed.count == 0 {
        return Err(search_failure(
            RenderFitError::InvalidSearchSeed,
            quality_report,
        ));
    }
    validate_config(config, initial)
        .map_err(|error| search_failure(error, quality_report.clone()))?;
    validate_search_options(options).map_err(|error| search_failure(error, quality_report))?;
    let candidates = fast_seeded_render_setting_candidates(initial, config, seed);
    find_render_settings_with_candidates(document, initial, config, candidates, options)
}

pub fn find_seeded_render_settings_with_options_and_timing(
    document: &KernDocument,
    initial: VerovioRenderSettings,
    config: RenderFitConfig,
    seed: RenderFitSearchSeed,
    options: RenderFitSearchOptions,
    timing_report: &mut RenderFitTimingReport,
) -> Result<RenderFitResult, RenderFitSearchFailure> {
    let quality_report = RenderFitQualityReport::for_options(options);
    if seed.count == 0 {
        return Err(search_failure(
            RenderFitError::InvalidSearchSeed,
            quality_report,
        ));
    }
    validate_config(config, initial)
        .map_err(|error| search_failure(error, quality_report.clone()))?;
    validate_search_options(options).map_err(|error| search_failure(error, quality_report))?;
    let candidates = fast_seeded_render_setting_candidates(initial, config, seed);
    find_render_settings_with_candidates_and_timing(
        document,
        initial,
        config,
        candidates,
        options,
        timing_report,
    )
}

fn find_render_settings_with_candidates(
    document: &KernDocument,
    initial: VerovioRenderSettings,
    config: RenderFitConfig,
    candidates: Vec<VerovioRenderSettings>,
    options: RenderFitSearchOptions,
) -> Result<RenderFitResult, RenderFitSearchFailure> {
    find_render_settings_with_candidates_and_renderer(
        document,
        initial,
        config,
        candidates,
        options,
        render_candidate,
        None,
    )
}

fn find_render_settings_with_candidates_and_timing(
    document: &KernDocument,
    initial: VerovioRenderSettings,
    config: RenderFitConfig,
    candidates: Vec<VerovioRenderSettings>,
    options: RenderFitSearchOptions,
    timing_report: &mut RenderFitTimingReport,
) -> Result<RenderFitResult, RenderFitSearchFailure> {
    find_render_settings_with_candidates_and_renderer(
        document,
        initial,
        config,
        candidates,
        options,
        render_candidate,
        Some(timing_report),
    )
}

fn find_render_settings_with_candidates_and_renderer<R>(
    document: &KernDocument,
    initial: VerovioRenderSettings,
    config: RenderFitConfig,
    candidates: Vec<VerovioRenderSettings>,
    options: RenderFitSearchOptions,
    mut renderer: R,
    mut timing_report: Option<&mut RenderFitTimingReport>,
) -> Result<RenderFitResult, RenderFitSearchFailure>
where
    R: FnMut(
        &KernDocument,
        VerovioRenderSettings,
        u32,
    ) -> Result<RenderedCandidatePages, RenderFitError>,
{
    let mut quality_report = RenderFitQualityReport::for_options(options);
    validate_config(config, initial)
        .map_err(|error| search_failure(error, quality_report.clone()))?;
    validate_search_options(options)
        .map_err(|error| search_failure(error, quality_report.clone()))?;

    let attempt_budget = candidates.len().min(config.max_attempts);
    if let Some(timing_report) = timing_report.as_mut() {
        timing_report.candidate_count = candidates.len();
        timing_report.attempt_budget = attempt_budget;
        timing_report.attempts.clear();
    }
    let mut seen_candidates = candidates.clone();
    let mut queue = candidates
        .into_iter()
        .map(|settings| ScheduledCandidate {
            settings,
            origin: CandidateOrigin::Primary,
        })
        .collect::<VecDeque<_>>();
    let mut attempts = Vec::new();

    while attempts.len() < attempt_budget {
        let Some(scheduled) = queue.pop_front() else {
            break;
        };
        let settings = scheduled.settings;
        if scheduled.origin == CandidateOrigin::SvgOverlapFallback {
            quality_report.record_svg_overlap_fallback_attempt();
        }

        let attempt_index = attempts.len() + 1;
        let render_start = Instant::now();
        let rendered = match renderer(document, settings, config.target_pages) {
            Ok(rendered) => rendered,
            Err(error) => {
                record_attempt_timing(
                    timing_report_mut(&mut timing_report),
                    RenderFitAttemptTiming {
                        attempt_index,
                        origin: scheduled.origin.into(),
                        render_millis: millis(render_start.elapsed()),
                        evaluate_millis: 0.0,
                        page_count: None,
                        outcome: RenderFitAttemptOutcome::RenderError,
                    },
                );
                return Err(search_failure(error, quality_report.clone()));
            }
        };
        let render_millis = millis(render_start.elapsed());

        let evaluate_start = Instant::now();
        let outcome =
            match evaluate_rendered_candidate(&rendered, initial, settings, config, options) {
                Ok(outcome) => outcome,
                Err(error) => {
                    record_attempt_timing(
                        timing_report_mut(&mut timing_report),
                        RenderFitAttemptTiming {
                            attempt_index,
                            origin: scheduled.origin.into(),
                            render_millis,
                            evaluate_millis: millis(evaluate_start.elapsed()),
                            page_count: Some(rendered.page_count),
                            outcome: RenderFitAttemptOutcome::EvaluateError,
                        },
                    );
                    return Err(search_failure(error, quality_report.clone()));
                }
            };
        let evaluate_millis = millis(evaluate_start.elapsed());
        let outcome_kind = (&outcome).into();
        let (whitespace, score) = outcome.attempt_fields();

        attempts.push(RenderFitAttempt {
            settings,
            page_count: rendered.page_count,
            whitespace,
            score,
        });
        record_attempt_timing(
            timing_report_mut(&mut timing_report),
            RenderFitAttemptTiming {
                attempt_index,
                origin: scheduled.origin.into(),
                render_millis,
                evaluate_millis,
                page_count: Some(rendered.page_count),
                outcome: outcome_kind,
            },
        );

        match outcome {
            CandidateOutcome::Accepted(candidate) => {
                return Ok(render_fit_result(candidate, attempts, quality_report));
            }
            CandidateOutcome::Rejected(CandidateRejection::SvgOverlap { summary, .. }) => {
                quality_report.record_svg_overlap_rejection(summary);
                enqueue_svg_overlap_fallbacks(
                    &mut queue,
                    &mut seen_candidates,
                    settings,
                    config,
                    options,
                );
            }
            CandidateOutcome::Rejected(CandidateRejection::PageCountMismatch)
            | CandidateOutcome::Rejected(CandidateRejection::Whitespace { .. }) => {}
        }
    }

    Err(search_failure(
        RenderFitError::NoFittingCandidate {
            target_pages: config.target_pages,
            attempt_count: attempts.len(),
        },
        quality_report,
    ))
}

fn record_attempt_timing(
    timing_report: Option<&mut RenderFitTimingReport>,
    attempt: RenderFitAttemptTiming,
) {
    if let Some(timing_report) = timing_report {
        timing_report.attempts.push(attempt);
    }
}

fn timing_report_mut<'a>(
    timing_report: &'a mut Option<&mut RenderFitTimingReport>,
) -> Option<&'a mut RenderFitTimingReport> {
    timing_report.as_mut().map(|report| &mut **report)
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn evaluate_rendered_candidate(
    rendered: &RenderedCandidatePages,
    initial: VerovioRenderSettings,
    settings: VerovioRenderSettings,
    config: RenderFitConfig,
    options: RenderFitSearchOptions,
) -> Result<CandidateOutcome, RenderFitError> {
    if rendered.page_count != config.target_pages {
        return Ok(CandidateOutcome::Rejected(
            CandidateRejection::PageCountMismatch,
        ));
    }

    let mut selected_pages = Vec::with_capacity(rendered.svgs.len());
    let mut overlap_summaries = Vec::new();
    let overlap_options = svg_overlap_options(options);
    for (index, svg) in rendered.svgs.iter().enumerate() {
        let page_number = index as u32 + 1;
        let whitespace = measure_svg_whitespace(svg).map_err(|error| {
            RenderFitError::WhitespaceMeasurementFailed {
                page_number,
                message: error.to_string(),
            }
        })?;
        if let Some(overlap_options) = &overlap_options {
            let summary =
                detect_svg_bbox_overlaps(svg, page_number, overlap_options).map_err(|error| {
                    RenderFitError::SvgOverlapDetectionFailed {
                        page_number,
                        message: error.to_string(),
                    }
                })?;
            if summary.overlap_count > 0 {
                overlap_summaries.push(summary);
            }
        }
        selected_pages.push(RenderedPageArtifact {
            page_number,
            svg: svg.clone(),
            whitespace,
        });
    }

    let whitespace = selected_pages[0].whitespace;
    if let Some(summary) = worst_overlap_summary(overlap_summaries) {
        return Ok(CandidateOutcome::Rejected(CandidateRejection::SvgOverlap {
            summary,
            whitespace,
        }));
    }

    if !candidate_fits_page(whitespace, config) {
        return Ok(CandidateOutcome::Rejected(CandidateRejection::Whitespace {
            whitespace,
        }));
    }

    let score = score_candidate(initial, settings, whitespace, config);
    Ok(CandidateOutcome::Accepted(RenderedCandidate {
        settings,
        page_count: rendered.page_count,
        whitespace,
        score,
        selected_pages,
        selected_render_source_lines: rendered.render_source_lines.clone(),
    }))
}

fn svg_overlap_options(options: RenderFitSearchOptions) -> Option<SvgOverlapOptions> {
    if !options.svg_overlap_enabled() {
        return None;
    }

    Some(SvgOverlapOptions {
        min_overlap_ratio: options.svg_overlap_min_ratio,
        ..SvgOverlapOptions::default()
    })
}

fn worst_overlap_summary(summaries: Vec<SvgOverlapSummary>) -> Option<SvgOverlapSummary> {
    summaries.into_iter().max_by(|a, b| {
        let a_pair = a.worst_pairs.first();
        let b_pair = b.worst_pairs.first();
        a.overlap_count
            .cmp(&b.overlap_count)
            .then_with(|| {
                a_pair
                    .map(|pair| pair.ratio)
                    .unwrap_or(0.0)
                    .total_cmp(&b_pair.map(|pair| pair.ratio).unwrap_or(0.0))
            })
            .then_with(|| {
                a_pair
                    .map(|pair| pair.area)
                    .unwrap_or(0.0)
                    .total_cmp(&b_pair.map(|pair| pair.area).unwrap_or(0.0))
            })
    })
}

fn enqueue_svg_overlap_fallbacks(
    queue: &mut VecDeque<ScheduledCandidate>,
    seen_candidates: &mut Vec<VerovioRenderSettings>,
    settings: VerovioRenderSettings,
    config: RenderFitConfig,
    options: RenderFitSearchOptions,
) {
    if options.svg_overlap_fallback != SvgOverlapFallbackMode::HorizontalRelief {
        return;
    }

    let mut fallback_candidates = Vec::new();
    for candidate in overlap_horizontal_relief_candidates(settings, config) {
        if let Some(position) = queue.iter().position(|queued| queued.settings == candidate) {
            queue.remove(position);
            fallback_candidates.push(candidate);
        } else if !seen_candidates.contains(&candidate) {
            seen_candidates.push(candidate);
            fallback_candidates.push(candidate);
        }
    }
    while let Some(candidate) = fallback_candidates.pop() {
        queue.push_front(ScheduledCandidate {
            settings: candidate,
            origin: CandidateOrigin::SvgOverlapFallback,
        });
    }
}

fn validate_search_options(options: RenderFitSearchOptions) -> Result<(), RenderFitError> {
    if options.svg_overlap_enabled()
        && (!options.svg_overlap_min_ratio.is_finite() || options.svg_overlap_min_ratio < 0.0)
    {
        return Err(RenderFitError::InvalidWeight {
            setting: "svg_overlap_min_ratio",
        });
    }

    Ok(())
}

fn render_fit_result(
    candidate: RenderedCandidate,
    attempts: Vec<RenderFitAttempt>,
    quality_report: RenderFitQualityReport,
) -> RenderFitResult {
    RenderFitResult {
        settings: candidate.settings,
        page_count: candidate.page_count,
        whitespace: candidate.whitespace,
        score: candidate.score,
        attempts,
        selected_pages: candidate.selected_pages,
        selected_render_source_lines: candidate.selected_render_source_lines,
        quality_report,
    }
}

fn search_failure(
    error: RenderFitError,
    quality_report: RenderFitQualityReport,
) -> RenderFitSearchFailure {
    RenderFitSearchFailure {
        error,
        quality_report,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateOrigin {
    Primary,
    SvgOverlapFallback,
}

impl From<CandidateOrigin> for RenderFitAttemptOrigin {
    fn from(origin: CandidateOrigin) -> Self {
        match origin {
            CandidateOrigin::Primary => Self::Primary,
            CandidateOrigin::SvgOverlapFallback => Self::SvgOverlapFallback,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ScheduledCandidate {
    settings: VerovioRenderSettings,
    origin: CandidateOrigin,
}

#[derive(Debug, Clone, PartialEq)]
enum CandidateOutcome {
    Accepted(RenderedCandidate),
    Rejected(CandidateRejection),
}

impl CandidateOutcome {
    fn attempt_fields(&self) -> (Option<SvgPageWhitespace>, Option<f64>) {
        match self {
            Self::Accepted(candidate) => (Some(candidate.whitespace), Some(candidate.score)),
            Self::Rejected(CandidateRejection::Whitespace { whitespace }) => {
                (Some(*whitespace), None)
            }
            Self::Rejected(CandidateRejection::PageCountMismatch) => (None, None),
            Self::Rejected(CandidateRejection::SvgOverlap { whitespace, .. }) => {
                (Some(*whitespace), None)
            }
        }
    }
}

impl From<&CandidateOutcome> for RenderFitAttemptOutcome {
    fn from(outcome: &CandidateOutcome) -> Self {
        match outcome {
            CandidateOutcome::Accepted(_) => Self::Accepted,
            CandidateOutcome::Rejected(CandidateRejection::PageCountMismatch) => {
                Self::PageCountMismatch
            }
            CandidateOutcome::Rejected(CandidateRejection::Whitespace { .. }) => Self::Whitespace,
            CandidateOutcome::Rejected(CandidateRejection::SvgOverlap { .. }) => Self::SvgOverlap,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum CandidateRejection {
    PageCountMismatch,
    Whitespace {
        whitespace: SvgPageWhitespace,
    },
    SvgOverlap {
        summary: SvgOverlapSummary,
        whitespace: SvgPageWhitespace,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct RenderedCandidate {
    settings: VerovioRenderSettings,
    page_count: u32,
    whitespace: SvgPageWhitespace,
    score: f64,
    selected_pages: Vec<RenderedPageArtifact>,
    selected_render_source_lines: Vec<Option<usize>>,
}

#[cfg(test)]
mod tests;
