use super::*;
use crate::analysis::{FloatRange, IntRange, SvgContentBox, SvgWhitespace};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {actual} to be close to {expected}"
    );
}

fn overlap_test_svg(overlap: bool) -> String {
    let second_x = if overlap { 5.0 } else { 100.0 };
    format!(
        r#"<svg viewBox="0 0 1050 1485">
  <g class="system content-bounding-box"><rect x="50" y="50" width="900" height="1000"/></g>
  <g id="bbox-note-a" class="note bounding-box"><rect x="0" y="0" width="10" height="10"/></g>
  <g id="bbox-note-b" class="note bounding-box"><rect x="{second_x}" y="5" width="10" height="10"/></g>
</svg>"#
    )
}

#[test]
fn rendered_unit_policy_derives_unit_ranges_from_page_width() {
    let bounds = VerovioRenderSettingsBounds::single_page_search();
    let policy = RenderedUnitBounds::default_for_single_page_search();

    let wide = policy.unit_range_for_page_width(bounds.unit, 3500).unwrap();
    let narrow = policy.unit_range_for_page_width(bounds.unit, 1250).unwrap();

    assert_close(wide.min, 9.0);
    assert_close(wide.max, 12.0);
    assert_close(narrow.min, 6.0);
    assert_close(narrow.max, 10.0);
}

#[test]
fn single_page_config_targets_one_page() {
    let config = RenderFitConfig::single_page_balanced();

    assert_eq!(config.target_pages, 1);
    assert!(config.max_attempts > 0);
    config.bounds.validate().unwrap();
}

#[test]
fn document_config_raises_left_margin_for_multi_spine_documents() {
    let single_spine = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
    let multi_spine = KernDocument::parse("**kern\t**kern\n4c\t4e\n*-\t*-\n").unwrap();

    assert_eq!(
        RenderFitConfig::single_page_balanced_for_document(&single_spine)
            .bounds
            .page_margin_left
            .min,
        0
    );
    assert_eq!(
        RenderFitConfig::single_page_balanced_for_document(&multi_spine)
            .bounds
            .page_margin_left
            .min,
        MULTI_SPINE_MIN_PAGE_MARGIN_LEFT
    );
}

#[test]
fn document_config_uses_kern_staff_count_for_preferred_bounds() {
    let high_staff_with_dynamics = KernDocument::parse(
        "**kern\t**dynam\t**kern\t**dynam\t**kern\t**dynam\t**kern\t**kern\t**kern\t**kern\n\
             4c\tf\t4d\tf\t4e\tf\t4f\t4g\t4a\t4b\n\
             *-\t*-\t*-\t*-\t*-\t*-\t*-\t*-\t*-\t*-\n",
    )
    .unwrap();
    let config = RenderFitConfig::single_page_balanced_for_document(&high_staff_with_dynamics);

    assert_eq!(
        RenderFitDocumentProfile::for_document(&high_staff_with_dynamics).staff_count,
        7
    );
    assert_eq!(config.preferred_bounds.spacing_staff.max, 8.0);
    assert_eq!(config.preferred_bounds.page_margin_top.max, 150);
    assert_eq!(config.preferred_bounds.page_margin_bottom.max, 150);
    assert!(config.preferred_bounds.page_width.min >= HIGH_STAFF_MIN_PAGE_WIDTH);
    assert_eq!(
        config.bounds.spacing_staff.max,
        VerovioRenderSettingsBounds::single_page_search()
            .spacing_staff
            .max
    );
}

#[test]
fn preferred_bounds_snap_high_staff_width_floor_to_valid_page_width() {
    let mut bounds = VerovioRenderSettingsBounds::single_page_search();
    bounds.page_width = IntRange {
        min: 1400,
        max: 1749,
    };
    let preferred = preferred_bounds_for_profile(
        bounds,
        RenderFitDocumentProfile {
            staff_count: 7,
            total_spine_count: 7,
            barline_count: 0,
        },
    );

    assert_eq!(preferred.page_width.min, 1500);
    assert_eq!(valid_page_widths(preferred), vec![1500]);
}

#[test]
#[cfg(feature = "engraving")]
fn search_returns_single_page_render_settings() {
    let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
    let result = find_render_settings(
        &document,
        VerovioRenderSettings::medium(),
        RenderFitConfig::single_page_balanced_for_document(&document),
    )
    .unwrap();

    assert_eq!(result.page_count, 1);
    assert_eq!(result.selected_pages.len(), 1);
    assert_eq!(result.selected_pages[0].page_number, 1);
    assert!(result.selected_pages[0].svg.contains("<svg"));
    assert!(result.whitespace.content_box.width > 0.0);
    assert!(result.whitespace.content_box.height > 0.0);
    assert!(!result.attempts.is_empty());
    assert_eq!(result.attempts.len(), 1);
    assert!(result
        .attempts
        .iter()
        .any(|attempt| attempt.score.is_some()));
}

#[test]
#[cfg(feature = "engraving")]
fn fast_search_returns_after_first_fitting_candidate() {
    let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
    let config = RenderFitConfig::single_page_balanced();
    let result = find_render_settings(&document, VerovioRenderSettings::medium(), config).unwrap();

    assert_eq!(result.page_count, 1);
    assert_eq!(result.selected_pages.len(), 1);
    assert_eq!(result.attempts.len(), 1);
    assert!(result.attempts[0].score.is_some());
}

#[test]
fn svg_overlap_gate_fast_mode_tries_horizontal_relief_before_accepting() {
    let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
    let initial = VerovioRenderSettings::medium();
    let mut config = RenderFitConfig::single_page_balanced();
    config.max_attempts = 8;
    let first = seeded_low_discrepancy_candidate(0, config);
    let candidates = fast_seeded_render_setting_candidates(
        initial,
        config,
        RenderFitSearchSeed { index: 0, count: 8 },
    );
    let options = RenderFitSearchOptions {
        svg_overlap_gate: SvgOverlapGateMode::Reject,
        svg_overlap_min_ratio: 0.15,
        svg_overlap_fallback: SvgOverlapFallbackMode::HorizontalRelief,
    };

    let result = find_render_settings_with_candidates_and_renderer(
        &document,
        initial,
        config,
        candidates,
        options,
        |_document, settings, _target_pages| {
            Ok(RenderedCandidatePages {
                page_count: 1,
                svgs: vec![if settings == first {
                    overlap_test_svg(true)
                } else {
                    overlap_test_svg(false)
                }],
                render_source_lines: vec![Some(1), None, Some(3)],
            })
        },
        None,
    )
    .unwrap();

    assert_eq!(
        result.selected_render_source_lines,
        vec![Some(1), None, Some(3)]
    );
    assert!(result.quality_report.svg_overlap_gate_hit);
    assert_eq!(result.quality_report.svg_overlap_rejected_attempt_count, 1);
    assert_eq!(result.quality_report.svg_overlap_fallback_attempt_count, 1);
    assert_eq!(result.attempts.len(), 2);
    assert_eq!(result.attempts[0].settings, first);
    assert!(result.attempts[0].score.is_none());
    assert!(result.attempts[1].settings.page_width >= first.page_width);
    assert!(result.attempts[1].score.is_some());
}

#[test]
fn svg_overlap_gate_reports_unresolved_overlap_rejections() {
    let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
    let initial = VerovioRenderSettings::medium();
    let mut config = RenderFitConfig::single_page_balanced();
    config.max_attempts = 4;
    let candidates = fast_seeded_render_setting_candidates(
        initial,
        config,
        RenderFitSearchSeed { index: 0, count: 8 },
    );
    let options = RenderFitSearchOptions {
        svg_overlap_gate: SvgOverlapGateMode::Reject,
        svg_overlap_min_ratio: 0.15,
        svg_overlap_fallback: SvgOverlapFallbackMode::HorizontalRelief,
    };

    let failure = find_render_settings_with_candidates_and_renderer(
        &document,
        initial,
        config,
        candidates,
        options,
        |_document, _settings, _target_pages| {
            Ok(RenderedCandidatePages {
                page_count: 1,
                svgs: vec![overlap_test_svg(true)],
                render_source_lines: vec![Some(1), None, Some(3)],
            })
        },
        None,
    )
    .unwrap_err();

    assert!(matches!(
        failure.error,
        RenderFitError::NoFittingCandidate { .. }
    ));
    assert!(failure.quality_report.svg_overlap_gate_hit);
    assert!(failure.quality_report.svg_overlap_rejected_attempt_count > 0);
    assert!(failure.quality_report.worst_svg_overlap.is_some());
}

#[test]
fn search_rejects_zero_attempt_budget() {
    let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
    let mut config = RenderFitConfig::single_page_balanced();
    config.max_attempts = 0;

    let error =
        find_render_settings(&document, VerovioRenderSettings::medium(), config).unwrap_err();

    assert_eq!(error, RenderFitError::InvalidMaxAttempts);
}

#[test]
fn search_rejects_unsupported_target_pages() {
    let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
    let mut config = RenderFitConfig::single_page_balanced();
    config.target_pages = 2;

    let error =
        find_render_settings(&document, VerovioRenderSettings::medium(), config).unwrap_err();

    assert_eq!(
        error,
        RenderFitError::UnsupportedTargetPages { target_pages: 2 }
    );
}

#[test]
fn search_rejects_invalid_bounds() {
    let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
    let mut config = RenderFitConfig::single_page_balanced();
    config.bounds.spacing_staff = FloatRange { min: 4.0, max: 3.0 };

    let error =
        find_render_settings(&document, VerovioRenderSettings::medium(), config).unwrap_err();

    assert!(matches!(error, RenderFitError::InvalidBounds { .. }));
}

#[test]
fn search_rejects_bounds_without_valid_page_width() {
    let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
    let mut config = RenderFitConfig::single_page_balanced();
    config.bounds.page_width = IntRange {
        min: 2998,
        max: 2999,
    };

    let error =
        find_render_settings(&document, VerovioRenderSettings::medium(), config).unwrap_err();

    assert!(matches!(error, RenderFitError::InvalidBounds { .. }));
}

#[test]
fn search_rejects_rendered_unit_bounds_that_make_page_widths_impossible() {
    let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
    let mut config = RenderFitConfig::single_page_balanced();
    config.rendered_unit_bounds = RenderedUnitBounds {
        min: 20.0,
        max: 21.0,
    };

    let error =
        find_render_settings(&document, VerovioRenderSettings::medium(), config).unwrap_err();

    assert!(matches!(error, RenderFitError::InvalidBounds { .. }));
}

#[test]
fn search_rejects_invalid_initial_settings() {
    let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
    let mut initial = VerovioRenderSettings::medium();
    initial.page_width = 2920;

    let error = find_render_settings(&document, initial, RenderFitConfig::single_page_balanced())
        .unwrap_err();

    assert!(matches!(
        error,
        RenderFitError::InvalidInitialSettings { .. }
    ));
}

#[test]
fn candidates_are_deterministic_and_respect_attempt_budget() {
    let initial = VerovioRenderSettings::medium();
    let mut config = RenderFitConfig::single_page_balanced();
    config.max_attempts = 8;

    let first = render_setting_candidates(initial, config);
    let second = render_setting_candidates(initial, config);

    assert_eq!(first, second);
    assert_eq!(first.len(), 8);
    assert_eq!(first[0], initial);
    assert!(first
        .iter()
        .all(|settings| config.bounds.contains(*settings)));
    assert!(first.iter().all(|settings| config
        .rendered_unit_bounds
        .contains_settings(*settings, config.bounds.unit)));
}

#[test]
fn seeded_first_candidates_vary_across_file_indices() {
    let initial = VerovioRenderSettings::medium();
    let config = RenderFitConfig::single_page_balanced();
    let mut fonts = Vec::new();
    let mut page_widths = Vec::new();
    let mut units = Vec::new();

    for index in 0..32 {
        let candidates = fast_seeded_render_setting_candidates(
            initial,
            config,
            RenderFitSearchSeed { index, count: 32 },
        );
        let first = candidates[0];
        fonts.push(first.font);
        page_widths.push(first.page_width);
        units.push(first.unit);
        assert_ne!(first, initial);
        assert!(config.bounds.contains(first));
        assert!(config
            .rendered_unit_bounds
            .contains_settings(first, config.bounds.unit));
    }

    fonts.sort_by_key(|font| font.as_str());
    fonts.dedup();
    page_widths.sort();
    page_widths.dedup();
    units.sort_by(f64::total_cmp);
    units.dedup();

    assert!(fonts.len() > 1);
    assert!(page_widths.len() > 1);
    assert!(units.len() > 1);
}

#[test]
fn compact_candidates_use_minimum_unit_for_scaled_page_width() {
    let initial = VerovioRenderSettings::medium();
    let config = RenderFitConfig::single_page_balanced();
    let candidates = render_setting_candidates(initial, config);

    let compact = candidates
        .iter()
        .find(|settings| {
            settings.page_width == 3500
                && settings.spacing_staff == config.bounds.spacing_staff.min
                && settings.spacing_system == config.bounds.spacing_system.min
                && settings.measure_min_width == config.bounds.measure_min_width.min
                && settings.spacing_linear == config.bounds.spacing_linear.min
                && settings.spacing_non_linear == config.bounds.spacing_non_linear.min
        })
        .expect("compact fallback candidate should be present");

    assert_close(compact.unit, 9.0);
    assert!(config
        .rendered_unit_bounds
        .contains_settings(*compact, config.bounds.unit));
}

#[test]
fn fast_seeded_candidates_respect_attempt_budget() {
    let initial = VerovioRenderSettings::medium();
    let config = RenderFitConfig::single_page_balanced();
    let seed = RenderFitSearchSeed {
        index: 3,
        count: 17,
    };
    let candidates = fast_seeded_render_setting_candidates(initial, config, seed);

    assert_eq!(candidates[0], seeded_low_discrepancy_candidate(3, config));
    assert!(candidates.len() <= config.max_attempts);
    assert!(candidates.len() <= 8);
    assert!(candidates
        .iter()
        .all(|settings| config.bounds.contains(*settings)));
    assert!(candidates.iter().all(|settings| config
        .rendered_unit_bounds
        .contains_settings(*settings, config.bounds.unit)));
}

#[test]
fn seeded_candidates_sample_horizontal_margins_as_page_width_ratios() {
    let initial = VerovioRenderSettings::medium();
    let config = RenderFitConfig::single_page_balanced();

    for index in 0..64 {
        let candidates = fast_seeded_render_setting_candidates(
            initial,
            config,
            RenderFitSearchSeed { index, count: 64 },
        );

        assert!(candidates.iter().all(|settings| {
            drawable_width_ratio(*settings) >= MIN_DRAWABLE_WIDTH_RATIO
                && settings.page_margin_left as f64
                    <= settings.page_width as f64 * MAX_HORIZONTAL_SIDE_MARGIN_RATIO
                && settings.page_margin_right as f64
                    <= settings.page_width as f64 * MAX_HORIZONTAL_SIDE_MARGIN_RATIO
        }));
    }
}

#[test]
fn horizontal_margin_sampling_caps_margins_by_page_width() {
    let page_width = 1250;
    let margins = sample_horizontal_margins(
        page_width,
        VerovioRenderSettingsBounds::single_page_search(),
        0.999,
        0.999,
    );

    assert!(margins.left < 495);
    assert!(margins.right < 487);
    assert!(margins.left + margins.right <= max_total_horizontal_margin(page_width));
}

#[test]
fn seeded_candidates_respect_multi_spine_left_margin_floor() {
    let document = KernDocument::parse("**kern\t**kern\n4c\t4e\n*-\t*-\n").unwrap();
    let initial = VerovioRenderSettings::medium();
    let config = RenderFitConfig::single_page_balanced_for_document(&document);
    let candidates = fast_seeded_render_setting_candidates(
        initial,
        config,
        RenderFitSearchSeed {
            index: 0,
            count: 32,
        },
    );

    assert_eq!(
        config.bounds.page_margin_left.min,
        MULTI_SPINE_MIN_PAGE_MARGIN_LEFT
    );
    assert!(candidates
        .iter()
        .all(|settings| settings.page_margin_left >= MULTI_SPINE_MIN_PAGE_MARGIN_LEFT));
}

#[test]
fn overlap_horizontal_relief_candidates_respect_bounds_and_margin_floor() {
    let document = KernDocument::parse("**kern\t**kern\n4c\t4e\n*-\t*-\n").unwrap();
    let config = RenderFitConfig::single_page_balanced_for_document(&document);
    let mut settings = seeded_low_discrepancy_candidate(0, config);
    settings.page_margin_left = config.bounds.page_margin_left.min;
    settings.page_margin_right = config.bounds.page_margin_right.min;
    let candidates = overlap_horizontal_relief_candidates(settings, config);

    assert!(!candidates.is_empty());
    assert!(candidates.len() <= 3);
    assert!(candidates.iter().all(|candidate| {
        config.bounds.contains(*candidate)
            && candidate.page_margin_left >= MULTI_SPINE_MIN_PAGE_MARGIN_LEFT
            && candidate.page_width >= settings.page_width
    }));
    let mut deduped = candidates.clone();
    deduped.dedup();
    assert_eq!(deduped.len(), candidates.len());
}

#[test]
fn high_staff_seeded_candidates_sample_from_preferred_bounds() {
    let document = KernDocument::parse(
        "**kern\t**kern\t**kern\t**kern\t**kern\t**kern\t**kern\n\
             4c\t4d\t4e\t4f\t4g\t4a\t4b\n\
             *-\t*-\t*-\t*-\t*-\t*-\t*-\n",
    )
    .unwrap();
    let initial = VerovioRenderSettings::medium();
    let config = RenderFitConfig::single_page_balanced_for_document(&document);
    let candidates = fast_seeded_render_setting_candidates(
        initial,
        config,
        RenderFitSearchSeed {
            index: 0,
            count: 32,
        },
    );

    assert!(candidates[0].spacing_staff <= 8.0);
    assert!(candidates[0].page_margin_top <= 150);
    assert!(candidates[0].page_margin_bottom <= 150);
    assert!(candidates[0].page_width >= HIGH_STAFF_MIN_PAGE_WIDTH);
    assert!(candidates
        .iter()
        .all(|settings| config.bounds.contains(*settings)));
}

#[test]
fn lower_score_prefers_balanced_and_filled_content() {
    let initial = VerovioRenderSettings::medium();
    let config = RenderFitConfig::single_page_balanced();
    let balanced = SvgPageWhitespace {
        content_box: SvgContentBox {
            x: 100.0,
            y: 100.0,
            width: 850.0,
            height: 1285.0,
        },
        whitespace: SvgWhitespace {
            left: 100.0,
            right: 100.0,
            top: 100.0,
            bottom: 100.0,
        },
    };
    let sparse = SvgPageWhitespace {
        content_box: SvgContentBox {
            x: 10.0,
            y: 10.0,
            width: 400.0,
            height: 500.0,
        },
        whitespace: SvgWhitespace {
            left: 10.0,
            right: 640.0,
            top: 10.0,
            bottom: 975.0,
        },
    };

    assert!(
        score_candidate(initial, initial, balanced, config)
            < score_candidate(initial, initial, sparse, config)
    );
}

#[test]
fn candidate_fit_allows_small_overflow_but_rejects_large_overflow() {
    let config = RenderFitConfig::single_page_balanced();
    let near_edge = SvgPageWhitespace {
        content_box: SvgContentBox {
            x: 20.0,
            y: 20.0,
            width: 1035.0,
            height: 1490.0,
        },
        whitespace: SvgWhitespace {
            left: 20.0,
            right: -5.0,
            top: 20.0,
            bottom: -25.0,
        },
    };
    let severe_overflow = SvgPageWhitespace {
        content_box: SvgContentBox {
            x: 172.2,
            y: 654.43,
            width: 759.86,
            height: 2610.89,
        },
        whitespace: SvgWhitespace {
            left: 172.2,
            right: 117.94,
            top: 654.43,
            bottom: -1780.32,
        },
    };

    assert!(candidate_fits_page(near_edge, config));
    assert!(!candidate_fits_page(severe_overflow, config));
}

#[test]
fn valid_page_width_sampling_uses_only_scale_compatible_widths() {
    let bounds = VerovioRenderSettingsBounds {
        page_width: IntRange {
            min: 2990,
            max: 3010,
        },
        ..VerovioRenderSettingsBounds::single_page_search()
    };

    assert_eq!(valid_page_widths(bounds), vec![3000]);
}
