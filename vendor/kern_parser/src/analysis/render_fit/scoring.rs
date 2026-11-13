use crate::analysis::{
    FloatRange, IntRange, SvgPageWhitespace, VerovioRenderSettings, VerovioRenderSettingsBounds,
    TARGET_RENDER_HEIGHT, TARGET_RENDER_WIDTH,
};

use super::{sampling::MAX_HORIZONTAL_SIDE_MARGIN_RATIO, RenderFitConfig};

pub(super) fn score_candidate(
    initial: VerovioRenderSettings,
    settings: VerovioRenderSettings,
    whitespace: SvgPageWhitespace,
    config: RenderFitConfig,
) -> f64 {
    let horizontal_balance = (whitespace.whitespace.left - whitespace.whitespace.right).abs()
        / TARGET_RENDER_WIDTH as f64;
    let vertical_balance = (whitespace.whitespace.top - whitespace.whitespace.bottom).abs()
        / TARGET_RENDER_HEIGHT as f64;
    let page_area = TARGET_RENDER_WIDTH as f64 * TARGET_RENDER_HEIGHT as f64;
    let content_area = whitespace.content_box.width * whitespace.content_box.height;
    let unused_area = (1.0 - (content_area / page_area)).clamp(0.0, 1.0);
    let distortion = settings_distance(initial, settings, config.bounds);

    config.whitespace_balance_weight * (horizontal_balance + vertical_balance)
        + config.whitespace_fill_weight * unused_area
        + config.settings_distortion_weight * distortion
}

pub(super) fn candidate_fits_page(whitespace: SvgPageWhitespace, config: RenderFitConfig) -> bool {
    [
        whitespace.whitespace.left,
        whitespace.whitespace.right,
        whitespace.whitespace.top,
        whitespace.whitespace.bottom,
    ]
    .into_iter()
    .all(|value| value >= config.min_whitespace)
}

fn settings_distance(
    initial: VerovioRenderSettings,
    settings: VerovioRenderSettings,
    bounds: VerovioRenderSettingsBounds,
) -> f64 {
    let distances = [
        int_distance(initial.page_width, settings.page_width, bounds.page_width),
        margin_ratio_distance(
            initial.page_margin_left,
            initial.page_width,
            settings.page_margin_left,
            settings.page_width,
        ),
        margin_ratio_distance(
            initial.page_margin_right,
            initial.page_width,
            settings.page_margin_right,
            settings.page_width,
        ),
        int_distance(
            initial.page_margin_top,
            settings.page_margin_top,
            bounds.page_margin_top,
        ),
        int_distance(
            initial.page_margin_bottom,
            settings.page_margin_bottom,
            bounds.page_margin_bottom,
        ),
        float_distance(
            initial.bar_line_width,
            settings.bar_line_width,
            bounds.bar_line_width,
        ),
        int_distance(
            initial.beam_max_slope,
            settings.beam_max_slope,
            bounds.beam_max_slope,
        ),
        float_distance(
            initial.staff_line_width,
            settings.staff_line_width,
            bounds.staff_line_width,
        ),
        float_distance(initial.stem_width, settings.stem_width, bounds.stem_width),
        float_distance(
            initial.ledger_line_thickness,
            settings.ledger_line_thickness,
            bounds.ledger_line_thickness,
        ),
        float_distance(
            initial.thick_barline_thickness,
            settings.thick_barline_thickness,
            bounds.thick_barline_thickness,
        ),
        if initial.font == settings.font {
            0.0
        } else {
            1.0
        },
        float_distance(initial.unit, settings.unit, bounds.unit),
        float_distance(
            initial.spacing_staff,
            settings.spacing_staff,
            bounds.spacing_staff,
        ),
        float_distance(
            initial.spacing_system,
            settings.spacing_system,
            bounds.spacing_system,
        ),
        float_distance(
            initial.measure_min_width,
            settings.measure_min_width,
            bounds.measure_min_width,
        ),
        float_distance(
            initial.spacing_linear,
            settings.spacing_linear,
            bounds.spacing_linear,
        ),
        float_distance(
            initial.spacing_non_linear,
            settings.spacing_non_linear,
            bounds.spacing_non_linear,
        ),
    ];

    distances.iter().sum::<f64>() / distances.len() as f64
}

fn int_distance(a: u32, b: u32, range: IntRange) -> f64 {
    if range.min == range.max {
        return 0.0;
    }

    a.abs_diff(b) as f64 / (range.max - range.min) as f64
}

fn margin_ratio_distance(
    a_margin: u32,
    a_page_width: u32,
    b_margin: u32,
    b_page_width: u32,
) -> f64 {
    if a_page_width == 0 || b_page_width == 0 {
        return 0.0;
    }

    let a_ratio = a_margin as f64 / a_page_width as f64;
    let b_ratio = b_margin as f64 / b_page_width as f64;
    (a_ratio - b_ratio).abs() / MAX_HORIZONTAL_SIDE_MARGIN_RATIO
}

fn float_distance(a: f64, b: f64, range: FloatRange) -> f64 {
    if (range.max - range.min).abs() < f64::EPSILON {
        return 0.0;
    }

    (a - b).abs() / (range.max - range.min)
}
