use crate::analysis::{
    FloatRange, IntRange, VerovioMusicFont, VerovioRenderSettings, VerovioRenderSettingsBounds,
    TARGET_RENDER_WIDTH,
};

use super::{RenderFitConfig, RenderFitSearchSeed};

pub(super) const MAX_HORIZONTAL_SIDE_MARGIN_RATIO: f64 = 0.16;
pub(super) const MIN_DRAWABLE_WIDTH_RATIO: f64 = 0.68;
const STRATIFIED_BIN_COUNT: usize = 16;
const FAST_SEEDED_ATTEMPT_BUDGET: usize = 8;
const HALTON_BASES: [usize; 36] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
];

pub(super) fn render_setting_candidates(
    initial: VerovioRenderSettings,
    config: RenderFitConfig,
) -> Vec<VerovioRenderSettings> {
    let mut candidates = Vec::with_capacity(config.max_attempts);
    push_candidate(&mut candidates, initial, config);

    for fallback in compact_fallback_candidates(initial, config) {
        push_candidate(&mut candidates, fallback, config);
    }

    let mut index = 0;
    while candidates.len() < config.max_attempts {
        push_candidate(
            &mut candidates,
            low_discrepancy_candidate(index, config),
            config,
        );
        index += 1;
    }

    candidates
}

pub(super) fn fast_seeded_render_setting_candidates(
    initial: VerovioRenderSettings,
    config: RenderFitConfig,
    seed: RenderFitSearchSeed,
) -> Vec<VerovioRenderSettings> {
    let budget = config.max_attempts.min(FAST_SEEDED_ATTEMPT_BUDGET);
    let sampled = seeded_low_discrepancy_candidate(seed.index, config);
    let mut candidates = Vec::with_capacity(budget);
    push_candidate(&mut candidates, sampled, config);

    for repair in compact_repair_candidates(sampled, config) {
        push_candidate(&mut candidates, repair, config);
    }

    let mut attempt_index = 1;
    while candidates.len() < budget {
        push_candidate(
            &mut candidates,
            seeded_candidate_for_attempt(seed, attempt_index, config),
            config,
        );
        attempt_index += 1;
    }

    if candidates.len() > budget {
        candidates.truncate(budget);
    }
    if candidates.is_empty() {
        push_candidate(&mut candidates, initial, config);
    }

    candidates
}

fn seeded_candidate_for_attempt(
    seed: RenderFitSearchSeed,
    attempt_index: usize,
    config: RenderFitConfig,
) -> VerovioRenderSettings {
    let sequence_index = seed.index + attempt_index * seed.count;
    seeded_low_discrepancy_candidate(sequence_index, config)
}

fn compact_repair_candidates(
    sampled: VerovioRenderSettings,
    config: RenderFitConfig,
) -> [VerovioRenderSettings; 7] {
    let mut lower_system_spacing = sampled;
    lower_system_spacing.spacing_system =
        lower_float_bin(sampled.spacing_system, config.bounds.spacing_system, 2);

    let mut lower_staff_spacing = sampled;
    lower_staff_spacing.spacing_staff =
        lower_float_bin(sampled.spacing_staff, config.bounds.spacing_staff, 2);

    let mut lower_vertical_margins = sampled;
    lower_vertical_margins.page_margin_top =
        lower_int_bin(sampled.page_margin_top, config.bounds.page_margin_top, 4);
    lower_vertical_margins.page_margin_bottom = lower_int_bin(
        sampled.page_margin_bottom,
        config.bounds.page_margin_bottom,
        4,
    );

    let mut lower_horizontal_spacing = sampled;
    lower_horizontal_spacing.measure_min_width = lower_float_bin(
        sampled.measure_min_width,
        config.bounds.measure_min_width,
        3,
    );
    lower_horizontal_spacing.spacing_linear =
        lower_float_bin(sampled.spacing_linear, config.bounds.spacing_linear, 3);
    lower_horizontal_spacing.spacing_non_linear = lower_float_bin(
        sampled.spacing_non_linear,
        config.bounds.spacing_non_linear,
        3,
    );

    let mut wider_page = sampled;
    wider_page.page_width = higher_valid_page_width(sampled.page_width, config.bounds, 3);
    wider_page.page_margin_left =
        lower_int_bin(sampled.page_margin_left, config.bounds.page_margin_left, 3);
    wider_page.page_margin_right = lower_int_bin(
        sampled.page_margin_right,
        config.bounds.page_margin_right,
        3,
    );

    let mut compact_layout = lower_horizontal_spacing;
    compact_layout.page_width = higher_valid_page_width(sampled.page_width, config.bounds, 4);
    compact_layout.page_margin_left =
        lower_int_bin(sampled.page_margin_left, config.bounds.page_margin_left, 4);
    compact_layout.page_margin_right = lower_int_bin(
        sampled.page_margin_right,
        config.bounds.page_margin_right,
        4,
    );
    compact_layout.spacing_staff =
        lower_float_bin(sampled.spacing_staff, config.bounds.spacing_staff, 4);
    compact_layout.spacing_system =
        lower_float_bin(sampled.spacing_system, config.bounds.spacing_system, 4);
    compact_layout.unit = clamp_unit_for_page_width(
        lower_float_bin(sampled.unit, config.bounds.unit, 4),
        compact_layout.page_width,
        config,
    );
    compact_layout.page_margin_top =
        lower_int_bin(sampled.page_margin_top, config.bounds.page_margin_top, 6);
    compact_layout.page_margin_bottom = lower_int_bin(
        sampled.page_margin_bottom,
        config.bounds.page_margin_bottom,
        6,
    );

    let mut hard_compact = sampled;
    hard_compact.page_width = widest_valid_page_width(config.bounds);
    hard_compact.page_margin_left = config.bounds.page_margin_left.min;
    hard_compact.page_margin_right = config.bounds.page_margin_right.min;
    hard_compact.page_margin_top = config.bounds.page_margin_top.min;
    hard_compact.page_margin_bottom = config.bounds.page_margin_bottom.min;
    hard_compact.spacing_staff = config.bounds.spacing_staff.min;
    hard_compact.spacing_system = config.bounds.spacing_system.min;
    hard_compact.unit = compact_unit_for_page_width(hard_compact.page_width, config);
    hard_compact.measure_min_width = config.bounds.measure_min_width.min;
    hard_compact.spacing_linear = config.bounds.spacing_linear.min;
    hard_compact.spacing_non_linear = config.bounds.spacing_non_linear.min;

    [
        lower_system_spacing,
        lower_staff_spacing,
        lower_vertical_margins,
        lower_horizontal_spacing,
        wider_page,
        compact_layout,
        hard_compact,
    ]
}

fn push_candidate(
    candidates: &mut Vec<VerovioRenderSettings>,
    settings: VerovioRenderSettings,
    config: RenderFitConfig,
) {
    if candidates.len() >= config.max_attempts {
        return;
    }
    let Some(settings) = normalize_candidate_sizing(settings, config) else {
        return;
    };
    if config.bounds.contains(settings)
        && drawable_width_ratio(settings) >= MIN_DRAWABLE_WIDTH_RATIO
        && !candidates.contains(&settings)
    {
        candidates.push(settings);
    }
}

pub(super) fn overlap_horizontal_relief_candidates(
    settings: VerovioRenderSettings,
    config: RenderFitConfig,
) -> Vec<VerovioRenderSettings> {
    let mut candidates = Vec::new();

    let mut wider_page = settings;
    wider_page.page_width = higher_valid_page_width(settings.page_width, config.bounds, 3);
    push_candidate(&mut candidates, wider_page, config);

    let mut wider_lower_margins = wider_page;
    wider_lower_margins.page_margin_left =
        lower_int_bin(settings.page_margin_left, config.bounds.page_margin_left, 4);
    wider_lower_margins.page_margin_right = lower_int_bin(
        settings.page_margin_right,
        config.bounds.page_margin_right,
        4,
    );
    push_candidate(&mut candidates, wider_lower_margins, config);

    let mut widest_min_margins = settings;
    widest_min_margins.page_width = widest_valid_page_width(config.bounds);
    widest_min_margins.page_margin_left = config.bounds.page_margin_left.min;
    widest_min_margins.page_margin_right = config.bounds.page_margin_right.min;
    push_candidate(&mut candidates, widest_min_margins, config);

    candidates
}

fn force_render_fit_svg_options(mut settings: VerovioRenderSettings) -> VerovioRenderSettings {
    // settings.no_justification = true;
    settings.svg_bounding_boxes = true;
    settings.svg_content_bounding_boxes = true;
    settings
}

fn normalize_candidate_sizing(
    settings: VerovioRenderSettings,
    config: RenderFitConfig,
) -> Option<VerovioRenderSettings> {
    let mut settings = force_render_fit_svg_options(settings);
    settings.unit = config.rendered_unit_bounds.clamp_unit_for_page_width(
        settings.unit,
        config.bounds.unit,
        settings.page_width,
    )?;
    Some(settings)
}

fn compact_unit_for_page_width(page_width: u32, config: RenderFitConfig) -> f64 {
    unit_range_for_page_width(page_width, config.bounds, config).min
}

fn clamp_unit_for_page_width(unit: f64, page_width: u32, config: RenderFitConfig) -> f64 {
    config
        .rendered_unit_bounds
        .clamp_unit_for_page_width(unit, config.bounds.unit, page_width)
        .expect("validated render-fit config must allow every hard-bounds page width")
}

fn sample_unit_for_page_width(
    page_width: u32,
    bounds: VerovioRenderSettingsBounds,
    sample: f64,
    config: RenderFitConfig,
) -> f64 {
    sample_float(
        unit_range_for_page_width(page_width, bounds, config),
        sample,
    )
}

fn unit_range_for_page_width(
    page_width: u32,
    bounds: VerovioRenderSettingsBounds,
    config: RenderFitConfig,
) -> FloatRange {
    config
        .rendered_unit_bounds
        .unit_range_for_page_width(bounds.unit, page_width)
        .expect("validated render-fit config must allow every sampled page width")
}

fn compact_fallback_candidates(
    initial: VerovioRenderSettings,
    config: RenderFitConfig,
) -> [VerovioRenderSettings; 3] {
    let mut vertical = initial;
    vertical.page_margin_top = config.bounds.page_margin_top.min;
    vertical.page_margin_bottom = config.bounds.page_margin_bottom.min;
    vertical.spacing_staff = config.bounds.spacing_staff.min;
    vertical.spacing_system = config.bounds.spacing_system.min;

    let mut horizontal = initial;
    horizontal.page_width = widest_valid_page_width(config.bounds);
    horizontal.page_margin_left = config.bounds.page_margin_left.min;
    horizontal.page_margin_right = config.bounds.page_margin_right.min;
    horizontal.measure_min_width = config.bounds.measure_min_width.min;
    horizontal.spacing_linear = config.bounds.spacing_linear.min;
    horizontal.spacing_non_linear = config.bounds.spacing_non_linear.min;

    let mut compact = horizontal;
    compact.page_margin_top = config.bounds.page_margin_top.min;
    compact.page_margin_bottom = config.bounds.page_margin_bottom.min;
    compact.spacing_staff = config.bounds.spacing_staff.min;
    compact.spacing_system = config.bounds.spacing_system.min;
    compact.unit = compact_unit_for_page_width(compact.page_width, config);

    [vertical, horizontal, compact]
}

fn low_discrepancy_candidate(index: usize, config: RenderFitConfig) -> VerovioRenderSettings {
    let bounds = config.preferred_bounds;
    let sample = |dimension: usize| halton(index + 1, HALTON_BASES[dimension]);
    let page_width = sample_valid_page_width(bounds, sample(0));
    let horizontal_margins = sample_horizontal_margins(page_width, bounds, sample(1), sample(2));

    VerovioRenderSettings {
        page_width,
        page_margin_left: horizontal_margins.left,
        page_margin_right: horizontal_margins.right,
        page_margin_top: sample_int(bounds.page_margin_top, sample(3)),
        page_margin_bottom: sample_int(bounds.page_margin_bottom, sample(4)),
        bar_line_width: sample_float(bounds.bar_line_width, sample(5)),
        beam_max_slope: sample_int(bounds.beam_max_slope, sample(6)),
        staff_line_width: sample_float(bounds.staff_line_width, sample(7)),
        stem_width: sample_float(bounds.stem_width, sample(8)),
        ledger_line_thickness: sample_float(bounds.ledger_line_thickness, sample(9)),
        thick_barline_thickness: sample_float(bounds.thick_barline_thickness, sample(10)),
        font: sample_font(bounds, sample(11)),
        unit: sample_unit_for_page_width(page_width, bounds, sample(12), config),
        spacing_staff: sample_float(bounds.spacing_staff, sample(13)),
        spacing_system: sample_float(bounds.spacing_system, sample(14)),
        measure_min_width: sample_float(bounds.measure_min_width, sample(15)),
        spacing_linear: sample_float(bounds.spacing_linear, sample(16)),
        spacing_non_linear: sample_float(bounds.spacing_non_linear, sample(17)),
        // no_justification: true,
        svg_bounding_boxes: true,
        svg_content_bounding_boxes: true,
    }
}

pub(super) fn seeded_low_discrepancy_candidate(
    index: usize,
    config: RenderFitConfig,
) -> VerovioRenderSettings {
    let bounds = config.preferred_bounds;
    let sample = |dimension: usize, bins: usize| stratified_unit(index, dimension, bins);
    let page_width = sample_valid_page_width(bounds, sample(0, valid_page_widths(bounds).len()));
    let horizontal_margins = sample_horizontal_margins(
        page_width,
        bounds,
        sample(1, STRATIFIED_BIN_COUNT),
        sample(2, STRATIFIED_BIN_COUNT),
    );

    VerovioRenderSettings {
        page_width,
        page_margin_left: horizontal_margins.left,
        page_margin_right: horizontal_margins.right,
        page_margin_top: sample_int_binned(bounds.page_margin_top, sample(3, STRATIFIED_BIN_COUNT)),
        page_margin_bottom: sample_int_binned(
            bounds.page_margin_bottom,
            sample(4, STRATIFIED_BIN_COUNT),
        ),
        bar_line_width: sample_float_binned(bounds.bar_line_width, sample(5, STRATIFIED_BIN_COUNT)),
        beam_max_slope: sample_int_binned(bounds.beam_max_slope, sample(6, STRATIFIED_BIN_COUNT)),
        staff_line_width: sample_float_binned(
            bounds.staff_line_width,
            sample(7, STRATIFIED_BIN_COUNT),
        ),
        stem_width: sample_float_binned(bounds.stem_width, sample(8, STRATIFIED_BIN_COUNT)),
        ledger_line_thickness: sample_float_binned(
            bounds.ledger_line_thickness,
            sample(9, STRATIFIED_BIN_COUNT),
        ),
        thick_barline_thickness: sample_float_binned(
            bounds.thick_barline_thickness,
            sample(10, STRATIFIED_BIN_COUNT),
        ),
        font: sample_font(bounds, sample(11, bounds.font.allowed.len())),
        unit: sample_unit_for_page_width(
            page_width,
            bounds,
            sample(12, STRATIFIED_BIN_COUNT),
            config,
        ),
        spacing_staff: sample_float_binned(bounds.spacing_staff, sample(13, STRATIFIED_BIN_COUNT)),
        spacing_system: sample_float_binned(
            bounds.spacing_system,
            sample(14, STRATIFIED_BIN_COUNT),
        ),
        measure_min_width: sample_float_binned(
            bounds.measure_min_width,
            sample(15, STRATIFIED_BIN_COUNT),
        ),
        spacing_linear: sample_float_binned(
            bounds.spacing_linear,
            sample(16, STRATIFIED_BIN_COUNT),
        ),
        spacing_non_linear: sample_float_binned(
            bounds.spacing_non_linear,
            sample(17, STRATIFIED_BIN_COUNT),
        ),
        // no_justification: true,
        svg_bounding_boxes: true,
        svg_content_bounding_boxes: true,
    }
}

fn halton(mut index: usize, base: usize) -> f64 {
    let mut result = 0.0;
    let mut fraction = 1.0 / base as f64;
    while index > 0 {
        result += fraction * (index % base) as f64;
        index /= base;
        fraction /= base as f64;
    }

    result
}

fn sample_int(range: IntRange, unit: f64) -> u32 {
    let span = range.max - range.min + 1;
    range.min + ((unit * span as f64).floor() as u32).min(span - 1)
}

fn sample_int_binned(range: IntRange, unit: f64) -> u32 {
    sample_int(range, unit)
}

fn sample_float(range: FloatRange, unit: f64) -> f64 {
    range.min + (range.max - range.min) * unit
}

fn sample_float_binned(range: FloatRange, unit: f64) -> f64 {
    sample_float(range, unit)
}

fn sample_font(bounds: VerovioRenderSettingsBounds, unit: f64) -> VerovioMusicFont {
    let fonts = bounds.font.allowed;
    let index = ((unit * fonts.len() as f64).floor() as usize).min(fonts.len() - 1);
    fonts[index]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HorizontalMargins {
    pub left: u32,
    pub right: u32,
}

pub(super) fn sample_horizontal_margins(
    page_width: u32,
    bounds: VerovioRenderSettingsBounds,
    left_unit: f64,
    right_unit: f64,
) -> HorizontalMargins {
    // If more render settings become relative, correlated, or perceptual, promote this
    // helper into a sampled-settings type that converts into concrete Verovio settings.
    let max_side_margin = ((page_width as f64) * MAX_HORIZONTAL_SIDE_MARGIN_RATIO)
        .round()
        .clamp(0.0, u32::MAX as f64) as u32;
    let max_total_margin = max_total_horizontal_margin(page_width);
    let left_max = bounds.page_margin_left.max.min(max_side_margin);
    let right_max = bounds.page_margin_right.max.min(max_side_margin);
    let mut left = sample_int(
        IntRange {
            min: bounds.page_margin_left.min.min(left_max),
            max: left_max.max(bounds.page_margin_left.min),
        },
        left_unit,
    )
    .clamp(bounds.page_margin_left.min, bounds.page_margin_left.max);
    let mut right = sample_int(
        IntRange {
            min: bounds.page_margin_right.min.min(right_max),
            max: right_max.max(bounds.page_margin_right.min),
        },
        right_unit,
    )
    .clamp(bounds.page_margin_right.min, bounds.page_margin_right.max);

    enforce_max_total_horizontal_margin(
        &mut left,
        &mut right,
        bounds.page_margin_left.min,
        bounds.page_margin_right.min,
        max_total_margin,
    );

    HorizontalMargins { left, right }
}

pub(super) fn max_total_horizontal_margin(page_width: u32) -> u32 {
    ((page_width as f64) * (1.0 - MIN_DRAWABLE_WIDTH_RATIO))
        .floor()
        .clamp(0.0, u32::MAX as f64) as u32
}

pub(super) fn drawable_width_ratio(settings: VerovioRenderSettings) -> f64 {
    if settings.page_width == 0 {
        return 0.0;
    }
    let margin_sum = settings
        .page_margin_left
        .saturating_add(settings.page_margin_right);
    let drawable_width = settings.page_width.saturating_sub(margin_sum);
    drawable_width as f64 / settings.page_width as f64
}

fn enforce_max_total_horizontal_margin(
    left: &mut u32,
    right: &mut u32,
    left_min: u32,
    right_min: u32,
    max_total: u32,
) {
    while left.saturating_add(*right) > max_total {
        let left_slack = left.saturating_sub(left_min);
        let right_slack = right.saturating_sub(right_min);
        if left_slack == 0 && right_slack == 0 {
            break;
        }
        if left_slack >= right_slack && left_slack > 0 {
            *left -= 1;
        } else if right_slack > 0 {
            *right -= 1;
        }
    }
}

fn stratified_unit(index: usize, dimension: usize, bins: usize) -> f64 {
    let bins = bins.max(1);
    let bin_unit = halton(index + 1, HALTON_BASES[dimension]);
    let jitter_unit = halton(index + 1, HALTON_BASES[dimension + 17]);
    let bin = ((bin_unit * bins as f64).floor() as usize).min(bins - 1);
    ((bin as f64) + jitter_unit) / bins as f64
}

fn sample_valid_page_width(bounds: VerovioRenderSettingsBounds, unit: f64) -> u32 {
    let widths = valid_page_widths(bounds);
    let index = ((unit * widths.len() as f64).floor() as usize).min(widths.len() - 1);
    widths[index]
}

fn widest_valid_page_width(bounds: VerovioRenderSettingsBounds) -> u32 {
    valid_page_widths(bounds)
        .into_iter()
        .max()
        .expect("validated bounds must contain a valid page width")
}

fn higher_valid_page_width(
    page_width: u32,
    bounds: VerovioRenderSettingsBounds,
    steps: usize,
) -> u32 {
    let widths = valid_page_widths(bounds);
    let current = widths
        .iter()
        .position(|width| *width >= page_width)
        .unwrap_or(widths.len() - 1);
    widths[(current + steps).min(widths.len() - 1)]
}

fn lower_int_bin(value: u32, range: IntRange, steps: usize) -> u32 {
    if range.min == range.max {
        return range.min;
    }
    let span = range.max - range.min + 1;
    let bin_width = (span as f64) / STRATIFIED_BIN_COUNT as f64;
    let current_bin = (((value.saturating_sub(range.min)) as f64) / bin_width).floor() as usize;
    let target_bin = current_bin
        .saturating_sub(steps)
        .min(STRATIFIED_BIN_COUNT - 1);
    let sampled = range.min as f64 + (target_bin as f64 + 0.5) * bin_width;
    sampled.round().clamp(range.min as f64, range.max as f64) as u32
}

fn lower_float_bin(value: f64, range: FloatRange, steps: usize) -> f64 {
    if (range.max - range.min).abs() < f64::EPSILON {
        return range.min;
    }
    let bin_width = (range.max - range.min) / STRATIFIED_BIN_COUNT as f64;
    let current_bin = ((value - range.min) / bin_width).floor().max(0.0) as usize;
    let target_bin = current_bin
        .saturating_sub(steps)
        .min(STRATIFIED_BIN_COUNT - 1);
    range.min + (target_bin as f64 + 0.5) * bin_width
}

pub(super) fn valid_page_widths(bounds: VerovioRenderSettingsBounds) -> Vec<u32> {
    (bounds.page_width.min..=bounds.page_width.max)
        .filter(|width| (TARGET_RENDER_WIDTH * 100).is_multiple_of(*width))
        .collect()
}
