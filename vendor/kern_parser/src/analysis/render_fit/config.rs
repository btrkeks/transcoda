use crate::{
    analysis::{FloatRange, IntRange, VerovioRenderSettings, VerovioRenderSettingsBounds},
    KernDocument, KernLine,
};

use super::{sampling::valid_page_widths, sizing::RenderedUnitBounds, RenderFitError};

pub(super) const MULTI_SPINE_MIN_PAGE_MARGIN_LEFT: u32 = 25;
pub(super) const HIGH_STAFF_MIN_PAGE_WIDTH: u32 = 1750;

/// Configuration for render-setting search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderFitConfig {
    pub target_pages: u32,
    pub max_attempts: usize,
    pub bounds: VerovioRenderSettingsBounds,
    pub preferred_bounds: VerovioRenderSettingsBounds,
    pub rendered_unit_bounds: RenderedUnitBounds,
    pub min_whitespace: f64,
    pub whitespace_balance_weight: f64,
    pub whitespace_fill_weight: f64,
    pub settings_distortion_weight: f64,
}

/// Document layout summary used to bias render-setting candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderFitDocumentProfile {
    pub staff_count: u32,
    pub total_spine_count: u32,
    pub barline_count: usize,
}

impl RenderFitConfig {
    pub fn single_page_balanced() -> Self {
        let bounds = VerovioRenderSettingsBounds::single_page_search();
        Self {
            target_pages: 1,
            max_attempts: 32,
            bounds,
            preferred_bounds: bounds,
            rendered_unit_bounds: RenderedUnitBounds::default_for_single_page_search(),
            min_whitespace: -25.0,
            whitespace_balance_weight: 1.0,
            whitespace_fill_weight: 1.0,
            settings_distortion_weight: 0.15,
        }
    }

    pub fn single_page_balanced_for_document(document: &KernDocument) -> Self {
        let mut config = Self::single_page_balanced();
        let profile = RenderFitDocumentProfile::for_document(document);
        if document.layout.initial_spine_count >= 2 {
            config.bounds.page_margin_left.min = config
                .bounds
                .page_margin_left
                .min
                .max(MULTI_SPINE_MIN_PAGE_MARGIN_LEFT);
            config.preferred_bounds.page_margin_left.min = config
                .preferred_bounds
                .page_margin_left
                .min
                .max(MULTI_SPINE_MIN_PAGE_MARGIN_LEFT);
        }
        config.preferred_bounds = preferred_bounds_for_profile(config.preferred_bounds, profile);
        config
    }

    fn bounds_contains(self, inner: VerovioRenderSettingsBounds) -> bool {
        int_range_contains(self.bounds.page_width, inner.page_width)
            && int_range_contains(self.bounds.page_margin_left, inner.page_margin_left)
            && int_range_contains(self.bounds.page_margin_right, inner.page_margin_right)
            && int_range_contains(self.bounds.page_margin_top, inner.page_margin_top)
            && int_range_contains(self.bounds.page_margin_bottom, inner.page_margin_bottom)
            && float_range_contains(self.bounds.bar_line_width, inner.bar_line_width)
            && int_range_contains(self.bounds.beam_max_slope, inner.beam_max_slope)
            && float_range_contains(self.bounds.staff_line_width, inner.staff_line_width)
            && float_range_contains(self.bounds.stem_width, inner.stem_width)
            && float_range_contains(
                self.bounds.ledger_line_thickness,
                inner.ledger_line_thickness,
            )
            && float_range_contains(
                self.bounds.thick_barline_thickness,
                inner.thick_barline_thickness,
            )
            && inner
                .font
                .allowed
                .iter()
                .all(|font| self.bounds.font.contains(*font))
            && float_range_contains(self.bounds.unit, inner.unit)
            && float_range_contains(self.bounds.spacing_staff, inner.spacing_staff)
            && float_range_contains(self.bounds.spacing_system, inner.spacing_system)
            && float_range_contains(self.bounds.measure_min_width, inner.measure_min_width)
            && float_range_contains(self.bounds.spacing_linear, inner.spacing_linear)
            && float_range_contains(self.bounds.spacing_non_linear, inner.spacing_non_linear)
    }
}

impl RenderFitDocumentProfile {
    pub fn for_document(document: &KernDocument) -> Self {
        Self {
            staff_count: document.staff_count() as u32,
            total_spine_count: document.layout.initial_spine_count,
            barline_count: document
                .lines()
                .iter()
                .filter(|line| matches!(line, KernLine::Barline(_)))
                .count(),
        }
    }
}

pub(super) fn validate_config(
    config: RenderFitConfig,
    initial: VerovioRenderSettings,
) -> Result<(), RenderFitError> {
    if config.target_pages != 1 {
        return Err(RenderFitError::UnsupportedTargetPages {
            target_pages: config.target_pages,
        });
    }
    if config.max_attempts == 0 {
        return Err(RenderFitError::InvalidMaxAttempts);
    }
    validate_weight(
        config.whitespace_balance_weight,
        "whitespace_balance_weight",
    )?;
    validate_weight(config.whitespace_fill_weight, "whitespace_fill_weight")?;
    validate_weight(
        config.settings_distortion_weight,
        "settings_distortion_weight",
    )?;
    config
        .bounds
        .validate()
        .map_err(|error| RenderFitError::InvalidBounds {
            message: error.to_string(),
        })?;
    config
        .preferred_bounds
        .validate()
        .map_err(|error| RenderFitError::InvalidBounds {
            message: format!("preferred bounds invalid: {error}"),
        })?;
    let hard_page_widths = valid_page_widths(config.bounds);
    if hard_page_widths.is_empty() {
        return Err(RenderFitError::InvalidBounds {
            message: "page_width bounds contain no scale-compatible width".to_owned(),
        });
    }
    let preferred_page_widths = valid_page_widths(config.preferred_bounds);
    if preferred_page_widths.is_empty() {
        return Err(RenderFitError::InvalidBounds {
            message: "preferred page_width bounds contain no scale-compatible width".to_owned(),
        });
    }
    if !config.bounds_contains(config.preferred_bounds) {
        return Err(RenderFitError::InvalidBounds {
            message: "preferred bounds must be contained in hard bounds".to_owned(),
        });
    }
    config
        .rendered_unit_bounds
        .validate()
        .map_err(|message| RenderFitError::InvalidBounds { message })?;
    validate_rendered_unit_page_widths(
        config.rendered_unit_bounds,
        config.bounds.unit,
        &hard_page_widths,
        "bounds",
    )?;
    validate_rendered_unit_page_widths(
        config.rendered_unit_bounds,
        config.preferred_bounds.unit,
        &preferred_page_widths,
        "preferred_bounds",
    )?;
    validate_min_whitespace(config.min_whitespace)?;
    config.bounds.validate_settings(initial).map_err(|error| {
        RenderFitError::InvalidInitialSettings {
            message: error.to_string(),
        }
    })?;
    if !config
        .rendered_unit_bounds
        .contains_settings(initial, config.bounds.unit)
    {
        return Err(RenderFitError::InvalidInitialSettings {
            message: "initial settings are outside rendered-unit bounds".to_owned(),
        });
    }

    Ok(())
}

fn validate_rendered_unit_page_widths(
    rendered_unit_bounds: RenderedUnitBounds,
    hard_unit_bounds: FloatRange,
    page_widths: &[u32],
    label: &'static str,
) -> Result<(), RenderFitError> {
    if let Some(page_width) = page_widths.iter().copied().find(|page_width| {
        rendered_unit_bounds
            .unit_range_for_page_width(hard_unit_bounds, *page_width)
            .is_none()
    }) {
        return Err(RenderFitError::InvalidBounds {
            message: format!(
                "{label} rendered-unit bounds leave no valid unit range for page_width={page_width}"
            ),
        });
    }

    Ok(())
}

pub(super) fn preferred_bounds_for_profile(
    mut bounds: VerovioRenderSettingsBounds,
    profile: RenderFitDocumentProfile,
) -> VerovioRenderSettingsBounds {
    match profile.staff_count {
        0..=3 => {}
        4 => {
            cap_float_max(&mut bounds.spacing_staff, 36.0);
            cap_int_max(&mut bounds.page_margin_top, 400);
            cap_int_max(&mut bounds.page_margin_bottom, 400);
        }
        5 => {
            cap_float_max(&mut bounds.spacing_staff, 28.0);
            cap_int_max(&mut bounds.page_margin_top, 350);
            cap_int_max(&mut bounds.page_margin_bottom, 350);
        }
        6 => {
            cap_float_max(&mut bounds.spacing_staff, 16.0);
            cap_int_max(&mut bounds.page_margin_top, 250);
            cap_int_max(&mut bounds.page_margin_bottom, 250);
            raise_page_width_min_to_valid_floor(&mut bounds, HIGH_STAFF_MIN_PAGE_WIDTH);
        }
        _ => {
            cap_float_max(&mut bounds.spacing_staff, 8.0);
            cap_int_max(&mut bounds.page_margin_top, 150);
            cap_int_max(&mut bounds.page_margin_bottom, 150);
            raise_page_width_min_to_valid_floor(&mut bounds, HIGH_STAFF_MIN_PAGE_WIDTH);
        }
    }

    bounds
}

fn cap_int_max(range: &mut IntRange, max: u32) {
    range.max = range.max.min(max).max(range.min);
}

fn raise_page_width_min_to_valid_floor(bounds: &mut VerovioRenderSettingsBounds, min: u32) {
    let widths = valid_page_widths(*bounds);
    let selected = widths
        .iter()
        .copied()
        .find(|width| *width >= min)
        .or_else(|| widths.last().copied());
    if let Some(width) = selected {
        bounds.page_width.min = width;
    }
}

fn cap_float_max(range: &mut FloatRange, max: f64) {
    range.max = range.max.min(max).max(range.min);
}

fn validate_weight(value: f64, setting: &'static str) -> Result<(), RenderFitError> {
    if !value.is_finite() || value < 0.0 {
        return Err(RenderFitError::InvalidWeight { setting });
    }

    Ok(())
}

fn validate_min_whitespace(value: f64) -> Result<(), RenderFitError> {
    if !value.is_finite() {
        return Err(RenderFitError::InvalidWeight {
            setting: "min_whitespace",
        });
    }

    Ok(())
}

fn int_range_contains(outer: IntRange, inner: IntRange) -> bool {
    outer.min <= inner.min && inner.max <= outer.max
}

fn float_range_contains(outer: FloatRange, inner: FloatRange) -> bool {
    outer.min <= inner.min && inner.max <= outer.max
}
