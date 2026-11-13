//! Typed Verovio render settings used by render-based analyses.

use serde_json::json;
use thiserror::Error;

pub const TARGET_RENDER_WIDTH: u32 = 1050;
pub const TARGET_RENDER_HEIGHT: u32 = 1485;
const VEROVIO_MIN_SCALE_PERCENT: u32 = 1;
const VEROVIO_MAX_SCALE_PERCENT: u32 = 1000;
const TARGET_RENDER_WIDTH_SCALE_PRODUCT: u32 = TARGET_RENDER_WIDTH * 100;

/// Render settings passed to Verovio layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerovioRenderSettings {
    pub page_width: u32,
    pub page_margin_left: u32,
    pub page_margin_right: u32,
    pub page_margin_top: u32,
    pub page_margin_bottom: u32,
    pub bar_line_width: f64,
    pub beam_max_slope: u32,
    pub staff_line_width: f64,
    pub stem_width: f64,
    pub ledger_line_thickness: f64,
    pub thick_barline_thickness: f64,
    pub font: VerovioMusicFont,
    pub unit: f64,
    pub spacing_staff: f64,
    pub spacing_system: f64,
    pub measure_min_width: f64,
    pub spacing_linear: f64,
    pub spacing_non_linear: f64,
    // pub no_justification: bool,
    pub svg_bounding_boxes: bool,
    pub svg_content_bounding_boxes: bool,
}

/// Inclusive integer bounds for a render setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntRange {
    pub min: u32,
    pub max: u32,
}

/// Inclusive floating-point bounds for a render setting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatRange {
    pub min: f64,
    pub max: f64,
}

/// Supported SMuFL music fonts bundled with Verovio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerovioMusicFont {
    Leipzig,
    Bravura,
    Gootville,
    Leland,
    // Petaluma,
}

/// Inclusive categorical bounds for Verovio music fonts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FontSet {
    pub allowed: &'static [VerovioMusicFont],
}

/// Hard-coded search bounds for Verovio render settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerovioRenderSettingsBounds {
    pub page_width: IntRange,
    pub page_margin_left: IntRange,
    pub page_margin_right: IntRange,
    pub page_margin_top: IntRange,
    pub page_margin_bottom: IntRange,
    pub bar_line_width: FloatRange,
    pub beam_max_slope: IntRange,
    pub staff_line_width: FloatRange,
    pub stem_width: FloatRange,
    pub ledger_line_thickness: FloatRange,
    pub thick_barline_thickness: FloatRange,
    pub font: FontSet,
    pub unit: FloatRange,
    pub spacing_staff: FloatRange,
    pub spacing_system: FloatRange,
    pub measure_min_width: FloatRange,
    pub spacing_linear: FloatRange,
    pub spacing_non_linear: FloatRange,
}

pub const SINGLE_PAGE_RENDER_FONTS: &[VerovioMusicFont] = &[
    VerovioMusicFont::Leipzig,
    VerovioMusicFont::Bravura,
    VerovioMusicFont::Gootville,
    VerovioMusicFont::Leland,
    // VerovioMusicFont::Petaluma,
];

/// Default render setting bounds for single-page render setting search.
pub const SINGLE_PAGE_RENDER_SETTINGS_BOUNDS: VerovioRenderSettingsBounds =
    VerovioRenderSettingsBounds {
        page_width: IntRange {
            min: 1239,
            max: 3500,
        },
        page_margin_left: IntRange { min: 0, max: 500 },
        page_margin_right: IntRange { min: 0, max: 500 },
        page_margin_top: IntRange { min: 0, max: 500 },
        page_margin_bottom: IntRange { min: 0, max: 500 },
        bar_line_width: FloatRange {
            min: 0.16,
            max: 0.72,
        },
        beam_max_slope: IntRange { min: 4, max: 16 },
        staff_line_width: FloatRange {
            min: 0.10,
            max: 0.26,
        },
        stem_width: FloatRange {
            min: 0.10,
            max: 0.36,
        },
        ledger_line_thickness: FloatRange {
            min: 0.10,
            max: 0.40,
        },
        thick_barline_thickness: FloatRange {
            min: 0.60,
            max: 1.80,
        },
        font: FontSet {
            allowed: SINGLE_PAGE_RENDER_FONTS,
        },
        unit: FloatRange {
            min: 6.0,
            max: 12.0,
        },
        spacing_staff: FloatRange {
            min: 4.0,
            max: 48.0,
        },
        spacing_system: FloatRange {
            min: 3.0,
            max: 20.0,
        },
        measure_min_width: FloatRange {
            min: 1.0,
            max: 30.0,
        },
        spacing_linear: FloatRange {
            min: 0.12,
            max: 0.42,
        },
        spacing_non_linear: FloatRange {
            min: 0.20,
            max: 0.85,
        },
    };

/// Semantic render adjustments that map onto concrete Verovio settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderAdjustment {
    LessVerticalSpace,
    MoreVerticalSpace,
    LessHorizontalSpace,
    MoreHorizontalSpace,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum VerovioRenderSettingsError {
    #[error("Verovio page width must be greater than zero")]
    InvalidPageWidth,
    #[error("Verovio page width must derive an exact integer scale for 1050px output")]
    InvalidDerivedScale,
    #[error("derived Verovio scale {scale} is outside the supported 1..=1000 range")]
    DerivedScaleOutOfBounds { scale: u32 },
    #[error("Verovio spacing values must be finite and greater than zero")]
    InvalidSpacing,
    #[error("Verovio page margins must leave positive drawable width")]
    InvalidMargins,
    #[error("invalid integer bounds for Verovio setting {setting}")]
    InvalidIntegerBounds { setting: &'static str },
    #[error("invalid float bounds for Verovio setting {setting}")]
    InvalidFloatBounds { setting: &'static str },
    #[error("invalid font bounds for Verovio setting {setting}")]
    InvalidFontBounds { setting: &'static str },
    #[error("Verovio setting {setting} is outside the configured bounds")]
    SettingOutOfBounds { setting: &'static str },
    #[error("Verovio render settings serialization failed: {message}")]
    SerializationFailed { message: String },
}

impl IntRange {
    pub fn midpoint(self) -> u32 {
        ((self.min as u64 + self.max as u64) / 2) as u32
    }

    pub fn contains(self, value: u32) -> bool {
        self.min <= value && value <= self.max
    }

    pub fn validate(self, setting: &'static str) -> Result<(), VerovioRenderSettingsError> {
        if self.min > self.max {
            return Err(VerovioRenderSettingsError::InvalidIntegerBounds { setting });
        }

        Ok(())
    }
}

impl FloatRange {
    pub fn midpoint(self) -> f64 {
        (self.min + self.max) / 2.0
    }

    pub fn contains(self, value: f64) -> bool {
        value.is_finite() && self.min <= value && value <= self.max
    }

    pub fn validate(self, setting: &'static str) -> Result<(), VerovioRenderSettingsError> {
        if !self.min.is_finite() || !self.max.is_finite() || self.min <= 0.0 || self.min > self.max
        {
            return Err(VerovioRenderSettingsError::InvalidFloatBounds { setting });
        }

        Ok(())
    }
}

impl VerovioMusicFont {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Leipzig => "Leipzig",
            Self::Bravura => "Bravura",
            Self::Gootville => "Gootville",
            Self::Leland => "Leland",
            // Self::Petaluma => "Petaluma",
        }
    }
}

impl FontSet {
    pub fn midpoint(self) -> VerovioMusicFont {
        self.allowed[self.allowed.len() / 2]
    }

    pub fn contains(self, value: VerovioMusicFont) -> bool {
        self.allowed.contains(&value)
    }

    pub fn validate(self, setting: &'static str) -> Result<(), VerovioRenderSettingsError> {
        if self.allowed.is_empty() {
            return Err(VerovioRenderSettingsError::InvalidFontBounds { setting });
        }

        Ok(())
    }
}

impl VerovioRenderSettingsBounds {
    pub const fn single_page_search() -> Self {
        SINGLE_PAGE_RENDER_SETTINGS_BOUNDS
    }

    pub fn contains(self, settings: VerovioRenderSettings) -> bool {
        settings.validate().is_ok()
            && self.page_width.contains(settings.page_width)
            && self.page_margin_left.contains(settings.page_margin_left)
            && self.page_margin_right.contains(settings.page_margin_right)
            && self.page_margin_top.contains(settings.page_margin_top)
            && self
                .page_margin_bottom
                .contains(settings.page_margin_bottom)
            && self.bar_line_width.contains(settings.bar_line_width)
            && self.beam_max_slope.contains(settings.beam_max_slope)
            && self.staff_line_width.contains(settings.staff_line_width)
            && self.stem_width.contains(settings.stem_width)
            && self
                .ledger_line_thickness
                .contains(settings.ledger_line_thickness)
            && self
                .thick_barline_thickness
                .contains(settings.thick_barline_thickness)
            && self.font.contains(settings.font)
            && self.unit.contains(settings.unit)
            && self.spacing_staff.contains(settings.spacing_staff)
            && self.spacing_system.contains(settings.spacing_system)
            && self.measure_min_width.contains(settings.measure_min_width)
            && self.spacing_linear.contains(settings.spacing_linear)
            && self
                .spacing_non_linear
                .contains(settings.spacing_non_linear)
    }

    pub fn validate(self) -> Result<(), VerovioRenderSettingsError> {
        self.page_width.validate("page_width")?;
        self.page_margin_left.validate("page_margin_left")?;
        self.page_margin_right.validate("page_margin_right")?;
        self.page_margin_top.validate("page_margin_top")?;
        self.page_margin_bottom.validate("page_margin_bottom")?;
        self.bar_line_width.validate("bar_line_width")?;
        self.beam_max_slope.validate("beam_max_slope")?;
        self.staff_line_width.validate("staff_line_width")?;
        self.stem_width.validate("stem_width")?;
        self.ledger_line_thickness
            .validate("ledger_line_thickness")?;
        self.thick_barline_thickness
            .validate("thick_barline_thickness")?;
        self.font.validate("font")?;
        self.unit.validate("unit")?;
        self.spacing_staff.validate("spacing_staff")?;
        self.spacing_system.validate("spacing_system")?;
        self.measure_min_width.validate("measure_min_width")?;
        self.spacing_linear.validate("spacing_linear")?;
        self.spacing_non_linear.validate("spacing_non_linear")?;
        Ok(())
    }

    pub fn validate_settings(
        self,
        settings: VerovioRenderSettings,
    ) -> Result<(), VerovioRenderSettingsError> {
        self.validate()?;
        settings.validate()?;

        if !self.page_width.contains(settings.page_width) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "page_width",
            });
        }
        if !self.page_margin_left.contains(settings.page_margin_left) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "page_margin_left",
            });
        }
        if !self.page_margin_right.contains(settings.page_margin_right) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "page_margin_right",
            });
        }
        if !self.page_margin_top.contains(settings.page_margin_top) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "page_margin_top",
            });
        }
        if !self
            .page_margin_bottom
            .contains(settings.page_margin_bottom)
        {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "page_margin_bottom",
            });
        }
        if !self.bar_line_width.contains(settings.bar_line_width) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "bar_line_width",
            });
        }
        if !self.beam_max_slope.contains(settings.beam_max_slope) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "beam_max_slope",
            });
        }
        if !self.staff_line_width.contains(settings.staff_line_width) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "staff_line_width",
            });
        }
        if !self.stem_width.contains(settings.stem_width) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "stem_width",
            });
        }
        if !self
            .ledger_line_thickness
            .contains(settings.ledger_line_thickness)
        {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "ledger_line_thickness",
            });
        }
        if !self
            .thick_barline_thickness
            .contains(settings.thick_barline_thickness)
        {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "thick_barline_thickness",
            });
        }
        if !self.font.contains(settings.font) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds { setting: "font" });
        }
        if !self.unit.contains(settings.unit) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds { setting: "unit" });
        }
        if !self.spacing_staff.contains(settings.spacing_staff) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "spacing_staff",
            });
        }
        if !self.spacing_system.contains(settings.spacing_system) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "spacing_system",
            });
        }
        if !self.measure_min_width.contains(settings.measure_min_width) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "measure_min_width",
            });
        }
        if !self.spacing_linear.contains(settings.spacing_linear) {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "spacing_linear",
            });
        }
        if !self
            .spacing_non_linear
            .contains(settings.spacing_non_linear)
        {
            return Err(VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "spacing_non_linear",
            });
        }

        Ok(())
    }
}

impl VerovioRenderSettings {
    /// Medium Verovio layout settings used as the current page-count baseline.
    pub fn medium() -> Self {
        let bounds = VerovioRenderSettingsBounds::single_page_search();
        Self {
            page_width: nearest_valid_page_width_midpoint(bounds.page_width),
            page_margin_left: bounds.page_margin_left.midpoint(),
            page_margin_right: bounds.page_margin_right.midpoint(),
            page_margin_top: bounds.page_margin_top.midpoint(),
            page_margin_bottom: bounds.page_margin_bottom.midpoint(),
            bar_line_width: bounds.bar_line_width.midpoint(),
            beam_max_slope: bounds.beam_max_slope.midpoint(),
            staff_line_width: bounds.staff_line_width.midpoint(),
            stem_width: bounds.stem_width.midpoint(),
            ledger_line_thickness: bounds.ledger_line_thickness.midpoint(),
            thick_barline_thickness: bounds.thick_barline_thickness.midpoint(),
            font: bounds.font.midpoint(),
            unit: bounds.unit.midpoint(),
            spacing_staff: bounds.spacing_staff.midpoint(),
            spacing_system: bounds.spacing_system.midpoint(),
            measure_min_width: bounds.measure_min_width.midpoint(),
            spacing_linear: bounds.spacing_linear.midpoint(),
            spacing_non_linear: bounds.spacing_non_linear.midpoint(),
            // no_justification: true,
            svg_bounding_boxes: true,
            svg_content_bounding_boxes: true,
        }
    }

    /// Return a copy of these settings with a semantic adjustment applied.
    pub fn adjusted(
        mut self,
        adjustment: RenderAdjustment,
    ) -> Result<Self, VerovioRenderSettingsError> {
        match adjustment {
            RenderAdjustment::LessVerticalSpace => {
                self.spacing_staff *= 0.9;
                self.spacing_system *= 0.9;
                self.page_margin_top = scaled_u32(self.page_margin_top, 0.9);
                self.page_margin_bottom = scaled_u32(self.page_margin_bottom, 0.9);
            }
            RenderAdjustment::MoreVerticalSpace => {
                self.spacing_staff *= 1.1;
                self.spacing_system *= 1.1;
                self.page_margin_top = scaled_u32(self.page_margin_top, 1.1);
                self.page_margin_bottom = scaled_u32(self.page_margin_bottom, 1.1);
            }
            RenderAdjustment::LessHorizontalSpace => {
                self.measure_min_width *= 0.9;
                self.spacing_linear *= 0.9;
                self.spacing_non_linear *= 0.9;
                self.page_margin_left = scaled_u32(self.page_margin_left, 0.9);
                self.page_margin_right = scaled_u32(self.page_margin_right, 0.9);
            }
            RenderAdjustment::MoreHorizontalSpace => {
                self.measure_min_width *= 1.1;
                self.spacing_linear *= 1.1;
                self.spacing_non_linear *= 1.1;
                self.page_margin_left = scaled_u32(self.page_margin_left, 1.1);
                self.page_margin_right = scaled_u32(self.page_margin_right, 1.1);
            }
        }

        self.validate()?;
        Ok(self)
    }

    pub fn scale_percent(self) -> u32 {
        TARGET_RENDER_WIDTH_SCALE_PRODUCT
            .checked_div(self.page_width)
            .unwrap_or(0)
    }

    pub fn page_height(self) -> u32 {
        let scale = self.scale_percent();
        (TARGET_RENDER_HEIGHT * 100).checked_div(scale).unwrap_or(0)
    }

    pub fn rendered_width(self) -> u32 {
        (self.page_width * self.scale_percent()) / 100
    }

    pub fn rendered_height(self) -> u32 {
        (((self.page_height() as f64) * (self.scale_percent() as f64)) / 100.0).ceil() as u32
    }

    /// Serialize these settings into Verovio's JSON option format.
    pub fn to_options_json(self) -> Result<String, VerovioRenderSettingsError> {
        self.validate()?;

        serde_json::to_string(&json!({
            "breaks": "auto",
            "footer": "none",
            "scale": self.scale_percent(),
            "pageWidth": self.page_width,
            "pageHeight": self.page_height(),
            "pageMarginLeft": self.page_margin_left,
            "pageMarginRight": self.page_margin_right,
            "pageMarginTop": self.page_margin_top,
            "pageMarginBottom": self.page_margin_bottom,
            "barLineWidth": self.bar_line_width,
            "beamMaxSlope": self.beam_max_slope,
            "staffLineWidth": self.staff_line_width,
            "stemWidth": self.stem_width,
            "ledgerLineThickness": self.ledger_line_thickness,
            "thickBarlineThickness": self.thick_barline_thickness,
            "font": self.font.as_str(),
            "unit": self.unit,
            "spacingStaff": self.spacing_staff,
            "spacingSystem": self.spacing_system,
            "measureMinWidth": self.measure_min_width,
            "spacingLinear": self.spacing_linear,
            "spacingNonLinear": self.spacing_non_linear,
            // "noJustification": self.no_justification,
            "svgBoundingBoxes": self.svg_bounding_boxes,
            "svgContentBoundingBoxes": self.svg_content_bounding_boxes,
        }))
        .map_err(|error| VerovioRenderSettingsError::SerializationFailed {
            message: error.to_string(),
        })
    }

    pub fn validate(self) -> Result<(), VerovioRenderSettingsError> {
        if self.page_width == 0 {
            return Err(VerovioRenderSettingsError::InvalidPageWidth);
        }

        if !TARGET_RENDER_WIDTH_SCALE_PRODUCT.is_multiple_of(self.page_width)
            || self.scale_percent() == 0
        {
            return Err(VerovioRenderSettingsError::InvalidDerivedScale);
        }

        let scale = self.scale_percent();
        if !(VEROVIO_MIN_SCALE_PERCENT..=VEROVIO_MAX_SCALE_PERCENT).contains(&scale) {
            return Err(VerovioRenderSettingsError::DerivedScaleOutOfBounds { scale });
        }

        if self.page_margin_left + self.page_margin_right >= self.page_width {
            return Err(VerovioRenderSettingsError::InvalidMargins);
        }

        let spacings = [
            self.spacing_staff,
            self.spacing_system,
            self.measure_min_width,
            self.spacing_linear,
            self.spacing_non_linear,
            self.unit,
            self.bar_line_width,
            self.staff_line_width,
            self.stem_width,
            self.ledger_line_thickness,
            self.thick_barline_thickness,
        ];
        if spacings
            .iter()
            .any(|spacing| !spacing.is_finite() || *spacing <= 0.0)
        {
            return Err(VerovioRenderSettingsError::InvalidSpacing);
        }

        Ok(())
    }
}

fn scaled_u32(value: u32, multiplier: f64) -> u32 {
    ((value as f64) * multiplier).round().max(1.0) as u32
}

fn nearest_valid_page_width_midpoint(range: IntRange) -> u32 {
    let target_sum = range.min as i64 + range.max as i64;
    (range.min..=range.max)
        .filter(|width| TARGET_RENDER_WIDTH_SCALE_PRODUCT.is_multiple_of(*width))
        .min_by_key(|width| ((*width as i64 * 2) - target_sum).abs())
        .expect("single-page page_width bounds must contain a valid scale-compatible width")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medium_settings_serialize_to_verovio_json() {
        let options: serde_json::Value =
            serde_json::from_str(&VerovioRenderSettings::medium().to_options_json().unwrap())
                .unwrap();

        assert_eq!(options["breaks"], "auto");
        assert_eq!(options["footer"], "none");
        assert_eq!(options["scale"], 42);
        assert_eq!(options["pageWidth"], 2500);
        assert_eq!(options["pageHeight"], 3535);
        assert_eq!(options["barLineWidth"], 0.44);
        assert_eq!(options["beamMaxSlope"], 10);
        assert_eq!(options["staffLineWidth"], 0.18);
        assert!((options["stemWidth"].as_f64().unwrap() - 0.23).abs() < f64::EPSILON);
        assert_eq!(options["ledgerLineThickness"], 0.25);
        assert_eq!(options["thickBarlineThickness"], 1.2);
        assert_eq!(options["font"], "Gootville");
        assert_eq!(options["unit"], 9.0);
        assert_eq!(options["spacingStaff"], 26.0);
        assert!(options.get("noJustification").is_none());
        assert_eq!(options["svgBoundingBoxes"], true);
        assert_eq!(options["svgContentBoundingBoxes"], true);
    }

    #[test]
    fn less_vertical_space_reduces_vertical_settings() {
        let medium = VerovioRenderSettings::medium();
        let adjusted = medium
            .adjusted(RenderAdjustment::LessVerticalSpace)
            .unwrap();

        assert!(adjusted.spacing_staff < medium.spacing_staff);
        assert!(adjusted.spacing_system < medium.spacing_system);
        assert!(adjusted.page_margin_top < medium.page_margin_top);
        assert!(adjusted.page_margin_bottom < medium.page_margin_bottom);
    }

    #[test]
    fn invalid_settings_fail_fast() {
        let mut settings = VerovioRenderSettings::medium();
        settings.page_width = 2920;

        assert_eq!(
            settings.to_options_json().unwrap_err(),
            VerovioRenderSettingsError::InvalidDerivedScale
        );
    }

    #[test]
    fn rejects_derived_scale_outside_verovio_range() {
        let mut settings = VerovioRenderSettings::medium();
        settings.page_width = 100;

        assert_eq!(
            settings.to_options_json().unwrap_err(),
            VerovioRenderSettingsError::DerivedScaleOutOfBounds { scale: 1050 }
        );
    }

    #[test]
    fn medium_settings_derive_a4_geometry() {
        let settings = VerovioRenderSettings::medium();

        assert_eq!(settings.scale_percent(), 42);
        assert_eq!(settings.page_height(), 3535);
        assert_eq!(settings.rendered_width(), TARGET_RENDER_WIDTH);
        assert_eq!(settings.rendered_height(), TARGET_RENDER_HEIGHT);
        settings.validate().unwrap();
    }

    #[test]
    fn medium_settings_are_derived_from_single_page_search_bounds() {
        let bounds = VerovioRenderSettingsBounds::single_page_search();
        let settings = VerovioRenderSettings::medium();

        assert_eq!(settings.page_width, 2500);
        assert_eq!(
            settings.page_margin_left,
            bounds.page_margin_left.midpoint()
        );
        assert_eq!(
            settings.page_margin_right,
            bounds.page_margin_right.midpoint()
        );
        assert_eq!(settings.page_margin_top, bounds.page_margin_top.midpoint());
        assert_eq!(
            settings.page_margin_bottom,
            bounds.page_margin_bottom.midpoint()
        );
        assert_eq!(settings.bar_line_width, bounds.bar_line_width.midpoint());
        assert_eq!(settings.beam_max_slope, bounds.beam_max_slope.midpoint());
        assert_eq!(
            settings.staff_line_width,
            bounds.staff_line_width.midpoint()
        );
        assert_eq!(settings.stem_width, bounds.stem_width.midpoint());
        assert_eq!(
            settings.ledger_line_thickness,
            bounds.ledger_line_thickness.midpoint()
        );
        assert_eq!(
            settings.thick_barline_thickness,
            bounds.thick_barline_thickness.midpoint()
        );
        assert_eq!(settings.font, bounds.font.midpoint());
        assert_eq!(settings.unit, bounds.unit.midpoint());
        assert_eq!(settings.spacing_staff, bounds.spacing_staff.midpoint());
        assert_eq!(settings.spacing_system, bounds.spacing_system.midpoint());
        assert_eq!(
            settings.measure_min_width,
            bounds.measure_min_width.midpoint()
        );
        assert_eq!(settings.spacing_linear, bounds.spacing_linear.midpoint());
        assert_eq!(
            settings.spacing_non_linear,
            bounds.spacing_non_linear.midpoint()
        );
    }

    #[test]
    fn medium_settings_are_inside_single_page_search_bounds() {
        let bounds = VerovioRenderSettingsBounds::single_page_search();
        let settings = VerovioRenderSettings::medium();

        assert!(bounds.contains(settings));
        bounds.validate_settings(settings).unwrap();
    }

    #[test]
    fn bounds_reject_out_of_range_settings() {
        let bounds = VerovioRenderSettingsBounds::single_page_search();
        let mut settings = VerovioRenderSettings::medium();
        settings.page_width = 5250;

        assert_eq!(
            bounds.validate_settings(settings).unwrap_err(),
            VerovioRenderSettingsError::SettingOutOfBounds {
                setting: "page_width"
            }
        );
    }
}
