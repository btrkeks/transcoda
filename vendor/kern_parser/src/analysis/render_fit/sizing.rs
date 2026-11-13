use crate::analysis::{FloatRange, VerovioRenderSettings, TARGET_RENDER_WIDTH};

/// Bounds for the staff-size-like unit after Verovio scaling to the target image width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderedUnitBounds {
    pub min: f64,
    pub max: f64,
}

impl RenderedUnitBounds {
    pub const fn default_for_single_page_search() -> Self {
        Self { min: 2.7, max: 8.4 }
    }

    pub fn validate(self) -> Result<(), String> {
        if !self.min.is_finite() || !self.max.is_finite() || self.min <= 0.0 || self.min > self.max
        {
            return Err("rendered unit bounds must be finite, positive, and ordered".to_owned());
        }
        Ok(())
    }

    pub fn rendered_unit(self, settings: VerovioRenderSettings) -> Option<f64> {
        if settings.page_width == 0 || !settings.unit.is_finite() {
            return None;
        }
        Some(settings.unit * (TARGET_RENDER_WIDTH as f64 / settings.page_width as f64))
    }

    pub fn unit_range_for_page_width(
        self,
        hard_unit_bounds: FloatRange,
        page_width: u32,
    ) -> Option<FloatRange> {
        if page_width == 0 {
            return None;
        }

        let inverse_scale = page_width as f64 / TARGET_RENDER_WIDTH as f64;
        let min = hard_unit_bounds.min.max(self.min * inverse_scale);
        let max = hard_unit_bounds.max.min(self.max * inverse_scale);
        if min.is_finite() && max.is_finite() && min <= max {
            Some(FloatRange { min, max })
        } else {
            None
        }
    }

    pub fn clamp_unit_for_page_width(
        self,
        unit: f64,
        hard_unit_bounds: FloatRange,
        page_width: u32,
    ) -> Option<f64> {
        let range = self.unit_range_for_page_width(hard_unit_bounds, page_width)?;
        Some(unit.clamp(range.min, range.max))
    }

    pub fn contains_settings(
        self,
        settings: VerovioRenderSettings,
        hard_unit_bounds: FloatRange,
    ) -> bool {
        self.unit_range_for_page_width(hard_unit_bounds, settings.page_width)
            .is_some_and(|range| range.contains(settings.unit))
    }
}
