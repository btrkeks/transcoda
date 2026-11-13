#[cfg(feature = "engraving")]
use std::sync::Mutex;

#[cfg(feature = "engraving")]
use crate::analysis::{detect_svg_page_size, TARGET_RENDER_HEIGHT, TARGET_RENDER_WIDTH};
#[cfg(any(feature = "engraving", test))]
use crate::serialize::render;
use crate::{analysis::VerovioRenderSettings, KernDocument};

use super::RenderFitError;

#[cfg(feature = "engraving")]
static VEROVIO_RENDER_FIT_MUTEX: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, PartialEq)]
pub(super) struct RenderedCandidatePages {
    pub page_count: u32,
    pub svgs: Vec<String>,
    pub render_source_lines: Vec<Option<usize>>,
}

#[cfg(feature = "engraving")]
pub(super) fn render_candidate(
    document: &KernDocument,
    settings: VerovioRenderSettings,
    target_pages: u32,
) -> Result<RenderedCandidatePages, RenderFitError> {
    let _layout_guard =
        VEROVIO_RENDER_FIT_MUTEX
            .lock()
            .map_err(|_| RenderFitError::RenderFailed {
                message: "Verovio render-fit mutex poisoned".into(),
            })?;
    let options_json =
        settings
            .to_options_json()
            .map_err(|error| RenderFitError::RenderFailed {
                message: error.to_string(),
            })?;
    let (render_text, render_source_lines) = render_input(document)?;
    let mut voxide = verovioxide::Toolkit::new().map_err(verovio_error)?;
    let first_page_svg = voxide
        .render_data(&render_text, Some(&options_json))
        .map_err(verovio_error)?;
    let page_count = voxide.page_count();
    let pages_to_collect = if page_count == target_pages {
        page_count
    } else {
        0
    };
    let mut svgs = Vec::with_capacity(pages_to_collect as usize);

    for page_number in 1..=pages_to_collect {
        let svg = if page_number == 1 {
            first_page_svg.clone()
        } else {
            voxide.render_to_svg(page_number).map_err(verovio_error)?
        };
        validate_page_size(page_number, &svg)?;
        svgs.push(svg);
    }

    Ok(RenderedCandidatePages {
        page_count,
        svgs,
        render_source_lines,
    })
}

#[cfg(not(feature = "engraving"))]
pub(super) fn render_candidate(
    _document: &KernDocument,
    _settings: VerovioRenderSettings,
    _target_pages: u32,
) -> Result<RenderedCandidatePages, RenderFitError> {
    Err(RenderFitError::EngravingUnavailable)
}

#[cfg(any(feature = "engraving", test))]
fn render_input(document: &KernDocument) -> Result<(String, Vec<Option<usize>>), RenderFitError> {
    let output = render::to_render_output(document, render::RenderSerializeOptions::default())?;
    Ok((output.text(), output.sources().to_vec()))
}

#[cfg(feature = "engraving")]
fn validate_page_size(page_number: u32, svg: &str) -> Result<(), RenderFitError> {
    let page_size =
        detect_svg_page_size(svg).map_err(|error| RenderFitError::WhitespaceMeasurementFailed {
            page_number,
            message: error.to_string(),
        })?;
    let actual_width = page_size.width.round() as u32;
    let actual_height = page_size.height.round() as u32;
    if actual_width != TARGET_RENDER_WIDTH || actual_height != TARGET_RENDER_HEIGHT {
        return Err(RenderFitError::UnexpectedPageSize {
            page_number,
            actual_width,
            actual_height,
        });
    }

    Ok(())
}

#[cfg(feature = "engraving")]
fn verovio_error(error: verovioxide::Error) -> RenderFitError {
    RenderFitError::RenderFailed {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "engraving")]
    use crate::augmentation::TupletVisibilityAugmentationConfig;
    #[cfg(feature = "engraving")]
    use crate::{augment_for_render_with_report, RenderAugmentationConfig, RenderAugmentationSeed};

    #[test]
    fn render_input_source_map_matches_render_line_count() {
        let document = KernDocument::parse("**kern\n*M3/4\nMM2r\n=\n*-\n").unwrap();

        let (text, sources) = render_input(&document).unwrap();

        let line_count = text.trim_end_matches('\n').split('\n').count();
        assert_eq!(sources.len(), line_count);
        assert!(sources.iter().any(Option::is_none));
    }

    #[test]
    #[cfg(feature = "engraving")]
    fn tuplet_visibility_augmentation_removes_rendered_tuplet_numbers() {
        let document = KernDocument::parse("**kern\n*M4/4\n12c\n12d\n12e\n4f\n*-\n").unwrap();
        let baseline = render_candidate(&document, VerovioRenderSettings::medium(), 1).unwrap();
        let baseline_count = baseline.svgs[0].matches("class=\"tupletNum\"").count();
        assert!(baseline_count > 0, "{}", baseline.svgs[0]);

        let mut config = RenderAugmentationConfig::default();
        config.articulations.enabled = false;
        config.fermatas.enabled = false;
        config.arpeggios.enabled = false;
        config.mordents.enabled = false;
        config.dynamics.enabled = false;
        config.hairpins.enabled = false;
        config.pedal.enabled = false;
        config.ottavas.enabled = false;
        config.reference_metadata.enabled = false;
        config.staff_names.enabled = false;
        config.metronome_tempos.enabled = false;
        config.text_directions.enabled = false;
        config.tuplet_visibility = TupletVisibilityAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 2,
        };
        let (augmented, report) =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = render_candidate(&augmented, VerovioRenderSettings::medium(), 1).unwrap();

        assert_eq!(report.tuplet_visibility_count, 1);
        assert_eq!(
            rendered.svgs[0].matches("class=\"tupletNum\"").count(),
            0,
            "{}",
            rendered.svgs[0]
        );
    }
}
