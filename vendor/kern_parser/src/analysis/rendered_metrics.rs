//! Render-derived document metrics using Verovio.

#[cfg(feature = "engraving")]
use std::sync::Mutex;

use thiserror::Error;

use crate::{
    analysis::{SvgContentBoxError, SvgPageSizeError, VerovioRenderSettings},
    KernDocument,
};

#[cfg(feature = "engraving")]
use crate::{
    analysis::{
        detect_svg_content_box, detect_svg_page_size, TARGET_RENDER_HEIGHT, TARGET_RENDER_WIDTH,
    },
    serialize::render,
};

#[cfg(feature = "engraving")]
static VEROVIO_LAYOUT_MUTEX: Mutex<()> = Mutex::new(());

/// Render-derived metrics for one Verovio page.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderedPageMetrics {
    pub page_number: u32,
    pub content_height: f64,
}

/// Render-derived metrics for a full document.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedDocumentMetrics {
    pub page_count: u32,
    pub total_content_height: f64,
    pub pages: Vec<RenderedPageMetrics>,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum RenderedMetricsError {
    #[error("native engraving is unavailable; enable the `engraving` Cargo feature")]
    EngravingUnavailable,
    #[error("Verovio render failed: {message}")]
    Verovio { message: String },
    #[error("SVG content box detection failed on page {page_number}: {source}")]
    SvgContentBox {
        page_number: u32,
        #[source]
        source: SvgContentBoxError,
    },
    #[error("SVG page size detection failed on page {page_number}: {source}")]
    SvgPageSize {
        page_number: u32,
        #[source]
        source: SvgPageSizeError,
    },
    #[error(
        "rendered page {page_number} has size {actual_width}x{actual_height}, expected 1050x1485"
    )]
    UnexpectedPageSize {
        page_number: u32,
        actual_width: u32,
        actual_height: u32,
    },
    #[error(transparent)]
    Dialect(#[from] crate::serialize::DialectError),
}

/// Return render-derived metrics for `document` under explicit Verovio settings.
///
/// The document is serialized to renderable Humdrum before loading, since
/// [`KernDocument`] is a typed representation rather than a lossless copy of
/// the source text.
#[cfg(feature = "engraving")]
pub fn verovio_render_metrics(
    document: &KernDocument,
    settings: VerovioRenderSettings,
) -> Result<RenderedDocumentMetrics, RenderedMetricsError> {
    let _layout_guard = VEROVIO_LAYOUT_MUTEX
        .lock()
        .map_err(|_| RenderedMetricsError::Verovio {
            message: "Verovio layout mutex poisoned".into(),
        })?;
    let options_json =
        settings
            .to_options_json()
            .map_err(|error| RenderedMetricsError::Verovio {
                message: error.to_string(),
            })?;
    let render_text = render::to_string(document)?;
    let mut voxide = verovioxide::Toolkit::new().map_err(verovio_error)?;
    let first_page_svg = voxide
        .render_data(&render_text, Some(&options_json))
        .map_err(verovio_error)?;

    let page_count = voxide.page_count();
    let mut pages = Vec::with_capacity(page_count as usize);
    let mut total_content_height = 0.0;

    for page_number in 1..=page_count {
        let svg = if page_number == 1 {
            first_page_svg.clone()
        } else {
            voxide.render_to_svg(page_number).map_err(verovio_error)?
        };
        let page_size =
            detect_svg_page_size(&svg).map_err(|source| RenderedMetricsError::SvgPageSize {
                page_number,
                source,
            })?;
        let actual_width = page_size.width.round() as u32;
        let actual_height = page_size.height.round() as u32;
        if actual_width != TARGET_RENDER_WIDTH || actual_height != TARGET_RENDER_HEIGHT {
            return Err(RenderedMetricsError::UnexpectedPageSize {
                page_number,
                actual_width,
                actual_height,
            });
        }

        let content_box =
            detect_svg_content_box(&svg).map_err(|source| RenderedMetricsError::SvgContentBox {
                page_number,
                source,
            })?;
        total_content_height += content_box.height;
        pages.push(RenderedPageMetrics {
            page_number,
            content_height: content_box.height,
        });
    }

    Ok(RenderedDocumentMetrics {
        page_count,
        total_content_height,
        pages,
    })
}

#[cfg(not(feature = "engraving"))]
pub fn verovio_render_metrics(
    _document: &KernDocument,
    _settings: VerovioRenderSettings,
) -> Result<RenderedDocumentMetrics, RenderedMetricsError> {
    Err(RenderedMetricsError::EngravingUnavailable)
}

/// Return render-derived metrics under the medium Verovio layout.
pub fn medium_verovio_render_metrics(
    document: &KernDocument,
) -> Result<RenderedDocumentMetrics, RenderedMetricsError> {
    verovio_render_metrics(document, VerovioRenderSettings::medium())
}

#[cfg(feature = "engraving")]
fn verovio_error(error: verovioxide::Error) -> RenderedMetricsError {
    RenderedMetricsError::Verovio {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "engraving")]
    use crate::analysis::VerovioMusicFont;

    #[test]
    #[cfg(not(feature = "engraving"))]
    fn missing_engine_is_an_explicit_capability_error() {
        let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
        assert_eq!(
            medium_verovio_render_metrics(&document),
            Err(RenderedMetricsError::EngravingUnavailable)
        );
    }

    #[test]
    fn medium_render_options_are_valid_json() {
        let options: serde_json::Value =
            serde_json::from_str(&VerovioRenderSettings::medium().to_options_json().unwrap())
                .unwrap();

        assert_eq!(options["breaks"], "auto");
        assert_eq!(options["footer"], "none");
        assert_eq!(options["scale"], 42);
        assert_eq!(options["pageWidth"], 2500);
        assert_eq!(options["pageHeight"], 3535);
        assert!(options.get("noJustification").is_none());
    }

    #[test]
    #[cfg(feature = "engraving")]
    fn medium_render_metrics_include_content_height() {
        let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
        let metrics = medium_verovio_render_metrics(&document).unwrap();

        assert_eq!(metrics.page_count, 1);
        assert_eq!(metrics.pages.len(), 1);
        assert_eq!(metrics.pages[0].page_number, 1);
        assert!(metrics.pages[0].content_height > 0.0);
        assert_eq!(
            metrics.total_content_height,
            metrics.pages[0].content_height
        );
    }

    #[test]
    #[cfg(feature = "engraving")]
    fn added_single_page_render_settings_smoke_render() {
        let document = KernDocument::parse("**kern\n=1\n8cL\n8dJ\n=2\n4cc\n==\n*-\n").unwrap();
        let fonts = [
            VerovioMusicFont::Leipzig,
            VerovioMusicFont::Bravura,
            VerovioMusicFont::Gootville,
            VerovioMusicFont::Leland,
        ];

        for font in fonts {
            let mut settings = VerovioRenderSettings::medium();
            settings.bar_line_width = 0.16;
            settings.beam_max_slope = 16;
            settings.staff_line_width = 0.26;
            settings.stem_width = 0.36;
            settings.ledger_line_thickness = 0.40;
            settings.thick_barline_thickness = 1.80;
            settings.font = font;

            let metrics = verovio_render_metrics(&document, settings).unwrap();

            assert_eq!(metrics.page_count, 1);
            assert!(metrics.total_content_height > 0.0);
        }
    }
}
