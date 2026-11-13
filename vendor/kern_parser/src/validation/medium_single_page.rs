//! Medium render page-count validation.

use thiserror::Error;

use crate::{analysis::medium_verovio_render_metrics, KernDocument};

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum MediumSinglePageValidationError {
    #[error("medium render failed: {0}")]
    Render(String),
    #[error("medium render produced {actual} pages, expected 1")]
    UnexpectedPageCount { actual: u32 },
}

pub fn validate(document: &KernDocument) -> Result<(), MediumSinglePageValidationError> {
    let actual = medium_verovio_render_metrics(document)
        .map(|metrics| metrics.page_count)
        .map_err(|error| MediumSinglePageValidationError::Render(error.to_string()))?;

    if actual != 1 {
        return Err(MediumSinglePageValidationError::UnexpectedPageCount { actual });
    }

    Ok(())
}

#[cfg(all(test, feature = "engraving"))]
mod tests {
    use super::*;

    #[test]
    fn accepts_tiny_single_page_document() {
        let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();

        validate(&document).unwrap();
    }
}
