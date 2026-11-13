//! SVG-level geometric augmentation for rendered music pages.

use thiserror::Error;

use crate::analysis::{detect_svg_page_size, SvgContentBox, SvgPageSize, SvgPageSizeError};

/// Configuration for safe page-level geometric augmentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricAugmentationConfig {
    pub enabled: bool,
    pub min_margin: f64,
    pub max_rotation_deg: f64,
    pub min_x_scale: f64,
    pub max_x_scale: f64,
    pub min_y_scale: f64,
    pub max_y_scale: f64,
    pub max_translation_fraction: f64,
    pub max_attempts: usize,
}

impl GeometricAugmentationConfig {
    pub fn conservative_page_geometry() -> Self {
        Self {
            enabled: true,
            min_margin: 2.0,
            max_rotation_deg: 1.8,
            min_x_scale: 0.92,
            max_x_scale: 1.04,
            min_y_scale: 0.97,
            max_y_scale: 1.03,
            max_translation_fraction: 0.04,
            max_attempts: 12,
        }
    }
}

/// Stable seed metadata used to sample deterministic transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometricAugmentationSeed {
    pub file_index: usize,
    pub file_count: usize,
    pub variant_index: usize,
}

/// Page-level transform applied around a fixed center.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricTransform {
    pub angle_deg: f64,
    pub translate_x: f64,
    pub translate_y: f64,
    pub scale_x: f64,
    pub scale_y: f64,
    pub center_x: f64,
    pub center_y: f64,
}

impl GeometricTransform {
    pub fn identity(page_size: SvgPageSize) -> Self {
        Self {
            angle_deg: 0.0,
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            center_x: page_size.width / 2.0,
            center_y: page_size.height / 2.0,
        }
    }

    pub fn is_identity(self) -> bool {
        self.angle_deg == 0.0
            && self.translate_x == 0.0
            && self.translate_y == 0.0
            && self.scale_x == 1.0
            && self.scale_y == 1.0
    }
}

/// Outcome status for the augmentation stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometricAugmentationStatus {
    Applied,
    IdentityFallback,
    Disabled,
}

/// Result of applying or attempting SVG-level geometric augmentation.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometricAugmentationResult {
    pub svg: String,
    pub transform: Option<GeometricTransform>,
    pub original_content_box: SvgContentBox,
    pub transformed_content_box: SvgContentBox,
    pub status: GeometricAugmentationStatus,
    pub attempt_count: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GeometricAugmentationError {
    #[error("invalid geometric augmentation config: {message}")]
    InvalidConfig { message: String },
    #[error("SVG page size detection failed: {0}")]
    PageSize(#[from] SvgPageSizeError),
    #[error("SVG root start tag is missing")]
    MissingSvgStart,
    #[error("SVG root end tag is missing")]
    MissingSvgEnd,
}

/// Apply a safe, deterministic page-level geometric augmentation to an SVG.
pub fn augment_svg_geometry(
    svg: &str,
    content_box: SvgContentBox,
    config: GeometricAugmentationConfig,
    seed: GeometricAugmentationSeed,
) -> Result<GeometricAugmentationResult, GeometricAugmentationError> {
    validate_config(config)?;
    let page_size = detect_svg_page_size(svg)?;

    if !config.enabled {
        return Ok(GeometricAugmentationResult {
            svg: svg.to_owned(),
            transform: None,
            original_content_box: content_box,
            transformed_content_box: content_box,
            status: GeometricAugmentationStatus::Disabled,
            attempt_count: 0,
        });
    }

    let mut rng = DeterministicRng::new(seed);
    for attempt in 1..=config.max_attempts {
        let transform = sample_transform(page_size, config, &mut rng);
        let transformed_content_box = transformed_content_box(content_box, transform);
        if transform_fits_page(transformed_content_box, page_size, config.min_margin) {
            return Ok(GeometricAugmentationResult {
                svg: apply_transform_to_svg(svg, transform)?,
                transform: Some(transform),
                original_content_box: content_box,
                transformed_content_box,
                status: GeometricAugmentationStatus::Applied,
                attempt_count: attempt,
            });
        }
    }

    Ok(GeometricAugmentationResult {
        svg: svg.to_owned(),
        transform: None,
        original_content_box: content_box,
        transformed_content_box: content_box,
        status: GeometricAugmentationStatus::IdentityFallback,
        attempt_count: config.max_attempts,
    })
}

/// Return the content box after applying a page-level geometric transform.
pub fn transformed_svg_content_box(
    content_box: SvgContentBox,
    transform: GeometricTransform,
) -> SvgContentBox {
    transformed_content_box(content_box, transform)
}

/// Return whether a transformed content box remains inside the SVG page.
pub fn transformed_svg_content_fits_page(
    content_box: SvgContentBox,
    page_size: SvgPageSize,
    min_margin: f64,
) -> bool {
    transform_fits_page(content_box, page_size, min_margin)
}

/// Apply a known page-level geometric transform to an SVG.
pub fn apply_geometric_transform_to_svg(
    svg: &str,
    transform: GeometricTransform,
) -> Result<String, GeometricAugmentationError> {
    apply_transform_to_svg(svg, transform)
}

fn validate_config(config: GeometricAugmentationConfig) -> Result<(), GeometricAugmentationError> {
    let finite = [
        config.min_margin,
        config.max_rotation_deg,
        config.min_x_scale,
        config.max_x_scale,
        config.min_y_scale,
        config.max_y_scale,
        config.max_translation_fraction,
    ]
    .into_iter()
    .all(f64::is_finite);
    if !finite {
        return Err(GeometricAugmentationError::InvalidConfig {
            message: "all numeric settings must be finite".to_owned(),
        });
    }
    if config.min_margin < 0.0 {
        return Err(GeometricAugmentationError::InvalidConfig {
            message: "min_margin must be non-negative".to_owned(),
        });
    }
    if config.max_rotation_deg < 0.0 {
        return Err(GeometricAugmentationError::InvalidConfig {
            message: "max_rotation_deg must be non-negative".to_owned(),
        });
    }
    if !(0.0 < config.min_x_scale && config.min_x_scale <= config.max_x_scale) {
        return Err(GeometricAugmentationError::InvalidConfig {
            message: "x scale bounds must satisfy 0 < min <= max".to_owned(),
        });
    }
    if !(0.0 < config.min_y_scale && config.min_y_scale <= config.max_y_scale) {
        return Err(GeometricAugmentationError::InvalidConfig {
            message: "y scale bounds must satisfy 0 < min <= max".to_owned(),
        });
    }
    if config.max_translation_fraction < 0.0 {
        return Err(GeometricAugmentationError::InvalidConfig {
            message: "max_translation_fraction must be non-negative".to_owned(),
        });
    }
    if config.max_attempts == 0 && config.enabled {
        return Err(GeometricAugmentationError::InvalidConfig {
            message: "max_attempts must be greater than zero when enabled".to_owned(),
        });
    }
    Ok(())
}

fn sample_transform(
    page_size: SvgPageSize,
    config: GeometricAugmentationConfig,
    rng: &mut DeterministicRng,
) -> GeometricTransform {
    let max_tx = page_size.width * config.max_translation_fraction;
    let max_ty = page_size.height * config.max_translation_fraction;
    GeometricTransform {
        angle_deg: rng.range(-config.max_rotation_deg, config.max_rotation_deg),
        translate_x: rng.range(-max_tx, max_tx),
        translate_y: rng.range(-max_ty, max_ty),
        scale_x: rng.range(config.min_x_scale, config.max_x_scale),
        scale_y: rng.range(config.min_y_scale, config.max_y_scale),
        center_x: page_size.width / 2.0,
        center_y: page_size.height / 2.0,
    }
}

fn transformed_content_box(
    content_box: SvgContentBox,
    transform: GeometricTransform,
) -> SvgContentBox {
    let x0 = content_box.x;
    let y0 = content_box.y;
    let x1 = content_box.x + content_box.width;
    let y1 = content_box.y + content_box.height;
    let points = [
        transform_point(x0, y0, transform),
        transform_point(x1, y0, transform),
        transform_point(x1, y1, transform),
        transform_point(x0, y1, transform),
    ];
    let min_x = points.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
    let max_x = points
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = points.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let max_y = points
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);

    SvgContentBox {
        x: min_x,
        y: min_y,
        width: max_x - min_x,
        height: max_y - min_y,
    }
}

fn transform_point(x: f64, y: f64, transform: GeometricTransform) -> (f64, f64) {
    let radians = transform.angle_deg.to_radians();
    let cos = radians.cos();
    let sin = radians.sin();
    let centered_x = (x - transform.center_x) * transform.scale_x;
    let centered_y = (y - transform.center_y) * transform.scale_y;
    (
        transform.center_x + centered_x * cos - centered_y * sin + transform.translate_x,
        transform.center_y + centered_x * sin + centered_y * cos + transform.translate_y,
    )
}

fn transform_fits_page(
    content_box: SvgContentBox,
    page_size: SvgPageSize,
    min_margin: f64,
) -> bool {
    content_box.x >= min_margin
        && content_box.y >= min_margin
        && content_box.x + content_box.width <= page_size.width - min_margin
        && content_box.y + content_box.height <= page_size.height - min_margin
}

fn apply_transform_to_svg(
    svg: &str,
    transform: GeometricTransform,
) -> Result<String, GeometricAugmentationError> {
    if transform.is_identity() {
        return Ok(svg.to_owned());
    }

    let svg_start = svg
        .find("<svg")
        .ok_or(GeometricAugmentationError::MissingSvgStart)?;
    let start_end = svg[svg_start..]
        .find('>')
        .ok_or(GeometricAugmentationError::MissingSvgStart)?
        + svg_start
        + 1;
    let end_start = svg
        .rfind("</svg>")
        .ok_or(GeometricAugmentationError::MissingSvgEnd)?;
    let mut augmented = String::with_capacity(svg.len() + 256);
    augmented.push_str(&svg[..start_end]);
    augmented.push_str("\n<g class=\"geometric-augmentation\" transform=\"");
    augmented.push_str(&svg_transform_attribute(transform));
    augmented.push_str("\">");
    augmented.push_str(&svg[start_end..end_start]);
    augmented.push_str("</g>\n");
    augmented.push_str(&svg[end_start..]);
    Ok(augmented)
}

fn svg_transform_attribute(transform: GeometricTransform) -> String {
    format!(
        "translate({:.4} {:.4}) translate({:.4} {:.4}) rotate({:.4}) scale({:.6} {:.6}) translate({:.4} {:.4})",
        transform.translate_x,
        transform.translate_y,
        transform.center_x,
        transform.center_y,
        transform.angle_deg,
        transform.scale_x,
        transform.scale_y,
        -transform.center_x,
        -transform.center_y,
    )
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: GeometricAugmentationSeed) -> Self {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        state ^= (seed.file_index as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state ^= (seed.file_count as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
        state ^= (seed.variant_index as u64).wrapping_mul(0xd2b7_4407_b1ce_6e93);
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        const DENOMINATOR: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / DENOMINATOR
    }

    fn range(&mut self, min: f64, max: f64) -> f64 {
        min + self.unit() * (max - min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page_size() -> SvgPageSize {
        SvgPageSize {
            width: 100.0,
            height: 200.0,
        }
    }

    fn content_box() -> SvgContentBox {
        SvgContentBox {
            x: 25.0,
            y: 60.0,
            width: 50.0,
            height: 80.0,
        }
    }

    fn seed() -> GeometricAugmentationSeed {
        GeometricAugmentationSeed {
            file_index: 3,
            file_count: 10,
            variant_index: 0,
        }
    }

    #[test]
    fn identity_transform_preserves_content_box() {
        assert_eq!(
            transformed_content_box(content_box(), GeometricTransform::identity(page_size())),
            content_box()
        );
    }

    #[test]
    fn translation_moves_content_box() {
        let transform = GeometricTransform {
            translate_x: 4.0,
            translate_y: -5.0,
            ..GeometricTransform::identity(page_size())
        };

        assert_eq!(
            transformed_content_box(content_box(), transform),
            SvgContentBox {
                x: 29.0,
                y: 55.0,
                width: 50.0,
                height: 80.0,
            }
        );
    }

    #[test]
    fn rotation_expands_axis_aligned_content_box() {
        let transform = GeometricTransform {
            angle_deg: 10.0,
            ..GeometricTransform::identity(page_size())
        };

        let transformed = transformed_content_box(content_box(), transform);

        assert!(transformed.width > content_box().width);
        assert!(transformed.height > content_box().height);
    }

    #[test]
    fn fit_gate_rejects_content_outside_page_margin() {
        assert!(!transform_fits_page(
            SvgContentBox {
                x: 1.0,
                y: 20.0,
                width: 40.0,
                height: 40.0,
            },
            page_size(),
            2.0
        ));
    }

    #[test]
    fn fit_gate_accepts_content_inside_page_margin() {
        assert!(transform_fits_page(content_box(), page_size(), 2.0));
    }

    #[test]
    fn wraps_svg_contents_in_geometric_group() {
        let svg = r#"<svg viewBox="0 0 100 200"><path d="M0 0 L10 0" /></svg>"#;
        let transform = GeometricTransform {
            angle_deg: 1.0,
            ..GeometricTransform::identity(page_size())
        };

        let augmented = apply_transform_to_svg(svg, transform).unwrap();

        assert!(augmented.contains(r#"<g class="geometric-augmentation" transform=""#));
        assert!(augmented.contains(r#"<path d="M0 0 L10 0" />"#));
        assert!(augmented.ends_with("</svg>"));
    }

    #[test]
    fn wraps_svg_contents_after_xml_declaration() {
        let svg =
            r#"<?xml version="1.0" encoding="UTF-8"?><svg viewBox="0 0 100 200"><path /></svg>"#;
        let transform = GeometricTransform {
            angle_deg: 1.0,
            ..GeometricTransform::identity(page_size())
        };

        let augmented = apply_transform_to_svg(svg, transform).unwrap();

        assert!(augmented.starts_with(r#"<?xml version="1.0" encoding="UTF-8"?><svg"#));
        assert!(augmented.contains(r#"<g class="geometric-augmentation" transform=""#));
    }

    #[test]
    fn disabled_augmentation_returns_original_svg() {
        let svg = r#"<svg viewBox="0 0 100 200"><path d="M0 0 L10 0" /></svg>"#;
        let result = augment_svg_geometry(
            svg,
            content_box(),
            GeometricAugmentationConfig {
                enabled: false,
                ..GeometricAugmentationConfig::conservative_page_geometry()
            },
            seed(),
        )
        .unwrap();

        assert_eq!(result.svg, svg);
        assert_eq!(result.status, GeometricAugmentationStatus::Disabled);
        assert_eq!(result.transform, None);
        assert_eq!(result.attempt_count, 0);
    }

    #[test]
    fn falls_back_to_identity_when_no_sample_fits() {
        let svg = r#"<svg viewBox="0 0 100 200"><path d="M0 0 L10 0" /></svg>"#;
        let result = augment_svg_geometry(
            svg,
            SvgContentBox {
                x: 1.0,
                y: 1.0,
                width: 98.0,
                height: 198.0,
            },
            GeometricAugmentationConfig {
                min_margin: 4.0,
                max_attempts: 2,
                ..GeometricAugmentationConfig::conservative_page_geometry()
            },
            seed(),
        )
        .unwrap();

        assert_eq!(result.svg, svg);
        assert_eq!(result.status, GeometricAugmentationStatus::IdentityFallback);
        assert_eq!(result.transform, None);
        assert_eq!(result.attempt_count, 2);
    }

    #[test]
    fn applies_deterministic_safe_augmentation() {
        let svg = r#"<svg viewBox="0 0 100 200"><path d="M0 0 L10 0" /></svg>"#;
        let first = augment_svg_geometry(
            svg,
            content_box(),
            GeometricAugmentationConfig::conservative_page_geometry(),
            seed(),
        )
        .unwrap();
        let second = augment_svg_geometry(
            svg,
            content_box(),
            GeometricAugmentationConfig::conservative_page_geometry(),
            seed(),
        )
        .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.status, GeometricAugmentationStatus::Applied);
        assert!(first.transform.is_some());
        assert_ne!(first.svg, svg);
        assert!(transform_fits_page(
            first.transformed_content_box,
            page_size(),
            GeometricAugmentationConfig::conservative_page_geometry().min_margin,
        ));
    }
}
