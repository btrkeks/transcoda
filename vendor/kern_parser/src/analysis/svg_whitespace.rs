//! SVG content-box and whitespace analysis for pages rendered by notation engines.

use thiserror::Error;

/// Remaining space between the measured SVG content and each page border.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvgWhitespace {
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
}

/// Axis-aligned bounds of the measured SVG content in SVG user units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvgContentBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Root SVG page dimensions in SVG user units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvgPageSize {
    pub width: f64,
    pub height: f64,
}

/// Page and content measurements for a rendered SVG page.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvgPageWhitespace {
    pub content_box: SvgContentBox,
    pub whitespace: SvgWhitespace,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SvgContentBoxError {
    #[error("malformed SVG XML: {message}")]
    MalformedXml { message: String },
    #[error("SVG does not contain a system content-bounding-box")]
    MissingSystemContentBox,
    #[error("system content-bounding-box is missing a direct rect child")]
    MissingRect,
    #[error("system content-bounding-box rect is missing {attribute:?}")]
    MissingRectAttribute { attribute: &'static str },
    #[error("system content-bounding-box rect has invalid {attribute:?} value {value:?}")]
    InvalidRectAttribute {
        attribute: &'static str,
        value: String,
    },
    #[error("definition-scale SVG is missing viewBox")]
    MissingDefinitionScaleViewBox,
    #[error("definition-scale SVG has invalid viewBox value {value:?}")]
    InvalidDefinitionScaleViewBox { value: String },
    #[error("unsupported SVG transform {value:?}; expected translate(x, y)")]
    InvalidTransform { value: String },
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SvgPageSizeError {
    #[error("malformed SVG XML: {message}")]
    MalformedXml { message: String },
    #[error("SVG root element is not <svg>")]
    MissingSvgRoot,
    #[error("SVG root is missing viewBox or width/height attributes")]
    MissingPageSize,
    #[error("SVG root has invalid {attribute:?} value {value:?}")]
    InvalidPageSizeAttribute {
        attribute: &'static str,
        value: String,
    },
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SvgWhitespaceError {
    #[error("SVG content box detection failed: {0}")]
    ContentBox(#[from] SvgContentBoxError),
    #[error("SVG page size detection failed: {0}")]
    PageSize(#[from] SvgPageSizeError),
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SvgSystemCountError {
    #[error("malformed SVG XML: {message}")]
    MalformedXml { message: String },
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SvgStripBoundingBoxesError {
    #[error("malformed SVG XML: {message}")]
    MalformedXml { message: String },
}

/// Detect the root SVG page size from `viewBox`, falling back to `width`/`height`.
pub fn detect_svg_page_size(svg: &str) -> Result<SvgPageSize, SvgPageSizeError> {
    let document =
        roxmltree::Document::parse(svg).map_err(|error| SvgPageSizeError::MalformedXml {
            message: error.to_string(),
        })?;
    let root = document.root_element();

    if root.tag_name().name() != "svg" {
        return Err(SvgPageSizeError::MissingSvgRoot);
    }

    if let Some(view_box) = root.attribute("viewBox") {
        return parse_view_box(view_box);
    }

    let width = root
        .attribute("width")
        .ok_or(SvgPageSizeError::MissingPageSize)
        .and_then(|value| parse_svg_length(value, "width"))?;
    let height = root
        .attribute("height")
        .ok_or(SvgPageSizeError::MissingPageSize)
        .and_then(|value| parse_svg_length(value, "height"))?;

    Ok(SvgPageSize { width, height })
}

/// Detect the union of Verovio system content bounds in SVG user units.
pub fn detect_svg_content_box(svg: &str) -> Result<SvgContentBox, SvgContentBoxError> {
    let document =
        roxmltree::Document::parse(svg).map_err(|error| SvgContentBoxError::MalformedXml {
            message: error.to_string(),
        })?;

    let mut union: Option<Bounds> = None;
    for node in document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "g")
        .filter(|node| has_class(*node, "system") && has_class(*node, "content-bounding-box"))
    {
        let rect = node
            .children()
            .find(|child| child.is_element() && child.tag_name().name() == "rect")
            .ok_or(SvgContentBoxError::MissingRect)?;
        let content_box = parse_rect(rect)?;

        if content_box.width == 0.0 && content_box.height == 0.0 {
            continue;
        }

        let bounds = Bounds::from_box(content_box);
        union = Some(match union {
            Some(existing) => existing.union(bounds),
            None => bounds,
        });
    }

    union
        .map(Bounds::into_box)
        .ok_or(SvgContentBoxError::MissingSystemContentBox)
}

/// Count rendered Verovio systems in an SVG page.
pub fn count_svg_systems(svg: &str) -> Result<usize, SvgSystemCountError> {
    let document =
        roxmltree::Document::parse(svg).map_err(|error| SvgSystemCountError::MalformedXml {
            message: error.to_string(),
        })?;

    Ok(document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "g")
        .filter(|node| {
            has_class(*node, "system")
                && !has_class(*node, "content-bounding-box")
                && !has_class(*node, "bounding-box")
        })
        .count())
}

/// Remove Verovio bounding-box groups from an SVG while preserving other content.
pub fn strip_svg_bounding_boxes(svg: &str) -> Result<String, SvgStripBoundingBoxesError> {
    let document = roxmltree::Document::parse(svg).map_err(|error| {
        SvgStripBoundingBoxesError::MalformedXml {
            message: error.to_string(),
        }
    })?;
    let mut ranges = document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "g")
        .filter(|node| has_class(*node, "bounding-box") || has_class(*node, "content-bounding-box"))
        .map(|node| node.range())
        .collect::<Vec<_>>();

    ranges.sort_by_key(|range| range.start);

    let mut stripped = String::with_capacity(svg.len());
    let mut cursor = 0;
    for range in ranges {
        if range.start < cursor {
            continue;
        }
        stripped.push_str(&svg[cursor..range.start]);
        cursor = range.end;
    }
    stripped.push_str(&svg[cursor..]);

    Ok(stripped)
}

/// Measure remaining space between the main rendered SVG content and page borders.
pub fn measure_svg_whitespace(svg: &str) -> Result<SvgPageWhitespace, SvgWhitespaceError> {
    let page_size = detect_svg_page_size(svg)?;
    let content_box = match detect_verovio_page_content_box(svg, page_size)? {
        Some(content_box) => content_box,
        None => detect_svg_content_box(svg)?,
    };

    Ok(SvgPageWhitespace {
        content_box,
        whitespace: SvgWhitespace {
            left: content_box.x,
            right: page_size.width - (content_box.x + content_box.width),
            top: content_box.y,
            bottom: page_size.height - (content_box.y + content_box.height),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_single_system_content_box() {
        let content_box = detect_svg_content_box(
            r#"<svg>
                <g class="system content-bounding-box">
                    <rect x="10" y="20" width="30" height="40" />
                </g>
            </svg>"#,
        )
        .unwrap();

        assert_eq!(
            content_box,
            SvgContentBox {
                x: 10.0,
                y: 20.0,
                width: 30.0,
                height: 40.0
            }
        );
    }

    #[test]
    fn detects_page_size_from_view_box() {
        let page_size =
            detect_svg_page_size(r#"<svg viewBox="0 0 1050 1485" width="1" height="1"></svg>"#)
                .unwrap();

        assert_eq!(
            page_size,
            SvgPageSize {
                width: 1050.0,
                height: 1485.0
            }
        );
    }

    #[test]
    fn detects_page_size_from_width_and_height() {
        let page_size =
            detect_svg_page_size(r#"<svg width="1050px" height="1485px"></svg>"#).unwrap();

        assert_eq!(
            page_size,
            SvgPageSize {
                width: 1050.0,
                height: 1485.0
            }
        );
    }

    #[test]
    fn errors_when_page_size_is_missing() {
        let error = detect_svg_page_size("<svg></svg>").unwrap_err();

        assert_eq!(error, SvgPageSizeError::MissingPageSize);
    }

    #[test]
    fn errors_when_page_size_is_invalid() {
        let error = detect_svg_page_size(r#"<svg viewBox="0 0 nope 1485"></svg>"#).unwrap_err();

        assert_eq!(
            error,
            SvgPageSizeError::InvalidPageSizeAttribute {
                attribute: "viewBox",
                value: "0 0 nope 1485".to_owned()
            }
        );
    }

    #[test]
    fn unions_multiple_system_content_boxes() {
        let content_box = detect_svg_content_box(
            r#"<svg>
                <g class="system content-bounding-box">
                    <rect x="10" y="20" width="30" height="40" />
                </g>
                <g class="system content-bounding-box">
                    <rect x="50" y="5" width="20" height="10" />
                </g>
            </svg>"#,
        )
        .unwrap();

        assert_eq!(
            content_box,
            SvgContentBox {
                x: 10.0,
                y: 5.0,
                width: 60.0,
                height: 55.0
            }
        );
    }

    #[test]
    fn ignores_non_system_content_boxes() {
        let content_box = detect_svg_content_box(
            r#"<svg>
                <g class="note content-bounding-box">
                    <rect x="1" y="2" width="300" height="400" />
                </g>
                <g class="system content-bounding-box">
                    <rect x="10" y="20" width="30" height="40" />
                </g>
            </svg>"#,
        )
        .unwrap();

        assert_eq!(
            content_box,
            SvgContentBox {
                x: 10.0,
                y: 20.0,
                width: 30.0,
                height: 40.0
            }
        );
    }

    #[test]
    fn ignores_zero_sized_system_content_boxes() {
        let content_box = detect_svg_content_box(
            r#"<svg>
                <g class="system content-bounding-box">
                    <rect x="1" y="2" width="0" height="0" />
                </g>
                <g class="system content-bounding-box">
                    <rect x="10" y="20" width="30" height="40" />
                </g>
            </svg>"#,
        )
        .unwrap();

        assert_eq!(
            content_box,
            SvgContentBox {
                x: 10.0,
                y: 20.0,
                width: 30.0,
                height: 40.0
            }
        );
    }

    #[test]
    fn counts_rendered_system_groups() {
        let system_count = count_svg_systems(
            r#"<svg>
                <g class="system">
                    <g class="system content-bounding-box">
                        <rect x="0" y="0" width="10" height="10" />
                    </g>
                    <g class="system bounding-box" />
                </g>
                <g class="system">
                    <g class="system content-bounding-box">
                        <rect x="0" y="20" width="10" height="10" />
                    </g>
                </g>
                <g class="section systemMilestone" />
                <g class="systemMilestoneEnd section-L1F1" />
            </svg>"#,
        )
        .unwrap();

        assert_eq!(system_count, 2);
    }

    #[test]
    fn errors_when_counting_systems_in_malformed_xml() {
        let error = count_svg_systems("<svg><g></svg>").unwrap_err();

        assert!(matches!(error, SvgSystemCountError::MalformedXml { .. }));
    }

    #[test]
    fn strips_bounding_box_groups() {
        let stripped = strip_svg_bounding_boxes(
            r##"<svg>
                <g class="system">
                    <g class="system content-bounding-box">
                        <rect x="0" y="0" width="10" height="10" />
                    </g>
                    <path d="M0 0 L10 0" />
                    <g class="system bounding-box" />
                </g>
                <g class="note">
                    <g class="note content-bounding-box">
                        <rect x="1" y="1" width="2" height="2" />
                    </g>
                    <use href="#notehead" />
                </g>
            </svg>"##,
        )
        .unwrap();

        assert!(stripped.contains(r#"<g class="system">"#));
        assert!(stripped.contains(r#"<path d="M0 0 L10 0" />"#));
        assert!(stripped.contains(r##"<use href="#notehead" />"##));
        assert!(!stripped.contains("bounding-box"));
        assert!(!stripped.contains("content-bounding-box"));
        assert!(!stripped.contains("<rect"));
    }

    #[test]
    fn errors_when_stripping_bounding_boxes_in_malformed_xml() {
        let error = strip_svg_bounding_boxes("<svg><g></svg>").unwrap_err();

        assert!(matches!(
            error,
            SvgStripBoundingBoxesError::MalformedXml { .. }
        ));
    }

    #[test]
    fn errors_when_system_content_boxes_are_absent() {
        let error = detect_svg_content_box(
            r#"<svg>
                <g class="note content-bounding-box">
                    <rect x="1" y="2" width="3" height="4" />
                </g>
            </svg>"#,
        )
        .unwrap_err();

        assert_eq!(error, SvgContentBoxError::MissingSystemContentBox);
    }

    #[test]
    fn errors_on_malformed_xml() {
        let error = detect_svg_content_box("<svg><g></svg>").unwrap_err();

        assert!(matches!(error, SvgContentBoxError::MalformedXml { .. }));
    }

    #[test]
    fn errors_when_system_content_box_is_missing_rect() {
        let error =
            detect_svg_content_box(r#"<svg><g class="system content-bounding-box"/></svg>"#)
                .unwrap_err();

        assert_eq!(error, SvgContentBoxError::MissingRect);
    }

    #[test]
    fn errors_when_rect_attribute_is_missing() {
        let error = detect_svg_content_box(
            r#"<svg>
                <g class="system content-bounding-box">
                    <rect x="10" y="20" width="30" />
                </g>
            </svg>"#,
        )
        .unwrap_err();

        assert_eq!(
            error,
            SvgContentBoxError::MissingRectAttribute {
                attribute: "height"
            }
        );
    }

    #[test]
    fn errors_when_rect_attribute_is_invalid() {
        let error = detect_svg_content_box(
            r#"<svg>
                <g class="system content-bounding-box">
                    <rect x="10" y="20" width="nope" height="40" />
                </g>
            </svg>"#,
        )
        .unwrap_err();

        assert_eq!(
            error,
            SvgContentBoxError::InvalidRectAttribute {
                attribute: "width",
                value: "nope".to_owned()
            }
        );
    }

    #[test]
    fn measures_whitespace_from_view_box() {
        let page = measure_svg_whitespace(
            r#"<svg viewBox="0 0 100 200">
                <g class="system content-bounding-box">
                    <rect x="10" y="20" width="30" height="40" />
                </g>
            </svg>"#,
        )
        .unwrap();

        assert_eq!(
            page,
            SvgPageWhitespace {
                content_box: SvgContentBox {
                    x: 10.0,
                    y: 20.0,
                    width: 30.0,
                    height: 40.0,
                },
                whitespace: SvgWhitespace {
                    left: 10.0,
                    right: 60.0,
                    top: 20.0,
                    bottom: 140.0,
                },
            }
        );
    }

    #[test]
    fn measures_whitespace_from_width_and_height() {
        let page = measure_svg_whitespace(
            r#"<svg width="100px" height="200px">
                <g class="system content-bounding-box">
                    <rect x="5" y="15" width="25" height="35" />
                </g>
            </svg>"#,
        )
        .unwrap();

        assert_eq!(
            page.whitespace,
            SvgWhitespace {
                left: 5.0,
                right: 70.0,
                top: 15.0,
                bottom: 150.0,
            }
        );
    }

    #[test]
    fn measures_verovio_definition_scale_whitespace_in_page_units() {
        let page = measure_svg_whitespace(
            r#"<svg width="1050px" height="1485px">
                <svg class="definition-scale" viewBox="0 0 30000 42420">
                    <g class="page-margin" transform="translate(2500, 2500)">
                        <g class="system">
                            <g class="system content-bounding-box">
                                <rect x="-180" y="932" width="23488" height="7268" />
                            </g>
                        </g>
                    </g>
                </svg>
            </svg>"#,
        )
        .unwrap();

        assert_content_box_close(
            page.content_box,
            SvgContentBox {
                x: 81.2,
                y: 120.14427157001415,
                width: 822.08,
                height: 254.43140028288548,
            },
        );
        assert_whitespace_close(
            page.whitespace,
            SvgWhitespace {
                left: 81.2,
                right: 146.72,
                top: 120.14427157001415,
                bottom: 1110.4243281471004,
            },
        );
    }

    #[test]
    fn unions_verovio_definition_scale_content_boxes_after_normalizing() {
        let page = measure_svg_whitespace(
            r#"<svg width="100px" height="200px">
                <svg class="definition-scale" viewBox="0 0 1000 2000">
                    <g transform="translate(100 200)">
                        <g class="system content-bounding-box">
                            <rect x="100" y="100" width="200" height="300" />
                        </g>
                    </g>
                    <g transform="translate(50, 1000)">
                        <g class="system content-bounding-box">
                            <rect x="700" y="100" width="100" height="200" />
                        </g>
                    </g>
                </svg>
            </svg>"#,
        )
        .unwrap();

        assert_eq!(
            page.whitespace,
            SvgWhitespace {
                left: 20.0,
                right: 15.0,
                top: 30.0,
                bottom: 70.0,
            }
        );
    }

    #[test]
    fn errors_on_malformed_verovio_translate_transform() {
        let error = measure_svg_whitespace(
            r#"<svg width="100px" height="200px">
                <svg class="definition-scale" viewBox="0 0 1000 2000">
                    <g transform="scale(2)">
                        <g class="system content-bounding-box">
                            <rect x="100" y="100" width="200" height="300" />
                        </g>
                    </g>
                </svg>
            </svg>"#,
        )
        .unwrap_err();

        assert_eq!(
            error,
            SvgWhitespaceError::ContentBox(SvgContentBoxError::InvalidTransform {
                value: "scale(2)".to_owned()
            })
        );
    }

    #[test]
    fn measures_whitespace_from_unioned_system_content_boxes() {
        let page = measure_svg_whitespace(
            r#"<svg viewBox="0 0 100 200">
                <g class="system content-bounding-box">
                    <rect x="10" y="20" width="30" height="40" />
                </g>
                <g class="system content-bounding-box">
                    <rect x="70" y="100" width="20" height="50" />
                </g>
            </svg>"#,
        )
        .unwrap();

        assert_eq!(
            page.whitespace,
            SvgWhitespace {
                left: 10.0,
                right: 10.0,
                top: 20.0,
                bottom: 50.0,
            }
        );
    }

    #[test]
    fn whitespace_errors_when_page_size_is_missing() {
        let error = measure_svg_whitespace(
            r#"<svg>
                <g class="system content-bounding-box">
                    <rect x="10" y="20" width="30" height="40" />
                </g>
            </svg>"#,
        )
        .unwrap_err();

        assert_eq!(
            error,
            SvgWhitespaceError::PageSize(SvgPageSizeError::MissingPageSize)
        );
    }

    fn assert_content_box_close(actual: SvgContentBox, expected: SvgContentBox) {
        assert_close(actual.x, expected.x);
        assert_close(actual.y, expected.y);
        assert_close(actual.width, expected.width);
        assert_close(actual.height, expected.height);
    }

    fn assert_whitespace_close(actual: SvgWhitespace, expected: SvgWhitespace) {
        assert_close(actual.left, expected.left);
        assert_close(actual.right, expected.right);
        assert_close(actual.top, expected.top);
        assert_close(actual.bottom, expected.bottom);
    }

    fn assert_close(actual: f64, expected: f64) {
        let delta = (actual - expected).abs();
        assert!(
            delta < 1e-9,
            "expected {actual} to be within 1e-9 of {expected}, delta was {delta}"
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct Bounds {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl Bounds {
    fn from_box(content_box: SvgContentBox) -> Self {
        Self {
            min_x: content_box.x,
            min_y: content_box.y,
            max_x: content_box.x + content_box.width,
            max_y: content_box.y + content_box.height,
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    fn into_box(self) -> SvgContentBox {
        SvgContentBox {
            x: self.min_x,
            y: self.min_y,
            width: self.max_x - self.min_x,
            height: self.max_y - self.min_y,
        }
    }
}

fn has_class(node: roxmltree::Node<'_, '_>, class: &str) -> bool {
    node.attribute("class")
        .is_some_and(|classes| classes.split_ascii_whitespace().any(|value| value == class))
}

fn detect_verovio_page_content_box(
    svg: &str,
    page_size: SvgPageSize,
) -> Result<Option<SvgContentBox>, SvgWhitespaceError> {
    let document =
        roxmltree::Document::parse(svg).map_err(|error| SvgContentBoxError::MalformedXml {
            message: error.to_string(),
        })?;

    let mut found_definition_scale = false;
    let mut union: Option<Bounds> = None;
    for definition_scale in document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "svg")
        .filter(|node| has_class(*node, "definition-scale"))
    {
        found_definition_scale = true;
        let view_box = definition_scale
            .attribute("viewBox")
            .ok_or(SvgContentBoxError::MissingDefinitionScaleViewBox)?;
        let definition_size = parse_view_box(view_box).map_err(|_| {
            SvgContentBoxError::InvalidDefinitionScaleViewBox {
                value: view_box.to_owned(),
            }
        })?;
        let scale_x = page_size.width / definition_size.width;
        let scale_y = page_size.height / definition_size.height;

        for node in definition_scale
            .descendants()
            .filter(|node| node.is_element() && node.tag_name().name() == "g")
            .filter(|node| has_class(*node, "system") && has_class(*node, "content-bounding-box"))
        {
            let rect = node
                .children()
                .find(|child| child.is_element() && child.tag_name().name() == "rect")
                .ok_or(SvgContentBoxError::MissingRect)?;
            let content_box = parse_rect(rect)?;

            if content_box.width == 0.0 && content_box.height == 0.0 {
                continue;
            }

            let (translate_x, translate_y) =
                accumulated_translate_to_definition_scale(node, definition_scale)?;
            let normalized = SvgContentBox {
                x: (content_box.x + translate_x) * scale_x,
                y: (content_box.y + translate_y) * scale_y,
                width: content_box.width * scale_x,
                height: content_box.height * scale_y,
            };
            let bounds = Bounds::from_box(normalized);
            union = Some(match union {
                Some(existing) => existing.union(bounds),
                None => bounds,
            });
        }
    }

    if !found_definition_scale {
        return Ok(None);
    }

    union
        .map(Bounds::into_box)
        .map(Some)
        .ok_or_else(|| SvgContentBoxError::MissingSystemContentBox.into())
}

fn accumulated_translate_to_definition_scale(
    node: roxmltree::Node<'_, '_>,
    definition_scale: roxmltree::Node<'_, '_>,
) -> Result<(f64, f64), SvgContentBoxError> {
    let mut translate_x = 0.0;
    let mut translate_y = 0.0;
    let mut current = Some(node);

    while let Some(current_node) = current {
        if current_node == definition_scale {
            return Ok((translate_x, translate_y));
        }
        if let Some(transform) = current_node.attribute("transform") {
            let (x, y) = parse_translate_transform(transform)?;
            translate_x += x;
            translate_y += y;
        }
        current = current_node.parent();
    }

    Ok((translate_x, translate_y))
}

fn parse_rect(rect: roxmltree::Node<'_, '_>) -> Result<SvgContentBox, SvgContentBoxError> {
    Ok(SvgContentBox {
        x: parse_rect_attribute(rect, "x")?,
        y: parse_rect_attribute(rect, "y")?,
        width: parse_rect_attribute(rect, "width")?,
        height: parse_rect_attribute(rect, "height")?,
    })
}

fn parse_rect_attribute(
    rect: roxmltree::Node<'_, '_>,
    attribute: &'static str,
) -> Result<f64, SvgContentBoxError> {
    let raw = rect
        .attribute(attribute)
        .ok_or(SvgContentBoxError::MissingRectAttribute { attribute })?;
    let value = raw
        .parse::<f64>()
        .map_err(|_| SvgContentBoxError::InvalidRectAttribute {
            attribute,
            value: raw.to_owned(),
        })?;

    if !value.is_finite() {
        return Err(SvgContentBoxError::InvalidRectAttribute {
            attribute,
            value: raw.to_owned(),
        });
    }

    Ok(value)
}

fn parse_translate_transform(raw: &str) -> Result<(f64, f64), SvgContentBoxError> {
    let trimmed = raw.trim();
    let Some(inner) = trimmed
        .strip_prefix("translate(")
        .and_then(|value| value.strip_suffix(')'))
    else {
        return Err(SvgContentBoxError::InvalidTransform {
            value: raw.to_owned(),
        });
    };
    let values: Vec<&str> = inner
        .split(|character: char| character.is_ascii_whitespace() || character == ',')
        .filter(|value| !value.is_empty())
        .collect();
    if !(1..=2).contains(&values.len()) {
        return Err(SvgContentBoxError::InvalidTransform {
            value: raw.to_owned(),
        });
    }

    let x = values[0]
        .parse::<f64>()
        .map_err(|_| SvgContentBoxError::InvalidTransform {
            value: raw.to_owned(),
        })?;
    let y = values
        .get(1)
        .map_or(Ok(0.0), |value| value.parse::<f64>())
        .map_err(|_| SvgContentBoxError::InvalidTransform {
            value: raw.to_owned(),
        })?;

    if !x.is_finite() || !y.is_finite() {
        return Err(SvgContentBoxError::InvalidTransform {
            value: raw.to_owned(),
        });
    }

    Ok((x, y))
}

fn parse_view_box(raw: &str) -> Result<SvgPageSize, SvgPageSizeError> {
    let values: Vec<&str> = raw
        .split(|character: char| character.is_ascii_whitespace() || character == ',')
        .filter(|value| !value.is_empty())
        .collect();
    if values.len() != 4 {
        return Err(SvgPageSizeError::InvalidPageSizeAttribute {
            attribute: "viewBox",
            value: raw.to_owned(),
        });
    }

    let width =
        values[2]
            .parse::<f64>()
            .map_err(|_| SvgPageSizeError::InvalidPageSizeAttribute {
                attribute: "viewBox",
                value: raw.to_owned(),
            })?;
    let height =
        values[3]
            .parse::<f64>()
            .map_err(|_| SvgPageSizeError::InvalidPageSizeAttribute {
                attribute: "viewBox",
                value: raw.to_owned(),
            })?;

    if !is_positive_finite(width) || !is_positive_finite(height) {
        return Err(SvgPageSizeError::InvalidPageSizeAttribute {
            attribute: "viewBox",
            value: raw.to_owned(),
        });
    }

    Ok(SvgPageSize { width, height })
}

fn parse_svg_length(raw: &str, attribute: &'static str) -> Result<f64, SvgPageSizeError> {
    let trimmed = raw.trim();
    let numeric = trimmed.strip_suffix("px").unwrap_or(trimmed);
    let value = numeric
        .parse::<f64>()
        .map_err(|_| SvgPageSizeError::InvalidPageSizeAttribute {
            attribute,
            value: raw.to_owned(),
        })?;

    if !is_positive_finite(value) {
        return Err(SvgPageSizeError::InvalidPageSizeAttribute {
            attribute,
            value: raw.to_owned(),
        });
    }

    Ok(value)
}

fn is_positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}
