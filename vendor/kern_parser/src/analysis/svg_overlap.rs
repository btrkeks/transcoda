//! Verovio SVG note/rest bounding-box overlap detection.

use std::collections::BTreeMap;

use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct SvgBBox {
    pub object_id: String,
    pub group_id: String,
    pub source_id: String,
    pub kind: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub content: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SvgOverlapPair {
    pub first_id: String,
    pub second_id: String,
    pub first_kind: String,
    pub second_kind: String,
    pub area: f64,
    pub ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SvgOverlapSummary {
    pub page_no: u32,
    pub candidate_count: usize,
    pub overlap_count: usize,
    pub worst_pairs: Vec<SvgOverlapPair>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SvgOverlapOptions {
    pub min_overlap_ratio: f64,
    pub max_reported_pairs: usize,
    pub target_kinds: Vec<String>,
    pub ignore_same_source: bool,
    pub ignore_shared_notehead_columns: bool,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SvgOverlapError {
    #[error("malformed SVG XML: {message}")]
    MalformedXml { message: String },
}

impl Default for SvgOverlapOptions {
    fn default() -> Self {
        Self {
            min_overlap_ratio: 0.15,
            max_reported_pairs: 3,
            target_kinds: vec!["note".to_owned(), "rest".to_owned()],
            ignore_same_source: true,
            ignore_shared_notehead_columns: true,
        }
    }
}

pub fn extract_verovio_svg_bboxes(svg: &str) -> Result<Vec<SvgBBox>, SvgOverlapError> {
    let document =
        roxmltree::Document::parse(svg).map_err(|error| SvgOverlapError::MalformedXml {
            message: error.to_string(),
        })?;

    let mut boxes = BTreeMap::new();
    for node in document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "g")
    {
        let Some(bbox) = bbox_from_group(node) else {
            continue;
        };
        match boxes.get(&bbox.object_id) {
            Some(current) if should_replace_box(current, &bbox) => {
                boxes.insert(bbox.object_id.clone(), bbox);
            }
            None => {
                boxes.insert(bbox.object_id.clone(), bbox);
            }
            _ => {}
        }
    }

    Ok(boxes.into_values().collect())
}

pub fn detect_bbox_overlaps(
    boxes: &[SvgBBox],
    page_no: u32,
    options: &SvgOverlapOptions,
) -> SvgOverlapSummary {
    let candidates = boxes
        .iter()
        .filter(|box_| options.target_kinds.iter().any(|kind| kind == &box_.kind))
        .collect::<Vec<_>>();
    let mut overlaps = Vec::new();

    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let first = candidates[i];
            let second = candidates[j];
            if should_ignore_pair(first, second, options) {
                continue;
            }

            let overlap_area = overlap_area(first, second);
            if overlap_area <= 0.0 {
                continue;
            }
            let min_area = area(first).min(area(second));
            if min_area <= 0.0 {
                continue;
            }
            let ratio = overlap_area / min_area;
            if ratio < options.min_overlap_ratio {
                continue;
            }

            overlaps.push(SvgOverlapPair {
                first_id: first.object_id.clone(),
                second_id: second.object_id.clone(),
                first_kind: first.kind.clone(),
                second_kind: second.kind.clone(),
                area: overlap_area,
                ratio,
            });
        }
    }

    overlaps.sort_by(|a, b| {
        b.ratio
            .total_cmp(&a.ratio)
            .then_with(|| b.area.total_cmp(&a.area))
    });

    let overlap_count = overlaps.len();
    overlaps.truncate(options.max_reported_pairs);

    SvgOverlapSummary {
        page_no,
        candidate_count: candidates.len(),
        overlap_count,
        worst_pairs: overlaps,
    }
}

pub fn detect_svg_bbox_overlaps(
    svg: &str,
    page_no: u32,
    options: &SvgOverlapOptions,
) -> Result<SvgOverlapSummary, SvgOverlapError> {
    let boxes = extract_verovio_svg_bboxes(svg)?;
    Ok(detect_bbox_overlaps(&boxes, page_no, options))
}

fn bbox_from_group(group: roxmltree::Node<'_, '_>) -> Option<SvgBBox> {
    let classes = group.attribute("class")?;
    let is_content = has_class(classes, "content-bounding-box");
    if !is_content && !has_class(classes, "bounding-box") {
        return None;
    }

    let id = group.attribute("id")?;
    let prefix = if is_content { "cbbox-" } else { "bbox-" };
    let object_id = id.strip_prefix(prefix)?.to_owned();
    let dash = object_id.find('-')?;
    let kind = object_id[..dash].to_owned();

    let mut bounds: Option<RectBounds> = None;
    for rect in group
        .children()
        .filter(|node| node.is_element() && node.tag_name().name() == "rect")
    {
        let x = rect
            .attribute("x")
            .unwrap_or("0")
            .parse::<f64>()
            .unwrap_or(0.0);
        let y = rect
            .attribute("y")
            .unwrap_or("0")
            .parse::<f64>()
            .unwrap_or(0.0);
        let width = rect
            .attribute("width")
            .unwrap_or("0")
            .parse::<f64>()
            .unwrap_or(0.0);
        let height = rect
            .attribute("height")
            .unwrap_or("0")
            .parse::<f64>()
            .unwrap_or(0.0);
        if width == 0.0 || height == 0.0 {
            continue;
        }

        let rect_bounds = RectBounds {
            left: x.min(x + width),
            top: y.min(y + height),
            right: x.max(x + width),
            bottom: y.max(y + height),
        };
        bounds = Some(match bounds {
            Some(current) => current.union(rect_bounds),
            None => rect_bounds,
        });
    }

    let bounds = bounds?;
    if bounds.right <= bounds.left || bounds.bottom <= bounds.top {
        return None;
    }

    Some(SvgBBox {
        group_id: logical_group_id(group, &object_id),
        source_id: source_line_id(&object_id),
        kind,
        object_id,
        x: bounds.left,
        y: bounds.top,
        width: bounds.right - bounds.left,
        height: bounds.bottom - bounds.top,
        content: is_content,
    })
}

fn should_replace_box(current: &SvgBBox, candidate: &SvgBBox) -> bool {
    if candidate.kind == "note" {
        return current.content && !candidate.content;
    }

    candidate.content && !current.content
}

fn logical_group_id(group: roxmltree::Node<'_, '_>, object_id: &str) -> String {
    if let Some(chord_id) = ancestor_id_with_prefix(group, "chord-") {
        return chord_id;
    }
    strip_trailing_voice_suffix(object_id).unwrap_or_default()
}

fn ancestor_id_with_prefix(group: roxmltree::Node<'_, '_>, prefix: &str) -> Option<String> {
    let mut parent = group.parent();
    while let Some(node) = parent {
        if let Some(id) = node.attribute("id") {
            if id.starts_with(prefix) {
                return Some(id.to_owned());
            }
        }
        parent = node.parent();
    }
    None
}

fn strip_trailing_voice_suffix(object_id: &str) -> Option<String> {
    let mut pos = object_id.len();
    while pos > 0 && object_id.as_bytes()[pos - 1].is_ascii_digit() {
        pos -= 1;
    }
    if pos == object_id.len() || pos == 0 || object_id.as_bytes()[pos - 1] != b'S' {
        return None;
    }
    Some(object_id[..pos - 1].to_owned())
}

fn source_line_id(object_id: &str) -> String {
    let Some(pos) = object_id.find("-L") else {
        return String::new();
    };
    let start = pos + 1;
    let mut end = start + 1;
    while end < object_id.len() && object_id.as_bytes()[end].is_ascii_digit() {
        end += 1;
    }
    object_id[start..end].to_owned()
}

fn should_ignore_pair(a: &SvgBBox, b: &SvgBBox, options: &SvgOverlapOptions) -> bool {
    if !a.group_id.is_empty() && a.group_id == b.group_id {
        return true;
    }
    if options.ignore_same_source && !a.source_id.is_empty() && a.source_id == b.source_id {
        return true;
    }
    is_shared_notehead_column(a, b, options)
}

fn is_shared_notehead_column(a: &SvgBBox, b: &SvgBBox, options: &SvgOverlapOptions) -> bool {
    if !options.ignore_shared_notehead_columns || a.kind != "note" || b.kind != "note" {
        return false;
    }

    let min_width = a.width.min(b.width);
    let min_height = a.height.min(b.height);
    if min_width <= 0.0 || min_height <= 0.0 {
        return false;
    }

    let y_tolerance = 4.0_f64.max(min_height * 0.10);
    let x_tolerance = 4.0_f64.max(min_width * 0.50);
    (center_y(a) - center_y(b)).abs() <= y_tolerance
        && (center_x(a) - center_x(b)).abs() <= x_tolerance
}

fn area(box_: &SvgBBox) -> f64 {
    box_.width * box_.height
}

fn overlap_area(a: &SvgBBox, b: &SvgBBox) -> f64 {
    let x_overlap = 0.0_f64.max((a.x + a.width).min(b.x + b.width) - a.x.max(b.x));
    let y_overlap = 0.0_f64.max((a.y + a.height).min(b.y + b.height) - a.y.max(b.y));
    x_overlap * y_overlap
}

fn center_x(box_: &SvgBBox) -> f64 {
    box_.x + box_.width / 2.0
}

fn center_y(box_: &SvgBBox) -> f64 {
    box_.y + box_.height / 2.0
}

fn has_class(classes: &str, token: &str) -> bool {
    classes.split_whitespace().any(|class| class == token)
}

#[derive(Debug, Clone, Copy)]
struct RectBounds {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
}

impl RectBounds {
    fn union(self, other: Self) -> Self {
        Self {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_overlaps_from_normalized_boxes() {
        let boxes = vec![
            SvgBBox {
                object_id: "note-a".to_owned(),
                group_id: String::new(),
                source_id: String::new(),
                kind: "note".to_owned(),
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
                content: false,
            },
            SvgBBox {
                object_id: "note-b".to_owned(),
                group_id: String::new(),
                source_id: String::new(),
                kind: "note".to_owned(),
                x: 5.0,
                y: 5.0,
                width: 10.0,
                height: 10.0,
                content: false,
            },
            SvgBBox {
                object_id: "clef-c".to_owned(),
                group_id: String::new(),
                source_id: String::new(),
                kind: "clef".to_owned(),
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
                content: false,
            },
        ];

        let summary = detect_bbox_overlaps(&boxes, 4, &SvgOverlapOptions::default());

        assert_eq!(summary.page_no, 4);
        assert_eq!(summary.candidate_count, 2);
        assert_eq!(summary.overlap_count, 1);
        assert_eq!(summary.worst_pairs.len(), 1);
        assert_eq!(summary.worst_pairs[0].first_id, "note-a");
        assert_eq!(summary.worst_pairs[0].second_id, "note-b");
        assert_eq!(summary.worst_pairs[0].ratio, 0.25);
    }

    #[test]
    fn detects_note_rest_bbox_collisions_and_ignores_score_furniture() {
        let svg = r#"
<svg>
  <g id="bbox-staff-L1F1" class="bounding-box"><rect x="0" y="0" width="200" height="20"/></g>
  <g id="bbox-clef-L1F1" class="bounding-box"><rect x="0" y="0" width="20" height="20"/></g>
  <g id="bbox-note-a" class="bounding-box"><rect x="0" y="0" width="10" height="10"/></g>
  <g id="cbbox-note-a" class="content-bounding-box"><rect x="0" y="0" width="20" height="20"/></g>
  <g id="bbox-note-b" class="bounding-box"><rect x="5" y="5" width="10" height="10"/></g>
  <g id="bbox-rest-c" class="bounding-box"><rect x="100" y="100" width="10" height="10"/></g>
</svg>
"#;

        let summary = detect_svg_bbox_overlaps(svg, 3, &SvgOverlapOptions::default()).unwrap();

        assert_eq!(summary.page_no, 3);
        assert_eq!(summary.candidate_count, 3);
        assert_eq!(summary.overlap_count, 1);
        assert_eq!(summary.worst_pairs.len(), 1);
        assert_eq!(summary.worst_pairs[0].first_id, "note-a");
        assert_eq!(summary.worst_pairs[0].second_id, "note-b");
        assert_eq!(summary.worst_pairs[0].ratio, 0.25);
    }

    #[test]
    fn prefers_notehead_boxes_over_note_content_boxes() {
        let svg = r#"
<svg>
  <g id="note-a" class="note">
    <g id="cbbox-note-a" class="note content-bounding-box"><rect x="0" y="0" width="10" height="120"/></g>
    <g id="bbox-note-a" class="note bounding-box"><rect x="0" y="110" width="10" height="10"/></g>
  </g>
  <g id="note-b" class="note">
    <g id="cbbox-note-b" class="note content-bounding-box"><rect x="0" y="0" width="10" height="10"/></g>
    <g id="bbox-note-b" class="note bounding-box"><rect x="0" y="0" width="10" height="10"/></g>
  </g>
</svg>
"#;

        let summary = detect_svg_bbox_overlaps(svg, 1, &SvgOverlapOptions::default()).unwrap();

        assert_eq!(summary.candidate_count, 2);
        assert_eq!(summary.overlap_count, 0);
    }

    #[test]
    fn ignores_expected_same_event_notehead_sharing() {
        let svg = r#"
<svg>
  <g id="chord-L10F1" class="chord">
    <g id="note-L10F1S1" class="note">
      <g id="bbox-note-L10F1S1" class="note bounding-box"><rect x="0" y="0" width="10" height="10"/></g>
    </g>
    <g id="note-L10F1S2" class="note">
      <g id="bbox-note-L10F1S2" class="note bounding-box"><rect x="0" y="0" width="10" height="10"/></g>
    </g>
  </g>
  <g id="bbox-note-L20F1" class="note bounding-box"><rect x="20" y="0" width="10" height="10"/></g>
  <g id="bbox-note-L20F2" class="note bounding-box"><rect x="20" y="0" width="10" height="10"/></g>
  <g id="bbox-note-L30F1" class="note bounding-box"><rect x="40" y="0" width="10" height="10"/></g>
  <g id="bbox-note-L31F2" class="note bounding-box"><rect x="44" y="0" width="10" height="10"/></g>
</svg>
"#;

        let summary = detect_svg_bbox_overlaps(svg, 2, &SvgOverlapOptions::default()).unwrap();

        assert_eq!(summary.candidate_count, 6);
        assert_eq!(summary.overlap_count, 0);
    }
}
