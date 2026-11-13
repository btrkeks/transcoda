//! Collapse canceling spine split/merge regions.
//!
//! This pass removes temporary split regions only when every interior line can
//! be projected back onto the pre-split topology without losing information.
//!
//! Example:
//! ```text
//! Before:
//! *^
//! 4c\t4c
//! *v\t*v
//!
//! After:
//! 4c
//! ```
//!
//! Because both split children carry the same content, the temporary split can
//! be projected back onto the original parent spine exactly.
//!
//! This is different from [`crate::normalization::spine_manipulator_lines`]:
//! the split and merge do not need to be adjacent, and this pass inspects the
//! intervening barline or tandem interpretation content before deciding whether
//! the whole region can be removed. It does not try to combine unrelated
//! adjacent topology changes such as two independent splits staged on
//! neighboring manipulator lines.

use std::collections::HashSet;

use crate::{
    normalization::NormalizationPass, KernDocument, KernLine, SpineManipulator,
    TandemInterpretation, TandemInterpretationLine,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Remove split/merge regions whose contents project exactly to parent width.
pub struct CollapseTrivialSplitMergeRegions {
    options: NormalizeOptions,
}

impl CollapseTrivialSplitMergeRegions {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for CollapseTrivialSplitMergeRegions {
    fn normalize(&self, document: &mut KernDocument) {
        loop {
            let regions = detect_regions(document);
            let mut rewrites = regions
                .iter()
                .filter_map(|region| classify_region(document, region))
                .collect::<Vec<_>>();

            if rewrites.is_empty() {
                break;
            }

            rewrites.sort_by_key(|rewrite| rewrite.split_line);
            for rewrite in rewrites.into_iter().rev() {
                // Each projected line stays attributed to its own record; a
                // tandem line keeps the left child's field of each collapsed
                // pair. The split and merge records disappear.
                let replacement = rewrite
                    .lines
                    .into_iter()
                    .filter(|(line, _)| !is_null_tandem_line(line))
                    .map(|(line, index)| {
                        let origin = match line {
                            KernLine::TandemInterpretationLine(_) => {
                                document.origin(index).projected(&rewrite.retained)
                            }
                            _ => document.origin(index).clone(),
                        };
                        (line, origin)
                    })
                    .collect::<Vec<_>>();
                document.splice_lines(rewrite.split_line..rewrite.merge_line + 1, replacement);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SpineId(u32);

#[derive(Debug, Clone, PartialEq, Eq)]
struct SplitFrame {
    parent: SpineId,
    children: [SpineId; 2],
    split_line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SplitMergeRegion {
    split_line: usize,
    merge_line: usize,
    parent: SpineId,
    children: [SpineId; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AppliedManipulators {
    live: Vec<SpineId>,
    splits: Vec<AppliedSplit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AppliedSplit {
    input_index: usize,
    children: [SpineId; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RegionRewrite {
    split_line: usize,
    merge_line: usize,
    /// Each projected interior line with its index in the document.
    lines: Vec<(KernLine, usize)>,
    /// For each parent-width column, the interior column it keeps: the left
    /// child of the collapsed split, or the column itself.
    retained: Vec<usize>,
}

fn detect_regions(document: &KernDocument) -> Vec<SplitMergeRegion> {
    let mut live = initial_live_spines(document);
    let mut next_id = live.len() as u32;
    let mut open_splits = Vec::new();
    let mut regions = Vec::new();

    for (line_index, line) in document.lines().iter().enumerate() {
        let KernLine::SpineManipulatorLine(manipulators) = line else {
            continue;
        };
        if manipulators.len() != live.len() {
            continue;
        }

        for (start, end) in merge_groups(manipulators.tokens()) {
            if let Some(frame_index) = open_splits
                .iter()
                .rposition(|frame: &SplitFrame| live[start..end] == frame.children)
            {
                let frame = open_splits.remove(frame_index);
                regions.push(SplitMergeRegion {
                    split_line: frame.split_line,
                    merge_line: line_index,
                    parent: frame.parent,
                    children: frame.children,
                });
            }
        }

        let previous_live = live.clone();
        let Some(applied) =
            apply_manipulators_with_splits(&live, manipulators.tokens(), &mut next_id)
        else {
            continue;
        };
        let next_live = applied.live;

        for split in applied.splits {
            let parent = previous_live[split.input_index];
            open_splits.push(SplitFrame {
                parent,
                children: split.children,
                split_line: line_index,
            });
        }

        let live_set = next_live.iter().copied().collect::<HashSet<_>>();
        open_splits.retain(|frame| frame.children.iter().all(|child| live_set.contains(child)));
        live = next_live;
    }

    regions
}

fn initial_live_spines(document: &KernDocument) -> Vec<SpineId> {
    let count = match document.lines().first() {
        Some(KernLine::SpineTypeLine(spine_types)) => spine_types.len(),
        _ => document.layout.initial_spine_count as usize,
    };
    (0..count).map(|index| SpineId(index as u32)).collect()
}

fn apply_manipulators(
    live: &[SpineId],
    manipulators: &[SpineManipulator],
    next_id: &mut u32,
) -> Option<Vec<SpineId>> {
    Some(apply_manipulators_with_splits(live, manipulators, next_id)?.live)
}

fn apply_manipulators_with_splits(
    live: &[SpineId],
    manipulators: &[SpineManipulator],
    next_id: &mut u32,
) -> Option<AppliedManipulators> {
    let mut next = Vec::new();
    let mut splits = Vec::new();
    let mut index = 0;

    while index < manipulators.len() {
        match manipulators[index] {
            SpineManipulator::Null => {
                next.push(*live.get(index)?);
                index += 1;
            }
            SpineManipulator::Terminate => {
                live.get(index)?;
                index += 1;
            }
            SpineManipulator::Split => {
                live.get(index)?;
                let children = [fresh_spine_id(next_id), fresh_spine_id(next_id)];
                next.extend(children);
                splits.push(AppliedSplit {
                    input_index: index,
                    children,
                });
                index += 1;
            }
            SpineManipulator::Merge => {
                let start = index;
                while index < manipulators.len()
                    && matches!(manipulators[index], SpineManipulator::Merge)
                {
                    live.get(index)?;
                    index += 1;
                }
                if index - start < 2 {
                    return None;
                }
                next.push(fresh_spine_id(next_id));
            }
        }
    }

    Some(AppliedManipulators { live: next, splits })
}

fn fresh_spine_id(next_id: &mut u32) -> SpineId {
    let id = SpineId(*next_id);
    *next_id += 1;
    id
}

fn classify_region(document: &KernDocument, region: &SplitMergeRegion) -> Option<RegionRewrite> {
    let parent_live = live_before_line(document, region.split_line)?;
    let split_position = parent_live
        .iter()
        .position(|spine| *spine == region.parent)?;
    if !is_exact_split_boundary(&document.lines()[region.split_line], split_position) {
        return None;
    }

    let mut expanded_live = parent_live.clone();
    expanded_live.splice(split_position..=split_position, region.children);
    let merge_position = expanded_live
        .windows(region.children.len())
        .position(|window| window == region.children)?;
    if !is_exact_merge_boundary(
        &document.lines()[region.merge_line],
        merge_position,
        region.children.len(),
    ) {
        return None;
    }

    let mut projected_lines = Vec::new();
    for line_index in region.split_line + 1..region.merge_line {
        let line = &document.lines()[line_index];
        if matches!(line, KernLine::SpineManipulatorLine(_)) {
            return None;
        }

        let projected = project_line(line, &expanded_live, &parent_live, region)?;
        projected_lines.push((projected, line_index));
    }

    let mut retained = Vec::with_capacity(parent_live.len());
    let mut expanded_index = 0;
    for spine in &parent_live {
        retained.push(expanded_index);
        expanded_index += if *spine == region.parent { 2 } else { 1 };
    }

    Some(RegionRewrite {
        split_line: region.split_line,
        merge_line: region.merge_line,
        lines: projected_lines,
        retained,
    })
}

fn is_exact_split_boundary(line: &KernLine, split_position: usize) -> bool {
    let KernLine::SpineManipulatorLine(manipulators) = line else {
        return false;
    };

    manipulators
        .tokens()
        .iter()
        .enumerate()
        .all(|(index, manipulator)| {
            if index == split_position {
                matches!(manipulator, SpineManipulator::Split)
            } else {
                matches!(manipulator, SpineManipulator::Null)
            }
        })
}

fn is_exact_merge_boundary(line: &KernLine, merge_position: usize, merge_len: usize) -> bool {
    let KernLine::SpineManipulatorLine(manipulators) = line else {
        return false;
    };
    let merge_range = merge_position..merge_position + merge_len;

    manipulators
        .tokens()
        .iter()
        .enumerate()
        .all(|(index, manipulator)| {
            if merge_range.contains(&index) {
                matches!(manipulator, SpineManipulator::Merge)
            } else {
                matches!(manipulator, SpineManipulator::Null)
            }
        })
}

fn live_before_line(document: &KernDocument, target_line: usize) -> Option<Vec<SpineId>> {
    let mut live = initial_live_spines(document);
    let mut next_id = live.len() as u32;

    for line in document.lines().iter().take(target_line) {
        let KernLine::SpineManipulatorLine(manipulators) = line else {
            continue;
        };
        if manipulators.len() != live.len() {
            return None;
        }
        live = apply_manipulators(&live, manipulators.tokens(), &mut next_id)?;
    }

    Some(live)
}

fn project_line(
    line: &KernLine,
    expanded_live: &[SpineId],
    parent_live: &[SpineId],
    region: &SplitMergeRegion,
) -> Option<KernLine> {
    if line_width(line).is_some_and(|width| width != expanded_live.len()) {
        return None;
    }

    match line {
        KernLine::TandemInterpretationLine(tandems) => {
            project_tandems(tandems, parent_live, region)
        }
        KernLine::Barline(barline) if barline.is_topology_invariant() => Some(line.clone()),
        KernLine::Barline(_) => None,
        KernLine::SpineTypeLine(_)
        | KernLine::SpineManipulatorLine(_)
        | KernLine::LocalCommentLine(_)
        | KernLine::Data(_) => None,
    }
}

fn project_tandems(
    tandems: &TandemInterpretationLine,
    parent_live: &[SpineId],
    region: &SplitMergeRegion,
) -> Option<KernLine> {
    let mut tokens = Vec::with_capacity(parent_live.len());
    let mut expanded_index = 0;

    for spine in parent_live {
        if *spine == region.parent {
            let left = tandems.tokens().get(expanded_index)?;
            let right = tandems.tokens().get(expanded_index + 1)?;
            if left != right {
                return None;
            }
            tokens.push(left.clone());
            expanded_index += 2;
        } else {
            tokens.push(tandems.tokens().get(expanded_index)?.clone());
            expanded_index += 1;
        }
    }

    if expanded_index != tandems.len() {
        return None;
    }

    Some(KernLine::TandemInterpretationLine(
        TandemInterpretationLine::new(tokens).ok()?,
    ))
}

fn line_width(line: &KernLine) -> Option<usize> {
    match line {
        KernLine::SpineTypeLine(spine_types) => Some(spine_types.len()),
        KernLine::TandemInterpretationLine(tandems) => Some(tandems.len()),
        KernLine::SpineManipulatorLine(manipulators) => Some(manipulators.len()),
        KernLine::LocalCommentLine(comments) => Some(comments.len()),
        KernLine::Barline(barline) => barline.spine_count(),
        KernLine::Data(contents) => Some(contents.len()),
    }
}

fn merge_groups(manipulators: &[SpineManipulator]) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let mut index = 0;

    while index < manipulators.len() {
        if !matches!(manipulators[index], SpineManipulator::Merge) {
            index += 1;
            continue;
        }

        let start = index;
        while index < manipulators.len() && matches!(manipulators[index], SpineManipulator::Merge) {
            index += 1;
        }
        if index - start >= 2 {
            groups.push((start, index));
        }
    }

    groups
}

fn is_null_tandem_line(line: &KernLine) -> bool {
    matches!(
        line,
        KernLine::TandemInterpretationLine(tandems)
            if !tandems.is_empty()
                && tandems
                    .tokens()
                    .iter()
                    .all(|tandem| matches!(tandem, TandemInterpretation::Null))
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn collapsed_region_lines_keep_their_own_records() {
        let mut document = KernDocument::parse(
            "**kern\t**kern\n*clefF4\t*clefG2\n*^\t*\n*k[]\t*k[]\t*\n*v\t*v\t*\n4c\t4e\n",
        )
        .unwrap();
        CollapseTrivialSplitMergeRegions::new(Default::default()).normalize(&mut document);
        let sources: Vec<_> = (0..document.lines().len())
            .map(|index| document.source(index))
            .collect();
        // The split (record 3) and merge (record 5) are gone; the key
        // signature line stays attributed to record 4. The parser's default
        // key signature line has no record.
        assert_eq!(sources, [Some(1), Some(2), None, Some(4), Some(6)]);
    }

    use super::CollapseTrivialSplitMergeRegions;
    use crate::{
        normalization::NormalizationPass, types::BarlineStyle, types::RepeatStyle, ClefSign,
        KernContent, KernDocument, KernLine, SpineContent, SpineManipulator, TandemInterpretation,
    };

    fn pass_string(input: &str) -> String {
        let mut document = KernDocument::parse(input).unwrap();
        CollapseTrivialSplitMergeRegions::new(Default::default()).normalize(&mut document);
        render_lines(&document)
    }

    fn render_lines(document: &KernDocument) -> String {
        document
            .lines()
            .iter()
            .enumerate()
            .filter(|(index, _)| document.source_line_number(*index) != 0)
            .map(|(_, line)| line)
            .filter_map(render_line)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    fn render_line(line: &KernLine) -> Option<String> {
        match line {
            KernLine::SpineTypeLine(_) => None,
            KernLine::TandemInterpretationLine(tandems) => Some(
                tandems
                    .tokens()
                    .iter()
                    .map(tandem_token)
                    .collect::<Vec<_>>()
                    .join("\t"),
            ),
            KernLine::SpineManipulatorLine(manipulators) => Some(
                manipulators
                    .tokens()
                    .iter()
                    .map(manipulator_token)
                    .collect::<Vec<_>>()
                    .join("\t"),
            ),
            KernLine::LocalCommentLine(comments) => {
                Some(comments.iter().map(|_| "!").collect::<Vec<_>>().join("\t"))
            }
            KernLine::Barline(barline) => Some(barline_token(barline).to_owned()),
            KernLine::Data(contents) => Some(
                contents
                    .iter()
                    .map(|content| match content {
                        SpineContent::Kern(KernContent::Null) => ".".to_owned(),
                        SpineContent::Kern(KernContent::Note(_))
                        | SpineContent::Kern(KernContent::Chord(_))
                        | SpineContent::Kern(KernContent::Rest(_))
                        | SpineContent::Kern(KernContent::InvisibleRestSlice(_))
                        | SpineContent::Kern(KernContent::MultiMeasureRest(_)) => "data".to_owned(),
                        SpineContent::Kern(KernContent::Grace(_))
                        | SpineContent::Dynam(_)
                        | SpineContent::Fing(_)
                        | SpineContent::Text(_)
                        | SpineContent::MxmlHarmony(_)
                        | SpineContent::Opaque { .. } => "?".to_owned(),
                    })
                    .collect::<Vec<_>>()
                    .join("\t"),
            ),
        }
    }

    fn tandem_token(tandem: &TandemInterpretation) -> String {
        match tandem {
            TandemInterpretation::Null => "*".to_owned(),
            TandemInterpretation::Clef(clef) => format!(
                "*clef{}{}{}",
                match clef.sign {
                    ClefSign::G => "G",
                    ClefSign::F => "F",
                    ClefSign::C => "C",
                    ClefSign::Tab => "X",
                    ClefSign::Percussion => "perc",
                },
                match clef.octave_shift {
                    -2 => "vv",
                    -1 => "v",
                    0 => "",
                    1 => "^",
                    2 => "^^",
                    _ => "?",
                },
                clef.line
            ),
            _ => "?".to_owned(),
        }
    }

    fn barline_token(barline: &crate::Barline) -> &'static str {
        if barline.styles().contains(&BarlineStyle::Double) {
            "=="
        } else if barline.repeat() == RepeatStyle::Start {
            "=|:"
        } else if barline.repeat() == RepeatStyle::End {
            "=:|"
        } else {
            "="
        }
    }

    fn manipulator_token(manipulator: &SpineManipulator) -> &'static str {
        match manipulator {
            SpineManipulator::Null => "*",
            SpineManipulator::Terminate => "*-",
            SpineManipulator::Split => "*^",
            SpineManipulator::Merge => "*v",
        }
    }

    #[test]
    fn removes_empty_canceling_region() {
        let input = "**kern\t**kern\n*^\t*\n*v\t*v\t*\n4c\t4e\n";

        assert_eq!(pass_string(input), "data\tdata\n");
    }

    #[test]
    fn collapses_compatible_barlines() {
        let input = "**kern\t**kern\n*^\t*\n=\t=\t=\n*v\t*v\t*\n4c\t4e\n";

        assert_eq!(pass_string(input), "=\ndata\tdata\n");
    }

    #[test]
    fn collapses_identical_tandems() {
        let input = "**kern\t**kern\n*^\t*\n*clefG2\t*clefG2\t*\n*v\t*v\t*\n4c\t4e\n";

        assert_eq!(pass_string(input), "*clefG2\t*\ndata\tdata\n");
    }

    #[test]
    fn drops_null_only_tandems_inside_region() {
        let input = "**kern\t**kern\n*^\t*\n*\t*\t*\n*v\t*v\t*\n4c\t4e\n";

        assert_eq!(pass_string(input), "data\tdata\n");
    }

    #[test]
    fn preserves_split_merge_region_containing_mixed_barlines() {
        let input = "**kern\t**kern\n*^\t*\n=\t==\t=\n*v\t*v\t*\n4c\t4e\n";
        let mut document = KernDocument::parse(input).unwrap();
        CollapseTrivialSplitMergeRegions::new(Default::default()).normalize(&mut document);

        assert!(document
            .lines()
            .iter()
            .any(|line| matches!(line, KernLine::Barline(barline) if barline.is_spine_specific())));
        assert_eq!(
            document
                .lines()
                .iter()
                .filter(|line| matches!(line, KernLine::SpineManipulatorLine(_)))
                .count(),
            2
        );
    }

    #[test]
    fn preserves_differing_tandems() {
        let input = "**kern\t**kern\n*^\t*\n*clefG2\t*clefF4\t*\n*v\t*v\t*\n4c\t4e\n";

        assert_eq!(
            pass_string(input),
            "*^\t*\n*clefG2\t*clefF4\t*\n*v\t*v\t*\ndata\tdata\n"
        );
    }

    #[test]
    fn preserves_null_vs_non_null_child_tandems() {
        let input = "**kern\t**kern\n*^\t*\n*clefG2\t*\t*\n*v\t*v\t*\n4c\t4e\n";

        assert_eq!(
            pass_string(input),
            "*^\t*\n*clefG2\t*\t*\n*v\t*v\t*\ndata\tdata\n"
        );
    }

    #[test]
    fn preserves_data_lines() {
        let input = "**kern\t**kern\n*^\t*\n4c\t4c\t4e\n*v\t*v\t*\n4d\t4f\n";

        assert_eq!(
            pass_string(input),
            "*^\t*\ndata\tdata\tdata\n*v\t*v\t*\ndata\tdata\n"
        );
    }

    #[test]
    fn preserves_region_with_interior_termination() {
        let input = "**kern\t**kern\n*^\t*\n*-\t*\t*\n*v\t*v\n";

        assert_eq!(pass_string(input), "*^\t*\n*-\t*\t*\n*v\t*v\n");
    }

    #[test]
    fn preserves_region_with_unrelated_interior_topology() {
        let input = "**kern\t**kern\n*^\t*\n*\t*\t*^\n*v\t*v\t*\t*\n";

        assert_eq!(pass_string(input), "*^\t*\n*\t*\t*^\n*v\t*v\t*\t*\n");
    }

    #[test]
    fn preserves_boundary_lines_with_other_topology() {
        let input = "**kern\t**kern\n*^\t*^\n=\t=\t=\t=\n*v\t*v\t*\t*\n";

        assert_eq!(pass_string(input), "*^\t*^\n=\n*v\t*v\t*\t*\n");
    }

    #[test]
    fn collapses_nested_regions_bottom_up() {
        let input = [
            "**kern",
            "*^",
            "*\t*^",
            "*clefG2\t*clefG2\t*clefG2",
            "*\t*v\t*v",
            "*v\t*v",
            "4c",
        ]
        .join("\n");

        assert_eq!(pass_string(&input), "*clefG2\ndata\n");
    }

    #[test]
    fn is_idempotent() {
        let input = "**kern\t**kern\n*^\t*\n=\t=\t=\n*v\t*v\t*\n4c\t4e\n";
        let mut document = KernDocument::parse(input).unwrap();
        CollapseTrivialSplitMergeRegions::new(Default::default()).normalize(&mut document);
        let once = document.clone();
        CollapseTrivialSplitMergeRegions::new(Default::default()).normalize(&mut document);

        assert_eq!(document, once);
    }
}
