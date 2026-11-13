//! Normalization for barlines adjacent to spine-path operations.
//!
//! These passes move compatible barlines across adjacent split-only or
//! merge-only spine manipulator lines. The rewrites keep barlines on the
//! stable side of the topology change when projected tokens are unambiguous.

use crate::{
    normalization::NormalizationPass, KernDocument, KernLine, SpineManipulator,
    SpineManipulatorLine,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Move merge-only spine manipulator lines before compatible preceding barlines.
pub struct MoveMergesBeforeBarlines {
    options: NormalizeOptions,
}

impl MoveMergesBeforeBarlines {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for MoveMergesBeforeBarlines {
    fn normalize(&self, document: &mut KernDocument) {
        let mut index = 0;
        while index + 1 < document.lines().len() {
            let Some(projected) = projected_barline_after_merge(
                &document.lines()[index],
                &document.lines()[index + 1],
            ) else {
                index += 1;
                continue;
            };

            document.swap_lines(index, index + 1);
            document.lines_mut()[index + 1] = KernLine::Barline(projected);
            index += 2;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Move split-only spine manipulator lines after compatible following barlines.
pub struct MoveSplitsAfterBarlines {
    options: NormalizeOptions,
}

impl MoveSplitsAfterBarlines {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for MoveSplitsAfterBarlines {
    fn normalize(&self, document: &mut KernDocument) {
        let mut index = 0;
        while index + 1 < document.lines().len() {
            let Some(projected) = projected_barline_before_split(
                &document.lines()[index],
                &document.lines()[index + 1],
            ) else {
                index += 1;
                continue;
            };

            document.swap_lines(index, index + 1);
            document.lines_mut()[index] = KernLine::Barline(projected);
            index += 2;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Collapse exact inverse merge/barline/split windows to the barline.
pub struct CollapseRedundantMergeSplitAroundBarlines {
    options: NormalizeOptions,
}

impl CollapseRedundantMergeSplitAroundBarlines {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for CollapseRedundantMergeSplitAroundBarlines {
    fn normalize(&self, document: &mut KernDocument) {
        let mut index = 0;
        while index + 2 < document.lines().len() {
            let Some(barline) = redundant_merge_barline_split(
                &document.lines()[index],
                &document.lines()[index + 1],
                &document.lines()[index + 2],
            ) else {
                index += 1;
                continue;
            };

            let source = document.source(index + 1);
            document.splice_lines(index..index + 3, vec![(KernLine::Barline(barline), source)]);
            index = index.saturating_sub(1);
        }
    }
}

fn projected_barline_after_merge(left: &KernLine, right: &KernLine) -> Option<crate::Barline> {
    let (KernLine::Barline(barline), KernLine::SpineManipulatorLine(manipulators)) = (left, right)
    else {
        return None;
    };
    (barline.is_topology_invariant() && is_merge_only_line(manipulators)).then(|| barline.clone())
}

fn projected_barline_before_split(left: &KernLine, right: &KernLine) -> Option<crate::Barline> {
    let (KernLine::SpineManipulatorLine(manipulators), KernLine::Barline(barline)) = (left, right)
    else {
        return None;
    };
    (barline.is_topology_invariant() && is_split_only_line(manipulators)).then(|| barline.clone())
}

fn redundant_merge_barline_split(
    left: &KernLine,
    middle: &KernLine,
    right: &KernLine,
) -> Option<crate::Barline> {
    let (
        KernLine::SpineManipulatorLine(merge_line),
        KernLine::Barline(barline),
        KernLine::SpineManipulatorLine(split_line),
    ) = (left, middle, right)
    else {
        return None;
    };

    (barline.is_topology_invariant() && exact_inverse_merge_split(merge_line, split_line))
        .then(|| barline.clone())
}

fn exact_inverse_merge_split(
    merge_line: &SpineManipulatorLine,
    split_line: &SpineManipulatorLine,
) -> bool {
    let mut merge_index = 0;
    let mut split_index = 0;
    let mut saw_merge = false;

    while merge_index < merge_line.tokens().len() {
        match merge_line.tokens()[merge_index] {
            SpineManipulator::Null => {
                if split_line.tokens().get(split_index) != Some(&SpineManipulator::Null) {
                    return false;
                }
                merge_index += 1;
                split_index += 1;
            }
            SpineManipulator::Merge => {
                if merge_line.tokens().get(merge_index + 1) != Some(&SpineManipulator::Merge)
                    || merge_line.tokens().get(merge_index + 2) == Some(&SpineManipulator::Merge)
                    || split_line.tokens().get(split_index) != Some(&SpineManipulator::Split)
                {
                    return false;
                }
                saw_merge = true;
                merge_index += 2;
                split_index += 1;
            }
            SpineManipulator::Split | SpineManipulator::Terminate => return false,
        }
    }

    saw_merge && split_index == split_line.tokens().len()
}

fn is_merge_only_line(manipulators: &SpineManipulatorLine) -> bool {
    manipulators
        .tokens()
        .iter()
        .any(|manipulator| matches!(manipulator, SpineManipulator::Merge))
        && manipulators.tokens().iter().all(|manipulator| {
            matches!(
                manipulator,
                SpineManipulator::Null | SpineManipulator::Merge
            )
        })
}

fn is_split_only_line(manipulators: &SpineManipulatorLine) -> bool {
    manipulators
        .tokens()
        .iter()
        .any(|manipulator| matches!(manipulator, SpineManipulator::Split))
        && manipulators.tokens().iter().all(|manipulator| {
            matches!(
                manipulator,
                SpineManipulator::Null | SpineManipulator::Split
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        types::{BarlineStyle, RepeatStyle},
        DocumentLayout, KernDocument, KernMetadata,
    };

    fn split_pass_string(input: &str) -> String {
        let mut document = KernDocument::parse(input).unwrap();
        MoveSplitsAfterBarlines::new(Default::default()).normalize(&mut document);
        rendered_document_string(&document)
    }

    fn merge_pass_string(input: &str) -> String {
        let mut document = KernDocument::parse(input).unwrap();
        MoveMergesBeforeBarlines::new(Default::default()).normalize(&mut document);
        rendered_document_string(&document)
    }

    fn collapse_pass_string(input: &str) -> String {
        let mut document = KernDocument::parse(input).unwrap();
        CollapseRedundantMergeSplitAroundBarlines::new(Default::default()).normalize(&mut document);
        rendered_document_string(&document)
    }

    fn rendered_document_string(document: &KernDocument) -> String {
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
            KernLine::SpineTypeLine(_) | KernLine::LocalCommentLine(_) | KernLine::Data(_) => None,
            KernLine::TandemInterpretationLine(tandems) => Some(
                tandems
                    .tokens()
                    .iter()
                    .map(|_| "*")
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
            KernLine::Barline(barline) => Some(barline_token(barline).to_owned()),
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

    #[test]
    fn moves_split_before_compatible_barline_after_barline() {
        let input = "**kern\t**kern\n*^\t*\n=\t=\t=\n4c\t4e\t4g\n";

        assert_eq!(split_pass_string(input), "=\n*^\t*\n");
    }

    #[test]
    fn preserves_split_before_mixed_child_barlines() {
        let input = "**kern\t**kern\n*^\t*\n=\t==\t=\n4c\t4e\t4g\n";
        let mut document = KernDocument::parse(input).unwrap();
        MoveSplitsAfterBarlines::new(Default::default()).normalize(&mut document);
        let split_index = (0..document.lines().len())
            .find(|index| document.source_line_number(*index) == 2)
            .unwrap();
        let barline_index = (0..document.lines().len())
            .find(|index| document.source_line_number(*index) == 3)
            .unwrap();

        assert!(split_index < barline_index);
        assert!(matches!(
            &document.lines()[barline_index],
            KernLine::Barline(barline) if barline.is_spine_specific()
        ));
    }

    #[test]
    fn preserves_merge_line_before_barline() {
        let input = "**kern\t**kern\t**kern\n*v\t*v\t*\n=\t=\n4c\t4e\n";

        assert_eq!(split_pass_string(input), "*v\t*v\t*\n=\n");
    }

    #[test]
    fn preserves_terminate_line_before_barline() {
        let input = "**kern\t**kern\n*-\t*-\n=\n";

        assert_eq!(split_pass_string(input), "*-\t*-\n=\n");
    }

    #[test]
    fn moves_compatible_barline_after_merge() {
        let input = "**kern\t**kern\t**kern\n=\t=\t=\n*\t*v\t*v\n4c\t4e\n";

        assert_eq!(merge_pass_string(input), "*\t*v\t*v\n=\n");
    }

    #[test]
    fn moves_compatible_barline_after_left_merge() {
        let input = "**kern\t**kern\t**kern\n=\t=\t=\n*v\t*v\t*\n4c\t4e\n";

        assert_eq!(merge_pass_string(input), "*v\t*v\t*\n=\n");
    }

    #[test]
    fn preserves_mixed_barlines_before_merge() {
        let input = "**kern\t**kern\t**kern\n=\t==\t=\n*\t*v\t*v\n4c\t4e\n";
        let mut document = KernDocument::parse(input).unwrap();
        MoveMergesBeforeBarlines::new(Default::default()).normalize(&mut document);
        let barline_index = (0..document.lines().len())
            .find(|index| document.source_line_number(*index) == 2)
            .unwrap();
        let merge_index = (0..document.lines().len())
            .find(|index| document.source_line_number(*index) == 3)
            .unwrap();

        assert!(barline_index < merge_index);
        assert!(matches!(
            &document.lines()[barline_index],
            KernLine::Barline(barline) if barline.is_spine_specific()
        ));
    }

    #[test]
    fn preserves_split_line_after_barline() {
        let input = "**kern\t**kern\n=\t=\n*^\t*\n4c\t4e\t4g\n";

        assert_eq!(merge_pass_string(input), "=\n*^\t*\n");
    }

    #[test]
    fn preserves_terminate_line_after_barline() {
        let input = "**kern\t**kern\n=\t=\n*-\t*-\n";

        assert_eq!(merge_pass_string(input), "=\n*-\t*-\n");
    }

    #[test]
    fn preserves_malformed_width_relationship() {
        let mut document = KernDocument::from_parts(
            KernMetadata::default(),
            DocumentLayout {
                initial_spine_count: 2,
            },
            vec![
                KernLine::SpineManipulatorLine(
                    SpineManipulatorLine::new(vec![
                        SpineManipulator::Split,
                        SpineManipulator::Null,
                    ])
                    .unwrap(),
                ),
                KernLine::Barline(basic_barline()),
            ]
            .into_iter()
            .enumerate()
            .map(|(index, line)| (line, Some(index + 1)))
            .collect(),
        );

        MoveSplitsAfterBarlines::new(Default::default()).normalize(&mut document);

        assert_eq!(rendered_document_string(&document), "=\n*^\t*\n");
    }

    #[test]
    fn collapses_redundant_right_merge_barline_split() {
        let input = "**kern\t**kern\t**kern\n*\t*v\t*v\n=\t=\n*\t*^\n4c\t4e\t4g\n";

        assert_eq!(collapse_pass_string(input), "=\n");
    }

    #[test]
    fn collapsed_barline_preserves_original_source_line() {
        let input = "**kern\t**kern\t**kern\n*\t*v\t*v\n=\t=\n*\t*^\n4c\t4e\t4g\n";
        let mut document = KernDocument::parse(input).unwrap();

        CollapseRedundantMergeSplitAroundBarlines::new(Default::default()).normalize(&mut document);

        let barline_index = document
            .lines()
            .iter()
            .position(|line| matches!(line, KernLine::Barline(_)))
            .expect("expected collapsed barline");
        assert_eq!(document.source(barline_index), Some(3));
    }

    #[test]
    fn collapses_redundant_left_merge_barline_split() {
        let input = "**kern\t**kern\t**kern\n*v\t*v\t*\n=\t=\n*^\t*\n4c\t4e\t4g\n";

        assert_eq!(collapse_pass_string(input), "=\n");
    }

    #[test]
    fn collapses_separated_independent_merge_barline_splits() {
        let input = concat!(
            "**kern\t**kern\t**kern\t**kern\t**kern\n",
            "*v\t*v\t*\t*v\t*v\n",
            "=\t=\t=\n",
            "*^\t*\t*^\n",
            "4c\t4d\t4e\t4f\t4g\n",
        );

        assert_eq!(collapse_pass_string(input), "=\n");
    }

    #[test]
    fn collapses_nested_merge_barline_split_wrappers() {
        let input = concat!(
            "**kern\t**kern\t**kern\t**kern\n",
            "*\t*\t*v\t*v\n",
            "*\t*v\t*v\n",
            "=\t=\n",
            "*\t*^\n",
            "*\t*\t*^\n",
            "4a\t4b\t4c\t4d\n",
        );

        assert_eq!(collapse_pass_string(input), "=\n");
    }

    #[test]
    fn preserves_three_spine_merge_barline_split() {
        let input = "**kern\t**kern\t**kern\n*v\t*v\t*v\n=\n*^\n4c\t4e\n";

        assert_eq!(collapse_pass_string(input), "*v\t*v\t*v\n=\n*^\n");
    }

    #[test]
    fn preserves_mismatched_merge_barline_split_positions() {
        let input = "**kern\t**kern\t**kern\n*\t*v\t*v\n=\t=\n*^\t*\n4c\t4e\t4g\n";

        assert_eq!(collapse_pass_string(input), "*\t*v\t*v\n=\n*^\t*\n");
    }

    #[test]
    fn preserves_terminating_or_mixed_manipulator_windows() {
        let terminating = "**kern\t**kern\t**kern\n*v\t*v\t*\n=\n*-\t*-\n";
        let mixed = "**kern\t**kern\t**kern\n*v\t*v\t*^\n=\n*^\t*\t*\n";

        assert_eq!(collapse_pass_string(terminating), "*v\t*v\t*\n=\n*-\t*-\n");
        assert_eq!(collapse_pass_string(mixed), "*v\t*v\t*^\n=\n*^\t*\t*\n");
    }

    fn basic_barline() -> crate::Barline {
        crate::Barline::new(vec![BarlineStyle::Basic], RepeatStyle::None)
    }
}
