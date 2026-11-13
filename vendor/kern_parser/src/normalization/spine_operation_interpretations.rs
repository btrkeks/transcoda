//! Move safe tandem interpretations across split and merge operations.
//!
//! A tandem line can move across an adjacent pure spine operation only when the
//! fields collapsed by that move are exactly identical. The pass is local and
//! intentionally does not validate final spine topology.
//!
//! Example:
//! ```text
//! Before:
//! *^
//! *clefG2\t*clefG2
//!
//! After:
//! *clefG2
//! *^
//! ```
//!
//! The two post-split clefs are identical, so the clef line can be represented
//! once on the parent spine before the split.

use crate::{
    analysis::leading_header_bounds, normalization::NormalizationPass, KernDocument, KernLine,
    SpineManipulator, TandemInterpretationLine,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Move collapsible tandem interpretation lines to the side of a spine
/// operation where they can be represented without duplicated fields.
pub struct MoveInterpretationsAcrossSpineOperations {
    options: NormalizeOptions,
}

impl MoveInterpretationsAcrossSpineOperations {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for MoveInterpretationsAcrossSpineOperations {
    fn normalize(&self, document: &mut KernDocument) {
        let leading_header = leading_header_bounds(document);
        let mut index = 0;
        while index < document.lines().len() {
            if is_pure_split_line(&document.lines()[index]) {
                while index + 1 < document.lines().len() {
                    if is_in_range(index + 1, leading_header) {
                        break;
                    }
                    let Some(collapsed) = collapse_line_after_split(
                        &document.lines()[index],
                        &document.lines()[index + 1],
                    ) else {
                        break;
                    };

                    document.lines_mut()[index + 1] = KernLine::TandemInterpretationLine(collapsed);
                    document.swap_lines(index, index + 1);
                    index += 1;
                }
            } else if is_pure_merge_line(&document.lines()[index]) {
                while index > 0 {
                    if is_in_range(index - 1, leading_header) {
                        break;
                    }
                    let Some(collapsed) = collapse_line_before_merge(
                        &document.lines()[index - 1],
                        &document.lines()[index],
                    ) else {
                        break;
                    };

                    document.lines_mut()[index - 1] = KernLine::TandemInterpretationLine(collapsed);
                    document.swap_lines(index - 1, index);
                    index -= 1;
                }
            }

            index += 1;
        }
    }
}

fn is_in_range(index: usize, range: (usize, usize)) -> bool {
    (range.0..range.1).contains(&index)
}

fn is_pure_split_line(line: &KernLine) -> bool {
    let KernLine::SpineManipulatorLine(manipulator_line) = line else {
        return false;
    };
    let manipulators = manipulator_line.tokens();

    manipulators
        .iter()
        .any(|manipulator| matches!(manipulator, SpineManipulator::Split))
        && manipulators.iter().all(|manipulator| {
            matches!(
                manipulator,
                SpineManipulator::Null | SpineManipulator::Split
            )
        })
}

fn is_pure_merge_line(line: &KernLine) -> bool {
    let KernLine::SpineManipulatorLine(manipulator_line) = line else {
        return false;
    };
    let manipulators = manipulator_line.tokens();

    manipulators.iter().all(|manipulator| {
        matches!(
            manipulator,
            SpineManipulator::Null | SpineManipulator::Merge
        )
    }) && merge_groups(manipulators)
        .into_iter()
        .any(|(_, end)| end > 0)
}

fn collapse_line_after_split(
    split_line: &KernLine,
    tandem_line: &KernLine,
) -> Option<TandemInterpretationLine> {
    let KernLine::SpineManipulatorLine(manipulator_line) = split_line else {
        return None;
    };
    let manipulators = manipulator_line.tokens();
    let KernLine::TandemInterpretationLine(tandems) = tandem_line else {
        return None;
    };
    if !is_movable_tandem_line(tandems) {
        return None;
    }

    let mut collapsed = Vec::with_capacity(manipulators.len());
    let mut post_index = 0;
    for manipulator in manipulators {
        match manipulator {
            SpineManipulator::Null => {
                collapsed.push(tandems.tokens().get(post_index)?.clone());
                post_index += 1;
            }
            SpineManipulator::Split => {
                let left = tandems.tokens().get(post_index)?;
                let right = tandems.tokens().get(post_index + 1)?;
                if left != right {
                    return None;
                }
                collapsed.push(left.clone());
                post_index += 2;
            }
            SpineManipulator::Terminate | SpineManipulator::Merge => return None,
        }
    }

    (post_index == tandems.len())
        .then(|| TandemInterpretationLine::from_homogeneous_tokens(collapsed))
}

fn collapse_line_before_merge(
    tandem_line: &KernLine,
    merge_line: &KernLine,
) -> Option<TandemInterpretationLine> {
    let KernLine::TandemInterpretationLine(tandems) = tandem_line else {
        return None;
    };
    let KernLine::SpineManipulatorLine(manipulator_line) = merge_line else {
        return None;
    };
    let manipulators = manipulator_line.tokens();
    if !is_movable_tandem_line(tandems) || tandems.len() != manipulators.len() {
        return None;
    }

    let mut collapsed = Vec::new();
    let mut index = 0;
    while index < manipulators.len() {
        match manipulators[index] {
            SpineManipulator::Null => {
                collapsed.push(tandems.tokens()[index].clone());
                index += 1;
            }
            SpineManipulator::Merge => {
                let start = index;
                while index < manipulators.len()
                    && matches!(manipulators[index], SpineManipulator::Merge)
                {
                    index += 1;
                }
                if index - start < 2 {
                    return None;
                }

                let first = &tandems.tokens()[start];
                if tandems.tokens()[start + 1..index]
                    .iter()
                    .any(|tandem| tandem != first)
                {
                    return None;
                }
                collapsed.push(first.clone());
            }
            SpineManipulator::Split | SpineManipulator::Terminate => return None,
        }
    }

    Some(TandemInterpretationLine::from_homogeneous_tokens(collapsed))
}

fn is_movable_tandem_line(tandems: &TandemInterpretationLine) -> bool {
    !tandems.is_empty()
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

#[cfg(test)]
mod tests {
    use super::MoveInterpretationsAcrossSpineOperations;
    use crate::{
        normalization::NormalizationPass, ClefSign, DocumentLayout, KernDocument, KernLine,
        KernMetadata, SpineManipulator, TandemInterpretation,
    };

    fn normalized_omr_string(input: &str) -> String {
        crate::normalization::omr::normalized_for_omr(&KernDocument::parse(input).unwrap())
            .to_omr_string()
    }

    fn pass_output(input: &str) -> Vec<String> {
        let mut document = KernDocument::parse(input).unwrap();
        MoveInterpretationsAcrossSpineOperations::new(Default::default()).normalize(&mut document);
        relevant_lines(&document)
    }

    fn relevant_lines(document: &KernDocument) -> Vec<String> {
        document
            .lines()
            .iter()
            .enumerate()
            .filter(|(index, _)| document.source_line_number(*index) != 0)
            .map(|(_, line)| line)
            .filter_map(|line| match line {
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
                KernLine::SpineTypeLine(_)
                | KernLine::LocalCommentLine(_)
                | KernLine::Barline(_)
                | KernLine::Data(_) => None,
            })
            .collect()
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
            TandemInterpretation::KeySignature(key_signature) => key_signature.as_kern(),
            TandemInterpretation::TimeSignature(time_signature) => time_signature.as_kern(),
            TandemInterpretation::FreeMeter => "*MX".to_owned(),
            TandemInterpretation::MeterSymbol(symbol) => symbol.as_kern().to_owned(),
            TandemInterpretation::Raw(raw) => raw.0.clone(),
            other => format!("{other:?}"),
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

    fn lines(expected: &[&str]) -> Vec<String> {
        expected.iter().map(|line| (*line).to_owned()).collect()
    }

    #[test]
    fn moves_split_on_right() {
        let input = "**kern\t**kern\n*\t*^\n*clefF4\t*clefG2\t*clefG2\n4C\t4c\t4e\n";

        assert_eq!(pass_output(input), lines(&["*clefF4\t*clefG2", "*\t*^"]));
    }

    #[test]
    fn moves_split_on_left() {
        let input = "**kern\t**kern\n*^\t*\n*clefF4\t*clefF4\t*clefG2\n4C\t4E\t4c\n";

        assert_eq!(pass_output(input), lines(&["*clefF4\t*clefG2", "*^\t*"]));
    }

    #[test]
    fn moves_split_in_middle() {
        let input = [
            "**kern\t**kern\t**kern",
            "*\t*^\t*",
            "*clefF4\t*clefG2\t*clefG2\t*clefG2",
            "4C\t4c\t4e\t4g",
        ]
        .join("\n");

        assert_eq!(
            pass_output(&input),
            lines(&["*clefF4\t*clefG2\t*clefG2", "*\t*^\t*"])
        );
    }

    #[test]
    fn moves_double_split() {
        let input = [
            "**kern\t**kern",
            "*^\t*^",
            "*clefF4\t*clefF4\t*clefG2\t*clefG2",
            "*M2/2\t*M2/2\t*M2/2\t*M2/2",
            "4C\t4E\t4c\t4e",
        ]
        .join("\n");

        assert_eq!(
            pass_output(&input),
            lines(&["*clefF4\t*clefG2", "*M2/2\t*M2/2", "*^\t*^"])
        );
    }

    #[test]
    fn moves_merge_on_right() {
        let input =
            "**kern\t**kern\t**kern\n4C\t4c\t4e\n*clefF4\t*clefG2\t*clefG2\n*\t*v\t*v\n4C\t4c\n";

        assert_eq!(
            pass_output(input),
            lines(&["*\t*v\t*v", "*clefF4\t*clefG2"])
        );
    }

    #[test]
    fn moves_merge_on_left() {
        let input =
            "**kern\t**kern\t**kern\n4C\t4E\t4c\n*clefF4\t*clefF4\t*clefG2\n*v\t*v\t*\n4C\t4c\n";

        assert_eq!(
            pass_output(input),
            lines(&["*v\t*v\t*", "*clefF4\t*clefG2"])
        );
    }

    #[test]
    fn moves_merge_in_middle() {
        let input = [
            "**kern\t**kern\t**kern\t**kern",
            "4C\t4c\t4e\t4g",
            "*clefF4\t*clefG2\t*clefG2\t*clefG2",
            "*\t*v\t*v\t*",
            "4C\t4c\t4e",
        ]
        .join("\n");

        assert_eq!(
            pass_output(&input),
            lines(&["*\t*v\t*v\t*", "*clefF4\t*clefG2\t*clefG2"])
        );
    }

    #[test]
    fn moves_two_merge_groups() {
        let input = [
            "**kern\t**kern\t**kern\t**kern\t**kern",
            "4C\t4E\t4c\t4e\t4g",
            "*clefF4\t*clefF4\t*clefC4\t*clefG2\t*clefG2",
            "*v\t*v\t*\t*v\t*v",
            "4C\t4E\t4c",
        ]
        .join("\n");

        assert_eq!(
            pass_output(&input),
            lines(&["*v\t*v\t*\t*v\t*v", "*clefF4\t*clefC4\t*clefG2"])
        );
    }

    #[test]
    fn stops_split_at_first_incompatible_line() {
        let input = [
            "**kern\t**kern",
            "*\t*^",
            "*clefF4\t*clefG2\t*clefG2",
            "*k[b-]\t*k[b-]\t*k[b-]",
            "*M3/4\t*M3/4\t*M2/2",
            "4C\t4c\t4e",
        ]
        .join("\n");

        assert_eq!(
            pass_output(&input),
            lines(&[
                "*clefF4\t*clefG2",
                "*k[b-]\t*k[b-]",
                "*\t*^",
                "*M3/4\t*M3/4\t*M2/2",
            ])
        );
    }

    #[test]
    fn stops_merge_at_first_incompatible_line() {
        let input = [
            "**kern\t**kern\t**kern",
            "*clefF4\t*clefG2\t*clefG2",
            "*k[b-]\t*k[]\t*k[]",
            "*M3/4\t*M3/4\t*M2/2",
            "*\t*v\t*v",
            "4C\t4c",
        ]
        .join("\n");

        assert_eq!(
            pass_output(&input),
            lines(&[
                "*clefF4\t*clefG2\t*clefG2",
                "*k[b-]\t*k[]\t*k[]",
                "*M3/4\t*M3/4\t*M2/2",
                "*\t*v\t*v",
            ])
        );
    }

    #[test]
    fn allows_different_interpretations_on_unchanged_spines() {
        let input = [
            "**kern\t**kern",
            "*\t*^",
            "*clefF4\t*clefG2\t*clefG2",
            "*k[b-]\t*k[]\t*k[]",
            "4C\t4c\t4e",
        ]
        .join("\n");

        assert_eq!(
            pass_output(&input),
            lines(&["*clefF4\t*clefG2", "*k[b-]\t*k[]", "*\t*^"])
        );
    }

    #[test]
    fn moves_sparse_tandem_lines_when_operation_fields_are_collapsible() {
        let input = "**kern\t**kern\t**kern\n4C\t4c\t4e\n*\t*clefF4\t*clefF4\n*\t*v\t*v\n4C\t4c\n";

        assert_eq!(pass_output(input), lines(&["*\t*v\t*v", "*\t*clefF4"]));
    }

    #[test]
    fn keeps_leading_header_before_following_merge() {
        let input = "**kern\t**kern\t**kern\n*clefF4\t*clefG2\t*clefG2\n*k[]\t*k[]\t*k[]\n*\t*v\t*v\n4C\t4c\n";

        assert_eq!(
            pass_output(input),
            lines(&["*clefF4\t*clefG2\t*clefG2", "*k[]\t*k[]\t*k[]", "*\t*v\t*v"])
        );
    }

    #[test]
    fn does_not_move_sparse_tandem_lines_when_operation_fields_conflict() {
        let input = "**kern\t**kern\n*\t*^\n*clefF4\t*\t*clefG2\n4C\t4c\t4e\n";
        let mut document = KernDocument::parse(input).unwrap();

        MoveInterpretationsAcrossSpineOperations::new(Default::default()).normalize(&mut document);

        let operation_index = document
            .lines()
            .iter()
            .enumerate()
            .position(|(index, line)| {
                document.source(index) == Some(2)
                    && matches!(line, KernLine::SpineManipulatorLine(_))
            })
            .expect("expected source-authored operation line");
        let tandem_index = document
            .lines()
            .iter()
            .enumerate()
            .position(|(index, line)| {
                document.source(index) == Some(3)
                    && matches!(line, KernLine::TandemInterpretationLine(_))
            })
            .expect("expected source-authored tandem line");

        assert!(matches!(
            document.lines()[operation_index],
            KernLine::SpineManipulatorLine(ref manipulators)
                if manipulators.tokens() == [SpineManipulator::Null, SpineManipulator::Split]
        ));
        assert!(matches!(
            document.lines()[tandem_index],
            KernLine::TandemInterpretationLine(ref tandems)
                if matches!(tandems.tokens()[1], TandemInterpretation::Null)
        ));
    }

    #[test]
    fn does_not_move_non_pure_operation_lines() {
        let documents = [
            vec![
                KernLine::parse(1, "*-\t*").unwrap(),
                KernLine::parse(2, "*clefF4\t*clefG2").unwrap(),
            ],
            vec![
                KernLine::parse(1, "*\t*-").unwrap(),
                KernLine::parse(2, "*clefF4\t*clefG2").unwrap(),
            ],
        ];

        for lines in documents {
            let mut document = KernDocument::from_parts(
                KernMetadata::default(),
                DocumentLayout {
                    initial_spine_count: 2,
                },
                lines
                    .into_iter()
                    .enumerate()
                    .map(|(index, line)| (line, Some(index + 1)))
                    .collect(),
            );
            let original = document.lines().to_vec();
            MoveInterpretationsAcrossSpineOperations::new(Default::default())
                .normalize(&mut document);
            assert_eq!(document.lines(), original);
        }
    }

    #[test]
    fn omr_serialization_uses_canonical_split_position() {
        let input = [
            "**kern\t**kern",
            "*clefF4\t*clefG2",
            "*k[b-]\t*k[b-]",
            "*M3/4\t*M3/4",
            "4C\t4c",
            "*\t*^",
            "*clefF4\t*clefG2\t*clefG2",
            "*k[b-]\t*k[b-]\t*k[b-]",
            "*M3/4\t*M3/4\t*M3/4",
            "4D\t4d\t4f",
            "*\t*v\t*v",
            "2G\t2g",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefF4\t*clefG2\n*k[b-]\t*k[b-]\n*M3/4\t*M3/4\n4C\t4c\n*clefF4\t*clefG2\n*k[b-]\t*k[b-]\n*M3/4\t*M3/4\n*\t*^\n4D\t4d\t4f\n*\t*v\t*v\n2G\t2g\n*-\t*-\n"
        );
    }
}
