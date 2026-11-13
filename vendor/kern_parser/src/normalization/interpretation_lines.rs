//! Normalization for tandem interpretation lines.
//!
//! Combines adjacent sparse tandem lines when doing so preserves the
//! per-spine order and keeps the line homogeneous. For example,
//! `*clefG2\t*` followed by `*\t*clefF4` can become
//! `*clefG2\t*clefF4`.
//!
//! The pass does not combine different tandem kinds, even when they occur in
//! different spines. `*\t*M4/4\t*` followed by `*\t*\t*clefG2` therefore stays
//! as two lines, because the result would mix meter and clef tokens.
//!
//! Example:
//! ```text
//! Before:
//! *clefG2\t*
//! *\t*clefF4
//!
//! After:
//! *clefG2\t*clefF4
//! ```

use crate::{
    normalization::NormalizationPass, KernDocument, KernLine, TandemInterpretation,
    TandemInterpretationLine, TandemInterpretationLineKind,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Collapse adjacent compatible tandem interpretation lines.
///
/// Two lines are compatible only when they have the same width, the same
/// non-null [`TandemInterpretationLineKind`] (or one line is all null), and no
/// spine has a non-null token in both lines.
pub struct CollapseInterpretationLines {
    options: NormalizeOptions,
}

impl CollapseInterpretationLines {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for CollapseInterpretationLines {
    fn normalize(&self, document: &mut KernDocument) {
        let mut index = 0;
        while index + 1 < document.lines().len() {
            let Some(collapsed) =
                collapse_pair(&document.lines()[index], &document.lines()[index + 1])
            else {
                index += 1;
                continue;
            };

            // The merged line keeps the left record; fields it took from the
            // right line keep the right record as their source.
            let width = collapsed.len();
            let taken_from_right: Vec<_> = (0..width)
                .map(|column| {
                    is_tandem_null_at(&document.lines()[index], column)
                        && !is_tandem_null_at(&document.lines()[index + 1], column)
                })
                .collect();
            let origin = document.origin(index).merged(
                document.origin(index + 1),
                |column| taken_from_right[column],
                width,
            );
            document.lines_mut()[index] = KernLine::TandemInterpretationLine(collapsed);
            document.set_origin(index, origin);
            document.remove_line(index + 1);
        }
    }
}

fn is_tandem_null_at(line: &KernLine, column: usize) -> bool {
    matches!(line, KernLine::TandemInterpretationLine(tandems) if tandems.tokens().get(column).is_none_or(is_null))
}

fn collapse_pair(left: &KernLine, right: &KernLine) -> Option<TandemInterpretationLine> {
    let (KernLine::TandemInterpretationLine(left), KernLine::TandemInterpretationLine(right)) =
        (left, right)
    else {
        return None;
    };

    if left.len() != right.len() {
        return None;
    }
    if !compatible_kinds(left.kind(), right.kind()) {
        return None;
    }

    let mut collapsed = Vec::with_capacity(left.len());
    let mut generated = Vec::with_capacity(left.len());
    for (index, (left_token, right_token)) in left.tokens().iter().zip(right.tokens()).enumerate() {
        match (is_null(left_token), is_null(right_token)) {
            (true, true) => {
                collapsed.push(TandemInterpretation::Null);
                generated.push(false);
            }
            (false, true) => {
                collapsed.push(left_token.clone());
                generated.push(left.is_generated_token(index));
            }
            (true, false) => {
                collapsed.push(right_token.clone());
                generated.push(right.is_generated_token(index));
            }
            (false, false) => return None,
        }
    }

    Some(TandemInterpretationLine::from_homogeneous_tokens_with_generated(collapsed, generated))
}

fn compatible_kinds(
    left: TandemInterpretationLineKind,
    right: TandemInterpretationLineKind,
) -> bool {
    matches!(left, TandemInterpretationLineKind::Null)
        || matches!(right, TandemInterpretationLineKind::Null)
        || left == right
}

fn is_null(tandem: &TandemInterpretation) -> bool {
    matches!(tandem, TandemInterpretation::Null)
}

#[cfg(test)]
mod tests {
    #[test]
    fn merged_line_names_the_record_of_each_field() {
        use crate::ast::{FieldOrigin, RecordId};
        let mut document =
            KernDocument::parse("**kern\t**kern\n*clefG2\t*\n*\t*clefF4\n*k[]\t*k[]\n4c\t4C\n")
                .unwrap();
        CollapseInterpretationLines::new(Default::default()).normalize(&mut document);
        let merged = document.origin(1);
        assert_eq!(merged.owner(), RecordId::new(2));
        assert_eq!(
            merged.field(1),
            FieldOrigin::Source {
                record: RecordId::new(3).unwrap(),
                field: 1
            }
        );
    }

    use crate::{
        normalization::{interpretation_lines::CollapseInterpretationLines, NormalizationPass},
        KernDocument, KernLine, TandemInterpretation,
    };

    fn collapse_interpretation_lines(document: &mut KernDocument) {
        CollapseInterpretationLines::new(Default::default()).normalize(document);
    }

    #[test]
    fn collapses_adjacent_compatible_interpretation_lines() {
        let mut document =
            KernDocument::parse("**kern\t**kern\n*clefG2\t*\n*\t*clefF4\n4c\t4E\n").unwrap();

        collapse_interpretation_lines(&mut document);

        let KernLine::TandemInterpretationLine(line) = &document.lines()[1] else {
            panic!("expected collapsed tandem line");
        };
        let tandems = line.tokens();
        assert!(matches!(tandems[0], TandemInterpretation::Clef(_)));
        assert!(matches!(tandems[1], TandemInterpretation::Clef(_)));
        assert_eq!(document.lines().len(), 4);
    }

    #[test]
    fn does_not_collapse_conflicting_interpretation_lines() {
        let mut document = KernDocument::parse("**kern\n*clefG2\n*k[]\n4c\n").unwrap();

        collapse_interpretation_lines(&mut document);

        assert_eq!(document.lines().len(), 4);
    }

    #[test]
    fn does_not_collapse_different_interpretation_kinds_in_different_spines() {
        let mut document =
            KernDocument::parse("**kern\t**kern\t**kern\n*\t*M4/4\t*\n*\t*\t*clefG2\n4c\t4e\t4g\n")
                .unwrap();

        collapse_interpretation_lines(&mut document);

        assert_eq!(document.lines().len(), 5);
    }
}
