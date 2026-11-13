//! Normalization for null-only lines.
//!
//! Removes AST lines that carry no new information:
//! - tandem interpretation lines where every token is `*`
//! - spine-manipulator lines where every token is `*`
//! - data lines where every typed spine token is a null continuation `.`
//!
//! Barline and spine-type lines are never removed by this pass. Empty line
//! vectors are also preserved because they should not appear in parsed input and
//! are better left visible if constructed manually.
//!
//! Example:
//! ```text
//! Before:
//! **kern\t**kern
//! *\t*
//! .\t.
//! 4c\t4e
//!
//! After:
//! **kern\t**kern
//! 4c\t4e
//! ```

use crate::{
    normalization::NormalizationPass, DynamContent, FingContent, KernContent, KernDocument,
    KernLine, MxmlHarmonyContent, SpineContent, SpineManipulator, TandemInterpretation,
    TextContent,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Remove lines whose every token is the line kind's null/no-op token.
///
/// Opaque data tokens are not treated as null even if their raw text is `.`,
/// because their semantics are unknown to the typed AST.
pub struct RemoveNullLines {
    options: NormalizeOptions,
}

impl RemoveNullLines {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for RemoveNullLines {
    fn normalize(&self, document: &mut KernDocument) {
        document.retain_lines(|line| !is_null_line(line));
    }
}

fn is_null_line(line: &KernLine) -> bool {
    match line {
        KernLine::TandemInterpretationLine(tandems) => {
            !tandems.is_empty()
                && tandems
                    .tokens()
                    .iter()
                    .all(|tandem| matches!(tandem, TandemInterpretation::Null))
        }
        KernLine::SpineManipulatorLine(manipulators) => {
            !manipulators.is_empty()
                && manipulators
                    .tokens()
                    .iter()
                    .all(|manipulator| matches!(manipulator, SpineManipulator::Null))
        }
        KernLine::Data(contents) => {
            !contents.is_empty() && contents.iter().all(is_null_spine_content)
        }
        KernLine::SpineTypeLine(_) | KernLine::LocalCommentLine(_) | KernLine::Barline(_) => false,
    }
}

fn is_null_spine_content(content: &SpineContent) -> bool {
    matches!(
        content,
        SpineContent::Kern(KernContent::Null)
            | SpineContent::Dynam(DynamContent::Null)
            | SpineContent::Fing(FingContent::Null)
            | SpineContent::Text(TextContent::Null)
            | SpineContent::MxmlHarmony(MxmlHarmonyContent::Null)
    )
}

#[cfg(test)]
mod tests {
    use crate::types::RepeatStyle;
    use crate::{
        normalization::{null_lines::RemoveNullLines, NormalizationPass},
        Barline, DocumentLayout, DynamContent, FingContent, KernContent, KernDocument, KernLine,
        KernMetadata, MxmlHarmonyContent, MxmlHarmonyLabel, RawToken, SpineContent,
        SpineManipulator, SpineType, TandemInterpretation, TextContent, TextUnderlay,
    };

    #[test]
    fn removes_null_only_lines() {
        let mut document = test_document(vec![
            KernLine::SpineTypeLine(vec![
                SpineType::Kern,
                SpineType::Dynam,
                SpineType::Fing,
                SpineType::Text,
                SpineType::MxmlChords,
            ]),
            KernLine::TandemInterpretationLine(
                crate::TandemInterpretationLine::new(vec![
                    TandemInterpretation::Null,
                    TandemInterpretation::Null,
                ])
                .unwrap(),
            ),
            KernLine::SpineManipulatorLine(
                crate::SpineManipulatorLine::new(vec![
                    SpineManipulator::Null,
                    SpineManipulator::Null,
                ])
                .unwrap(),
            ),
            KernLine::Data(vec![
                SpineContent::Kern(KernContent::Null),
                SpineContent::Dynam(DynamContent::Null),
                SpineContent::Fing(FingContent::Null),
                SpineContent::Text(TextContent::Null),
                SpineContent::MxmlHarmony(MxmlHarmonyContent::Null),
            ]),
            KernLine::Barline(Barline::new(Vec::new(), RepeatStyle::None)),
        ]);

        RemoveNullLines::new(Default::default()).normalize(&mut document);

        assert_eq!(
            document.lines(),
            vec![
                KernLine::SpineTypeLine(vec![
                    SpineType::Kern,
                    SpineType::Dynam,
                    SpineType::Fing,
                    SpineType::Text,
                    SpineType::MxmlChords,
                ]),
                KernLine::Barline(Barline::new(Vec::new(), RepeatStyle::None)),
            ]
        );
    }

    #[test]
    fn preserves_mixed_and_non_null_lines() {
        let mixed_tandem = KernLine::TandemInterpretationLine(
            crate::TandemInterpretationLine::new(vec![
                TandemInterpretation::Null,
                TandemInterpretation::Raw(RawToken("*custom".to_owned())),
            ])
            .unwrap(),
        );
        let mixed_manipulator = KernLine::SpineManipulatorLine(
            crate::SpineManipulatorLine::new(vec![SpineManipulator::Null, SpineManipulator::Split])
                .unwrap(),
        );
        let mixed_data = KernLine::Data(vec![
            SpineContent::Kern(KernContent::Null),
            SpineContent::Text(TextContent::Text(TextUnderlay {
                text: "word".to_owned(),
            })),
        ]);
        let opaque_dot = KernLine::Data(vec![SpineContent::Opaque {
            spine_type: SpineType::Other("**foo".to_owned()),
            raw: RawToken(".".to_owned()),
        }]);
        let empty_data = KernLine::Data(Vec::new());
        let mut document = test_document(vec![
            KernLine::SpineTypeLine(vec![SpineType::Kern]),
            mixed_tandem.clone(),
            mixed_manipulator.clone(),
            mixed_data.clone(),
            opaque_dot.clone(),
            empty_data.clone(),
        ]);

        RemoveNullLines::new(Default::default()).normalize(&mut document);

        assert_eq!(
            document.lines(),
            vec![
                KernLine::SpineTypeLine(vec![SpineType::Kern]),
                mixed_tandem,
                mixed_manipulator,
                mixed_data,
                opaque_dot,
                empty_data,
            ]
        );
    }

    #[test]
    fn preserves_non_null_data_lines() {
        let harmony = KernLine::Data(vec![SpineContent::MxmlHarmony(MxmlHarmonyContent::Label(
            MxmlHarmonyLabel {
                text: "G major".to_owned(),
            },
        ))]);
        let mut document = test_document(vec![harmony.clone()]);

        RemoveNullLines::new(Default::default()).normalize(&mut document);

        assert_eq!(document.lines(), vec![harmony]);
    }

    #[test]
    fn preserves_non_null_fingering_data_lines() {
        let fingering = KernLine::Data(vec![SpineContent::Fing(FingContent::Fingerings(vec![
            Some(crate::Fingering::single(
                crate::FingerNumber::try_from(3).unwrap(),
            )),
        ]))]);
        let mut document = test_document(vec![fingering.clone()]);

        RemoveNullLines::new(Default::default()).normalize(&mut document);

        assert_eq!(document.lines(), vec![fingering]);
    }

    fn test_document(lines: Vec<KernLine>) -> KernDocument {
        KernDocument::from_parts(
            KernMetadata::default(),
            DocumentLayout {
                initial_spine_count: 1,
            },
            lines
                .into_iter()
                .enumerate()
                .map(|(index, line)| (line, Some(index + 1)))
                .collect(),
        )
    }
}
