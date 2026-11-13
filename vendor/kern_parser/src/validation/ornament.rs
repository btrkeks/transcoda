//! Validation for ornament spellings that are parseable but underspecified.

use crate::types::{Ornament, Placed, Turn, TurnForm};
use crate::validation::{SourceLocation, ValidationSeverity, ValidationWarning};
use crate::{KernContent, KernDocument, SpineContent};

pub fn validate(_document: &KernDocument) -> Result<(), std::convert::Infallible> {
    Ok(())
}

pub(crate) fn check(document: &KernDocument) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();
    for row in document.data_rows() {
        for (column_index, content) in row.contents.iter().enumerate() {
            let SpineContent::Kern(content) = content else {
                continue;
            };
            match content {
                KernContent::Chord(chord) => {
                    push_underspecified_turn_warnings(
                        &mut warnings,
                        row.line_number,
                        column_index + 1,
                        &chord.modifiers.ornaments,
                    );
                }
                KernContent::Grace(grace) => {
                    push_underspecified_turn_warnings(
                        &mut warnings,
                        row.line_number,
                        column_index + 1,
                        &grace.modifiers.ornaments,
                    );
                }
                KernContent::Null
                | KernContent::Rest(_)
                | KernContent::InvisibleRestSlice(_)
                | KernContent::MultiMeasureRest(_)
                | KernContent::Note(_) => {}
            }
        }
    }
    warnings
}

fn push_underspecified_turn_warnings(
    warnings: &mut Vec<ValidationWarning>,
    line: usize,
    column: usize,
    ornaments: &[Placed<Ornament>],
) {
    for ornament in ornaments {
        let Ornament::Turn(turn) = ornament.mark else {
            continue;
        };
        if turn.intervals.is_some() {
            continue;
        }
        warnings.push(ValidationWarning {
            span: None,
            severity: ValidationSeverity::Warning,
            loose_end: None,
            code: "ornament.underspecified-turn",
            location: SourceLocation {
                line,
                column: Some(column),
            },
            message: underspecified_turn_message(turn),
        });
    }
}

fn underspecified_turn_message(turn: Turn) -> String {
    let label = match turn.form {
        TurnForm::Regular => "regular turn",
        TurnForm::Inverted => "inverted turn",
    };
    let example = match turn.form {
        TurnForm::Regular => "SSS",
        TurnForm::Inverted => "$SS",
    };
    format!(
        "{label} is underspecified for VHV rendering; use an explicit spelling such as `{example}`"
    )
}
