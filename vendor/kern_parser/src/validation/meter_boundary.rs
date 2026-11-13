//! Meter-change boundary validation.

use thiserror::Error;

use crate::{
    score::{Score, TrackId},
    DynamContent, KernContent, KernLine, MxmlHarmonyContent, SpineContent, TandemInterpretation,
    TandemInterpretationLine, TextContent,
};

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum MeterBoundaryValidationError {
    #[error(
        "line {line}: meter interpretation must be associated with a barline boundary in columns {columns:?}"
    )]
    MissingBoundary { line: usize, columns: Vec<usize> },
}

/// A numeric meter change sits next to a barline: only interpretations,
/// comments, spine manipulations and null data in the changing tracks may
/// separate them. Meter is track state, so every subspine of a changing track
/// must be null there.
pub fn validate(score: &Score<'_>) -> Result<(), MeterBoundaryValidationError> {
    let document = score.document();
    let topology = score.topology();
    let first_content_or_barline = first_content_or_barline_index(score);

    for (line_index, line) in document.lines().iter().enumerate() {
        let KernLine::TandemInterpretationLine(tandems) = line else {
            continue;
        };
        let affected_columns = numeric_meter_columns(tandems);
        if affected_columns.is_empty() || line_index < first_content_or_barline {
            continue;
        }
        let mut tracks = affected_columns
            .iter()
            .map(|&column| topology.track_of(topology.layer_at(line_index, column)))
            .collect::<Vec<_>>();
        tracks.dedup();

        if scan_for_boundary(score, line_index, &tracks, Direction::Backward)
            || scan_for_boundary(score, line_index, &tracks, Direction::Forward)
        {
            continue;
        }

        return Err(MeterBoundaryValidationError::MissingBoundary {
            line: document.source_line_number(line_index),
            columns: affected_columns
                .iter()
                .map(|column_index| column_index + 1)
                .collect(),
        });
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Backward,
    Forward,
}

fn first_content_or_barline_index(score: &Score<'_>) -> usize {
    let lines = score.document().lines();
    lines
        .iter()
        .position(|line| matches!(line, KernLine::Barline(_) | KernLine::Data(_)))
        .unwrap_or(lines.len())
}

fn numeric_meter_columns(tandems: &TandemInterpretationLine) -> Vec<usize> {
    tandems
        .tokens()
        .iter()
        .enumerate()
        .filter_map(|(index, tandem)| match tandem {
            TandemInterpretation::TimeSignature(_) | TandemInterpretation::FreeMeter => Some(index),
            _ => None,
        })
        .collect()
}

fn scan_for_boundary(
    score: &Score<'_>,
    line_index: usize,
    tracks: &[TrackId],
    direction: Direction,
) -> bool {
    let lines = score.document().lines();
    let mut next_index = match direction {
        Direction::Backward => line_index.checked_sub(1),
        Direction::Forward => Some(line_index + 1).filter(|index| *index < lines.len()),
    };

    while let Some(index) = next_index {
        match &lines[index] {
            KernLine::Barline(_) => return true,
            KernLine::Data(contents) => {
                let topology = score.topology();
                let null = tracks.iter().all(|&track| {
                    topology
                        .track_columns(index, track)
                        .all(|column| content_is_null(contents.get(column)))
                });
                if !null {
                    return false;
                }
            }
            KernLine::SpineTypeLine(_)
            | KernLine::TandemInterpretationLine(_)
            | KernLine::SpineManipulatorLine(_)
            | KernLine::LocalCommentLine(_) => {}
        }

        next_index = match direction {
            Direction::Backward => index.checked_sub(1),
            Direction::Forward => Some(index + 1).filter(|index| *index < lines.len()),
        };
    }

    false
}

fn content_is_null(content: Option<&SpineContent>) -> bool {
    matches!(
        content,
        Some(
            SpineContent::Kern(KernContent::Null)
                | SpineContent::Dynam(DynamContent::Null)
                | SpineContent::Text(TextContent::Null)
                | SpineContent::MxmlHarmony(MxmlHarmonyContent::Null)
        )
    ) || matches!(content, Some(SpineContent::Opaque { raw, .. }) if raw.0 == ".")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(input: &str) -> Result<(), MeterBoundaryValidationError> {
        let document = crate::KernDocument::parse(input).unwrap();
        validate(&Score::resolve(&document, crate::document::Framing::Excerpt).unwrap())
    }

    fn assert_valid(input: &str) {
        check(input).unwrap();
    }

    fn assert_invalid(input: &str, expected: &str) {
        let error = check(input).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "expected {error:?} to contain {expected:?}"
        );
    }

    #[test]
    fn accepts_initial_header_meter_without_barline() {
        assert_valid("**kern\n*M4/4\n4c\n*-\n");
    }

    #[test]
    fn accepts_meter_immediately_after_barline() {
        assert_valid("**kern\n*M4/4\n4c\n=\n*M3/4\n4d\n*-\n");
    }

    #[test]
    fn accepts_meter_immediately_before_barline() {
        assert_valid("**kern\n*M4/4\n4c\n*M3/4\n=\n4d\n*-\n");
    }

    #[test]
    fn ignores_interpretations_spine_operations_and_null_data_between_meter_and_barline() {
        assert_valid("**kern\n*M4/4\n4c\n*M3/4\n*k[]\n*^\n.\t.\n=\t=\n4d\t4f\n*-\t*-\n");
    }

    #[test]
    fn rejects_meter_between_musical_data_lines_without_boundary() {
        assert_invalid("**kern\n*M4/4\n4c\n*M3/4\n4d\n*-\n", "line 4");
    }

    #[test]
    fn accepts_scalar_barline_for_all_affected_meter_columns() {
        assert_valid("**kern\t**kern\n*M4/4\t*M4/4\n4c\t4e\n*M3/4\t*M3/4\n=\n4d\t4g\n*-\t*-\n");
    }

    #[test]
    fn accepts_scalar_barline_when_one_meter_column_changes() {
        assert_valid("**kern\t**kern\n*M4/4\t*M4/4\n4c\t4e\n*M3/4\t*\n=\n4d\t4g\n*-\t*-\n");
    }

    #[test]
    fn meter_in_one_subspine_needs_every_sibling_null_before_the_barline() {
        assert_valid("**kern\n*M4/4\n*^\n2c\t2e\n*M3/4\t*\n.\t.\n=\t=\n2.d\t2.f\n*-\t*-\n");
        assert_invalid(
            "**kern\n*M4/4\n*^\n2c\t2e\n*M3/4\t*\n.\t4f\n=\t=\n*-\t*-\n",
            "line 5",
        );
    }
}
