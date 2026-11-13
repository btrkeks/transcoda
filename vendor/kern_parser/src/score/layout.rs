//! `!LO:` layout parameters, linked to the token each one applies to.
//!
//! humlib links a `!LO:` local comment to the next token below it in the same
//! layer that is data (a null `.` included), a barline, or an interpretation
//! other than `*` (`HumdrumFileStructure::processLocalParametersForStrand`).
//! A parameter above a spine manipulator belongs to the manipulator, so it
//! never reaches the layers the manipulator creates.

use std::collections::HashMap;

use super::{StateError, Topology};
use crate::{
    parse::token::duration_spelling, Duration, KernDocument, KernLine, LocalComment,
    SpineManipulator, TandemInterpretation,
};

/// Layout parameters by `(line_index, column)` of the token they apply to.
#[derive(Debug, Clone, Default)]
pub(super) struct LinkedLayouts {
    visual_durations: HashMap<(usize, usize), Duration>,
}

impl LinkedLayouts {
    /// The `!LO:N:vis=` duration of the whole token, if one is linked to it.
    pub(super) fn visual_duration(&self, line_index: usize, column: usize) -> Option<&Duration> {
        self.visual_durations.get(&(line_index, column))
    }
}

/// One `!LO:` comment still looking for its token.
struct Pending<'d> {
    line: usize,
    column: usize,
    raw: &'d str,
}

pub(super) fn link(
    document: &KernDocument,
    topology: &Topology,
) -> (LinkedLayouts, Vec<StateError>) {
    let mut pending: Vec<Vec<Pending<'_>>> =
        (0..topology.layer_count()).map(|_| Vec::new()).collect();
    let mut linked = LinkedLayouts::default();
    let mut errors = Vec::new();
    for (index, line) in document.lines().iter().enumerate() {
        let layers = topology.layers_at(index);
        let takes: Vec<bool> = match line {
            KernLine::LocalCommentLine(comments) => {
                for (column, (&layer, comment)) in layers.iter().zip(comments).enumerate() {
                    if let LocalComment::Unsupported(raw) = comment {
                        if raw.starts_with("!LO:") {
                            pending[layer.index()].push(Pending {
                                line: document.source_line_number(index),
                                column: column + 1,
                                raw,
                            });
                        }
                    }
                }
                continue;
            }
            KernLine::TandemInterpretationLine(tandems) => tandems
                .tokens()
                .iter()
                .map(|tandem| *tandem != TandemInterpretation::Null)
                .collect(),
            KernLine::SpineManipulatorLine(manipulators) => manipulators
                .tokens()
                .iter()
                .map(|manipulator| *manipulator != SpineManipulator::Null)
                .collect(),
            KernLine::Data(_) | KernLine::Barline(_) | KernLine::SpineTypeLine(_) => {
                vec![true; layers.len()]
            }
        };
        for (column, (&layer, takes)) in layers.iter().zip(takes).enumerate() {
            if !takes {
                continue;
            }
            for comment in pending[layer.index()].drain(..) {
                let Some(value) = chord_visual_duration(comment.raw) else {
                    continue;
                };
                match duration_spelling(value) {
                    Some(duration) => {
                        linked
                            .visual_durations
                            .entry((index, column))
                            .or_insert(duration);
                    }
                    None => errors.push(StateError::InvalidVisualDuration {
                        line: comment.line,
                        column: comment.column,
                        value: value.to_owned(),
                    }),
                }
            }
        }
    }
    (linked, errors)
}

/// The `vis` value of an `!LO:N:` comment that addresses the whole token.
/// A comment with `n=` addresses one note of a chord.
fn chord_visual_duration(raw: &str) -> Option<&str> {
    let parameters: Vec<&str> = raw.strip_prefix("!LO:N:")?.split(':').collect();
    if parameters
        .iter()
        .any(|parameter| parameter.starts_with("n="))
    {
        return None;
    }
    parameters
        .iter()
        .find_map(|parameter| parameter.strip_prefix("vis="))
}
