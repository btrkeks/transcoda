//! Spine types while parsing.
//!
//! A token's grammar depends on its spine's type, so the parser follows `*^`,
//! `*v` and `*-` for spine types alone, before a document exists to resolve.
//! Everything else about spine identity comes from
//! [`crate::score::Topology`] once the document is parsed.

use crate::{types::*, ParseError};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct SpineState {
    active: Vec<SpineType>,
}

impl SpineState {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn initialize(&mut self, active: Vec<SpineType>) {
        self.active = active;
    }

    pub(super) fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    pub(super) fn expected_spine_count(&self) -> Option<usize> {
        (!self.active.is_empty()).then_some(self.active.len())
    }

    pub(super) fn validate_line_width(&self, line: usize, found: usize) -> Result<(), ParseError> {
        let Some(expected) = self.expected_spine_count() else {
            return Ok(());
        };

        if found != expected {
            return Err(ParseError::SpineCountMismatch {
                line,
                expected,
                found,
            });
        }

        Ok(())
    }

    pub(super) fn spine_type(&self, index: SpineIndex) -> Result<SpineType, ParseError> {
        if self.active.is_empty() {
            return Ok(SpineType::Kern);
        }

        self.active.get(index.0).cloned().ok_or_else(|| {
            invalid(format!(
                "missing active spine type for column {}",
                index.0 + 1
            ))
        })
    }

    /// Spine types after a manipulator line, which already has one token per
    /// active spine. A merge must join at least two spines of one type.
    pub(super) fn apply_manipulators(
        &mut self,
        manipulators: &[SpineManipulator],
    ) -> Result<(), ParseError> {
        if manipulators.len() != self.active.len() {
            return Err(invalid(
                "spine manipulator count does not match active spine count".to_owned(),
            ));
        }
        let mut next = Vec::with_capacity(self.active.len());
        let mut column = 0;
        while column < manipulators.len() {
            let spine_type = &self.active[column];
            match manipulators[column] {
                SpineManipulator::Null => next.push(spine_type.clone()),
                SpineManipulator::Terminate => {}
                SpineManipulator::Split => {
                    next.push(spine_type.clone());
                    next.push(spine_type.clone());
                }
                SpineManipulator::Merge => {
                    let start = column;
                    while column + 1 < manipulators.len()
                        && manipulators[column + 1] == SpineManipulator::Merge
                    {
                        column += 1;
                    }
                    if column == start {
                        return Err(invalid(
                            "merge requires at least 2 consecutive *v tokens".to_owned(),
                        ));
                    }
                    if self.active[start..=column]
                        .iter()
                        .any(|other| other != spine_type)
                    {
                        return Err(invalid(
                            "cannot merge spines with different spine types".to_owned(),
                        ));
                    }
                    next.push(spine_type.clone());
                }
            }
            column += 1;
        }
        self.active = next;
        Ok(())
    }
}

fn invalid(message: String) -> ParseError {
    ParseError::InvalidSpineOperation { message }
}
