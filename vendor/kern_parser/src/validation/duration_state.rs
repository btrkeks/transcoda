//! Per-spine duration-state validation.
//!
//! This pass checks the timeline invariant used by Transcoda target
//! projection: duration-bearing events must not start in an active `**kern`
//! spine while that spine still carries remaining duration from a previous
//! event, including across split and merge topology changes.

use thiserror::Error;

use crate::{
    analysis::{score_line_durations, DataLineDurationError},
    score::Score,
};

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum DurationStateValidationError {
    #[error("invalid duration state: {0}")]
    InvalidDurationState(#[from] DataLineDurationError),
}

impl DurationStateValidationError {
    pub fn line(&self) -> usize {
        match self {
            Self::InvalidDurationState(source) => source.line(),
        }
    }

    pub fn column(&self) -> Option<usize> {
        match self {
            Self::InvalidDurationState(source) => source.column(),
        }
    }
}

pub fn validate(score: &Score<'_>) -> Result<(), DurationStateValidationError> {
    score_line_durations(score)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(input: &str) -> Result<(), DurationStateValidationError> {
        let document = crate::KernDocument::parse(input).unwrap();
        validate(&Score::resolve(&document, crate::document::Framing::Excerpt).unwrap())
    }

    #[test]
    fn accepts_split_after_carried_duration_finishes() {
        let result = check(
            "**kern\t**kern\n\
*M4/4\t*M4/4\n\
4c\t4e\n\
*^\t*\n\
4d\t4f\t4a\n\
*-\t*-\t*-\n",
        );

        result.unwrap();
    }

    #[test]
    fn rejects_new_duration_in_split_child_before_carried_duration_finishes() {
        let result = check(
            "**kern\t**kern\n\
*M4/4\t*M4/4\n\
2c\t4e\n\
*^\t*\n\
.\t4d\t4f\n\
*-\t*-\t*-\n",
        );

        let error = result.unwrap_err();

        assert!(error
            .to_string()
            .contains("duration-bearing token starts before previous token has finished"));
    }
}
