use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalViolation {
    pub rule: String,
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl CanonicalViolation {
    #[must_use]
    pub fn new(
        rule: impl Into<String>,
        message: impl Into<String>,
        line: usize,
        column: usize,
    ) -> Self {
        Self {
            rule: rule.into(),
            message: message.into(),
            line,
            column,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SpineCheckError {
    #[error("spine checker is not implemented yet")]
    NotYetImplemented,

    #[error("{0}")]
    CanonicalViolation(CanonicalViolationDisplay),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalViolationDisplay(pub CanonicalViolation);

impl std::fmt::Display for CanonicalViolationDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let violation = &self.0;
        write!(
            f,
            "{} at line {}, column {}: {}",
            violation.rule, violation.line, violation.column, violation.message
        )
    }
}

impl From<CanonicalViolation> for SpineCheckError {
    fn from(value: CanonicalViolation) -> Self {
        Self::CanonicalViolation(CanonicalViolationDisplay(value))
    }
}
