use crate::error::SpineCheckError;

#[derive(Debug, Default, Clone, Copy)]
pub struct SpineChecker;

impl SpineChecker {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    pub fn check_snippet(&self, _snippet: &str) -> Result<(), SpineCheckError> {
        Err(SpineCheckError::NotYetImplemented)
    }
}
