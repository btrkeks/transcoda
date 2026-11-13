//! Parsers for the typed Humdrum `**kern` AST.

mod barline;
mod data;
mod document;
mod finalize;
mod line;
mod spine_state;
mod tandem;
pub(crate) mod token;

#[cfg(test)]
mod tests;

use crate::{KernContent, ParseError};

pub use document::{ParseDialect, ParseOptions};

/// Parse one `**kern` data token without document-level validation.
pub fn parse_kern_token(raw: &str, options: ParseOptions) -> Result<KernContent, ParseError> {
    KernContent::from_token(raw, &options)
}
