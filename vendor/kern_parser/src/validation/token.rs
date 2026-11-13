//! Marks a `**kern` data token keeps as atoms but leaves out of its AST.

use super::{SourceLocation, ValidationSeverity, ValidationWarning};
use crate::{document::Document, parse::token, KernContent};

pub(crate) fn check(document: &Document) -> Vec<ValidationWarning> {
    let dialect = document.options().dialect;
    document
        .kern_fields()
        .filter(|(.., content)| !matches!(content, KernContent::Null))
        .flat_map(|(line, field, text, _)| {
            token::warnings(text, dialect)
                .into_iter()
                .map(move |(span, warning)| ValidationWarning {
                    code: warning.code(),
                    severity: ValidationSeverity::Warning,
                    location: SourceLocation {
                        line,
                        column: Some(field + 1),
                    },
                    message: warning.message(),
                    loose_end: None,
                    span: Some(token::char_span(text, span)),
                })
        })
        .collect()
}
