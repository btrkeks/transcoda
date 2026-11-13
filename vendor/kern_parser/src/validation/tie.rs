//! Warning-only validation for ordinary document-level tie balance.

use crate::{
    analysis::{
        score_line_durations, tie_chains::analyze_timed_tie_chains, TieChain, TieChainAnalysisError,
    },
    score::Score,
    validation::{SourceLocation, ValidationSeverity, ValidationWarning},
};

pub(crate) struct TieCheckReport {
    pub chains: Vec<TieChain>,
    pub warnings: Vec<ValidationWarning>,
}

pub(crate) fn check(score: &Score<'_>) -> Result<TieCheckReport, TieChainAnalysisError> {
    let analysis = analyze_timed_tie_chains(score, score_line_durations(score)?)?;
    Ok(TieCheckReport {
        chains: analysis.chains,
        warnings: analysis
            .findings
            .into_iter()
            .map(|finding| ValidationWarning {
                span: None,
                severity: ValidationSeverity::Warning,
                code: finding.code,
                loose_end: finding.loose_end,
                location: SourceLocation {
                    line: finding.source_line,
                    column: Some(finding.field_index + 1),
                },
                message: finding.message,
            })
            .collect(),
    })
}
