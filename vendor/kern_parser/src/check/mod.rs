//! Editor-oriented diagnostics for single `**kern` documents.

use std::ops::Range;

use serde::Serialize;

mod acknowledgements;
pub(crate) use acknowledgements::acknowledged_tokens;
use acknowledgements::{acknowledge_diagnostics, apply_acknowledgements};

use crate::{
    analysis::{LooseEnd, TieChain},
    parse::{token, ParseDialect},
    validation::{
        self, beam::BeamValidationError, duration_state::DurationStateValidationError,
        meter_boundary::MeterBoundaryValidationError, rhythm::RhythmValidationError,
        ValidationSeverity, ValidationWarning,
    },
    KernDocument, ParseError, ValidationConfig, ValidationPass,
};

pub const CHECK_SCHEMA_VERSION: &str = "2";
/// Schema of [`CheckEvidence`], deliberately apart from the report's: a
/// consumer that reads one must never mistake the other for it.
pub const CHECK_EVIDENCE_SCHEMA_VERSION: &str = "evidence-1";

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CheckConfig {
    pub input_dialect: ParseDialect,
    pub framing: crate::document::Framing,
    pub deny_warnings: bool,
    pub validation: ValidationConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CheckReport {
    pub schema_version: &'static str,
    pub success: bool,
    pub diagnostics: Vec<CheckDiagnostic>,
    pub rules: Vec<validation::RuleOutcome>,
    pub tie_chains: Vec<TieChain>,
}

/// What `check` observed, before any conclusion is drawn from it.
///
/// Diagnostics carry the severity their rule gave them; an `!ack:` comment
/// has not demoted any of them yet. `acknowledgement_targets` say which token
/// each comment accepts, so a consumer can apply them, alongside findings of
/// its own, without following the layer topology itself.
/// [`CheckEvidence::into_report`] draws exactly the conclusion `check` reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CheckEvidence {
    pub schema_version: &'static str,
    pub diagnostics: Vec<CheckDiagnostic>,
    pub rules: Vec<validation::RuleOutcome>,
    pub tie_chains: Vec<TieChain>,
    pub acknowledgement_targets: Vec<AcknowledgementTarget>,
}

/// The token an `!ack:` comment accepts, addressed as diagnostics address
/// theirs: the source line counted from 1 and the field counted from 0.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AcknowledgementTarget {
    #[serde(rename = "line_1based")]
    pub line: usize,
    #[serde(rename = "field_0based")]
    pub field: usize,
    /// The comment after `!ack:`, as written.
    pub text: String,
}

impl CheckEvidence {
    fn observed(
        diagnostics: Vec<CheckDiagnostic>,
        rules: Vec<validation::RuleOutcome>,
        tie_chains: Vec<TieChain>,
        acknowledged: std::collections::BTreeMap<(usize, usize), String>,
    ) -> Self {
        Self {
            schema_version: CHECK_EVIDENCE_SCHEMA_VERSION,
            diagnostics,
            rules,
            tie_chains,
            acknowledgement_targets: acknowledged
                .into_iter()
                .map(|((line, field), text)| AcknowledgementTarget { line, field, text })
                .collect(),
        }
    }

    /// The report `check` writes: acknowledged validation findings demoted to
    /// info, and success decided over what remains.
    pub fn into_report(self, deny_warnings: bool) -> CheckReport {
        let acknowledged = self
            .acknowledgement_targets
            .into_iter()
            .map(|target| ((target.line, target.field), target.text))
            .collect();
        let mut diagnostics = self.diagnostics;
        apply_acknowledgements(&acknowledged, &mut diagnostics);
        CheckReport::from_diagnostics(diagnostics, self.rules, self.tie_chains, deny_warnings)
    }
}

impl CheckReport {
    fn from_diagnostics(
        diagnostics: Vec<CheckDiagnostic>,
        rules: Vec<validation::RuleOutcome>,
        tie_chains: Vec<TieChain>,
        deny_warnings: bool,
    ) -> Self {
        let success = !diagnostics.iter().any(|diagnostic| {
            diagnostic.severity == CheckSeverity::Error
                || (deny_warnings && diagnostic.severity == CheckSeverity::Warning)
        }) && !rules.iter().any(|outcome| {
            matches!(
                outcome.status,
                validation::RuleStatus::Blocked { .. } | validation::RuleStatus::Failed { .. }
            )
        });
        Self {
            schema_version: CHECK_SCHEMA_VERSION,
            success,
            diagnostics,
            rules,
            tie_chains,
        }
    }

    pub fn has_execution_failure(&self) -> bool {
        self.rules
            .iter()
            .any(|outcome| matches!(outcome.status, validation::RuleStatus::Failed { .. }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CheckDiagnostic {
    pub phase: CheckPhase,
    pub rule: Option<ValidationPass>,
    pub code: String,
    pub severity: CheckSeverity,
    pub message: String,
    #[serde(rename = "line_1based")]
    pub line: Option<usize>,
    #[serde(rename = "field_0based")]
    pub column: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loose_end: Option<LooseEnd>,
    /// One-based ordinal among repeated slur findings with the same code and location.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ordinal_1based: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acknowledged: Option<String>,
    /// Zero-based char offsets in the field.
    #[serde(rename = "span_chars_0based", skip_serializing_if = "Option::is_none")]
    pub span: Option<Range<usize>>,
    /// The token scanner's reason for refusing a `**kern` data token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckPhase {
    Parse,
    Validation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckSeverity {
    Error,
    Warning,
    Info,
}

pub fn check_text(input: &str, config: &CheckConfig) -> CheckReport {
    check_text_evidence(input, config).into_report(config.deny_warnings)
}

/// What [`check_text`] observes, with the `!ack:` targets it would apply.
pub fn check_text_evidence(input: &str, config: &CheckConfig) -> CheckEvidence {
    let document = match crate::document::Document::parse(
        input,
        crate::document::InputOptions {
            dialect: config.input_dialect,
            framing: config.framing,
        },
    ) {
        Ok(document) => document,
        Err(error) => {
            let (line, column) = (error.line(), error.field());
            let refusal = error
                .parse_error()
                .and_then(|error| token_refusal(error, config.input_dialect));
            let diagnostics = vec![CheckDiagnostic {
                phase: CheckPhase::Parse,
                rule: None,
                code: error.code().into(),
                severity: CheckSeverity::Error,
                message: error.to_string(),
                line,
                column,
                loose_end: None,
                ordinal_1based: None,
                acknowledged: None,
                span: refusal.as_ref().map(|(_, span)| span.clone()),
                reason: refusal.map(|(reason, _)| reason.to_owned()),
            }];
            let rules = config
                .validation
                .passes()
                .iter()
                .map(|&rule| validation::RuleOutcome {
                    rule,
                    status: validation::RuleStatus::Blocked {
                        causes: vec![error.code().into()],
                    },
                })
                .collect();
            return CheckEvidence::observed(diagnostics, rules, Vec::new(), Default::default());
        }
    };
    source_document_evidence(&document, config)
}

/// The scanner's reason and char span for a refused `**kern` data token. The
/// token must refuse with the reported error again, so a refused field of
/// another spine type gets none.
fn token_refusal(
    error: &ParseError,
    dialect: ParseDialect,
) -> Option<(&'static str, Range<usize>)> {
    let ParseError::InvalidSpineToken { raw, source, .. } = error else {
        return None;
    };
    let refusal = token::refusal(raw, dialect).filter(|refusal| refusal.error == **source)?;
    Some((
        refusal.fault.reason.code(),
        token::char_span(raw, refusal.fault.span),
    ))
}

pub fn check_document(document: &KernDocument, config: &CheckConfig) -> CheckReport {
    let mut report = check_score(document, None, config);
    let active_hide = document.data_rows().any(|row| {
        row.contents.iter().any(|content| {
            let crate::SpineContent::Kern(content) = content else {
                return false;
            };
            content.noteheads().any(|head| {
                let modifiers = head.note.modifiers();
                modifiers.invisible_accidental && !modifiers.invisible
            })
        })
    });
    if !active_hide
        || report
            .diagnostics
            .iter()
            .any(|d| d.rule == Some(ValidationPass::State) && d.severity == CheckSeverity::Error)
    {
        return report;
    }
    // A document whose score does not resolve already reports that through
    // the rules behind `state`.
    if let Ok(output) = crate::serialize::render::to_render_output(
        document,
        crate::serialize::render::RenderSerializeOptions {
            preserve_unsupported_local_comments: true,
            ..Default::default()
        },
    ) {
        if let Ok(source) = crate::document::Document::parse(
            &output.canonical_text(),
            crate::document::InputOptions::default(),
        ) {
            let mut extra = hidden_pitch_diagnostics(&source);
            acknowledge_diagnostics(source.score(), &mut extra);
            for diagnostic in &mut extra {
                if let Some(p) = output.note_provenance().iter().find(|p| {
                    Some(p.rendered.line_number) == diagnostic.line
                        && Some(p.rendered.spine_index) == diagnostic.column
                }) {
                    diagnostic.line = Some(p.source.line_number);
                    diagnostic.column = Some(p.source.spine_index);
                }
            }
            report.diagnostics.extend(extra);
        }
    }
    CheckReport::from_diagnostics(
        report.diagnostics,
        report.rules,
        report.tie_chains,
        config.deny_warnings,
    )
}

pub(crate) fn check_source_document(
    document: &crate::document::Document,
    config: &CheckConfig,
) -> CheckReport {
    source_document_evidence(document, config).into_report(config.deny_warnings)
}

fn source_document_evidence(
    document: &crate::document::Document,
    config: &CheckConfig,
) -> CheckEvidence {
    let (mut diagnostics, rules, tie_chains) =
        observe_score(document.score(), Some(document), config);
    diagnostics.extend(hidden_pitch_diagnostics(document));
    CheckEvidence::observed(
        diagnostics,
        rules,
        tie_chains,
        acknowledged_tokens(document.score()),
    )
}

fn hidden_pitch_diagnostics(document: &crate::document::Document) -> Vec<CheckDiagnostic> {
    use crate::analysis::{accidental_display::HiddenRiskClass, diagnostics_index::ScoreIndex};
    if !document
        .source()
        .lines()
        .filter(|line| !line.starts_with(['*', '!', '=']))
        .any(|line| {
            line.split(['\t', ' '])
                .any(crate::analysis::accidental_display::active_hide)
        })
    {
        return Vec::new();
    }
    let index = ScoreIndex::build(document);
    let mut diagnostics = Vec::new();
    for cell in &index.view.cells {
        for note in cell.events.iter().flat_map(|e| &e.notes) {
            let Some(risk) = note.display.as_ref().and_then(|d| d.hidden_risk.as_ref()) else {
                continue;
            };
            let (code, severity) = match risk.class {
                HiddenRiskClass::GraceCarry => continue,
                HiddenRiskClass::OctaveCarry => {
                    ("accidental.hidden-octave-carry", CheckSeverity::Info)
                }
                HiddenRiskClass::TieCarry => ("accidental.hidden-tie-carry", CheckSeverity::Info),
                HiddenRiskClass::OttavaCarry => {
                    ("accidental.hidden-ottava-carry", CheckSeverity::Info)
                }
                HiddenRiskClass::Unexplained => {
                    ("accidental.hidden-pitch-mismatch", CheckSeverity::Warning)
                }
            };
            diagnostics.push(CheckDiagnostic {
                phase: CheckPhase::Validation, rule: None, code: code.into(), severity,
                message: format!("Bar {}, staff {}, note {}: hidden accidental encodes {}; without the glyph the reader infers {} ({:?}).",
                    cell.bar, risk.staff.map_or_else(|| "unknown".into(), |s| s.to_string()),
                    note.id.as_ref().map_or_else(|| format!("L{}F{}S{}", note.address.line, note.address.field + 1, note.address.chord_note + 1), |id| id.0.clone()),
                    risk.encoded_pitch, risk.inferred_pitch, risk.class),
                line: Some(note.address.line), column: Some(note.address.field), loose_end: None, ordinal_1based: None, acknowledged: None,
 span: None,
 reason: None,
            });
        }
    }
    diagnostics.extend(
        index
            .diagnostics
            .iter()
            .filter(|d| d.code == "pitch.display-unavailable")
            .map(|d| CheckDiagnostic {
                phase: CheckPhase::Validation,
                rule: None,
                code: "accidental.analysis-unavailable".into(),
                severity: CheckSeverity::Warning,
                message: d.message.clone(),
                line: d.line,
                column: d.field,
                loose_end: None,
                ordinal_1based: None,
                acknowledged: None,
                span: None,
                reason: None,
            }),
    );
    diagnostics
}

fn check_score(
    document: &KernDocument,
    source: Option<&crate::document::Document>,
    config: &CheckConfig,
) -> CheckReport {
    let (mut diagnostics, rules, tie_chains) = observe_score(document, source, config);
    acknowledge_diagnostics(document, &mut diagnostics);
    CheckReport::from_diagnostics(diagnostics, rules, tie_chains, config.deny_warnings)
}

/// Every rule's findings at the severity the rule gave them.
fn observe_score(
    document: &KernDocument,
    source: Option<&crate::document::Document>,
    config: &CheckConfig,
) -> (
    Vec<CheckDiagnostic>,
    Vec<validation::RuleOutcome>,
    Vec<TieChain>,
) {
    let mut diagnostics = Vec::new();
    let mut tie_chains = Vec::new();
    let mut rules = Vec::new();
    for result in validation::run_rules(document, source, &config.validation, config.framing) {
        let reports_error = result.reports_error(config.validation.passes());
        for warning in result.warnings {
            diagnostics.push(warning_diagnostic(result.rule, warning));
        }
        if let Some(error) = result.error.filter(|_| reports_error) {
            diagnostics.extend(error_diagnostics(error).into_iter().map(|mut diagnostic| {
                diagnostic.rule = Some(result.rule);
                diagnostic
            }));
        }
        rules.push(validation::RuleOutcome {
            rule: result.rule,
            status: result.status,
        });
        tie_chains.extend(result.tie_chains);
    }
    assign_slur_ordinals(&mut diagnostics);
    diagnostics.extend(unison_head_diagnostics(document, config.framing));
    (diagnostics, rules, tie_chains)
}

fn assign_slur_ordinals(diagnostics: &mut [CheckDiagnostic]) {
    let mut groups = std::collections::BTreeMap::<_, Vec<usize>>::new();
    for (index, diagnostic) in diagnostics.iter().enumerate() {
        if matches!(
            diagnostic.code.as_str(),
            "slur.unclosed-open" | "slur.unmatched-close" | "slur.linked-unpaired"
        ) {
            groups
                .entry((diagnostic.line, diagnostic.column, diagnostic.code.clone()))
                .or_default()
                .push(index);
        }
    }
    for indices in groups.values().filter(|indices| indices.len() > 1) {
        for (ordinal, &index) in indices.iter().enumerate() {
            diagnostics[index].ordinal_1based = Some(ordinal + 1);
        }
    }
}

/// Diagnose each note in a unison that Verovio draws as two heads.
fn unison_head_diagnostics(
    document: &KernDocument,
    framing: crate::document::Framing,
) -> Vec<CheckDiagnostic> {
    let Ok(score) = crate::score::Score::resolve(document, framing) else {
        return Vec::new();
    };
    crate::analysis::unison_heads::unmerged_unisons(&score)
        .into_iter()
        .flat_map(|unison| {
            unison.fields.map(|field| CheckDiagnostic {
                phase: CheckPhase::Validation,
                rule: None,
                code: "unison-head.unmerged".into(),
                severity: CheckSeverity::Info,
                message: format!(
                    "Verovio draws this note and the unison in field {} as two noteheads; `!LO:N:vis={}` on field {} makes them share one.",
                    unison.partner(field) + 1,
                    unison.visual,
                    unison.target + 1,
                ),
                line: Some(unison.line),
                column: Some(field),
                loose_end: None,
                ordinal_1based: None,
                acknowledged: None,
                span: None,
                reason: None,
            })
        })
        .collect()
}

fn error_diagnostics(error: crate::ValidationError) -> Vec<CheckDiagnostic> {
    match error {
        crate::ValidationError::State(errors) => {
            errors.0.into_iter().map(state_error_diagnostic).collect()
        }
        error => vec![error_diagnostic(error)],
    }
}

fn state_error_diagnostic(error: crate::score::StateError) -> CheckDiagnostic {
    use crate::score::{StateError, TopologyError};
    let (code, line, column) = match &error {
        StateError::Topology(TopologyError::WidthMismatch { line, .. }) => {
            ("state.width-mismatch", *line, None)
        }
        StateError::Topology(TopologyError::MergeRequiresAtLeastTwo { line }) => {
            ("state.invalid-merge", *line, None)
        }
        StateError::Topology(TopologyError::CrossTrackMerge { line, column }) => {
            ("state.cross-track-merge", *line, Some(*column))
        }
        StateError::Conflicting { line, column, .. } => {
            ("state.conflicting-values", *line, Some(*column))
        }
        StateError::UnmatchedClose { line, column, .. } => {
            ("state.unmatched-close", *line, Some(*column))
        }
        StateError::SpanMismatch { line, column, .. } => {
            ("state.span-mismatch", *line, Some(*column))
        }
        StateError::CueRenumbered { line, column, .. } => {
            ("state.cue-renumbered", *line, Some(*column))
        }
        StateError::GraceUnderRhythmScale { line, column, .. } => {
            ("state.grace-under-rscale", *line, Some(*column))
        }
        StateError::UndrawableDuration { line, column, .. } => {
            ("state.undrawable-duration", *line, Some(*column))
        }
        StateError::OpenAtEnd { line, column, .. } => ("state.open-at-end", *line, Some(*column)),
        StateError::InvalidVisualDuration { line, column, .. } => {
            ("state.invalid-visual-duration", *line, Some(*column))
        }
        StateError::StrayNatural { line, column, .. } => {
            ("state.stray-natural", *line, Some(*column))
        }
        StateError::KeyCancellation { line, column, .. } => {
            ("state.key-cancellation", *line, Some(*column))
        }
    };
    validation_diagnostic(code, error.to_string(), Some(line), column)
}

fn error_diagnostic(error: crate::ValidationError) -> CheckDiagnostic {
    use crate::ValidationError;
    match error {
        ValidationError::State(_) => unreachable!("expanded by error_diagnostics"),
        ValidationError::DurationState(error) => duration_state_error_diagnostic(error),
        ValidationError::MeterBoundary(error) => meter_boundary_error_diagnostic(error),
        ValidationError::Beam(error) => beam_error_diagnostic(error),
        ValidationError::Rhythm(error) => rhythm_error_diagnostic(error),
        ValidationError::MediumSinglePage(error) => {
            let code = match &error {
                validation::medium_single_page::MediumSinglePageValidationError::Render(_) => "medium-single-page.render-failed",
                validation::medium_single_page::MediumSinglePageValidationError::UnexpectedPageCount { .. } => "medium-single-page.unexpected-page-count",
            };
            validation_diagnostic(code, error.to_string(), None, None)
        }
        ValidationError::OmrTraining(error) => {
            validation_diagnostic("conversion.invalid-target", error.to_string(), None, None)
        }
    }
}

fn meter_boundary_error_diagnostic(error: MeterBoundaryValidationError) -> CheckDiagnostic {
    let (line, column) = match &error {
        MeterBoundaryValidationError::MissingBoundary { line, columns } => {
            (Some(*line), columns.first().copied())
        }
    };
    validation_diagnostic(
        "meter-boundary.missing-boundary",
        error.to_string(),
        line,
        column,
    )
}

fn duration_state_error_diagnostic(error: DurationStateValidationError) -> CheckDiagnostic {
    let line = Some(error.line());
    let column = error.column();
    validation_diagnostic(duration_state_code(&error), error.to_string(), line, column)
}

fn duration_state_code(error: &DurationStateValidationError) -> &'static str {
    use crate::analysis::DataLineDurationError;
    let DurationStateValidationError::InvalidDurationState(error) = error;
    match error {
        DataLineDurationError::State(_) => {
            unreachable!("duration-state times a resolved score")
        }
        DataLineDurationError::KernSpineContainsNonKern { .. } => "duration-state.non-kern-content",
        DataLineDurationError::TokenDuration { .. } => "duration-state.invalid-token-duration",
        DataLineDurationError::OverlappingDuration { .. } => "duration-state.overlap",
        DataLineDurationError::MultiMeasureRestWithoutMeter { .. } => {
            "duration-state.missing-meter"
        }
        DataLineDurationError::DurationUnderflow { .. } => "duration-state.underflow",
    }
}

fn beam_error_diagnostic(error: BeamValidationError) -> CheckDiagnostic {
    let line = Some(error.line());
    let column = error.column();
    validation_diagnostic(error.code(), error.to_string(), line, column)
}

fn warning_diagnostic(rule: ValidationPass, warning: ValidationWarning) -> CheckDiagnostic {
    let mut diagnostic = validation_diagnostic_with_severity(
        warning.code,
        match warning.severity {
            ValidationSeverity::Error => CheckSeverity::Error,
            ValidationSeverity::Warning => CheckSeverity::Warning,
        },
        warning.to_string(),
        Some(warning.location.line),
        warning.location.column,
    );
    diagnostic.rule = Some(rule);
    diagnostic.loose_end = warning.loose_end;
    diagnostic.span = warning.span;
    diagnostic
}

fn rhythm_error_diagnostic(error: RhythmValidationError) -> CheckDiagnostic {
    let (line, column) = match error.line() {
        Some(line) => (Some(line), error.column()),
        // Wrapped errors (e.g. rest-slice rehydration) may still embed their
        // location in the message text only.
        None => source_line_column(&error.to_string()),
    };
    validation_diagnostic(error.code(), error.message_without_location(), line, column)
}

fn validation_diagnostic(
    code: &str,
    message: String,
    line: Option<usize>,
    column: Option<usize>,
) -> CheckDiagnostic {
    validation_diagnostic_with_severity(code, CheckSeverity::Error, message, line, column)
}

fn validation_diagnostic_with_severity(
    code: &str,
    severity: CheckSeverity,
    message: String,
    line: Option<usize>,
    column: Option<usize>,
) -> CheckDiagnostic {
    CheckDiagnostic {
        phase: CheckPhase::Validation,
        rule: None,
        code: code.to_owned(),
        severity,
        message,
        line,
        column: column.map(|column| column.saturating_sub(1)),
        loose_end: None,
        ordinal_1based: None,
        acknowledged: None,
        span: None,
        reason: None,
    }
}

fn source_line_column(message: &str) -> (Option<usize>, Option<usize>) {
    // Scan every "line " occurrence and take the first one followed by a
    // number, so non-location uses such as "data line has 3 spines" or
    // "barline" never shadow the actual location marker.
    let mut remaining = message;
    while let Some(index) = remaining.find("line ") {
        let after_marker = &remaining[index + "line ".len()..];
        let (line, after_line) = parse_usize_prefix(after_marker);
        if let Some(line) = line {
            let column = after_line
                .strip_prefix(", column ")
                .or_else(|| after_line.strip_prefix(" column "))
                .and_then(|value| parse_usize_prefix(value).0);
            return (Some(line), column);
        }
        remaining = after_marker;
    }

    (None, None)
}

fn parse_usize_prefix(input: &str) -> (Option<usize>, &str) {
    let end = input
        .char_indices()
        .take_while(|(_, character)| character.is_ascii_digit())
        .map(|(index, character)| index + character.len_utf8())
        .last()
        .unwrap_or(0);
    if end == 0 {
        return (None, input);
    }

    (input[..end].parse().ok(), &input[end..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "engraving"))]
    fn requested_engraving_rule_reports_missing_engine_as_failure() {
        let report = check_text(
            "**kern\n4c\n*-\n",
            &CheckConfig {
                validation: ValidationConfig::with_passes(vec![ValidationPass::MediumSinglePage]),
                ..Default::default()
            },
        );
        assert!(!report.success);
        assert!(report.has_execution_failure());
        assert!(matches!(
            report.rules[0].status,
            validation::RuleStatus::Failed { .. }
        ));
        assert!(report.diagnostics[0].message.contains("engraving"));
    }

    #[test]
    fn note_rules_report_not_applicable_for_rest_only_input() {
        let report = check_text(
            "**kern\n4r\n*-\n",
            &CheckConfig {
                validation: ValidationConfig::with_passes(vec![
                    ValidationPass::Ornament,
                    ValidationPass::Slur,
                    ValidationPass::Tie,
                ]),
                ..Default::default()
            },
        );
        assert!(report.success);
        assert_eq!(report.rules.len(), 3);
        assert!(report
            .rules
            .iter()
            .all(|outcome| outcome.status == validation::RuleStatus::NotApplicable));
    }

    #[test]
    fn tie_only_reports_blocked_duration_analysis() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::Tie]),
            ..Default::default()
        };
        let report = check_text(
            "**kern\t**kern\n*M4/4\t*M4/4\n2c[\t4e\n4c]\t4f\n*-\t*-\n",
            &config,
        );
        assert!(!report.success);
        assert_eq!(report.diagnostics[0].code, "duration-state.overlap");
        assert_eq!(report.diagnostics[0].rule, Some(ValidationPass::Tie));
        assert_eq!(report.diagnostics[0].line, Some(4));
        assert_eq!(report.diagnostics[0].column, Some(0));
        assert_eq!(
            report.rules[0].status,
            validation::RuleStatus::Blocked {
                causes: vec!["duration-state".into()]
            }
        );
        assert!(!report.has_execution_failure());
    }

    #[test]
    fn slur_loose_ends_preserve_marker_multiplicity() {
        use crate::analysis::{LooseEndSide, Spanner};
        for (token, count, code, side, message) in [
            (
                "(4c(",
                2,
                "slur.unclosed-open",
                LooseEndSide::Open,
                "2 slurs opened here are never closed",
            ),
            (
                "((4c",
                2,
                "slur.unclosed-open",
                LooseEndSide::Open,
                "2 slurs opened here are never closed",
            ),
            (
                "4c))",
                2,
                "slur.unmatched-close",
                LooseEndSide::Close,
                "2 closing slur markers have no matching open slurs",
            ),
            (
                "(4c",
                1,
                "slur.unclosed-open",
                LooseEndSide::Open,
                "slur opened here is never closed",
            ),
            (
                "4c)",
                1,
                "slur.unmatched-close",
                LooseEndSide::Close,
                "closing slur marker has no matching open slur",
            ),
        ] {
            for framing in [
                crate::document::Framing::Excerpt,
                crate::document::Framing::Complete,
            ] {
                let report = check_text(
                    &format!("**kern\n{token}\n4d\n*-\n"),
                    &CheckConfig {
                        framing,
                        validation: ValidationConfig::with_passes(vec![ValidationPass::Slur]),
                        ..Default::default()
                    },
                );
                assert_eq!(report.diagnostics.len(), count, "{token}");
                assert_eq!(report.success, framing == crate::document::Framing::Excerpt);
                for (index, diagnostic) in report.diagnostics.iter().enumerate() {
                    assert_eq!(diagnostic.code, code);
                    assert_eq!(diagnostic.message, message);
                    assert_eq!(
                        diagnostic.severity,
                        if report.success {
                            CheckSeverity::Warning
                        } else {
                            CheckSeverity::Error
                        }
                    );
                    assert_eq!((diagnostic.line, diagnostic.column), (Some(2), Some(0)));
                    assert_eq!(
                        diagnostic.loose_end,
                        Some(LooseEnd {
                            spanner: Spanner::Slur,
                            side,
                            staff: 1,
                            pitch: None
                        })
                    );
                    assert_eq!(diagnostic.ordinal_1based, (count > 1).then_some(index + 1));
                }
            }
        }
    }

    #[test]
    fn warnings_only_succeed_unless_explicitly_denied() {
        let input = "**kern\n(4c\n*-\n";
        let mut config = CheckConfig {
            framing: crate::document::Framing::Excerpt,
            validation: ValidationConfig::with_passes(vec![ValidationPass::Slur]),
            ..Default::default()
        };
        let report = check_text(input, &config);
        assert!(report.success);
        assert_eq!(report.rules[0].status, validation::RuleStatus::Findings);
        assert_eq!(report.diagnostics[0].code, "slur.unclosed-open");
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["schema_version"], "2");
        assert_eq!(json["diagnostics"][0]["line_1based"], 2);
        assert_eq!(json["diagnostics"][0]["field_0based"], 0);
        config.deny_warnings = true;
        assert!(!check_text(input, &config).success);
    }

    #[test]
    fn state_warns_about_restatements_that_draw_nothing() {
        // A restated ottava draws nothing; a restated clef draws a clef.
        let input = "**kern\n*clefG2\n*8va\n4c\n*8va\n*clefG2\n4d\n*X8va\n*-\n";
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::State]),
            ..Default::default()
        };
        let report = check_text(input, &config);
        assert!(report.success);
        let codes = report
            .diagnostics
            .iter()
            .map(|diagnostic| {
                (
                    diagnostic.code.as_str(),
                    diagnostic.line,
                    diagnostic.severity,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            codes,
            [("state.redundant", Some(5), CheckSeverity::Warning)]
        );
    }

    #[test]
    fn all_adapters_accept_regular_beam_crossing_barline() {
        let input = "**kern\n8cL\n=\n8dJ\n*-\n";
        let document = KernDocument::parse(input).unwrap();
        let validation = ValidationConfig::with_passes(vec![ValidationPass::Beam]);
        assert!(validation::validate_with_config(&document, &validation).is_ok());
        let report = validation::validate_report_with_config(&document, &validation);
        assert!(!report.has_errors());
        let report = check_text(
            input,
            &CheckConfig {
                validation,
                ..Default::default()
            },
        );
        assert!(report.success);
        assert!(report.diagnostics.is_empty());
    }

    fn valid_kern() -> &'static str {
        "**kern\n*clefG2\n*k[]\n*M4/4\n4c\n*-\n"
    }

    #[test]
    fn check_valid_kern_returns_success() {
        let report = check_text(valid_kern(), &CheckConfig::default());

        assert!(report.success);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn check_parse_error_reports_line_and_column() {
        let report = check_text("**kern\t\n*-\t*-\n", &CheckConfig::default());

        assert!(!report.success);
        assert_eq!(report.diagnostics.len(), 1);
        let diagnostic = &report.diagnostics[0];
        assert_eq!(diagnostic.phase, CheckPhase::Parse);
        assert_eq!(diagnostic.rule, None);
        assert_eq!(diagnostic.line, Some(1));
        assert_eq!(diagnostic.column, Some(1));
    }

    #[test]
    fn check_state_error_reports_line_and_column() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::State]),
            ..Default::default()
        };
        let report = check_text("**kern\t**kern\n4c\t4e\n*v\t*v\n2g\n*-\n", &config);

        assert!(!report.success);
        assert_eq!(report.diagnostics.len(), 1);
        let diagnostic = &report.diagnostics[0];
        assert_eq!(diagnostic.phase, CheckPhase::Validation);
        assert_eq!(diagnostic.rule, Some(ValidationPass::State));
        assert_eq!(diagnostic.code, "state.cross-track-merge");
        assert_eq!(diagnostic.line, Some(3));
        assert_eq!(diagnostic.column, Some(1));
    }

    #[test]
    fn check_duration_state_error_reports_line_and_column() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::DurationState]),
            ..Default::default()
        };
        let report = check_text(
            "**kern\t**kern\n*M4/4\t*M4/4\n2c\t4e\n*^\t*\n.\t4d\t4f\n*-\t*-\t*-\n",
            &config,
        );

        assert!(!report.success);
        assert_eq!(report.diagnostics.len(), 1);
        let diagnostic = &report.diagnostics[0];
        assert_eq!(diagnostic.phase, CheckPhase::Validation);
        assert_eq!(diagnostic.rule, Some(ValidationPass::DurationState));
        assert_eq!(diagnostic.line, Some(5));
        assert_eq!(diagnostic.column, Some(1));
    }

    #[test]
    fn check_default_includes_rhythm_pass() {
        let report = check_text(
            "**kern\n*M4/4\n2c\n2d\n4e\n=\n*-\n",
            &CheckConfig::default(),
        );

        assert!(!report.success);
        assert!(report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.rule == Some(ValidationPass::Rhythm)));
    }

    #[test]
    fn check_rhythm_diagnostic_reports_location_fields_without_embedded_location() {
        let report = check_text(
            "**kern\n*M4/4\n2c\n2d\n4e\n=\n*-\n",
            &CheckConfig::default(),
        );

        assert!(!report.success);
        let diagnostic = report
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.rule == Some(ValidationPass::Rhythm))
            .expect("expected rhythm diagnostic");
        assert_eq!(diagnostic.line, Some(5));
        assert_eq!(diagnostic.column, Some(0));
        assert!(!diagnostic.message.contains("line 5"));
        assert!(diagnostic
            .message
            .contains("measure duration exceeds meter"));
    }

    #[test]
    fn check_rhythm_diagnostic_hints_for_unmarked_invisible_system_break() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::Rhythm]),
            ..Default::default()
        };
        let report = check_text("**kern\n*M4/4\n1c\n=\n2d\n=-\n2e\n=\n*-\n", &config);

        assert!(!report.success);
        assert_eq!(report.diagnostics.len(), 1);
        let diagnostic = &report.diagnostics[0];
        assert_eq!(diagnostic.rule, Some(ValidationPass::Rhythm));
        assert_eq!(diagnostic.line, Some(6));
        assert_eq!(diagnostic.column, Some(0));
        assert!(!diagnostic.message.contains("line 6"));
        assert!(diagnostic
            .message
            .contains("add `!!LO:LB:g=original` before the `=-`"));
    }

    #[test]
    fn check_default_includes_beam_pass() {
        let report = check_text(
            "**kern\n*M4/4\n8cJ\n8d\n8e\n8f\n8g\n8a\n8b\n8cc\n=\n*-\n",
            &CheckConfig::default(),
        );

        assert!(!report.success);
        assert!(report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.rule == Some(ValidationPass::Beam)));
    }

    #[test]
    fn check_default_includes_state_pass() {
        let report = check_text("**kern\n*8va\n4c\n*-\n", &CheckConfig::default());

        assert!(!report.success);
        assert!(report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.rule == Some(ValidationPass::State)));
    }

    #[test]
    fn check_reports_a_blocking_state_error_once() {
        let source = "**kern\n*8va\n4c\n*-\n";
        let report = check_text(source, &CheckConfig::default());

        assert_eq!(report.diagnostics.len(), 1);
        assert_eq!(report.diagnostics[0].rule, Some(ValidationPass::State));
        let rhythm = report
            .rules
            .iter()
            .find(|outcome| outcome.rule == ValidationPass::Rhythm)
            .unwrap();
        assert_eq!(
            rhythm.status,
            validation::RuleStatus::Blocked {
                causes: vec!["state".into()]
            }
        );

        // Without the state rule in the run, the blocked rule carries it.
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![
                ValidationPass::Rhythm,
                ValidationPass::Tie,
            ]),
            ..Default::default()
        };
        let report = check_text(source, &config);
        assert_eq!(
            report
                .diagnostics
                .iter()
                .map(|diagnostic| (diagnostic.rule, diagnostic.code.as_str()))
                .collect::<Vec<_>>(),
            [
                (Some(ValidationPass::Rhythm), "state.open-at-end"),
                (Some(ValidationPass::Tie), "state.open-at-end"),
            ]
        );
    }

    #[test]
    fn check_selected_state_pass_reports_open_ottava_at_spine_end() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::State]),
            ..Default::default()
        };
        let report = check_text("**kern\n*8va\n4c\n*-\n", &config);

        assert!(!report.success);
        assert_eq!(report.diagnostics.len(), 1);
        let diagnostic = &report.diagnostics[0];
        assert_eq!(diagnostic.phase, CheckPhase::Validation);
        assert_eq!(diagnostic.rule, Some(ValidationPass::State));
        assert_eq!(diagnostic.code, "state.open-at-end");
        assert_eq!(diagnostic.line, Some(4));
        assert_eq!(diagnostic.column, Some(0));
        assert!(diagnostic
            .message
            .contains("ottava opened at line 2 is still open where its spine ends"));
    }

    #[test]
    fn check_selected_beam_pass_reports_line_and_column() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::Beam]),
            ..Default::default()
        };
        let report = check_text(
            "**kern\n*M4/4\n8cJ\n8d\n8e\n8f\n8g\n8a\n8b\n8cc\n=\n*-\n",
            &config,
        );

        assert!(!report.success);
        assert_eq!(report.diagnostics.len(), 1);
        let diagnostic = &report.diagnostics[0];
        assert_eq!(diagnostic.phase, CheckPhase::Validation);
        assert_eq!(diagnostic.rule, Some(ValidationPass::Beam));
        assert_eq!(diagnostic.line, Some(3));
        assert_eq!(diagnostic.column, Some(0));
    }

    #[test]
    fn check_bare_turn_is_warning() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::Ornament]),
            ..Default::default()
        };
        let report = check_text("**kern\n4cS\n*-\n", &config);

        assert!(report.success);
        assert_eq!(report.diagnostics.len(), 1);
        let diagnostic = &report.diagnostics[0];
        assert_eq!(diagnostic.phase, CheckPhase::Validation);
        assert_eq!(diagnostic.rule, Some(ValidationPass::Ornament));
        assert_eq!(diagnostic.severity, CheckSeverity::Warning);
        assert_eq!(diagnostic.line, Some(2));
        assert_eq!(diagnostic.column, Some(0));
        assert!(diagnostic
            .message
            .contains("regular turn is underspecified for VHV rendering"));
    }

    #[test]
    fn check_explicit_turn_has_no_ornament_warning() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::Ornament]),
            ..Default::default()
        };
        let report = check_text("**kern\n4cSss\n4c$SS\n*-\n", &config);

        assert!(report.success);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn check_same_pitch_slur_has_no_warning() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::Slur]),
            ..Default::default()
        };
        let report = check_text("**kern\n(4c\n4c)\n*-\n", &config);

        assert!(report.success);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn check_default_allows_same_pitch_slur() {
        let report = check_text("**kern\n*M4/4\n(4c\n4c)\n*-\n", &CheckConfig::default());

        assert!(report.success);
        assert!(report
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.rule != Some(ValidationPass::Slur)));
    }

    #[test]
    fn check_three_event_slur_has_no_slur_warning() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::Slur]),
            ..Default::default()
        };
        let report = check_text("**kern\n(4c\n4d\n4c)\n*-\n", &config);

        assert!(report.success);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn check_dangling_slur_is_warning_in_excerpt() {
        let config = CheckConfig {
            framing: crate::document::Framing::Excerpt,
            validation: ValidationConfig::with_passes(vec![ValidationPass::Slur]),
            ..Default::default()
        };
        let report = check_text("**kern\n(4c\n*-\n", &config);

        assert!(report.success);
        assert_eq!(report.diagnostics.len(), 1);
        let diagnostic = &report.diagnostics[0];
        assert_eq!(diagnostic.phase, CheckPhase::Validation);
        assert_eq!(diagnostic.rule, Some(ValidationPass::Slur));
        assert_eq!(diagnostic.severity, CheckSeverity::Warning);
        assert_eq!(diagnostic.line, Some(2));
        assert_eq!(diagnostic.column, Some(0));
        assert!(diagnostic.message.contains("never closed"));
    }

    #[test]
    fn check_regular_beam_crossing_barline_still_reports_later_beam_error() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::Beam]),
            ..Default::default()
        };
        let report = check_text("**kern\n8cL\n=\n4d\n*-\n", &config);

        assert!(!report.success);
        assert_eq!(report.diagnostics.len(), 1);
        assert_eq!(report.diagnostics[0].severity, CheckSeverity::Error);
        assert!(report.diagnostics[0]
            .message
            .contains("unbeamable duration participates in a beam"));
    }

    #[test]
    fn check_diagnostics_report_source_lines_past_skipped_global_comments() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::Beam]),
            ..Default::default()
        };
        let report = check_text(
            "!! global comment\n!! another global comment\n**kern\n*M4/4\n8cJ\n4d\n4e\n4f\n=\n*-\n",
            &config,
        );

        assert!(!report.success);
        assert_eq!(report.diagnostics.len(), 1);
        let diagnostic = &report.diagnostics[0];
        assert_eq!(diagnostic.rule, Some(ValidationPass::Beam));
        assert_eq!(diagnostic.line, Some(5));
        assert_eq!(diagnostic.column, Some(0));
    }

    #[test]
    fn check_selected_rhythm_pass_skips_beam_errors() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::Rhythm]),
            ..Default::default()
        };
        let report = check_text(
            "**kern\n*M4/4\n8cJ\n8d\n8e\n8f\n8g\n8a\n8b\n8cc\n=\n*-\n",
            &config,
        );

        assert!(report.success);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn check_selected_state_pass_skips_rhythm_errors() {
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::State]),
            ..Default::default()
        };
        let report = check_text("**kern\n*M4/4\n2c\n2d\n4e\n=\n*-\n", &config);

        assert!(report.success);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn check_text_input_dialect_controls_durationless_invisible_rest_slices() {
        let input =
            "**kern\t**kern\n*M4/4\t*M4/4\n4c\tryy\n4d\tryy\n4e\tryy\n4f\tryy\n=\t=\n*-\t*-\n";

        let strict = check_text(input, &CheckConfig::default());
        assert!(!strict.success);
        assert_eq!(strict.diagnostics[0].phase, CheckPhase::Parse);

        let transcoda_v2 = check_text(
            input,
            &CheckConfig {
                input_dialect: ParseDialect::TranscodaV2,
                ..Default::default()
            },
        );
        assert!(
            transcoda_v2.success,
            "expected Transcoda v2 input to pass, got {:?}",
            transcoda_v2.diagnostics
        );
    }

    fn token_rule() -> CheckConfig {
        CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::Token]),
            ..Default::default()
        }
    }

    #[test]
    fn token_rule_reports_ignored_marks_at_their_span() {
        let report = check_text("**kern\t**kern\n4c 4e)>\t4c;;\n*-\t*-\n", &token_rule());
        let found: Vec<_> = report
            .diagnostics
            .iter()
            .map(|d| {
                (
                    d.rule,
                    d.code.as_str(),
                    d.severity,
                    d.line,
                    d.column,
                    d.span.clone(),
                    d.message.as_str(),
                )
            })
            .collect();
        assert_eq!(
            found,
            [
                (
                    Some(ValidationPass::Token),
                    "token.unattached-signifier",
                    CheckSeverity::Warning,
                    Some(2),
                    Some(0),
                    Some(6..7),
                    "`>` follows slur-close, which takes no placement; the parser ignores it",
                ),
                (
                    Some(ValidationPass::Token),
                    "token.duplicate-mark",
                    CheckSeverity::Warning,
                    Some(2),
                    Some(1),
                    Some(3..4),
                    "second fermata on one head; the parser keeps the first and ignores this one",
                ),
            ]
        );
        assert!(report.success);
    }

    #[test]
    fn token_rule_needs_the_source_text() {
        let document = KernDocument::parse("**kern\n4c)>\n*-\n").unwrap();
        let report = check_document(&document, &token_rule());
        assert!(report.diagnostics.is_empty());
        assert_eq!(
            report.rules[0].status,
            validation::RuleStatus::NotApplicable
        );
    }

    #[test]
    fn a_refused_token_names_its_reason_and_char_span_without_changing_the_error() {
        let parse_error = |input: &str| {
            let report = check_text(input, &CheckConfig::default());
            let d = report.diagnostics[0].clone();
            (d.code, d.message, d.span, d.reason)
        };
        assert_eq!(
            parse_error("**kern\n4c 16r# c##L\n*-\n"),
            (
                "parse.invalid-chord".to_owned(),
                "line 2: invalid spine token \"4c 16r# c##L\" at column 1: invalid chord \"4c 16r# c##L\"".to_owned(),
                Some(3..7),
                Some("token.rest-in-chord".to_owned()),
            )
        );
        assert_eq!(
            parse_error("**kern\n4c 4e\u{2192}\n*-\n"),
            (
                "parse.invalid-note".to_owned(),
                "line 2: invalid spine token \"4c 4e\u{2192}\" at column 1: invalid note \"4e\u{2192}\": character 3 '\u{2192}'".to_owned(),
                Some(5..6),
                Some("token.unknown-character".to_owned()),
            )
        );
        // `r` is a refused **kern token too, but not with this error.
        let (code, _, span, reason) = parse_error("**kern\t**fing\n4c\tr\n*-\t*-\n");
        assert_eq!(
            (code.as_str(), span, reason),
            ("parse.invalid-fingering", None, None)
        );
    }
}
