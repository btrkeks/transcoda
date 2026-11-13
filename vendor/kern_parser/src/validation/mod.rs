//! Document-level validators for parsed Humdrum records.

use std::{fmt, ops::Range};

use serde::Serialize;
use thiserror::Error;

mod runner;
pub(crate) use runner::run_rules;

use crate::{document::Framing, KernDocument};

pub mod beam;
pub mod clef_barline;
mod cross_staff;
pub mod duration_state;
mod dynamics;
mod fermata_stack;
pub mod medium_single_page;
pub mod meter_boundary;
pub mod ornament;
pub mod quality_filter;
pub mod rest_placement;
pub mod rhythm;
pub mod slur;
pub mod tie;
mod token;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ValidationPass {
    State,
    ClefBarline,
    CrossStaff,
    DurationState,
    RestPlacement,
    FermataStack,
    MeterBoundary,
    Beam,
    Rhythm,
    Ornament,
    Slur,
    Tie,
    Token,
    Dynam,
    MediumSinglePage,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct RuleDefinition {
    pub id: &'static str,
    pub rule: ValidationPass,
    pub default: bool,
    pub description: &'static str,
    pub prerequisites: &'static [ValidationPass],
}

macro_rules! rules {
    ($(($variant:ident, $id:literal, $default:literal, $description:literal, $prerequisites:expr)),* $(,)?) => {
        pub const RULES: &[RuleDefinition] = &[$(RuleDefinition {
            id: $id, rule: ValidationPass::$variant, default: $default, description: $description, prerequisites: $prerequisites,
        }),*];
        impl ValidationPass {
            pub const fn id(self) -> &'static str { match self { $(Self::$variant => $id),* } }
            pub fn definition(self) -> &'static RuleDefinition { RULES.iter().find(|rule| rule.rule == self).expect("registered validation rule") }
            pub const fn code(self) -> ValidationCode { match self { $(Self::$variant => ValidationCode::$variant),* } }
            pub fn from_id(id: &str) -> Option<Self> { RULES.iter().find(|rule| rule.id == id).map(|rule| rule.rule) }
        }
    }
}
rules! {
    (State, "state", true, "Spine topology and interpretation state", &[]),
    (ClefBarline, "clef-barline", true, "Clef changes placed after a barline instead of before it", &[]),
    (CrossStaff, "cross-staff", true, "Pitched notes moved beyond the outermost score staff", &[ValidationPass::State]),
    (DurationState, "duration-state", true, "Overlapping and incomplete duration state", &[ValidationPass::State]),
    (RestPlacement, "rest-placement", true, "Rests Verovio cannot place among three or more voices", &[ValidationPass::State, ValidationPass::DurationState]),
    (FermataStack, "fermata-stack", true, "Simultaneous fermatas Verovio draws on the same side of one staff", &[ValidationPass::State]),
    (MeterBoundary, "meter-boundary", true, "Meter changes at measure boundaries", &[ValidationPass::State]),
    (Beam, "beam", true, "Beam balance and boundaries", &[ValidationPass::State]),
    (Ornament, "ornament", true, "Underspecified ornament spelling", &[]),
    (Slur, "slur", true, "Unpaired and ineffective slurs", &[ValidationPass::State]),
    (Tie, "tie", true, "Tie chains and unmatched tie markers", &[ValidationPass::State, ValidationPass::DurationState]),
    (Rhythm, "rhythm", true, "Local measure rhythm, allowing page fragments", &[ValidationPass::State]),
    (Token, "token", true, "Signifiers and repeated marks the parser leaves out of the AST", &[]),
    (Dynam, "dynam", true, "**dynam hairpins Verovio does not draw and conflicting tokens", &[]),
    (MediumSinglePage, "medium-single-page", false, "A single page at medium rendering size", &[]),
}

impl fmt::Display for ValidationPass {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.id())
    }
}
impl std::str::FromStr for ValidationPass {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::from_id(value).ok_or_else(|| {
            format!(
                "unknown rule `{value}`; expected {}",
                RULES
                    .iter()
                    .map(|rule| rule.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }
}
pub fn rules() -> &'static [RuleDefinition] {
    RULES
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum RuleStatus {
    Passed,
    Findings,
    NotApplicable,
    Blocked { causes: Vec<String> },
    Failed { diagnostic: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuleOutcome {
    pub rule: ValidationPass,
    #[serde(flatten)]
    pub status: RuleStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationConfig {
    passes: Vec<ValidationPass>,
}

impl ValidationConfig {
    pub fn default_passes() -> Self {
        Self {
            passes: RULES
                .iter()
                .filter(|rule| rule.default)
                .map(|rule| rule.rule)
                .collect(),
        }
    }

    pub fn with_passes(passes: Vec<ValidationPass>) -> Self {
        assert!(
            !passes.is_empty(),
            "validation config requires at least one pass"
        );
        let mut unique = Vec::new();
        for pass in passes {
            if !unique.contains(&pass) {
                unique.push(pass);
            }
        }
        Self { passes: unique }
    }

    pub fn passes(&self) -> &[ValidationPass] {
        &self.passes
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self::default_passes()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ValidationReport {
    pub findings: Vec<ValidationFinding>,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.findings.is_empty()
    }

    pub fn has_errors(&self) -> bool {
        self.findings
            .iter()
            .any(|finding| finding.severity == ValidationSeverity::Error)
    }

    pub fn push_error(&mut self, code: ValidationCode, message: impl Into<String>) {
        self.findings.push(ValidationFinding {
            code,
            severity: ValidationSeverity::Error,
            message: message.into(),
        });
    }

    pub fn push_warning(&mut self, code: ValidationCode, message: impl Into<String>) {
        self.findings.push(ValidationFinding {
            code,
            severity: ValidationSeverity::Warning,
            message: message.into(),
        });
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.findings.is_empty() {
            return write!(formatter, "validation report contains no findings");
        }

        let message = self
            .findings
            .iter()
            .map(|finding| format!("{:?}: {}", finding.code, finding.message))
            .collect::<Vec<_>>()
            .join("; ");
        formatter.write_str(&message)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationFinding {
    pub code: ValidationCode,
    pub severity: ValidationSeverity,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SourceLocation {
    pub line: usize,
    pub column: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidationWarning {
    pub code: &'static str,
    pub severity: ValidationSeverity,
    pub location: SourceLocation,
    pub message: String,
    pub loose_end: Option<crate::analysis::LooseEnd>,
    /// Zero-based char offsets in the field.
    pub span: Option<Range<usize>>,
}

impl fmt::Display for ValidationWarning {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationCode {
    Beam,
    ClefBarline,
    CrossStaff,
    DurationState,
    Dynam,
    EmptyModelTarget,
    FermataStack,
    MeterBoundary,
    MediumSinglePage,
    NoSerializableKernContent,
    Ornament,
    Rhythm,
    RestPlacement,
    Slur,
    Tie,
    State,
    Token,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ValidationError {
    #[error(transparent)]
    Beam(#[from] beam::BeamValidationError),
    #[error(transparent)]
    DurationState(#[from] duration_state::DurationStateValidationError),
    #[error(transparent)]
    MeterBoundary(#[from] meter_boundary::MeterBoundaryValidationError),
    #[error(transparent)]
    MediumSinglePage(#[from] medium_single_page::MediumSinglePageValidationError),
    #[error("{0}")]
    OmrTraining(ValidationReport),
    #[error(transparent)]
    Rhythm(#[from] rhythm::RhythmValidationError),
    #[error(transparent)]
    State(#[from] crate::score::StateErrors),
}

impl ValidationError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Beam(_) => "BeamValidationError",
            Self::DurationState(_) => "DurationStateValidationError",
            Self::MeterBoundary(_) => "MeterBoundaryValidationError",
            Self::MediumSinglePage(_) => "MediumSinglePageValidationError",
            Self::OmrTraining(_) => "OmrTrainingValidationError",
            Self::Rhythm(_) => "RhythmValidationError",
            Self::State(_) => "StateValidationError",
        }
    }
}

pub fn validate(document: &KernDocument) -> Result<(), ValidationError> {
    validate_with_config(document, &ValidationConfig::default())
}

pub fn validate_with_config(
    document: &KernDocument,
    config: &ValidationConfig,
) -> Result<(), ValidationError> {
    for result in run_rules(document, None, config, Framing::Excerpt) {
        if let Some(error) = result.error {
            return Err(error);
        }
    }
    Ok(())
}

pub fn validate_report(document: &KernDocument) -> ValidationReport {
    validate_report_with_config(document, &ValidationConfig::default())
}

pub fn validate_report_with_config(
    document: &KernDocument,
    config: &ValidationConfig,
) -> ValidationReport {
    let mut report = ValidationReport::new();
    for result in run_rules(document, None, config, Framing::Excerpt) {
        let reports_error = result.reports_error(config.passes());
        for warning in result.warnings {
            match warning.severity {
                ValidationSeverity::Error => report.push_error(result.rule.code(), warning.message),
                ValidationSeverity::Warning => {
                    report.push_warning(result.rule.code(), warning.message)
                }
            }
        }
        if let Some(error) = result.error.filter(|_| reports_error) {
            report.push_error(result.rule.code(), error.to_string());
        }
    }
    report
}

/// Resolve `document` as an excerpt and check its measure rhythm.
pub fn validate_rhythm(document: &KernDocument) -> Result<(), ValidationError> {
    let score = crate::score::Score::resolve(document, Framing::Excerpt)?;
    rhythm::validate(&score)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> KernDocument {
        KernDocument::parse(input).unwrap()
    }

    fn invalid_beam_complete_measure() -> &'static str {
        "**kern\n*M4/4\n8cJ\n8d\n8e\n8f\n8g\n8a\n8b\n8cc\n=\n*-\n"
    }

    /// The ottava is open where its track ends before the document does.
    fn unclosed_ottava() -> &'static str {
        "**kern\t**kern\n*8va\t*\n4c\t4e\n*-\t*\n4f\n*-\n"
    }

    #[test]
    fn default_validation_rejects_invalid_beams() {
        let document = parse(invalid_beam_complete_measure());

        let error = validate(&document).unwrap_err();

        assert!(matches!(error, ValidationError::Beam(_)));
        assert!(error.to_string().contains("closes before it is open"));
    }

    #[test]
    fn selected_beam_pass_rejects_beam_errors() {
        let document = parse(invalid_beam_complete_measure());
        let config = ValidationConfig::with_passes(vec![ValidationPass::Beam]);

        let report = validate_report_with_config(&document, &config);

        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].code, ValidationCode::Beam);
    }

    #[test]
    fn default_validation_rejects_unclosed_ottava() {
        let document = parse(unclosed_ottava());

        let error = validate(&document).unwrap_err();

        assert!(matches!(error, ValidationError::State(_)));
        assert!(error.to_string().contains("ottava opened at line"));
    }

    #[test]
    fn selected_state_pass_reports_unclosed_ottava() {
        let document = parse(unclosed_ottava());
        let config = ValidationConfig::with_passes(vec![ValidationPass::State]);

        let report = validate_report_with_config(&document, &config);

        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].code, ValidationCode::State);
    }

    #[test]
    fn selected_rule_is_blocked_by_state_errors() {
        let document = parse(unclosed_ottava());
        let config = ValidationConfig::with_passes(vec![ValidationPass::Rhythm]);

        let error = validate_with_config(&document, &config).unwrap_err();

        assert!(matches!(error, ValidationError::State(_)));
    }

    #[test]
    fn selected_rhythm_pass_skips_beam_errors() {
        let document = parse(invalid_beam_complete_measure());
        let config = ValidationConfig::with_passes(vec![ValidationPass::Rhythm]);

        validate_with_config(&document, &config).unwrap();
    }

    #[test]
    fn selected_slur_pass_allows_same_pitch_slur() {
        let document = parse("**kern\n(4c\n4c)\n*-\n");
        let config = ValidationConfig::with_passes(vec![ValidationPass::Slur]);

        validate_with_config(&document, &config).unwrap();
        let report = validate_report_with_config(&document, &config);

        assert!(report.findings.is_empty());
    }

    #[test]
    fn selected_slur_pass_reports_dangling_slur_warning() {
        let document = parse("**kern\n(4c\n*-\n");
        let config = ValidationConfig::with_passes(vec![ValidationPass::Slur]);

        validate_with_config(&document, &config).unwrap();
        let report = validate_report_with_config(&document, &config);

        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].code, ValidationCode::Slur);
        assert_eq!(report.findings[0].severity, ValidationSeverity::Warning);
        assert!(report.findings[0].message.contains("never closed"));
    }
}
