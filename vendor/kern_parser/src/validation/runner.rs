use super::*;
use crate::{
    analysis::{TieChain, TieChainAnalysisError},
    document::{Document, Framing},
    score::{Score, StateErrors},
    KernDocument, KernLine,
};

#[derive(Clone)]
pub(crate) struct RuleRun {
    pub rule: ValidationPass,
    pub status: RuleStatus,
    pub warnings: Vec<ValidationWarning>,
    pub error: Option<ValidationError>,
    pub tie_chains: Vec<TieChain>,
}

impl RuleRun {
    /// A blocked rule carries the error of the rule that blocked it. That
    /// error belongs in a report under this rule only when the failed rule
    /// is not part of the run, which reports it already.
    pub(crate) fn reports_error(&self, passes: &[ValidationPass]) -> bool {
        match &self.status {
            RuleStatus::Blocked { causes } => !causes
                .iter()
                .all(|cause| passes.iter().any(|pass| pass.id() == cause)),
            _ => true,
        }
    }
}

/// The sole dispatch point for document validation. Adapters decide how to
/// display findings; they do not choose different validation semantics.
/// `source` is the authored text `document` was parsed from, when there is one.
pub(crate) fn run_rules(
    document: &KernDocument,
    source: Option<&Document>,
    config: &ValidationConfig,
    framing: Framing,
) -> Vec<RuleRun> {
    let mut cache = Vec::<RuleRun>::new();
    let score = Score::resolve(document, framing);
    config
        .passes()
        .iter()
        .map(|&rule| cached_rule(document, source, &score, rule, framing, &mut cache))
        .collect()
}

/// The document resolved once per run; rules that need interpretation state
/// list [`ValidationPass::State`] as a prerequisite and read it from here.
type Resolved<'d> = Result<Score<'d>, StateErrors>;

fn cached_rule(
    document: &KernDocument,
    source: Option<&Document>,
    score: &Resolved<'_>,
    rule: ValidationPass,
    framing: Framing,
    cache: &mut Vec<RuleRun>,
) -> RuleRun {
    if let Some(result) = cache.iter().find(|result| result.rule == rule) {
        return result.clone();
    }
    let mut blocked = None;
    let mut causes = Vec::<String>::new();
    for &dependency in rule.definition().prerequisites {
        let result = cached_rule(document, source, score, dependency, framing, cache);
        if result.error.is_some() {
            // Name the rules that failed, not those they blocked in turn.
            let failed = match result.status {
                RuleStatus::Blocked { causes } => causes,
                _ => vec![dependency.id().to_owned()],
            };
            for cause in failed {
                if !causes.contains(&cause) {
                    causes.push(cause);
                }
            }
            if blocked.is_none() {
                blocked = result.error;
            }
        }
    }
    let result = if let Some(error) = blocked {
        RuleRun {
            rule,
            status: RuleStatus::Blocked { causes },
            warnings: Vec::new(),
            error: Some(error),
            tie_chains: Vec::new(),
        }
    } else {
        run_rule(document, source, score, rule, framing)
    };
    cache.push(result.clone());
    result
}

fn run_rule(
    document: &KernDocument,
    source: Option<&Document>,
    score: &Resolved<'_>,
    rule: ValidationPass,
    framing: Framing,
) -> RuleRun {
    let resolved = || {
        score
            .as_ref()
            .expect("rules reading the score are blocked by the state rule")
    };
    let mut result = RuleRun {
        rule,
        status: RuleStatus::Passed,
        warnings: Vec::new(),
        error: None,
        tie_chains: Vec::new(),
    };
    if matches!(
        rule,
        ValidationPass::Ornament
            | ValidationPass::Slur
            | ValidationPass::Tie
            | ValidationPass::CrossStaff
    ) && !document.data_rows().any(|row| {
        row.contents.iter().any(|content| {
            matches!(
                content,
                crate::SpineContent::Kern(
                    crate::KernContent::Note(_)
                        | crate::KernContent::Chord(_)
                        | crate::KernContent::Grace(_)
                )
            )
        })
    }) {
        result.status = RuleStatus::NotApplicable;
        return result;
    }
    result.error = match rule {
        ValidationPass::State => match score {
            Ok(score) => {
                result.warnings = redundant_state(score);
                None
            }
            Err(errors) => Some(errors.clone().into()),
        },
        ValidationPass::ClefBarline => {
            result.warnings = clef_barline::check(document);
            None
        }
        ValidationPass::DurationState => duration_state::validate(resolved()).err().map(Into::into),
        ValidationPass::RestPlacement => match rest_placement::check(resolved()) {
            Ok(warnings) => {
                result.warnings = warnings;
                None
            }
            Err(error) => blocked_by_timing(&mut result, error),
        },
        ValidationPass::FermataStack => {
            result.warnings = fermata_stack::check(resolved());
            None
        }
        ValidationPass::MeterBoundary => meter_boundary::validate(resolved()).err().map(Into::into),
        ValidationPass::Beam => beam::validate(resolved()).err().map(Into::into),
        ValidationPass::Ornament => {
            result.warnings = ornament::check(document);
            None
        }
        ValidationPass::Dynam => {
            result.warnings = dynamics::check(document, framing);
            None
        }
        ValidationPass::CrossStaff => {
            result.warnings = cross_staff::check(resolved());
            None
        }
        ValidationPass::Token => {
            match source {
                Some(source) => result.warnings = token::check(source),
                None => result.status = RuleStatus::NotApplicable,
            }
            None
        }
        ValidationPass::Slur => {
            result.warnings = slur::check(resolved(), framing);
            None
        }
        ValidationPass::Tie => match tie::check(resolved()) {
            Ok(report) => {
                result.warnings = report.warnings;
                result.tie_chains = report.chains;
                None
            }
            Err(TieChainAnalysisError::Timing(error)) => {
                blocked_by_timing(&mut result, error.into())
            }
        },
        ValidationPass::Rhythm => rhythm::validate(resolved()).err().map(Into::into),
        ValidationPass::MediumSinglePage => {
            medium_single_page::validate(document).err().map(Into::into)
        }
    };
    if let Some(ValidationError::MediumSinglePage(
        medium_single_page::MediumSinglePageValidationError::Render(message),
    )) = &result.error
    {
        result.status = RuleStatus::Failed {
            diagnostic: message.clone(),
        };
    } else if !matches!(result.status, RuleStatus::Blocked { .. })
        && (result.error.is_some() || !result.warnings.is_empty())
    {
        result.status = RuleStatus::Findings;
    }
    result
}

/// One warning per tandem that restates the value in force and that Verovio
/// draws nothing for; normalization removes these.
fn redundant_state(score: &Score<'_>) -> Vec<ValidationWarning> {
    let document = score.document();
    score
        .events()
        .iter()
        .filter(|event| event.outcome == crate::score::Outcome::Redundant)
        .map(|event| {
            let line = document.source_line_number(event.line_index);
            let column = event.column + 1;
            let KernLine::TandemInterpretationLine(tandems) = &document.lines()[event.line_index]
            else {
                unreachable!("state events come from tandem lines")
            };
            let written = crate::serialize::omr::spell_tandem(
                &tandems.tokens()[event.column],
                crate::serialize::omr::VisibilityMarkerStyle::Preserve,
            )
            .unwrap_or_default();
            ValidationWarning {
                code: "state.redundant",
                severity: ValidationSeverity::Warning,
                location: SourceLocation {
                    line,
                    column: Some(column),
                },
                message: format!(
                    "line {line} column {column}: {written} restates the {} already in force",
                    event.key.name()
                ),
                loose_end: None,
                span: None,
            }
        })
        .collect()
}

/// Rules that time lines themselves list `duration-state` as a prerequisite,
/// which runs the same timing first; a timing error here is reported as that
/// rule's.
fn blocked_by_timing(
    result: &mut RuleRun,
    error: duration_state::DurationStateValidationError,
) -> Option<ValidationError> {
    result.status = RuleStatus::Blocked {
        causes: vec![ValidationPass::DurationState.id().into()],
    };
    Some(error.into())
}
