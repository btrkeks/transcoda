//! Rhythm validation for typed `**kern` documents.
//!
//! This pass is a dataset quality filter for obvious local rhythm defects:
//! overfull measures, ordinary underfull internal measures, inconsistent
//! pickup handling across active `**kern` spines, and invalid beam structure.
//! It intentionally does not try to prove global musical repeat structure.
//! Page-level OMR samples may contain a back-repeat whose target is on a
//! previous page, so underfull measures at repeat-end barlines are accepted.

use std::collections::BTreeSet;

use thiserror::Error;

use crate::{
    analysis::initial_time_signature::resolve_initial_time_signature_in,
    measure::{MeasureDuration, MeasureLocation},
    normalization::invisible_rest_slices::{
        contains_durationless_invisible_rest_slices, rehydrate_durationless_invisible_rest_slices,
    },
    score::Score,
    BarLength, KernDocument, KernLine, TandemInterpretation,
};

mod boundary;
mod state;

use state::MeasureStateTracker;

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum RhythmValidationError {
    #[error("invalid rhythm structure")]
    Invalid,
    #[error("invalid measure structure: {}", located_message(.line, .column, .message))]
    InvalidMeasureStructure {
        code: &'static str,
        line: Option<usize>,
        column: Option<usize>,
        message: String,
    },
}

impl RhythmValidationError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Invalid => "rhythm.invalid",
            Self::InvalidMeasureStructure { code, .. } => code,
        }
    }

    pub fn line(&self) -> Option<usize> {
        match self {
            Self::Invalid => None,
            Self::InvalidMeasureStructure { line, .. } => *line,
        }
    }

    pub fn column(&self) -> Option<usize> {
        match self {
            Self::Invalid => None,
            Self::InvalidMeasureStructure { column, .. } => *column,
        }
    }

    /// Message without the embedded source location, for diagnostics that
    /// report the location in dedicated line/column fields.
    pub fn message_without_location(&self) -> String {
        match self {
            Self::Invalid => self.to_string(),
            Self::InvalidMeasureStructure { message, .. } => {
                format!("invalid measure structure: {message}")
            }
        }
    }
}

fn located_message(line: &Option<usize>, column: &Option<usize>, message: &str) -> String {
    match (line, column) {
        (Some(line), Some(column)) => format!("line {line}, column {column}: {message}"),
        (Some(line), None) => format!("line {line}: {message}"),
        (None, _) => message.to_string(),
    }
}

/// Filters locally rhythmically broken pieces from the dataset so the model
/// learns better measure structure without rejecting valid page-level repeat
/// fragments whose full musical context is outside the sample.
///
/// Meter is track state: it comes from the score, and measure positions are
/// kept per layer.
pub fn validate(score: &Score<'_>) -> Result<(), RhythmValidationError> {
    walk(score).result
}

/// Line indexes of barlines that divide a bar instead of ending it: a dashed
/// barline (`=:`) or a system-break `=-` reached before the bar is full. The
/// reading stops at the first rhythm error, so barlines after one are absent.
pub fn bar_dividing_barlines(score: &Score<'_>) -> BTreeSet<usize> {
    walk(score).dividing_barlines
}

struct Walk {
    result: Result<(), RhythmValidationError>,
    dividing_barlines: BTreeSet<usize>,
}

impl From<RhythmValidationError> for Walk {
    fn from(error: RhythmValidationError) -> Self {
        Self {
            result: Err(error),
            dividing_barlines: BTreeSet::new(),
        }
    }
}

fn walk(score: &Score<'_>) -> Walk {
    let document = score.document();
    if contains_durationless_invisible_rest_slices(document) {
        let mut document = document.clone();
        if let Err(source) = rehydrate_durationless_invisible_rest_slices(&mut document) {
            return RhythmValidationError::InvalidMeasureStructure {
                code: "rhythm.invalid-rest-slices",
                line: None,
                column: None,
                message: source.to_string(),
            }
            .into();
        }
        return walk_measures(score, &document);
    }

    walk_measures(score, document)
}

/// `document` is the score's own document, or a copy whose rehydrated rest
/// slices changed data tokens only, so the score's layout still applies.
fn walk_measures(score: &Score<'_>, document: &KernDocument) -> Walk {
    if !document
        .lines()
        .iter()
        .any(|line| matches!(line, KernLine::Data(_)))
    {
        return Walk {
            result: Ok(()),
            dividing_barlines: BTreeSet::new(),
        };
    }

    let initial_meter = if declares_initial_free_meter(document) {
        BarLength::Free
    } else {
        match resolve_initial_time_signature_in(score, document) {
            Ok(time_signature) => {
                BarLength::Fixed(MeasureDuration::from_time_signature(&time_signature))
            }
            Err(error) => {
                return RhythmValidationError::InvalidMeasureStructure {
                    code: "rhythm.initial-meter-unavailable",
                    line: None,
                    column: None,
                    message: error.to_string(),
                }
                .into()
            }
        }
    };
    let mut tracker = MeasureStateTracker::new(score, initial_meter);
    let result = (0..document.lines().len())
        .try_for_each(|line_index| tracker.apply_line(document.lines(), line_index))
        .and_then(|()| tracker.finish_document(document.lines().len() + 1));
    Walk {
        result,
        dividing_barlines: tracker.dividing_barlines().clone(),
    }
}

/// An initial `*MX` declares that bars have no fixed length, so there is no
/// initial meter to declare or infer.
fn declares_initial_free_meter(document: &KernDocument) -> bool {
    document.lines()[document.leading_tandem_header_range()]
        .iter()
        .any(|line| {
            matches!(
                line,
                KernLine::TandemInterpretationLine(tandems)
                    if tandems.tokens().contains(&TandemInterpretation::FreeMeter)
            )
        })
}

fn measure_error_at<T>(
    location: MeasureLocation,
    code: &'static str,
    message: impl Into<String>,
) -> Result<T, RhythmValidationError> {
    Err(measure_error_value(location, code, message))
}

fn measure_error_value(
    location: MeasureLocation,
    code: &'static str,
    message: impl Into<String>,
) -> RhythmValidationError {
    RhythmValidationError::InvalidMeasureStructure {
        code,
        line: Some(location.source_line),
        column: location.column,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        types::{Chord, ChordBeams, DurationValue},
        validation::beam,
        Duration, KernContent, KernDocument, ParseDialect, ParseOptions, SpineContent,
    };

    fn check(document: &KernDocument) -> Result<(), RhythmValidationError> {
        validate(&Score::resolve(document, crate::document::Framing::Excerpt).unwrap())
    }

    fn assert_valid(input: &str) {
        let document = KernDocument::parse(input).unwrap();
        check(&document).unwrap();
    }

    fn assert_invalid(input: &str, expected: &str) {
        let document = KernDocument::parse(input).unwrap();
        let error = check(&document).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "expected {error:?} to contain {expected:?}"
        );
    }

    fn assert_invalid_beams(input: &str, expected: &str) {
        let document = KernDocument::parse(input).unwrap();
        let error = beam::validate_document(&document).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "expected {error:?} to contain {expected:?}"
        );
    }

    fn assert_valid_transcoda_v2(input: &str) {
        let document = KernDocument::parse_with_options(
            input,
            ParseOptions {
                dialect: ParseDialect::TranscodaV2,
            },
        )
        .unwrap();
        check(&document).unwrap();
    }

    #[test]
    fn validates_simple_eighth_beam_group() {
        assert_valid("**kern\n*M4/4\n8cL\n8dJ\n*-\n");
    }

    #[test]
    fn validates_sixteenth_beam_group() {
        assert_valid("**kern\n*M4/4\n16cLL\n16d\n16e\n16fJJ\n*-\n");
    }

    #[test]
    fn validates_partial_beam_pair() {
        assert_valid("**kern\n*M4/4\n8.cL\n16dJk\n*-\n");
    }

    #[test]
    fn validates_parallel_spines_independently() {
        assert_valid(
            "**kern\t**kern\n*M4/4\t*M4/4\n8cL\t16eLL\n8dJ\t16f\n.\t16g\n.\t16aJJ\n*-\t*-\n",
        );
    }

    #[test]
    fn validates_transcoda_v2_durationless_invisible_rest_slices() {
        assert_valid_transcoda_v2(
            "**kern\t**kern\n*M4/4\t*M4/4\n4c\tryy\n4d\tryy\n4e\tryy\n4f\tryy\n=\t=\n*-\t*-\n",
        );
    }

    #[test]
    fn validates_rest_inside_active_beam() {
        assert_valid("**kern\n*M4/4\n8cL\n16r\n16dJk\n*-\n");
    }

    #[test]
    fn validates_complete_four_four_measure() {
        assert_valid("**kern\n*M4/4\n4c\n4d\n4e\n4f\n=\n*-\n");
    }

    #[test]
    fn meter_symbol_updates_meter_after_numeric_meter() {
        assert_valid("**kern\n*M2/4\n4c\n=\n*met(c)\n4d\n4e\n4f\n4g\n=\n*-\n");
    }

    #[test]
    fn cut_meter_symbol_updates_meter_after_numeric_meter() {
        assert_valid("**kern\n*M4/4\n1c\n=\n*met(c|)\n2d\n=\n*-\n");
    }

    #[test]
    fn numeric_meter_after_meter_symbol_group_updates_meter() {
        assert_valid("**kern\n*M4/4\n*met(c)\n1c\n=\n*M3/4\n2.d\n=\n*-\n");
    }

    #[test]
    fn rejects_mid_piece_meter_group_with_conflicting_meter_symbol() {
        for group in ["*M3/4\n*met(c)", "*met(c)\n*M3/4"] {
            assert_invalid(
                &format!("**kern\n*M4/4\n1c\n=\n{group}\n2.d\n=\n*-\n"),
                "time signature 3/4 conflicts with meter symbol implying 4/4",
            );
        }
    }

    #[test]
    fn accepts_mid_piece_meter_group_with_matching_meter_symbol() {
        assert_valid("**kern\n*M4/4\n1c\n=\n*M2/2\n*met(c|)\n1d\n=\n*-\n");
    }

    #[test]
    fn hidden_time_signatures_keep_their_meter() {
        assert_valid("**kern\n*M3/4\n*met()\n2.c\n=\n*-\n");
        assert_valid("**kern\n*M3/4yy\n2.c\n=\n*-\n");
    }

    #[test]
    fn hide_only_meter_group_keeps_the_meter_in_force() {
        assert_valid("**kern\n*M3/4\n2.c\n=\n*met()\n2.d\n=\n*-\n");
    }

    #[test]
    fn hide_only_meter_group_without_a_meter_does_not_panic() {
        assert_valid("**kern\n*met()\n4c\n4d\n4e\n4f\n=\n*-\n");
    }

    #[test]
    fn initial_free_meter_accepts_bars_of_any_length() {
        assert_valid("**kern\n*MX\n8c\n=1\n8.a\n16a\n16a\n16a\n=2\n4g\n8r\n=3\n2.c\n==\n*-\n");
    }

    #[test]
    fn free_meter_bar_returns_to_the_declared_meter() {
        assert_valid("**kern\n*M4/4\n1c\n=\n*MX\n2.d\n16e\n=\n*M4/4\n*met()\n1f\n==\n*-\n");
        assert_invalid(
            "**kern\n*M4/4\n1c\n=\n*MX\n2.d\n=\n*M4/4\n2f\n=\n1g\n==\n*-\n",
            "measure duration does not match meter",
        );
    }

    #[test]
    fn free_meter_bar_requires_equal_voice_lengths() {
        assert_valid("**kern\t**kern\n*MX\t*MX\n2.c\t4e\n.\t2f\n=\t=\n*-\t*-\n");
        assert_invalid(
            "**kern\t**kern\n*MX\t*MX\n2.c\t2e\n=\t=\n*-\t*-\n",
            "voices in a free-meter (*MX) measure differ in length",
        );
    }

    #[test]
    fn free_meter_rejects_multi_measure_rests() {
        assert_invalid(
            "**kern\n*MX\nMM2r\n=\n*-\n",
            "multi-measure rest has no measure length under free meter",
        );
    }

    #[test]
    fn infers_missing_initial_meter_from_second_measure() {
        assert_valid("**kern\n4c\n=\n4d\n4e\n4f\n=\n4g\n4a\n4b\n=\n*-\n");
    }

    #[test]
    fn infers_missing_initial_meter_from_only_measure() {
        assert_valid("**kern\n4c\n4d\n4e\n4f\n=\n*-\n");
    }

    #[test]
    fn inferred_missing_initial_meter_rejects_legacy_four_four_assumption() {
        assert_invalid(
            "**kern\n4c\n=\n4d\n4e\n4f\n=\n4g\n4a\n4b\n4cc\n=\n*-\n",
            "measure duration exceeds meter",
        );
    }

    #[test]
    fn validates_consistent_pickup_across_parallel_spines() {
        assert_valid("**kern\t**kern\n*M4/4\t*M4/4\n4c\t4e\n=\t=\n2d\t2f\n4e\t4g\n=\t=\n*-\t*-\n");
    }

    #[test]
    fn rejects_pickup_in_one_spine_against_complete_parallel_measure() {
        assert_invalid(
            "**kern\t**kern\n\
*clefF4\t*clefG2\n\
*k[b-e-]\t*k[b-e-]\n\
*M4/4\t*M4/4\n\
8B-L\t.\n\
8f\t.\n\
8d\t.\n\
8fJ\t.\n\
8B-L\t4dd\n\
8f\t.\n\
8d\t4ff\n\
8fJ\t.\n\
=\t=\n\
8B-L\t2ff\n\
8f\t.\n\
8d\t.\n\
8fJ\t.\n\
8B-L\t8ffL\n\
8f\t8ff\n\
8d\t8ff\n\
8fJ\t8ffJ\n\
=\t=\n\
8B-L\t4ff\n\
8f\t.\n\
8d\t4bb-\n\
8fJ\t.\n\
8B-L\t4.ddd\n\
8f\t.\n\
8d\t.\n\
8fJ\t8bb-\n\
=\t=\n",
            "pickup duration is inconsistent across kern spines",
        );
    }

    #[test]
    fn validates_multi_measure_rest_at_measure_boundary() {
        assert_valid("**kern\n*M3/4\nMM2r\n=\n*-\n");
    }

    #[test]
    fn rejects_content_after_multi_measure_rest_before_barline() {
        assert_invalid(
            "**kern\n*M3/4\nMM2r\n4c\n4d\n4e\n=\n*-\n",
            "measure duration exceeds meter",
        );
    }

    #[test]
    fn rejects_multi_measure_rest_inside_partial_measure() {
        assert_invalid(
            "**kern\n*M4/4\n4c\nMM2r\n=\n*-\n",
            "multi-measure rest must start at a measure boundary",
        );
    }

    #[test]
    fn rejects_underfull_internal_measure() {
        assert_invalid(
            "**kern\n*M4/4\n4c\n4d\n4e\n4f\n=\n4g\n4a\n=\n4b\n4cc\n4dd\n4ee\n=\n*-\n",
            "measure duration does not match meter",
        );
    }

    #[test]
    fn marked_invisible_mid_measure_break_preserves_duration_until_real_barline() {
        for marker in ["!!LO:LB", "!!LO:LB:g=original"] {
            assert_valid(&format!(
                "**kern\n*M4/4\n1c\n=\n2d\n{marker}\n=-\n2e\n=\n1f\n=\n*-\n"
            ));
        }
    }

    #[test]
    fn marked_invisible_break_at_measure_boundary_resets_normally() {
        assert_valid("**kern\n*M4/4\n1c\n!!LO:LB:g=original\n=-\n1d\n=\n*-\n");
    }

    #[test]
    fn marked_invisible_mid_measure_break_is_global_across_kern_spines() {
        assert_valid(
            "**kern\t**kern\n\
*M4/4\t*M4/4\n\
1c\t2e\n\
!!LO:LB:g=original\n\
=-\t=-\n\
.\t2f\n\
=\t=\n\
*-\t*-\n",
        );
    }

    #[test]
    fn marked_invisible_break_defers_underfill_to_next_real_barline() {
        let document = KernDocument::parse(
            "**kern\n*M4/4\n1c\n=\n2d\n!!LO:LB:g=original\n=-\n4e\n=\n1f\n=\n*-\n",
        )
        .unwrap();
        let error = check(&document).unwrap_err();

        assert_eq!(error.line(), Some(9));
        assert_eq!(error.column(), Some(1));
        assert!(error
            .to_string()
            .contains("measure duration does not match meter"));
    }

    #[test]
    fn marked_invisible_break_reports_overfill_at_contributing_data() {
        let document = KernDocument::parse(
            "**kern\n*M4/4\n1c\n=\n2d\n!!LO:LB:g=original\n=-\n2e\n4f\n=\n*-\n",
        )
        .unwrap();
        let error = check(&document).unwrap_err();

        assert_eq!(error.line(), Some(9));
        assert_eq!(error.column(), Some(1));
        assert!(error.to_string().contains("measure duration exceeds meter"));
    }

    #[test]
    fn marked_invisible_break_keeps_incomplete_tail_validation_active() {
        let document =
            KernDocument::parse("**kern\n*M4/4\n1c\n=\n2d\n!!LO:LB:g=original\n=-\n4e\n*-\n")
                .unwrap();
        let error = check(&document).unwrap_err();

        assert_eq!(error.line(), Some(9));
        assert_eq!(error.column(), Some(1));
        assert!(error
            .to_string()
            .contains("spine terminates with incomplete measure"));
    }

    #[test]
    fn dashed_barline_inside_a_bar_divides_it() {
        // Ekier's B. 49, bar 32: a [5/4] bar split by a dashed line into
        // 2/4 + 3/4.
        assert_valid("**kern\n*M5/4\n2c\n=:\n2.d\n=\n*-\n");
    }

    #[test]
    fn dashed_barline_at_a_full_bar_ends_it() {
        assert_valid("**kern\n*M4/4\n1c\n=:\n1d\n=\n*-\n");
    }

    #[test]
    fn dashed_barline_defers_underfill_to_the_next_barline() {
        let document =
            KernDocument::parse("**kern\n*M4/4\n1c\n=\n2d\n=:\n4e\n=\n1f\n=\n*-\n").unwrap();
        let error = check(&document).unwrap_err();

        assert_eq!(error.line(), Some(8));
        assert!(error
            .to_string()
            .contains("measure duration does not match meter"));
    }

    #[test]
    fn bar_dividing_barlines_lists_dashed_and_marked_breaks_inside_bars() {
        let document = KernDocument::parse(
            "**kern\n*M4/4\n1c\n=\n2d\n=:\n2e\n=\n2f\n!!LO:LB:g=original\n=-\n2g\n=:\n1a\n=\n*-\n",
        )
        .unwrap();
        let score = Score::resolve(&document, crate::document::Framing::Excerpt).unwrap();
        let lines: Vec<usize> = bar_dividing_barlines(&score)
            .into_iter()
            .map(|index| document.source_line_number(index))
            .collect();

        assert_eq!(lines, vec![6, 11]);
    }

    #[test]
    fn unmarked_invisible_internal_break_stays_strict_with_system_break_hint() {
        let document = KernDocument::parse("**kern\n*M4/4\n1c\n=\n2d\n=-\n2e\n=\n*-\n").unwrap();
        let error = check(&document).unwrap_err();

        assert_eq!(error.line(), Some(6));
        assert_eq!(error.column(), Some(1));
        assert!(error
            .message_without_location()
            .contains("add `!!LO:LB:g=original` before the `=-`"));
    }

    #[test]
    fn underfull_internal_measure_error_reports_structured_location_and_hint() {
        let document = KernDocument::parse(
            "**kern\n*M4/4\n4c\n4d\n4e\n4f\n=\n4g\n4a\n=\n4b\n4cc\n4dd\n4ee\n=\n*-\n",
        )
        .unwrap();
        let error = check(&document).unwrap_err();

        assert_eq!(error.line(), Some(10));
        assert_eq!(error.column(), Some(1));
        assert!(error.to_string().contains("line 10, column 1"));
        assert!(!error.message_without_location().contains("line 10"));
        assert!(error
            .message_without_location()
            .contains("mark the barline as a section boundary with `=||`"));
    }

    #[test]
    fn validates_underfull_measure_before_internal_thin_piece_boundary() {
        assert_valid("**kern\n*M4/4\n1c\n=\n4d\n4e\n=||\n*k[f#]\n*M3/4\n4e\n4f\n4g\n=\n*-\n");
    }

    #[test]
    fn validates_underfull_measure_before_internal_double_piece_boundary() {
        assert_valid("**kern\n*M4/4\n1c\n=\n4d\n4e\n==\n*k[f#]\n*M3/4\n4e\n4f\n4g\n=\n*-\n");
    }

    #[test]
    fn validates_pickup_after_internal_thin_piece_boundary() {
        assert_valid("**kern\n*M4/4\n1c\n=||\n4d\n=\n4e\n4f\n4g\n=\n*-\n");
    }

    #[test]
    fn validates_pickup_after_internal_double_piece_boundary() {
        assert_valid("**kern\n*M4/4\n1c\n==\n4d\n=\n4e\n4f\n4g\n=\n*-\n");
    }

    #[test]
    fn internal_piece_boundary_resets_pickup_state_for_next_section() {
        assert_valid("**kern\n*M4/4\n4c\n=\n4d\n4e\n4f\n=||\n8g\n=\n4a\n4b\n4cc\n8dd\n=||\n*-\n");
    }

    #[test]
    fn rejects_underfull_measure_before_internal_basic_barline() {
        assert_invalid(
            "**kern\n*M4/4\n1c\n=\n4d\n4e\n=\n4f\n4g\n4a\n4b\n=\n*-\n",
            "measure duration does not match meter",
        );
    }

    #[test]
    fn validates_underfull_measure_before_internal_repeat_end_barline() {
        assert_valid("**kern\n*M4/4\n1c\n=\n4d\n4e\n=:|!\n4f\n4g\n4a\n4b\n=\n*-\n");
    }

    #[test]
    fn validates_underfull_measure_before_repeat_start_barline() {
        assert_valid(
            "**kern\n*M3/8\n8c\n16d\n=\n8e\n16f\n16g\n16a\n16b\n=!|:\n16cc\n=\n8dd\n16ee\n=\n*-\n",
        );
    }

    #[test]
    fn validates_parallel_underfull_measure_before_repeat_start_barline() {
        assert_valid("**kern\t**kern\n*M3/8\t*M3/8\n8c\t8e\n16d\t16f\n=\t=\n8e\t8g\n16f\t16a\n16g\t16b\n16a\t16cc\n16b\t16dd\n=!|:\t=!|:\n16cc\t16ee\n=\t=\n8dd\t8ff\n16ee\t16gg\n=\t=\n*-\t*-\n");
    }

    #[test]
    fn validates_repeat_end_complementing_implicit_pickup_then_full_repeat_end() {
        assert_valid("**kern\n*M4/4\n4c\n=\n4d\n4e\n4f\n=:|:\n1g\n=:|!\n*-\n");
    }

    #[test]
    fn validates_plain_repeat_end_linear_continuation_after_pickup() {
        assert_valid("**kern\n*M6/8\n8c\n=\n4d\n8e\n4f\n=:|!\n8g\n=\n*-\n");
    }

    #[test]
    fn validates_full_measure_after_complementing_repeat_end() {
        assert_valid("**kern\n*M6/8\n8c\n=\n4d\n8e\n4f\n=:|!\n4g\n8a\n4b\n8cc\n=\n*-\n");
    }

    #[test]
    fn validates_full_repeat_end_for_downbeat_section_after_pickup_section() {
        // Pickup section ends complementing the pickup at the first back
        // repeat; the next section starts and ends with full measures.
        assert_valid(
            "**kern\n*M3/8\n8c\n=\n8d\n8e\n8f\n=\n4g\n=:|!\n8a\n8b\n8cc\n=\n8d\n8e\n8f\n=:|!\n8g\n8a\n8b\n=\n*-\n",
        );
    }

    #[test]
    fn validates_new_pickup_fragment_after_complementing_repeat_end() {
        assert_valid(
            "**kern\n*M3/8\n8c\n=\n8d\n8e\n8f\n=\n4g\n=:|!\n8a\n=\n8b\n8cc\n8dd\n=\n4ee\n=:|!\n*-\n",
        );
    }

    #[test]
    fn validates_full_repeat_end_after_meter_change_inside_downbeat_section() {
        assert_valid("**kern\n*M4/4\n1c\n=:|!\n1d\n=\n*M3/4\n2.e\n=:|!\n2.f\n=\n*-\n");
    }

    #[test]
    fn rejects_full_repeat_end_after_nonzero_section_entry_fragment() {
        assert_invalid(
            "**kern\n*M3/8\n8c\n=\n8d\n8e\n8f\n=\n4g\n=:|!\n16a\n=\n8b\n8cc\n8dd\n=\n8ee\n8ff\n8gg\n=:|!\n*-\n",
            "repeat ending measure does not complement repeat entry",
        );
    }

    #[test]
    fn rejects_plain_repeat_end_linear_continuation_that_stays_underfull() {
        // The fragment after the back repeat is a valid new-section pickup,
        // but the following internal measure must still fill the meter.
        assert_invalid(
            "**kern\n*M6/8\n8c\n=\n4d\n8e\n4f\n=:|!\n16g\n=\n4a\n8b\n4c\n=\n4d\n4e\n4f\n=\n*-\n",
            "measure duration does not match meter",
        );
    }

    #[test]
    fn rejects_full_repeat_end_after_partial_repeat_entry() {
        assert_invalid(
            "**kern\n*M4/4\n1c\n=:|:\n4d\n=\n1e\n=:|!\n*-\n",
            "repeat ending measure does not complement repeat entry",
        );
    }

    #[test]
    fn validates_underfull_repeat_end_after_complete_repeat_entry() {
        assert_valid("**kern\n*M4/4\n1c\n=:|:\n1d\n=\n4e\n4f\n4g\n=:|!\n*-\n");
    }

    #[test]
    fn validates_underfull_final_repeat_end_without_pickup_or_entry() {
        assert_valid("**kern\n*M4/4\n1c\n=\n4d\n4e\n4f\n=:|!\n*-\n");
    }

    #[test]
    fn validates_full_repeat_end_without_pickup_or_entry() {
        assert_valid("**kern\n*M4/4\n1c\n=:|!\n*-\n");
    }

    #[test]
    fn rejects_overfull_measure() {
        assert_invalid(
            "**kern\n*M4/4\n2c\n2d\n4e\n=\n*-\n",
            "measure duration exceeds meter",
        );
    }

    #[test]
    fn validates_dotted_and_rational_durations() {
        assert_valid("**kern\n*M4/4\n2.c\n8%2d\n=\n*-\n");
    }

    #[test]
    fn validates_pickup_with_complementary_final_measure() {
        assert_valid("**kern\n*M4/4\n4c\n=\n4d\n4e\n4f\n=\n*-\n");
    }

    #[test]
    fn validates_underfull_final_measure() {
        assert_valid("**kern\n*M4/4\n4c\n=\n4d\n4e\n=\n*-\n");
    }

    #[test]
    fn validates_meter_change_at_boundary() {
        assert_valid("**kern\n*M4/4\n4c\n4d\n4e\n4f\n=\n*M3/4\n4g\n4a\n4b\n=\n*-\n");
    }

    #[test]
    fn split_duplicates_elapsed_measure_position() {
        assert_valid("**kern\n*M4/4\n4c\n*^\n4d\t4e\n2f\t2a\n=\t=\n*-\t*-\n");
    }

    #[test]
    fn merge_accepts_equal_elapsed_measure_positions() {
        assert_valid("**kern\n*M4/4\n*^\n4c\t4e\n*v\t*v\n4d\n2f\n=\n*-\n");
    }

    #[test]
    fn merge_rejects_unequal_elapsed_measure_positions_with_columns_and_ahead_delta() {
        assert_invalid(
            "**kern\n*M4/4\n*^\n4c\t2e\n*v\t*v\n2f\n=\n*-\n",
            "cannot merge spines at different measure positions: merge columns 1-2; column 1 elapsed 1/4, column 2 elapsed 1/2; column 2 is ahead by 1/4",
        );
    }

    #[test]
    fn merge_rejects_unequal_elapsed_measure_positions_with_short_delta() {
        assert_invalid(
            "**kern\n*M4/4\n*^\n2c\t4e\n*v\t*v\n2f\n=\n*-\n",
            "cannot merge spines at different measure positions: merge columns 1-2; column 1 elapsed 1/2, column 2 elapsed 1/4; column 2 is short by 1/4",
        );
    }

    #[test]
    fn chord_duration_counts_once() {
        assert_valid("**kern\n*M4/4\n4c 4e 4g\n4d\n2f\n=\n*-\n");
    }

    #[test]
    fn rejects_chord_with_mismatched_note_durations() {
        let mut document = KernDocument::parse("**kern\n*M4/4\n4c 4e\n4d\n2f\n=\n*-\n").unwrap();
        let KernLine::Data(contents) = document
            .lines_mut()
            .iter_mut()
            .find(|line| matches!(line, KernLine::Data(_)))
            .expect("expected data line")
        else {
            panic!("expected data line");
        };
        let SpineContent::Kern(KernContent::Chord(chord)) = &mut contents[0] else {
            panic!("expected kern chord");
        };
        chord.notes[1].duration = Duration {
            value: DurationValue::Reciprocal(8),
            augmentation_dots: 0,
        };

        let error = check(&document).unwrap_err();
        assert!(error.to_string().contains("chord contains mixed durations"));
    }

    #[test]
    fn grace_notes_do_not_contribute_to_measure_duration() {
        assert_valid("**kern\n*M4/4\n4c\n8qdd\n4d\n2e\n=\n*-\n");
    }

    #[test]
    fn rejects_close_without_open() {
        assert_invalid_beams("**kern\n8cJ\n*-\n", "closes before it is open");
    }

    #[test]
    fn rejects_unterminated_beam_at_document_end() {
        assert_invalid_beams("**kern\n8cL\n", "beam crosses end of document");
    }

    #[test]
    fn rejects_too_many_active_beams_for_duration() {
        assert_invalid_beams(
            "**kern\n16cLL\n8dJ\n16eJ\n*-\n",
            "incoming beam count exceeds duration depth",
        );
    }

    #[test]
    fn beam_errors_include_source_line_column_and_state_context() {
        let document = KernDocument::parse("**kern\n16cLL\n8dJ\n16eJ\n*-\n").unwrap();
        let error = beam::validate_document(&document).unwrap_err().to_string();

        assert!(error.contains("line 3, column 1"));
        assert!(error.contains("expected_depth=1"));
        assert!(error.contains("entering=2"));
        assert!(error.contains("starts=0"));
        assert!(error.contains("ends=1"));
        assert!(error.contains("outgoing=1"));
    }

    #[test]
    fn rejects_beam_mark_on_unbeamable_duration() {
        assert_invalid_beams("**kern\n4cL\n4dJ\n*-\n", "unbeamable duration");
    }

    #[test]
    fn rejects_invalid_partial_left_without_incoming_beam() {
        assert_invalid_beams(
            "**kern\n16ck\n*-\n",
            "partial-left beam has no incoming beam",
        );
    }

    #[test]
    fn rejects_invalid_partial_right_without_outgoing_beam() {
        let mut document = KernDocument::parse("**kern\n16c\n*-\n").unwrap();
        let KernLine::Data(contents) = document
            .lines_mut()
            .iter_mut()
            .find(|line| matches!(line, KernLine::Data(_)))
            .expect("expected data line")
        else {
            panic!("expected data line");
        };
        let SpineContent::Kern(KernContent::Chord(chord)) = &mut contents[0] else {
            panic!("expected kern chord");
        };
        chord.modifiers.beams = ChordBeams::StartAndPartialRight {
            start_level: 0,
            partial_right_level: 1,
        };

        let error = beam::validate_document(&document).unwrap_err();
        assert!(error.to_string().contains("partial-right beam"));
    }

    #[test]
    fn rejects_beam_crossing_spine_termination() {
        assert_invalid_beams("**kern\n8cL\n*-\n", "beam crosses spine termination");
    }

    #[test]
    fn rejects_beam_jump_to_inserted_child_after_spine_split() {
        assert_invalid_beams(
            "**kern\n8cL\n*^\n4c\t8d\n.\t8eJ\n*v\t*v\n*-\n",
            "unbeamable duration participates in a beam",
        );
    }

    #[test]
    fn rejects_beam_merge_of_two_open_streams() {
        assert_invalid_beams(
            "**kern\n*^\n8cL\t8eL\n8d\t8f\n*v\t*v\n8gJ\n*-\n",
            "multiple open beams enter one spine merge",
        );
    }

    #[test]
    fn keeps_normalized_chord_beam_parse_behavior_out_of_scope() {
        let chord = Chord::try_from("4cLJ").unwrap();
        assert_eq!(chord.modifiers.beams, ChordBeams::None);
    }
}
