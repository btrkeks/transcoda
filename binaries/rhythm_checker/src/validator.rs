//! Core rhythm validation algorithm.

use crate::duration::RhythmDuration;
use crate::error::{FileResult, RhythmError};
use crate::parser::{parse_token, split_tokens, LineType, TokenInfo};
use crate::repeat::RepeatTracker;
use crate::spine::SpineTracker;
use std::fs;
use std::path::{Path, PathBuf};

/// Tracks errors from the last barline that may be suppressed if it's the final measure.
#[derive(Debug)]
struct PendingFinalValidation {
    errors: Vec<RhythmError>,
}

/// Options for rhythm validation.
#[derive(Debug, Clone)]
pub struct ValidatorOptions {
    /// Allow incomplete first measure (anacrusis/pickup).
    pub allow_anacrusis: bool,
    /// Allow incomplete final measure.
    pub allow_incomplete_final: bool,
    /// Allow incomplete measures at repeat boundaries to pair with initial pickup.
    pub allow_repeat_pairing: bool,
    /// Verbose output (per-measure details).
    pub verbose: bool,
}

impl Default for ValidatorOptions {
    fn default() -> Self {
        ValidatorOptions {
            allow_anacrusis: true,
            allow_incomplete_final: true,
            allow_repeat_pairing: true,
            verbose: false,
        }
    }
}

/// Stateful line-by-line validator for **kern rhythm semantics.
pub struct IncrementalValidator {
    options: ValidatorOptions,
    result: FileResult,
    tracker: SpineTracker,
    repeat_tracker: RepeatTracker,
    is_first_barline: bool,
    pending_final: Option<PendingFinalValidation>,
    finished: bool,
}

impl IncrementalValidator {
    /// Create an incremental validator for a logical file path.
    pub fn new(path: impl Into<PathBuf>, options: ValidatorOptions) -> Self {
        IncrementalValidator {
            options,
            result: FileResult::new(path.into()),
            tracker: SpineTracker::new(),
            repeat_tracker: RepeatTracker::new(),
            is_first_barline: true,
            pending_final: None,
            finished: false,
        }
    }

    /// Create an incremental validator with default options.
    pub fn with_defaults(path: impl Into<PathBuf>) -> Self {
        Self::new(path, ValidatorOptions::default())
    }

    /// Feed one 1-indexed source line into the validator.
    pub fn accept_line(&mut self, line_num: usize, line: &str) {
        if self.finished {
            return;
        }

        let line_type = LineType::classify(line);
        match line_type {
            LineType::Empty => {}
            LineType::ExclusiveInterpretation => self.handle_exclusive(line),
            LineType::TandemInterpretation => self.handle_tandem(line_num, line),
            LineType::Data => self.handle_data(line_num, line),
            LineType::Barline {
                is_double,
                is_visual_double,
                repeat,
                ..
            } => self.handle_barline(line_num, is_double, is_visual_double, repeat),
            LineType::SpineTerminator => self.handle_spine_terminator(line_num),
        }
    }

    /// Finalize validation after all lines have been consumed.
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }

        if let Some(pending) = self.pending_final.take() {
            self.result.errors.extend(pending.errors);
        }
        self.finished = true;
    }

    /// Borrow the current result.
    pub fn result(&self) -> &FileResult {
        &self.result
    }

    /// Consume the validator and return the accumulated result.
    pub fn into_result(mut self) -> FileResult {
        self.finish();
        self.result
    }

    fn handle_exclusive(&mut self, line: &str) {
        let tokens = split_tokens(line);
        self.tracker.initialize(&tokens);
    }

    fn handle_tandem(&mut self, line_num: usize, line: &str) {
        let tokens = split_tokens(line);

        let has_spine_op = tokens
            .iter()
            .any(|t| *t == "*^" || *t == "*v" || *t == "*x");
        if has_spine_op {
            if let Err(e) = self.tracker.process_tandem(&tokens) {
                self.result
                    .warnings
                    .push(format!("line {}: spine operation error: {}", line_num, e));
            }
        }

        if tokens.len() == self.tracker.spine_count() {
            self.tracker.update_time_signatures(&tokens);
        }
    }

    fn handle_data(&mut self, line_num: usize, line: &str) {
        let tokens = split_tokens(line);

        for (spine_idx, token) in tokens.iter().enumerate() {
            if let Some(spine) = self.tracker.get_mut(spine_idx) {
                if !spine.is_kern {
                    continue;
                }

                match parse_token(token) {
                    TokenInfo::Duration(d) | TokenInfo::Chord(d) => spine.add_duration(d),
                    TokenInfo::GraceNote | TokenInfo::Null => {}
                    TokenInfo::Unknown(s) => {
                        if !s.is_empty() && s != "." {
                            self.result.warnings.push(format!(
                                "line {}: unrecognized token in spine {}: {}",
                                line_num,
                                spine_idx + 1,
                                s
                            ));
                        }
                    }
                }
            }
        }
    }

    fn handle_barline(
        &mut self,
        line_num: usize,
        is_double: bool,
        is_visual_double: bool,
        repeat: crate::parser::RepeatInfo,
    ) {
        if let Some(pending) = self.pending_final.take() {
            self.result.errors.extend(pending.errors);
        }

        let has_content = self.tracker.kern_indices().iter().any(|&i| {
            self.tracker
                .get(i)
                .map(|s| !s.measure_duration.is_zero())
                .unwrap_or(false)
        });

        if self.is_first_barline && !has_content {
            self.tracker.start_all_measures();
            return;
        }

        let kern_indices = self.tracker.kern_indices();
        let mut spine_durations: Vec<RhythmDuration> = Vec::new();
        let mut expected_durations: Vec<RhythmDuration> = Vec::new();

        for &spine_idx in &kern_indices {
            if let Some(spine) = self.tracker.get(spine_idx) {
                spine_durations.push(spine.measure_duration);
                expected_durations.push(spine.expected_duration().unwrap_or(RhythmDuration::ZERO));
            }
        }

        if repeat.is_start_repeat {
            self.repeat_tracker.on_start_repeat(&expected_durations);
        }

        let mut deferred_errors: Vec<RhythmError> = Vec::new();

        for (local_idx, &spine_idx) in kern_indices.iter().enumerate() {
            if let Some(spine) = self.tracker.get(spine_idx) {
                if let Some(expected) = spine.expected_duration() {
                    let actual = spine.measure_duration;
                    if actual != expected {
                        let is_first = self.is_first_barline;
                        let is_final = is_double || is_visual_double;
                        let is_short = actual < expected;
                        let at_end_repeat = repeat.is_end_repeat && is_short;

                        let should_report = if is_first && self.options.allow_anacrusis && is_short
                        {
                            false
                        } else if self.repeat_tracker.is_awaiting_section_pickup()
                            && self.options.allow_anacrusis
                            && is_short
                        {
                            false
                        } else if is_final && self.options.allow_incomplete_final && is_short {
                            false
                        } else if at_end_repeat
                            && self.options.allow_repeat_pairing
                            && (self.repeat_tracker.has_section_pickup()
                                || self.repeat_tracker.has_initial_pickup())
                        {
                            let pickup = if self.repeat_tracker.has_section_pickup() {
                                self.repeat_tracker
                                    .get_section_pickup(local_idx)
                                    .unwrap_or(RhythmDuration::ZERO)
                            } else {
                                self.repeat_tracker
                                    .get_initial_pickup(local_idx)
                                    .unwrap_or(RhythmDuration::ZERO)
                            };
                            let sum = pickup + actual;

                            if sum == expected {
                                false
                            } else {
                                self.result.errors.push(RhythmError {
                                    file: self.result.file.clone(),
                                    line: line_num,
                                    spine: spine_idx,
                                    measure: spine.measure_number,
                                    expected,
                                    actual,
                                    time_signature: spine
                                        .effective_time_signature()
                                        .unwrap_or_default(),
                                    is_first_measure: false,
                                    is_final_measure: is_final,
                                    is_repeat_boundary: true,
                                    paired_pickup: Some(pickup),
                                });
                                false
                            }
                        } else {
                            true
                        };

                        if should_report {
                            let error = RhythmError {
                                file: self.result.file.clone(),
                                line: line_num,
                                spine: spine_idx,
                                measure: spine.measure_number,
                                expected,
                                actual,
                                time_signature: spine
                                    .effective_time_signature()
                                    .unwrap_or_default(),
                                is_first_measure: is_first,
                                is_final_measure: is_final,
                                is_repeat_boundary: repeat.is_end_repeat,
                                paired_pickup: None,
                            };

                            if !is_final && self.options.allow_incomplete_final && is_short {
                                deferred_errors.push(error);
                            } else {
                                self.result.errors.push(error);
                            }
                        }
                    }
                }
            }
        }

        if !deferred_errors.is_empty() {
            self.pending_final = Some(PendingFinalValidation {
                errors: deferred_errors,
            });
        }

        if self.is_first_barline && self.options.allow_anacrusis {
            let has_incomplete = kern_indices.iter().any(|&spine_idx| {
                self.tracker
                    .get(spine_idx)
                    .and_then(|s| s.expected_duration().map(|e| s.measure_duration < e))
                    .unwrap_or(false)
            });
            if has_incomplete {
                self.repeat_tracker.record_initial_pickup(&spine_durations);
            }
        }

        if self.repeat_tracker.is_awaiting_section_pickup() && self.options.allow_anacrusis {
            let has_incomplete = kern_indices.iter().any(|&spine_idx| {
                self.tracker
                    .get(spine_idx)
                    .and_then(|s| s.expected_duration().map(|e| s.measure_duration < e))
                    .unwrap_or(false)
            });
            if has_incomplete {
                self.repeat_tracker.record_section_pickup(&spine_durations);
            } else {
                self.repeat_tracker.clear_awaiting_section_pickup();
            }
        }

        if repeat.is_end_repeat {
            self.repeat_tracker.clear_section_pickup();
        }

        let is_section_boundary = is_double || is_visual_double;
        if (repeat.is_start_repeat || is_section_boundary) && self.options.allow_anacrusis {
            self.repeat_tracker.mark_awaiting_section_pickup();
        }

        if let Some(&first_kern) = kern_indices.first() {
            if let Some(spine) = self.tracker.get(first_kern) {
                self.result.measures_checked = spine.measure_number;
            }
        }

        self.tracker.reset_all_measures();
        self.is_first_barline = false;
    }

    fn handle_spine_terminator(&mut self, line_num: usize) {
        let kern_indices = self.tracker.kern_indices();
        let has_content = kern_indices.iter().any(|&i| {
            self.tracker
                .get(i)
                .map(|s| !s.measure_duration.is_zero())
                .unwrap_or(false)
        });

        if has_content {
            if let Some(pending) = self.pending_final.take() {
                self.result.errors.extend(pending.errors);
            }

            for &spine_idx in &kern_indices {
                if let Some(spine) = self.tracker.get(spine_idx) {
                    if spine.measure_duration.is_zero() {
                        continue;
                    }
                    if let Some(expected) = spine.expected_duration() {
                        let actual = spine.measure_duration;
                        if actual != expected {
                            let is_short = actual < expected;
                            if !(self.options.allow_incomplete_final && is_short) {
                                self.result.errors.push(RhythmError {
                                    file: self.result.file.clone(),
                                    line: line_num,
                                    spine: spine_idx,
                                    measure: spine.measure_number,
                                    expected,
                                    actual,
                                    time_signature: spine
                                        .effective_time_signature()
                                        .unwrap_or_default(),
                                    is_first_measure: false,
                                    is_final_measure: true,
                                    is_repeat_boundary: false,
                                    paired_pickup: None,
                                });
                            }
                        }
                    }
                }
            }
        } else if let Some(pending) = self.pending_final.take() {
            if !self.options.allow_incomplete_final {
                self.result.errors.extend(pending.errors);
            }
        }
    }
}

/// Validate rhythm in a **kern file.
pub struct Validator {
    options: ValidatorOptions,
}

impl Validator {
    /// Create a validator with the given options.
    pub fn new(options: ValidatorOptions) -> Self {
        Validator { options }
    }

    /// Create a validator with default options.
    pub fn with_defaults() -> Self {
        Self::new(ValidatorOptions::default())
    }

    /// Validate a file and return the result.
    pub fn validate_file(&self, path: &Path) -> FileResult {
        let content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                let mut result = FileResult::new(path.to_path_buf());
                result.warnings.push(format!("failed to read file: {}", e));
                return result;
            }
        };

        let mut incremental = IncrementalValidator::new(path.to_path_buf(), self.options.clone());
        for (line_num, line) in content.lines().enumerate() {
            incremental.accept_line(line_num + 1, line);
        }
        incremental.into_result()
    }

    /// Validate content string and populate the result.
    pub fn validate_content(&self, content: &str, result: &mut FileResult) {
        let mut incremental = IncrementalValidator::new(result.file.clone(), self.options.clone());
        for (line_num, line) in content.lines().enumerate() {
            incremental.accept_line(line_num + 1, line);
        }
        *result = incremental.into_result();
    }
}

/// Convenience function to validate a single file.
pub fn validate_file(path: &Path) -> FileResult {
    Validator::with_defaults().validate_file(path)
}

/// Convenience function to validate file content.
pub fn validate_content(content: &str, path: &Path) -> FileResult {
    let mut result = FileResult::new(path.to_path_buf());
    Validator::with_defaults().validate_content(content, &mut result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn validate_str(content: &str) -> FileResult {
        validate_content(content, Path::new("test.krn"))
    }

    fn validate_incrementally(content: &str) -> FileResult {
        let mut incremental = IncrementalValidator::with_defaults("test.krn");
        for (line_num, line) in content.lines().enumerate() {
            incremental.accept_line(line_num + 1, line);
        }
        incremental.into_result()
    }

    #[test]
    fn test_simple_valid_file() {
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=2
4g
4a
4b
4cc
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
        assert_eq!(result.measures_checked, 2);
    }

    #[test]
    fn test_incremental_validator_matches_batch_validation() {
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=2
4g
4a
4b
4cc
==
*-";

        let batch = validate_str(content);
        let incremental = validate_incrementally(content);

        assert_eq!(incremental.measures_checked, batch.measures_checked);
        assert_eq!(incremental.errors.len(), batch.errors.len());
        assert_eq!(incremental.warnings, batch.warnings);
    }

    #[test]
    fn test_incremental_validator_finish_commits_pending_errors() {
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=2
4c
4d
4e
=3";

        let batch = validate_str(content);
        let incremental = validate_incrementally(content);

        assert_eq!(incremental.errors.len(), batch.errors.len());
        assert_eq!(incremental.errors[0].measure, 2);
    }

    #[test]
    fn test_incremental_validator_preserves_final_measure_exception() {
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=2
4g
4a
*-
";

        let result = validate_incrementally(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_dotted_rhythm() {
        let content = "\
**kern
*M4/4
=1
4.c
8d
4e
4f
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_invalid_measure_short() {
        // Short measure in position 2 (not first, so anacrusis doesn't apply)
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=2
4c
4d
4e
=3
4g
4a
4b
4cc
==
*-";

        let result = validate_str(content);
        assert!(!result.is_ok());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].measure, 2);
        assert!(result.errors[0].is_short());
    }

    #[test]
    fn test_invalid_measure_long() {
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
4g
=2
4c
4d
4e
4f
==
*-";

        let result = validate_str(content);
        assert!(!result.is_ok());
        assert_eq!(result.errors.len(), 1);
        assert!(!result.errors[0].is_short()); // too long
    }

    #[test]
    fn test_anacrusis_allowed() {
        // First measure is incomplete (pickup)
        let content = "\
**kern
*M4/4
=1
4c
=2
4d
4e
4f
4g
==
*-";

        let result = validate_str(content);
        // Anacrusis allowed by default
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_anacrusis_disallowed() {
        let content = "\
**kern
*M4/4
=1
4c
=2
4d
4e
4f
4g
==
*-";

        let validator = Validator::new(ValidatorOptions {
            allow_anacrusis: false,
            ..Default::default()
        });
        let mut result = FileResult::new(PathBuf::from("test.krn"));
        validator.validate_content(content, &mut result);

        assert!(!result.is_ok());
        assert_eq!(result.errors[0].measure, 1);
        assert!(result.errors[0].is_first_measure);
    }

    #[test]
    fn test_triplets() {
        let content = "\
**kern
*M4/4
=1
12c
12d
12e
4f
4g
4a
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_rests() {
        let content = "\
**kern
*M4/4
=1
4r
4c
4r
4d
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_grace_notes_ignored() {
        let content = "\
**kern
*M4/4
=1
8qc
4d
4e
4f
4g
==
*-";

        let result = validate_str(content);
        // Grace notes have zero duration
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_chords() {
        let content = "\
**kern
*M4/4
=1
4c 4e 4g
4d
4e
4f
==
*-";

        let result = validate_str(content);
        // Chord should count as one quarter note
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_multiple_spines() {
        let content = "\
**kern\t**kern
*M4/4\t*M4/4
=1\t=1
4c\t4g
4d\t4a
4e\t4b
4f\t4cc
=2\t=2
4g\t4dd
4a\t4ee
4b\t4ff
4cc\t4gg
==\t==
*-\t*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_time_signature_change() {
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
*M3/4
=2
4g
4a
4b
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_time_signature_change_after_barline_applies_to_current_measure() {
        let content = "\
**kern
*M2/4
=1
4c
4d
=2
*M3/4
4e
4f
4g
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_time_signature_change_after_barline_with_non_kern_spines() {
        let content = "\
**kern\t**dynam\t**kern\t**text
*M2/4\t*\t*M2/4\t*
=1\t=1\t=1\t=1
4c\tp\t4e\tla
4d\t.\t4f\t.
=2\t=2\t=2\t=2
*M3/4\t*\t*M3/4\t*
4e\t.\t4g\t.
4f\t.\t4a\t.
4g\t.\t4b\tha
==\t==\t==\t==
*-\t*-\t*-\t*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_six_eight_time() {
        let content = "\
**kern
*M6/8
=1
8c
8d
8e
8f
8g
8a
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_whole_note() {
        let content = "\
**kern
*M4/4
=1
1c
=2
2d
2e
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_null_tokens_ignored() {
        let content = "\
**kern\t**dynam
*M4/4\t*
=1\t=1
4c\t.
4d\tf
4e\t.
4f\t.
==\t==
*-\t*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_non_kern_spine_ignored() {
        // Dynamics spine has no valid rhythm but should be ignored
        let content = "\
**kern\t**dynam
*M4/4\t*
=1\t=1
4c\tf
4d\t.
4e\t.
4f\tp
==\t==
*-\t*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_repeat_with_anacrusis() {
        // 6/8 time with pickup (1/8) and pre-repeat measure (5/8)
        // When paired: 1/8 + 5/8 = 6/8 (complete)
        let content = "\
**kern
*M6/8
=1
8cc
=2
8a
8b-
8cc
8cc
8dd
8cc
=3
4a
8g
4f
=:|!|:
8ee
8ff
8gg
8aa
8bb-
8ccc
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_repeat_with_anacrusis_mismatch() {
        // 6/8 time with pickup (1/8) and pre-repeat measure (4/8)
        // When paired: 1/8 + 4/8 = 5/8 (incomplete - should fail)
        let content = "\
**kern
*M6/8
=1
8cc
=2
8a
8b-
8cc
8cc
8dd
8cc
=3
4a
4f
=:|!|:
8ee
8ff
8gg
8aa
8bb-
8ccc
==
*-";

        let result = validate_str(content);
        assert!(
            !result.is_ok(),
            "expected error for mismatched repeat pairing"
        );
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].is_repeat_boundary);
        assert!(result.errors[0].paired_pickup.is_some());
    }

    #[test]
    fn test_repeat_pairing_disabled() {
        // Same as test_repeat_with_anacrusis but with pairing disabled
        let content = "\
**kern
*M6/8
=1
8cc
=2
8a
8b-
8cc
8cc
8dd
8cc
=3
4a
8g
4f
=:|!|:
8ee
8ff
8gg
8aa
8bb-
8ccc
==
*-";

        let validator = Validator::new(ValidatorOptions {
            allow_repeat_pairing: false,
            ..Default::default()
        });
        let mut result = FileResult::new(PathBuf::from("test.krn"));
        validator.validate_content(content, &mut result);

        // Should report an error because pairing is disabled
        assert!(!result.is_ok());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].measure, 3);
    }

    #[test]
    fn test_end_repeat_without_anacrusis() {
        // End repeat but with complete measures (no anacrusis)
        let content = "\
**kern
*M6/8
=1
8a
8b-
8cc
8cc
8dd
8cc
=2
4a
8g
4f
=:|!|:
8ee
8ff
8gg
8aa
8bb-
8ccc
==
*-";

        let result = validate_str(content);
        // No anacrusis, so pre-repeat measure (5/8) should be an error
        assert!(!result.is_ok());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].measure, 2);
    }

    #[test]
    fn test_start_repeat_only() {
        // Start repeat without end repeat - no pairing needed
        let content = "\
**kern
*M4/4
=|:
4c
4d
4e
4f
=2
4g
4a
4b
4cc
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_multiple_spines_repeat() {
        // Two kern spines with repeat pairing
        let content = "\
**kern\t**kern
*M6/8\t*M6/8
=1\t=1
8cc\t8cc
=2\t=2
8a\t8e
8b-\t8f
8cc\t8g
8cc\t8a
8dd\t8b
8cc\t8cc
=3\t=3
4a\t4a
8g\t8g
4f\t4f
=:|!|:\t=:|!|:
8ee\t8ee
8ff\t8ff
8gg\t8gg
8aa\t8aa
8bb-\t8bb-
8ccc\t8ccc
==\t==
*-\t*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_section_anacrusis_after_segue() {
        // AABB form: A section ends, B section starts with pickup
        // 6/8 time: initial pickup (1/8), B section pickup (1/8)
        let content = "\
**kern
*M6/8
=1
8cc
=2
8a
8b-
8cc
8cc
8dd
8cc
=3
4a
8g
4f
=:|!|:
8ee
=5
8ff
8gg
8aa
8aa
8bb-
8ccc
=6
4aa
8gg
4ff
=:|!
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_section_anacrusis_multiple_spines() {
        // AABB form with two kern spines, both with section pickup
        let content = "\
**kern\t**kern
*M6/8\t*M6/8
=1\t=1
8cc\t8cc
=2\t=2
8a\t8e
8b-\t8f
8cc\t8g
8cc\t8a
8dd\t8b
8cc\t8cc
=3\t=3
4a\t4a
8g\t8g
4f\t4f
=:|!|:\t=:|!|:
8ee\t8ee
=5\t=5
8ff\t8ff
8gg\t8gg
8aa\t8aa
8aa\t8aa
8bb-\t8bb-
8ccc\t8ccc
=6\t=6
4aa\t4aa
8gg\t8gg
4ff\t4ff
=:|!\t=:|!
*-\t*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_start_repeat_full_measure() {
        // Start repeat followed by full measure (no section anacrusis)
        let content = "\
**kern
*M6/8
=1
8cc
=2
8a
8b-
8cc
8cc
8dd
8cc
=3
4a
8g
4f
=:|!|:
8ff
8gg
8aa
8aa
8bb-
8ccc
=5
4aa
8gg
4ff
=:|!
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_section_anacrusis_mismatch() {
        // Section pickup (1/8) + pre-repeat (4/8) = 5/8, expected 6/8 - should fail
        let content = "\
**kern
*M6/8
=1
8cc
=2
8a
8b-
8cc
8cc
8dd
8cc
=3
4a
8g
4f
=:|!|:
8ee
=5
8ff
8gg
8aa
8aa
8bb-
8ccc
=6
4aa
4ff
=:|!
*-";

        let result = validate_str(content);
        assert!(
            !result.is_ok(),
            "expected error for mismatched section pairing"
        );
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].is_repeat_boundary);
        assert!(result.errors[0].paired_pickup.is_some());
    }

    #[test]
    fn test_incomplete_final_via_terminator() {
        // Plain barline + *- should allow incomplete final measure
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=2
4g
4a
4b
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_incomplete_final_visual_double_barline() {
        // =|| (visual double barline) + *- should allow incomplete final measure
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=||
4g
4a
4b
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_incomplete_middle_measure_still_errors() {
        // Incomplete middle measure (not first, not final) should still error
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=2
4g
4a
4b
=3
4cc
4dd
4ee
4ff
*-";

        let result = validate_str(content);
        assert!(
            !result.is_ok(),
            "expected error for incomplete middle measure"
        );
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].measure, 2);
    }

    #[test]
    fn test_deferred_error_committed_without_terminator() {
        // If file ends without *- after a barline with deferred error,
        // the deferred error should be committed (conservative behavior)
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=2
4g
4a
4b
=3";

        let result = validate_str(content);
        // Measure 2 is short (3/4) and was deferred at =3
        // Since no *- follows, the error is committed
        assert!(
            !result.is_ok(),
            "expected error for incomplete measure without terminator"
        );
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].measure, 2);
    }

    #[test]
    fn test_incomplete_final_disabled_via_terminator() {
        // With allow_incomplete_final=false, plain barline + *- should still error
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=2
4g
4a
4b
*-";

        let validator = Validator::new(ValidatorOptions {
            allow_incomplete_final: false,
            ..Default::default()
        });
        let mut result = FileResult::new(PathBuf::from("test.krn"));
        validator.validate_content(content, &mut result);

        assert!(!result.is_ok());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].measure, 2);
    }

    #[test]
    fn test_incomplete_final_barline_immediately_before_terminator() {
        // =|| immediately followed by *- (no content between)
        // This pattern appears in real scores where the final measure ends with a visual double barline
        let content = "\
**kern
*M3/4
=1
4c
4d
4e
=2
4f
4g
4a
=3
2b
=||
*-";

        let result = validate_str(content);
        // Measure 3 has 2/4 (half note) vs expected 3/4 - should be allowed as final
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_incomplete_final_disabled_barline_immediately_before_terminator() {
        // Same as above but with allow_incomplete_final=false
        let content = "\
**kern
*M3/4
=1
4c
4d
4e
=2
4f
4g
4a
=3
2b
=||
*-";

        let validator = Validator::new(ValidatorOptions {
            allow_incomplete_final: false,
            ..Default::default()
        });
        let mut result = FileResult::new(PathBuf::from("test.krn"));
        validator.validate_content(content, &mut result);

        assert!(!result.is_ok());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].measure, 3);
    }

    #[test]
    fn test_visual_double_allows_incomplete_mid_piece() {
        // =|| in middle of piece should allow incomplete measure
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=||
4g
4a
4b
=3
4cc
4dd
4ee
4ff
==
*-";
        let result = validate_str(content);
        assert!(
            result.is_ok(),
            "=|| should allow incomplete measure: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_semantic_double_allows_incomplete_mid_piece() {
        // == in middle of piece should allow incomplete measure
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
==
4g
4a
4b
=3
4cc
4dd
4ee
4ff
==
*-";
        let result = validate_str(content);
        assert!(
            result.is_ok(),
            "== should allow incomplete measure: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_spine_split() {
        let content = "\
**kern
*M4/4
=1
2c
*^
2d\t2e
==
*-\t*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_spine_merge() {
        let content = "\
**kern\t**kern
*M4/4\t*M4/4
=1\t=1
2c\t2e
2d\t2f
*v\t*v
=2
1g
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_repeat_pairing() {
        // 6/8 with 1/8 pickup and 5/8 pre-repeat
        let content = "\
**kern
*M6/8
=1
8g
=2
4c
8d
4e
=3:|!
8e
8f
8g
8a
8b
8cc
==
*-";

        let result = validate_str(content);
        assert!(result.is_ok(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_repeat_pairing_mismatch() {
        // 6/8 with 1/8 pickup but only 4/8 pre-repeat (should fail)
        let content = "\
**kern
*M6/8
=1
8g
=2
4c
4d
=3:|!
8e
8f
8g
8a
8b
8cc
==
*-";

        let validator = Validator::new(ValidatorOptions {
            allow_anacrusis: true,
            allow_incomplete_final: true,
            allow_repeat_pairing: false,
            verbose: false,
        });
        let mut result = FileResult::new(PathBuf::from("test.krn"));
        validator.validate_content(content, &mut result);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_final_measure_without_terminator_commits_pending() {
        let content = "\
**kern
*M4/4
=1
4c
4d
4e
4f
=2
4g
4a
4b
=3";

        let result = validate_str(content);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].measure, 2);
    }
}
