//! Error types for the rhythm checker.

use crate::duration::RhythmDuration;
use crate::time_signature::TimeSignature;
use serde::Serialize;
use std::fmt;
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during parsing.
#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("invalid time signature: {0}")]
    InvalidTimeSignature(String),

    #[error("invalid duration: {0}")]
    InvalidDuration(String),

    #[error("invalid spine operation: {0}")]
    InvalidSpineOperation(String),
}

/// A rhythm validation error for a specific measure.
#[derive(Debug, Clone, Serialize)]
pub struct RhythmError {
    /// Path to the file containing the error.
    pub file: PathBuf,
    /// Line number (1-indexed) where the barline was found.
    pub line: usize,
    /// Spine index (0-indexed) where the error occurred.
    pub spine: usize,
    /// Measure number (1-indexed).
    pub measure: u32,
    /// Expected duration based on time signature.
    pub expected: RhythmDuration,
    /// Actual accumulated duration.
    pub actual: RhythmDuration,
    /// Time signature in effect.
    pub time_signature: TimeSignature,
    /// Whether this is the first measure (potential anacrusis).
    pub is_first_measure: bool,
    /// Whether this is the final measure (at double barline).
    pub is_final_measure: bool,
    /// Whether this error is at a repeat boundary.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_repeat_boundary: bool,
    /// If this is a repeat pairing error, the paired pickup duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paired_pickup: Option<RhythmDuration>,
}

impl RhythmError {
    /// Calculate the difference between expected and actual duration.
    pub fn difference(&self) -> RhythmDuration {
        if self.expected >= self.actual {
            self.expected - self.actual
        } else {
            self.actual - self.expected
        }
    }

    /// Returns true if actual duration is less than expected.
    pub fn is_short(&self) -> bool {
        self.actual < self.expected
    }
}

impl fmt::Display for RhythmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.is_short() { "-" } else { "+" };

        // Special message for repeat pairing errors
        if self.is_repeat_boundary {
            if let Some(pickup) = self.paired_pickup {
                let sum = pickup + self.actual;
                return write!(
                    f,
                    "{}:{}:{}: error: measure {} at repeat boundary doesn't pair with pickup\n  \
                     pickup:     {} (measure 1)\n  \
                     pre-repeat: {} (measure {})\n  \
                     sum:        {}\n  \
                     expected:   {} ({})",
                    self.file.display(),
                    self.line,
                    self.spine + 1,
                    self.measure,
                    pickup,
                    self.actual,
                    self.measure,
                    sum,
                    self.expected,
                    self.time_signature,
                );
            }
        }

        write!(
            f,
            "{}:{}:{}: error: measure {} duration mismatch\n  \
             expected: {} ({})\n  \
             actual:   {}\n  \
             diff:     {}{}",
            self.file.display(),
            self.line,
            self.spine + 1,
            self.measure,
            self.expected,
            self.time_signature,
            self.actual,
            sign,
            self.difference()
        )
    }
}

/// Result of validating a single file.
#[derive(Debug, Clone, Serialize)]
pub struct FileResult {
    /// Path to the validated file.
    pub file: PathBuf,
    /// Total number of measures validated.
    pub measures_checked: u32,
    /// Rhythm errors found.
    pub errors: Vec<RhythmError>,
    /// Any warnings (e.g., unrecognized tokens).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl FileResult {
    /// Create a new file result.
    pub fn new(file: PathBuf) -> Self {
        FileResult {
            file,
            measures_checked: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Returns true if validation passed (no errors).
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Result of validating multiple files.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationSummary {
    /// Total files processed.
    pub files_processed: usize,
    /// Files with errors.
    pub files_with_errors: usize,
    /// Total errors across all files.
    pub total_errors: usize,
    /// Total measures checked across all files.
    pub total_measures: u32,
    /// Per-file results.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub file_results: Vec<FileResult>,
}

impl ValidationSummary {
    /// Create a new empty summary.
    pub fn new() -> Self {
        ValidationSummary {
            files_processed: 0,
            files_with_errors: 0,
            total_errors: 0,
            total_measures: 0,
            file_results: Vec::new(),
        }
    }

    /// Add a file result to the summary.
    pub fn add(&mut self, result: FileResult) {
        self.files_processed += 1;
        self.total_measures += result.measures_checked;
        if !result.is_ok() {
            self.files_with_errors += 1;
            self.total_errors += result.errors.len();
        }
        self.file_results.push(result);
    }

    /// Returns true if all files passed validation.
    pub fn is_ok(&self) -> bool {
        self.total_errors == 0
    }
}

impl Default for ValidationSummary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhythm_error_display() {
        let error = RhythmError {
            file: PathBuf::from("test.krn"),
            line: 42,
            spine: 0,
            measure: 5,
            expected: RhythmDuration::new(1, 1),
            actual: RhythmDuration::new(7, 8),
            time_signature: TimeSignature::new(4, 4),
            is_first_measure: false,
            is_final_measure: false,
            is_repeat_boundary: false,
            paired_pickup: None,
        };

        let display = error.to_string();
        assert!(display.contains("test.krn:42:1"));
        assert!(display.contains("measure 5"));
        assert!(display.contains("expected: 1 (4/4)"));
        assert!(display.contains("actual:   7/8"));
        assert!(display.contains("diff:     -1/8"));
    }

    #[test]
    fn test_file_result() {
        let mut result = FileResult::new(PathBuf::from("test.krn"));
        assert!(result.is_ok());

        result.errors.push(RhythmError {
            file: PathBuf::from("test.krn"),
            line: 10,
            spine: 0,
            measure: 1,
            expected: RhythmDuration::new(1, 1),
            actual: RhythmDuration::new(3, 4),
            time_signature: TimeSignature::new(4, 4),
            is_first_measure: true,
            is_final_measure: false,
            is_repeat_boundary: false,
            paired_pickup: None,
        });
        assert!(!result.is_ok());
    }

    #[test]
    fn test_validation_summary() {
        let mut summary = ValidationSummary::new();
        assert!(summary.is_ok());

        let mut good_result = FileResult::new(PathBuf::from("good.krn"));
        good_result.measures_checked = 10;
        summary.add(good_result);

        assert!(summary.is_ok());
        assert_eq!(summary.files_processed, 1);
        assert_eq!(summary.files_with_errors, 0);

        let mut bad_result = FileResult::new(PathBuf::from("bad.krn"));
        bad_result.measures_checked = 5;
        bad_result.errors.push(RhythmError {
            file: PathBuf::from("bad.krn"),
            line: 10,
            spine: 0,
            measure: 1,
            expected: RhythmDuration::new(1, 1),
            actual: RhythmDuration::new(3, 4),
            time_signature: TimeSignature::new(4, 4),
            is_first_measure: false,
            is_final_measure: false,
            is_repeat_boundary: false,
            paired_pickup: None,
        });
        summary.add(bad_result);

        assert!(!summary.is_ok());
        assert_eq!(summary.files_processed, 2);
        assert_eq!(summary.files_with_errors, 1);
        assert_eq!(summary.total_errors, 1);
        assert_eq!(summary.total_measures, 15);
    }
}
