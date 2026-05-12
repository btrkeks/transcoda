//! Rhythm Checker for Humdrum **kern files.
//!
//! This library validates that measures in **kern files contain the correct
//! total duration based on the time signature. It uses exact rational arithmetic
//! to avoid floating-point errors.
//!
//! # Example
//!
//! ```rust
//! use rhythm_checker::validator::validate_file;
//! use std::path::Path;
//!
//! let result = validate_file(Path::new("song.krn"));
//! if result.is_ok() {
//!     println!("All {} measures valid!", result.measures_checked);
//! } else {
//!     for error in &result.errors {
//!         eprintln!("{}", error);
//!     }
//! }
//! ```

pub mod duration;
pub mod error;
pub mod parser;
pub mod repeat;
pub mod spine;
pub mod time_signature;
pub mod validator;

pub use duration::RhythmDuration;
pub use error::{FileResult, RhythmError, ValidationSummary};
pub use repeat::RepeatTracker;
pub use time_signature::TimeSignature;
pub use validator::{validate_file, IncrementalValidator, Validator, ValidatorOptions};
