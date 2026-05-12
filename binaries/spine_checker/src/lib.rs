//! Library entrypoint for strict canonical `**kern` spine checking.

pub mod checker;
pub mod error;

pub use checker::SpineChecker;
pub use error::{CanonicalViolation, SpineCheckError};
