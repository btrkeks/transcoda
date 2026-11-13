//! Compatibility facade for the typed Humdrum `**kern` AST.
//!
//! New code can import from [`crate::ast`]. This module remains as the
//! stable historical path for callers using `kern_parser::types::*`.

pub use crate::ast::*;
