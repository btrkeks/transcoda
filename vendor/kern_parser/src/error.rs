use std::fmt;

use thiserror::Error;

use crate::ast::TimeSignature;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidNoteLocation {
    pub position: Option<usize>,
    pub character: Option<char>,
}

impl fmt::Display for InvalidNoteLocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.position, self.character) {
            (Some(position), Some(character)) => {
                write!(formatter, "character {position} {character:?}")
            }
            (Some(position), None) => write!(formatter, "after character {position}"),
            (None, Some(character)) => write!(formatter, "character {character:?}"),
            (None, None) => formatter.write_str("no specific character"),
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("empty kern document")]
    EmptyKernDocument,
    #[error("line {line}: empty kern line")]
    EmptyKernLine { line: usize },
    #[error("line {line}: empty spine token at column {column}")]
    EmptySpineToken { line: usize, column: usize },
    #[error("line {line}: invalid spine token {raw:?} at column {column}: {source}")]
    InvalidSpineToken {
        line: usize,
        column: usize,
        raw: String,
        #[source]
        source: Box<ParseError>,
    },
    #[error(
        "line {line}: mixed line types at column {column}; expected {expected}, found {found}"
    )]
    MixedKernLineTypes {
        line: usize,
        column: usize,
        expected: &'static str,
        found: &'static str,
    },
    #[error(
        "line {line}: mixed tandem interpretation types at column {column}; expected {expected}, found {found}"
    )]
    MixedTandemInterpretationTypes {
        line: usize,
        column: usize,
        expected: &'static str,
        found: &'static str,
    },
    #[error("line {line}: kern line contains only null interpretation tokens")]
    NullKernLine { line: usize },
    #[error("line {line}: spine type line must appear before any non-comment transcription line")]
    SpineTypeLineNotAtStart { line: usize },
    #[error("unknown spine type {raw:?}")]
    UnknownSpineType { raw: String },
    #[error("unsupported tandem interpretation {raw:?}")]
    UnsupportedTandemInterpretation { raw: String },
    #[error(
        "invalid rhythm scale {raw:?}: expected *rscale:N or *rscale:N/M with positive integers"
    )]
    InvalidRhythmScale { raw: String },
    #[error("tandem interpretation {raw:?} is only valid in **kern spines")]
    KernOnlyTandemInterpretation { raw: String },
    #[error("unsupported spine manipulator {raw:?}")]
    UnsupportedSpineManipulator { raw: String },
    #[error(
        "line {line}: mixed spine manipulator types at column {column}; expected {expected}, found {found}"
    )]
    MixedSpineManipulatorTypes {
        line: usize,
        column: usize,
        expected: &'static str,
        found: &'static str,
    },
    #[error("line {line}: invalid spine manipulator line at column {column}: {message}")]
    InvalidSpineManipulatorLine {
        line: usize,
        column: usize,
        message: String,
    },
    #[error("line {line}: spine count mismatch; expected {expected}, found {found}")]
    SpineCountMismatch {
        line: usize,
        expected: usize,
        found: usize,
    },
    #[error("invalid spine operation: {message}")]
    InvalidSpineOperation { message: String },
    #[error("line {line}: repeated barline tokens require an active spine count; found {found}")]
    RepeatedBarlineWithoutActiveSpines { line: usize, found: usize },
    #[error(
        "line {line}: inconsistent barline token at column {column}; expected {expected:?}, found {found:?}"
    )]
    InconsistentBarlineTokens {
        line: usize,
        column: usize,
        expected: String,
        found: String,
    },
    #[error("invalid clef {raw:?}")]
    InvalidClef { raw: String },
    #[error("invalid key signature {raw:?}")]
    InvalidKeySignature { raw: String },
    #[error("invalid key designation {raw:?}")]
    InvalidKeyDesignation { raw: String },
    #[error("invalid time signature {raw:?}")]
    InvalidTimeSignature { raw: String },
    #[error("invalid meter symbol {raw:?}")]
    InvalidMeterSymbol { raw: String },
    #[error("invalid stria {raw:?}")]
    InvalidStria { raw: String },
    #[error("invalid part {raw:?}")]
    InvalidPart { raw: String },
    #[error("invalid staff {raw:?}")]
    InvalidStaff { raw: String },
    #[error("invalid group {raw:?}")]
    InvalidGroup { raw: String },
    #[error("invalid instrument {raw:?}")]
    InvalidInstrument { raw: String },
    #[error("invalid tempo {raw:?}")]
    InvalidTempo { raw: String },
    #[error("invalid local comment {raw:?}")]
    InvalidLocalComment { raw: String },
    #[error("line {line}: invalid ornament convention record {raw:?} (expected `!!!ornament-convention: baroque|classical|modern|baroque-v2|classical-v2|modern-v2`, declared once)")]
    InvalidOrnamentConvention { line: usize, raw: String },
    #[error("line {line}: invalid system break {raw:?}")]
    InvalidSystemBreak { line: usize, raw: String },
    #[error("invalid ottava {raw:?}")]
    InvalidOttava { raw: String },
    #[error("invalid pedal {raw:?}")]
    InvalidPedal { raw: String },
    #[error("invalid tuplet {raw:?}")]
    InvalidTuplet { raw: String },
    #[error("invalid tremolo {raw:?}")]
    InvalidTremolo { raw: String },
    #[error("invalid bracket span {raw:?}")]
    InvalidBracketSpan { raw: String },
    #[error("invalid section label {raw:?}")]
    InvalidSectionLabel { raw: String },
    #[error("invalid expansion list {raw:?}")]
    InvalidExpansionList { raw: String },
    #[error("invalid barline {raw:?}")]
    InvalidBarline { raw: String },
    #[error("invalid rest {raw:?}")]
    InvalidRest { raw: String },
    #[error("line {line}, column {column}: invalid durationless invisible rest slice: {message}")]
    InvalidDurationlessInvisibleRestSlice {
        line: usize,
        column: usize,
        message: String,
    },
    #[error("invalid durationless invisible rest slices: {message}")]
    InvalidDurationlessInvisibleRestSlices { message: String },
    #[error("invalid fingering {raw:?}")]
    InvalidFingering { raw: String },
    #[error("invalid chord {raw:?}")]
    InvalidChord { raw: String },
    #[error("invalid note {raw:?}: {location}")]
    InvalidNote {
        raw: String,
        location: InvalidNoteLocation,
    },
    #[error("unsupported data token {raw:?}")]
    UnsupportedDataToken { raw: String },
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum InitialTimeSignatureError {
    #[error(
        "line {line}: conflicting initial time signature at column {column}; expected {expected:?}, found {found:?}"
    )]
    ConflictingInitialTimeSignature {
        line: usize,
        column: usize,
        expected: TimeSignature,
        found: TimeSignature,
    },
}
