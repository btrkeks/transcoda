//! Humdrum **kern parsing and data preparation for Transcoda.

pub mod analysis;
pub mod ast;
pub mod augmentation;
pub mod batch;
pub mod check;
#[cfg(feature = "cli")]
pub mod cli;
pub mod concatenation;
pub mod dataset_manifest;
pub mod document;
mod document_boundaries;
mod error;
mod initial_state;
pub mod layout_targets;
mod line;
mod measure;
pub mod meter;
pub mod normalization;
mod parse;
pub mod score;
pub mod serialize;
pub mod transforms;
pub mod types;
pub mod validation;

pub use augmentation::{
    augment_for_render, augment_for_render_with_report, AugmentationError,
    RenderAugmentationConfig, RenderAugmentationReport, RenderAugmentationSeed,
    TupletVisibilitySpanReport,
};
pub use batch::input::{
    collect_batch_tasks, collect_sample_transcriptions, BatchInputLayout, BatchTask,
    SampleTranscription, DEFAULT_SAMPLE_TRANSCRIPTION_NAME,
};
pub use check::{
    check_document, check_text, check_text_evidence, AcknowledgementTarget, CheckConfig,
    CheckDiagnostic, CheckEvidence, CheckPhase, CheckReport, CheckSeverity,
    CHECK_EVIDENCE_SCHEMA_VERSION, CHECK_SCHEMA_VERSION,
};
pub use error::{InitialTimeSignatureError, InvalidNoteLocation, ParseError};
pub use measure::{KernContentDurationError, MusicalDuration};
pub use meter::{
    ActiveMeter, BarLength, Meter, MeterChange, MeterConflict, MeterDeclaration, MeterDisplay,
    MeterOutput,
};
pub use normalization::batch::{
    normalize_dir, CanonicalizeNoteOrderStats, FailedExample as NormalizeFailedExample,
    FileOutcome as NormalizeFileOutcome, NormalizationPassStats, NormalizeDirOptions,
    NormalizeDirReport, StatsPayload as NormalizeStatsPayload,
};
pub use parse::{parse_kern_token, ParseDialect, ParseOptions};
pub use serialize::{
    BarNumberStyle, ContentProfile, KernDialect, SerializeError, SerializeOptions, SerializePreset,
};
pub use transforms::{slice_measure_range, SliceMeasuresError};
pub use types::{
    pitch_to_midi_pitch, Articulation, Barline, BarlineFermata, BracketSpan, BracketSpanKind,
    Chord, Clef, ClefSign, CrossStaffPlacement, Cue, DataRowRef, DecimalTempo, DocumentEvent,
    DocumentEventKind, DocumentLayout, Duration, DynamContent, DynamMarks, DynamicHairpinDirection,
    DynamicLevel, DynamicLevelValue, DynamicProvenance, EditorialInterpretation, Elision, Fifths,
    FingContent, FingerNumber, Fingering, GraceChord, GraceKind, GraceNote, Group, HairpinBegin,
    HairpinContinue, HairpinEnd, HairpinSlots, HalfNoteStemSide, Instrument, InstrumentKind,
    InstrumentTextVariant, InvalidFingerNumber, InvisibleRestSlice, KernContent, KernDocument,
    KernLine, KernMetadata, KernNoteRef, KernPitchedNote, KeyCancellation, KeyDesignation, KeyMode,
    KeySignature, LayoutText, LocalComment, LocatedKernNoteRef, MensurationSign, MeterSymbol,
    MultiMeasureRest, MxmlHarmonyContent, MxmlHarmonyLabel, Note, OrnamentConvention, Ottava,
    OttavaKind, Part, Pedal, Phrase, Pitch, RawToken, Repetition, Rest, RestPlacement, RhythmScale,
    SourceLocation, SpineBarline, SpineContent, SpineIndex, SpineManipulator, SpineManipulatorLine,
    SpineManipulatorLineKind, SpineType, Staff, StaffPlacement, Stria, SystemBreak,
    TandemInterpretation, TandemInterpretationLine, TandemInterpretationLineKind, Tempo,
    TextContent, TextPlacement, TextUnderlay, TimeSignature, TimeSignatureTandem, Tremolo, Tuplet,
    TupletBracket, VisualStyle,
};
pub use validation::quality_filter::{
    quality_filter_dir, quality_filter_dir_with_progress, QualityFilterOptions,
    QualityFilterOutcome, QualityFilterReport, QualityFilterStatsPayload,
};
pub use validation::{
    ValidationCode, ValidationConfig, ValidationError, ValidationFinding, ValidationPass,
    ValidationReport, ValidationSeverity,
};
