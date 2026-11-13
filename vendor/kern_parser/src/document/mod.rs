use crate::{
    KernContent, KernDocument, KernLine, ParseDialect, ParseError, ParseOptions, SpineContent,
};
use sha2::{Digest, Sha256};
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Framing {
    #[default]
    Complete,
    /// Missing outer wrappers imply **kern spines at the first record's width.
    Excerpt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InputOptions {
    pub dialect: ParseDialect,
    pub framing: Framing,
}

#[derive(Debug, Clone)]
pub struct Document {
    source: String,
    score: KernDocument,
    options: InputOptions,
    revision: String,
    records: Vec<Range<usize>>,
    score_lines: Vec<Option<usize>>,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct DocumentError {
    execution_failure: bool,
    code: &'static str,
    line: Option<usize>,
    field: Option<usize>,
    message: String,
    #[source]
    parse: Option<Box<ParseError>>,
}

impl DocumentError {
    pub fn is_execution_failure(&self) -> bool {
        self.execution_failure
    }

    pub(crate) fn into_execution_failure(mut self) -> Self {
        self.execution_failure = true;
        self
    }

    pub(crate) fn internal(
        code: &'static str,
        line: Option<usize>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(code, line, message).into_execution_failure()
    }

    pub fn code(&self) -> &str {
        self.code
    }
    pub fn line(&self) -> Option<usize> {
        self.line
    }
    pub fn field(&self) -> Option<usize> {
        self.field
    }
    pub fn parse_error(&self) -> Option<&ParseError> {
        self.parse.as_deref()
    }

    pub(crate) fn new(code: &'static str, line: Option<usize>, message: impl Into<String>) -> Self {
        Self {
            execution_failure: false,
            code,
            line,
            field: None,
            message: message.into(),
            parse: None,
        }
    }
    pub(crate) fn at(mut self, line: usize, field: usize) -> Self {
        self.line = Some(line);
        self.field = Some(field);
        self
    }
}

impl From<ParseError> for DocumentError {
    fn from(error: ParseError) -> Self {
        let line = match &error {
            ParseError::EmptyKernLine { line }
            | ParseError::EmptySpineToken { line, .. }
            | ParseError::InvalidSpineToken { line, .. }
            | ParseError::MixedKernLineTypes { line, .. }
            | ParseError::MixedTandemInterpretationTypes { line, .. }
            | ParseError::NullKernLine { line }
            | ParseError::SpineTypeLineNotAtStart { line }
            | ParseError::MixedSpineManipulatorTypes { line, .. }
            | ParseError::InvalidSpineManipulatorLine { line, .. }
            | ParseError::SpineCountMismatch { line, .. }
            | ParseError::RepeatedBarlineWithoutActiveSpines { line, .. }
            | ParseError::InconsistentBarlineTokens { line, .. }
            | ParseError::InvalidOrnamentConvention { line, .. }
            | ParseError::InvalidSystemBreak { line, .. }
            | ParseError::InvalidDurationlessInvisibleRestSlice { line, .. } => Some(*line),
            _ => None,
        };
        let field = match &error {
            ParseError::EmptySpineToken { column, .. }
            | ParseError::InvalidSpineToken { column, .. }
            | ParseError::MixedKernLineTypes { column, .. }
            | ParseError::MixedTandemInterpretationTypes { column, .. }
            | ParseError::MixedSpineManipulatorTypes { column, .. }
            | ParseError::InvalidSpineManipulatorLine { column, .. }
            | ParseError::InconsistentBarlineTokens { column, .. }
            | ParseError::InvalidDurationlessInvisibleRestSlice { column, .. } => {
                Some(column.saturating_sub(1))
            }
            _ => None,
        };
        Self {
            execution_failure: false,
            code: parse_code(&error),
            line,
            field,
            message: error.to_string(),
            parse: Some(Box::new(error)),
        }
    }
}

impl Document {
    pub fn parse(source: &str, options: InputOptions) -> Result<Self, DocumentError> {
        let mut offset = 0;
        let mut records = Vec::new();
        let mut first_record = None;
        for (index, record) in source.split_inclusive('\n').enumerate() {
            let text = record.strip_suffix('\n').unwrap_or(record);
            let text = text.strip_suffix('\r').unwrap_or(text);
            if text.is_empty() {
                return Err(DocumentError::new(
                    "framing.blank-record",
                    Some(index + 1),
                    "blank records are not permitted",
                ));
            }
            if first_record.is_none() && !text.starts_with("!!") {
                first_record = Some((index + 1, text));
            }
            records.push(offset..offset + text.len());
            offset += record.len();
        }
        if options.framing == Framing::Complete {
            if let Some((line, raw)) = first_record {
                if !raw.starts_with("**") {
                    return Err(DocumentError::new("framing.missing-header", Some(line), "complete input must start with exclusive interpretations such as **kern; use excerpt framing for a fragment").at(line, 0));
                }
            }
        }
        let score = KernDocument::parse_with_options(
            source,
            ParseOptions {
                dialect: options.dialect,
            },
        )?;
        let mut score_lines = vec![None; records.len()];
        for index in 0..score.lines().len() {
            if let Some(line) = score.source(index) {
                score_lines[line - 1] = Some(index);
            }
        }
        validate_lifecycle(&score, &records, &score_lines, source, options.framing)?;
        let mut digest = Sha256::new();
        digest.update(b"kern.document.v1\0");
        digest.update([
            match options.dialect {
                ParseDialect::HumdrumKern => 0,
                ParseDialect::TranscodaV2 => 1,
            },
            match options.framing {
                Framing::Complete => 0,
                Framing::Excerpt => 1,
            },
        ]);
        digest.update(source.as_bytes());
        let revision = format!("{:x}", digest.finalize());
        Ok(Self {
            source: source.to_owned(),
            score,
            options,
            revision,
            records,
            score_lines,
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn score(&self) -> &KernDocument {
        &self.score
    }
    pub fn revision(&self) -> &str {
        &self.revision
    }
    pub fn options(&self) -> InputOptions {
        self.options
    }

    pub(crate) fn data_at(&self, source_line: usize) -> Option<&[SpineContent]> {
        let index = self
            .score_lines
            .get(source_line.checked_sub(1)?)
            .copied()
            .flatten()?;
        match &self.score.lines()[index] {
            KernLine::Data(contents) => Some(contents.as_slice()),
            _ => None,
        }
    }

    /// Every `**kern` data field: its one-based line, zero-based field, text and content.
    pub(crate) fn kern_fields(&self) -> impl Iterator<Item = (usize, usize, &str, &KernContent)> {
        self.records
            .iter()
            .enumerate()
            .flat_map(move |(index, range)| {
                let contents = self.data_at(index + 1).unwrap_or_default();
                self.source[range.clone()]
                    .split('\t')
                    .zip(contents)
                    .enumerate()
                    .filter_map(move |(field, (text, content))| match content {
                        SpineContent::Kern(content) => Some((index + 1, field, text, content)),
                        _ => None,
                    })
            })
    }
}

fn validate_lifecycle(
    score: &KernDocument,
    records: &[Range<usize>],
    score_lines: &[Option<usize>],
    source: &str,
    framing: Framing,
) -> Result<(), DocumentError> {
    let mut terminated = false;
    let mut seen = false;
    for (record_index, record) in records.iter().enumerate() {
        let source_line = record_index + 1;
        let raw = &source[record.clone()];
        if raw.starts_with("!!") {
            continue;
        }
        if terminated {
            return Err(DocumentError::new(
                "framing.data-after-termination",
                Some(source_line),
                "spine records are not permitted after all spines terminate",
            ));
        }
        let Some(score_index) = score_lines[record_index] else {
            continue;
        };
        let line = &score.lines()[score_index];
        if matches!(line, KernLine::LocalCommentLine(_)) {
            continue;
        }
        seen = true;
        // A manipulator record holds one token per active spine, so it ends
        // every spine when each of its tokens is `*-`.
        terminated = crate::document_boundaries::is_terminal_line(line);
    }
    if framing == Framing::Complete && seen && !terminated {
        return Err(DocumentError::new(
            "framing.missing-terminator",
            Some(records.len().max(1)),
            "complete input must terminate every active spine with *-",
        ));
    }
    Ok(())
}

fn parse_code(error: &ParseError) -> &'static str {
    match error {
        ParseError::InvalidSpineToken { source, .. } => parse_code(source),
        ParseError::EmptyKernDocument => "parse.empty-document",
        ParseError::EmptyKernLine { .. } => "parse.empty-line",
        ParseError::EmptySpineToken { .. } => "parse.empty-field",
        ParseError::MixedKernLineTypes { .. } => "parse.mixed-record",
        ParseError::MixedTandemInterpretationTypes { .. } => "parse.mixed-interpretations",
        ParseError::NullKernLine { .. } => "parse.null-interpretation-record",
        ParseError::SpineTypeLineNotAtStart { .. } => "parse.misplaced-exclusive-interpretation",
        ParseError::UnknownSpineType { .. } => "parse.unknown-spine-type",
        ParseError::UnsupportedTandemInterpretation { .. } => "parse.unsupported-interpretation",
        ParseError::InvalidRhythmScale { .. } => "parse.invalid-rhythm-scale",
        ParseError::KernOnlyTandemInterpretation { .. } => "parse.kern-only-interpretation",
        ParseError::UnsupportedSpineManipulator { .. } => "parse.unsupported-manipulator",
        ParseError::MixedSpineManipulatorTypes { .. } => "parse.mixed-manipulators",
        ParseError::InvalidSpineManipulatorLine { .. } => "parse.invalid-manipulators",
        ParseError::SpineCountMismatch { .. } => "parse.spine-count",
        ParseError::InvalidSpineOperation { .. } => "parse.invalid-spine-operation",
        ParseError::RepeatedBarlineWithoutActiveSpines { .. } => "parse.barline-without-spines",
        ParseError::InconsistentBarlineTokens { .. } => "parse.inconsistent-barlines",
        ParseError::InvalidClef { .. } => "parse.invalid-clef",
        ParseError::InvalidKeySignature { .. } => "parse.invalid-key-signature",
        ParseError::InvalidKeyDesignation { .. } => "parse.invalid-key-designation",
        ParseError::InvalidTimeSignature { .. } => "parse.invalid-time-signature",
        ParseError::InvalidMeterSymbol { .. } => "parse.invalid-meter-symbol",
        ParseError::InvalidStria { .. } => "parse.invalid-stria",
        ParseError::InvalidPart { .. } => "parse.invalid-part",
        ParseError::InvalidStaff { .. } => "parse.invalid-staff",
        ParseError::InvalidGroup { .. } => "parse.invalid-group",
        ParseError::InvalidInstrument { .. } => "parse.invalid-instrument",
        ParseError::InvalidTempo { .. } => "parse.invalid-tempo",
        ParseError::InvalidLocalComment { .. } => "parse.invalid-local-comment",
        ParseError::InvalidOrnamentConvention { .. } => "parse.invalid-ornament-convention",
        ParseError::InvalidSystemBreak { .. } => "parse.invalid-system-break",
        ParseError::InvalidOttava { .. } => "parse.invalid-ottava",
        ParseError::InvalidPedal { .. } => "parse.invalid-pedal",
        ParseError::InvalidTuplet { .. } => "parse.invalid-tuplet",
        ParseError::InvalidTremolo { .. } => "parse.invalid-tremolo",
        ParseError::InvalidBracketSpan { .. } => "parse.invalid-bracket-span",
        ParseError::InvalidSectionLabel { .. } => "parse.invalid-section-label",
        ParseError::InvalidExpansionList { .. } => "parse.invalid-expansion-list",
        ParseError::InvalidBarline { .. } => "parse.invalid-barline",
        ParseError::InvalidRest { .. } => "parse.invalid-rest",
        ParseError::InvalidDurationlessInvisibleRestSlice { .. }
        | ParseError::InvalidDurationlessInvisibleRestSlices { .. } => "parse.invalid-rest-slice",
        ParseError::InvalidFingering { .. } => "parse.invalid-fingering",
        ParseError::InvalidChord { .. } => "parse.invalid-chord",
        ParseError::InvalidNote { .. } => "parse.invalid-note",
        ParseError::UnsupportedDataToken { .. } => "parse.unsupported-data-token",
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DisplayEnclosure {
    #[default]
    None,
    Paren,
    Brack,
}

impl Document {
    pub fn check(&self, config: &crate::check::CheckConfig) -> crate::check::CheckReport {
        crate::check::check_source_document(self, config)
    }
}
