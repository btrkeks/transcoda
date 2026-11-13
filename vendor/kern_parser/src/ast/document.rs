use std::ops::Range;

use super::origin::{LineOrigin, RecordId};
use super::{
    Barline, KernPitchedNote, Ornament, Placed, SpineContent, SpineIndex, SpineManipulator,
    SpineType, TandemInterpretation, TimeSignature,
};
use crate::{InitialTimeSignatureError, ParseError};

/// Typed representation of a complete Humdrum `**kern` document.
///
/// Example input could start with `**kern\t**kern` and contain many
/// [`KernLine`] entries for interpretations, barlines, and note data.
#[derive(Debug, Clone, Eq)]
pub struct KernDocument {
    pub metadata: KernMetadata,
    pub layout: DocumentLayout,
    lines: Vec<KernLine>,
    origins: Vec<LineOrigin>,
}

/// Equality ignores [`LineOrigin`]s: two documents with the same content are
/// equal however their lines were derived.
impl PartialEq for KernDocument {
    fn eq(&self, other: &Self) -> bool {
        self.metadata == other.metadata && self.layout == other.layout && self.lines == other.lines
    }
}

fn unzip_origins<O: Into<LineOrigin>>(
    lines: Vec<(KernLine, O)>,
) -> (Vec<KernLine>, Vec<LineOrigin>) {
    lines
        .into_iter()
        .map(|(line, origin)| (line, origin.into()))
        .unzip()
}

impl KernDocument {
    pub fn from_parts<O: Into<LineOrigin>>(
        metadata: KernMetadata,
        layout: DocumentLayout,
        lines: Vec<(KernLine, O)>,
    ) -> Self {
        let (lines, origins) = unzip_origins(lines);
        Self {
            metadata,
            layout,
            lines,
            origins,
        }
    }

    pub fn lines(&self) -> &[KernLine] {
        &self.lines
    }

    /// The 1-based source record that owns line `line_index`.
    pub fn source(&self, line_index: usize) -> Option<usize> {
        self.origins
            .get(line_index)
            .and_then(LineOrigin::owner)
            .map(RecordId::get)
    }

    /// Field-level provenance of line `line_index`.
    pub fn origin(&self, line_index: usize) -> &LineOrigin {
        &self.origins[line_index]
    }

    pub fn origins(&self) -> &[LineOrigin] {
        &self.origins
    }

    pub fn set_origin(&mut self, line_index: usize, origin: LineOrigin) {
        self.origins[line_index] = origin;
    }

    pub fn insert_line(&mut self, index: usize, line: KernLine, origin: impl Into<LineOrigin>) {
        self.lines.insert(index, line);
        self.origins.insert(index, origin.into());
        self.assert_invariants();
    }

    pub fn push_line(&mut self, line: KernLine, origin: impl Into<LineOrigin>) {
        self.lines.push(line);
        self.origins.push(origin.into());
        self.assert_invariants();
    }

    pub fn remove_line(&mut self, index: usize) -> (KernLine, LineOrigin) {
        let line = self.lines.remove(index);
        let origin = self.origins.remove(index);
        self.assert_invariants();
        (line, origin)
    }

    pub fn truncate_lines(&mut self, len: usize) {
        self.lines.truncate(len);
        self.origins.truncate(len);
        self.assert_invariants();
    }

    pub fn retain_lines(&mut self, mut keep: impl FnMut(&KernLine) -> bool) {
        let old = self.take_lines();
        self.replace_lines(old.into_iter().filter(|(line, _)| keep(line)).collect());
    }

    pub fn drain_lines(&mut self, range: Range<usize>) -> Vec<(KernLine, LineOrigin)> {
        let lines = self.lines.drain(range.clone()).collect::<Vec<_>>();
        let origins = self.origins.drain(range).collect::<Vec<_>>();
        self.assert_invariants();
        lines.into_iter().zip(origins).collect()
    }

    pub fn swap_lines(&mut self, a: usize, b: usize) {
        self.lines.swap(a, b);
        self.origins.swap(a, b);
        self.assert_invariants();
    }

    pub fn splice_lines<O: Into<LineOrigin>>(
        &mut self,
        range: Range<usize>,
        replacement: Vec<(KernLine, O)>,
    ) {
        let (lines, origins) = unzip_origins(replacement);
        self.lines.splice(range.clone(), lines);
        self.origins.splice(range, origins);
        self.assert_invariants();
    }

    pub fn replace_lines<O: Into<LineOrigin>>(&mut self, lines: Vec<(KernLine, O)>) {
        let (lines, origins) = unzip_origins(lines);
        self.lines = lines;
        self.origins = origins;
        self.assert_invariants();
    }

    pub fn take_lines(&mut self) -> Vec<(KernLine, LineOrigin)> {
        let lines = std::mem::take(&mut self.lines);
        let origins = std::mem::take(&mut self.origins);
        self.assert_invariants();
        lines.into_iter().zip(origins).collect()
    }

    pub fn lines_mut(&mut self) -> &mut [KernLine] {
        &mut self.lines
    }

    fn assert_invariants(&self) {
        debug_assert_eq!(self.lines.len(), self.origins.len());
    }

    /// Iterate over parsed Humdrum document events with source line identity.
    ///
    /// This is a low-level structural stream over existing parsed lines. It
    /// does not expand repeats, build a musical timeline, or interpret events
    /// for playback.
    pub fn events(&self) -> impl Iterator<Item = DocumentEvent<'_>> {
        self.lines
            .iter()
            .enumerate()
            .map(move |(line_index, line)| DocumentEvent {
                line_number: self.source_line_number(line_index),
                line_index,
                kind: DocumentEventKind::from(line),
            })
    }

    /// Iterate over parsed data rows with their 1-based source line numbers.
    ///
    /// Documents produced directly by [`KernDocument::parse`] preserve exact
    /// source line numbers, excluding skipped blank/global-comment records.
    /// Generated or transformed documents fall back to stored line indexes when
    /// no source line number is available for a row.
    pub fn data_rows(&self) -> impl Iterator<Item = DataRowRef<'_>> {
        self.lines
            .iter()
            .enumerate()
            .filter_map(move |(line_index, line)| {
                let KernLine::Data(contents) = line else {
                    return None;
                };

                Some(DataRowRef {
                    line_number: self.source_line_number(line_index),
                    contents,
                })
            })
    }

    /// Iterate over every pitched notehead, grace notes included, with its
    /// source location.
    ///
    /// Rests, nulls, and non-`**kern` spines are skipped. Chord and grace-chord
    /// noteheads preserve their zero-based note indexes.
    pub fn located_noteheads(&self) -> impl Iterator<Item = LocatedKernNoteRef<'_>> {
        self.lines
            .iter()
            .enumerate()
            .flat_map(move |(line_index, line)| {
                let KernLine::Data(contents) = line else {
                    return Vec::new();
                };

                let line_number = self.source_line_number(line_index);
                contents
                    .iter()
                    .enumerate()
                    .flat_map(move |(column_index, content)| {
                        let SpineContent::Kern(content) = content else {
                            return Vec::new();
                        };

                        let token_ornaments = content.token_ornaments();
                        content
                            .noteheads()
                            .map(move |notehead| LocatedKernNoteRef {
                                location: SourceLocation {
                                    line_number,
                                    line_index,
                                    spine_index: SpineIndex(column_index),
                                    chord_note_index: notehead.chord_note_index,
                                },
                                note: notehead.note,
                                token_ornaments,
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
    }

    pub(crate) fn source_line_number(&self, line_index: usize) -> usize {
        match self.origins.get(line_index) {
            Some(LineOrigin::Generated) => 0,
            Some(origin) => origin.owner().map_or(line_index + 1, RecordId::get),
            None => line_index + 1,
        }
    }

    /// Return the half-open range containing the leading tandem header block.
    ///
    /// The header starts after an optional exclusive interpretation line and
    /// includes the initial contiguous run of tandem interpretation lines.
    pub fn leading_tandem_header_range(&self) -> Range<usize> {
        let start = usize::from(matches!(
            self.lines.first(),
            Some(KernLine::SpineTypeLine(_))
        ));
        let end = self.lines[start..]
            .iter()
            .position(|line| !matches!(line, KernLine::TandemInterpretationLine(_)))
            .map_or(self.lines.len(), |offset| start + offset);

        start..end
    }

    /// Return the leading contiguous block of tandem interpretation lines.
    pub fn leading_tandem_header_lines(&self) -> &[KernLine] {
        let range = self.leading_tandem_header_range();
        &self.lines[range]
    }

    /// Return the declared initial time signature from the leading tandem header.
    ///
    /// Numeric meter interpretations (`*M`) and meter symbols (`*met`) in
    /// the leading tandem header must agree when both are present.
    pub fn declared_initial_time_signature(
        &self,
    ) -> Result<Option<TimeSignature>, InitialTimeSignatureError> {
        let mut initial: Option<TimeSignature> = None;
        let header_range = self.leading_tandem_header_range();

        for (line_index, line) in self
            .lines
            .iter()
            .enumerate()
            .skip(header_range.start)
            .take(header_range.len())
        {
            let KernLine::TandemInterpretationLine(tandems) = line else {
                continue;
            };

            for (column_index, tandem) in tandems.tokens().iter().enumerate() {
                let Some(time_signature) = crate::meter::implied_time_signature(tandem) else {
                    continue;
                };

                if let Some(expected) = &initial {
                    if expected != &time_signature {
                        return Err(InitialTimeSignatureError::ConflictingInitialTimeSignature {
                            line: self.source_line_number(line_index),
                            column: column_index + 1,
                            expected: *expected,
                            found: time_signature,
                        });
                    }
                } else {
                    initial = Some(time_signature);
                }
            }
        }

        Ok(initial)
    }

    /// Return the initial time signature, defaulting to legacy Humdrum `**kern` 4/4.
    ///
    /// New measure-sensitive code should prefer
    /// [`crate::analysis::resolve_initial_time_signature`] so missing meters are
    /// inferred or reported explicitly instead of silently becoming 4/4.
    pub fn initial_time_signature(&self) -> Result<TimeSignature, InitialTimeSignatureError> {
        Ok(self
            .declared_initial_time_signature()?
            .unwrap_or_else(default_time_signature))
    }

    /// Number of staves represented by the document's initial `**kern` spines.
    ///
    /// Humdrum encodes the initial staff layout with the opening exclusive
    /// interpretation line. Non-`**kern` spines such as `**dynam` and `**text`
    /// do not contribute to the staff count. Documents without an explicit
    /// spine-type line are parsed as all-`**kern` slices.
    pub fn staff_count(&self) -> usize {
        match self.lines.first() {
            Some(KernLine::SpineTypeLine(spine_types)) => spine_types
                .iter()
                .filter(|spine_type| matches!(spine_type, SpineType::Kern))
                .count(),
            _ => self.layout.initial_spine_count as usize,
        }
    }

    /// Convenience wrapper for [`crate::normalization::omr::normalize_for_omr`].
    ///
    /// New internal code should call the module function directly so
    /// [`KernDocument`] stays focused on structural document behavior.
    pub fn normalize_for_omr(&mut self) {
        crate::normalization::omr::normalize_for_omr(self);
    }

    /// Convenience wrapper for [`crate::normalization::omr::normalized_for_omr`].
    ///
    /// New internal code should call the module function directly so
    /// [`KernDocument`] stays focused on structural document behavior.
    pub fn normalized_for_omr(&self) -> Self {
        crate::normalization::omr::normalized_for_omr(self)
    }

    /// Convenience wrapper for [`crate::serialize::omr::to_string`].
    ///
    /// Prefer serializer module APIs in new code.
    pub fn to_omr_string(&self) -> String {
        crate::serialize::omr::to_string(self)
    }

    /// Convenience wrapper for [`crate::serialize::SerializePreset::TranscodaV2`].
    ///
    /// Prefer [`crate::serialize::SerializePreset`] in new code.
    pub fn to_transcoda_v2_string(&self) -> String {
        crate::serialize::SerializePreset::TranscodaV2.serialize_document(self)
    }

    /// Convenience wrapper for [`crate::serialize::SerializePreset::HumdrumKern`].
    ///
    /// Prefer [`crate::serialize::SerializePreset`] in new code.
    pub fn to_humdrum_kern_string(&self) -> String {
        crate::serialize::SerializePreset::HumdrumKern.serialize_document(self)
    }

    /// Convenience wrapper for [`crate::serialize::render::to_string`].
    ///
    /// Prefer serializer module APIs in new code.
    pub fn to_render_string(&self) -> Result<String, crate::serialize::DialectError> {
        crate::serialize::render::to_string(self)
    }

    /// Convenience wrapper for [`crate::validation::validate`].
    ///
    /// Prefer validation module APIs in new code.
    pub fn validate(&self) -> Result<(), crate::ValidationError> {
        crate::validation::validate(self)
    }

    /// Convenience wrapper for [`crate::validation::validate_rhythm`].
    ///
    /// Prefer validation module APIs in new code.
    pub fn validate_rhythm(&self) -> Result<(), crate::ValidationError> {
        crate::validation::validate_rhythm(self)
    }
}

/// Borrowed view of one Humdrum data row in a [`KernDocument`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataRowRef<'a> {
    pub line_number: usize,
    pub contents: &'a [SpineContent],
}

/// Borrowed low-level event in a [`KernDocument`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocumentEvent<'a> {
    /// 1-based source line number from the original Humdrum input when known.
    pub line_number: usize,
    /// Zero-based parsed document line index.
    pub line_index: usize,
    pub kind: DocumentEventKind<'a>,
}

/// Borrowed payload for one [`DocumentEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentEventKind<'a> {
    SpineTypeLine(&'a [SpineType]),
    TandemInterpretationLine(&'a TandemInterpretationLine),
    SpineManipulatorLine(&'a SpineManipulatorLine),
    LocalCommentLine(&'a [LocalComment]),
    Barline(&'a Barline),
    Data(&'a [SpineContent]),
}

impl<'a> From<&'a KernLine> for DocumentEventKind<'a> {
    fn from(line: &'a KernLine) -> Self {
        match line {
            KernLine::SpineTypeLine(spine_types) => Self::SpineTypeLine(spine_types),
            KernLine::TandemInterpretationLine(tandems) => Self::TandemInterpretationLine(tandems),
            KernLine::SpineManipulatorLine(manipulators) => {
                Self::SpineManipulatorLine(manipulators)
            }
            KernLine::LocalCommentLine(comments) => Self::LocalCommentLine(comments),
            KernLine::Barline(barline) => Self::Barline(barline),
            KernLine::Data(contents) => Self::Data(contents),
        }
    }
}

/// Source identity for a pitched notehead, regular or grace, inside a [`KernDocument`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLocation {
    /// 1-based source line number from the original Humdrum input when known.
    pub line_number: usize,
    /// Zero-based parsed document line index.
    pub line_index: usize,
    /// Zero-based spine/column index on the Humdrum line.
    pub spine_index: SpineIndex,
    /// Zero-based source-subtoken index, or `None` for single-note tokens.
    pub chord_note_index: Option<usize>,
}

/// Borrowed view of one pitched notehead, regular or grace, with its source location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocatedKernNoteRef<'a> {
    pub location: SourceLocation,
    pub note: KernPitchedNote<'a>,
    pub token_ornaments: &'a [Placed<Ornament>],
}

impl<'a> LocatedKernNoteRef<'a> {
    pub fn token_ornaments(&self) -> &'a [Placed<Ornament>] {
        self.token_ornaments
    }
}

fn default_time_signature() -> TimeSignature {
    TimeSignature {
        numerator: 4,
        denominator: 4,
    }
}

/// Performance-practice convention a source's ornament signs follow, from
/// `!!!ornament-convention: baroque`.
///
/// The record decides how ambiguous ornament signs are performed and never
/// changes engraving. Each value names a fixed table of readings: a later
/// change to what a value means is a new value, never a reinterpretation of
/// sources that already carry the old one. An absent record means
/// [`OrnamentConvention::Modern`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OrnamentConvention {
    /// Plain-squiggle mordent signs (`M`/`m`) are short trills from above
    /// (J. S. Bach's 1720 Explication, C. P. E. Bach).
    Baroque,
    /// Same readings as [`OrnamentConvention::Baroque`] today (L. Mozart,
    /// Türk keep the upper start); a separate value so a later divergence
    /// needs no re-authoring of files.
    Classical,
    /// Plain-squiggle mordent signs are inverted mordents: main, upper, main.
    Modern,
    /// [`OrnamentConvention::Baroque`], plus: written trills (`T`/`t`) start
    /// on the upper note, unless the previous token in their spine is a grace
    /// one diatonic step above, which is then the written upper start; and a
    /// trill line continued onto a tied note holds that note instead of
    /// starting a second trill.
    #[serde(rename = "baroque-v2")]
    BaroqueV2,
    /// [`OrnamentConvention::Classical`], plus the same upper-start trills and
    /// held tied continuations as [`OrnamentConvention::BaroqueV2`].
    #[serde(rename = "classical-v2")]
    ClassicalV2,
    /// [`OrnamentConvention::Modern`], plus held tied trill continuations as
    /// in [`OrnamentConvention::BaroqueV2`]. Written trills keep their main
    /// note start.
    #[serde(rename = "modern-v2")]
    ModernV2,
}

impl OrnamentConvention {
    /// The reference record key, as in `!!!ornament-convention: baroque`.
    pub const RECORD_KEY: &'static str = "ornament-convention";

    pub const ALL: [Self; 6] = [
        Self::Baroque,
        Self::Classical,
        Self::Modern,
        Self::BaroqueV2,
        Self::ClassicalV2,
        Self::ModernV2,
    ];

    /// Parses the record value. Exact lowercase match only.
    pub fn from_record_value(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|convention| convention.record_value() == value)
    }

    /// The record value, the inverse of [`Self::from_record_value`].
    pub fn record_value(self) -> &'static str {
        match self {
            Self::Baroque => "baroque",
            Self::Classical => "classical",
            Self::Modern => "modern",
            Self::BaroqueV2 => "baroque-v2",
            Self::ClassicalV2 => "classical-v2",
            Self::ModernV2 => "modern-v2",
        }
    }
}

/// Parsed formal metadata from supported Humdrum reference records.
///
/// Examples:
/// - `!!!OTL: Fugue in C minor` maps to [`KernMetadata::title`].
/// - `!!!COM: Bach, Johann Sebastian` maps to [`KernMetadata::composer`].
/// - `!!!OPS: BWV 847` maps to [`KernMetadata::opus`].
/// - `!!!OMV: 2` maps to [`KernMetadata::movement`].
/// - `!!!ODT: 1722` maps to [`KernMetadata::date`].
/// - `!!!URL: https://example.test/source.krn` can map to [`KernMetadata::source`].
/// - `!!!ornament-convention: baroque` maps to [`KernMetadata::ornament_convention`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KernMetadata {
    /// Work title from `!!!OTL:`.
    ///
    /// Example: `!!!OTL: Fugue in C minor`.
    pub title: Option<String>,
    /// Composer/creator from `!!!COM:`.
    ///
    /// Example: `!!!COM: Bach, Johann Sebastian`.
    pub composer: Option<String>,
    /// Parent work title from `!!!OPR:`.
    ///
    /// Example: `!!!OPR: Piano Sonata No. 14`.
    pub parent_work: Option<String>,
    /// Alternative/popular titles from repeatable `!!!OTA:` records.
    ///
    /// Language-qualified records (`!!!OTA@@DE:`) are folded into plain
    /// values; the qualifier is dropped.
    ///
    /// Example: `!!!OTA: Moonlight Sonata`.
    pub alternative_titles: Vec<String>,
    /// Opus, catalog, or work number from `!!!OPS:`.
    ///
    /// Example: `!!!OPS: BWV 847`.
    pub opus: Option<String>,
    /// Movement number or movement name from `!!!OMV:`.
    ///
    /// Example: `!!!OMV: 2`.
    pub movement: Option<String>,
    /// Movement designation/name from `!!!OMD:`.
    ///
    /// Example: `!!!OMD: Allegro con brio`.
    pub movement_designation: Option<String>,
    /// Date associated with the work or source from `!!!ODT:`.
    ///
    /// Example: `!!!ODT: 1722`.
    pub date: Option<String>,
    /// Source identifier or URL from a supported source reference record.
    ///
    /// Examples: `!!!URL: https://example.test/fugue.krn`, `!!!SCT: manuscript`.
    pub source: Option<String>,
    /// Ornament performance convention from `!!!ornament-convention:`.
    ///
    /// `None` when the source declares none; consumers read that as
    /// [`OrnamentConvention::Modern`]. Unlike the bibliographic records, an
    /// unknown or repeated value is a parse error because it changes how the
    /// score is performed.
    pub ornament_convention: Option<OrnamentConvention>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentLayout {
    pub initial_spine_count: u32,
}

/// Top-level category for a Humdrum line.
///
/// Examples: `!! title` is a global comment, `*clefG2` is a tandem
/// interpretation, `4c` is data, and `=1` is a barline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernLine {
    SpineTypeLine(Vec<SpineType>),
    TandemInterpretationLine(TandemInterpretationLine),
    SpineManipulatorLine(SpineManipulatorLine),
    LocalCommentLine(Vec<LocalComment>),
    Barline(Barline),
    Data(Vec<SpineContent>),
}

/// Global layout system break record: `!!LO:LB:g=original`.
///
/// This global comment kind attaches to the following [`Barline`] as
/// [`Barline::system_break_before`]. It spans the whole system rather than
/// individual spines, so it must never be treated as measure-body content.
/// Equality compares the layout group; source provenance is not musical content.
#[derive(Debug, Clone, Eq)]
pub struct SystemBreak {
    /// Layout group, e.g. `original` from `g=original`.
    pub group: Option<String>,
    /// 1-based source line for the original `!!LO:LB` record when known.
    pub source_line: Option<usize>,
}

impl PartialEq for SystemBreak {
    fn eq(&self, other: &Self) -> bool {
        self.group == other.group
    }
}

/// Local comment token in a spine-specific Humdrum comment line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalComment {
    /// `!`
    Null,
    /// `!LO:TX:a:t=Allegro` / `!LO:TX:b:i:t=dolce`.
    LayoutText(LayoutText),

    /// Unsupported local comment text.
    ///
    /// This keeps corpus-authored `!LO` variants parseable even when we do not
    /// yet model their exact semantics. Extend the typed variants above when a
    /// specific local comment starts carrying useful render or target signal.
    Unsupported(String),
}

/// Minimal VHV/Verovio layout text direction used by render augmentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutText {
    pub placement: TextPlacement,
    pub italic: bool,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextPlacement {
    Above,
    Below,
}

/// A Humdrum spine-path manipulator line.
///
/// Split and merge operations may appear on the same line, but terminations
/// must remain separate from other non-null spine-path operations. Merge
/// operations are local groups of adjacent `*v` tokens. A lone `*v` cannot
/// identify which neighboring spine it should merge with, so every contiguous
/// merge run must contain at least two tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpineManipulatorLine {
    tokens: Vec<SpineManipulator>,
}

impl SpineManipulatorLine {
    pub fn new(tokens: Vec<SpineManipulator>) -> Result<Self, ParseError> {
        Self::new_on_line(0, tokens)
    }

    pub fn new_on_line(line: usize, tokens: Vec<SpineManipulator>) -> Result<Self, ParseError> {
        let mut non_terminate_kind = SpineManipulatorLineKind::Null;
        let mut terminate_column = None;
        let mut index = 0;

        while index < tokens.len() {
            let token_kind = SpineManipulatorLineKind::from_manipulator(&tokens[index]);

            match token_kind {
                SpineManipulatorLineKind::Null => {}
                SpineManipulatorLineKind::Terminate => {
                    if !matches!(non_terminate_kind, SpineManipulatorLineKind::Null) {
                        return Err(ParseError::MixedSpineManipulatorTypes {
                            line,
                            column: index + 1,
                            expected: non_terminate_kind.name(),
                            found: token_kind.name(),
                        });
                    }
                    terminate_column = Some(index + 1);
                }
                SpineManipulatorLineKind::Split | SpineManipulatorLineKind::Merge => {
                    if let Some(column) = terminate_column {
                        return Err(ParseError::MixedSpineManipulatorTypes {
                            line,
                            column,
                            expected: token_kind.name(),
                            found: SpineManipulatorLineKind::Terminate.name(),
                        });
                    }
                    if matches!(non_terminate_kind, SpineManipulatorLineKind::Null) {
                        non_terminate_kind = token_kind;
                    }
                }
            }

            if !matches!(tokens[index], SpineManipulator::Merge) {
                index += 1;
                continue;
            }

            let column = index + 1;
            let mut merge_count = 0;
            while index < tokens.len() && matches!(tokens[index], SpineManipulator::Merge) {
                merge_count += 1;
                index += 1;
            }

            if merge_count < 2 {
                return Err(ParseError::InvalidSpineManipulatorLine {
                    line,
                    column,
                    message: "merge requires at least 2 consecutive *v tokens".to_owned(),
                });
            }
        }

        Ok(Self { tokens })
    }

    pub fn tokens(&self) -> &[SpineManipulator] {
        &self.tokens
    }

    pub fn into_tokens(self) -> Vec<SpineManipulator> {
        self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpineManipulatorLineKind {
    Null,
    Terminate,
    Split,
    Merge,
}

impl SpineManipulatorLineKind {
    pub fn from_manipulator(manipulator: &SpineManipulator) -> Self {
        match manipulator {
            SpineManipulator::Null => Self::Null,
            SpineManipulator::Terminate => Self::Terminate,
            SpineManipulator::Split => Self::Split,
            SpineManipulator::Merge => Self::Merge,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Terminate => "terminate",
            Self::Split => "split",
            Self::Merge => "merge",
        }
    }
}

/// A homogeneous Humdrum tandem interpretation line.
///
/// Every non-null token in the line has the same
/// [`TandemInterpretationLineKind`]. Null tokens may appear in any column.
#[derive(Debug, Clone, Eq)]
pub struct TandemInterpretationLine {
    kind: TandemInterpretationLineKind,
    tokens: Vec<TandemInterpretation>,
    generated_tokens: Vec<bool>,
}

impl PartialEq for TandemInterpretationLine {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.tokens == other.tokens
    }
}

impl TandemInterpretationLine {
    pub fn new(tokens: Vec<TandemInterpretation>) -> Result<Self, ParseError> {
        Self::new_on_line(0, tokens)
    }

    pub fn new_on_line(line: usize, tokens: Vec<TandemInterpretation>) -> Result<Self, ParseError> {
        let mut kind = TandemInterpretationLineKind::Null;

        for (index, tandem) in tokens.iter().enumerate() {
            let token_kind = TandemInterpretationLineKind::from_tandem(tandem);
            if matches!(token_kind, TandemInterpretationLineKind::Null) {
                continue;
            }

            if matches!(kind, TandemInterpretationLineKind::Null) {
                kind = token_kind;
                continue;
            }

            if kind != token_kind {
                return Err(ParseError::MixedTandemInterpretationTypes {
                    line,
                    column: index + 1,
                    expected: kind.name(),
                    found: token_kind.name(),
                });
            }
        }

        let generated_tokens = vec![false; tokens.len()];
        Ok(Self {
            kind,
            tokens,
            generated_tokens,
        })
    }

    pub(crate) fn from_homogeneous_tokens(tokens: Vec<TandemInterpretation>) -> Self {
        Self::new(tokens).expect("tandem interpretation line tokens are homogeneous")
    }

    pub(crate) fn from_homogeneous_tokens_with_generated(
        tokens: Vec<TandemInterpretation>,
        generated_tokens: Vec<bool>,
    ) -> Self {
        let mut line = Self::from_homogeneous_tokens(tokens);
        if generated_tokens.len() == line.tokens.len() {
            line.generated_tokens = generated_tokens;
        }
        line
    }

    pub fn kind(&self) -> TandemInterpretationLineKind {
        self.kind
    }

    pub fn tokens(&self) -> &[TandemInterpretation] {
        &self.tokens
    }

    pub fn tokens_mut(&mut self) -> &mut [TandemInterpretation] {
        &mut self.tokens
    }

    pub(crate) fn mark_generated_token(&mut self, index: usize) {
        if let Some(generated) = self.generated_tokens.get_mut(index) {
            *generated = true;
        }
    }

    pub(crate) fn is_generated_token(&self, index: usize) -> bool {
        self.generated_tokens.get(index).copied().unwrap_or(false)
    }

    pub(crate) fn generated_tokens(&self) -> &[bool] {
        &self.generated_tokens
    }

    pub fn into_tokens(self) -> Vec<TandemInterpretation> {
        self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TandemInterpretationLineKind {
    Null,
    Clef,
    KeySignature,
    KeyDesignation,
    TimeSignature,
    MeterSymbol,
    Ottava,
    Pedal,
    Tuplet,
    TupletBracket,
    Repetition,
    TupletBreak,
    Stria,
    Part,
    Staff,
    Group,
    Instrument,
    Tempo,
    Tremolo,
    BracketSpan,
    SectionLabel,
    ExpansionList,
    VisualStyle,
    Cue,
    RhythmScale,
    Raw,
}

impl TandemInterpretationLineKind {
    pub fn from_tandem(tandem: &TandemInterpretation) -> Self {
        match tandem {
            TandemInterpretation::Null => Self::Null,
            TandemInterpretation::Clef(_) => Self::Clef,
            TandemInterpretation::KeySignature(_) => Self::KeySignature,
            TandemInterpretation::KeyDesignation(_) => Self::KeyDesignation,
            TandemInterpretation::TimeSignature(_) | TandemInterpretation::FreeMeter => {
                Self::TimeSignature
            }
            TandemInterpretation::MeterSymbol(_) => Self::MeterSymbol,
            TandemInterpretation::Ottava(_) => Self::Ottava,
            TandemInterpretation::Pedal(_) => Self::Pedal,
            TandemInterpretation::Tuplet(_) => Self::Tuplet,
            TandemInterpretation::TupletBracket(_) => Self::TupletBracket,
            TandemInterpretation::Repetition(_) => Self::Repetition,
            TandemInterpretation::TupletBreak => Self::TupletBreak,
            TandemInterpretation::Stria(_) => Self::Stria,
            TandemInterpretation::Part(_) => Self::Part,
            TandemInterpretation::Staff(_) => Self::Staff,
            TandemInterpretation::Group(_) => Self::Group,
            TandemInterpretation::Instrument(_) => Self::Instrument,
            TandemInterpretation::Tempo(_) => Self::Tempo,
            TandemInterpretation::Tremolo(_) => Self::Tremolo,
            TandemInterpretation::BracketSpan(_) => Self::BracketSpan,
            TandemInterpretation::SectionLabel(_) => Self::SectionLabel,
            TandemInterpretation::ExpansionList(_) => Self::ExpansionList,
            TandemInterpretation::VisualStyle(_) => Self::VisualStyle,
            TandemInterpretation::Cue(_) => Self::Cue,
            TandemInterpretation::RhythmScale(_) => Self::RhythmScale,
            TandemInterpretation::Raw(_) => Self::Raw,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Clef => "clef",
            Self::KeySignature => "key signature",
            Self::KeyDesignation => "key designation",
            Self::TimeSignature => "time signature",
            Self::MeterSymbol => "meter symbol",
            Self::Ottava => "ottava",
            Self::Pedal => "pedal",
            Self::Tuplet => "tuplet",
            Self::TupletBracket => "tuplet bracket",
            Self::Repetition => "repetition",
            Self::TupletBreak => "tuplet break",
            Self::Stria => "stria",
            Self::Part => "part",
            Self::Staff => "staff",
            Self::Group => "group",
            Self::Instrument => "instrument",
            Self::Tempo => "tempo",
            Self::Tremolo => "tremolo",
            Self::BracketSpan => "bracket span",
            Self::SectionLabel => "section label",
            Self::ExpansionList => "expansion list",
            Self::VisualStyle => "visual style",
            Self::Cue => "cue",
            Self::RhythmScale => "rhythm scale",
            Self::Raw => "raw",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        types::{AuxInterval, DiatonicStep, Ornament, Turn, TurnForm, TurnIntervals, TurnTiming},
        InitialTimeSignatureError, KernContent,
    };

    fn parse(input: &str) -> KernDocument {
        KernDocument::parse(input).expect("test input parses")
    }

    fn time_signature(numerator: u32, denominator: u32) -> TimeSignature {
        TimeSignature {
            numerator,
            denominator,
        }
    }

    fn turn(
        form: TurnForm,
        timing: TurnTiming,
        intervals: Option<(AuxInterval, AuxInterval)>,
    ) -> Ornament {
        Ornament::Turn(Turn {
            form,
            timing,
            intervals: intervals.map(|(upper, lower)| TurnIntervals { upper, lower }),
        })
    }

    #[test]
    fn initial_time_signature_defaults_to_four_four() {
        let document = parse("**kern\n4c\n*-\n");

        assert_eq!(
            document.initial_time_signature().unwrap(),
            time_signature(4, 4)
        );
    }

    #[test]
    fn declared_initial_time_signature_returns_none_without_meter() {
        let document = parse("**kern\n4c\n*-\n");

        assert_eq!(document.declared_initial_time_signature().unwrap(), None);
    }

    #[test]
    fn declared_initial_time_signature_uses_numeric_meter() {
        let document = parse("**kern\n*M3/4\n4c\n*-\n");

        assert_eq!(
            document.declared_initial_time_signature().unwrap(),
            Some(time_signature(3, 4))
        );
    }

    #[test]
    fn initial_time_signature_uses_numeric_meter() {
        let document = parse("**kern\n*M3/4\n4c\n*-\n");

        assert_eq!(
            document.initial_time_signature().unwrap(),
            time_signature(3, 4)
        );
    }

    #[test]
    fn initial_time_signature_maps_common_meter_symbol() {
        for input in ["**kern\n*met(C)\n4c\n*-\n", "**kern\n*met(c)\n4c\n*-\n"] {
            let document = parse(input);

            assert_eq!(
                document.initial_time_signature().unwrap(),
                time_signature(4, 4)
            );
        }
    }

    #[test]
    fn initial_time_signature_maps_cut_meter_symbol() {
        for input in ["**kern\n*met(C|)\n4c\n*-\n", "**kern\n*met(c|)\n4c\n*-\n"] {
            let document = parse(input);

            assert_eq!(
                document.initial_time_signature().unwrap(),
                time_signature(2, 2)
            );
        }
    }

    #[test]
    fn initial_time_signature_allows_compatible_numeric_meter_and_meter_symbol() {
        let document = parse("**kern\n*M4/4\n*met(C)\n4c\n*-\n");

        assert_eq!(
            document.initial_time_signature().unwrap(),
            time_signature(4, 4)
        );
    }

    #[test]
    fn initial_time_signature_errors_on_conflicting_initial_declarations() {
        let document = parse("**kern\n*M3/4\n*met(C)\n4c\n*-\n");
        let error = document.initial_time_signature().unwrap_err();

        assert_eq!(
            error,
            InitialTimeSignatureError::ConflictingInitialTimeSignature {
                line: 3,
                column: 1,
                expected: time_signature(3, 4),
                found: time_signature(4, 4),
            }
        );
    }

    #[test]
    fn initial_time_signature_ignores_later_meter_changes() {
        let document = parse("**kern\n*M3/4\n4c\n*M4/4\n4d\n*-\n");

        assert_eq!(
            document.initial_time_signature().unwrap(),
            time_signature(3, 4)
        );
    }

    #[test]
    fn initial_time_signature_does_not_use_meter_after_non_header_line() {
        for input in [
            "**kern\n4c\n*M3/4\n*-\n",
            "**kern\n=\n*M3/4\n*-\n",
            "**kern\n*^\n*M3/4\t*M3/4\n*-\t*-\n",
        ] {
            let document = parse(input);

            assert_eq!(
                document.initial_time_signature().unwrap(),
                time_signature(4, 4)
            );
            assert_eq!(document.declared_initial_time_signature().unwrap(), None);
        }
    }

    #[test]
    fn leading_tandem_header_range_stops_before_non_tandem_lines() {
        let document = parse("**kern\n*clefG2\n*k[]\n=\n*M3/4\n4c\n*-\n");

        assert_eq!(document.leading_tandem_header_range(), 1..3);
        assert_eq!(document.leading_tandem_header_lines().len(), 2);
    }

    #[test]
    fn leading_tandem_header_range_starts_at_zero_without_spine_type_line() {
        let document = parse("*clefG2\n*k[]\n4c\n");

        assert_eq!(document.leading_tandem_header_range(), 0..2);
        assert_eq!(document.leading_tandem_header_lines().len(), 2);
    }

    #[test]
    fn data_rows_yield_data_contents_with_source_line_numbers() {
        let document = parse(
            "!!!COM: Example\n\
             !! global comment\n\
             **kern\t**dynam\n\
             *M4/4\t*\n\
             =1\t=1\n\
             4c\tp\n\
             4d 4f\t.\n\
             *-\t*-\n",
        );

        let rows: Vec<_> = document.data_rows().collect();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].line_number, 6);
        assert_eq!(rows[1].line_number, 7);
        assert!(matches!(rows[0].contents[0], SpineContent::Kern(_)));
        assert!(matches!(rows[0].contents[1], SpineContent::Dynam(_)));
        assert!(matches!(
            rows[1].contents[0],
            SpineContent::Kern(KernContent::Chord(_))
        ));
    }

    #[test]
    fn events_yield_all_line_kinds_with_source_line_numbers() {
        let document = parse(
            "!!!COM: Example\n\
             !! global comment\n\
             **kern\t**dynam\n\
             *M4/4\t*\n\
             !LO:TX:a:t=Allegro\t!\n\
             =1\t=1\n\
             4c\tp\n\
             *-\t*-\n",
        );

        let events: Vec<_> = document.events().collect();

        assert_eq!(events.len(), 8);
        assert_eq!(events[0].line_number, 3);
        assert!(matches!(
            events[0].kind,
            DocumentEventKind::SpineTypeLine(spine_types) if spine_types.len() == 2
        ));
        assert_eq!(events[1].line_number, 0);
        assert!(matches!(
            events[1].kind,
            DocumentEventKind::TandemInterpretationLine(tandems) if tandems.len() == 2
        ));
        assert_eq!(events[2].line_number, 0);
        assert!(matches!(
            events[2].kind,
            DocumentEventKind::TandemInterpretationLine(tandems) if tandems.len() == 2
        ));
        assert!(matches!(
            events[3].kind,
            DocumentEventKind::TandemInterpretationLine(tandems) if tandems.len() == 2
        ));
        assert!(matches!(
            events[4].kind,
            DocumentEventKind::LocalCommentLine(comments) if comments.len() == 2
        ));
        assert!(matches!(events[5].kind, DocumentEventKind::Barline(_)));
        assert!(matches!(
            events[6].kind,
            DocumentEventKind::Data(contents)
                if matches!(contents[0], SpineContent::Kern(_))
                    && matches!(contents[1], SpineContent::Dynam(_))
        ));
        assert!(matches!(
            events[7].kind,
            DocumentEventKind::SpineManipulatorLine(manipulators) if manipulators.len() == 2
        ));
    }

    #[test]
    fn events_fall_back_to_source_line_numbers_for_constructed_documents() {
        let document = KernDocument::from_parts(
            KernMetadata::default(),
            DocumentLayout {
                initial_spine_count: 1,
            },
            vec![(
                KernLine::Data(vec![SpineContent::Kern(KernContent::Null)]),
                None,
            )],
        );

        let events: Vec<_> = document.events().collect();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].line_number, 1);
        assert_eq!(events[0].line_index, 0);
        assert!(matches!(events[0].kind, DocumentEventKind::Data(_)));
    }

    #[test]
    fn located_noteheads_yields_single_note_location() {
        let document = parse("**kern\n4c\n*-\n");
        let noteheads: Vec<_> = document.located_noteheads().collect();

        assert_eq!(noteheads.len(), 1);
        assert_eq!(
            noteheads[0].location,
            SourceLocation {
                line_number: 2,
                line_index: 3,
                spine_index: SpineIndex(0),
                chord_note_index: Some(0),
            }
        );
        assert_eq!(noteheads[0].note.pitch().step, DiatonicStep::C);
    }

    #[test]
    fn located_noteheads_preserves_source_chord_note_indexes() {
        let document = parse("**kern\n4g 4c 4e\n*-\n");
        let noteheads: Vec<_> = document
            .located_noteheads()
            .map(|notehead| {
                (
                    notehead.location.chord_note_index,
                    notehead.note.pitch().step,
                )
            })
            .collect();

        assert_eq!(
            noteheads,
            vec![
                (Some(0), DiatonicStep::G),
                (Some(1), DiatonicStep::C),
                (Some(2), DiatonicStep::E),
            ]
        );
    }

    #[test]
    fn located_noteheads_skip_rests_nulls_and_other_spines() {
        let document = parse(
            "**kern\t**kern\t**kern\t**dynam\t**text\t**mxhm\t**opaque\n\
             4c\t4r\t.\tp\tword\tC:maj\tanything\n\
             *-\t*-\t*-\t*-\t*-\t*-\t*-\n",
        );
        let noteheads: Vec<_> = document.located_noteheads().collect();

        assert_eq!(noteheads.len(), 1);
        assert_eq!(noteheads[0].location.spine_index, SpineIndex(0));
        assert_eq!(noteheads[0].note.pitch().step, DiatonicStep::C);
    }

    #[test]
    fn located_noteheads_preserves_source_line_numbers_after_skipped_lines() {
        let document = parse(
            "!!!COM: Example\n\
             !! global comment\n\
             **kern\n\
             4c\n\
             *-\n",
        );
        let noteheads: Vec<_> = document.located_noteheads().collect();

        assert_eq!(noteheads.len(), 1);
        assert_eq!(noteheads[0].location.line_number, 4);
        assert_eq!(noteheads[0].location.line_index, 3);
    }

    #[test]
    fn located_noteheads_include_grace_notes() {
        let document = parse(
            "!!!COM: Example\n\
             **kern\t**kern\n\
             8qcc\t.\n\
             4c\t8qee 8qgg\n\
             *-\t*-\n",
        );
        let noteheads: Vec<_> = document
            .located_noteheads()
            .map(|notehead| {
                (
                    notehead.location.line_number,
                    notehead.location.spine_index,
                    notehead.location.chord_note_index,
                    notehead.note.pitch().step,
                    matches!(notehead.note, KernPitchedNote::Grace(_)),
                )
            })
            .collect();

        assert_eq!(
            noteheads,
            vec![
                (3, SpineIndex(0), Some(0), DiatonicStep::C, true),
                (4, SpineIndex(0), Some(0), DiatonicStep::C, false),
                (4, SpineIndex(1), Some(0), DiatonicStep::E, true),
                (4, SpineIndex(1), Some(1), DiatonicStep::G, true),
            ]
        );
    }

    #[test]
    fn located_noteheads_expose_turn_ornaments() {
        let document = parse(
            "**kern\n\
             4cS\n\
             4d$\n\
             4esSss\n\
             4fSss\n\
             4gSSs\n\
             4a\n\
             *-\n",
        );

        let ornaments: Vec<_> = document
            .located_noteheads()
            .map(|notehead| {
                notehead
                    .token_ornaments()
                    .iter()
                    .map(|ornament| ornament.mark.clone())
                    .collect::<Vec<_>>()
            })
            .collect();

        assert_eq!(
            ornaments,
            vec![
                vec![turn(TurnForm::Regular, TurnTiming::Delayed, None)],
                vec![turn(TurnForm::Inverted, TurnTiming::Delayed, None)],
                vec![turn(
                    TurnForm::Regular,
                    TurnTiming::OnNote,
                    Some((AuxInterval::Semitone, AuxInterval::Semitone)),
                )],
                vec![turn(
                    TurnForm::Regular,
                    TurnTiming::Delayed,
                    Some((AuxInterval::Semitone, AuxInterval::Semitone)),
                )],
                vec![turn(
                    TurnForm::Regular,
                    TurnTiming::Delayed,
                    Some((AuxInterval::WholeTone, AuxInterval::Semitone)),
                )],
                Vec::new(),
            ]
        );
    }

    #[test]
    fn located_noteheads_exposes_chord_token_ornaments_on_every_notehead() {
        let document = parse("**kern\n4c 4eSss 4g\n*-\n");

        let ornaments: Vec<_> = document
            .located_noteheads()
            .map(|notehead| {
                notehead
                    .token_ornaments()
                    .iter()
                    .map(|ornament| ornament.mark.clone())
                    .collect::<Vec<_>>()
            })
            .collect();

        assert_eq!(ornaments.len(), 3);
        assert!(ornaments.iter().all(|ornaments| ornaments
            == &vec![turn(
                TurnForm::Regular,
                TurnTiming::Delayed,
                Some((AuxInterval::Semitone, AuxInterval::Semitone)),
            )]));
    }

    #[test]
    fn located_noteheads_reports_empty_ornaments_for_grace_tokens() {
        let document = parse("**kern\n8qccSss\n*-\n");

        let ornaments: Vec<_> = document
            .located_noteheads()
            .map(|notehead| notehead.token_ornaments().to_vec())
            .collect();

        assert_eq!(ornaments, vec![Vec::new()]);
    }
}

#[cfg(test)]
impl KernDocument {
    pub(crate) fn to_compact_omr_string(&self) -> String {
        crate::serialize::omr::to_compact_string(self)
    }
}

#[cfg(test)]
impl KernDocument {
    pub(crate) fn to_compact_omr_no_note_modifiers_string(&self) -> String {
        crate::serialize::omr::to_compact_no_note_modifiers_string(self)
    }
}
