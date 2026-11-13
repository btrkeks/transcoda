//! Render-facing serialization for typed `**kern` documents.

use super::barline::{serialize_barline_parts, BarlineSpelling};

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

mod cross_staff;

use crate::{
    analysis::explicit_accidentals::{explicit_accidental, ExplicitAccidental},
    document::Framing,
    normalization::{
        initial_header::{defines_staff, CanonicalizeInitialHeader},
        omr::remove_unread_staff_declarations,
        NormalizationPass,
    },
    score::{LayerId, Score, StateErrors, Topology, TrackId},
    serialize::bar_numbers::{BarNumberPlan, BarNumberStyle},
    serialize::omr::{
        ArpeggioStyle, ArticulationStyle, BarlineEmissionPolicy, BeamStyle, BowingStyle,
        ChordDurationStyle, ChordNoteOrder, CrossStaffStyle, InvisibleNoteStyle,
        InvisibleRestModifierStyle, InvisibleRestStyle, MarkPlacementStyle, NoteModifierStyle,
        NullDataTokenStyle, OmrSerializeOptions, OrnamentStyle, RestPlacementStyle,
        SlurPlacementStyle, SpineTopologyStyle, StemDirectionStyle, SystemBreakStyle,
        TerminalLineStyle, TiePlacementStyle, TrillWavyLineStyle, VisibilityMarkerStyle,
    },
    serialize::tandem_policy::DialectError,
    serialize::tandem_policy::TandemPolicy,
    types::{
        Accidental, DiatonicStep, ForcedAccidental, NoteModifiers, Ornament, OrnamentConvention,
        Slur,
    },
    Clef, ClefSign, DynamContent, DynamicHairpinDirection, DynamicLevel, DynamicLevelValue,
    DynamicProvenance, Fifths, FingContent, KernDocument, KernLine, KeySignature, LocalComment,
    MeterOutput, MeterSymbol, SpineContent, SpineType, SystemBreak, TandemInterpretation,
    TandemInterpretationLine, TandemInterpretationLineKind, TextPlacement,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderSerializeOptions {
    pub bar_number_style: BarNumberStyle,
    pub accidental_suppression_style: AccidentalSuppressionStyle,
    pub forced_accidental_style: ForcedAccidentalStyle,
    pub ornament_style: OrnamentStyle,
    pub trill_wavy_line_style: TrillWavyLineStyle,
    pub visibility_marker_style: VisibilityMarkerStyle,
    pub include_dynamics: bool,
    pub include_fingering: bool,
    /// Which tandem interpretations the output writes.
    pub tandem_policy: TandemPolicy,
    pub include_metadata: bool,
    /// Emit the full titling record set (OPR/OTA/OPS/OMV) in addition to the
    /// legacy COM/OTL/OMD records. Off by default: TranscodaV1 output is
    /// frozen and must stay byte-identical.
    pub extended_metadata_records: bool,
    pub include_local_comments: bool,
    /// Emit local comments the typed AST could not model, such as `!LO:DY`,
    /// verbatim instead of collapsing them to `!`.
    ///
    /// Off by default: the Transcoda V1 target is frozen and must stay
    /// byte-identical, and the notation engines do not need them.
    pub preserve_unsupported_local_comments: bool,
    /// How the initial header declares meter.
    pub meter_output: MeterOutput,
    pub include_system_breaks: bool,
    pub include_cross_staff_rdf_footer: bool,
    pub slur_placement_style: SlurPlacementStyle,
    pub tie_placement_style: TiePlacementStyle,
    pub mark_placement_style: MarkPlacementStyle,
    pub rest_placement_style: RestPlacementStyle,
    pub cross_staff_style: CrossStaffStyle,
}

impl Default for RenderSerializeOptions {
    fn default() -> Self {
        Self {
            bar_number_style: BarNumberStyle::default(),
            accidental_suppression_style: AccidentalSuppressionStyle::default(),
            forced_accidental_style: ForcedAccidentalStyle::default(),
            ornament_style: OrnamentStyle::default(),
            trill_wavy_line_style: TrillWavyLineStyle::Preserve,
            visibility_marker_style: VisibilityMarkerStyle::default(),
            include_dynamics: true,
            include_fingering: true,
            tandem_policy: TandemPolicy::HumdrumKern,
            include_metadata: true,
            extended_metadata_records: false,
            include_local_comments: true,
            preserve_unsupported_local_comments: false,
            meter_output: MeterOutput::ExplicitTimeSignature,
            include_system_breaks: true,
            include_cross_staff_rdf_footer: true,
            slur_placement_style: SlurPlacementStyle::Preserve,
            tie_placement_style: TiePlacementStyle::Preserve,
            mark_placement_style: MarkPlacementStyle::Preserve,
            rest_placement_style: RestPlacementStyle::Preserve,
            cross_staff_style: CrossStaffStyle::Preserve,
        }
    }
}

/// Whether render serialization honors source-authored accidental suppression.
///
/// `IgnoreSource` omits the single `y` attached to an accidental while preserving
/// independent visibility syntax such as the `yy` whole-note marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccidentalSuppressionStyle {
    #[default]
    HonorSource,
    IgnoreSource,
}

/// Which accidentals render serialization forces to display with `X`.
///
/// `Explicit` forces every written accidental that is not suppressed with `y`,
/// and a natural on every plain step the active key signature alters. Tie
/// continuations and ends are left to the note that starts the tie.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ForcedAccidentalStyle {
    #[default]
    Source,
    Explicit,
}

/// A pitched-note address in either serialized render text or canonical source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderNoteAddress {
    /// 1-based Humdrum line number.
    pub line_number: usize,
    /// Zero-based spine/field index on that line.
    pub spine_index: usize,
    /// Zero-based chord subtoken index, or `None` for a single-note token.
    pub chord_note_index: Option<usize>,
}

/// Maps a serialized note address back to the canonical source note that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderNoteProvenance {
    pub rendered: RenderNoteAddress,
    pub source: RenderNoteAddress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct RenderAddress {
    pub line: usize,
    pub field: usize,
    pub subtoken: Option<usize>,
}

impl RenderAddress {
    pub const fn field(line: usize, field: usize) -> Self {
        Self {
            line,
            field,
            subtoken: None,
        }
    }

    pub const fn subtoken(line: usize, field: usize, subtoken: usize) -> Self {
        Self {
            line,
            field,
            subtoken: Some(subtoken),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RenderProvenanceRelation {
    Preserved,
    Normalized,
    Expanded,
    Synthesized,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RenderTokenProvenance {
    pub rendered: RenderAddress,
    pub source: Option<RenderAddress>,
    pub relation: RenderProvenanceRelation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RenderSymbolKind {
    Note,
    Rest,
    MultiMeasureRest,
    Clef,
    Articulation,
    Fermata,
    Trill,
    Mordent,
    Turn,
    Arpeggio,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RenderSymbolProvenance {
    pub kind: RenderSymbolKind,
    pub rendered: RenderAddress,
    pub source: Option<RenderAddress>,
    pub relation: RenderProvenanceRelation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RenderControlKind {
    Ottava,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RenderControlSpanRendered {
    pub open: RenderAddress,
    pub close: RenderAddress,
    pub start_anchor: RenderAddress,
    pub end_anchor: RenderAddress,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RenderControlSpanSource {
    pub open: Option<RenderAddress>,
    pub close: Option<RenderAddress>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RenderControlSpan {
    pub kind: RenderControlKind,
    pub value: String,
    pub rendered: RenderControlSpanRendered,
    pub source: RenderControlSpanSource,
    pub relation: RenderProvenanceRelation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderOutput {
    pub(super) lines: Vec<String>,
    pub(super) sources: Vec<Option<usize>>,
    pub(super) note_provenance: Vec<RenderNoteProvenance>,
    pub(super) token_provenance: Vec<RenderTokenProvenance>,
    pub(super) symbol_provenance: Vec<RenderSymbolProvenance>,
    pub(super) control_spans: Vec<RenderControlSpan>,
}

impl RenderOutput {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            sources: Vec::new(),
            note_provenance: Vec::new(),
            token_provenance: Vec::new(),
            symbol_provenance: Vec::new(),
            control_spans: Vec::new(),
        }
    }

    fn push(&mut self, line: String, source: Option<usize>) {
        self.lines.push(line);
        self.sources.push(source);
    }

    /// Verovio input. Cross-staff display tokens use a render-only spelling.
    /// Use `canonical_text()` when parsing or analyzing the prepared data.
    pub fn text(&self) -> String {
        cross_staff::lower_notes(
            &self.canonical_text(),
            self.note_provenance
                .iter()
                .map(|p| crate::analysis::accidental_display::NoteAddress {
                    line: p.rendered.line_number,
                    field: p.rendered.spine_index + 1,
                    subtoken: p.rendered.chord_note_index.unwrap_or(0) + 1,
                }),
        )
    }

    /// Prepared, source-grammar text for analysis. Addresses match `text()`.
    pub fn canonical_text(&self) -> String {
        format!("{}\n", self.lines.join("\n"))
    }

    pub fn sources(&self) -> &[Option<usize>] {
        &self.sources
    }

    pub fn note_provenance(&self) -> &[RenderNoteProvenance] {
        &self.note_provenance
    }

    pub fn token_provenance(&self) -> &[RenderTokenProvenance] {
        &self.token_provenance
    }

    pub fn symbol_provenance(&self) -> &[RenderSymbolProvenance] {
        &self.symbol_provenance
    }

    pub fn control_spans(&self) -> &[RenderControlSpan] {
        &self.control_spans
    }

    /// Canonical prepared lines and their source mapping, before render lowering.
    pub fn into_parts(self) -> (Vec<String>, Vec<Option<usize>>) {
        (self.lines, self.sources)
    }

    fn push_tokens(
        &mut self,
        line: String,
        legacy_source: Option<usize>,
        source_fields: Vec<Option<RenderAddress>>,
        expanded_source_lines: &HashMap<usize, usize>,
    ) {
        let rendered_line = self.lines.len() + 1;
        debug_assert_eq!(line.split('\t').count(), source_fields.len());
        self.token_provenance
            .extend(
                source_fields
                    .into_iter()
                    .enumerate()
                    .map(|(field_index, source)| {
                        let rendered = RenderAddress::field(rendered_line, field_index + 1);
                        RenderTokenProvenance {
                            rendered,
                            source,
                            relation: provenance_relation(rendered, source, expanded_source_lines),
                        }
                    }),
            );
        self.push(line, legacy_source);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OriginSymbol {
    kind: RenderSymbolKind,
    subtoken: Option<usize>,
}

#[derive(Debug, Default)]
struct RenderOriginIndex {
    symbols: HashMap<(usize, usize), Vec<OriginSymbol>>,
    initial_tandem_sources: Vec<(TandemInterpretationLineKind, usize, RenderAddress)>,
}

impl RenderOriginIndex {
    fn new(document: &KernDocument) -> Self {
        let mut index = Self::default();
        let initial_header = document.leading_tandem_header_range();
        for (line_index, line) in document.lines().iter().enumerate() {
            let Some(source_line) = document.source(line_index) else {
                continue;
            };
            match line {
                KernLine::Data(contents) => {
                    for (field_index, content) in contents.iter().enumerate() {
                        let descriptors = symbol_descriptors(content);
                        if !descriptors.is_empty() {
                            index
                                .symbols
                                .insert((source_line, field_index + 1), descriptors);
                        }
                    }
                }
                KernLine::TandemInterpretationLine(tandems) => {
                    for (field_index, tandem) in tandems.tokens().iter().enumerate() {
                        if initial_header.contains(&line_index) && defines_staff(tandem) {
                            index.initial_tandem_sources.push((
                                tandems.kind(),
                                field_index + 1,
                                RenderAddress::field(source_line, field_index + 1),
                            ));
                        }
                        if matches!(tandem, TandemInterpretation::Clef(_)) {
                            index.symbols.insert(
                                (source_line, field_index + 1),
                                vec![OriginSymbol {
                                    kind: RenderSymbolKind::Clef,
                                    subtoken: None,
                                }],
                            );
                        }
                    }
                }
                KernLine::SpineTypeLine(_)
                | KernLine::SpineManipulatorLine(_)
                | KernLine::LocalCommentLine(_)
                | KernLine::Barline(_) => {}
            }
        }
        index
    }

    fn symbols_for(
        &self,
        source: Option<RenderAddress>,
        fallback: Vec<OriginSymbol>,
    ) -> Vec<OriginSymbol> {
        source
            .and_then(|address| self.symbols.get(&(address.line, address.field)).cloned())
            .unwrap_or(fallback)
    }

    fn initial_tandem_source(
        &self,
        kind: TandemInterpretationLineKind,
        field: usize,
    ) -> Option<RenderAddress> {
        self.initial_tandem_sources.iter().rev().find_map(
            |(candidate_kind, candidate_field, address)| {
                (*candidate_kind == kind && *candidate_field == field).then_some(*address)
            },
        )
    }
}

fn provenance_relation(
    rendered: RenderAddress,
    source: Option<RenderAddress>,
    expanded_source_lines: &HashMap<usize, usize>,
) -> RenderProvenanceRelation {
    let Some(source) = source else {
        return RenderProvenanceRelation::Synthesized;
    };
    if expanded_source_lines
        .get(&source.line)
        .copied()
        .unwrap_or(0)
        > 1
    {
        RenderProvenanceRelation::Expanded
    } else if rendered == source {
        RenderProvenanceRelation::Preserved
    } else {
        RenderProvenanceRelation::Normalized
    }
}

fn source_field_address(
    document: &KernDocument,
    line_index: usize,
    field_index: usize,
    generated: bool,
) -> Option<RenderAddress> {
    if generated {
        None
    } else {
        document
            .source(line_index)
            .map(|line| RenderAddress::field(line, field_index + 1))
    }
}

fn append_symbols(
    output: &mut RenderOutput,
    origins: &RenderOriginIndex,
    rendered_line: usize,
    rendered_field: usize,
    source_field: Option<RenderAddress>,
    fallback: Vec<OriginSymbol>,
    expanded_source_lines: &HashMap<usize, usize>,
) {
    for symbol in origins.symbols_for(source_field, fallback) {
        let rendered = match symbol.subtoken {
            Some(subtoken) => RenderAddress::subtoken(rendered_line, rendered_field, subtoken),
            None => RenderAddress::field(rendered_line, rendered_field),
        };
        let source = source_field.map(|address| match symbol.subtoken {
            Some(subtoken) => RenderAddress::subtoken(address.line, address.field, subtoken),
            None => address,
        });
        output.symbol_provenance.push(RenderSymbolProvenance {
            kind: symbol.kind,
            rendered,
            source,
            relation: provenance_relation(rendered, source, expanded_source_lines),
        });
    }
}

fn symbol_descriptors(content: &SpineContent) -> Vec<OriginSymbol> {
    let SpineContent::Kern(content) = content else {
        return Vec::new();
    };
    let mut symbols = Vec::new();
    match content {
        crate::KernContent::Null => {}
        crate::KernContent::Rest(rest) => {
            symbols.push(OriginSymbol {
                kind: RenderSymbolKind::Rest,
                subtoken: None,
            });
            if rest.fermata.is_some() {
                symbols.push(OriginSymbol {
                    kind: RenderSymbolKind::Fermata,
                    subtoken: None,
                });
            }
        }
        crate::KernContent::InvisibleRestSlice(rest) => {
            symbols.push(OriginSymbol {
                kind: RenderSymbolKind::Rest,
                subtoken: None,
            });
            if rest.fermata.is_some() {
                symbols.push(OriginSymbol {
                    kind: RenderSymbolKind::Fermata,
                    subtoken: None,
                });
            }
        }
        crate::KernContent::MultiMeasureRest(_) => symbols.push(OriginSymbol {
            kind: RenderSymbolKind::MultiMeasureRest,
            subtoken: None,
        }),
        crate::KernContent::Note(_) => {
            let note_count = content.noteheads().count();
            symbols.extend(content.noteheads().map(|note| {
                OriginSymbol {
                    kind: RenderSymbolKind::Note,
                    subtoken: (note_count > 1)
                        .then_some(note.chord_note_index)
                        .flatten()
                        .map(|index| index + 1),
                }
            }));
        }
        crate::KernContent::Grace(grace) => {
            let note_count = grace.notes.len();
            symbols.extend(
                grace
                    .notes
                    .iter()
                    .enumerate()
                    .map(|(index, _)| OriginSymbol {
                        kind: RenderSymbolKind::Note,
                        subtoken: (note_count > 1).then_some(index + 1),
                    }),
            );
            append_chord_modifier_descriptors(&mut symbols, &grace.modifiers);
        }
        crate::KernContent::Chord(chord) => {
            let note_count = chord.notes.len();
            symbols.extend(
                chord
                    .notes
                    .iter()
                    .enumerate()
                    .map(|(index, _)| OriginSymbol {
                        kind: RenderSymbolKind::Note,
                        subtoken: (note_count > 1).then_some(index + 1),
                    }),
            );
            append_chord_modifier_descriptors(&mut symbols, &chord.modifiers);
        }
    }
    symbols
}

fn append_chord_modifier_descriptors(
    symbols: &mut Vec<OriginSymbol>,
    modifiers: &crate::types::ChordModifiers,
) {
    symbols.extend(
        std::iter::repeat_n(
            RenderSymbolKind::Articulation,
            modifiers.articulations.len(),
        )
        .map(|kind| OriginSymbol {
            kind,
            subtoken: None,
        }),
    );
    if modifiers.fermata.is_some() {
        symbols.push(OriginSymbol {
            kind: RenderSymbolKind::Fermata,
            subtoken: None,
        });
    }
    for ornament in &modifiers.ornaments {
        let kind = match ornament.mark {
            Ornament::Trill { .. } => RenderSymbolKind::Trill,
            Ornament::Mordent { .. } => RenderSymbolKind::Mordent,
            Ornament::Turn(_) | Ornament::ConcludingTurn => RenderSymbolKind::Turn,
            Ornament::Generic => continue,
        };
        symbols.push(OriginSymbol {
            kind,
            subtoken: None,
        });
    }
    if modifiers.arpeggio.is_some() {
        symbols.push(OriginSymbol {
            kind: RenderSymbolKind::Arpeggio,
            subtoken: None,
        });
    }
}

/// Serialize a document to renderable Humdrum for notation engines such as Verovio.
///
/// Unlike the OMR training serializers, this emits the required exclusive
/// interpretation header and terminal spine line.
/// Fails when the document's score does not resolve or sets state the
/// options' [`TandemPolicy`] rejects.
pub fn to_string(document: &KernDocument) -> Result<String, DialectError> {
    to_string_with_options(document, RenderSerializeOptions::default())
}

pub fn to_string_with_options(
    document: &KernDocument,
    serialize_options: RenderSerializeOptions,
) -> Result<String, DialectError> {
    Ok(to_render_output(document, serialize_options)?.text())
}

pub(crate) fn to_string_with_options_and_arpeggio_style(
    document: &KernDocument,
    serialize_options: RenderSerializeOptions,
    arpeggio_style: ArpeggioStyle,
) -> Result<String, DialectError> {
    Ok(
        to_render_output_with_arpeggio_style(document, serialize_options, arpeggio_style)?
            .canonical_text(),
    )
}

/// Decode Verovio-bound accidental placement for a consumer of prepared render text.
/// Authored data must still be parsed with the ordinary source grammar.
pub fn canonical_data_text(text: &str) -> Result<String, crate::document::DocumentError> {
    cross_staff::canonical_data(text)
}

/// Lower already-prepared canonical data without repeating render serialization.
/// This supports display transforms performed on `RenderOutput::canonical_text()`.
/// The input must satisfy the ordinary source grammar.
pub fn lower_prepared_text(text: &str) -> Result<String, crate::document::DocumentError> {
    cross_staff::lower_prepared(text)
}

pub(crate) fn lower_cross_staff_head(token: &str) -> String {
    cross_staff::lower_head(token)
}

pub fn to_render_output(
    document: &KernDocument,
    serialize_options: RenderSerializeOptions,
) -> Result<RenderOutput, DialectError> {
    to_render_output_with_arpeggio_style(document, serialize_options, ArpeggioStyle::Preserve)
}

fn to_render_output_with_arpeggio_style(
    document: &KernDocument,
    serialize_options: RenderSerializeOptions,
    arpeggio_style: ArpeggioStyle,
) -> Result<RenderOutput, DialectError> {
    let origins = RenderOriginIndex::new(document);
    let first_bar_number = crate::analysis::written_measures::written_measure_numbers(document)
        .ok()
        .and_then(|measures| measures.first().map(|measure| measure.measure.number()))
        .unwrap_or(1);
    let document = prepare_render_document(document, serialize_options)?;
    let score = Score::resolve(&document, Framing::Excerpt)?;
    serialize_options.tandem_policy.check(&score)?;

    Ok(render_spine_document(
        &score,
        serialize_options,
        arpeggio_style,
        &origins,
        first_bar_number,
    ))
}

pub(crate) fn prepare_render_document(
    document: &KernDocument,
    serialize_options: RenderSerializeOptions,
) -> Result<KernDocument, StateErrors> {
    let mut document =
        crate::normalization::multi_measure_rests::expanded_multi_measure_rests(document);
    move_breaks_after_expanded_multi_measure_rests_for_verovio(&mut document);

    if !matches!(document.lines().first(), Some(KernLine::SpineTypeLine(_))) {
        if let Some((index, _)) = crate::initial_state::initial_spine_type_declaration(&document) {
            let declaration = document.remove_line(index);
            document.insert_line(0, declaration.0, declaration.1);
        } else {
            document.insert_line(
                0,
                KernLine::SpineTypeLine(vec![
                    SpineType::Kern;
                    document.layout.initial_spine_count as usize
                ]),
                None,
            );
        }
    }
    CanonicalizeInitialHeader::new(Default::default(), serialize_options.meter_output)
        .normalize(&mut document);
    spell_hidden_time_signatures_with_empty_symbols(&mut document);
    synthesize_default_kern_header(&mut document);
    remove_unread_staff_declarations(&mut document)
        .map_err(|error| StateErrors(vec![error.into()]))?;
    crate::normalization::invisible_rest_slices::rehydrate_durationless_invisible_rest_slices_if_present(
        &mut document,
    )
    .expect("render invisible rest slice rehydration failed");
    let emits_invisible_accidentals = matches!(
        serialize_options.accidental_suppression_style,
        AccidentalSuppressionStyle::HonorSource
    ) && matches!(
        serialize_options.visibility_marker_style,
        VisibilityMarkerStyle::Preserve
    );
    if !emits_invisible_accidentals {
        strip_invisible_accidental_markers(&mut document);
    }
    if matches!(
        serialize_options.forced_accidental_style,
        ForcedAccidentalStyle::Explicit
    ) {
        let draws_invisible_notes = matches!(
            serialize_options.visibility_marker_style,
            VisibilityMarkerStyle::Preserve
        );
        force_explicit_accidentals(&mut document, draws_invisible_notes)?;
    }

    Ok(document)
}

fn force_explicit_accidentals(
    document: &mut KernDocument,
    draws_invisible_notes: bool,
) -> Result<(), StateErrors> {
    let key_signatures: Vec<Vec<Option<KeySignature>>> = {
        let score = Score::resolve(document, Framing::Excerpt)?;
        document
            .lines()
            .iter()
            .enumerate()
            .map(|(line_index, line)| match line {
                KernLine::Data(contents) => (0..contents.len())
                    .map(|column| score.state_at(line_index, column).track.key_signature)
                    .collect(),
                _ => Vec::new(),
            })
            .collect()
    };
    for (line_index, line) in document.lines_mut().iter_mut().enumerate() {
        let KernLine::Data(contents) = line else {
            continue;
        };
        for (spine_index, content) in contents.iter_mut().enumerate() {
            let SpineContent::Kern(content) = content else {
                continue;
            };
            let key_signature = key_signatures[line_index][spine_index].as_ref();
            match content {
                crate::KernContent::Note(note) => force_explicit_accidental(
                    note.pitch.step,
                    &mut note.accidental,
                    &mut note.modifiers,
                    draws_invisible_notes,
                    key_signature,
                ),
                crate::KernContent::Chord(chord) => {
                    for note in &mut chord.notes {
                        force_explicit_accidental(
                            note.pitch.step,
                            &mut note.accidental,
                            &mut note.modifiers,
                            draws_invisible_notes,
                            key_signature,
                        );
                    }
                }
                crate::KernContent::Grace(grace) => {
                    for note in &mut grace.notes {
                        force_explicit_accidental(
                            note.pitch.step,
                            &mut note.accidental,
                            &mut note.modifiers,
                            draws_invisible_notes,
                            key_signature,
                        );
                    }
                }
                crate::KernContent::Null
                | crate::KernContent::Rest(_)
                | crate::KernContent::InvisibleRestSlice(_)
                | crate::KernContent::MultiMeasureRest(_) => {}
            }
        }
    }
    Ok(())
}

fn force_explicit_accidental(
    step: DiatonicStep,
    accidental: &mut Option<Accidental>,
    modifiers: &mut NoteModifiers,
    draws_invisible_notes: bool,
    key_signature: Option<&KeySignature>,
) {
    let drawn_invisible = draws_invisible_notes && modifiers.invisible;
    let Some(explicit) = explicit_accidental(
        step,
        accidental.as_ref(),
        modifiers,
        drawn_invisible,
        key_signature,
    ) else {
        return;
    };
    if explicit == ExplicitAccidental::ForceNatural {
        *accidental = Some(Accidental::Natural);
    }
    modifiers.forced_accidental = Some(ForcedAccidental);
}

fn strip_invisible_accidental_markers(document: &mut KernDocument) {
    for line in document.lines_mut() {
        let KernLine::Data(contents) = line else {
            continue;
        };
        for content in contents {
            let SpineContent::Kern(content) = content else {
                continue;
            };
            match content {
                crate::KernContent::Note(note) => {
                    note.modifiers.invisible_accidental = false;
                }
                crate::KernContent::Chord(chord) => {
                    for note in &mut chord.notes {
                        note.modifiers.invisible_accidental = false;
                    }
                }
                crate::KernContent::Grace(grace) => {
                    for note in &mut grace.notes {
                        note.modifiers.invisible_accidental = false;
                    }
                }
                crate::KernContent::Null
                | crate::KernContent::Rest(_)
                | crate::KernContent::InvisibleRestSlice(_)
                | crate::KernContent::MultiMeasureRest(_) => {}
            }
        }
    }
}

fn move_breaks_after_expanded_multi_measure_rests_for_verovio(document: &mut KernDocument) {
    let original = document.take_lines();
    let mut lines = Vec::with_capacity(original.len());
    let mut index = 0;

    while index < original.len() {
        let (mut line, source) = original[index].clone();
        // Verovio condenses expanded full-rest runs into <multiRest>. A break
        // before the final boundary is dropped during that condensation, but a
        // break before the first internal boundary survives as a post-multiRest
        // system break.
        let movable_break = match &mut line {
            KernLine::Barline(barline) => barline.system_break_before.take(),
            _ => None,
        };
        if let Some(system_break) = movable_break {
            if let Some(insert_index) = expanded_multi_measure_rest_break_index(&lines) {
                if let KernLine::Barline(barline) = &mut lines[insert_index].0 {
                    if barline.system_break_before.is_none() {
                        barline.system_break_before = Some(system_break);
                        lines.push((line, source));
                        index += 1;
                        continue;
                    }
                }
            }
            if let KernLine::Barline(barline) = &mut line {
                barline.system_break_before = Some(system_break);
            }
        }

        lines.push((line, source));
        index += 1;
    }

    document.replace_lines(lines);
}

fn expanded_multi_measure_rest_break_index(
    lines: &[(KernLine, crate::ast::LineOrigin)],
) -> Option<usize> {
    let (last_line, origin) = lines.last()?;
    if !matches!(last_line, KernLine::Data(_)) {
        return None;
    }
    let source = Some(origin.owner()?);
    let mut start = lines.len() - 1;
    while start > 0
        && lines[start - 1].1.owner() == source
        && matches!(lines[start - 1].0, KernLine::Barline(_) | KernLine::Data(_))
    {
        start -= 1;
    }

    let mut saw_first_data = false;
    for index in start..lines.len() {
        match &lines[index].0 {
            KernLine::Data(_) => {
                saw_first_data = true;
            }
            KernLine::Barline(_) if saw_first_data => {
                let has_later_data = lines[index + 1..].iter().any(|(line, line_source)| {
                    line_source.owner() == source && matches!(line, KernLine::Data(_))
                });
                if has_later_data {
                    return Some(index);
                }
            }
            KernLine::Barline(_) => {}
            _ => return None,
        }
    }

    None
}

/// Verovio honors `yy` on `*M` only for the first meter of a staff and draws a
/// later `*M4/4yy`, so every hidden time signature becomes a visible `*M`
/// followed by `*met()`.
fn spell_hidden_time_signatures_with_empty_symbols(document: &mut KernDocument) {
    let mut index = 0;
    while index < document.lines().len() {
        let KernLine::TandemInterpretationLine(tandems) = &mut document.lines_mut()[index] else {
            index += 1;
            continue;
        };
        let mut symbols = vec![TandemInterpretation::Null; tandems.len()];
        for (token, symbol) in tandems.tokens_mut().iter_mut().zip(&mut symbols) {
            if let TandemInterpretation::TimeSignature(time_signature) = token {
                if time_signature.invisible {
                    time_signature.invisible = false;
                    *symbol = TandemInterpretation::MeterSymbol(MeterSymbol::Empty);
                }
            }
        }
        index += 1;
        if symbols
            .iter()
            .any(|symbol| !matches!(symbol, TandemInterpretation::Null))
        {
            let source = document.source(index - 1);
            document.insert_line(
                index,
                KernLine::TandemInterpretationLine(
                    TandemInterpretationLine::from_homogeneous_tokens(symbols),
                ),
                source,
            );
            index += 1;
        }
    }
}

fn render_spine_document(
    score: &Score<'_>,
    serialize_options: RenderSerializeOptions,
    arpeggio_style: ArpeggioStyle,
    origins: &RenderOriginIndex,
    first_bar_number: u32,
) -> RenderOutput {
    let document = score.document();
    let options = OmrSerializeOptions {
        chord_duration_style: ChordDurationStyle::Repeated,
        chord_note_order: ChordNoteOrder::Source,
        terminal_line_style: TerminalLineStyle::Emit,
        note_modifier_style: NoteModifierStyle::Preserve,
        slur_placement_style: serialize_options.slur_placement_style,
        tie_placement_style: serialize_options.tie_placement_style,
        mark_placement_style: serialize_options.mark_placement_style,
        rest_placement_style: serialize_options.rest_placement_style,
        beam_style: BeamStyle::Preserve,
        articulation_style: ArticulationStyle::Preserve,
        ornament_style: serialize_options.ornament_style,
        trill_wavy_line_style: serialize_options.trill_wavy_line_style,
        bowing_style: BowingStyle::Preserve,
        stem_direction_style: StemDirectionStyle::Preserve,
        cross_staff_style: serialize_options.cross_staff_style,
        visibility_marker_style: serialize_options.visibility_marker_style,
        invisible_note_style: InvisibleNoteStyle::Preserve,
        invisible_rest_style: InvisibleRestStyle::DurationBearing,
        invisible_rest_modifier_style: InvisibleRestModifierStyle::Preserve,
        spine_topology_style: SpineTopologyStyle::AsWritten,
        tandem_policy: serialize_options.tandem_policy,
        system_break_style: SystemBreakStyle::Omit,
        compact_multi_measure_rests: false,
        compact_consecutive_invisible_rests: false,
        preserve_leading_barlines: true,
        barline_emission_policy: BarlineEmissionPolicy::PerActiveSpine,
        null_data_token_style: NullDataTokenStyle::Dot,
        bar_number_style: BarNumberStyle::Omit,
    };
    let bar_number_plan = BarNumberPlan::with_first_number(
        document,
        serialize_options.bar_number_style,
        first_bar_number,
    );
    let initial_header = document.leading_tandem_header_range();
    let topology = score.topology();
    let mut output = RenderOutput::new();
    let mut expanded_source_lines = HashMap::new();
    for line_index in 0..document.lines().len() {
        if let Some(source) = document.source(line_index) {
            *expanded_source_lines.entry(source).or_insert(0) += 1;
        }
    }
    if serialize_options.include_metadata {
        push_metadata_records(
            document,
            &mut output,
            serialize_options.extended_metadata_records,
        );
    }

    for (line_index, line) in document.lines().iter().enumerate() {
        let spines = spine_types(topology, topology.layers_at(line_index));
        match line {
            KernLine::SpineTypeLine(spine_types) => {
                let fields = spine_types
                    .iter()
                    .enumerate()
                    .filter_map(|(field_index, spine_type)| {
                        serialize_spine_type(spine_type, serialize_options).map(|token| {
                            (
                                token,
                                source_field_address(document, line_index, field_index, false),
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                if !fields.is_empty() {
                    output.push_tokens(
                        fields
                            .iter()
                            .map(|(token, _)| *token)
                            .collect::<Vec<_>>()
                            .join("\t"),
                        document.source(line_index),
                        fields.into_iter().map(|(_, source)| source).collect(),
                        &expanded_source_lines,
                    );
                }
            }
            KernLine::TandemInterpretationLine(tandems) => {
                let companions = match serialize_options.visibility_marker_style {
                    VisibilityMarkerStyle::Preserve => {
                        clef_companions(document, topology, line_index, tandems.tokens())
                    }
                    VisibilityMarkerStyle::Omit => vec![None; tandems.len()],
                };
                let fields = tandems
                    .tokens()
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| is_render_spine(&spines, *index, serialize_options))
                    .map(|(field_index, tandem)| {
                        if let Some(companion) = &companions[field_index] {
                            let source = match companion.drawn_from {
                                Some(shown) => {
                                    source_field_address(document, line_index, shown, false)
                                }
                                None => (*tandem != TandemInterpretation::Null)
                                    .then(|| {
                                        source_field_address(
                                            document,
                                            line_index,
                                            field_index,
                                            false,
                                        )
                                    })
                                    .flatten(),
                            };
                            return (companion.clef.as_kern(), true, source);
                        }
                        let generated = tandems.is_generated_token(field_index);
                        let source = if generated {
                            None
                        } else if initial_header.contains(&line_index) {
                            origins
                                .initial_tandem_source(tandems.kind(), field_index + 1)
                                .or_else(|| {
                                    source_field_address(document, line_index, field_index, false)
                                })
                        } else {
                            source_field_address(document, line_index, field_index, false)
                        };
                        (
                            crate::serialize::omr::serialize_tandem(tandem, &options)
                                .unwrap_or_else(|| "*".to_owned()),
                            matches!(tandem, TandemInterpretation::Clef(_)),
                            source,
                        )
                    })
                    .collect::<Vec<_>>();
                if let Some(serialized) = non_null_line(
                    fields.iter().map(|(token, _, _)| token.clone()).collect(),
                    "*",
                ) {
                    let rendered_line = output.lines.len() + 1;
                    for (rendered_field_index, (_, is_clef, source)) in fields.iter().enumerate() {
                        if *is_clef {
                            append_symbols(
                                &mut output,
                                origins,
                                rendered_line,
                                rendered_field_index + 1,
                                *source,
                                vec![OriginSymbol {
                                    kind: RenderSymbolKind::Clef,
                                    subtoken: None,
                                }],
                                &expanded_source_lines,
                            );
                        }
                    }
                    output.push_tokens(
                        serialized,
                        document.source(line_index),
                        fields.into_iter().map(|(_, _, source)| source).collect(),
                        &expanded_source_lines,
                    );
                }
            }
            KernLine::SpineManipulatorLine(manipulators) => {
                let fields = manipulators
                    .tokens()
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| is_render_spine(&spines, *index, serialize_options))
                    .map(|(field_index, manipulator)| {
                        (
                            crate::serialize::omr::serialize_spine_manipulator(manipulator),
                            source_field_address(document, line_index, field_index, false),
                        )
                    })
                    .collect::<Vec<_>>();
                if let Some(serialized) =
                    non_null_line(fields.iter().map(|(token, _)| token.clone()).collect(), "*")
                {
                    output.push_tokens(
                        serialized,
                        document.source(line_index),
                        fields.into_iter().map(|(_, source)| source).collect(),
                        &expanded_source_lines,
                    );
                }
            }
            KernLine::LocalCommentLine(comments) => {
                if !serialize_options.include_local_comments {
                    continue;
                }
                let fields = comments
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| is_render_spine(&spines, *index, serialize_options))
                    .map(|(field_index, comment)| {
                        (
                            serialize_local_comment(
                                comment,
                                serialize_options.preserve_unsupported_local_comments,
                            ),
                            source_field_address(document, line_index, field_index, false),
                        )
                    })
                    .collect::<Vec<_>>();
                if let Some(serialized) =
                    non_null_line(fields.iter().map(|(token, _)| token.clone()).collect(), "!")
                {
                    output.push_tokens(
                        serialized,
                        document.source(line_index),
                        fields.into_iter().map(|(_, source)| source).collect(),
                        &expanded_source_lines,
                    );
                }
            }
            KernLine::Barline(barline) => {
                if serialize_options.include_system_breaks {
                    if let Some(system_break) = &barline.system_break_before {
                        output.push(
                            serialize_system_break(system_break),
                            system_break
                                .source_line
                                .or_else(|| document.source(line_index)),
                        );
                    }
                }
                let fields = spines
                    .iter()
                    .enumerate()
                    .filter(|(_, spine_type)| is_render_spine_type(spine_type, serialize_options))
                    .map(|(field_index, _)| {
                        (
                            serialize_render_barline_for_spine(
                                barline,
                                field_index,
                                bar_number_plan.barline_number(line_index),
                                serialize_options.mark_placement_style,
                            ),
                            source_field_address(document, line_index, field_index, false),
                        )
                    })
                    .collect::<Vec<_>>();
                if !fields.is_empty() {
                    output.push_tokens(
                        fields
                            .iter()
                            .map(|(token, _)| token.as_str())
                            .collect::<Vec<_>>()
                            .join("\t"),
                        document.source(line_index),
                        fields.into_iter().map(|(_, source)| source).collect(),
                        &expanded_source_lines,
                    );
                }
            }
            KernLine::Data(contents) => {
                if let Some(line) = crate::serialize::omr::serialize_synthetic_render_barline_line(
                    &spines,
                    bar_number_plan.insertion_before(line_index),
                    serialize_options.include_dynamics,
                    serialize_options.include_fingering,
                ) {
                    let field_count = line.split('\t').count();
                    output.push_tokens(
                        line,
                        document.source(line_index),
                        vec![None; field_count],
                        &expanded_source_lines,
                    );
                }
                let mut tokens = Vec::new();
                let mut note_provenance = Vec::new();
                let mut rendered_fields = Vec::new();
                let source_line_number = document.source_line_number(line_index);
                for (source_spine_index, content) in contents.iter().enumerate() {
                    if !is_render_spine(&spines, source_spine_index, serialize_options) {
                        continue;
                    }
                    let Some(token) = serialize_render_content(content, &options, arpeggio_style)
                    else {
                        continue;
                    };
                    let rendered_spine_index = tokens.len();
                    let source =
                        source_field_address(document, line_index, source_spine_index, false);
                    if let SpineContent::Kern(content) = content {
                        let notehead_count = content.noteheads().count();
                        note_provenance.extend(content.noteheads().map(|notehead| {
                            (
                                rendered_spine_index,
                                (notehead_count > 1)
                                    .then_some(notehead.chord_note_index)
                                    .flatten(),
                                source_spine_index,
                                source_line_number,
                            )
                        }));
                    }
                    rendered_fields.push((content, source));
                    tokens.push(token);
                }
                if !tokens.is_empty() {
                    let rendered_line_number = output.lines.len() + 1;
                    output
                        .note_provenance
                        .extend(note_provenance.into_iter().map(
                            |(
                                rendered_spine_index,
                                chord_note_index,
                                source_spine_index,
                                source_line_number,
                            )| RenderNoteProvenance {
                                rendered: RenderNoteAddress {
                                    line_number: rendered_line_number,
                                    spine_index: rendered_spine_index,
                                    chord_note_index,
                                },
                                source: RenderNoteAddress {
                                    line_number: source_line_number,
                                    spine_index: source_spine_index,
                                    chord_note_index,
                                },
                            },
                        ));
                    for (rendered_field_index, (content, source)) in
                        rendered_fields.iter().enumerate()
                    {
                        append_symbols(
                            &mut output,
                            origins,
                            rendered_line_number,
                            rendered_field_index + 1,
                            *source,
                            symbol_descriptors(content),
                            &expanded_source_lines,
                        );
                    }
                    output.push_tokens(
                        tokens.join("\t"),
                        document.source(line_index),
                        rendered_fields
                            .into_iter()
                            .map(|(_, source)| source)
                            .collect(),
                        &expanded_source_lines,
                    );
                }
            }
        }
    }

    let render_width = spine_types(topology, topology.layers_after_last())
        .iter()
        .filter(|spine_type| is_render_spine_type(spine_type, serialize_options))
        .count();
    if render_width > 0
        && !output
            .lines
            .last()
            .is_some_and(|line| line.split('\t').all(|token| token == "*-"))
    {
        output.push_tokens(
            vec!["*-"; render_width].join("\t"),
            None,
            vec![None; render_width],
            &expanded_source_lines,
        );
    }
    let includes_default_rdf_footer =
        serialize_options.include_metadata && serialize_options.include_cross_staff_rdf_footer;
    if includes_default_rdf_footer {
        push_cross_staff_rdf_footer(&mut output);
    } else if (matches!(
        serialize_options.slur_placement_style,
        SlurPlacementStyle::Preserve
    ) && document_has_slur(document, |slur| {
        matches!(
            slur,
            Slur::Open {
                placement: Some(_),
                ..
            }
        )
    })) || (matches!(
        serialize_options.tie_placement_style,
        TiePlacementStyle::Preserve
    ) && document_has_placed_tie(document))
        || (matches!(
            serialize_options.mark_placement_style,
            MarkPlacementStyle::Preserve
        ) && document_has_placed_mark(document))
    {
        output.push("!!!RDF**kern: > = above".to_owned(), None);
        output.push("!!!RDF**kern: < = below".to_owned(), None);
    }

    if !includes_default_rdf_footer {
        for (marker, record) in [("Z", DEFAULT_RDF_RECORDS[3]), ("j", DEFAULT_RDF_RECORDS[4])] {
            if document.located_noteheads().any(|head| {
                let modifiers = head.note.modifiers();
                modifiers
                    .user_signifiers
                    .iter()
                    .any(|sign| sign.marker == marker)
            }) {
                output.push(record.to_owned(), None);
            }
        }
        if document_has_slur(document, |slur| {
            matches!(
                slur,
                Slur::Open { linked: true, .. } | Slur::Close { linked: true, .. }
            )
        }) {
            output.push(DEFAULT_RDF_RECORDS[2].to_owned(), None);
        }
    }

    populate_control_spans(&mut output, score);

    output
}

fn document_has_slur(document: &KernDocument, predicate: impl Fn(&Slur) -> bool) -> bool {
    document.lines().iter().any(|line| {
        let KernLine::Data(contents) = line else {
            return false;
        };
        contents.iter().any(|content| {
            let SpineContent::Kern(content) = content else {
                return false;
            };
            crate::analysis::slur_spans::slurs_for_content(content)
                .iter()
                .any(&predicate)
        })
    })
}

fn document_has_placed_mark(document: &KernDocument) -> bool {
    document.lines().iter().any(|line| {
        if let KernLine::Barline(barline) = line {
            return (0..barline.spine_count().unwrap_or(1)).any(|index| {
                matches!(
                    barline.fermata_for_spine(index),
                    Some(crate::BarlineFermata::PlacedSingle(_))
                )
            });
        }
        let KernLine::Data(contents) = line else {
            return false;
        };
        contents.iter().any(|content| {
            let (fermata, modifiers) = match content {
                SpineContent::Kern(crate::KernContent::Chord(chord)) => {
                    (chord.modifiers.fermata, Some(&chord.modifiers))
                }
                SpineContent::Kern(crate::KernContent::Grace(grace)) => {
                    (grace.modifiers.fermata, Some(&grace.modifiers))
                }
                SpineContent::Kern(crate::KernContent::Rest(rest)) => (rest.fermata, None),
                SpineContent::Kern(crate::KernContent::InvisibleRestSlice(slice)) => {
                    (slice.fermata, None)
                }
                _ => (None, None),
            };
            fermata.is_some_and(|fermata| fermata.placement.is_some())
                || modifiers.is_some_and(|modifiers| {
                    modifiers
                        .articulations
                        .iter()
                        .any(|mark| mark.placement.is_some())
                        || modifiers
                            .ornaments
                            .iter()
                            .any(|mark| mark.placement.is_some())
                        || modifiers
                            .bowing
                            .as_ref()
                            .is_some_and(|mark| mark.placement.is_some())
                        || modifiers
                            .breath
                            .is_some_and(|mark| mark.placement.is_some())
                        || modifiers
                            .harmonic
                            .is_some_and(|mark| mark.placement.is_some())
                })
        })
    })
}

fn document_has_placed_tie(document: &KernDocument) -> bool {
    document.lines().iter().any(|line| {
        let KernLine::Data(contents) = line else {
            return false;
        };
        contents.iter().any(|content| {
            let notes = match content {
                SpineContent::Kern(crate::KernContent::Note(note)) => {
                    return note
                        .modifiers
                        .ties
                        .iter()
                        .any(|tie| tie.placement().is_some());
                }
                SpineContent::Kern(crate::KernContent::Chord(chord)) => &chord.notes,
                SpineContent::Kern(crate::KernContent::Grace(grace)) => {
                    return grace.notes.iter().any(|note| {
                        note.modifiers
                            .ties
                            .iter()
                            .any(|tie| tie.placement().is_some())
                    });
                }
                _ => return false,
            };
            notes.iter().any(|note| {
                note.modifiers
                    .ties
                    .iter()
                    .any(|tie| tie.placement().is_some())
            })
        })
    })
}

/// One ottava bracket of the source: the tandems that open and close it and
/// the first and last notes or rests it covers, as 1-based source line and
/// field.
struct OttavaSpan {
    kind: crate::OttavaKind,
    open: (usize, usize),
    close: (usize, usize),
    start_anchor: Option<(usize, usize)>,
    end_anchor: Option<(usize, usize)>,
}

/// Ottava brackets in source order. An ottava is track state, so a bracket
/// covers every subspine of its track; Verovio ends it on the lowest layer of
/// the last onset. An ottava still open where the document ends has no
/// bracket.
fn ottava_spans(score: &Score<'_>) -> Vec<OttavaSpan> {
    let document = score.document();
    let topology = score.topology();
    let address = |line: usize, column: usize| (document.source_line_number(line), column + 1);
    let mut open: BTreeMap<crate::score::TrackId, (crate::OttavaKind, usize, usize)> =
        BTreeMap::new();
    let mut spans = Vec::new();
    for event in score.events() {
        if event.key != crate::score::StateKey::Ottava
            || event.outcome != crate::score::Outcome::Changed
        {
            continue;
        }
        let KernLine::TandemInterpretationLine(tandems) = &document.lines()[event.line_index]
        else {
            continue;
        };
        let TandemInterpretation::Ottava(ottava) = &tandems.tokens()[event.column] else {
            continue;
        };
        let track = topology.track_of(topology.layer_at(event.line_index, event.column));
        if ottava.active {
            open.insert(track, (ottava.kind, event.line_index, event.column));
            continue;
        }
        let Some((kind, open_line, open_column)) = open.remove(&track) else {
            continue;
        };
        let mut start_anchor = None;
        let mut end_anchor = None;
        for line in open_line + 1..event.line_index {
            let KernLine::Data(fields) = &document.lines()[line] else {
                continue;
            };
            let first = topology
                .track_columns(line, track)
                .find(|column| is_renderable_anchor(&fields[*column]));
            if let Some(column) = first {
                start_anchor.get_or_insert(address(line, column));
                end_anchor = Some(address(line, column));
            }
        }
        spans.push(OttavaSpan {
            kind,
            open: address(open_line, open_column),
            close: address(event.line_index, event.column),
            start_anchor,
            end_anchor,
        });
    }
    spans
}

fn is_renderable_anchor(content: &SpineContent) -> bool {
    matches!(
        content,
        SpineContent::Kern(
            crate::KernContent::Rest(_)
                | crate::KernContent::InvisibleRestSlice(_)
                | crate::KernContent::MultiMeasureRest(_)
                | crate::KernContent::Note(_)
                | crate::KernContent::Chord(_)
                | crate::KernContent::Grace(_)
        )
    )
}

fn populate_control_spans(output: &mut RenderOutput, score: &Score<'_>) {
    let provenance_by_source = source_provenance_index(output);

    for span in ottava_spans(score) {
        let kind = span.kind;
        let (Some(start_anchor), Some(end_anchor)) = (span.start_anchor, span.end_anchor) else {
            continue;
        };
        let source_open = RenderAddress::field(span.open.0, span.open.1);
        let source_close = RenderAddress::field(span.close.0, span.close.1);
        let source_start = RenderAddress::field(start_anchor.0, start_anchor.1);
        let source_end = RenderAddress::field(end_anchor.0, end_anchor.1);
        let Some(open) = provenance_by_source
            .get(&source_open)
            .map(|occurrences| occurrences.first)
        else {
            continue;
        };
        let Some(close) = provenance_by_source
            .get(&source_close)
            .map(|occurrences| occurrences.last)
        else {
            continue;
        };
        let Some(start) = provenance_by_source
            .get(&source_start)
            .map(|occurrences| occurrences.first)
        else {
            continue;
        };
        let Some(end) = provenance_by_source
            .get(&source_end)
            .map(|occurrences| occurrences.last)
        else {
            continue;
        };
        let rendered = RenderControlSpanRendered {
            open: open.rendered,
            close: close.rendered,
            start_anchor: start.rendered,
            end_anchor: end.rendered,
        };
        let relation =
            combined_relation([open.relation, close.relation, start.relation, end.relation]);
        output.control_spans.push(RenderControlSpan {
            kind: RenderControlKind::Ottava,
            value: match kind {
                crate::OttavaKind::OctaveUp => "8va",
                crate::OttavaKind::OctaveDown => "8ba",
                crate::OttavaKind::TwoOctavesUp => "15ma",
                crate::OttavaKind::TwoOctavesDown => "15ba",
            }
            .to_owned(),
            rendered,
            source: RenderControlSpanSource {
                open: Some(source_open),
                close: Some(source_close),
            },
            relation,
        });
    }
}

#[derive(Clone, Copy)]
struct SourceProvenanceEndpoint {
    rendered: RenderAddress,
    relation: RenderProvenanceRelation,
}

#[derive(Clone, Copy)]
struct SourceProvenanceOccurrences {
    first: SourceProvenanceEndpoint,
    last: SourceProvenanceEndpoint,
}

fn source_provenance_index(
    output: &RenderOutput,
) -> HashMap<RenderAddress, SourceProvenanceOccurrences> {
    let mut index = HashMap::new();
    for entry in &output.token_provenance {
        let Some(source) = entry.source else {
            continue;
        };
        let endpoint = SourceProvenanceEndpoint {
            rendered: entry.rendered,
            relation: entry.relation,
        };
        index
            .entry(source)
            .and_modify(|occurrences: &mut SourceProvenanceOccurrences| {
                occurrences.last = endpoint;
            })
            .or_insert(SourceProvenanceOccurrences {
                first: endpoint,
                last: endpoint,
            });
    }
    index
}

fn combined_relation(
    relations: impl IntoIterator<Item = RenderProvenanceRelation>,
) -> RenderProvenanceRelation {
    let mut result = RenderProvenanceRelation::Preserved;
    for relation in relations {
        result = match (result, relation) {
            (RenderProvenanceRelation::Synthesized, _)
            | (_, RenderProvenanceRelation::Synthesized) => RenderProvenanceRelation::Synthesized,
            (RenderProvenanceRelation::Expanded, _) | (_, RenderProvenanceRelation::Expanded) => {
                RenderProvenanceRelation::Expanded
            }
            (RenderProvenanceRelation::Normalized, _)
            | (_, RenderProvenanceRelation::Normalized) => RenderProvenanceRelation::Normalized,
            _ => RenderProvenanceRelation::Preserved,
        };
    }
    result
}

fn serialize_render_barline_for_spine(
    barline: &crate::Barline,
    spine_index: usize,
    bar_number: Option<u32>,
    placement_style: MarkPlacementStyle,
) -> String {
    serialize_barline_parts(
        barline.styles_for_spine(spine_index),
        barline.repeat_for_spine(spine_index),
        barline.fermata_for_spine(spine_index),
        bar_number,
        BarlineSpelling::Humdrum,
        placement_style,
    )
}

fn synthesize_default_kern_header(document: &mut KernDocument) {
    let needs_clef = !leading_header_has(document, TandemInterpretationLineKind::Clef);
    let needs_key_signature =
        !leading_header_has(document, TandemInterpretationLineKind::KeySignature);
    if !needs_clef && !needs_key_signature {
        return;
    }

    let Some(KernLine::SpineTypeLine(spine_types)) = document.lines().first() else {
        return;
    };
    let mut replacement = Vec::new();
    if needs_clef {
        replacement.push((default_kern_clef_line(spine_types), None));
    }
    if needs_key_signature {
        replacement.push((default_kern_key_signature_line(spine_types), None));
    }

    document.splice_lines(
        default_kern_header_insert_index(document)..default_kern_header_insert_index(document),
        replacement,
    );
}

fn default_kern_header_insert_index(document: &KernDocument) -> usize {
    let header = document.leading_tandem_header_range();
    if leading_header_has(document, TandemInterpretationLineKind::Clef)
        || leading_header_has(document, TandemInterpretationLineKind::KeySignature)
        || leading_header_has(document, TandemInterpretationLineKind::Instrument)
    {
        let last_pre_default_header = document.lines()[header.clone()].iter().rposition(|line| {
            matches!(
                line,
                KernLine::TandemInterpretationLine(tandems)
                    if matches!(
                        tandems.kind(),
                        TandemInterpretationLineKind::Clef
                            | TandemInterpretationLineKind::KeySignature
                            | TandemInterpretationLineKind::Instrument
                    )
            )
        });
        last_pre_default_header.map_or(header.start, |offset| header.start + offset + 1)
    } else {
        header.start
    }
}

fn push_metadata_records(
    document: &KernDocument,
    output: &mut RenderOutput,
    extended_records: bool,
) {
    if let Some(composer) = &document.metadata.composer {
        output.push(format!("!!!COM: {composer}"), None);
    }
    if extended_records {
        if let Some(parent_work) = &document.metadata.parent_work {
            output.push(format!("!!!OPR: {parent_work}"), None);
        }
    }
    if let Some(title) = &document.metadata.title {
        output.push(format!("!!!OTL: {title}"), None);
    }
    if extended_records {
        for alternative_title in &document.metadata.alternative_titles {
            output.push(format!("!!!OTA: {alternative_title}"), None);
        }
        if let Some(opus) = &document.metadata.opus {
            output.push(format!("!!!OPS: {opus}"), None);
        }
        if let Some(movement) = &document.metadata.movement {
            output.push(format!("!!!OMV: {movement}"), None);
        }
    }
    if let Some(movement_designation) = &document.metadata.movement_designation {
        output.push(format!("!!!OMD: {movement_designation}"), None);
    }
    // Semantic, not bibliographic: emitted whenever metadata is, never only
    // behind the extended record set, because dropping it changes how the
    // score is performed.
    if let Some(convention) = document.metadata.ornament_convention {
        output.push(
            format!(
                "!!!{}: {}",
                OrnamentConvention::RECORD_KEY,
                convention.record_value()
            ),
            None,
        );
    }
}

pub(crate) const DEFAULT_RDF_RECORDS: [&str; 5] = [
    "!!!RDF**kern: > = above",
    "!!!RDF**kern: < = below",
    "!!!RDF**kern: N = linked",
    "!!!RDF**kern: Z = editorial accidental, parentheses",
    "!!!RDF**kern: j = editorial accidental, brackets",
];

fn push_cross_staff_rdf_footer(output: &mut RenderOutput) {
    for record in DEFAULT_RDF_RECORDS {
        output.push(record.to_owned(), None);
    }
}

fn leading_header_has(document: &KernDocument, kind: TandemInterpretationLineKind) -> bool {
    document
        .leading_tandem_header_lines()
        .iter()
        .any(|line| matches!(line, KernLine::TandemInterpretationLine(tandems) if tandems.kind() == kind))
}

fn default_kern_clef_line(spine_types: &[SpineType]) -> KernLine {
    KernLine::TandemInterpretationLine(TandemInterpretationLine::from_homogeneous_tokens(
        spine_types
            .iter()
            .map(|spine_type| {
                if matches!(spine_type, SpineType::Kern) {
                    TandemInterpretation::Clef(Clef {
                        sign: ClefSign::G,
                        line: 2,
                        octave_shift: 0,
                        invisible: false,
                    })
                } else {
                    TandemInterpretation::Null
                }
            })
            .collect(),
    ))
}

fn default_kern_key_signature_line(spine_types: &[SpineType]) -> KernLine {
    KernLine::TandemInterpretationLine(TandemInterpretationLine::from_homogeneous_tokens(
        spine_types
            .iter()
            .map(|spine_type| {
                if matches!(spine_type, SpineType::Kern) {
                    TandemInterpretation::KeySignature(KeySignature::from_fifths(
                        Fifths::new(0).expect("zero fifths is a valid key signature"),
                    ))
                } else {
                    TandemInterpretation::Null
                }
            })
            .collect(),
    ))
}

fn serialize_local_comment(comment: &LocalComment, preserve_unsupported: bool) -> String {
    match comment {
        LocalComment::Null => "!".to_owned(),
        LocalComment::LayoutText(text) => {
            let placement = match text.placement {
                TextPlacement::Above => "a",
                TextPlacement::Below => "b",
            };
            let italic = if text.italic { ":i" } else { "" };
            format!("!LO:TX:{placement}{italic}:t={}", text.text)
        }
        LocalComment::Unsupported(raw) if preserve_unsupported || raw.starts_with("!LO:N:") => {
            raw.clone()
        }
        LocalComment::Unsupported(_) => "!".to_owned(),
    }
}

fn serialize_system_break(system_break: &SystemBreak) -> String {
    match &system_break.group {
        Some(group) => format!("!!LO:LB:g={group}"),
        None => "!!LO:LB".to_owned(),
    }
}

fn serialize_spine_type(
    spine_type: &SpineType,
    options: RenderSerializeOptions,
) -> Option<&'static str> {
    match spine_type {
        SpineType::Kern => Some("**kern"),
        SpineType::Dynam if options.include_dynamics => Some("**dynam"),
        SpineType::Dynam => None,
        SpineType::Fing if options.include_fingering => Some("**fing"),
        SpineType::Fing => None,
        SpineType::Text | SpineType::MxmlChords | SpineType::Other(_) => None,
    }
}

/// A clef the render output writes in place of a subspine's own token.
#[derive(Debug, Clone)]
struct ClefCompanion {
    clef: Clef,
    /// The field whose written clef this visible copy draws; `None` for a
    /// hidden companion.
    drawn_from: Option<usize>,
}

/// Clefs for the subspines of a track whose clef changes in only some of
/// them. Verovio copies a clef written in one subspine into every layer of
/// its staff and draws it once per layer that has no clef of its own, and
/// once per visible clef. A clef inside a layer's held note becomes a copy of
/// another clef at the same time and is not drawn. The measure's layers past
/// the subspines on the clef line get the first subspine's clef. The next
/// system's opening clef comes from the last layer whose clef is not such a
/// copy, so a hidden clef there leaves that system without one. The first
/// visible clef of the track is therefore drawn by the first subspine when
/// the measure has more layers than the line, otherwise by the last subspine
/// not holding a note, or the last subspine when all of them hold one. Every
/// other subspine gets it with `yy`.
fn clef_companions(
    document: &KernDocument,
    topology: &Topology,
    line_index: usize,
    tandems: &[TandemInterpretation],
) -> Vec<Option<ClefCompanion>> {
    let mut companions = vec![None; tandems.len()];
    let mut column = 0;
    while column < tandems.len() {
        let track = topology.track_of(topology.layer_at(line_index, column));
        let siblings = topology.track_columns(line_index, track);
        column = siblings.end;
        let Some(shown) = siblings.clone().find(|&sibling| {
            matches!(&tandems[sibling], TandemInterpretation::Clef(clef) if !clef.invisible)
        }) else {
            continue;
        };
        let TandemInterpretation::Clef(clef) = &tandems[shown] else {
            unreachable!("found above")
        };
        let drawing = if measure_layer_count(document, topology, line_index, track) > siblings.len()
        {
            siblings.start
        } else {
            siblings
                .clone()
                .rev()
                .find(|&sibling| !holds_a_note(document, topology, line_index, sibling))
                .unwrap_or(siblings.end - 1)
        };
        for sibling in siblings {
            if sibling == drawing {
                if sibling != shown {
                    companions[sibling] = Some(ClefCompanion {
                        clef: clef.clone(),
                        drawn_from: Some(shown),
                    });
                }
            } else if !matches!(&tandems[sibling], TandemInterpretation::Clef(clef) if clef.invisible)
            {
                companions[sibling] = Some(ClefCompanion {
                    clef: Clef {
                        invisible: true,
                        ..clef.clone()
                    },
                    drawn_from: None,
                });
            }
        }
    }
    companions
}

/// The most subspines `track` has on any line of the measure around
/// `line_index`, which is how many layers Verovio gives its staff there.
fn measure_layer_count(
    document: &KernDocument,
    topology: &Topology,
    line_index: usize,
    track: TrackId,
) -> usize {
    let lines = document.lines();
    let is_barline = |line: &KernLine| matches!(line, KernLine::Barline(_));
    let start = lines[..line_index]
        .iter()
        .rposition(is_barline)
        .unwrap_or(0);
    let end = lines[line_index..]
        .iter()
        .position(is_barline)
        .map_or(lines.len(), |offset| line_index + offset);
    (start..end)
        .map(|line| topology.track_columns(line, track).len())
        .max()
        .unwrap_or(0)
}

/// Whether the layer at `column` of interpretation line `line_index` is inside
/// a note there: its next data token is a null continuation.
fn holds_a_note(
    document: &KernDocument,
    topology: &Topology,
    line_index: usize,
    column: usize,
) -> bool {
    let layer = topology.layer_at(line_index, column);
    let Some((data_index, contents)) = document
        .lines()
        .iter()
        .enumerate()
        .skip(line_index + 1)
        .find_map(|(index, line)| match line {
            KernLine::Data(contents) => Some((index, contents)),
            _ => None,
        })
    else {
        return false;
    };
    topology
        .layers_at(data_index)
        .iter()
        .position(|&id| id == layer)
        .is_some_and(|column| {
            matches!(
                contents.get(column),
                Some(SpineContent::Kern(crate::KernContent::Null))
            )
        })
}

fn spine_types(topology: &Topology, layers: &[LayerId]) -> Vec<SpineType> {
    layers
        .iter()
        .map(|&layer| topology.spine_type(layer).clone())
        .collect()
}

fn is_render_spine(
    active_spines: &[SpineType],
    index: usize,
    options: RenderSerializeOptions,
) -> bool {
    active_spines
        .get(index)
        .is_some_and(|spine_type| is_render_spine_type(spine_type, options))
}

fn is_render_spine_type(spine_type: &SpineType, options: RenderSerializeOptions) -> bool {
    matches!(spine_type, SpineType::Kern)
        || (options.include_dynamics && matches!(spine_type, SpineType::Dynam))
        || (options.include_fingering && matches!(spine_type, SpineType::Fing))
}

fn serialize_render_content(
    content: &SpineContent,
    options: &OmrSerializeOptions,
    arpeggio_style: ArpeggioStyle,
) -> Option<String> {
    match content {
        SpineContent::Kern(content) => Some(match arpeggio_style {
            ArpeggioStyle::Preserve => {
                crate::serialize::omr::serialize_kern_content_preserving_arpeggio_scope(
                    content, options,
                )
            }
            ArpeggioStyle::CollapseCrossStaff => {
                crate::serialize::omr::serialize_kern_content(content, options)
            }
        }),
        SpineContent::Dynam(content) => Some(serialize_dynamic_content(content)),
        SpineContent::Fing(content) => Some(serialize_fing_content(content)),
        SpineContent::Text(_) | SpineContent::MxmlHarmony(_) | SpineContent::Opaque { .. } => None,
    }
}

fn serialize_fing_content(content: &FingContent) -> String {
    content.to_string()
}

/// Canonical text of a `**dynam` token: end, continuation, level, silence,
/// placeholder, start, with crescendo before diminuendo among the hairpins.
pub(crate) fn serialize_dynamic_content(content: &DynamContent) -> String {
    let marks = match content {
        DynamContent::Null => return ".".to_owned(),
        DynamContent::Raw { text, .. } => return text.0.clone(),
        DynamContent::Marks(marks) => marks,
    };
    let (mut ends, mut continues, mut begins) = (Vec::new(), Vec::new(), Vec::new());
    for direction in DynamicHairpinDirection::ALL {
        let [begin_symbol, continue_symbol, end_symbol] = hairpin_symbols(direction);
        let slots = marks.hairpin(direction);
        if let Some(end) = slots.end {
            let run = if end.to_line_end { 2 } else { 1 };
            ends.push(with_provenance(
                end_symbol.to_string().repeat(run),
                end.provenance,
            ));
        }
        if let Some(continuation) = slots.continues {
            continues.push(with_provenance(
                continue_symbol.to_string(),
                continuation.provenance,
            ));
        }
        if let Some(begin) = slots.begin {
            let mut part = begin_symbol.to_string();
            if begin.single_note {
                part.push(end_symbol);
            }
            begins.push(with_provenance(part, begin.provenance));
        }
    }
    let mut parts = ends;
    parts.extend(continues);
    if let Some(level) = &marks.level {
        parts.push(serialize_dynamic_level(level));
    }
    if marks.silence {
        parts.push("r".to_owned());
    }
    if marks.placeholder {
        parts.push("???".to_owned());
    }
    parts.extend(begins);
    if parts.is_empty() {
        return ".".to_owned();
    }
    parts.join(" ")
}

/// Begin, continuation and end characters of each direction.
fn hairpin_symbols(direction: DynamicHairpinDirection) -> [char; 3] {
    match direction {
        DynamicHairpinDirection::Crescendo => ['<', '(', '['],
        DynamicHairpinDirection::Diminuendo => ['>', ')', ']'],
    }
}

fn with_provenance(mut part: String, provenance: Option<DynamicProvenance>) -> String {
    match provenance {
        Some(DynamicProvenance::Explicit) => part.push('X'),
        Some(DynamicProvenance::Editorial) => part.push('x'),
        None => {}
    }
    part
}

fn serialize_dynamic_level(level: &DynamicLevel) -> String {
    let mut output = String::new();
    if level.subito {
        output.push('s');
    }
    match &level.value {
        DynamicLevelValue::Piano { count } => output.push_str(&"p".repeat((*count).into())),
        DynamicLevelValue::Forte { count } => output.push_str(&"f".repeat((*count).into())),
        DynamicLevelValue::MezzoPiano => output.push_str("mp"),
        DynamicLevelValue::MezzoForte => output.push_str("mf"),
        DynamicLevelValue::FortePiano => output.push_str("fp"),
        DynamicLevelValue::Rinforzando => output.push_str("rf"),
    }
    if level.sforzando {
        output.push('z');
    }
    if level.accent {
        output.push('v');
    }
    with_provenance(output, level.provenance)
}

fn non_null_line(tokens: Vec<String>, null_token: &str) -> Option<String> {
    if tokens.is_empty() || tokens.iter().all(|token| token == null_token) {
        None
    } else {
        Some(tokens.join("\t"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Address = (usize, usize);

    /// Open, close, start anchor and end anchor of every ottava bracket.
    fn spans(input: &str) -> Vec<(Address, Address, Option<Address>, Option<Address>)> {
        let document = KernDocument::parse(input).unwrap();
        let score = Score::resolve(&document, Framing::Complete).unwrap();
        ottava_spans(&score)
            .into_iter()
            .map(|span| (span.open, span.close, span.start_anchor, span.end_anchor))
            .collect()
    }

    #[test]
    fn ottava_span_anchors_follow_the_track_through_splits() {
        assert_eq!(
            spans("**kern\n*8va\n*^\n4c\t4e\n*v\t*v\n4g\n*X8va\n*-\n"),
            [((2, 1), (7, 1), Some((4, 1)), Some((6, 1)))]
        );
        // A marker in either subspine opens the bracket for the track.
        assert_eq!(
            spans("**kern\n*^\n*\t*8va\n4c\t4e\n*v\t*v\n4g\n*X8va\n*-\n"),
            [((3, 2), (7, 1), Some((4, 1)), Some((6, 1)))]
        );
    }

    #[test]
    fn ottava_span_ends_on_the_lowest_layer_of_the_last_onset() {
        assert_eq!(
            spans("**kern\n*8va\n*^\n4c\t4e\n4d\t4f\n*X8va\t*\n*v\t*v\n*-\n"),
            [((2, 1), (6, 1), Some((4, 1)), Some((5, 1)))]
        );
    }

    #[test]
    fn empty_ottava_span_has_no_anchors() {
        assert_eq!(
            spans("**kern\n*8va\n*X8va\n*-\n"),
            [((2, 1), (3, 1), None, None)]
        );
    }

    fn with_cross_staff_rdf_footer(body: &str) -> String {
        format!(
            "{body}!!!RDF**kern: > = above\n!!!RDF**kern: < = below\n!!!RDF**kern: N = linked\n!!!RDF**kern: Z = editorial accidental, parentheses\n!!!RDF**kern: j = editorial accidental, brackets\n"
        )
    }

    #[test]
    fn keeps_only_the_staff_declarations_verovio_numbers_staves_from() {
        let render = |text: &str| {
            let rendered = to_string(&KernDocument::parse(text).unwrap()).unwrap();
            let staff_lines = rendered
                .lines()
                .filter(|line| line.contains("*staff"))
                .map(str::to_owned)
                .collect::<Vec<_>>();
            (rendered, staff_lines)
        };

        // Dropping a left-to-right declaration would flip the staves.
        let (rendered, staff_lines) =
            render("**kern\t**dynam\t**kern\n*staff1\t*staff1\t*staff2\n*clefF4\t*\t*clefG2\n4C\tp\t4c\n*staff2\t*\t*staff1\n4D\t.\t4d\n*-\t*-\t*-\n");
        assert_eq!(staff_lines, ["*staff1\t*\t*staff2"], "{rendered}");
        // Standard order numbers the staves the same without them.
        let (rendered, staff_lines) =
            render("**kern\t**kern\n*staff2\t*staff1\n*clefF4\t*clefG2\n4C\t4c\n*-\t*-\n");
        assert!(staff_lines.is_empty(), "{rendered}");
    }

    #[test]
    fn hides_the_clef_verovio_would_copy_into_split_siblings() {
        let render = |text: &str| {
            to_string(&KernDocument::parse(text).unwrap())
                .unwrap()
                .lines()
                .filter(|line| line.contains("clefF4"))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };

        assert_eq!(
            render("**kern\n*^\n*clefF4\t*\n4C\t4E\n*v\t*v\n*-\n"),
            ["*clefF4yy\t*clefF4"]
        );
        assert_eq!(
            render("**kern\n*^\n*\t*clefF4\n4C\t4E\n*v\t*v\n*-\n"),
            ["*clefF4yy\t*clefF4"]
        );
        assert_eq!(
            render("**kern\n*^\n*clefF4\t*clefF4\n4C\t4E\n*v\t*v\n*-\n"),
            ["*clefF4yy\t*clefF4"]
        );
        assert_eq!(
            render("**kern\n*^\n*^\t*\n*clefF4\t*\t*\n4C\t4E\t4G\n*v\t*v\t*v\n*-\n"),
            ["*clefF4yy\t*clefF4yy\t*clefF4"]
        );
        assert_eq!(
            render("**kern\n*^\n*clefF4yy\t*\n4C\t4E\n*v\t*v\n*-\n"),
            ["*clefF4yy\t*"]
        );
        // An author's hidden clef in the last subspine would hide the next
        // system's opening clef, so the visible one replaces it.
        assert_eq!(
            render("**kern\n*^\n*clefF4\t*clefF4yy\n4C\t4E\n*v\t*v\n*-\n"),
            ["*clefF4yy\t*clefF4"]
        );
        assert_eq!(
            render("**kern\t**kern\n*clefG2\t*clefG2\n4c\t4e\n*clefF4\t*\n4C\t4e\n*-\t*-\n"),
            ["*clefF4\t*"]
        );
    }

    #[test]
    fn draws_the_split_spine_clef_from_the_last_subspine_not_holding_a_note() {
        let render = |text: &str| {
            to_string(&KernDocument::parse(text).unwrap())
                .unwrap()
                .lines()
                .filter(|line| line.contains("clefF4"))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };

        assert_eq!(
            render("**kern\n*^\n4c\t2e\n*clefF4\t*\n4C\t.\n4C\t4E\n*v\t*v\n*-\n"),
            ["*clefF4\t*clefF4yy"]
        );
        assert_eq!(
            render("**kern\n*^\n*^\t*\n4c\t4e\t2g\n*\t*clefF4\t*\n4C\t4E\t.\n2C\t2E\t2G\n*v\t*v\t*v\n*-\n"),
            ["*clefF4yy\t*clefF4\t*clefF4yy"]
        );
        // A note that ends at the clef does not hold across it.
        assert_eq!(
            render("**kern\n*^\n8c\t4e\n8d\t.\n*clefF4\t*\n4C\t4E\n*v\t*v\n*-\n"),
            ["*clefF4yy\t*clefF4"]
        );
        // Verovio copies the first subspine's clef into the measure's
        // remaining layers and opens the next system from the last one.
        assert_eq!(
            render("**kern\n*^\n*^\t*\n4c\t4e\t4g\n*\t*v\t*v\n4c\t4e\n*\t*clefF4\n2C\t2E\n=\t=\n*v\t*v\n*-\n"),
            ["*clefF4\t*clefF4yy"]
        );
        // With every subspine holding a note, the last one still draws it.
        assert_eq!(
            render("**kern\t**kern\n*^\t*\n2c\t2e\t4cc\n*clefF4\t*\t*\n.\t.\t4dd\n2C\t2E\t2cc\n*v\t*v\t*\n*-\t*-\n"),
            ["*clefF4yy\t*clefF4\t*"]
        );
    }

    #[test]
    fn clef_companions_trace_the_visible_clef_to_its_written_field() {
        let document = KernDocument::parse("**kern\n*^\n*clefF4\t*\n4C\t4E\n*v\t*v\n*-\n").unwrap();
        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();
        let line = output
            .canonical_text()
            .lines()
            .position(|line| line == "*clefF4yy\t*clefF4")
            .unwrap()
            + 1;
        let written = RenderAddress::field(3, 1);
        let drawn = RenderAddress::field(line, 2);

        assert!(output
            .token_provenance()
            .iter()
            .any(|token| token.rendered == drawn && token.source == Some(written)));
        assert!(output
            .symbol_provenance()
            .iter()
            .any(|symbol| symbol.rendered == drawn
                && symbol.kind == RenderSymbolKind::Clef
                && symbol.source == Some(written)));
    }

    #[test]
    fn refuses_a_document_whose_score_does_not_resolve() {
        let document = KernDocument::parse("**kern\t**kern\n4c\t4e\n*v\t*v\n*-\n").unwrap();

        assert!(matches!(
            to_string(&document),
            Err(DialectError::State(errors))
                if errors.to_string().contains("cannot merge spines from different original spines")
        ));
    }

    #[test]
    fn includes_exclusive_interpretation_and_terminal_line() {
        let document = KernDocument::parse("*clefG2\n4c\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n4c\n*-\n")
        );
    }

    #[test]
    fn preserves_invisible_clef_markers() {
        let document = KernDocument::parse("*clefG2yy\n4c\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2yy\n*k[]\n4c\n*-\n")
        );
    }

    #[test]
    fn ignoring_accidental_suppression_strips_only_single_y_markers() {
        let document =
            KernDocument::parse("**kern\n4c#y\n4dny\n4e-y\n4f##y\n4g--y\n4a#yy\n8qb-y\n*-\n")
                .unwrap();

        let serialized = to_string_with_options(
            &document,
            RenderSerializeOptions {
                accidental_suppression_style: AccidentalSuppressionStyle::IgnoreSource,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(
            serialized,
            with_cross_staff_rdf_footer(
                "**kern\n*clefG2\n*k[]\n4c#\n4dn\n4e-\n4f##\n4g--\n4a#yy\n8qb-\n*-\n"
            )
        );
    }

    #[test]
    fn render_note_provenance_maps_normalized_output_back_to_source_notes() {
        let document = KernDocument::parse(
            "!!!OTL: Provenance\n**text\t**kern\t**dynam\nlyrics\t4g 4c\tmf\n*-\t*-\t*-\n",
        )
        .unwrap();

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();

        assert_eq!(
            output.note_provenance(),
            &[
                RenderNoteProvenance {
                    rendered: RenderNoteAddress {
                        line_number: 5,
                        spine_index: 0,
                        chord_note_index: Some(0),
                    },
                    source: RenderNoteAddress {
                        line_number: 3,
                        spine_index: 1,
                        chord_note_index: Some(0),
                    },
                },
                RenderNoteProvenance {
                    rendered: RenderNoteAddress {
                        line_number: 5,
                        spine_index: 0,
                        chord_note_index: Some(1),
                    },
                    source: RenderNoteAddress {
                        line_number: 3,
                        spine_index: 1,
                        chord_note_index: Some(1),
                    },
                },
            ]
        );
    }

    #[test]
    fn render_provenance_covers_initial_symbols_and_paired_ottava() {
        let document = KernDocument::parse(
            "**kern\n*clefG2\n*k[]\n*8va\n4c'\n4r;\n4eT\n4fM\n4gSss\n4a:\n*X8va\n*-\n",
        )
        .unwrap();

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();
        let symbols = output.symbol_provenance();

        assert!(symbols.iter().any(|entry| {
            entry.kind == RenderSymbolKind::Clef
                && entry.rendered == RenderAddress::field(2, 1)
                && entry.source == Some(RenderAddress::field(2, 1))
                && entry.relation == RenderProvenanceRelation::Preserved
        }));
        assert!(symbols.iter().any(|entry| {
            entry.kind == RenderSymbolKind::Articulation
                && entry.source == Some(RenderAddress::field(5, 1))
        }));
        assert!(symbols.iter().any(|entry| {
            entry.kind == RenderSymbolKind::Rest && entry.source == Some(RenderAddress::field(6, 1))
        }));
        assert!(symbols.iter().any(|entry| {
            entry.kind == RenderSymbolKind::Fermata
                && entry.source == Some(RenderAddress::field(6, 1))
        }));
        assert!(symbols
            .iter()
            .any(|entry| entry.kind == RenderSymbolKind::Trill));
        assert!(symbols
            .iter()
            .any(|entry| entry.kind == RenderSymbolKind::Mordent));
        assert!(symbols
            .iter()
            .any(|entry| entry.kind == RenderSymbolKind::Turn));
        assert!(symbols
            .iter()
            .any(|entry| entry.kind == RenderSymbolKind::Arpeggio));
        assert_eq!(
            output.control_spans(),
            &[RenderControlSpan {
                kind: RenderControlKind::Ottava,
                value: "8va".to_owned(),
                rendered: RenderControlSpanRendered {
                    open: RenderAddress::field(4, 1),
                    close: RenderAddress::field(11, 1),
                    start_anchor: RenderAddress::field(5, 1),
                    end_anchor: RenderAddress::field(10, 1),
                },
                source: RenderControlSpanSource {
                    open: Some(RenderAddress::field(4, 1)),
                    close: Some(RenderAddress::field(11, 1)),
                },
                relation: RenderProvenanceRelation::Preserved,
            }]
        );
    }

    #[test]
    fn render_provenance_marks_chord_subtokens_expansion_and_synthesis() {
        let chord = KernDocument::parse("**kern\n*clefG2\n*k[]\n4c 4e\n*-\n").unwrap();
        let chord_output = to_render_output(&chord, RenderSerializeOptions::default()).unwrap();
        let note_symbols = chord_output
            .symbol_provenance()
            .iter()
            .filter(|entry| entry.kind == RenderSymbolKind::Note)
            .collect::<Vec<_>>();

        assert_eq!(note_symbols.len(), 2);
        assert_eq!(note_symbols[0].rendered.subtoken, Some(1));
        assert_eq!(note_symbols[0].source.unwrap().subtoken, Some(1));
        assert_eq!(note_symbols[1].rendered.subtoken, Some(2));
        assert_eq!(note_symbols[1].source.unwrap().subtoken, Some(2));

        let expanded = KernDocument::parse("**kern\n*M4/4\n=\nMM2r\n=\n*-\n").unwrap();
        let expanded_output =
            to_render_output(&expanded, RenderSerializeOptions::default()).unwrap();
        let multi_rests = expanded_output
            .symbol_provenance()
            .iter()
            .filter(|entry| entry.kind == RenderSymbolKind::MultiMeasureRest)
            .collect::<Vec<_>>();
        assert_eq!(multi_rests.len(), 2);
        assert!(multi_rests
            .iter()
            .all(|entry| entry.relation == RenderProvenanceRelation::Expanded));
        assert!(expanded_output.token_provenance().iter().any(|entry| {
            entry.source.is_none() && entry.relation == RenderProvenanceRelation::Synthesized
        }));
    }

    #[test]
    fn render_provenance_keeps_per_field_sources_when_initial_header_is_coalesced() {
        let document = KernDocument::parse(
            "**kern\t**kern\n*clefG2\t*\n*\t*clefF4\n*k[]\t*k[]\n4c\t4e\n*-\t*-\n",
        )
        .unwrap();

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();
        let clefs = output
            .symbol_provenance()
            .iter()
            .filter(|entry| entry.kind == RenderSymbolKind::Clef)
            .collect::<Vec<_>>();

        assert_eq!(clefs.len(), 2);
        assert_eq!(clefs[0].source, Some(RenderAddress::field(2, 1)));
        assert_eq!(clefs[1].source, Some(RenderAddress::field(3, 2)));
    }

    #[test]
    fn paired_ottava_anchors_cover_first_and_last_multi_rest_expansions() {
        let document = KernDocument::parse("**kern\n*M4/4\n*8va\n=\nMM2r\n=\n*X8va\n*-\n").unwrap();

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();
        let [span] = output.control_spans() else {
            panic!("expected one rendered Ottava span");
        };

        assert!(span.rendered.start_anchor.line < span.rendered.end_anchor.line);
        assert_eq!(span.source.open, Some(RenderAddress::field(3, 1)));
        assert_eq!(span.source.close, Some(RenderAddress::field(7, 1)));
        assert_eq!(span.relation, RenderProvenanceRelation::Expanded);
        assert_eq!(
            output
                .token_provenance()
                .iter()
                .find(|entry| entry.rendered == span.rendered.start_anchor)
                .and_then(|entry| entry.source),
            Some(RenderAddress::field(5, 1))
        );
        assert_eq!(
            output
                .token_provenance()
                .iter()
                .find(|entry| entry.rendered == span.rendered.end_anchor)
                .and_then(|entry| entry.source),
            Some(RenderAddress::field(5, 1))
        );
    }

    #[test]
    fn paired_ottava_provenance_keeps_initial_passthrough_lines_separate() {
        let document = KernDocument::parse(
            "**kern\t**kern\n*8va\t*\n*\t*8va\n4c\t4e\n*X8va\t*\n*\t*X8va\n*-\t*-\n",
        )
        .unwrap();

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();
        let sources = output
            .control_spans()
            .iter()
            .map(|span| (span.source.open, span.source.close))
            .collect::<Vec<_>>();

        assert_eq!(
            sources,
            vec![
                (
                    Some(RenderAddress::field(2, 1)),
                    Some(RenderAddress::field(5, 1)),
                ),
                (
                    Some(RenderAddress::field(3, 2)),
                    Some(RenderAddress::field(6, 2)),
                ),
            ]
        );
    }

    #[test]
    fn paired_ottava_provenance_preserves_repeated_initial_passthrough_origins() {
        let document = KernDocument::parse("**kern\n*8va\n*X8va\n*8va\n4c\n*X8va\n*-\n").unwrap();

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();
        let [span] = output.control_spans() else {
            panic!("expected the populated Ottava span");
        };

        assert_eq!(span.rendered.open, RenderAddress::field(6, 1));
        assert_eq!(span.rendered.start_anchor, RenderAddress::field(7, 1));
        assert_eq!(span.rendered.close, RenderAddress::field(8, 1));
        assert_eq!(span.source.open, Some(RenderAddress::field(4, 1)));
    }

    #[test]
    fn paired_ottava_provenance_resolves_dense_sequential_spans() {
        let mut source = String::from("**kern\n");
        for _ in 0..64 {
            source.push_str("*8va\n4c\n*X8va\n");
        }
        source.push_str("*-\n");
        let document = KernDocument::parse(&source).unwrap();

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();

        assert_eq!(output.control_spans().len(), 64);
        for (index, span) in output.control_spans().iter().enumerate() {
            let open_line = index * 3 + 2;
            assert_eq!(
                span.source,
                RenderControlSpanSource {
                    open: Some(RenderAddress::field(open_line, 1)),
                    close: Some(RenderAddress::field(open_line + 2, 1)),
                }
            );
            assert_eq!(span.rendered.start_anchor, span.rendered.end_anchor);
        }
    }

    #[test]
    fn paired_ottava_opened_in_one_subspine_yields_a_control_span() {
        let document =
            KernDocument::parse("**kern\n*^\n*\t*8va\n4c\t4e\n*v\t*v\n4g\n*X8va\n4a\n*-\n")
                .unwrap();

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();

        assert_eq!(output.control_spans().len(), 1);
        assert_eq!(
            output.control_spans()[0].source,
            RenderControlSpanSource {
                open: Some(RenderAddress::field(3, 2)),
                close: Some(RenderAddress::field(7, 1)),
            }
        );
    }

    #[test]
    fn synthesizes_time_signature_from_common_meter_symbol() {
        let document = KernDocument::parse("*clefG2\n*met(c)\n=\n1r\n=\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n*M4/4\n*met(c)\n=\n1r\n=\n*-\n")
        );
    }

    #[test]
    fn synthesizes_time_signature_from_cut_meter_symbol_per_spine() {
        let document =
            KernDocument::parse("*clefG2\t*clefF4\n*met(c|)\t*met(c|)\n=\t=\n1r\t1r\n=\t=\n")
                .unwrap();

        let serialized = to_string(&document).unwrap();

        assert!(serialized.contains("*M2/2\t*M2/2\n*met(c|)\t*met(c|)\n"));
    }

    #[test]
    fn does_not_synthesize_time_signature_when_meter_is_declared() {
        let document = KernDocument::parse("*clefG2\n*M3/4\n=\n2.c\n=\n").unwrap();

        let serialized = to_string(&document).unwrap();

        assert!(serialized.contains("*M3/4\n"));
        assert!(!serialized.contains("*M4/4"));
    }

    #[test]
    fn does_not_synthesize_over_declared_meter_with_meter_symbol() {
        let document = KernDocument::parse("*clefG2\n*M3/4\n*met(c)\n=\n2.c\n=\n").unwrap();

        let serialized = to_string(&document).unwrap();

        assert!(serialized.contains("*M3/4\n*met(c)\n"));
        assert!(!serialized.contains("*M4/4"));
    }

    #[test]
    fn hides_a_later_meter_with_an_empty_symbol_instead_of_yy() {
        for change in ["*M4/4yy", "*M4/4\n*met()"] {
            let document =
                KernDocument::parse(&format!("*clefG2\n*M3/4\n=\n2.c\n=\n{change}\n1d\n=\n"))
                    .unwrap();

            let serialized = to_string(&document).unwrap();

            assert!(
                serialized.contains("=\n*M4/4\n*met()\n1d\n"),
                "{serialized}"
            );
            assert!(!serialized.contains("yy"), "{serialized}");
        }
    }

    #[test]
    fn keeps_declared_time_signature_and_symbol_in_either_order() {
        for header in ["*M4/4\n*met(c)", "*met(c)\n*M4/4"] {
            let document = KernDocument::parse(&format!("*clefG2\n{header}\n=\n1c\n=\n")).unwrap();

            assert!(to_string(&document)
                .unwrap()
                .contains("*k[]\n*M4/4\n*met(c)\n=\n"));
        }
    }

    #[test]
    fn synthesized_time_signature_precedes_expanded_multi_measure_rests() {
        let document = KernDocument::parse("*clefG2\n*met(c)\n=\nMM2r\n=\n1cc\n=\n").unwrap();

        let serialized = to_string(&document).unwrap();

        assert!(serialized.contains("*M4/4\n*met(c)\n"));
        assert_eq!(serialized.matches("1r").count(), 2);
    }

    #[test]
    fn includes_one_exclusive_interpretation_per_staff() {
        let document = KernDocument::parse("**kern\t**kern\n4c\t4e\n*-\t*-\n").unwrap();

        assert!(to_string(&document)
            .unwrap()
            .starts_with("**kern\t**kern\n"));
    }

    #[test]
    fn preserves_fingering_spines_for_rendering() {
        let document = KernDocument::parse(
            "**kern\t**fing\t**dynam\n*\t*left\t*\n=1\t=1\t=1\n4c\t1\tmf\n4e 4g\t2 4\t.\n*-\t*-\t*-\n",
        )
        .unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer(
                "**kern\t**fing\t**dynam\n*clefG2\t*\t*\n*k[]\t*\t*\n*\t*left\t*\n=\t=\t=\n4c\t1\tmf\n4e 4g\t2 4\t.\n*-\t*-\t*-\n"
            )
        );
    }

    #[test]
    fn preserves_source_chord_order_for_rendered_notehead_ids() {
        let document = KernDocument::parse("**kern\n4g 4c 4e\n*-\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n4g 4c 4e\n*-\n")
        );
    }

    #[test]
    fn preserves_invisible_barlines_for_rendering() {
        let document = KernDocument::parse("**kern\n=-\n4c\n=1-\n4d\n*-\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n=-\n4c\n=-\n4d\n*-\n")
        );
    }

    #[test]
    fn sequential_bar_numbers_preserve_invisible_barlines_for_rendering() {
        let document = KernDocument::parse("**kern\n*M4/4\n4c\n=-\n4d\n*-\n").unwrap();

        assert_eq!(
            to_string_with_options(
                &document,
                RenderSerializeOptions {
                    bar_number_style: BarNumberStyle::Sequential,
                    ..Default::default()
                },
            )
            .unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n=2-\n4d\n*-\n")
        );
    }

    #[test]
    fn preserves_beam_markers() {
        let document = KernDocument::parse("**kern\n8cL\n8dJ\n*-\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n8cL\n8dJ\n*-\n")
        );
    }

    #[test]
    fn preserves_stem_direction_markers() {
        let document = KernDocument::parse("**kern\n4c/\n4d\\\n4e 4g/\n*-\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n4c/\n4d\\\n4e 4g/\n*-\n")
        );
    }

    #[test]
    fn preserves_cross_staff_placement_markers() {
        let document = KernDocument::parse("**kern\n4c>\n4d<\n8qee>\n*-\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n4c>\n4d<\n8qee>\n*-\n")
        );
    }

    #[test]
    fn cross_staff_display_lowering_keeps_analysis_and_source_grammar_canonical() {
        let input = "**kern\t**kern\n*clefF4\t*clefG2\n4C#X>\t4f#X< 4anX< 4b-y<\n*-\t*-\n";
        let document = KernDocument::parse(input).unwrap();
        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();
        assert!(output.text().contains("4C>#X\t4f<#X 4a<nX 4b<-y"));
        assert!(output.canonical_text().contains("4C#X>\t4f#X< 4anX< 4b-y<"));
        assert!(KernDocument::parse(&output.text()).is_err());
        assert!(KernDocument::parse(&output.canonical_text()).is_ok());
        assert_eq!(
            canonical_data_text(&output.text()).unwrap(),
            output.canonical_text()
        );
        assert_eq!(
            crate::serialize::SerializePreset::HumdrumKern.serialize_document(&document),
            output.canonical_text()
        );
    }

    #[test]
    fn render_decoding_never_changes_text_spines() {
        let canonical = "**kern\t**text\n4F#X<\tF<#X\n*-\t*-\n!!!RDF**kern: < = below\n";
        let lowered = "**kern\t**text\n4F<#X\tF<#X\n*-\t*-\n!!!RDF**kern: < = below\n";
        assert_eq!(canonical_data_text(lowered).unwrap(), canonical);
    }

    #[test]
    fn prepared_data_lowers_without_reserializing_other_content() {
        let canonical = "!! untouched\n**kern\t**text\n=47\t=47\n4F#X<\tF<#X\n*-\t*-\n!!!RDF**kern: < = below\n";
        let lowered = "!! untouched\n**kern\t**text\n=47\t=47\n4F<#X\tF<#X\n*-\t*-\n!!!RDF**kern: < = below\n";
        assert_eq!(lower_prepared_text(canonical).unwrap(), lowered);
        assert!(lower_prepared_text(lowered).is_err());
    }

    #[test]
    fn preserves_staff_local_and_cross_staff_arpeggios_with_source_mapping() {
        let document =
            KernDocument::parse("**kern\t**kern\n4c:\t4e:\n4d::\t4f::\n*-\t*-\n").unwrap();

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();
        let text = output.text();
        let lines = text.lines().collect::<Vec<_>>();
        let cross_staff_row = lines
            .iter()
            .position(|line| *line == "4d::\t4f::")
            .expect("cross-staff arpeggio row is preserved");

        assert!(lines.contains(&"4c:\t4e:"));
        assert_eq!(output.sources()[cross_staff_row], Some(3));

        let reparsed = KernDocument::parse(&text).unwrap();
        let arpeggios = reparsed
            .lines()
            .iter()
            .filter_map(|line| match line {
                KernLine::Data(contents) => Some(contents),
                _ => None,
            })
            .flatten()
            .filter_map(|content| match content {
                SpineContent::Kern(crate::KernContent::Chord(chord)) => chord.modifiers.arpeggio,
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            arpeggios,
            vec![
                crate::types::Arpeggio::StaffLocal,
                crate::types::Arpeggio::StaffLocal,
                crate::types::Arpeggio::CrossStaff,
                crate::types::Arpeggio::CrossStaff,
            ]
        );
    }

    #[test]
    fn preserves_disjunct_and_linked_ties() {
        let document = KernDocument::parse("**kern\n4c[[\n4c]]\nN[8f<\n2.fN]\n*-\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n4c[[\n4c]]\n8f<N[\n2.fN]\n*-\n")
        );
    }

    #[test]
    fn declares_n_for_linked_slurs_without_the_default_footer() {
        let document = KernDocument::parse("**kern\t**kern\n4C\t4cN(\n4DN)\t4d\n*-\t*-\n").unwrap();
        let body = "**kern\t**kern\n*clefG2\t*clefG2\n*k[]\t*k[]\n4C\t4cN(\n4DN)\t4d\n*-\t*-\n";

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer(body)
        );
        assert_eq!(
            to_render_output(
                &document,
                RenderSerializeOptions {
                    include_cross_staff_rdf_footer: false,
                    ..RenderSerializeOptions::default()
                },
            )
            .unwrap()
            .text(),
            format!("{body}!!!RDF**kern: N = linked\n")
        );
    }

    #[test]
    fn declares_above_and_below_for_placed_marks_without_the_default_footer() {
        let document = KernDocument::parse("**kern\n4cT>\n*-\n").unwrap();
        let options = RenderSerializeOptions {
            include_cross_staff_rdf_footer: false,
            ..RenderSerializeOptions::default()
        };

        assert_eq!(
            to_render_output(&document, options).unwrap().text(),
            "**kern\n*clefG2\n*k[]\n4cT>\n*-\n!!!RDF**kern: > = above\n!!!RDF**kern: < = below\n"
        );
    }

    #[test]
    fn cross_staff_rdf_footer_is_last_and_unsourced() {
        let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();
        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();

        assert!(output.text().ends_with(
            "*-\n!!!RDF**kern: > = above\n!!!RDF**kern: < = below\n!!!RDF**kern: N = linked\n!!!RDF**kern: Z = editorial accidental, parentheses\n!!!RDF**kern: j = editorial accidental, brackets\n"
        ));
        assert_eq!(output.sources().last(), Some(&None));
        assert_eq!(
            output.sources().get(output.sources().len() - 2),
            Some(&None)
        );
        assert_eq!(
            output.sources().get(output.sources().len() - 3),
            Some(&None)
        );
    }

    #[test]
    fn canonicalizes_initial_header_across_leading_barline() {
        let document = KernDocument::parse(
            [
                "!!!COM: Traditional",
                "!!!OTL: I Will Go Home to Kintail",
                "**kern",
                "*clefG2",
                "*k[]",
                "=",
                "*MM40",
                "*M4/4",
                "*k[]",
                "*clefG2",
                "8ggqq",
                "*-",
            ]
            .join("\n")
            .as_str(),
        )
        .unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer(
                "!!!COM: Traditional\n!!!OTL: I Will Go Home to Kintail\n**kern\n*clefG2\n*k[]\n*M4/4\n*MM40\n8qqgg\n*-\n"
            )
        );
    }

    #[test]
    fn expands_multi_measure_rests_to_standard_kern() {
        let document = KernDocument::parse("**kern\n*M3/4\nMM2r\n=\n*-\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n*M3/4\n=\n2.r\n=\n2.r\n=\n*-\n")
        );
    }

    #[test]
    fn render_output_sources_track_expanded_multi_measure_rests() {
        let document = KernDocument::parse("**kern\n*M3/4\nMM2r\n=\n*-\n").unwrap();

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();

        assert_eq!(
            output.text(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n*M3/4\n=\n2.r\n=\n2.r\n=\n*-\n")
        );
        assert_eq!(
            output.sources(),
            &[
                Some(1),
                None,
                None,
                Some(2),
                Some(3),
                Some(3),
                Some(3),
                Some(3),
                Some(4),
                Some(5),
                None,
                None,
                None,
                None,
                None,
            ]
        );
    }

    #[test]
    fn generated_line_keeps_its_origin_through_multi_measure_rest_expansion() {
        let mut document = KernDocument::parse("**kern\n*M3/4\nMM2r\n=\n4c\n*-\n").unwrap();
        let grace = KernLine::parse(1, "8qd").unwrap();
        let note_index = document.lines().len() - 2;
        document.insert_line(
            note_index,
            grace.clone(),
            crate::types::LineOrigin::Generated,
        );

        let prepared = prepare_render_document(&document, RenderSerializeOptions::default())
            .expect("render preparation succeeds");
        let grace_index = prepared
            .lines()
            .iter()
            .position(|line| *line == grace)
            .expect("the generated grace survives render preparation");
        assert_eq!(
            prepared.origin(grace_index),
            &crate::types::LineOrigin::Generated
        );

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();
        assert_eq!(
            output.text(),
            with_cross_staff_rdf_footer(
                "**kern\n*clefG2\n*k[]\n*M3/4\n=\n2.r\n=\n2.r\n=\n8qd\n4c\n*-\n"
            )
        );
        let sources_by_rendered_line = output
            .note_provenance()
            .iter()
            .map(|note| (note.rendered.line_number, note.source.line_number))
            .collect::<Vec<_>>();
        assert_eq!(sources_by_rendered_line, [(10, 0), (11, 5)]);
        assert_eq!(output.sources()[9], None);
    }

    #[test]
    fn moves_system_break_after_expanded_multi_measure_rest_for_verovio() {
        let document =
            KernDocument::parse("**kern\n*M3/4\nMM2r\n!!LO:LB:g=original\n=\n4c\n*-\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer(
                "**kern\n*clefG2\n*k[]\n*M3/4\n=\n2.r\n!!LO:LB:g=original\n=\n2.r\n=\n4c\n*-\n"
            )
        );
    }

    #[test]
    fn moved_multi_measure_rest_system_break_keeps_source_line() {
        let document =
            KernDocument::parse("**kern\n*M3/4\nMM2r\n!!LO:LB:g=original\n=\n4c\n*-\n").unwrap();

        let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();

        assert_eq!(
            output.sources(),
            &[
                Some(1),
                None,
                None,
                Some(2),
                Some(3),
                Some(3),
                Some(4),
                Some(3),
                Some(3),
                Some(5),
                Some(6),
                Some(7),
                None,
                None,
                None,
                None,
                None,
            ]
        );
    }

    #[test]
    fn render_output_has_one_source_per_output_line() {
        let inputs = [
            "**kern\n4c\n*-\n",
            "**kern\t**fing\t**dynam\n*\t*left\t*\n=1\t=1\t=1\n4c\t1\tmf\n4e 4g\t2 4\t.\n*-\t*-\t*-\n",
            "!!!COM: Adler, Clara\n!!!OTL: Prelude\n!!!OMD: Allegro\n**kern\n*I\"Cello\n*MM96\n!LO:TX:a:t=Allegro\n4c\n*-\n",
            "**kern\n*M3/4\nMM2r\n=\n*-\n",
        ];

        for input in inputs {
            let document = KernDocument::parse(input).unwrap();
            let output = to_render_output(&document, RenderSerializeOptions::default()).unwrap();
            let line_count = output.text().trim_end_matches('\n').split('\n').count();

            assert_eq!(output.sources().len(), line_count, "{input}");
        }
    }

    #[test]
    fn does_not_duplicate_existing_barline_before_expanded_multi_measure_rests() {
        let document = KernDocument::parse("**kern\n*M3/4\n=\nMM2r\n=\n*-\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n*M3/4\n=\n2.r\n=\n2.r\n=\n*-\n")
        );
    }

    #[test]
    fn optionally_emits_sequential_bar_numbers() {
        let document = KernDocument::parse("**kern\n*M4/4\n4c\n=\n4d\n==\n*-\n").unwrap();

        assert_eq!(
            to_string_with_options(
                &document,
                RenderSerializeOptions {
                    bar_number_style: BarNumberStyle::Sequential,
                    ..Default::default()
                },
            )
            .unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n=2\n4d\n==\n*-\n")
        );
    }

    #[test]
    fn renders_metadata_instrument_tempo_and_layout_text() {
        let document = KernDocument::parse(
            "!!!COM: Adler, Clara\n!!!OTL: Prelude\n!!!OMD: Allegro\n**kern\n*I\"Cello\n*MM96\n!LO:TX:a:t=Allegro\n4c\n*-\n",
        )
        .unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer(
                "!!!COM: Adler, Clara\n!!!OTL: Prelude\n!!!OMD: Allegro\n**kern\n*clefG2\n*k[]\n*I\"Cello\n*MM96\n!LO:TX:a:t=Allegro\n4c\n*-\n"
            )
        );
        assert_eq!(document.to_compact_omr_string(), "*clefG2\n*k[]\n4c\n");
    }

    #[test]
    fn renders_note_layout_comments_but_omr_serialization_omits_them() {
        let document =
            KernDocument::parse("**kern\t**kern\n!LO:N:head=invis\t!LO:N:head=x\n4c\t4e\n*-\t*-\n")
                .unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer(
                "**kern\t**kern\n*clefG2\t*clefG2\n*k[]\t*k[]\n!LO:N:head=invis\t!LO:N:head=x\n4c\t4e\n*-\t*-\n"
            )
        );
        assert_eq!(
            document.to_compact_omr_string(),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n4c\t4e\n"
        );
    }

    #[test]
    fn render_preserves_modern_and_original_instrument_labels() {
        let document = KernDocument::parse("**kern\n*mI\"Violin\n*oI'Vl.\n4c\n*-\n").unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n*mI\"Violin\n*oI'Vl.\n4c\n*-\n")
        );
        assert_eq!(document.to_compact_omr_string(), "*clefG2\n*k[]\n4c\n");
    }

    #[test]
    fn omits_unsupported_local_comments() {
        let document = KernDocument::parse(
            "**kern\n!LO:TX:a:t=[quarter]=70\n!LO:TX:a:t=8va  _  _  _\n4c\n*-\n",
        )
        .unwrap();

        assert_eq!(
            to_string(&document).unwrap(),
            with_cross_staff_rdf_footer("**kern\n*clefG2\n*k[]\n4c\n*-\n")
        );
    }
}
