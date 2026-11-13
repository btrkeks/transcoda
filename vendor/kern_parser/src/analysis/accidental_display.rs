//! Accidental display in Verovio 6.2.1, independently ported from humlib's
//! `analyzeKernAccidentals` / `analyzeOttavas` and `iohumdrum.cpp::convertNote`.
//!
//! This models engraving, never sounding pitch. Quirks are intentional: source
//! tracks own cross-staff state, subspines share it, natural concurrent duplicates
//! do not force display, ornaments use unshifted base-40 pitches during ottavas,
//! and negative diatonic indexes skip humlib analysis. The upstream 70-step array
//! is preserved; indexes beyond it return an error instead of invoking undefined
//! behavior. Pass the parsed standard text actually sent to Verovio, not a
//! transformed AST whose source spellings or addresses have changed.

pub(crate) mod canonical;
mod hidden;
pub(crate) use hidden::active_hide;
pub use hidden::{
    AccidentalConventions, HiddenRisk, HiddenRiskClass, OctaveCarryVariant, TieCarryVariant,
};
mod layout;
pub use canonical::{
    analyze_source_accidental_display, canonicalize_accidental_display, CanonicalAccidentalDisplay,
    DisplayChange, DisplayChangeKind, DisplaySkip, GraceFollowers,
};
mod rest_layers;
mod tremolo;

use crate::{
    document::{Document, Framing},
    score::{LayerId, LayerOrigin, Score, StateErrors},
    types::{Accidental, DiatonicStep},
    ClefSign, KernLine, OttavaKind, ParseDialect, SpineContent, SpineType, TandemInterpretation,
};
use serde::Serialize;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]

pub struct NoteAddress {
    pub line: usize,
    pub field: usize,
    /// Always one-based, including 1 for single-note tokens (whose SVG id omits S).
    pub subtoken: usize,
}
impl NoteAddress {
    pub fn from_verovio_id(id: &str) -> Option<Self> {
        let (line, rest) = id.strip_prefix("note-L")?.split_once('F')?;
        let (field, subtoken) = rest.split_once('S').unwrap_or((rest, "1"));
        Some(Self {
            line: line.parse().ok()?,
            field: field.parse().ok()?,
            subtoken: subtoken.parse().ok()?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]

pub enum AccidentalGlyph {
    Flat,
    Natural,
    Sharp,
    DoubleSharp,
    DoubleFlat,
    NaturalFlat,
    NaturalSharp,
    TwoSharps,
    TripleSharp,
    TripleFlat,
}
impl AccidentalGlyph {
    fn from_count(count: i32) -> Option<Self> {
        Some(match count {
            -3 => Self::TripleFlat,
            -2 => Self::DoubleFlat,
            -1 => Self::Flat,
            0 => Self::Natural,
            1 => Self::Sharp,
            2 => Self::DoubleSharp,
            3 => Self::TripleSharp,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]

pub enum DisplayCause {
    DiffersFromState,
    AfterGraceCautionary,
    TieAcrossBarlineCautionary,
    OrnamentStateCautionary,
    ConcurrentDuplicate,
    RedundantNatural,
    ForcedAccidental,
    HiddenAccidental,
    HiddenNote,
    TieContinuation,
    SatisfiedByKey,
    SatisfiedByMeasureMemory,
    BelowAnalysisRange,
    EditorialAccidental,
    LayoutOverride,
    TremoloSuppression,
    TremoloAccidentalTransfer,
    UnpitchedClef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct AccidentalDisplay {
    pub address: NoteAddress,
    pub source_address: Option<NoteAddress>,
    pub token: String,
    pub shown: bool,
    /// The glyph actually printed, or None when no glyph is drawn.
    pub glyph: Option<AccidentalGlyph>,
    pub cautionary: bool,
    /// Whether the printed accidental has parentheses or brackets.
    pub enclosed: bool,
    pub enclosure: crate::document::DisplayEnclosure,
    pub cause: DisplayCause,
    pub default_shown: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_risk: Option<HiddenRisk>,
    pub conventions: AccidentalConventions,
    #[serde(skip)]
    pub(crate) accidental_layout: bool,
    #[serde(skip)]
    pub(crate) bare_glyph: Option<AccidentalGlyph>,
    /// Previous event at this written diatonic pitch in the bar was a grace
    /// note with the same absolute accidental, and this event is normal.
    pub grace_predecessor_same_accidental: bool,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct AccidentalDisplayError {
    pub line: Option<usize>,
    pub message: String,
}
fn error(line: usize, message: impl Into<String>) -> AccidentalDisplayError {
    AccidentalDisplayError {
        line: Some(line),
        message: format!("line {line}: {}", message.into()),
    }
}
impl From<StateErrors> for AccidentalDisplayError {
    fn from(errors: StateErrors) -> Self {
        Self {
            line: None,
            message: errors.to_string(),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct State {
    count: i32,
    caution: Option<DisplayCause>,
    memory: bool,
}
/// humlib's accidental memory for one `**kern` track.
struct Track {
    key: [i32; 7],
    normal: [State; 70],
    grace: [State; 70],
    first: bool,
    /// Verovio's tremolo flag, which skips full-measure rest layers; see
    /// [`rest_layers`].
    tremolo: bool,
    events: BTreeMap<i32, (i32, bool)>,
}
impl Default for Track {
    fn default() -> Self {
        Self {
            key: [0; 7],
            normal: [State::default(); 70],
            grace: [State::default(); 70],
            first: false,
            tremolo: false,
            events: BTreeMap::new(),
        }
    }
}
impl Track {
    fn reset(&mut self) {
        for i in 0..70 {
            let state = State {
                count: self.key[i % 7],
                ..State::default()
            };
            self.normal[i] = state;
            self.grace[i] = state;
        }
    }
}

/// Layout parameters and beam state of one layer.
#[derive(Clone, Default)]
struct Field {
    layouts: Vec<(usize, String)>,
    beam: tremolo::BeamCollector,
}
/// Analyze exact retained source spelling using its parsed pitches and topology.
/// Only complete standard Humdrum documents are accepted. Lower other dialects
/// through `serialize::render::to_render_output` first and attach its provenance.
pub fn analyze_accidental_display(
    document: &Document,
) -> Result<Vec<AccidentalDisplay>, AccidentalDisplayError> {
    if document.options().dialect != ParseDialect::HumdrumKern
        || document.options().framing != Framing::Complete
    {
        return Err(error(
            1,
            "accidental analysis requires complete standard Humdrum render text",
        ));
    }
    let score = Score::resolve(document.score(), Framing::Excerpt)?;
    analyze(document, &score)
}

/// [`analyze_accidental_display`] over the document's resolved score.
fn analyze(
    document: &Document,
    score: &Score<'_>,
) -> Result<Vec<AccidentalDisplay>, AccidentalDisplayError> {
    let topology = score.topology();
    let raw: Vec<_> = document.source().lines().collect();
    let editorial: Vec<_> = raw
        .iter()
        .filter_map(|line| {
            let (sign, value) = line.strip_prefix("!!!RDF**kern:")?.split_once('=')?;
            value
                .contains("editorial accidental")
                .then(|| (sign.trim().to_owned(), value.to_owned()))
        })
        .collect();
    // Indexed by `LayerId`.
    let mut fields = vec![Field::default(); topology.layer_count()];
    // humlib tracks are `**kern` spines; indexed by `TrackId`.
    let mut kern_tracks = 0;
    let slots: Vec<Option<usize>> = topology
        .track_ids()
        .map(|track| {
            (topology.track(track).spine_type == SpineType::Kern).then(|| {
                kern_tracks += 1;
                kern_tracks - 1
            })
        })
        .collect();
    let slot = |layer: LayerId| slots[topology.track_of(layer).index()];
    let mut tracks: Vec<Track> = (0..kern_tracks).map(|_| Track::default()).collect();
    let mut exclusive_seen = false;
    let mut result = Vec::new();
    let mut visual = Vec::new();
    let mut previous_source = 0;
    let mut global_layouts = Vec::new();
    for (li, line) in document.score().lines().iter().enumerate() {
        let layers = topology.layers_at(li);
        let Some(ln) = document.score().source(li) else {
            continue;
        };
        // The AST drops some unsupported local comments. Their layout
        // parameters still belong to the next source token. Retain them using
        // the topology established by typed records.
        if ln > previous_source {
            for (offset, skipped) in raw[previous_source..ln - 1].iter().enumerate() {
                retain_layouts(
                    &mut fields,
                    layers,
                    &mut global_layouts,
                    skipped,
                    previous_source + offset + 1,
                )?;
            }
            previous_source = ln;
        }

        let tokens: Vec<_> = raw
            .get(ln - 1)
            .ok_or_else(|| error(ln, "missing retained source"))?
            .split('\t')
            .collect();
        if !matches!(line, KernLine::LocalCommentLine(_))
            && tokens.iter().any(|t| !matches!(*t, "*" | "."))
        {
            if matches!(line, KernLine::Data(_)) {
                for layer in layers {
                    let field = &mut fields[layer.index()];
                    field.layouts.extend(global_layouts.iter().cloned());
                    field.layouts.sort_by_key(|(line, _)| *line);
                }
            }
            global_layouts.clear();
        }
        match line {
            KernLine::SpineTypeLine(_) => {
                if std::mem::replace(&mut exclusive_seen, true) {
                    return Err(error(ln, "exclusive interpretation change is unsupported"));
                }
            }
            KernLine::SpineManipulatorLine(_) => {
                for (layer, token) in layers.iter().zip(&tokens) {
                    if *token != "*" {
                        fields[layer.index()].layouts.clear();
                    }
                }
                for &layer in topology.layers_after(li) {
                    if layers.contains(&layer) {
                        continue;
                    }
                    fields[layer.index()] = match &topology.layer(layer).origin {
                        LayerOrigin::SplitOf(parent) => fields[parent.index()].clone(),
                        LayerOrigin::MergeOf(inputs) => fields[inputs[0].index()].clone(),
                        LayerOrigin::Exclusive => Field::default(),
                    };
                }
            }
            KernLine::TandemInterpretationLine(tandems) => {
                for (fi, (tandem, &layer)) in tandems.tokens().iter().zip(layers).enumerate() {
                    if tokens[fi] != "*" {
                        fields[layer.index()].layouts.clear();
                    }
                    let Some(ti) = slot(layer) else { continue };
                    let track = &mut tracks[ti];
                    match tandem {
                        TandemInterpretation::KeySignature(_) if tokens[fi].starts_with("*k[") => {
                            // humlib resets accidental memory at every key
                            // signature token, even a restated one.
                            let key = score.state_after(li, fi).track.key_signature;
                            for step in [
                                crate::types::DiatonicStep::C,
                                crate::types::DiatonicStep::D,
                                crate::types::DiatonicStep::E,
                                crate::types::DiatonicStep::F,
                                crate::types::DiatonicStep::G,
                                crate::types::DiatonicStep::A,
                                crate::types::DiatonicStep::B,
                            ] {
                                track.key[step_index(step)] = accidental_count(
                                    key.and_then(|key| key.accidental_for_step(step)).as_ref(),
                                );
                            }
                            track.reset();
                        }
                        TandemInterpretation::Tremolo(tremolo)
                            if !rest_layers::skips_staff_state(
                                document.score(),
                                li,
                                fi,
                                score.state_at(li, fi).track.meter.measure_duration(),
                            ) =>
                        {
                            track.tremolo = tremolo.active;
                        }
                        _ => {}
                    }
                }
            }
            KernLine::LocalCommentLine(_) => {
                retain_layouts(&mut fields, layers, &mut global_layouts, raw[ln - 1], ln)?;
            }
            KernLine::Barline(bar) => {
                for layer in layers {
                    let field = &mut fields[layer.index()];
                    field.beam.clear();
                    field.layouts.clear();
                }
                for (fi, &layer) in layers.iter().enumerate() {
                    if let Some(ti) = slot(layer) {
                        tracks[ti].events.clear();
                        if !bar.is_invisible_for_spine(fi) {
                            for track in &mut tracks {
                                track.first = true;
                            }
                            tracks[ti].reset();
                        }
                    }
                }
            }
            KernLine::Data(contents) => {
                let mut concurrent = [0; 70];
                let mut last = None;
                for (fi, (content, &layer)) in contents.iter().zip(layers).enumerate() {
                    let SpineContent::Kern(content) = content else {
                        continue;
                    };
                    let ti = slot(layer).expect("kern content lies in a **kern track");
                    let state = score.state_at(li, fi).track;
                    let shift = state.ottava.map_or(0, octaves);
                    let unpitched = state
                        .clef
                        .as_ref()
                        .is_some_and(|clef| clef.sign == ClefSign::Percussion);
                    if content.noteheads().next().is_none() {
                        continue;
                    }
                    if last != Some(ti) {
                        concurrent.fill(0);
                    }
                    last = Some(ti);
                    let subtokens: Vec<_> = tokens[fi].split(' ').collect();
                    let record_start = result.len();
                    for head in content.noteheads() {
                        let si = head.chord_note_index.unwrap_or(0);
                        let token = *subtokens
                            .get(si)
                            .ok_or_else(|| error(ln, "chord subtoken mapping failed"))?;
                        let address = NoteAddress {
                            line: ln,
                            field: fi + 1,
                            subtoken: si + 1,
                        };
                        let pitch = head.note.pitch();
                        let step = step_index(pitch.step);
                        let acc = accidental_count(head.note.accidental());
                        let track = &mut tracks[ti];
                        let dia = i32::from(pitch.octave) * 7 + step as i32 - shift * 7;
                        let written_dia = i32::from(pitch.octave) * 7 + step as i32;
                        let grace = tokens[fi].contains('q');
                        let previous_grace =
                            !grace && track.events.get(&written_dia) == Some(&(acc, true));
                        track.events.insert(written_dia, (acc, grace));
                        let mut answer = AccidentalDisplay {
                            address,
                            source_address: None,
                            token: token.into(),
                            shown: false,
                            glyph: None,
                            cautionary: false,
                            enclosed: false,
                            enclosure: crate::document::DisplayEnclosure::None,
                            cause: DisplayCause::BelowAnalysisRange,
                            default_shown: false,
                            hidden_risk: None,
                            conventions: AccidentalConventions::default(),
                            accidental_layout: false,
                            bare_glyph: None,
                            grace_predecessor_same_accidental: previous_grace,
                        };
                        if dia >= 0 {
                            let di = checked_index(dia, ln)?;
                            let hidden = !token.contains("yy")
                                && ["ny", "#y", "-y"].iter().any(|m| token.contains(m));
                            let tie = token.contains('_') || token.contains(']');
                            let forced = token.find('X').is_some_and(|i| {
                                i > 0 && matches!(token.as_bytes()[i - 1], b'#' | b'-' | b'n')
                            });
                            if tie && acc != track.key[di % 7] && track.first {
                                let state = State {
                                    count: -1000 + acc,
                                    caution: Some(DisplayCause::TieAcrossBarlineCautionary),
                                    memory: true,
                                };
                                track.normal[di] = state;
                                track.grace[di] = state;
                            }
                            if tie && !forced {
                                answer.cause = DisplayCause::TieContinuation;
                            } else {
                                let b40 = i32::from(pitch.octave) * 40 + BASE40[step] + acc;
                                ornament_states(track, b40, token, ln)?;
                                let grace = tokens[fi].contains('q');
                                let previous = if grace {
                                    track.grace[di]
                                } else {
                                    track.normal[di]
                                };
                                if !tie
                                    && (acc != previous.count
                                        || (!grace && concurrent[di] != 0 && concurrent[di] == acc))
                                {
                                    answer.bare_glyph = AccidentalGlyph::from_count(acc);
                                }
                                let duplicate =
                                    !grace && concurrent[di] != 0 && concurrent[di] == acc;
                                if acc != previous.count || duplicate {
                                    answer.shown = !hidden;
                                    answer.cautionary = !hidden && previous.count < -900;
                                    answer.cause = if answer.cautionary {
                                        previous
                                            .caution
                                            .unwrap_or(DisplayCause::OrnamentStateCautionary)
                                    } else if duplicate {
                                        DisplayCause::ConcurrentDuplicate
                                    } else {
                                        DisplayCause::DiffersFromState
                                    };
                                    let state = State {
                                        count: acc,
                                        caution: None,
                                        memory: true,
                                    };
                                    track.grace[di] = state;
                                    if grace {
                                        track.normal[di] = State {
                                            count: -1000 + acc,
                                            caution: Some(DisplayCause::AfterGraceCautionary),
                                            memory: true,
                                        };
                                    } else {
                                        track.normal[di] = state;
                                        if !hidden {
                                            concurrent[di] = acc;
                                        }
                                    }
                                } else if acc == 0 && token.contains('n') && !hidden {
                                    answer.shown = true;
                                    answer.cautionary = true;
                                    answer.cause = DisplayCause::RedundantNatural;
                                } else if !token.contains("XX") && forced {
                                    answer.shown = true;
                                    answer.cautionary = true;
                                    answer.cause = DisplayCause::ForcedAccidental;
                                } else {
                                    answer.cause = if previous.memory {
                                        DisplayCause::SatisfiedByMeasureMemory
                                    } else {
                                        DisplayCause::SatisfiedByKey
                                    };
                                }
                                if hidden && !answer.shown {
                                    answer.cause = DisplayCause::HiddenAccidental;
                                }
                            }
                        }
                        visual.push(answer.shown);
                        let mut glyph = AccidentalGlyph::from_count(acc);
                        let layout = note_layout(
                            &fields[layer.index()].layouts,
                            (subtokens.len() > 1).then_some(si),
                            ln,
                        )?;
                        answer.accidental_layout =
                            has_accidental_layout(&fields[layer.index()].layouts);
                        let declared_editorial =
                            editorial.iter().find(|(sign, _)| token.contains(sign));
                        let edit = token.contains(['Z', 'j'])
                            || declared_editorial.is_some()
                            || layout.contains_key("A.edit");
                        let editorial_style = declared_editorial
                            .map(|(_, style)| style.as_str())
                            .or_else(|| layout.get("A.edit").map(String::as_str));
                        use crate::document::DisplayEnclosure;
                        answer.enclosure = if token.contains('j')
                            || editorial_style.is_some_and(|s| s.contains("brac"))
                            || (!edit && layout.contains_key("ACC.brack"))
                        {
                            DisplayEnclosure::Brack
                        } else if token.contains('Z')
                            || editorial_style.is_some_and(|s| s.contains("paren"))
                            || (!edit && layout.contains_key("ACC.paren"))
                        {
                            DisplayEnclosure::Paren
                        } else {
                            DisplayEnclosure::None
                        };
                        answer.enclosed = answer.enclosure != DisplayEnclosure::None;
                        // Cautionary flags create a glyph before the visual override branch.
                        if let Some(vis) = layout.get("A.vis").or_else(|| layout.get("N.acc")) {
                            glyph = Some(layout_glyph(vis, edit, ln)?);
                            answer.shown = true;
                            answer.cause = DisplayCause::LayoutOverride;
                            answer.bare_glyph = glyph;
                        }
                        if edit {
                            answer.shown = true;
                            answer.cautionary = false;
                            answer.cause = DisplayCause::EditorialAccidental;
                            answer.bare_glyph = glyph;
                        }
                        if !edit
                            && (layout.contains_key("ACC.brack")
                                || layout.contains_key("ACC.paren"))
                        {
                            answer.cautionary = true;
                        }
                        if unpitched {
                            answer.shown = false;
                            answer.cautionary = false;
                            answer.cause = DisplayCause::UnpitchedClef;
                            answer.bare_glyph = None;
                        }
                        if token.contains("yy") {
                            answer.shown = false;
                            answer.cause = DisplayCause::HiddenNote;
                            answer.bare_glyph = None;
                        }
                        answer.enclosed &= answer.shown;
                        answer.glyph = if answer.shown { glyph } else { None };
                        result.push(answer);
                    }
                    fields[layer.index()].beam.observe(
                        tracks[ti].tremolo,
                        content,
                        record_start..result.len(),
                        &mut result,
                        &visual,
                    );
                    fields[layer.index()].layouts.clear();
                }
                for layer in layers {
                    fields[layer.index()].layouts.clear();
                }
                for track in &mut tracks {
                    track.first = false;
                }
            }
        }
    }
    for note in &mut result {
        note.enclosed &= note.shown;
        if !note.shown {
            note.enclosure = crate::document::DisplayEnclosure::None;
        }
        if note.cause == DisplayCause::TremoloSuppression {
            note.bare_glyph = None;
        }
    }
    hidden::classify(document, score, &mut result, None);
    Ok(result)
}
fn retain_layouts(
    fields: &mut [Field],
    layers: &[LayerId],
    globals: &mut Vec<(usize, String)>,
    raw: &str,
    line: usize,
) -> Result<(), AccidentalDisplayError> {
    if raw.starts_with("!!LO:") && !raw.contains(":ignore") {
        globals.push((line, raw.into()));
    }
    if raw.starts_with('!') && !raw.starts_with("!!") && raw.contains("!LO:") {
        let tokens: Vec<_> = raw.split('\t').collect();
        width(line, layers.len(), tokens.len())?;
        for (layer, token) in layers.iter().zip(tokens) {
            if token.starts_with("!LO:") && !token.contains(":ignore") {
                fields[layer.index()].layouts.push((line, token.into()));
            }
        }
    }
    Ok(())
}

fn octaves(kind: OttavaKind) -> i32 {
    match kind {
        OttavaKind::OctaveUp => 1,
        OttavaKind::OctaveDown => -1,
        OttavaKind::TwoOctavesUp => 2,
        OttavaKind::TwoOctavesDown => -2,
    }
}
fn width(line: usize, expected: usize, actual: usize) -> Result<(), AccidentalDisplayError> {
    if expected != actual {
        Err(error(
            line,
            format!("expected {expected} active fields, found {actual}"),
        ))
    } else {
        Ok(())
    }
}
fn accidental_count(accidental: Option<&Accidental>) -> i32 {
    match accidental {
        None | Some(Accidental::Natural) => 0,
        Some(Accidental::Flat) => -1,
        Some(Accidental::DoubleFlat) => -2,
        Some(Accidental::Sharp) => 1,
        Some(Accidental::DoubleSharp) => 2,
    }
}
fn step_index(step: DiatonicStep) -> usize {
    match step {
        DiatonicStep::C => 0,
        DiatonicStep::D => 1,
        DiatonicStep::E => 2,
        DiatonicStep::F => 3,
        DiatonicStep::G => 4,
        DiatonicStep::A => 5,
        DiatonicStep::B => 6,
    }
}
const BASE40: [i32; 7] = [2, 8, 14, 19, 25, 31, 37];
fn checked_index(dia: i32, line: usize) -> Result<usize, AccidentalDisplayError> {
    if (0..70).contains(&dia) {
        Ok(dia as usize)
    } else {
        Err(error(
            line,
            format!("diatonic index {dia} outside Verovio's 70-step accidental array"),
        ))
    }
}
fn base40_parts(b40: i32) -> (i32, i32) {
    if b40 < 0 {
        return (-1, 0);
    }
    for (step, base) in BASE40.iter().enumerate() {
        let acc = b40 % 40 - base;
        if (-2..=2).contains(&acc) {
            return (b40 / 40 * 7 + step as i32, acc);
        }
    }
    (-1, 1000)
}
fn ornament_states(
    track: &mut Track,
    b40: i32,
    token: &str,
    line: usize,
) -> Result<(), AccidentalDisplayError> {
    let interval = if token.contains('t') {
        Some(5)
    } else if token.contains('T') || token.contains('M') {
        Some(6)
    } else if token.contains('m') {
        Some(5)
    } else if token.contains('W') {
        Some(-6)
    } else if token.contains('w') {
        Some(-5)
    } else {
        None
    };
    if let Some(interval) = interval {
        let (dia, acc) = base40_parts(b40 + interval);
        set_ornament_state(track, checked_index(dia, line)?, acc);
    } else if let Some((pos, inverted)) = token
        .find('$')
        .map(|p| (p, true))
        .or_else(|| token.find('S').map(|p| (p, false)))
    {
        let (dia, _) = base40_parts(b40);
        let lower = checked_index(dia - 1, line)?;
        let upper = checked_index(dia + 1, line)?;
        let chars = token.as_bytes();
        let interval = |offset| match chars.get(pos + offset) {
            Some(b's') => Some(5),
            Some(b'S') => Some(6),
            _ => None,
        };
        let up = interval(if inverted { 2 } else { 1 });
        let down = interval(if inverted { 1 } else { 2 });
        // Unspecified intervals use the current state, including humlib's -1000 sentinel.
        let upper40 = up.map_or(
            (upper / 7) as i32 * 40 + BASE40[upper % 7] + track.normal[upper].count,
            |v| b40 + v,
        );
        let lower40 = down.map_or(
            (lower / 7) as i32 * 40 + BASE40[lower % 7] + track.normal[lower].count,
            |v| b40 - v,
        );
        set_ornament_state(track, upper, base40_parts(upper40).1);
        set_ornament_state(track, lower, base40_parts(lower40).1);
    }
    Ok(())
}
fn set_ornament_state(track: &mut Track, dia: usize, acc: i32) {
    if track.normal[dia].count != acc {
        track.normal[dia] = State {
            count: -1000 + acc,
            caution: Some(DisplayCause::OrnamentStateCautionary),
            memory: true,
        };
    }
}
fn has_accidental_layout(layouts: &[(usize, String)]) -> bool {
    layouts.iter().any(|(_, layout)| {
        let parts: Vec<_> = layout.split(':').collect();
        match parts.get(1).copied() {
            Some("ACC") => true,
            Some("A") => parts[2..]
                .iter()
                .any(|p| p.starts_with("edit=") || p.starts_with("vis=")),
            Some("N") => parts[2..].iter().any(|p| p.starts_with("acc=")),
            _ => false,
        }
    })
}
fn note_layout(
    layouts: &[(usize, String)],
    subtoken: Option<usize>,
    line: usize,
) -> Result<BTreeMap<String, String>, AccidentalDisplayError> {
    let mut result = BTreeMap::new();
    for (category, key) in [
        ("A", "edit"),
        ("A", "vis"),
        ("N", "acc"),
        ("ACC", "brack"),
        ("ACC", "paren"),
    ] {
        let mut selector = None;
        for (_, layout) in layouts {
            let parts: Vec<_> = layout.split(':').collect();
            if parts.get(1).copied() != Some(category) {
                continue;
            }
            let mut output = None;
            for part in &parts[2..] {
                let (name, value) = part.split_once('=').unwrap_or((part, "true"));
                if name == key {
                    output = Some(value);
                }
                if name == "n" && category != "ACC" {
                    selector =
                        Some(value.parse::<usize>().map_err(|_| {
                            error(line, "invalid accidental layout subtoken selector")
                        })?);
                }
            }
            if category != "ACC" && subtoken.is_some() && selector == subtoken.map(|s| s + 1) {
                if let Some(value) = output {
                    result.insert(format!("{category}.{key}"), value.into());
                }
                // humlib returns even an empty value when n matches, before
                // inspecting later parameter sets for the requested key.
                break;
            }
            if let Some(value) = output {
                // hasLayoutParameter ignores selectors for ACC enclosure flags.
                // getLayoutParameter carries n forward across linked parameter
                // sets and ignores it for single notes. Both quirks are upstream.
                if category == "ACC" {
                    if !matches!(value, "false" | "0") {
                        result.insert(format!("{category}.{key}"), value.into());
                    }
                    break;
                }
                if subtoken.is_none() || selector.is_none() || selector == subtoken.map(|s| s + 1) {
                    result.insert(format!("{category}.{key}"), value.into());
                    break;
                }
            }
        }
    }
    Ok(result)
}
fn layout_glyph(
    value: &str,
    editorial: bool,
    line: usize,
) -> Result<AccidentalGlyph, AccidentalDisplayError> {
    Ok(match value {
        "#" => AccidentalGlyph::Sharp,
        "-" => AccidentalGlyph::Flat,
        "n" => AccidentalGlyph::Natural,
        "x" => AccidentalGlyph::DoubleSharp,
        "##" => AccidentalGlyph::TwoSharps,
        "--" => AccidentalGlyph::DoubleFlat,
        "n#" => AccidentalGlyph::NaturalSharp,
        "n-" => AccidentalGlyph::NaturalFlat,
        "###" | "sx" | "xs" | "#x" => AccidentalGlyph::TripleSharp,
        "---" if editorial => AccidentalGlyph::TripleSharp,
        "---" => AccidentalGlyph::TripleFlat,
        _ => {
            return Err(error(
                line,
                format!("unsupported visual accidental layout {value:?}"),
            ))
        }
    })
}

#[cfg(test)]
#[path = "../../tests/support/accidental_display_rules.rs"]
mod rules;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::InputOptions;
    macro_rules! rule_tests {
        ($($name:ident),* $(,)?) => { $(
            #[test]
            fn $name() {
                let fixture = super::rules::fixtures().into_iter().find(|f|f.name == stringify!($name)).unwrap();
                let document = Document::parse(&fixture.text,InputOptions::default()).unwrap();
                let notes = analyze_accidental_display(&document).unwrap();
                assert_eq!(notes.iter().map(|n|n.glyph).collect::<Vec<_>>(),fixture.glyphs);
            }
        )* };
    }
    #[test]
    fn rejects_merges_of_independent_tracks() {
        // humlib merges them into one track; the resolved score rejects them.
        let document = Document::parse(
            "**kern\t**kern\n=1\t=1\n4c#\t4c\n*v\t*v\n4d#\n*-\n",
            InputOptions::default(),
        )
        .unwrap();
        let error = analyze_accidental_display(&document).unwrap_err();
        assert!(error
            .message
            .contains("cannot merge spines from different original spines"));
    }
    rule_tests!(
        tremolo_control_in_full_rest_layer,
        appoggiatura_p_is_not_humlib_grace,
        unpitched_clef,
        unpitched_clef_return,
        layout_ignored,
        layout_global,
        layout_global_priority,
        layout_single_selector_ignored,
        layout_chord_selector,
        layout_sticky_selector,
        layout_does_not_cross_interpretation,
        layout_does_not_cross_null,
        below_range,
        tremolo_repeated_forced,
        tremolo_alternate_transfer,
        tremolo_internal,
        tremolo_chord_transfer_quirk,
        tremolo_unequal_duration,
        tremolo_internal_tie,
        measure_memory,
        key_signature,
        key_change_mid_bar,
        flat_key,
        invisible_barline,
        grace_main,
        grace_next_bar,
        grace_natural_next_bar,
        grace_repeated,
        grace_after_cross_bar_tie,
        ties_within_bar,
        ties_across_bar,
        tie_in_key,
        tie_not_first_slice,
        forced_tie,
        forced_and_redundant,
        hide_accidental,
        hide_flat,
        hidden_note_updates_state,
        octave_specific,
        ottava_state,
        ottava_down,
        two_octaves,
        two_octaves_down,
        trill_minor,
        trill_major,
        mordent_upper_major,
        mordent_upper_minor,
        mordent_lower_major,
        mordent_lower_minor,
        turn_regular,
        turn_inverted,
        turn_implicit,
        ornament_during_ottava,
        chord_subtokens,
        double_flat,
        same_line_duplicates,
        natural_duplicates_quirk,
        independent_tracks,
        editorial,
        layout_override,
        layout_editorial,
        cross_staff_source_track
    );
    #[test]
    fn causes_and_cautionary_flags() {
        let text = "**kern\n=1\n8qf#\n4f#\n4f#X\n4fny\n4fn\n*-\n";
        let document = Document::parse(text, InputOptions::default()).unwrap();
        let notes = analyze_accidental_display(&document).unwrap();
        assert_eq!(
            notes
                .iter()
                .map(|n| (n.cause, n.cautionary))
                .collect::<Vec<_>>(),
            vec![
                (DisplayCause::DiffersFromState, false),
                (DisplayCause::AfterGraceCautionary, true),
                (DisplayCause::ForcedAccidental, true),
                (DisplayCause::HiddenAccidental, false),
                (DisplayCause::RedundantNatural, true)
            ]
        );
    }
    #[test]
    fn below_range_skips_humlib_and_high_range_fails() {
        let document = Document::parse("**kern\n4CCCCC#\n*-\n", InputOptions::default()).unwrap();
        let note = analyze_accidental_display(&document).unwrap().remove(0);
        assert_eq!(
            (note.shown, note.cause),
            (false, DisplayCause::BelowAnalysisRange)
        );
        let high = Document::parse(
            "**kern\n*8ba\n4cccccc#\n*X8ba\n*-\n",
            InputOptions::default(),
        )
        .unwrap();
        assert!(analyze_accidental_display(&high)
            .unwrap_err()
            .message
            .contains("70-step"));
    }
    #[test]
    fn id_numbering() {
        assert_eq!(
            NoteAddress::from_verovio_id("note-L10F1"),
            Some(NoteAddress {
                line: 10,
                field: 1,
                subtoken: 1
            })
        );
        assert_eq!(
            NoteAddress::from_verovio_id("note-L10F1S2"),
            Some(NoteAddress {
                line: 10,
                field: 1,
                subtoken: 2
            })
        );
    }
}
