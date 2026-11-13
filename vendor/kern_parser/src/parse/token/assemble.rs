use std::ops::Range;

use super::{
    scan::{
        merge_arpeggios, positive, push_slur, scan, BeamCounts, ChordFault, Core, Fault, Missing,
        Reason, Scanned, Warning,
    },
    table::{is_grace_byte, EventKind, Marker, MarkerId},
    AtomSink, Part,
};
use crate::{
    parse::ParseDialect,
    types::{
        Chord, ChordModifiers, Fermata, GraceChord, GraceNote, InvisibleRestSlice, KernContent,
        MultiMeasureRest, Note, Pitch, Rest, RestPlacement,
    },
    InvalidNoteLocation, ParseError,
};

/// A refused token: the public error, and its typed reason with a span in the whole token.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Refusal {
    pub error: ParseError,
    pub fault: Fault,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Null,
    Slice,
    Grace,
    MultiRest,
    Rest,
    Chord { rest: bool },
}

impl Route {
    fn of(raw: &str, dialect: ParseDialect) -> Self {
        if raw == "." {
            return Self::Null;
        }
        if dialect == ParseDialect::TranscodaV2 && raw.starts_with("ryy") {
            return Self::Slice;
        }
        let mut rest = false;
        for byte in raw.bytes() {
            if is_grace_byte(byte) {
                return Self::Grace;
            }
            rest |= byte == b'r';
        }
        if is_multi_rest(raw) {
            Self::MultiRest
        } else if rest && !raw.contains(' ') {
            Self::Rest
        } else {
            Self::Chord { rest }
        }
    }
}

/// `MM`, digits, `r`. Any other `MM` prefix is a doubled mordent.
fn is_multi_rest(raw: &str) -> bool {
    raw.strip_prefix("MM")
        .and_then(|body| body.strip_suffix('r'))
        .is_some_and(|count| !count.is_empty() && count.bytes().all(|b| b.is_ascii_digit()))
}

pub(crate) fn content(raw: &str, dialect: ParseDialect) -> Result<KernContent, ParseError> {
    content_with(raw, dialect, &mut ()).map_err(|refusal| refusal.error)
}

pub(crate) fn refusal(raw: &str, dialect: ParseDialect) -> Option<Refusal> {
    content_with(raw, dialect, &mut ()).err()
}

/// The marks `content` keeps as atoms but leaves out of the AST, with spans in `raw`.
pub(crate) fn warnings(raw: &str, dialect: ParseDialect) -> Vec<(Range<usize>, Warning)> {
    let mut warnings = Warnings::default();
    let _ = content_with(raw, dialect, &mut warnings);
    warnings.found
}

#[derive(Default)]
struct Warnings {
    subtoken: usize,
    found: Vec<(Range<usize>, Warning)>,
}

impl AtomSink for Warnings {
    fn atom(&mut self, _: Range<usize>, _: MarkerId, _: &'static Marker) {}

    fn subtoken(&mut self, start: usize) {
        self.subtoken = start;
    }

    fn warn(&mut self, span: Range<usize>, warning: Warning) {
        self.found.push((
            span.start + self.subtoken..span.end + self.subtoken,
            warning,
        ));
    }
}

fn content_with<S: AtomSink>(
    raw: &str,
    dialect: ParseDialect,
    sink: &mut S,
) -> Result<KernContent, Refusal> {
    match Route::of(raw, dialect) {
        Route::Null => Ok(KernContent::Null),
        Route::Slice => slice(raw).map(KernContent::InvisibleRestSlice),
        Route::Grace => grace_chord(raw, sink).map(KernContent::Grace),
        Route::MultiRest => MultiMeasureRest::try_from(raw)
            .map(KernContent::MultiMeasureRest)
            .map_err(|error| Refusal {
                error,
                fault: Fault {
                    span: 0..raw.len(),
                    reason: Reason::MeasureCount,
                },
            }),
        Route::Rest => rest(raw, sink).map(KernContent::Rest),
        Route::Chord { rest: false } => {
            chord(raw, |_| EventKind::Note, sink).map(KernContent::Chord)
        }
        Route::Chord { rest: true } => chord(raw, sub_event, sink).map(KernContent::Chord),
    }
}

pub(super) fn head_event(sub: &str) -> EventKind {
    if sub.bytes().any(is_grace_byte) {
        EventKind::Grace
    } else {
        EventKind::Note
    }
}

pub(super) fn sub_event(sub: &str) -> EventKind {
    match head_event(sub) {
        EventKind::Note if sub.contains('r') => EventKind::Rest,
        event => event,
    }
}

impl TryFrom<&str> for Chord {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        chord(raw, head_event, &mut ()).map_err(|refusal| refusal.error)
    }
}

impl TryFrom<&str> for GraceChord {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        grace_chord(raw, &mut ()).map_err(|refusal| refusal.error)
    }
}

impl TryFrom<&str> for Note {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let scanned = scan_lone(raw, head_event(raw), Part::First, &mut ())?;
        if let Some(at) = scanned
            .grace
            .as_ref()
            .map(|(_, at)| *at)
            .or(scanned.first_token_mark)
        {
            return Err(head_error(raw, &Fault::at(at, Reason::Scope)));
        }
        Ok(Note {
            duration: scanned.duration.ok_or_else(|| unlocated(raw))?,
            pitch: head_pitch(scanned.core),
            accidental: scanned.accidental,
            modifiers: scanned.head,
        })
    }
}

impl TryFrom<&str> for GraceNote {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let scanned = scan_lone(raw, head_event(raw), Part::First, &mut ())?;
        let Some((kind, _)) = scanned.grace else {
            return Err(head_error(
                raw,
                &Fault::at(0, Reason::Missing(Missing::Grace)),
            ));
        };
        if let Some(at) = scanned.first_token_mark {
            return Err(head_error(raw, &Fault::at(at, Reason::Scope)));
        }
        Ok(GraceNote {
            kind,
            duration: scanned.duration,
            pitch: head_pitch(scanned.core),
            accidental: scanned.accidental,
            modifiers: scanned.head,
        })
    }
}

fn chord<S: AtomSink>(
    raw: &str,
    event_of: impl Fn(&str) -> EventKind,
    sink: &mut S,
) -> Result<Chord, Refusal> {
    check_shape(raw)?;
    let mut notes = Vec::new();
    let mut modifiers = ChordModifiers::default();
    for (index, (at, sub)) in subtokens(raw).enumerate() {
        let span = at..at + sub.len();
        let event = event_of(sub);
        if event == EventKind::Rest {
            return Err(chord_refusal(raw, span, ChordFault::RestHead));
        }
        let scanned = scan_head(sub, at, event, Part::new(index > 0), sink)?;
        if scanned.grace.is_some() {
            return Err(chord_refusal(raw, span, ChordFault::GraceHead));
        }
        merge_token(&mut modifiers, scanned.token)
            .ok_or_else(|| chord_refusal(raw, span.clone(), ChordFault::Merge))?;
        let first = notes.first().map(|note: &Note| &note.duration);
        let duration = match (scanned.duration, first) {
            (Some(duration), Some(first)) if duration != *first => {
                return Err(chord_refusal(raw, span, ChordFault::Duration))
            }
            (duration, first) => duration.or_else(|| first.cloned()).ok_or_else(|| Refusal {
                error: unlocated(sub),
                fault: Fault {
                    span,
                    reason: Reason::Missing(Missing::Duration),
                },
            })?,
        };
        notes.push(Note {
            duration,
            pitch: head_pitch(scanned.core),
            accidental: scanned.accidental,
            modifiers: scanned.head,
        });
    }
    Ok(Chord { notes, modifiers })
}

fn grace_chord<S: AtomSink>(raw: &str, sink: &mut S) -> Result<GraceChord, Refusal> {
    check_shape(raw)?;
    let mut notes: Vec<GraceNote> = Vec::new();
    let mut modifiers = ChordModifiers::default();
    for (index, (at, sub)) in subtokens(raw).enumerate() {
        let span = at..at + sub.len();
        let scanned = scan_head(sub, at, EventKind::Grace, Part::new(index > 0), sink)?;
        let first = notes.first();
        let Some(kind) = scanned
            .grace
            .map(|(kind, _)| kind)
            .or_else(|| first.map(|note| note.kind.clone()))
        else {
            return Err(chord_refusal(raw, span, ChordFault::MissingGrace));
        };
        let duration = scanned
            .duration
            .or_else(|| first.and_then(|note| note.duration.clone()));
        merge_token(&mut modifiers, scanned.token)
            .ok_or_else(|| chord_refusal(raw, span, ChordFault::Merge))?;
        notes.push(GraceNote {
            kind,
            duration,
            pitch: head_pitch(scanned.core),
            accidental: scanned.accidental,
            modifiers: scanned.head,
        });
    }
    let duration = &notes[0].duration;
    if let Some((_, (at, sub))) = notes
        .iter()
        .zip(subtokens(raw))
        .find(|(note, _)| note.duration != *duration)
    {
        return Err(chord_refusal(raw, at..at + sub.len(), ChordFault::Duration));
    }
    Ok(GraceChord { notes, modifiers })
}

fn subtokens(raw: &str) -> impl Iterator<Item = (usize, &str)> {
    raw.split(' ')
        .map(move |sub| (sub.as_ptr() as usize - raw.as_ptr() as usize, sub))
}

fn head_pitch(core: Core) -> Pitch {
    match core {
        Core::Pitch(pitch) => pitch,
        Core::Rest { .. } => unreachable!("rest rows are On::No for notes and grace notes"),
    }
}

fn check_shape(raw: &str) -> Result<(), Refusal> {
    if raw.split(' ').any(str::is_empty)
        || raw
            .chars()
            .any(|character| character.is_whitespace() && character != ' ')
    {
        return Err(chord_refusal(raw, 0..raw.len(), ChordFault::Spacing));
    }
    Ok(())
}

fn chord_refusal(raw: &str, span: Range<usize>, fault: ChordFault) -> Refusal {
    Refusal {
        error: ParseError::InvalidChord {
            raw: raw.to_owned(),
        },
        fault: Fault {
            span,
            reason: Reason::Chord(fault),
        },
    }
}

impl TryFrom<&str> for Rest {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        rest(raw, &mut ()).map_err(|refusal| refusal.error)
    }
}

fn rest<S: AtomSink>(raw: &str, sink: &mut S) -> Result<Rest, Refusal> {
    let Scanned {
        duration,
        core,
        head,
        token,
        ..
    } = scan_rest(raw, sink)?;
    let (Some(duration), Core::Rest { position }) = (duration, core) else {
        unreachable!("Fold::finish requires a duration and a rest core on rests")
    };
    Ok(Rest {
        duration,
        cross_staff: head.cross_staff,
        placement: position.map(|position| RestPlacement { position }),
        invisible: head.invisible,
        breath: token.breath.map(|breath| breath.mark),
        fermata: token.fermata,
        slurs: token.slurs,
        phrases: token.phrases,
    })
}

pub(super) fn scan_rest<S: AtomSink>(raw: &str, sink: &mut S) -> Result<Scanned, Refusal> {
    scan(raw, EventKind::Rest, Part::First, sink).map_err(|fault| Refusal {
        error: rest_error(raw),
        fault,
    })
}

impl TryFrom<&str> for MultiMeasureRest {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        raw.strip_prefix("MM")
            .and_then(|body| body.strip_suffix('r'))
            .filter(|count| !count.starts_with('0') && count.bytes().all(|b| b.is_ascii_digit()))
            .and_then(positive)
            .filter(|count| *count >= 2)
            .map(|measure_count| Self { measure_count })
            .ok_or_else(|| rest_error(raw))
    }
}

fn slice(raw: &str) -> Result<InvisibleRestSlice, Refusal> {
    let fermata = match &raw["ryy".len()..] {
        "" => None,
        ";" => Some(Fermata::default()),
        _ => {
            return Err(Refusal {
                error: rest_error(raw),
                fault: Fault {
                    span: "ryy".len()..raw.len(),
                    reason: Reason::Slice,
                },
            })
        }
    };
    Ok(InvisibleRestSlice { fermata })
}

fn scan_head<S: AtomSink>(
    sub: &str,
    at: usize,
    event: EventKind,
    part: Part,
    sink: &mut S,
) -> Result<Scanned, Refusal> {
    sink.subtoken(at);
    scan(sub, event, part, sink).map_err(|fault| Refusal {
        error: head_error(sub, &fault),
        fault: Fault {
            span: fault.span.start + at..fault.span.end + at,
            reason: fault.reason,
        },
    })
}

pub(crate) fn scan_lone<S: AtomSink>(
    sub: &str,
    event: EventKind,
    part: Part,
    sink: &mut S,
) -> Result<Scanned, ParseError> {
    precheck(sub)
        .and_then(|()| scan(sub, event, part, sink))
        .map_err(|fault| head_error(sub, &fault))
}

fn precheck(sub: &str) -> Result<(), Fault> {
    if sub.is_empty() {
        return Err(Fault::at(0, Reason::Malformed));
    }
    match sub.char_indices().find(|(_, c)| c.is_whitespace()) {
        Some((at, c)) => Err(Fault {
            span: at..at + c.len_utf8(),
            reason: Reason::Unknown,
        }),
        None => Ok(()),
    }
}

pub(crate) fn head_error(sub: &str, fault: &Fault) -> ParseError {
    if fault.reason == Reason::Malformed {
        return unlocated(sub);
    }
    let byte = fault.span.start;
    let index = sub[..byte].chars().count();
    let character = sub[byte..].chars().next();
    ParseError::InvalidNote {
        raw: sub.to_owned(),
        location: InvalidNoteLocation {
            position: Some(if character.is_some() {
                index + 1
            } else {
                index
            }),
            character,
        },
    }
}

fn unlocated(sub: &str) -> ParseError {
    ParseError::InvalidNote {
        raw: sub.to_owned(),
        location: InvalidNoteLocation {
            position: None,
            character: None,
        },
    }
}

fn rest_error(raw: &str) -> ParseError {
    ParseError::InvalidRest {
        raw: raw.to_owned(),
    }
}

fn merge_token(target: &mut ChordModifiers, source: ChordModifiers) -> Option<()> {
    target.arpeggio = merge_arpeggios(target.arpeggio, source.arpeggio);
    target.articulations.extend(source.articulations);
    let mut beams = BeamCounts::default();
    target.beams = beams
        .add_chord_beams(&target.beams)
        .and_then(|()| beams.add_chord_beams(&source.beams))
        .and_then(|()| beams.into_chord_beams())?;
    if target.stem_direction.is_none() {
        target.stem_direction = source.stem_direction;
    }
    target.breath = target.breath.or(source.breath);
    target.harmonic = target.harmonic.or(source.harmonic);
    // The same ornament on another head says nothing more about the chord, and
    // written back as one run (`TT`, `MM`) it would read as another ornament.
    let earlier_heads = target.ornaments.len();
    for ornament in source.ornaments {
        if !target.ornaments[..earlier_heads].contains(&ornament) {
            target.ornaments.push(ornament);
        }
    }
    target.fermata = target.fermata.or(source.fermata);
    for slur in source.slurs {
        push_slur(&mut target.slurs, slur)?;
    }
    target.phrases.extend(source.phrases);
    if target.bowing.is_none() {
        target.bowing = source.bowing;
    }
    Some(())
}
