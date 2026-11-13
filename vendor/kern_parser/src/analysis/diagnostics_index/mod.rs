//! Structural pitch addresses and views derived without rhythm validation.
mod clefs;
pub use clefs::*;
mod pitch;
mod staff;
use crate::{
    document::{Document, Framing},
    score::{Score, StateError, StateErrors},
    KernContent, KernLine, SpineContent,
};
pub use pitch::ScientificPitch;
pub(crate) use pitch::{alteration, staff_position, STEPS};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub use staff::{StaffIdentity, StaffMap};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NoteId(pub String);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]

pub struct NoteAddress {
    pub line: usize,
    pub field: usize,
    pub chord_note: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct PitchDiagnostic {
    pub code: String,
    pub message: String,
    pub line: Option<usize>,
    pub field: Option<usize>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct NoteView {
    pub id: Option<NoteId>,
    pub note: usize,
    pub address: NoteAddress,
    pub voice: String,
    pub written: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sounding: Option<String>,
    pub pos: Option<i16>,
    pub accidental_source: String,
    pub ties: Vec<String>,
    pub grace: bool,
    pub shown_on: Option<u32>,
    pub clef: Option<String>,
    pub key: Option<BTreeMap<String, i8>>,
    pub ottava: Option<OttavaView>,
    pub display: Option<DisplayView>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct DisplayView {
    pub shown: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glyph: Option<super::accidental_display::AccidentalGlyph>,
    pub cautionary: bool,
    pub enclosure: crate::document::DisplayEnclosure,
    pub cause: super::accidental_display::DisplayCause,
    #[serde(rename = "override")]
    pub override_mode: DisplayOverride,
    pub default_shown: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_risk: Option<super::accidental_display::HiddenRisk>,
    pub conventions: super::accidental_display::AccidentalConventions,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]

pub enum DisplayOverride {
    None,
    Force,
    Hide,
}
impl DisplayOverride {
    pub(crate) fn from_token(token: &str) -> Self {
        if token
            .as_bytes()
            .windows(2)
            .any(|w| matches!(w[0], b'#' | b'-' | b'n') && w[1] == b'X')
        {
            Self::Force
        } else if token
            .as_bytes()
            .windows(3)
            .any(|w| matches!(w[0], b'#' | b'-' | b'n') && w[1] == b'y' && w[2] != b'y')
            || token.ends_with("#y")
            || token.ends_with("-y")
            || token.ends_with("ny")
        {
            Self::Hide
        } else {
            Self::None
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct OttavaView {
    pub encoding: String,
    pub octaves: i8,
    pub pass: Option<u32>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct RestView {
    pub field: usize,
    pub voice: String,
    pub hidden: bool,
    pub bars: Option<u32>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct EventView {
    pub event: usize,
    pub line: usize,
    pub beat: Option<String>,
    pub notes: Vec<NoteView>,
    pub hidden_count: usize,
    pub rests: Vec<RestView>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct PitchCell {
    pub cell: String,
    pub bar: usize,
    pub staff: Option<u32>,
    pub gradable: bool,
    pub hash: String,
    pub timing: String,
    pub tremolo: bool,
    pub events: Vec<EventView>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct BarView {
    pub bar: usize,
    pub system: usize,
    pub start_line: usize,
    pub end_line: usize,
}
#[derive(Debug, Clone, Serialize)]

pub struct PitchView {
    pub bars: Vec<BarView>,
    pub staves: Vec<u32>,
    pub cells: Vec<PitchCell>,
}
#[derive(Debug, Clone)]
pub(crate) struct IndexedNote {
    pub view: NoteView,
    pub hidden: bool,

    pub cell: usize,
}
#[derive(Debug, Clone)]
pub(crate) struct IndexedRest {
    pub address: NoteAddress,
    pub rest: usize,
    pub content: KernContent,
    pub cell: usize,
}
#[derive(Debug, Clone)]
pub struct ScoreIndex {
    pub view: PitchView,
    pub diagnostics: Vec<PitchDiagnostic>,
}
impl ScoreIndex {
    /// Index a document's resolved score. A document whose score does not
    /// resolve gets one `pitch.state` diagnostic per state error and no cells.
    pub fn build(document: &Document) -> Self {
        match Score::resolve(document.score(), Framing::Excerpt) {
            Ok(resolved) => Self::build_resolved(document, &resolved),
            Err(errors) => Self::unresolved(document, errors),
        }
    }

    fn unresolved(_document: &Document, errors: StateErrors) -> Self {
        Self {
            view: PitchView {
                bars: Vec::new(),
                staves: Vec::new(),
                cells: Vec::new(),
            },
            diagnostics: errors
                .0
                .iter()
                .map(|error| PitchDiagnostic {
                    code: "pitch.state".into(),
                    message: error.to_string(),
                    line: state_error_line(error),
                    field: None,
                })
                .collect(),
        }
    }

    fn build_resolved(document: &Document, resolved: &Score<'_>) -> Self {
        let score = document.score();
        let staff_map = StaffMap::build(resolved);
        let spans = super::measure_spans(score);
        let mut diagnostics = Vec::new();
        let durations = match super::score_line_durations(resolved) {
            Ok(values) => Some(
                values
                    .into_iter()
                    .map(|d| (d.line_index, d.duration))
                    .collect::<BTreeMap<_, _>>(),
            ),
            Err(error) => {
                diagnostics.push(PitchDiagnostic {
                    code: "pitch.timing".into(),
                    message: error.to_string(),
                    line: Some(error.line()),
                    field: error.column().map(|c| c - 1),
                });
                None
            }
        };
        let mut view = PitchView {
            bars: Vec::new(),
            staves: Vec::new(),
            cells: Vec::new(),
        };
        let mut notes = Vec::new();
        let mut rests = Vec::new();
        let mut reported = BTreeSet::new();
        let mut system = 1;
        let mut next_bar = 1;
        for (bar_index, span) in spans.iter().enumerate() {
            let bar = next_bar;
            next_bar += span.bar_count(score) as usize;
            if span.leading_barline.is_some_and(|i| matches!(&score.lines()[i], KernLine::Barline(b) if b.system_break_before.is_some())) && bar_index > 0 { system += 1; }
            view.bars.push(BarView {
                bar,
                system,
                start_line: score
                    .source(span.content.start)
                    .unwrap_or(span.content.start + 1),
                end_line: score
                    .source(span.content.end.saturating_sub(1))
                    .unwrap_or(span.content.end),
            });
            let mut cells: BTreeMap<StaffIdentity, usize> = BTreeMap::new();
            let mut beat = crate::MusicalDuration::new(1, 1);
            let mut hash_events: BTreeMap<usize, Vec<serde_json::Value>> = BTreeMap::new();
            for line_index in span.content.clone() {
                let KernLine::Data(contents) = &score.lines()[line_index] else {
                    continue;
                };
                let line = score.source(line_index).unwrap_or(line_index + 1);
                let columns = staff_map.line(line_index);
                let mut row_events: BTreeMap<usize, usize> = BTreeMap::new();
                let mut semantics: BTreeMap<usize, Vec<serde_json::Value>> = BTreeMap::new();
                for (field, content) in contents.iter().enumerate() {
                    let SpineContent::Kern(content) = content else {
                        continue;
                    };
                    if matches!(content, KernContent::Null) {
                        continue;
                    }
                    let column = &columns[field];
                    let cell_index = *cells.entry(column.staff).or_insert_with(|| {
                        let index = view.cells.len();
                        let staff = column.staff.number();
                        let cell = match column.staff {
                            StaffIdentity::Known(n) => format!("b{bar}s{n}"),
                            StaffIdentity::Unknown(n) => format!("b{bar}u{}", n.saturating_add(1)),
                        };
                        view.cells.push(PitchCell {
                            cell,
                            bar,
                            staff,
                            gradable: staff.is_some(),
                            hash: String::new(),
                            timing: if durations.is_some() {
                                "exact"
                            } else {
                                "line-order"
                            }
                            .into(),
                            tremolo: false,
                            events: Vec::new(),
                        });
                        index
                    });
                    let cell = &mut view.cells[cell_index];
                    cell.tremolo |= column.tremolo;
                    if column.staff.number().is_none()
                        && reported.insert(("pitch.staff", cell_index))
                    {
                        diagnostics.push(PitchDiagnostic {
                            code: "pitch.staff".into(),
                            message: "mixed or inconsistent staff declarations; cell is ungradable"
                                .into(),
                            line: Some(line),
                            field: Some(field),
                        });
                    }
                    let event_index = *row_events.entry(cell_index).or_insert_with(|| {
                        let index = cell.events.len();
                        cell.events.push(EventView {
                            event: index + 1,
                            line,
                            beat: durations.as_ref().map(|_| beat.to_string()),
                            notes: Vec::new(),
                            hidden_count: 0,
                            rests: Vec::new(),
                        });
                        index
                    });
                    let event = &mut cell.events[event_index];
                    let first_ordinal = event.notes.len() + event.hidden_count + 1;
                    for (ordinal, head) in (first_ordinal..).zip(content.noteheads()) {
                        let modifiers = head.note.modifiers();
                        let stored = ScientificPitch {
                            pitch: head.note.pitch(),
                            accidental: head.note.accidental().cloned(),
                        };
                        let grace = head.note.grace_kind().is_some();
                        let known_ottava = column.ottava;
                        let written =
                            known_ottava.and_then(|shift| stored.shifted(-i16::from(shift) * 7));
                        let sounding = if known_ottava.is_some_and(|s| s != 0) {
                            Some(stored.spn())
                        } else {
                            None
                        };
                        let shown_on = match modifiers.cross_staff {
                            Some(crate::CrossStaffPlacement::Above) => column
                                .staff
                                .number()
                                .and_then(|n| n.checked_sub(1))
                                .filter(|n| *n > 0),
                            Some(crate::CrossStaffPlacement::Below) => {
                                column.staff.number().map(|n| n + 1)
                            }
                            None => column.staff.number(),
                        };
                        let staff_clef = shown_on.map_or(StaffClef::Absent, |staff| {
                            staff_map.staff_clef(line_index, staff)
                        });
                        let pos = written.as_ref().and_then(|p| {
                            staff_clef.known().and_then(|c| c.staff_position(p.pitch))
                        });
                        let clef_token = columns
                            .iter()
                            .find(|c| {
                                c.spine_type == crate::SpineType::Kern
                                    && c.staff.number() == shown_on
                            })
                            .and_then(|c| c.clef.as_ref())
                            .map(crate::Clef::as_kern);
                        for (code, bad, message) in [
                            (
                                "pitch.clef",
                                pos.is_none(),
                                "clef unavailable or conflicted; staff position is unknown",
                            ),
                            (
                                "pitch.ottava",
                                written.is_none(),
                                "ottava unavailable or conflicted; written pitch is unknown",
                            ),
                        ] {
                            if bad && reported.insert((code, cell_index)) {
                                diagnostics.push(PitchDiagnostic {
                                    code: code.into(),
                                    message: message.into(),
                                    line: Some(line),
                                    field: Some(field),
                                });
                            }
                        }
                        let source = if let Some(key) = column.key {
                            let a = alteration(stored.accidental.as_ref());
                            let k = alteration(key.accidental_for_step(stored.pitch.step).as_ref());
                            if a == k && k != 0 {
                                "key"
                            } else if a != k {
                                "written"
                            } else {
                                "none"
                            }
                        } else if alteration(stored.accidental.as_ref()) == 0 {
                            "none"
                        } else {
                            "written"
                        };
                        let note_view = NoteView {
                            id: None,
                            display: None,
                            note: ordinal,
                            address: NoteAddress {
                                line,
                                field,
                                chord_note: head.chord_note_index.unwrap_or(0),
                            },
                            voice: column.voice.clone(),
                            written: written.map(|p| p.spn()),
                            sounding,
                            pos,
                            accidental_source: source.into(),
                            ties: modifiers.ties.iter().map(|t| format!("{t:?}")).collect(),
                            grace,
                            shown_on,
                            clef: clef_token,
                            key: column.key.map(|k| {
                                STEPS
                                    .iter()
                                    .map(|s| {
                                        (
                                            pitch::letter(*s).to_string(),
                                            alteration(k.accidental_for_step(*s).as_ref()),
                                        )
                                    })
                                    .collect()
                            }),
                            ottava: known_ottava.filter(|s| *s != 0).map(|shift| OttavaView {
                                encoding: "tandem".into(),
                                octaves: shift,
                                pass: None,
                            }),
                        };
                        semantics
                            .entry(cell_index)
                            .or_default()
                            .push(serde_json::json!([
                                note_view.written.as_ref().unwrap_or(&stored.spn()),
                                grace,
                                modifiers.invisible
                            ]));
                        if modifiers.invisible {
                            event.hidden_count += 1;
                        } else {
                            event.notes.push(note_view.clone());
                        }
                        notes.push(IndexedNote {
                            view: note_view,
                            hidden: modifiers.invisible,

                            cell: cell_index,
                        });
                    }
                    let rest = match content {
                        KernContent::Rest(r) => Some((r.invisible, None)),
                        KernContent::MultiMeasureRest(r) => Some((false, Some(r.measure_count))),
                        KernContent::InvisibleRestSlice(_) => Some((true, None)),
                        _ => None,
                    };
                    if let Some((hidden, bars)) = rest {
                        rests.push(IndexedRest {
                            address: NoteAddress {
                                line,
                                field,
                                chord_note: 0,
                            },
                            rest: event.rests.len() + 1,
                            content: content.clone(),
                            cell: cell_index,
                        });
                        event.rests.push(RestView {
                            field,
                            voice: column.voice.clone(),
                            hidden,
                            bars,
                        });
                        semantics
                            .entry(cell_index)
                            .or_default()
                            .push(serde_json::json!(["rest", hidden, bars]));
                    }
                }
                for (cell, semantic) in semantics {
                    hash_events
                        .entry(cell)
                        .or_default()
                        .push(serde_json::json!(semantic));
                }
                if let Some(duration) = durations.as_ref().and_then(|d| d.get(&line_index)) {
                    beat += *duration * 4;
                }
            }
            for (cell, events) in hash_events {
                view.cells[cell].hash =
                    format!("{:x}", Sha256::digest(serde_json::to_vec(&events).unwrap()))[..6]
                        .to_owned();
            }
        }
        let mut ids = BTreeMap::new();
        let mut addresses = BTreeMap::new();
        for note in &mut notes {
            let cell = &view.cells[note.cell];
            if let Some(staff) = cell.staff {
                let event = cell
                    .events
                    .iter()
                    .find(|e| e.line == note.view.address.line)
                    .unwrap()
                    .event;
                let id = NoteId(format!(
                    "b{}s{}e{}n{}@{}",
                    cell.bar, staff, event, note.view.note, cell.hash
                ));
                ids.insert(id.clone(), note.view.address);
                addresses.insert(note.view.address, id.clone());
                note.view.id = Some(id);
            }
        }
        for rest in &rests {
            let cell = &view.cells[rest.cell];
            // Preserve the legacy hash input, including its existing rest entries.
            if let (Some(staff), KernContent::Rest(_)) = (cell.staff, &rest.content) {
                let event = cell
                    .events
                    .iter()
                    .find(|e| e.line == rest.address.line)
                    .unwrap()
                    .event;
                let id = NoteId(format!(
                    "b{}s{}e{}r{}@{}",
                    cell.bar, staff, event, rest.rest, cell.hash
                ));
                ids.insert(id.clone(), rest.address);
                addresses.insert(rest.address, id);
            }
        }
        for cell in &mut view.cells {
            for event in &mut cell.events {
                for note in &mut event.notes {
                    note.id = addresses.get(&note.address).cloned();
                }
            }
        }
        view.staves = view
            .cells
            .iter()
            .filter_map(|c| c.staff)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        match super::accidental_display::analyze_source_accidental_display(document) {
            Ok(records) => {
                let records: BTreeMap<_, _> = records
                    .into_iter()
                    .filter_map(|record| {
                        record.source_address.map(|a| {
                            (
                                NoteAddress {
                                    line: a.line,
                                    field: a.field - 1,
                                    chord_note: a.subtoken - 1,
                                },
                                record,
                            )
                        })
                    })
                    .collect();
                let source_ranges =
                    super::accidental_display::canonical::source_ranges(document.source());
                for note in &mut notes {
                    if let Some(record) = records.get(&note.view.address) {
                        let address = super::accidental_display::NoteAddress {
                            line: note.view.address.line,
                            field: note.view.address.field + 1,
                            subtoken: note.view.address.chord_note + 1,
                        };
                        let source_token = source_ranges
                            .get(&address)
                            .map(|r| &document.source()[r.clone()])
                            .unwrap_or("");
                        note.view.display = Some(DisplayView {
                            shown: record.shown,
                            glyph: record.glyph,
                            cautionary: record.cautionary,
                            enclosure: record.enclosure,
                            cause: record.cause,
                            override_mode: DisplayOverride::from_token(source_token),
                            default_shown: record.default_shown,
                            hidden_risk: record.hidden_risk.clone(),
                            conventions: record.conventions.clone(),
                        });
                    } else if !note.hidden {
                        diagnostics.push(PitchDiagnostic {
                            code: "pitch.display-unavailable".into(),
                            message: "no render provenance for notehead".into(),
                            line: Some(note.view.address.line),
                            field: Some(note.view.address.field),
                        });
                    }
                }
                let displays: BTreeMap<_, _> = notes
                    .iter()
                    .map(|n| (n.view.address, n.view.display.clone()))
                    .collect();
                for cell in &mut view.cells {
                    for event in &mut cell.events {
                        for note in &mut event.notes {
                            note.display = displays.get(&note.address).cloned().flatten();
                        }
                    }
                }
            }
            Err(error) => diagnostics.push(PitchDiagnostic {
                code: "pitch.display-unavailable".into(),
                message: error.to_string(),
                line: error.line(),
                field: error.field(),
            }),
        }
        Self { view, diagnostics }
    }
}

impl ScoreIndex {}

fn state_error_line(error: &StateError) -> Option<usize> {
    Some(match error {
        StateError::Topology(error) => error.line(),
        StateError::Conflicting { line, .. }
        | StateError::UnmatchedClose { line, .. }
        | StateError::SpanMismatch { line, .. }
        | StateError::CueRenumbered { line, .. }
        | StateError::GraceUnderRhythmScale { line, .. }
        | StateError::UndrawableDuration { line, .. }
        | StateError::OpenAtEnd { line, .. }
        | StateError::InvalidVisualDuration { line, .. }
        | StateError::StrayNatural { line, .. }
        | StateError::KeyCancellation { line, .. } => *line,
    })
}
