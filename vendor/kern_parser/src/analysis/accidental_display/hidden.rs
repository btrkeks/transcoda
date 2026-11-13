use super::{accidental_count, step_index, AccidentalDisplay, AccidentalGlyph, NoteAddress};
use crate::analysis::diagnostics_index::{staff_position, StaffIdentity, StaffMap};
use crate::{document::Document, score::Score, CrossStaffPlacement, KernLine, SpineContent};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]

pub enum HiddenRiskClass {
    GraceCarry,
    OctaveCarry,
    OttavaCarry,
    TieCarry,
    Unexplained,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]

pub enum OctaveCarryVariant {
    SameStaff,
    CrossStaff,
    SameChord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]

pub enum TieCarryVariant {
    Simultaneous,
    LaterInBar,
}

/// Opportunities are independent and may overlap. A carry opportunity requires
/// a needed glyph; reminder opportunities require a glyph Verovio would omit.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]

pub struct AccidentalConventions {
    pub octave_carry: Option<OctaveCarryVariant>,
    pub grace_carry: bool,
    pub tie_carry: Option<TieCarryVariant>,
    pub courtesy_after_barline: bool,
    pub cross_staff_reminder: bool,
    pub restatement_after_tie: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct HiddenRisk {
    pub class: HiddenRiskClass,
    pub octave_variant: Option<OctaveCarryVariant>,
    pub tie_variant: Option<TieCarryVariant>,
    pub bar: usize,
    pub staff: Option<u32>,
    pub encoded_pitch: String,
    pub inferred_pitch: String,
}

pub(crate) fn active_hide(token: &str) -> bool {
    !token.contains("yy")
        && token
            .as_bytes()
            .windows(2)
            .any(|w| matches!(w[0], b'#' | b'-' | b'n') && w[1] == b'y')
}

struct Event {
    record: usize,
    staff: StaffIdentity,
    part: Option<u32>,
    voice: String,
    written: i32,
    sounding: i32,
    position: Option<i16>,
    alteration: i32,
    inferred: i32,
    grace: bool,
    shown: bool,
    invisible: bool,
    tied: bool,
    entered_tie: bool,
    visible: Option<i32>,
    understood: bool,
}
impl Event {
    fn step(&self) -> i32 {
        self.written.rem_euclid(7)
    }
    fn same_part(&self, other: &Self) -> bool {
        self.part == other.part
    }
}
fn pitch(diatonic: i32, alteration: i32) -> String {
    format!(
        "{}{}{}",
        ['C', 'D', 'E', 'F', 'G', 'A', 'B'][diatonic.rem_euclid(7) as usize],
        match alteration {
            -3 => "bbb",
            -2 => "bb",
            -1 => "b",
            1 => "#",
            2 => "##",
            3 => "###",
            _ => "",
        },
        diatonic.div_euclid(7)
    )
}
fn glyph_alteration(g: AccidentalGlyph) -> i32 {
    match g {
        AccidentalGlyph::Flat | AccidentalGlyph::NaturalFlat => -1,
        AccidentalGlyph::Natural => 0,
        AccidentalGlyph::Sharp | AccidentalGlyph::NaturalSharp => 1,
        AccidentalGlyph::DoubleSharp | AccidentalGlyph::TwoSharps => 2,
        AccidentalGlyph::DoubleFlat => -2,
        AccidentalGlyph::TripleSharp => 3,
        AccidentalGlyph::TripleFlat => -3,
    }
}

/// Reader memory follows displayed staves, visible glyphs and written octaves.
/// Convention evidence never changes the inferred pitch or sounding pitch.
pub(super) fn classify(
    document: &Document,
    resolved: &Score<'_>,
    records: &mut [AccidentalDisplay],
    source_parts: Option<&BTreeMap<NoteAddress, Option<u32>>>,
) {
    let score = document.score();
    let map = StaffMap::build(resolved);
    let lookup: BTreeMap<_, _> = records
        .iter()
        .enumerate()
        .map(|(i, r)| (r.address, i))
        .collect();
    let mut voices: BTreeMap<(String, i32), (i32, usize)> = BTreeMap::new();
    let mut last_voice_events: BTreeMap<String, Vec<(i32, bool, bool, i32)>> = BTreeMap::new();
    let mut memory = BTreeMap::new();
    let mut keys: BTreeMap<StaffIdentity, [i32; 7]> = BTreeMap::new();
    let mut previous_altered = BTreeSet::new();
    let mut altered = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut tied_across = BTreeSet::new();
    let mut tie_sources = BTreeMap::new();
    if let Ok(ties) = crate::analysis::analyze_tie_chains(resolved) {
        for chain in ties.chains {
            if chain.status == crate::analysis::TieChainStatus::Ambiguous {
                continue;
            }
            for pair in chain.members.windows(2) {
                let record_index = |member: &crate::analysis::TieChainMember| {
                    lookup
                        .get(&NoteAddress {
                            line: member.source_line,
                            field: member.field_index + 1,
                            subtoken: member.source_chord_note_index.unwrap_or(0) + 1,
                        })
                        .copied()
                };
                if let (Some(previous), Some(current)) =
                    (record_index(&pair[0]), record_index(&pair[1]))
                {
                    tie_sources.insert(current, previous);
                }
            }
        }
    }
    let mut past_events = BTreeMap::new();
    let mut tie_pitches = BTreeSet::new();
    let mut bar = 1;
    let mut cursor = 0;
    for span in crate::analysis::measure_spans(score) {
        let mut history: Vec<Event> = Vec::new();
        for li in cursor..span.content.end {
            if let KernLine::Barline(b) = &score.lines()[li] {
                previous_altered = std::mem::take(&mut altered);
                seen.clear();
                tied_across.clear();
                tie_pitches.clear();
                for (fi, col) in map.line(li).iter().enumerate() {
                    if !b.is_invisible_for_spine(fi) {
                        memory.retain(|(staff, _), _| *staff != col.staff);
                    }
                }
            }
            if let KernLine::TandemInterpretationLine(tandems) = &score.lines()[li] {
                for (fi, tandem) in tandems.tokens().iter().enumerate() {
                    if let crate::TandemInterpretation::KeySignature(_) = tandem {
                        let key = resolved.state_after(li, fi).track.key_signature;
                        if let Some(col) = map.get(li, fi) {
                            memory.retain(|(staff, _), _| *staff != col.staff);
                            history.retain(|e| e.staff != col.staff);
                            previous_altered.retain(|(staff, _)| *staff != col.staff);
                            altered.retain(|(staff, _)| *staff != col.staff);
                            let mut alterations = [0; 7];
                            for step in crate::analysis::diagnostics_index::STEPS {
                                alterations[step_index(step)] = accidental_count(
                                    key.and_then(|key| key.accidental_for_step(step)).as_ref(),
                                );
                            }
                            keys.insert(col.staff, alterations);
                        }
                    }
                }
            }
            let KernLine::Data(contents) = &score.lines()[li] else {
                continue;
            };
            let mut pending = Vec::new();
            for (fi, content) in contents.iter().enumerate() {
                let SpineContent::Kern(content) = content else {
                    continue;
                };
                let Some(col) = map.get(li, fi) else { continue };
                if content.noteheads().next().is_none()
                    && score
                        .source(li)
                        .and_then(|line| document.source().lines().nth(line - 1))
                        .and_then(|line| line.split('\t').nth(fi))
                        .is_some_and(|token| token.contains('r'))
                {
                    voices.retain(|(voice, _), _| *voice != col.voice);
                    last_voice_events.remove(&col.voice);
                }
                for head in content.noteheads() {
                    let address = NoteAddress {
                        line: score.source(li).unwrap_or(li + 1),
                        field: fi + 1,
                        subtoken: head.chord_note_index.unwrap_or(0) + 1,
                    };
                    let Some(&ri) = lookup.get(&address) else {
                        continue;
                    };
                    let record = &mut records[ri];
                    record.default_shown = record.bare_glyph.is_some();
                    record.hidden_risk = None;
                    let modifiers = head.note.modifiers();
                    let staff = match (col.staff, modifiers.cross_staff) {
                        (StaffIdentity::Known(n), Some(CrossStaffPlacement::Above)) if n > 1 => {
                            StaffIdentity::Known(n - 1)
                        }
                        (StaffIdentity::Known(n), Some(CrossStaffPlacement::Below)) => {
                            StaffIdentity::Known(n + 1)
                        }
                        _ => col.staff,
                    };
                    let p = head.note.pitch();
                    let step = step_index(p.step);
                    let acc = accidental_count(head.note.accidental());
                    let shift = i32::from(col.ottava.unwrap_or(0));
                    let sounding = i32::from(p.octave) * 7 + step as i32;
                    let written = sounding - shift * 7;
                    let displayed = map
                        .line(li)
                        .iter()
                        .find(|c| c.staff == staff)
                        .unwrap_or(col);
                    let mut wp = p;
                    wp.octave -= shift as i8;
                    let clef = displayed.clef.as_ref().filter(|c| {
                        map.line(li).iter().filter(|c| c.staff == staff).all(|o| {
                            o.clef.as_ref().is_some_and(|o| {
                                o.sign == c.sign
                                    && o.line == c.line
                                    && o.octave_shift == c.octave_shift
                            })
                        })
                    });
                    let inferred = *memory
                        .get(&(staff, written))
                        .unwrap_or(&keys.get(&staff).map_or(0, |k| k[step]));
                    let tied = record.token.contains(['_', ']']);
                    let inferred = if tied {
                        voices
                            .get(&(col.voice.clone(), sounding))
                            .map_or(inferred, |e| e.0)
                    } else {
                        inferred
                    };
                    let entered_tie = tied
                        && tie_sources
                            .get(&ri)
                            .and_then(|previous| past_events.get(previous))
                            .is_some_and(|&(old_staff, old_written, old_acc, old_bar)| {
                                old_staff == staff
                                    && old_written == written
                                    && old_acc == acc
                                    && old_bar < bar
                            });
                    let visible = record.glyph.map(glyph_alteration);
                    pending.push(Event {
                        record: ri,
                        staff,
                        part: source_parts
                            .and_then(|parts| parts.get(&address))
                            .copied()
                            .unwrap_or(col.part),
                        voice: col.voice.clone(),
                        written,
                        sounding,
                        position: clef.and_then(|c| staff_position(wp, c)),
                        alteration: acc,
                        inferred,
                        grace: head.note.grace_kind().is_some(),
                        shown: record.shown && visible == Some(acc),
                        invisible: record.token.contains("yy"),
                        tied,
                        entered_tie,
                        visible,
                        understood: !record.token.contains("yy")
                            && visible.map_or(inferred == acc, |a| a == acc),
                    });
                }
            }
            for e in &pending {
                if e.invisible {
                    continue;
                }
                let record = &mut records[e.record];
                let grace = !e.grace
                    && last_voice_events.get(&e.voice).is_some_and(|notes| {
                        notes.iter().any(|&(sounding, grace, shown, acc)| {
                            sounding == e.sounding && grace && shown && acc == e.alteration
                        })
                    });
                let matches = |other: &&Event| {
                    e.same_part(other)
                        && other.understood
                        && other.step() == e.step()
                        && other.alteration == e.alteration
                };
                let same_chord = pending
                    .iter()
                    .filter(|o| o.record != e.record)
                    .filter(matches)
                    .any(|o| o.written != e.written || o.staff != e.staff);
                let same_staff = history
                    .iter()
                    .filter(matches)
                    .any(|o| o.staff == e.staff && o.written != e.written);
                let cross_staff = history.iter().filter(matches).any(|o| o.staff != e.staff);
                let octave = if same_chord {
                    Some(OctaveCarryVariant::SameChord)
                } else if same_staff {
                    Some(OctaveCarryVariant::SameStaff)
                } else if cross_staff {
                    Some(OctaveCarryVariant::CrossStaff)
                } else {
                    None
                };
                let ottava = history.iter().any(|o| {
                    o.staff == e.staff
                        && o.position.is_some()
                        && o.position == e.position
                        && o.written == e.written
                        && o.sounding != e.sounding
                        && o.alteration == e.alteration
                        && o.understood
                });
                let simultaneous_tie = pending.iter().any(|o| {
                    o.entered_tie
                        && !o.invisible
                        && e.same_part(o)
                        && o.staff == e.staff
                        && o.written == e.written
                        && o.alteration == e.alteration
                });
                let tie = if simultaneous_tie {
                    Some(TieCarryVariant::Simultaneous)
                } else if tie_pitches.contains(&(e.part, e.staff, e.written, e.alteration)) {
                    Some(TieCarryVariant::LaterInBar)
                } else {
                    None
                };
                let needed = e.alteration != e.inferred || record.default_shown;
                record.conventions = AccidentalConventions {
                    octave_carry: if needed && !e.tied { octave } else { None },
                    grace_carry: needed && grace && !e.tied,
                    tie_carry: if needed && !e.tied { tie } else { None },
                    courtesy_after_barline: !record.default_shown
                        && !e.tied
                        && !seen.contains(&(e.staff, e.step()))
                        && previous_altered.contains(&(e.staff, e.step())),
                    cross_staff_reminder: !record.default_shown
                        && !e.tied
                        && history.iter().any(|o| {
                            e.same_part(o)
                                && o.staff != e.staff
                                && o.step() == e.step()
                                && o.understood
                                && o.alteration
                                    != keys.get(&o.staff).map_or(0, |k| k[o.step() as usize])
                        }),
                    restatement_after_tie: !e.tied
                        && tied_across.contains(&(e.voice.clone(), e.written, e.alteration)),
                };
                if !record.shown && active_hide(&record.token) && e.alteration != e.inferred {
                    record.hidden_risk = Some(HiddenRisk {
                        class: if grace {
                            HiddenRiskClass::GraceCarry
                        } else if tie.is_some() {
                            HiddenRiskClass::TieCarry
                        } else if ottava {
                            HiddenRiskClass::OttavaCarry
                        } else if octave.is_some() {
                            HiddenRiskClass::OctaveCarry
                        } else {
                            HiddenRiskClass::Unexplained
                        },
                        octave_variant: octave,
                        tie_variant: tie,
                        bar,
                        staff: e.staff.number(),
                        encoded_pitch: pitch(e.sounding, e.alteration),
                        inferred_pitch: pitch(e.sounding, e.inferred),
                    });
                }
            }
            let mut current_voice_events: BTreeMap<String, Vec<(i32, bool, bool, i32)>> =
                BTreeMap::new();
            for mut e in pending {
                past_events.insert(e.record, (e.staff, e.written, e.alteration, bar));
                if e.entered_tie && !e.invisible {
                    tie_pitches.insert((e.part, e.staff, e.written, e.alteration));
                }
                current_voice_events
                    .entry(e.voice.clone())
                    .or_default()
                    .push((e.sounding, e.grace, e.shown, e.alteration));
                let crossed_bar = voices
                    .get(&(e.voice.clone(), e.sounding))
                    .is_some_and(|v| v.1 < bar);
                if !e.invisible {
                    e.understood |= records[e.record]
                        .hidden_risk
                        .as_ref()
                        .is_some_and(|risk| risk.class != HiddenRiskClass::Unexplained);
                }
                voices.insert(
                    (e.voice.clone(), e.sounding),
                    (
                        if e.understood {
                            e.alteration
                        } else {
                            e.visible.unwrap_or(e.inferred)
                        },
                        bar,
                    ),
                );
                if e.invisible {
                    continue;
                }
                if e.tied
                    && e.understood
                    && crossed_bar
                    && e.alteration != keys.get(&e.staff).map_or(0, |k| k[e.step() as usize])
                {
                    tied_across.insert((e.voice.clone(), e.written, e.alteration));
                }
                if !e.tied {
                    tied_across.remove(&(e.voice.clone(), e.written, e.alteration));
                }
                seen.insert((e.staff, e.step()));
                if e.understood
                    && e.alteration != keys.get(&e.staff).map_or(0, |k| k[e.step() as usize])
                {
                    altered.insert((e.staff, e.step()));
                }
                if let Some(a) = e.visible {
                    memory.insert((e.staff, e.written), a);
                }
                history.push(e);
            }
            last_voice_events.extend(current_voice_events);
        }
        cursor = span.content.end;
        bar += span.bar_count(score) as usize;
    }
}
