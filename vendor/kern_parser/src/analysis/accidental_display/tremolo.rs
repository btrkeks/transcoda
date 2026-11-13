//! `iohumdrum.cpp::checkForTremolo` changes which source noteheads are drawn.
//! Humlib state still sees every note. Apply suppression after the whole beam
//! group, preserving its accidental transfer quirk (subtoken zero only).
use super::{base40_parts, AccidentalDisplay, AccidentalGlyph, DisplayCause};
use crate::{KernContent, MusicalDuration};
use std::ops::Range;

#[derive(Clone, Default)]
pub(super) struct BeamCollector {
    depth: i32,
    notes: Vec<BeamNote>,
}
#[derive(Clone)]
struct BeamNote {
    records: Range<usize>,
    duration: MusicalDuration,
    pitches: Vec<i32>,
    tied: bool,
}
impl BeamCollector {
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }
    pub(super) fn observe(
        &mut self,
        active: bool,
        content: &KernContent,
        records: Range<usize>,
        result: &mut [AccidentalDisplay],
        visual: &[bool],
    ) {
        let KernContent::Chord(chord) = content else {
            return;
        };
        let starts = i32::from(chord.modifiers.beams.beam_start_count());
        let ends = i32::from(chord.modifiers.beams.beam_end_count());
        if self.depth == 0 && starts > 0 && active {
            self.notes.clear();
            self.depth = starts;
        } else if self.depth > 0 {
            self.depth += starts;
        } else {
            return;
        }
        let pitches = chord
            .notes
            .iter()
            .map(|n| {
                let step = super::step_index(n.pitch.step);
                i32::from(n.pitch.octave) * 40
                    + super::BASE40[step]
                    + super::accidental_count(n.accidental.as_ref())
            })
            .collect();
        self.notes.push(BeamNote {
            records,
            duration: chord.notes[0].duration.as_rational_whole_notes(),
            pitches,
            tied: chord.notes.iter().any(|n| !n.modifiers.ties.is_empty()),
        });
        self.depth -= ends;
        if self.depth <= 0 {
            apply(&self.notes, result, visual);
            self.clear();
        }
    }
}
fn suppress(note: &BeamNote, result: &mut [AccidentalDisplay]) {
    for record in &mut result[note.records.clone()] {
        record.shown = false;
        record.glyph = None;
        record.cautionary = false;
        record.cause = DisplayCause::TremoloSuppression;
    }
}
fn apply(notes: &[BeamNote], result: &mut [AccidentalDisplay], visual: &[bool]) {
    if notes.len() < 2
        || notes.iter().any(|n| n.duration != notes[0].duration)
        || notes[1..notes.len() - 1].iter().any(|n| n.tied)
    {
        return;
    }
    let duration = notes[0].duration;
    let qdur = duration.numerator() as f64 / duration.denominator() as f64 * 4.0;
    if qdur <= 0.0 {
        return;
    }
    let ties_at_ends = notes[0].tied || notes.last().unwrap().tied;
    let mut nextsame = vec![true; notes.len()];
    let mut equal = !ties_at_ends;
    if !ties_at_ends {
        for i in 1..notes.len() {
            if notes[i].pitches != notes[i - 1].pitches {
                equal = false;
                nextsame[i - 1] = false;
            }
        }
    }
    if equal {
        let total = qdur * notes.len() as f64;
        let mut slashes = qdur.log2() as i32;
        let noteslash = total.log2() as i32;
        if noteslash < 0 {
            slashes -= noteslash;
        }
        if -slashes > 0 {
            for note in &notes[1..] {
                suppress(note, result);
            }
        }
        return;
    }
    let mut internal = true;
    for i in 1..nextsame.len() - 1 {
        if !nextsame[i] && (!nextsame[i - 1] || !nextsame[i + 1]) {
            internal = false;
            break;
        }
    }
    if nextsame.len() == 2 && !nextsame[0] && nextsame[1] {
        internal = false;
    }
    if internal {
        let mut groups = vec![0];
        for (i, same) in nextsame.iter().enumerate().take(notes.len() - 1) {
            if !same {
                groups.push(i + 1);
            }
        }
        groups.push(notes.len());
        if groups.windows(2).all(|g| (g[1] - g[0]).is_power_of_two()) {
            for group in groups.windows(2) {
                for note in &notes[group[0] + 1..group[1]] {
                    suppress(note, result);
                }
            }
            return;
        }
    }
    if notes.len() < 3
        || notes
            .iter()
            .enumerate()
            .skip(2)
            .any(|(i, n)| n.pitches != notes[i - 2].pitches)
        || (-qdur.log2()) as i32 <= 0
    {
        return;
    }
    let last = notes.last().unwrap();
    let transfer = visual[notes[1].records.start];
    for note in &notes[1..notes.len() - 1] {
        suppress(note, result);
    }
    if transfer {
        let record = &mut result[last.records.start];
        if !record.token.contains("yy")
            && !matches!(
                record.cause,
                DisplayCause::EditorialAccidental
                    | DisplayCause::LayoutOverride
                    | DisplayCause::UnpitchedClef
            )
        {
            record.shown = true;
            record.glyph = AccidentalGlyph::from_count(base40_parts(last.pitches[0]).1);
            record.cause = DisplayCause::TremoloAccidentalTransfer;
            record.bare_glyph = record.glyph;
        }
    }
}
