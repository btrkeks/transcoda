//! Verovio's key signature cancellation, simulated.
//!
//! Verovio reads `*kcancel`/`*Xkcancel` from every `**kern` spine into one
//! flag for all staves (`m_show_cautionary_keysig`), and encoders write it
//! once for the whole score, so a `*kcancel` in any `**kern` spine asks for
//! cancellation naturals on every staff from the next line on. Verovio builds
//! a key change that opens a measure before it reads that measure's tokens,
//! so such a change sees the flag as the previous measure left it. It draws
//! automatic cancellation naturals only for a change every staff makes to the
//! same key at the start of a measure (a change to `*k[]` draws them without
//! `*kcancel`), and none when the new or the previous key spells naturals of
//! its own. A key signature with naturals in the middle of a measure is
//! dropped. The resolver refuses a document where what Verovio draws differs
//! from what the written flag asks for, as it does for `*cue`.

use std::{collections::BTreeMap, fmt};

use super::{StateError, TrackId};
use crate::KeySignature;

/// One key signature token in a `**kern` track, as the resolver applied it.
#[derive(Debug, Clone, Copy)]
pub(super) struct KeyChange {
    /// 1-based source line and column.
    pub line: usize,
    pub column: usize,
    pub track: TrackId,
    pub key: KeySignature,
}

#[derive(Debug, Clone, Copy)]
struct Written {
    line: usize,
    column: usize,
    active: bool,
}

#[derive(Debug, Clone, Copy)]
struct Pending {
    change: KeyChange,
    /// The key it replaces.
    previous: Option<KeySignature>,
    /// The written flag at the change.
    requested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placement {
    MidMeasure,
    /// At the start of a measure, where not every staff changes to one key:
    /// Verovio writes one `<staffDef>` per staff.
    PerStaff,
    /// At the start of a measure, every staff to one key: one `<scoreDef>`.
    System,
}

#[derive(Debug, Default)]
pub(super) struct CancellationSimulation {
    /// Verovio's flag as of the last barline.
    flag: bool,
    /// Per track, the last `*kcancel`/`*Xkcancel` since that barline.
    written: BTreeMap<TrackId, Written>,
    /// The last of those in line order.
    last_written: Option<bool>,
    seen_data: bool,
    data_in_measure: bool,
    /// Key changes after the barline and before the measure's first data
    /// line.
    pending: Vec<Pending>,
    /// The last key signature token per track, naturals included.
    drawn: BTreeMap<TrackId, KeySignature>,
}

impl CancellationSimulation {
    /// A `*kcancel` or `*Xkcancel` token. Returns whether it moves Verovio's
    /// flag, so that restating the spine's own value is not redundant.
    pub fn write(&mut self, track: TrackId, line: usize, column: usize, active: bool) -> bool {
        let moves = self.written_flag() != active;
        self.written.insert(
            track,
            Written {
                line,
                column,
                active,
            },
        );
        self.last_written = Some(active);
        moves
    }

    /// The flag as the file writes it, in line order.
    fn written_flag(&self) -> bool {
        self.last_written.unwrap_or(self.flag)
    }

    pub fn key_change(&mut self, change: KeyChange, errors: &mut Vec<StateError>) {
        let pending = Pending {
            change,
            previous: self.drawn.insert(change.track, change.key),
            requested: self.written_flag(),
        };
        // The opening staff definition draws no cancellation.
        if !self.seen_data {
            return;
        }
        if self.data_in_measure {
            self.check(pending, Placement::MidMeasure, errors);
        } else {
            self.pending.push(pending);
        }
    }

    /// A data line. `kern_tracks` are the `**kern` tracks active on it.
    pub fn data_line(&mut self, kern_tracks: &[TrackId], errors: &mut Vec<StateError>) {
        self.flush(kern_tracks, errors);
        self.seen_data = true;
        self.data_in_measure = true;
    }

    pub fn barline(&mut self, kern_tracks: &[TrackId], errors: &mut Vec<StateError>) {
        self.flush(kern_tracks, errors);
        let mut written = std::mem::take(&mut self.written).into_values();
        if let Some(first) = written.next() {
            match written.find(|other| other.active != first.active) {
                // Verovio reads staff by staff, so which value it ends the
                // measure with depends on staff order.
                Some(other) => {
                    let later = if (other.line, other.column) > (first.line, first.column) {
                        other
                    } else {
                        first
                    };
                    errors.push(StateError::KeyCancellation {
                        line: later.line,
                        column: later.column,
                        issue: CancellationIssue::ConflictingFlags,
                    });
                }
                None => self.flag = first.active,
            }
        }
        self.last_written = None;
        self.data_in_measure = false;
    }

    pub fn finish(&mut self, kern_tracks: &[TrackId], errors: &mut Vec<StateError>) {
        self.flush(kern_tracks, errors);
    }

    /// Check the key changes that open a measure, once its first data line
    /// (or its end) shows which staves change.
    fn flush(&mut self, kern_tracks: &[TrackId], errors: &mut Vec<StateError>) {
        let pending = std::mem::take(&mut self.pending);
        // Verovio compares the last key token of each staff.
        let mut last = BTreeMap::new();
        for entry in &pending {
            last.insert(entry.change.track, entry.change.key);
        }
        let first = last.values().next().copied();
        let system = kern_tracks.iter().all(|track| last.contains_key(track))
            && last.values().all(|key| Some(*key) == first);
        let placement = if system {
            Placement::System
        } else {
            Placement::PerStaff
        };
        for entry in pending {
            self.check(entry, placement, errors);
        }
    }

    fn check(&self, pending: Pending, placement: Placement, errors: &mut Vec<StateError>) {
        let Pending {
            change,
            previous,
            requested,
        } = pending;
        let Some(previous) = previous else {
            return;
        };
        let issue = |issue| StateError::KeyCancellation {
            line: change.line,
            column: change.column,
            issue,
        };
        if change.key.has_naturals() {
            if placement == Placement::MidMeasure {
                errors.push(issue(CancellationIssue::NaturalsMidMeasure));
            }
            return;
        }
        let dropped = change.key.dropped_from(previous);
        if dropped.is_empty() {
            return;
        }
        let to_no_accidentals = change.key.fifths().get() == 0;
        let not_drawn = match placement {
            Placement::MidMeasure => Some(NotDrawn::MidMeasure),
            Placement::PerStaff => Some(NotDrawn::PerStaff),
            Placement::System if previous.has_naturals() => Some(NotDrawn::AfterNaturals),
            Placement::System if !to_no_accidentals && !self.flag => Some(NotDrawn::FlagOff),
            Placement::System => None,
        };
        match not_drawn {
            Some(reason) if requested => {
                errors.push(issue(CancellationIssue::NotDrawn {
                    reason,
                    spelled: change
                        .key
                        .with_naturals(&dropped)
                        .expect("dropped steps are not in the new key"),
                }));
            }
            None if !requested && !to_no_accidentals => {
                errors.push(issue(CancellationIssue::Unrequested));
            }
            _ => {}
        }
    }
}

/// Where Verovio's drawing of a key change differs from what the file says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellationIssue {
    /// `*kcancel` asks for cancellation naturals Verovio leaves out.
    /// `spelled` writes them out.
    NotDrawn {
        reason: NotDrawn,
        spelled: KeySignature,
    },
    /// Verovio draws cancellation naturals after an `*Xkcancel` in the same
    /// measure.
    Unrequested,
    /// A key signature with naturals in the middle of a measure, which
    /// Verovio drops.
    NaturalsMidMeasure,
    /// `**kern` spines leave different `*kcancel`/`*Xkcancel` values at one
    /// barline.
    ConflictingFlags,
}

/// Why Verovio draws no automatic cancellation naturals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotDrawn {
    MidMeasure,
    PerStaff,
    AfterNaturals,
    /// Verovio's flag was off at the barline that opens the measure.
    FlagOff,
}

impl fmt::Display for CancellationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotDrawn { reason, spelled } => {
                let spelled = spelled.as_kern();
                write!(
                    f,
                    "*kcancel is in force, but Verovio draws no cancellation naturals for this key change "
                )?;
                match reason {
                    NotDrawn::MidMeasure => write!(
                        f,
                        "in the middle of a measure; move the key change to the start of a measure"
                    ),
                    NotDrawn::PerStaff => write!(
                        f,
                        "because not every staff changes to this key here; write them out as `{spelled}`"
                    ),
                    NotDrawn::AfterNaturals => write!(
                        f,
                        "after a key signature that spells its own naturals; write them out as `{spelled}`"
                    ),
                    NotDrawn::FlagOff => write!(
                        f,
                        "because it reads *kcancel at the barline that opens the measure; write *kcancel before that barline, or write the naturals out as `{spelled}`"
                    ),
                }
            }
            Self::Unrequested => write!(
                f,
                "*Xkcancel is in force, but Verovio draws cancellation naturals for this key change because it reads *Xkcancel at the barline that opens the measure; write *Xkcancel before that barline"
            ),
            Self::NaturalsMidMeasure => write!(
                f,
                "Verovio drops a key signature with cancellation naturals in the middle of a measure; move the key change to the start of a measure"
            ),
            Self::ConflictingFlags => write!(
                f,
                "**kern spines leave both *kcancel and *Xkcancel in force at the end of this measure; Verovio keeps one flag for all staves, and which value it takes depends on staff order. Write the same one in every **kern spine"
            ),
        }
    }
}

#[cfg(test)]
mod tests;
