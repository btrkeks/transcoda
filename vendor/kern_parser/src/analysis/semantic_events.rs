//! Topology-stable sounding-event projection for normalization checks.
//!
//! This intentionally models written pitched events, not performed playback:
//! repeats are not expanded, tie continuations remain events, and rests only
//! advance time. Physical columns and transient split branches are excluded
//! because measure-slot canonicalization is allowed to rewrite them.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::{
    measure::MusicalDuration,
    score::{NoteSize, Score},
    types::Accidental,
    KernContent, KernLine, Pitch, SpineContent, SpineType,
};

use super::line_durations::{score_line_durations, DataLineDurationError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SemanticAccidental {
    DoubleFlat,
    Flat,
    Natural,
    Sharp,
    DoubleSharp,
}

impl From<&Accidental> for SemanticAccidental {
    fn from(accidental: &Accidental) -> Self {
        match accidental {
            Accidental::DoubleFlat => Self::DoubleFlat,
            Accidental::Flat => Self::Flat,
            Accidental::Natural => Self::Natural,
            Accidental::Sharp => Self::Sharp,
            Accidental::DoubleSharp => Self::DoubleSharp,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SemanticPitch {
    pub pitch: Pitch,
    pub accidental: Option<SemanticAccidental>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SemanticEvent {
    pub onset: MusicalDuration,
    pub duration: MusicalDuration,
    pub pitch: SemanticPitch,
    pub grace: bool,
    /// Drawn at cue size (`*cue`).
    pub cue: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SemanticEventError {
    #[error(transparent)]
    Duration(#[from] DataLineDurationError),
}

/// Return the sorted multiset of written sounding notehead events.
///
/// Multiset multiplicity preserves simultaneous unisons without assigning
/// unstable identities to voice branches that canonicalization may reshape.
pub fn semantic_events(score: &Score<'_>) -> Result<Vec<SemanticEvent>, SemanticEventError> {
    let document = score.document();
    let topology = score.topology();
    let duration_by_line = score_line_durations(score)?
        .into_iter()
        .map(|duration| (duration.line_index, duration.duration))
        .collect::<BTreeMap<_, _>>();
    let mut onset = MusicalDuration::zero();
    let mut events = Vec::new();

    for (line_index, line) in document.lines().iter().enumerate() {
        let KernLine::Data(contents) = line else {
            continue;
        };
        for (column, (&layer, content)) in topology
            .layers_at(line_index)
            .iter()
            .zip(contents)
            .enumerate()
        {
            if *topology.spine_type(layer) == SpineType::Kern {
                let cue = score.drawn().size(line_index, column) == NoteSize::Cue;
                collect_events(&mut events, content, onset, cue);
            }
        }
        onset += duration_by_line
            .get(&line_index)
            .copied()
            .unwrap_or_else(MusicalDuration::zero);
    }

    events.sort_unstable();
    Ok(events)
}

fn collect_events(
    events: &mut Vec<SemanticEvent>,
    content: &SpineContent,
    onset: MusicalDuration,
    cue: bool,
) {
    let SpineContent::Kern(content) = content else {
        return;
    };
    match content {
        KernContent::Note(note) => events.push(regular_event(
            note.pitch,
            note.accidental.as_ref(),
            note.duration.as_rational_whole_notes(),
            onset,
            cue,
        )),
        KernContent::Chord(chord) => {
            events.extend(chord.notes.iter().map(|note| {
                regular_event(
                    note.pitch,
                    note.accidental.as_ref(),
                    note.duration.as_rational_whole_notes(),
                    onset,
                    cue,
                )
            }));
        }
        KernContent::Grace(grace) => {
            events.extend(grace.notes.iter().map(|note| SemanticEvent {
                onset,
                duration: MusicalDuration::zero(),
                pitch: semantic_pitch(note.pitch, note.accidental.as_ref()),
                grace: true,
                cue,
            }));
        }
        KernContent::Null
        | KernContent::Rest(_)
        | KernContent::InvisibleRestSlice(_)
        | KernContent::MultiMeasureRest(_) => {}
    }
}

fn regular_event(
    pitch: Pitch,
    accidental: Option<&Accidental>,
    duration: MusicalDuration,
    onset: MusicalDuration,
    cue: bool,
) -> SemanticEvent {
    SemanticEvent {
        onset,
        duration,
        pitch: semantic_pitch(pitch, accidental),
        grace: false,
        cue,
    }
}

fn semantic_pitch(pitch: Pitch, accidental: Option<&Accidental>) -> SemanticPitch {
    SemanticPitch {
        pitch,
        accidental: accidental.map(SemanticAccidental::from),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        document::Framing,
        normalization::{omr::normalized_for_omr_with_options, NormalizeOptions},
        types::DiatonicStep,
        KernDocument,
    };

    fn events(input: &str) -> Vec<SemanticEvent> {
        events_of(&KernDocument::parse(input).expect("fixture should parse"))
    }

    fn events_of(document: &KernDocument) -> Vec<SemanticEvent> {
        let score = Score::resolve(document, Framing::Excerpt).expect("fixture should resolve");
        semantic_events(&score).expect("fixture should have valid durations")
    }

    #[test]
    fn pitch_spelling_is_absolute_under_key_signatures() {
        let projected = events("**kern\n*k[f#]\n4f\n4f#\n*-");
        assert_eq!(projected.len(), 2);
        assert_eq!(projected[0].pitch.pitch.step, DiatonicStep::F);
        assert_eq!(projected[0].pitch.accidental, None);
        assert_eq!(
            projected[1].pitch.accidental,
            Some(SemanticAccidental::Sharp)
        );
    }

    #[test]
    fn null_carry_chords_and_rational_durations_use_contextual_onsets() {
        let projected = events("**kern\t**kern\n4c 4e\t2g\n3%5d\t.\n*-\t*-");
        assert_eq!(projected.len(), 4);
        let d = projected
            .iter()
            .find(|event| event.pitch.pitch.step == DiatonicStep::D)
            .expect("D event should exist");
        assert_eq!(d.onset.as_fraction(), (1, 4));
        assert_eq!(d.duration.as_fraction(), (5, 3));
        let g = projected
            .iter()
            .find(|event| event.pitch.pitch.step == DiatonicStep::G)
            .expect("G event should exist");
        assert_eq!(g.onset.as_fraction(), (0, 1));
        assert_eq!(g.duration.as_fraction(), (1, 2));
    }

    #[test]
    fn grace_events_are_zero_duration_at_the_next_ordinary_onset() {
        let projected = events("**kern\n8qcc\n4c\n*-");
        assert_eq!(projected.len(), 2);
        assert!(projected[0].grace);
        assert_eq!(projected[0].duration.as_fraction(), (0, 1));
        assert_eq!(projected[0].onset, projected[1].onset);
        assert!(!projected[1].grace);
    }

    #[test]
    fn split_voices_preserve_unison_multiplicity_without_column_identity() {
        let projected = events("**kern\n*^\n4c\t4c\n*v\t*v\n4d\n*-");
        assert_eq!(projected.len(), 3);
        assert_eq!(
            projected
                .iter()
                .filter(|event| event.pitch.pitch.step == DiatonicStep::C)
                .count(),
            2
        );
    }

    #[test]
    fn canonical_measure_slot_rewrite_preserves_event_multiset() {
        let document = KernDocument::parse(
            "**kern\t**kern\n*clefG2\t*clefG2\n=\n2d\t4f\n*\t*^\n.\t4a\t4b\n*\t*v\t*v\n=\n*-\t*-",
        )
        .expect("fixture should parse");
        let before = events_of(&document);
        let normalized = normalized_for_omr_with_options(
            &document,
            &NormalizeOptions {
                canonicalize_measure_grid: true,
                remove_invisible_rest_measures: false,
                ..Default::default()
            },
        );
        let after = events_of(&normalized);
        assert_eq!(after, before);
    }

    #[test]
    fn immediate_tie_canonicalization_preserves_written_events() {
        let document = KernDocument::parse("**kern\n4c[\n4c]\n*-").expect("fixture should parse");
        let before = events_of(&document);
        let normalized = normalized_for_omr_with_options(
            &document,
            &NormalizeOptions {
                remove_invisible_rest_measures: false,
                ..Default::default()
            },
        );
        assert_eq!(events_of(&normalized), before);
    }
}
