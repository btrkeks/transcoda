//! Unisons Verovio draws as two noteheads, and the `!LO:N:vis=` that makes
//! them share one.
//!
//! Verovio shares a notehead between notes of different layers on one staff
//! at the same onset and pitch only when both draw half-note heads or both
//! draw black heads, with the same number of dots
//! (`LayerElement::CalcElementHorizontalOverlap`). Any other unison it draws
//! side by side. Hiding one head with `!LO:N:head=invis` does not help: the
//! hidden head keeps its displaced position, and its stem stays with it.
//!
//! A `vis` that gives one note the other's head type and dots makes Verovio
//! share the head. The note to change is the one with fewer dots, or the
//! black head against a half-note head. A black head may take a half-note
//! `vis` only when it draws no flag, because Verovio draws the flag of the
//! `vis` value; a beamed note keeps its beams.

use crate::{
    score::{NoteSize, Score},
    types::{Accidental, ChordBeams, DurationValue},
    KernContent, KernLine, Note, SpineContent, SpineType,
};

/// Two notes of one staff that Verovio draws as separate unison heads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnmergedUnison {
    /// One-based source line of both notes.
    pub line: usize,
    /// Zero-based fields of the two notes, in field order.
    pub fields: [usize; 2],
    /// Zero-based field of the note that takes the `vis`.
    pub target: usize,
    /// The `!LO:N:vis=` value that makes the heads shared.
    pub visual: String,
}

impl UnmergedUnison {
    pub fn partner(&self, field: usize) -> usize {
        if self.fields[0] == field {
            self.fields[1]
        } else {
            self.fields[0]
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Head {
    Half,
    Black,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum StaffKey {
    Staff(u32),
    Track(usize),
}

struct Candidate<'d> {
    column: usize,
    staff: StaffKey,
    note: &'d Note,
    head: Head,
    /// Drawn value without dots, as a `**kern` recip.
    value: String,
    dots: u8,
    /// Draws a flag when it is not beamed.
    flagged: bool,
}

pub fn unmerged_unisons(score: &Score<'_>) -> Vec<UnmergedUnison> {
    let document = score.document();
    let topology = score.topology();
    let mut open_beams = vec![0u8; topology.layer_count()];
    let mut unisons = Vec::new();
    for (line_index, line) in document.lines().iter().enumerate() {
        let KernLine::Data(contents) = line else {
            continue;
        };
        let mut candidates = Vec::new();
        for (column, content) in contents.iter().enumerate() {
            let SpineContent::Kern(content) = content else {
                continue;
            };
            let layer = topology.layer_at(line_index, column);
            if *topology.spine_type(layer) != SpineType::Kern {
                continue;
            }
            let (note, beams) = match content {
                KernContent::Note(note) => (note, &ChordBeams::None),
                KernContent::Chord(chord) => match chord.notes.as_slice() {
                    [note] => (note, &chord.modifiers.beams),
                    _ => {
                        step_beams(&mut open_beams[layer.index()], &chord.modifiers.beams);
                        continue;
                    }
                },
                _ => continue,
            };
            let beamed = step_beams(&mut open_beams[layer.index()], beams);
            if note.modifiers.invisible
                || note.modifiers.cross_staff.is_some()
                || score.drawn().size(line_index, column) != NoteSize::Normal
                || !score
                    .state_at(line_index, column)
                    .track
                    .rhythm_scale
                    .is_unit()
                || !is_plain_value(&note.duration.value)
            {
                continue;
            }
            let drawn = score.drawn().duration(line_index, column, &note.duration);
            let DurationValue::Reciprocal(value) = *drawn.value() else {
                continue;
            };
            let head = match value {
                2 => Head::Half,
                4.. => Head::Black,
                _ => continue,
            };
            candidates.push(Candidate {
                column,
                staff: topology.staff_of(layer).map_or(
                    StaffKey::Track(topology.track_of(layer).index()),
                    StaffKey::Staff,
                ),
                note,
                head,
                value: value.to_string(),
                dots: drawn.dots(),
                flagged: drawn.beam_depth() > 0 && !beamed,
            });
        }
        let line = document.source_line_number(line_index);
        for (index, first) in candidates.iter().enumerate() {
            for second in &candidates[index + 1..] {
                if first.staff == second.staff && same_pitch(first.note, second.note) {
                    unisons.extend(merge(first, second, line));
                }
            }
        }
    }
    unisons
}

/// Applies one token's full beams to its layer and reports whether the token
/// is under a beam.
fn step_beams(open: &mut u8, beams: &ChordBeams) -> bool {
    let starts = beams.beam_start_count();
    let beamed = *open > 0 || starts > 0;
    *open = open
        .saturating_sub(beams.beam_end_count())
        .saturating_add(starts);
    beamed
}

/// A written value whose `vis` spelling is the drawn value itself: a power of
/// two, with no tuplet scaling.
fn is_plain_value(value: &DurationValue) -> bool {
    matches!(value, DurationValue::Reciprocal(value) if value.is_power_of_two())
}

/// Verovio's unison test: same step and octave, and enharmonic, where a note
/// without an accidental is natural.
fn same_pitch(first: &Note, second: &Note) -> bool {
    let natural =
        |accidental: &Option<Accidental>| accidental.clone().unwrap_or(Accidental::Natural);
    first.pitch == second.pitch && natural(&first.accidental) == natural(&second.accidental)
}

fn merge(first: &Candidate<'_>, second: &Candidate<'_>, line: usize) -> Option<UnmergedUnison> {
    if first.head == second.head && first.dots == second.dots {
        return None;
    }
    let (target, other, value) = if first.head == second.head {
        let (target, other) = if first.dots < second.dots {
            (first, second)
        } else {
            (second, first)
        };
        (target, other, target.value.as_str())
    } else {
        let (target, other) = if first.head == Head::Black {
            (first, second)
        } else {
            (second, first)
        };
        if target.flagged {
            return None;
        }
        (target, other, "2")
    };
    Some(UnmergedUnison {
        line,
        fields: [first.column, second.column],
        target: target.column,
        visual: format!("{value}{}", ".".repeat(other.dots.into())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{document::Framing, KernDocument};

    fn unisons(body: &str) -> Vec<UnmergedUnison> {
        let input = format!("**kern\n*clefG2\n*M3/4\n*^\n{body}*v\t*v\n*-\n");
        let document = KernDocument::parse(&input).unwrap();
        unmerged_unisons(&Score::resolve(&document, Framing::Complete).unwrap())
    }

    fn visual(body: &str) -> Option<(usize, String)> {
        let found = unisons(body);
        assert!(found.len() <= 1, "{found:?}");
        found.into_iter().next().map(|u| (u.target, u.visual))
    }

    #[test]
    fn a_half_note_and_a_beamed_eighth_take_a_half_note_vis() {
        let found = unisons("2c\\\t8cL/\n.\t8d\n.\t8e\n.\t8fJ\n");
        assert_eq!(
            found,
            vec![UnmergedUnison {
                line: 5,
                fields: [0, 1],
                target: 1,
                visual: "2".to_owned(),
            }]
        );
    }

    #[test]
    fn the_half_note_dots_carry_into_the_vis() {
        assert_eq!(
            visual("2.c\\\t8cL/\n.\t8d\n.\t8e\n.\t8f\n.\t8g\n.\t8aJ\n"),
            Some((1, "2.".to_owned()))
        );
    }

    #[test]
    fn a_dotted_quarter_and_an_eighth_add_only_the_dot() {
        assert_eq!(visual("4.c\\\t8c/\n.\t4d/\n"), Some((1, "8.".to_owned())));
        assert_eq!(visual("8c/\t4.c\\\n4d/\t.\n"), Some((0, "8.".to_owned())));
    }

    #[test]
    fn a_flagged_eighth_cannot_take_a_half_note_head() {
        assert_eq!(visual("2c\\\t8c/\n.\t8d/\n.\t4e/\n"), None);
    }

    #[test]
    fn a_quarter_takes_a_half_note_head() {
        assert_eq!(visual("2c\\\t4c/\n.\t4d/\n"), Some((1, "2".to_owned())));
    }

    #[test]
    fn heads_verovio_already_shares_are_not_reported() {
        assert_eq!(visual("4c\\\t8cL/\n.\t8dJ\n"), None);
        assert_eq!(visual("2c\\\t2c/\n"), None);
        assert_eq!(visual("4.c\\\t4.c/\n"), None);
    }

    #[test]
    fn an_existing_vis_counts_as_drawn() {
        assert_eq!(
            visual("!\t!LO:N:vis=2\n2c\\\t8cL/\n.\t8d\n.\t8e\n.\t8fJ\n"),
            None
        );
    }

    #[test]
    fn a_hidden_head_does_not_merge() {
        assert_eq!(
            visual("!\t!LO:N:head=invis\n2c\\\t8cL/\n.\t8d\n.\t8e\n.\t8fJ\n"),
            Some((1, "2".to_owned()))
        );
    }

    #[test]
    fn different_pitches_and_whole_notes_are_ignored() {
        assert_eq!(visual("2c\\\t8dL/\n.\t8d\n.\t8e\n.\t8fJ\n"), None);
        assert_eq!(visual("2c#\\\t8cL/\n.\t8d\n.\t8e\n.\t8fJ\n"), None);
        assert_eq!(visual("1c\\\t2c/\n.\t2d/\n"), None);
    }

    #[test]
    fn a_natural_sign_is_the_same_pitch_as_none() {
        assert_eq!(
            visual("2cn\\\t8cL/\n.\t8d\n.\t8e\n.\t8fJ\n"),
            Some((1, "2".to_owned()))
        );
    }

    #[test]
    fn invisible_and_chord_notes_are_ignored() {
        assert_eq!(visual("2cyy\\\t8cL/\n.\t8d\n.\t8e\n.\t8fJ\n"), None);
        assert_eq!(visual("2c 2e\\\t8cL/\n.\t8d\n.\t8e\n.\t8fJ\n"), None);
    }

    #[test]
    fn notes_on_different_staves_are_ignored() {
        let input = "**kern\t**kern\n*staff2\t*staff1\n2c\t8cL\n.\t8d\n.\t8e\n.\t8fJ\n*-\t*-\n";
        let document = KernDocument::parse(input).unwrap();
        let score = Score::resolve(&document, Framing::Complete).unwrap();
        assert!(unmerged_unisons(&score).is_empty());
    }
}
