use super::{
    Chord, DynamContent, FingContent, GraceChord, GraceKind, GraceNote, InvisibleRestSlice,
    MultiMeasureRest, MxmlHarmonyContent, Note, Ornament, Pitch, Placed, RawToken, Rest, SpineType,
    TextContent,
};

/// Data content in one spine position.
///
/// The outer variant records the active exclusive interpretation for the spine
/// where the token appeared. This keeps `**kern`, `**dynam`, `**text`, and
/// `**mxhm` content distinct even when they share token spellings such as `.`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpineContent {
    Kern(KernContent),
    /// NOTE: This should not be serialized
    Dynam(DynamContent),
    Fing(FingContent),
    /// NOTE: This should not be serialized
    Text(TextContent),
    /// NOTE: This should not be serialized
    MxmlHarmony(MxmlHarmonyContent),
    /// NOTE: This should not be serialized
    Opaque {
        spine_type: SpineType,
        raw: RawToken,
    },
}

/// Musical content in one `**kern` spine position on a data line.
///
/// Examples: `.` is null, `4r` is a rest, and `4c` or `4c 4e 4g` are chords.
///
/// Data-line pitched tokens are represented as chords even when the spine entry
/// contains only one note. `Note` is the notehead-level representation inside a
/// chord or grace chord.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernContent {
    Null,
    Rest(Rest),
    InvisibleRestSlice(InvisibleRestSlice),
    MultiMeasureRest(MultiMeasureRest),
    Note(Note),
    Chord(Chord),
    Grace(GraceChord),
}

impl KernContent {
    /// Return this token's duration as a normalized rational count of whole notes.
    ///
    /// Null tokens, grace notes, and multi-measure rests return zero because
    /// they do not carry an ordinary duration-bearing `**kern` note/rest value
    /// in the current AST.
    pub fn duration(&self) -> Result<crate::MusicalDuration, crate::KernContentDurationError> {
        crate::measure::kern_content_duration(self)
    }

    /// Iterate over every pitched notehead in this token: regular notes,
    /// chord notes, and grace notes.
    ///
    /// There is no regular-only view. A caller that must skip grace notes
    /// tests [`KernPitchedNote::Grace`] itself, so the skip shows at the call site.
    pub fn noteheads(&self) -> impl Iterator<Item = KernNoteRef<'_>> {
        let notes = match self {
            KernContent::Note(note) => KernNoteheads::SingleRegular(Some(note)),
            KernContent::Chord(chord) => {
                KernNoteheads::RegularChord(chord.notes.iter().enumerate())
            }
            KernContent::Grace(grace) => KernNoteheads::GraceChord(grace.notes.iter().enumerate()),
            KernContent::Null
            | KernContent::Rest(_)
            | KernContent::InvisibleRestSlice(_)
            | KernContent::MultiMeasureRest(_) => KernNoteheads::Empty,
        };

        notes
    }

    /// Return ornaments attached to the source token that owns this content.
    ///
    /// Regular chords carry token-level modifiers. Grace and manually
    /// constructed bare [`KernContent::Note`] values report no ornaments for
    /// this accessor.
    pub fn token_ornaments(&self) -> &[Placed<Ornament>] {
        match self {
            KernContent::Chord(chord) => &chord.modifiers.ornaments,
            KernContent::Note(_)
            | KernContent::Grace(_)
            | KernContent::Null
            | KernContent::Rest(_)
            | KernContent::InvisibleRestSlice(_)
            | KernContent::MultiMeasureRest(_) => &[],
        }
    }
}

/// Borrowed view of one pitched notehead inside [`KernContent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernNoteRef<'a> {
    pub note: KernPitchedNote<'a>,
    /// Zero-based source-subtoken index, or `None` for single-note tokens.
    pub chord_note_index: Option<usize>,
}

/// Borrowed pitched notehead, preserving whether it is regular or grace-sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernPitchedNote<'a> {
    Regular(&'a Note),
    Grace(&'a GraceNote),
}

impl<'a> KernPitchedNote<'a> {
    pub const fn pitch(self) -> Pitch {
        match self {
            KernPitchedNote::Regular(note) => note.pitch,
            KernPitchedNote::Grace(note) => note.pitch,
        }
    }

    pub fn accidental(self) -> Option<&'a super::Accidental> {
        match self {
            KernPitchedNote::Regular(note) => note.accidental.as_ref(),
            KernPitchedNote::Grace(note) => note.accidental.as_ref(),
        }
    }

    pub fn modifiers(self) -> &'a super::NoteModifiers {
        match self {
            KernPitchedNote::Regular(note) => &note.modifiers,
            KernPitchedNote::Grace(note) => &note.modifiers,
        }
    }

    pub fn grace_kind(self) -> Option<&'a GraceKind> {
        match self {
            KernPitchedNote::Regular(_) => None,
            KernPitchedNote::Grace(note) => Some(&note.kind),
        }
    }
}

enum KernNoteheads<'a> {
    Empty,
    SingleRegular(Option<&'a Note>),
    RegularChord(std::iter::Enumerate<std::slice::Iter<'a, Note>>),
    GraceChord(std::iter::Enumerate<std::slice::Iter<'a, GraceNote>>),
}

impl<'a> Iterator for KernNoteheads<'a> {
    type Item = KernNoteRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            KernNoteheads::Empty => None,
            KernNoteheads::SingleRegular(note) => note.take().map(|note| KernNoteRef {
                note: KernPitchedNote::Regular(note),
                chord_note_index: None,
            }),
            KernNoteheads::RegularChord(notes) => notes.next().map(|(index, note)| KernNoteRef {
                note: KernPitchedNote::Regular(note),
                chord_note_index: Some(index),
            }),
            KernNoteheads::GraceChord(notes) => notes.next().map(|(index, note)| KernNoteRef {
                note: KernPitchedNote::Grace(note),
                chord_note_index: Some(index),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DiatonicStep, Duration, DurationValue, Pitch};

    fn steps(content: &KernContent) -> Vec<(Option<usize>, DiatonicStep, Option<GraceKind>)> {
        content
            .noteheads()
            .map(|notehead| {
                (
                    notehead.chord_note_index,
                    notehead.note.pitch().step,
                    notehead.note.grace_kind().cloned(),
                )
            })
            .collect()
    }

    #[test]
    fn noteheads_yields_regular_and_grace_heads_with_source_indexes() {
        let regular = KernContent::Chord(Chord::try_from("4g 4c 4e").unwrap());
        assert_eq!(
            steps(&regular),
            vec![
                (Some(0), DiatonicStep::G, None),
                (Some(1), DiatonicStep::C, None),
                (Some(2), DiatonicStep::E, None),
            ]
        );

        let grace = KernContent::Grace(GraceChord::try_from("8qdd 8qcc").unwrap());
        assert_eq!(
            steps(&grace),
            vec![
                (Some(0), DiatonicStep::D, Some(GraceKind::Acciaccatura)),
                (Some(1), DiatonicStep::C, Some(GraceKind::Acciaccatura)),
            ]
        );
    }

    #[test]
    fn noteheads_allows_legacy_direct_note_variant() {
        let content = KernContent::Note(Note {
            duration: Duration {
                value: DurationValue::Reciprocal(4),
                augmentation_dots: 0,
            },
            pitch: Pitch {
                step: DiatonicStep::A,
                octave: 4,
            },
            accidental: None,
            modifiers: Default::default(),
        });

        assert_eq!(steps(&content), vec![(None, DiatonicStep::A, None)]);
    }

    #[test]
    fn noteheads_skip_unpitched_content() {
        let unpitched = vec![
            KernContent::Null,
            KernContent::Rest(Rest::try_from("4r").unwrap()),
            KernContent::InvisibleRestSlice(InvisibleRestSlice { fermata: None }),
            KernContent::MultiMeasureRest(MultiMeasureRest { measure_count: 2 }),
        ];

        for content in unpitched {
            assert_eq!(steps(&content), vec![]);
        }
    }
}
