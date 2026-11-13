//! What each Transcoda V2 label dialect does with each kind of content.
//!
//! [`ContentPolicy::action`] is the only place that decides it, the way
//! [`TandemPolicy`] decides tandems. Lowering projects a document through
//! [`ContentPolicy::project_content`], the projected-invariant check asks the
//! same projection whether a token would change, and the printer reads its
//! layout rewrites from the table, so the three cannot disagree. The token
//! projections destructure every modifier struct, so a new field does not
//! compile until it is mapped to a row. `docs/dialects/transcoda-v2.md` is
//! generated from this table.

use crate::{
    types::{Arpeggio, ChordBeams, ChordModifiers, NoteModifiers},
    KernContent, LocalComment, Rest,
};

use super::TandemPolicy;

/// One dialect's decision for one kind of content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentAction {
    /// Written as in the source.
    Keep,
    /// Left out.
    Drop,
    /// Written in a different, canonical spelling ([`ContentKind::meaning`]
    /// says which).
    Rewrite,
}

/// A kind of content a Transcoda V2 label may or may not carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentKind {
    Tie,
    ForcedAccidental,
    HiddenAccidental,
    RedundantNatural,
    EditorialAccidental,
    EditorialInterpretation,
    UserSignifier,
    InvisibleNote,
    CrossStaff,
    Beam,
    BeamSpelling,
    Arpeggio,
    CrossStaffArpeggio,
    Articulation,
    Ornament,
    StemDirection,
    Bowing,
    Harmonic,
    Breath,
    Fermata,
    FermataPlacement,
    Slur,
    Phrase,
    ChordDuration,
    ChordNoteOrder,
    GraceDuration,
    RestPlacement,
    InvisibleRest,
    InvisibleRestFermata,
    InvisibleRestRun,
    InvisibleRestMeasure,
    MultiMeasureRest,
    ExclusiveInterpretation,
    NonKernSpine,
    LocalComment,

    SystemBreak,
    LeadingBarline,
    SpineSlots,
    TerminalLine,
}

/// Where a [`ContentKind`] lives, for the generated overview.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentGroup {
    Note,
    Mark,
    Chord,
    Rest,
    Document,
}

impl ContentGroup {
    pub fn name(self) -> &'static str {
        match self {
            Self::Note => "Notes",
            Self::Mark => "Marks on notes, chords and rests",
            Self::Chord => "Chords and grace notes",
            Self::Rest => "Rests",
            Self::Document => "Lines and spines",
        }
    }
}

impl ContentKind {
    pub const ALL: [Self; 39] = [
        Self::Tie,
        Self::ForcedAccidental,
        Self::HiddenAccidental,
        Self::RedundantNatural,
        Self::EditorialAccidental,
        Self::EditorialInterpretation,
        Self::UserSignifier,
        Self::InvisibleNote,
        Self::CrossStaff,
        Self::Beam,
        Self::BeamSpelling,
        Self::Arpeggio,
        Self::CrossStaffArpeggio,
        Self::Articulation,
        Self::Ornament,
        Self::StemDirection,
        Self::Bowing,
        Self::Harmonic,
        Self::Breath,
        Self::Fermata,
        Self::FermataPlacement,
        Self::Slur,
        Self::Phrase,
        Self::ChordDuration,
        Self::ChordNoteOrder,
        Self::GraceDuration,
        Self::RestPlacement,
        Self::InvisibleRest,
        Self::InvisibleRestFermata,
        Self::InvisibleRestRun,
        Self::InvisibleRestMeasure,
        Self::MultiMeasureRest,
        Self::ExclusiveInterpretation,
        Self::NonKernSpine,
        Self::LocalComment,
        Self::SystemBreak,
        Self::LeadingBarline,
        Self::SpineSlots,
        Self::TerminalLine,
    ];

    pub fn group(self) -> ContentGroup {
        match self {
            Self::Tie
            | Self::ForcedAccidental
            | Self::HiddenAccidental
            | Self::RedundantNatural
            | Self::EditorialAccidental
            | Self::EditorialInterpretation
            | Self::UserSignifier
            | Self::InvisibleNote
            | Self::CrossStaff => ContentGroup::Note,
            Self::Beam
            | Self::BeamSpelling
            | Self::Arpeggio
            | Self::CrossStaffArpeggio
            | Self::Articulation
            | Self::Ornament
            | Self::StemDirection
            | Self::Bowing
            | Self::Harmonic
            | Self::Breath
            | Self::Fermata
            | Self::FermataPlacement
            | Self::Slur
            | Self::Phrase => ContentGroup::Mark,
            Self::ChordDuration | Self::ChordNoteOrder | Self::GraceDuration => ContentGroup::Chord,
            Self::RestPlacement
            | Self::InvisibleRest
            | Self::InvisibleRestFermata
            | Self::InvisibleRestRun
            | Self::InvisibleRestMeasure
            | Self::MultiMeasureRest => ContentGroup::Rest,
            Self::ExclusiveInterpretation
            | Self::NonKernSpine
            | Self::LocalComment
            | Self::SystemBreak
            | Self::LeadingBarline
            | Self::SpineSlots
            | Self::TerminalLine => ContentGroup::Document,
        }
    }

    /// The source spelling and what a [`ContentAction::Rewrite`] turns it
    /// into.
    pub fn meaning(self) -> &'static str {
        match self {
            Self::Tie => "`[`, `_`, `]` ties",
            Self::ForcedAccidental => "`X` forced accidental",
            Self::HiddenAccidental => "`y` hidden accidental",
            Self::RedundantNatural => {
                "`n` on a note whose natural Verovio draws anyway, from the key signature or an \
                 earlier accidental in the measure (`4c#` then `4cn` is written `4c#` then `4c`)"
            }
            Self::EditorialAccidental => {
                "`Z`, `j` editorial accidental signifier, with its `X` or `y`"
            }
            Self::EditorialInterpretation => "`x` editorial interpretation",
            Self::UserSignifier => "other RDF user signifiers",
            Self::InvisibleNote => {
                "`yy` invisible note; it leaves its chord, and a lone note or a chord of only \
                 invisible notes becomes an invisible rest of its duration (a grace note becomes \
                 a null token)"
            }
            Self::CrossStaff => "`<`, `>` cross-staff note or rest",
            Self::Beam => "`L`, `J`, `k`, `K` beams",
            Self::BeamSpelling => {
                "beam marks, respelled from the groups and secondary breaks Verovio draws: \
                 full `L`/`J` beams as deep as both notes take, `k`/`K` for the rest of a \
                 note's depth, and `qq` for a beamed `q` grace note"
            }
            Self::Arpeggio => "`:` arpeggio",
            Self::CrossStaffArpeggio => "`::` cross-staff arpeggio, written as `:`",
            Self::Articulation => "articulations (`'`, `~`, `^`, ...)",
            Self::Ornament => "ornaments (`t`, `T`, `m`, `M`, `S`, ...)",
            Self::StemDirection => "`/`, `\\` stem direction",
            Self::Bowing => "`u`, `v` bowing",
            Self::Harmonic => "`o` harmonic",
            Self::Breath => "`,` breath mark",
            Self::Fermata => "`;` fermata on a note, chord or visible rest",
            Self::FermataPlacement => "`<`, `>` after a fermata",
            Self::Slur => "`(`, `)` slurs",
            Self::Phrase => "`{`, `}` phrases",
            Self::ChordDuration => {
                "durations after the first chord note: `4c 4e` is written `4c e`"
            }
            Self::ChordNoteOrder => "chord notes, written from low to high",
            Self::GraceDuration => {
                "grace notes without a duration get `8`: `qcc` is written `8qcc`"
            }
            Self::RestPlacement => "pitch letters that place a rest (`4rB`)",
            Self::InvisibleRest => "`yy` invisible rest",
            Self::InvisibleRestFermata => "`;` on an invisible rest",
            Self::InvisibleRestRun => {
                "consecutive invisible rests in one spine, merged into the fewest rests"
            }
            Self::InvisibleRestMeasure => "measures where every spine holds only invisible rests",
            Self::MultiMeasureRest => "runs of whole-measure rests, written `MM3r`",
            Self::ExclusiveInterpretation => "the `**kern` line",
            Self::NonKernSpine => "non-`**kern` spines (`**dynam`, `**text`, ...)",
            Self::LocalComment => "local comments",

            Self::SystemBreak => "`!!LO:LB` system breaks",
            Self::LeadingBarline => "non-repeat barlines before the first data line",
            Self::SpineSlots => {
                "split and merge placement, rewritten to each measure's positional grid: a staff \
                 keeps as many columns as its widest data line for the whole measure, splits \
                 follow the opening barline, merges precede the closing one, and a column a \
                 line lacks holds `ryy` unless its last token still sounds; a measure stays as \
                 written where the grid would change how iohumdrum reads a layer (its gaps, \
                 rests, tuplets, beams, clefs, cross-staff notes) or holds a clef or other \
                 emitted tandem inside a split"
            }
            Self::TerminalLine => "the final `*-` line",
        }
    }

    /// The actions a dialect may choose for this kind. Kinds with one action
    /// are fixed by the OMR printer; they are listed for the overview.
    pub fn allowed(self) -> &'static [ContentAction] {
        use ContentAction::{Drop, Keep, Rewrite};
        match self {
            Self::InvisibleRest => &[Keep],
            Self::ExclusiveInterpretation | Self::NonKernSpine => &[Drop],
            Self::CrossStaffArpeggio => &[Keep, Drop, Rewrite],
            Self::ChordDuration
            | Self::ChordNoteOrder
            | Self::GraceDuration
            | Self::InvisibleRestRun
            | Self::MultiMeasureRest
            | Self::SpineSlots
            | Self::BeamSpelling
            | Self::RedundantNatural => &[Keep, Rewrite],
            _ => &[Keep, Drop],
        }
    }
}

/// The content policy of one Transcoda V2 label dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContentPolicy {
    /// The `transcoda-v2` preset.
    #[default]
    TranscodaV2,
    /// The `transcoda-v2-training` preset: V2 without editorial accidentals.
    TranscodaV2Training,
}

impl ContentPolicy {
    pub const ALL: [Self; 2] = [Self::TranscodaV2, Self::TranscodaV2Training];

    pub fn action(self, kind: ContentKind) -> ContentAction {
        use ContentAction::{Drop, Keep, Rewrite};
        // Columns: TranscodaV2, TranscodaV2Training.
        let row: [ContentAction; 2] = match kind {
            ContentKind::Tie => [Drop, Drop],
            ContentKind::ForcedAccidental => [Drop, Drop],
            ContentKind::HiddenAccidental => [Drop, Drop],
            ContentKind::RedundantNatural => [Rewrite, Rewrite],
            ContentKind::EditorialAccidental => [Keep, Drop],
            ContentKind::EditorialInterpretation => [Drop, Drop],
            ContentKind::UserSignifier => [Drop, Drop],
            ContentKind::InvisibleNote => [Drop, Drop],
            ContentKind::CrossStaff => [Keep, Keep],
            ContentKind::Beam => [Keep, Keep],
            ContentKind::BeamSpelling => [Rewrite, Rewrite],
            ContentKind::Arpeggio => [Keep, Keep],
            ContentKind::CrossStaffArpeggio => [Rewrite, Rewrite],
            ContentKind::Articulation => [Drop, Drop],
            ContentKind::Ornament => [Drop, Drop],
            ContentKind::StemDirection => [Drop, Drop],
            ContentKind::Bowing => [Drop, Drop],
            ContentKind::Harmonic => [Drop, Drop],
            ContentKind::Breath => [Drop, Drop],
            ContentKind::Fermata => [Keep, Keep],
            ContentKind::FermataPlacement => [Drop, Drop],
            ContentKind::Slur => [Drop, Drop],
            ContentKind::Phrase => [Drop, Drop],
            ContentKind::ChordDuration => [Rewrite, Rewrite],
            ContentKind::ChordNoteOrder => [Rewrite, Rewrite],
            ContentKind::GraceDuration => [Rewrite, Rewrite],
            ContentKind::RestPlacement => [Drop, Drop],
            ContentKind::InvisibleRest => [Keep, Keep],
            ContentKind::InvisibleRestFermata => [Drop, Drop],
            ContentKind::InvisibleRestRun => [Rewrite, Rewrite],
            ContentKind::InvisibleRestMeasure => [Drop, Drop],
            ContentKind::MultiMeasureRest => [Rewrite, Rewrite],
            ContentKind::ExclusiveInterpretation => [Drop, Drop],
            ContentKind::NonKernSpine => [Drop, Drop],
            ContentKind::LocalComment => [Drop, Drop],

            ContentKind::SystemBreak => [Keep, Keep],
            ContentKind::LeadingBarline => [Drop, Drop],
            ContentKind::SpineSlots => [Rewrite, Rewrite],
            ContentKind::TerminalLine => [Drop, Drop],
        };
        row[self as usize]
    }

    pub fn drops(self, kind: ContentKind) -> bool {
        self.action(kind) == ContentAction::Drop
    }

    pub fn rewrites(self, kind: ContentKind) -> bool {
        self.action(kind) == ContentAction::Rewrite
    }

    /// Both label dialects share the Transcoda V2 tandem column.
    pub fn tandem_policy(self) -> TandemPolicy {
        TandemPolicy::TranscodaV2
    }

    pub fn preset_name(self) -> &'static str {
        match self {
            Self::TranscodaV2 => "transcoda-v2",
            Self::TranscodaV2Training => "transcoda-v2-training",
        }
    }

    /// Whether a local comment survives, ignoring the null comment `!`.
    pub fn keeps_local_comment(self, comment: &LocalComment) -> bool {
        let kind = match comment {
            LocalComment::Null => return true,

            _ => ContentKind::LocalComment,
        };
        !self.drops(kind)
    }

    /// Whether projecting `content` would leave it unchanged.
    pub fn keeps_content(self, content: &KernContent) -> bool {
        let mut projected = content.clone();
        self.project_content(&mut projected);
        projected == *content
    }

    /// Rewrite one `**kern` token to what this dialect writes.
    pub fn project_content(self, content: &mut KernContent) {
        match content {
            KernContent::Note(note) => {
                if note.modifiers.invisible && self.drops(ContentKind::InvisibleNote) {
                    *content = KernContent::Rest(Rest::invisible(note.duration.clone()));
                } else {
                    self.project_note_modifiers(&mut note.modifiers);
                }
            }
            KernContent::Chord(chord) => {
                let duration = chord.notes.first().map(|note| note.duration.clone());
                if self.drops(ContentKind::InvisibleNote) {
                    chord.notes.retain(|note| !note.modifiers.invisible);
                }
                for note in &mut chord.notes {
                    self.project_note_modifiers(&mut note.modifiers);
                }
                self.project_chord_modifiers(&mut chord.modifiers);
                if chord.notes.is_empty() {
                    *content = duration.map_or(KernContent::Null, |duration| {
                        KernContent::Rest(Rest::invisible(duration))
                    });
                }
            }
            KernContent::Grace(grace) => {
                if self.drops(ContentKind::InvisibleNote) {
                    grace.notes.retain(|note| !note.modifiers.invisible);
                }
                for note in &mut grace.notes {
                    self.project_note_modifiers(&mut note.modifiers);
                }
                self.project_chord_modifiers(&mut grace.modifiers);
                if grace.notes.is_empty() {
                    *content = KernContent::Null;
                }
            }
            KernContent::Rest(rest) => self.project_rest(rest),
            KernContent::Null
            | KernContent::InvisibleRestSlice(_)
            | KernContent::MultiMeasureRest(_) => {}
        }
    }

    fn project_note_modifiers(self, modifiers: &mut NoteModifiers) {
        let keeps_editorial =
            modifiers.has_editorial_accidental() && !self.drops(ContentKind::EditorialAccidental);
        let NoteModifiers {
            ties,
            forced_accidental,
            invisible_accidental,
            editorial_interpretation,
            // Removed with its token in `project_content`.
            invisible: _,
            cross_staff,
            user_signifiers,
        } = modifiers;
        if self.drops(ContentKind::Tie) {
            ties.clear();
        }
        if !keeps_editorial {
            if self.drops(ContentKind::ForcedAccidental) {
                *forced_accidental = None;
            }
            if self.drops(ContentKind::HiddenAccidental) {
                *invisible_accidental = false;
            }
        }
        if self.drops(ContentKind::EditorialInterpretation) {
            *editorial_interpretation = None;
        }
        if self.drops(ContentKind::CrossStaff) {
            *cross_staff = None;
        }
        user_signifiers.retain(|sign| {
            let kind = if sign.is_editorial_accidental() {
                ContentKind::EditorialAccidental
            } else {
                ContentKind::UserSignifier
            };
            !self.drops(kind)
        });
    }

    fn project_chord_modifiers(self, modifiers: &mut ChordModifiers) {
        let ChordModifiers {
            arpeggio,
            articulations,
            beams,
            stem_direction,
            breath,
            harmonic,
            ornaments,
            fermata,
            slurs,
            phrases,
            bowing,
        } = modifiers;
        match arpeggio {
            Some(Arpeggio::StaffLocal) if self.drops(ContentKind::Arpeggio) => *arpeggio = None,
            Some(Arpeggio::CrossStaff) => match self.action(ContentKind::CrossStaffArpeggio) {
                ContentAction::Keep => {}
                ContentAction::Drop => *arpeggio = None,
                ContentAction::Rewrite => *arpeggio = Some(Arpeggio::StaffLocal),
            },
            _ => {}
        }
        if self.drops(ContentKind::Articulation) {
            articulations.clear();
        }
        if self.drops(ContentKind::Beam) {
            *beams = ChordBeams::default();
        }
        if self.drops(ContentKind::StemDirection) {
            *stem_direction = None;
        }
        if self.drops(ContentKind::Breath) {
            *breath = None;
        }
        if self.drops(ContentKind::Harmonic) {
            *harmonic = None;
        }
        if self.drops(ContentKind::Ornament) {
            ornaments.clear();
        }
        self.project_fermata(fermata, ContentKind::Fermata);
        if self.drops(ContentKind::Slur) {
            slurs.clear();
        }
        if self.drops(ContentKind::Phrase) {
            phrases.clear();
        }
        if self.drops(ContentKind::Bowing) {
            *bowing = None;
        }
    }

    fn project_rest(self, rest: &mut Rest) {
        let Rest {
            duration: _,
            cross_staff,
            placement,
            invisible,
            breath,
            fermata,
            slurs,
            phrases,
        } = rest;
        if self.drops(ContentKind::CrossStaff) {
            *cross_staff = None;
        }
        if self.drops(ContentKind::RestPlacement) {
            *placement = None;
        }
        if self.drops(ContentKind::Breath) {
            *breath = None;
        }
        let fermata_kind = if *invisible {
            ContentKind::InvisibleRestFermata
        } else {
            ContentKind::Fermata
        };
        self.project_fermata(fermata, fermata_kind);
        if self.drops(ContentKind::Slur) {
            slurs.clear();
        }
        if self.drops(ContentKind::Phrase) {
            phrases.clear();
        }
    }

    fn project_fermata(self, fermata: &mut Option<crate::types::Fermata>, kind: ContentKind) {
        if self.drops(kind) {
            *fermata = None;
        } else if let Some(fermata) = fermata {
            if self.drops(ContentKind::FermataPlacement) {
                fermata.placement = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_policy_chooses_an_allowed_action() {
        for policy in ContentPolicy::ALL {
            for kind in ContentKind::ALL {
                assert!(
                    kind.allowed().contains(&policy.action(kind)),
                    "{policy:?} {kind:?} chooses {:?}",
                    policy.action(kind)
                );
            }
        }
    }

    #[test]
    fn a_rest_loses_its_slur_and_fermata_placement_like_a_note() {
        let document = crate::KernDocument::parse("**kern\n*M2/4\n4r(\n4c);>\n*-").unwrap();
        for preset in [
            crate::serialize::SerializePreset::TranscodaV2,
            crate::serialize::SerializePreset::TranscodaV2Training,
        ] {
            assert_eq!(
                preset.try_serialize_document(&document).unwrap(),
                "*clefG2\n*k[]\n*M2/4\n4r\n4c;\n"
            );
        }
    }

    #[test]
    fn a_token_of_only_invisible_notes_becomes_an_invisible_rest_of_its_duration() {
        let token = |raw: &str| crate::parse::parse_kern_token(raw, Default::default()).unwrap();
        for (written, projected) in [
            ("4.b-yy", "4.ryy"),
            ("2b-yy bb-yy", "2ryy"),
            ("12cyy", "12ryy"),
            ("3%2cyy", "3%2ryy"),
            ("[8cyyL", "8ryy"),
            ("(4cyy;' 4eyy)", "4ryy"),
            ("4c 4eyy", "4c"),
            ("8qcyy", "."),
        ] {
            for policy in ContentPolicy::ALL {
                let mut content = token(written);
                policy.project_content(&mut content);
                assert_eq!(content, token(projected), "{policy:?} {written:?}");
            }
        }
    }
}
