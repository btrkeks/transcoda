/// Regular notes in one `**kern` chord token, preserved in source-subtoken order.
///
/// Semantic consumers that need staff order must use [`Chord::notes_by_pitch`],
/// [`Chord::lowest_note`], or [`Chord::highest_note`] explicitly. Keeping the
/// stored order source-faithful makes zero-based chord-note indexes stable for
/// diagnostics, rendered-element mapping, and semantic edits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chord {
    pub notes: Vec<Note>,
    pub modifiers: ChordModifiers,
}

impl Chord {
    /// Borrow chord notes in ascending diatonic staff position without
    /// changing their source-subtoken identity.
    pub fn notes_by_pitch(&self) -> Vec<&Note> {
        let mut notes = self.notes.iter().collect::<Vec<_>>();
        notes.sort_by_key(|note| note.pitch);
        notes
    }

    pub fn lowest_note(&self) -> Option<&Note> {
        self.notes.iter().min_by_key(|note| note.pitch)
    }

    pub fn highest_note(&self) -> Option<&Note> {
        self.notes.iter().max_by_key(|note| note.pitch)
    }
}

/// Modifiers that apply to a chord as a whole.
///
/// Example: in `(4c 4e)`, the slur belongs to the chord grouping rather than
/// to only one notehead.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChordModifiers {
    pub arpeggio: Option<Arpeggio>,
    /// `'`, `~`
    pub articulations: Vec<Placed<Articulation>>,
    /// `L`, `J`, `k`, `K`.
    pub beams: ChordBeams,
    /// `/`, `\`
    ///
    /// NOTE: This should not be serialized
    pub stem_direction: Option<StemDirection>,
    /// `,`, `,<`
    pub breath: Option<Placed<Breath>>,
    /// `o`, `o>`
    pub harmonic: Option<Placed<Harmonic>>,
    pub ornaments: Vec<Placed<Ornament>>,
    pub fermata: Option<Fermata>,
    /// `(`, `)`.
    ///
    /// Slurs are chord-level in `**kern` multiple-stops: VHV documents that
    /// slurs are not duplicated per chord note, unlike rhythm/articulation
    /// marks. Parsing is permissive, so duplicate slur markers in a chord are
    /// retained as `count`. Strict OMR serialization must resolve document-
    /// level slur spans before collapsing duplicates: multiple slurs with
    /// distinct start/end events are visually meaningful, while duplicate
    /// slurs with the same start and same end should be emitted only once.
    pub slurs: Vec<Slur>,
    /// - `{` = beginning/opening of a phrase
    /// - `}` = end/closing of a phrase
    /// - `&` marks elision for slurs or phrases
    /// - `&{` opens an elided phrase, and the matching close is `&}`
    /// - Multiple elision levels repeat ampersands, e.g. `&&{ ... &&}`
    ///
    /// NOTE: This should not be serialized
    pub phrases: Vec<Phrase>,
    /// `v` up-bow, `u` down-bow.
    pub bowing: Option<Placed<Bowing>>,
}

/// A mark with the staff side an RDF signifier forces on it.
///
/// Example: `'<` in `4c'<` is a staccato placed below the staff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placed<T> {
    pub mark: T,
    pub placement: Option<MarkPlacement>,
}

impl<T> From<T> for Placed<T> {
    fn from(mark: T) -> Self {
        Self {
            mark,
            placement: None,
        }
    }
}

/// Forced staff side for an articulation, ornament, bowing, breath or harmonic
/// mark, encoded by an RDF signifier right after the mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkPlacement {
    Above,
    Below,
}

/// Optional marks attached to a note.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NoteModifiers {
    /// Tie markers: ordinary `[`, `_`, `]`; sparse `[[`, `__`, `]]`;
    /// linked `N[`, `N_`, `N]`. Opening markers may include RDF placement.
    pub ties: Vec<Tie>,
    /// `X` immediately after an accidental, forcing the accidental to display.
    pub forced_accidental: Option<ForcedAccidental>,
    /// `y` immediately after an accidental, suppressing accidental display.
    pub invisible_accidental: bool,
    /// `x` editorial interpretation marker.
    ///
    /// NOTE: This should not be serialized
    pub editorial_interpretation: Option<EditorialInterpretation>,
    /// `yy` visual invisibility marker.
    ///
    /// NOTE: This should not be serialized
    pub invisible: bool,
    /// Cross-staff notehead placement marker.
    ///
    /// `>` displays the notehead on the staff above, and `<` displays it on
    /// the staff below while keeping it connected to the same note stem. In
    /// source text this marker belongs immediately after the pitch spelling,
    /// optional accidental, and accidental display marker (`X` or `y`).
    pub cross_staff: Option<CrossStaffPlacement>,
    /// RDF/user-defined note markers, e.g. `i` in `4f#i` when declared as
    /// `!!!RDF**kern: i = editorial accidental`.
    ///
    /// Serializers preserve `Z` for parentheses and `j` for square brackets
    /// around editorial accidentals. Other user signifiers are omitted.
    pub user_signifiers: Vec<UserSignifier>,
}

impl NoteModifiers {
    /// `Z` and `j` mark editorial accidentals in both supported dialects.
    pub fn has_editorial_accidental(&self) -> bool {
        self.user_signifiers
            .iter()
            .any(UserSignifier::is_editorial_accidental)
    }
}

/// Group of one or more grace notes in a single spine, preserved in
/// source-subtoken order.
///
/// Example: `8qcc 8qdd` or a groupetto-like `16qqd`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraceChord {
    pub notes: Vec<GraceNote>,
    pub modifiers: ChordModifiers,
}

impl GraceChord {
    /// Borrow grace notes in ascending diatonic staff position without
    /// changing their source-subtoken identity.
    pub fn notes_by_pitch(&self) -> Vec<&GraceNote> {
        let mut notes = self.notes.iter().collect::<Vec<_>>();
        notes.sort_by_key(|note| note.pitch);
        notes
    }

    pub fn lowest_note(&self) -> Option<&GraceNote> {
        self.notes.iter().min_by_key(|note| note.pitch)
    }

    pub fn highest_note(&self) -> Option<&GraceNote> {
        self.notes.iter().max_by_key(|note| note.pitch)
    }
}

/// One grace or cue-sized note.
///
/// Examples: `8qcc`, `16qqd`, `4Qe`, `ccq`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraceNote {
    pub kind: GraceKind,
    /// NOTE: When serializing, use 8 as default value here
    pub duration: Option<Duration>,
    pub pitch: Pitch,
    pub accidental: Option<Accidental>,
    pub modifiers: NoteModifiers,
}

/// Grace-note category signaled by `q`, `qq`, `Q`, `P`, or `p`.
///
/// Examples: `q` for acciaccatura, `qq` for accented grace, `Q` for groupetto.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraceKind {
    Acciaccatura,
    AccentedGrace,
    Groupetto,
    Appoggiatura,
    PostAppoggiatura,
}

/// Regular pitched note token.
///
/// Examples: `4c`, `[8ff#L`, `16.b-J`, `4G#?`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    pub duration: Duration,
    pub pitch: Pitch,
    pub accidental: Option<Accidental>,
    pub modifiers: NoteModifiers,
}

/// Duration spelling and augmentation dots.
///
/// Examples: `4`, `2.`, `8..`, `3%5`, `0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Duration {
    pub value: DurationValue,
    pub augmentation_dots: u8,
}

impl Duration {
    /// Return this duration as a normalized rational count of whole notes.
    ///
    /// For example, `4` is `1/4`, `4.` is `3/8`, and `0` is `2/1`.
    pub fn as_rational_whole_notes(&self) -> crate::MusicalDuration {
        crate::measure::MusicalDuration::from_duration(self)
    }

    /// Return this duration as a normalized rational count of quarter notes.
    ///
    /// For example, `4` is `1/1`, `2` is `2/1`, and `8` is `1/2`.
    pub fn as_quarter_notes(&self) -> crate::MusicalDuration {
        self.as_rational_whole_notes().multiplied_by(4)
    }
}

/// Base duration value before augmentation dots.
///
/// Examples: reciprocal `4` for quarter note, rational `3%5`, `0` for breve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurationValue {
    Reciprocal(u32),
    Rational { numerator: u32, denominator: u32 },
    Breve,
    Long,
    Maxima,
}

/// Pitch spelling without accidental.
///
/// The octave uses scientific pitch notation, where middle C is `C4`.
/// Ordering follows diatonic staff position: octave first, then `C` through `B`.
///
/// Examples: `c` parses as `C4`, `cc` parses as `C5`, and `C` parses as `C3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pitch {
    pub step: DiatonicStep,
    pub octave: i8,
}

impl Pitch {
    pub(crate) const LOWEST_SUPPORTED_KERN_PITCH: Self = Self {
        step: DiatonicStep::C,
        octave: -2,
    };
    pub(crate) const HIGHEST_SUPPORTED_KERN_PITCH: Self = Self {
        step: DiatonicStep::B,
        octave: 9,
    };

    pub(crate) fn is_supported_kern_pitch(self) -> bool {
        (Self::LOWEST_SUPPORTED_KERN_PITCH..=Self::HIGHEST_SUPPORTED_KERN_PITCH).contains(&self)
    }
}

/// Return the sounding MIDI note number for a parsed `**kern` pitch.
///
/// This is correct for the semantic pitch represented by this AST: `**kern`
/// encodes accidentals on each note, so callers should not infer pitch from key
/// signatures, previous accidentals in the measure, or barline state. `None` and
/// [`Accidental::Natural`] both mean an unaltered diatonic pitch.
///
/// This does not model engraving/display accidental state such as forced,
/// cautionary, or invisible accidentals.
///
/// Returns `None` when the spelled pitch falls outside the MIDI `0..=127`
/// range.
pub fn pitch_to_midi_pitch(pitch: Pitch, accidental: Option<&Accidental>) -> Option<u8> {
    let midi_pitch = pitch_to_sounding_semitone(pitch, accidental);

    u8::try_from(midi_pitch).ok().filter(|pitch| *pitch <= 127)
}

/// Return an unbounded signed semitone key for a parsed pitch.
///
/// Unlike [`pitch_to_midi_pitch`], this preserves every pitch accepted by the
/// `**kern` parser, including supported spellings outside MIDI `0..=127`.
pub(crate) fn pitch_to_sounding_semitone(pitch: Pitch, accidental: Option<&Accidental>) -> i16 {
    let base = (i16::from(pitch.octave) + 1) * 12 + i16::from(pitch.step.chromatic_index());
    let alteration = match accidental {
        None | Some(Accidental::Natural) => 0,
        Some(Accidental::DoubleFlat) => -2,
        Some(Accidental::Flat) => -1,
        Some(Accidental::Sharp) => 1,
        Some(Accidental::DoubleSharp) => 2,
    };
    base + alteration
}

impl PartialOrd for Pitch {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Pitch {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.octave, self.step.staff_index()).cmp(&(other.octave, other.step.staff_index()))
    }
}

/// Diatonic pitch letter.
///
/// Examples: `C`/`c` map to `C`, `F`/`f` map to `F`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiatonicStep {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
}

impl DiatonicStep {
    /// The step a lowercase pitch letter names, as key signatures spell it.
    pub fn from_lowercase(letter: char) -> Option<Self> {
        Some(match letter {
            'a' => Self::A,
            'b' => Self::B,
            'c' => Self::C,
            'd' => Self::D,
            'e' => Self::E,
            'f' => Self::F,
            'g' => Self::G,
            _ => return None,
        })
    }

    pub fn lowercase(self) -> char {
        match self {
            Self::A => 'a',
            Self::B => 'b',
            Self::C => 'c',
            Self::D => 'd',
            Self::E => 'e',
            Self::F => 'f',
            Self::G => 'g',
        }
    }

    fn staff_index(self) -> u8 {
        match self {
            DiatonicStep::C => 0,
            DiatonicStep::D => 1,
            DiatonicStep::E => 2,
            DiatonicStep::F => 3,
            DiatonicStep::G => 4,
            DiatonicStep::A => 5,
            DiatonicStep::B => 6,
        }
    }

    fn chromatic_index(self) -> u8 {
        match self {
            DiatonicStep::C => 0,
            DiatonicStep::D => 2,
            DiatonicStep::E => 4,
            DiatonicStep::F => 5,
            DiatonicStep::G => 7,
            DiatonicStep::A => 9,
            DiatonicStep::B => 11,
        }
    }
}

/// Accidental immediately following a pitch spelling.
///
/// Examples: `#`, `-`, `n`, `##`, `--`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Accidental {
    DoubleFlat,
    Flat,
    Natural,
    Sharp,
    DoubleSharp,
}

/// Tie state marker on a note.
///
/// Examples: `[` starts an ordinary tie, `[[` starts a disjunct/sparse tie,
/// and `N[` starts an explicitly linked tie. An opening marker may carry
/// placement, such as `[>` for an above tie or `_<` for a below continuation
/// tie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tie {
    Start,
    Continue,
    End,
    DisjunctStart,
    DisjunctContinue,
    DisjunctEnd,
    LinkedStart,
    LinkedContinue,
    LinkedEnd,
    /// A tie opener (`[` or `_`) with a forced side for the curve.
    PlacedOpen {
        phase: TieOpenPhase,
        resolution: TieResolution,
        placement: TiePlacement,
    },
}

impl Tie {
    pub const fn new(phase: TiePhase, resolution: TieResolution) -> Self {
        match (phase, resolution) {
            (TiePhase::Start, TieResolution::Ordinary) => Self::Start,
            (TiePhase::Continue, TieResolution::Ordinary) => Self::Continue,
            (TiePhase::End, TieResolution::Ordinary) => Self::End,
            (TiePhase::Start, TieResolution::Disjunct) => Self::DisjunctStart,
            (TiePhase::Continue, TieResolution::Disjunct) => Self::DisjunctContinue,
            (TiePhase::End, TieResolution::Disjunct) => Self::DisjunctEnd,
            (TiePhase::Start, TieResolution::Linked) => Self::LinkedStart,
            (TiePhase::Continue, TieResolution::Linked) => Self::LinkedContinue,
            (TiePhase::End, TieResolution::Linked) => Self::LinkedEnd,
        }
    }

    pub const fn phase(self) -> TiePhase {
        match self {
            Self::Start | Self::DisjunctStart | Self::LinkedStart => TiePhase::Start,
            Self::Continue | Self::DisjunctContinue | Self::LinkedContinue => TiePhase::Continue,
            Self::End | Self::DisjunctEnd | Self::LinkedEnd => TiePhase::End,
            Self::PlacedOpen {
                phase: TieOpenPhase::Start,
                ..
            } => TiePhase::Start,
            Self::PlacedOpen {
                phase: TieOpenPhase::Continue,
                ..
            } => TiePhase::Continue,
        }
    }

    pub const fn resolution(self) -> TieResolution {
        match self {
            Self::Start | Self::Continue | Self::End => TieResolution::Ordinary,
            Self::DisjunctStart | Self::DisjunctContinue | Self::DisjunctEnd => {
                TieResolution::Disjunct
            }
            Self::LinkedStart | Self::LinkedContinue | Self::LinkedEnd => TieResolution::Linked,
            Self::PlacedOpen { resolution, .. } => resolution,
        }
    }

    pub const fn is_start(self) -> bool {
        matches!(self.phase(), TiePhase::Start)
    }

    pub const fn is_continue(self) -> bool {
        matches!(self.phase(), TiePhase::Continue)
    }

    pub const fn is_end(self) -> bool {
        matches!(self.phase(), TiePhase::End)
    }

    pub const fn placement(self) -> Option<TiePlacement> {
        match self {
            Self::PlacedOpen { placement, .. } => Some(placement),
            _ => None,
        }
    }

    pub const fn with_placement(self, placement: TiePlacement) -> Self {
        let phase = match self.phase() {
            TiePhase::Start => TieOpenPhase::Start,
            TiePhase::Continue => TieOpenPhase::Continue,
            TiePhase::End => return self,
        };
        Self::PlacedOpen {
            phase,
            resolution: self.resolution(),
            placement,
        }
    }
}

/// An opening tie phase that may carry a curve placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TieOpenPhase {
    Start,
    Continue,
}

/// Forced side for a tie curve, encoded after `[` or `_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiePlacement {
    Above,
    Below,
}

/// Position of a tie marker in a tie chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TiePhase {
    Start,
    Continue,
    End,
}

/// Tie pairing/resolution mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TieResolution {
    /// Ordinary same-spine tie spelling: `[`, `_`, `]`.
    Ordinary,
    /// Sparse/disjunct same-spine tie spelling: `[[`, `__`, `]]`.
    Disjunct,
    /// Explicitly linked tie spelling: `N[`, `N_`, `N]`.
    Linked,
}

/// Forced/cautionary accidental marker.
///
/// Example: `X` in `4c#X`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForcedAccidental;

/// Editorial interpretation marker.
///
/// Example: `x` in `4cx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditorialInterpretation;

/// Slur boundary marker and optional RDF-forced placement on its opener.
///
/// Examples: `(` opens a slur, `)` closes it, `&(` opens an elided slur, and
/// `N(` opens a linked slur.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slur {
    Open {
        elided: bool,
        depth: u8,
        count: u8,
        placement: Option<SlurPlacement>,
        linked: bool,
    },
    Close {
        elided: bool,
        depth: u8,
        count: u8,
        linked: bool,
    },
}

/// Forced staff side for a slur, encoded by an RDF signifier after `(`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlurPlacement {
    Above,
    Below,
}

/// Phrase boundary marker.
///
/// Examples: `{` opens a phrase, `}` closes it, `&{` opens an elided phrase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phrase {
    Open { elision: Elision },
    Close { elision: Elision },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Elision {
    pub depth: u8, // 0 = ordinary, 1 = &, 2 = &&, ...
}

/// Auxiliary-note interval used by ornaments.
///
/// In `**kern` ornament spelling, uppercase markers generally indicate a
/// whole-tone auxiliary interval and lowercase markers indicate a semitone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxInterval {
    WholeTone,
    Semitone,
}

impl AuxInterval {
    pub const fn semitones(self) -> i8 {
        match self {
            Self::WholeTone => 2,
            Self::Semitone => 1,
        }
    }
}

/// Turn direction.
///
/// Examples: `S` is regular, `$` is inverted/Wagnerian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnForm {
    Regular,
    Inverted,
}

/// Turn placement.
///
/// Examples: `Sss` is delayed/centered, `sSss` is on-note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnTiming {
    Delayed,
    OnNote,
}

/// Explicit auxiliary intervals for a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnIntervals {
    pub upper: AuxInterval,
    pub lower: AuxInterval,
}

/// Structured VHV turn spelling.
///
/// `intervals: None` represents historical bare `S` or `$`, which is
/// parseable but underspecified for VHV rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Turn {
    pub form: TurnForm,
    pub timing: TurnTiming,
    pub intervals: Option<TurnIntervals>,
}

/// Mordent direction.
///
/// In `**kern`, `M`/`m` alternate with the upper neighbor and `W`/`w` alternate
/// with the lower neighbor. Avoid "inverted" for mordents: traditions disagree
/// about what that word means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MordentDirection {
    Upper,
    Lower,
}

/// Wavy-line state encoded by a trill marker run.
///
/// `TT`/`tt` starts a line after a trill symbol. `TTT`/`ttt` continues an
/// existing line without starting another trill symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrillLine {
    None,
    Start,
    Continuation,
}

/// Ornament marker.
///
/// Examples: `T` trill, `M` mordent, `S` turn, `$` Wagnerian turn, `O` generic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ornament {
    Trill {
        interval: AuxInterval,
        line: TrillLine,
    },
    Mordent {
        direction: MordentDirection,
        interval: Option<AuxInterval>,
        doubled: bool,
    },
    Turn(Turn),
    ConcludingTurn,
    Generic,
}

/// Articulation marker.
///
/// Examples: `'` staccato, `` ` `` staccatissimo, `~` tenuto, `^` accent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Articulation {
    Staccato,
    Pizzicato,
    Staccatissimo,
    Tenuto,
    Accent,
    Generic,
}

/// Bowing direction for string instruments.
///
/// Examples: `v` up-bow, `u` down-bow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Bowing {
    UpBow,
    DownBow,
}

/// Stem direction marker.
///
/// Examples: `/` up-stem, `\` down-stem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StemDirection {
    Up,
    Down,
}

/// Normalized chord-level beam markers.
///
/// Matching `L` and `J` markers are cancelled before beam state reaches the
/// AST, so a spelling like `LJ` is represented as [`ChordBeams::None`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ChordBeams {
    #[default]
    None,
    Start {
        level: u8,
    },
    End {
        level: u8,
    },
    PartialLeft {
        level: u8,
    },
    EndAndPartialLeft {
        end_level: u8,
        partial_left_level: u8,
    },
    StartAndPartialRight {
        start_level: u8,
        partial_right_level: u8,
    },
}

impl ChordBeams {
    pub fn from_counts(
        start_count: u8,
        end_count: u8,
        partial_left_count: u8,
        partial_right_count: u8,
    ) -> Option<Self> {
        let canceling_count = start_count.min(end_count);
        let start_count = start_count - canceling_count;
        let end_count = end_count - canceling_count;

        match (
            start_count,
            end_count,
            partial_left_count,
            partial_right_count,
        ) {
            (0, 0, 0, 0) => Some(Self::None),
            (level, 0, 0, 0) if level > 0 => Some(Self::Start { level }),
            (0, level, 0, 0) if level > 0 => Some(Self::End { level }),
            (0, 0, level, 0) if level > 0 => Some(Self::PartialLeft { level }),
            (0, end_level, partial_left_level, 0) if end_level > 0 && partial_left_level > 0 => {
                Some(Self::EndAndPartialLeft {
                    end_level,
                    partial_left_level,
                })
            }
            (start_level, 0, 0, partial_right_level)
                if start_level > 0 && partial_right_level > 0 =>
            {
                Some(Self::StartAndPartialRight {
                    start_level,
                    partial_right_level,
                })
            }
            _ => None,
        }
    }

    pub fn beam_start_count(&self) -> u8 {
        match self {
            Self::Start { level } => *level,
            Self::StartAndPartialRight { start_level, .. } => *start_level,
            _ => 0,
        }
    }

    pub fn beam_end_count(&self) -> u8 {
        match self {
            Self::End { level } => *level,
            Self::EndAndPartialLeft { end_level, .. } => *end_level,
            _ => 0,
        }
    }

    pub fn partial_left_count(&self) -> u8 {
        match self {
            Self::PartialLeft { level } => *level,
            Self::EndAndPartialLeft {
                partial_left_level, ..
            } => *partial_left_level,
            _ => 0,
        }
    }

    pub fn partial_right_count(&self) -> u8 {
        match self {
            Self::StartAndPartialRight {
                partial_right_level,
                ..
            } => *partial_right_level,
            _ => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Self::None)
    }

    pub fn write_kern_beam_spelling(&self, output: &mut String) {
        match self {
            Self::None => {}
            Self::Start { level } => output.push_str(&"L".repeat((*level).into())),
            Self::End { level } => output.push_str(&"J".repeat((*level).into())),
            Self::PartialLeft { level } => output.push_str(&"k".repeat((*level).into())),
            Self::EndAndPartialLeft {
                end_level,
                partial_left_level,
            } => {
                output.push_str(&"J".repeat((*end_level).into()));
                output.push_str(&"k".repeat((*partial_left_level).into()));
            }
            Self::StartAndPartialRight {
                start_level,
                partial_right_level,
            } => {
                output.push_str(&"L".repeat((*start_level).into()));
                output.push_str(&"K".repeat((*partial_right_level).into()));
            }
        }
    }
}

/// Beam marker, including stacked beams.
///
/// Examples:
/// `L` starts one beam,
/// `JJ` ends two beams,
/// `k` is partial-left,
/// `K` is partial-right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Beam {
    Start { level: u8 },
    End { level: u8 },
    PartialLeft { level: u8 },
    PartialRight { level: u8 },
}

/// Fermata/pause marker.
///
/// Examples: `;` in `4c;`, `;<` for a fermata forced below the staff. Barline
/// pauses use [`super::barline::BarlineFermata`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fermata {
    pub placement: Option<FermataPlacement>,
}

/// Forced staff side for a fermata, encoded by an RDF signifier after `;`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FermataPlacement {
    Above,
    Below,
}

/// Breath mark.
///
/// Example: `,` in `4c,`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Breath;

/// Natural harmonic marker.
///
/// Example: `o` in a string-instrument note token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Harmonic;

/// Arpeggiation scope, usually for a chord.
///
/// Humdrum uses `:` for a staff-local arpeggio and `::` for an arpeggio that
/// spans staves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arpeggio {
    StaffLocal,
    CrossStaff,
}

/// User-defined signifier from RDF/custom markup.
///
/// Example: a custom marker declared by `!!!RDF**kern:`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserSignifier {
    pub marker: String,
}

impl UserSignifier {
    pub fn is_editorial_accidental(&self) -> bool {
        matches!(self.marker.as_str(), "Z" | "j")
    }
}

/// Cross-staff notehead placement marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossStaffPlacement {
    /// `>`: display on the staff above.
    Above,
    /// `<`: display on the staff below.
    Below,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_to_midi_pitch_uses_scientific_pitch_numbering() {
        assert_eq!(
            pitch_to_midi_pitch(
                Pitch {
                    step: DiatonicStep::C,
                    octave: 4,
                },
                None
            ),
            Some(60)
        );
        assert_eq!(
            pitch_to_midi_pitch(
                Pitch {
                    step: DiatonicStep::A,
                    octave: 4,
                },
                None
            ),
            Some(69)
        );
    }

    #[test]
    fn pitch_to_midi_pitch_applies_explicit_accidentals() {
        let c4 = Pitch {
            step: DiatonicStep::C,
            octave: 4,
        };

        assert_eq!(pitch_to_midi_pitch(c4, None), Some(60));
        assert_eq!(
            pitch_to_midi_pitch(c4, Some(&Accidental::Natural)),
            Some(60)
        );
        assert_eq!(pitch_to_midi_pitch(c4, Some(&Accidental::Sharp)), Some(61));
        assert_eq!(
            pitch_to_midi_pitch(c4, Some(&Accidental::DoubleSharp)),
            Some(62)
        );
        assert_eq!(pitch_to_midi_pitch(c4, Some(&Accidental::Flat)), Some(59));
        assert_eq!(
            pitch_to_midi_pitch(c4, Some(&Accidental::DoubleFlat)),
            Some(58)
        );
    }

    #[test]
    fn pitch_to_midi_pitch_returns_none_outside_midi_range() {
        assert_eq!(
            pitch_to_midi_pitch(
                Pitch {
                    step: DiatonicStep::C,
                    octave: -1,
                },
                None
            ),
            Some(0)
        );
        assert_eq!(
            pitch_to_midi_pitch(
                Pitch {
                    step: DiatonicStep::C,
                    octave: -1,
                },
                Some(&Accidental::Flat)
            ),
            None
        );
        assert_eq!(
            pitch_to_midi_pitch(
                Pitch {
                    step: DiatonicStep::G,
                    octave: 9,
                },
                None
            ),
            Some(127)
        );
        assert_eq!(
            pitch_to_midi_pitch(
                Pitch {
                    step: DiatonicStep::G,
                    octave: 9,
                },
                Some(&Accidental::Sharp)
            ),
            None
        );
    }

    #[test]
    fn aux_interval_reports_chromatic_distance() {
        assert_eq!(AuxInterval::WholeTone.semitones(), 2);
        assert_eq!(AuxInterval::Semitone.semitones(), 1);
    }
}
