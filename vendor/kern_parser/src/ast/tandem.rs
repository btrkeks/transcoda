use super::{Accidental, DiatonicStep, RawToken};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TandemInterpretation {
    /// `*`
    Null,
    /// `*clefG2`, `*clefF4`, etc.
    Clef(Clef),
    /// `*k[]`, `*k[f#c#]`, etc.
    KeySignature(KeySignature),
    /// `*C:`, `*a:`, `*B-:`, etc.
    ///
    /// NOTE: This should be skipped during serialization
    KeyDesignation(KeyDesignation),
    /// `*M4/4`, `*M6/8`, `*M3/4yy`, etc.
    TimeSignature(TimeSignatureTandem),
    /// `*met(c)`, `*met(C|)`, `*met()`, etc.
    MeterSymbol(MeterSymbol),
    /// `*MX`: irregular meter. Bars have no fixed length until the next `*M`.
    FreeMeter,
    /// `*8va`, `*X8va`, `*8ba`, `*15ma`, etc.
    Ottava(Ottava),
    /// `*ped`, `*Xped`
    Pedal(Pedal),
    /// `*tuplet`, `*Xtuplet`
    Tuplet(Tuplet),
    /// `*brackettup`, `*Xbrackettup`
    TupletBracket(TupletBracket),
    /// `*rep`, `*Xrep`
    Repetition(Repetition),
    /// `*tupbreak`, a boundary between visual tuplet groups.
    TupletBreak,
    /// `*stria5`, etc.
    ///
    /// NOTE: This should not be serialized
    Stria(Stria),
    /// `*part1`, `*part2`, etc.
    ///
    /// NOTE: This should not be serialized
    Part(Part),
    /// `*staff1`, `*staff2`, etc.
    ///
    /// NOTE: This should not be serialized
    Staff(Staff),
    /// `*group1`, `*group2`, etc.
    ///
    /// NOTE: This should not be serialized
    Group(Group),
    /// `*Ivioln`, `*Icello`, etc.
    ///
    /// NOTE: This should not be serialized
    Instrument(Instrument),
    /// `*MM96`, etc.
    ///
    /// NOTE: This should not be serialized
    Tempo(Tempo),
    /// `*tremolo`, `*Xtremolo`
    ///
    /// NOTE: This should not be serialized
    Tremolo(Tremolo),
    /// `*lig`, `*Xlig`, `*col`, `*Xcol`
    BracketSpan(BracketSpan),
    /// Humdrum `thru` section label, such as `*>A`.
    SectionLabel(SectionLabel),
    /// Humdrum `thru` expansion list, such as `*>[A,A,B,B]`.
    ExpansionList(ExpansionList),
    /// Visual/rendering state tandems such as `*2\right` and `*2\left`.
    VisualStyle(VisualStyle),
    /// `*cue`, `*Xcue`
    Cue(Cue),
    /// `*rscale:2`, `*rscale:1/2`
    RhythmScale(RhythmScale),
    /// Opaque tandem interpretation in an opaque non-kern spine.
    ///
    /// NOTE: This should not be serialized
    Raw(RawToken),
}

/// Clef tandem interpretation.
///
/// Examples: `*clefG2`, `*clefF4`, `*clefC3`, `*clefGv2`, `*clefG2yy`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clef {
    pub sign: ClefSign,
    /// Staff line number encoded at the end of the token, not an input line.
    /// *clefG2 -> 2
    /// *clefF4 -> 4
    /// *clefC3 -> 3
    pub line: u8,
    /// *clefG2    octave_shift = 0
    /// *clefGv2   octave_shift = -1
    /// *clefGvv2  octave_shift = -2
    /// *clefG^2   octave_shift = 1
    /// *clefG^^2  octave_shift = 2
    pub octave_shift: i8,
    /// `yy` visual invisibility marker.
    pub invisible: bool,
}

impl Clef {
    /// Return the canonical Humdrum tandem token for this clef.
    pub fn as_kern(&self) -> String {
        let sign = match self.sign {
            ClefSign::G => "G",
            ClefSign::F => "F",
            ClefSign::C => "C",
            ClefSign::Tab => "TAB",
            ClefSign::Percussion => "X",
        };
        let shift = if self.octave_shift < 0 {
            "v".repeat(self.octave_shift.unsigned_abs().into())
        } else {
            "^".repeat(self.octave_shift as usize)
        };
        let visibility = if self.invisible { "yy" } else { "" };

        if matches!(self.sign, ClefSign::Tab | ClefSign::Percussion) && self.line == 0 {
            return format!("*clef{sign}{shift}{visibility}");
        }

        format!("*clef{sign}{shift}{}{visibility}", self.line)
    }
}

/// Clef family/sign used by a [`Clef`].
///
/// Examples: `G` in `*clefG2`, `F` in `*clefF4`, `C` in `*clefC3`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClefSign {
    G,
    F,
    C,
    Tab,
    Percussion,
}

/// Key designation tandem interpretation.
///
/// Examples: `*C:` for C major, `*a:` for A minor, `*B-:` for B-flat major.
///
/// These indicate tonal center and mode, not key signature. They are parsed
/// for input compatibility and should not be emitted by OMR serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyDesignation {
    pub step: DiatonicStep,
    pub accidental: Option<Accidental>,
    pub mode: KeyMode,
}

/// Mode encoded by a key designation pitch-letter's case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyMode {
    Major,
    Minor,
}

/// Key signature tandem interpretation.
///
/// Examples: `*k[]`, `*k[f#c#]`, `*k[b-e-a-]`, and with cancellation
/// naturals `*k[bnenf#]`.
///
/// The naturals are drawn before the new accidentals to cancel the previous
/// key's. They belong to the token, not to the key: pitch and accidental
/// state read only [`Self::fifths`], and track state stores the key
/// [`Self::without_naturals`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeySignature {
    fifths: Fifths,
    /// Cancellation naturals in written order, packed at the front.
    naturals: [Option<DiatonicStep>; 7],
}

impl KeySignature {
    pub fn from_fifths(fifths: Fifths) -> Self {
        Self {
            fifths,
            naturals: [None; 7],
        }
    }

    /// This key with cancellation naturals on `steps`, drawn in the given
    /// order. `None` when a step repeats or carries one of the key's own
    /// accidentals.
    pub fn with_naturals(self, steps: &[DiatonicStep]) -> Option<Self> {
        let mut naturals = [None; 7];
        for (index, &step) in steps.iter().enumerate() {
            if steps[..index].contains(&step) || self.accidental_for_step(step).is_some() {
                return None;
            }
            naturals[index] = Some(step);
        }
        Some(Self {
            fifths: self.fifths,
            naturals,
        })
    }

    pub fn fifths(self) -> Fifths {
        self.fifths
    }

    /// Cancellation naturals in written order.
    pub fn naturals(self) -> impl Iterator<Item = DiatonicStep> {
        self.naturals.into_iter().map_while(|step| step)
    }

    pub fn has_naturals(self) -> bool {
        self.naturals[0].is_some()
    }

    /// The key alone, as track state holds it.
    pub fn without_naturals(self) -> Self {
        Self::from_fifths(self.fifths)
    }

    /// The previous key's accidentals that this key does not keep, in
    /// `previous`'s order: what cancellation naturals would cancel.
    pub fn dropped_from(self, previous: KeySignature) -> Vec<DiatonicStep> {
        previous
            .accidental_steps()
            .iter()
            .copied()
            .filter(|&step| self.accidental_for_step(step) != previous.accidental_for_step(step))
            .collect()
    }

    /// The steps the key alters, in circle-of-fifths order.
    fn accidental_steps(self) -> &'static [DiatonicStep] {
        let fifths = self.fifths.get();
        if fifths >= 0 {
            &SHARP_ORDER[..fifths as usize]
        } else {
            &FLAT_ORDER[..(-fifths) as usize]
        }
    }

    /// Return the key-signature accidental for a diatonic step.
    ///
    /// Key designations such as `*C:` are tonal centers and are intentionally
    /// not modeled here.
    pub fn accidental_for_step(self, step: DiatonicStep) -> Option<Accidental> {
        if !self.accidental_steps().contains(&step) {
            return None;
        }
        Some(if self.fifths.get() > 0 {
            Accidental::Sharp
        } else {
            Accidental::Flat
        })
    }

    /// The canonical spelling: naturals first in written order, then the
    /// key's accidentals in circle-of-fifths order.
    pub fn as_kern(self) -> String {
        let key = self.key_as_kern();
        if !self.has_naturals() {
            return key.to_owned();
        }
        let naturals = self
            .naturals()
            .map(|step| format!("{}n", step.lowercase()))
            .collect::<String>();
        format!("*k[{naturals}{}", &key[3..])
    }

    fn key_as_kern(self) -> &'static str {
        match self.fifths.get() {
            -7 => "*k[b-e-a-d-g-c-f-]",
            -6 => "*k[b-e-a-d-g-c-]",
            -5 => "*k[b-e-a-d-g-]",
            -4 => "*k[b-e-a-d-]",
            -3 => "*k[b-e-a-]",
            -2 => "*k[b-e-]",
            -1 => "*k[b-]",
            0 => "*k[]",
            1 => "*k[f#]",
            2 => "*k[f#c#]",
            3 => "*k[f#c#g#]",
            4 => "*k[f#c#g#d#]",
            5 => "*k[f#c#g#d#a#]",
            6 => "*k[f#c#g#d#a#e#]",
            7 => "*k[f#c#g#d#a#e#b#]",
            _ => unreachable!("Fifths only permits values from -7 through 7"),
        }
    }
}

const SHARP_ORDER: [DiatonicStep; 7] = [
    DiatonicStep::F,
    DiatonicStep::C,
    DiatonicStep::G,
    DiatonicStep::D,
    DiatonicStep::A,
    DiatonicStep::E,
    DiatonicStep::B,
];

const FLAT_ORDER: [DiatonicStep; 7] = [
    DiatonicStep::B,
    DiatonicStep::E,
    DiatonicStep::A,
    DiatonicStep::D,
    DiatonicStep::G,
    DiatonicStep::C,
    DiatonicStep::F,
];

/// Circle-of-fifths position for a key signature.
///
/// Values range from seven flats (`-7`) through no accidentals (`0`) to
/// seven sharps (`7`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fifths(i8);

impl Fifths {
    pub const MIN: i8 = -7;
    pub const MAX: i8 = 7;

    pub fn new(value: i8) -> Option<Self> {
        (Self::MIN..=Self::MAX)
            .contains(&value)
            .then_some(Self(value))
    }

    pub fn get(self) -> i8 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(fifths: i8) -> KeySignature {
        KeySignature::from_fifths(Fifths::new(fifths).expect("valid fifths"))
    }

    #[test]
    fn empty_key_signature_returns_no_accidentals() {
        for step in [
            DiatonicStep::A,
            DiatonicStep::B,
            DiatonicStep::C,
            DiatonicStep::D,
            DiatonicStep::E,
            DiatonicStep::F,
            DiatonicStep::G,
        ] {
            assert_eq!(key(0).accidental_for_step(step), None);
        }
    }

    #[test]
    fn sharp_key_signature_applies_sharp_order() {
        let signature = key(2);

        assert_eq!(
            signature.accidental_for_step(DiatonicStep::F),
            Some(Accidental::Sharp)
        );
        assert_eq!(
            signature.accidental_for_step(DiatonicStep::C),
            Some(Accidental::Sharp)
        );
        assert_eq!(signature.accidental_for_step(DiatonicStep::G), None);
    }

    #[test]
    fn flat_key_signature_applies_flat_order() {
        let signature = key(-3);

        assert_eq!(
            signature.accidental_for_step(DiatonicStep::B),
            Some(Accidental::Flat)
        );
        assert_eq!(
            signature.accidental_for_step(DiatonicStep::E),
            Some(Accidental::Flat)
        );
        assert_eq!(
            signature.accidental_for_step(DiatonicStep::A),
            Some(Accidental::Flat)
        );
        assert_eq!(signature.accidental_for_step(DiatonicStep::D), None);
    }

    #[test]
    fn all_fifths_values_map_without_panic() {
        for fifths in Fifths::MIN..=Fifths::MAX {
            let signature = key(fifths);
            let _ = signature.accidental_for_step(DiatonicStep::C);
        }
    }
}

/// Numeric time signature.
///
/// Example: `*M3/8` has numerator `3` and denominator `8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSignature {
    pub numerator: u32,
    pub denominator: u32,
}

impl TimeSignature {
    /// Return the measure duration as a normalized rational count of whole notes.
    ///
    /// For example, `4/4` is `1/1`, `3/4` is `3/4`, and `6/8` is `3/4`.
    pub fn as_rational_whole_notes(&self) -> crate::MusicalDuration {
        crate::measure::MusicalDuration::from_time_signature(self)
    }

    pub fn as_kern(&self) -> String {
        format!("*M{}/{}", self.numerator, self.denominator)
    }
}

/// Time signature tandem, `*M3/4` or `*M3/4yy`.
///
/// The `yy` marker hides the printed signature without changing the meter,
/// like `*met()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSignatureTandem {
    pub time_signature: TimeSignature,
    /// `yy` visual invisibility marker.
    pub invisible: bool,
}

impl TimeSignatureTandem {
    pub fn visible(time_signature: TimeSignature) -> Self {
        Self {
            time_signature,
            invisible: false,
        }
    }

    pub fn as_kern(&self) -> String {
        let visibility = if self.invisible { "yy" } else { "" };
        format!("{}{visibility}", self.time_signature.as_kern())
    }
}

/// Displayed meter symbol, `*met(...)`.
///
/// The symbol is a glyph; the time signature it implies only governs measure
/// duration when no `*M` is declared alongside it. [`MeterSymbol::Empty`]
/// shows no glyph and implies no time signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeterSymbol {
    /// `*met(c)`
    CommonTime,
    /// `*met(c|)`
    CutTime,
    /// `*met(C)`, `*met(C|)`
    Mensural(MensurationSign),
    /// `*met()`: hides the meter signature.
    Empty,
}

/// Mensural sign inside [`MeterSymbol::Mensural`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MensurationSign {
    /// `C`
    TempusImperfectum,
    /// `C|`
    TempusImperfectumDiminutum,
}

impl MeterSymbol {
    pub fn implied_time_signature(self) -> Option<TimeSignature> {
        match self {
            Self::CommonTime | Self::Mensural(MensurationSign::TempusImperfectum) => {
                Some(TimeSignature {
                    numerator: 4,
                    denominator: 4,
                })
            }
            Self::CutTime | Self::Mensural(MensurationSign::TempusImperfectumDiminutum) => {
                Some(TimeSignature {
                    numerator: 2,
                    denominator: 2,
                })
            }
            Self::Empty => None,
        }
    }

    pub fn as_kern(self) -> &'static str {
        match self {
            Self::CommonTime => "*met(c)",
            Self::CutTime => "*met(c|)",
            Self::Mensural(MensurationSign::TempusImperfectum) => "*met(C)",
            Self::Mensural(MensurationSign::TempusImperfectumDiminutum) => "*met(C|)",
            Self::Empty => "*met()",
        }
    }
}

/// Visual/rendering tandem interpretation.
///
/// These tandems affect score appearance without changing the musical content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualStyle {
    /// RGB colour inherited by split spine descendants, `*color:#RRGGBB`.
    Color([u8; 3]),
    /// Placement policy for down-stems on half notes.
    HalfNoteStemSide(HalfNoteStemSide),
    /// `*above` or `*below`, applying to subsequent contents of the spine.
    StaffPlacement(StaffPlacement),
    /// `*kcancel` or `*Xkcancel`, applying to later key signature changes.
    KeyCancellation(KeyCancellation),
}

/// Whether a key signature change draws naturals that cancel the previous
/// key's accidentals: `*kcancel` turns them on, `*Xkcancel` back off (the
/// default). A change to `*k[]` draws them either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyCancellation {
    pub active: bool,
}

impl KeyCancellation {
    pub fn as_kern(self) -> &'static str {
        if self.active {
            "*kcancel"
        } else {
            "*Xkcancel"
        }
    }
}

/// Side of the staff used for a spine's annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaffPlacement {
    Above,
    Below,
}

/// Side used for down-stems on half-note noteheads.
///
/// `*2\left` is the standard modern placement; `*2\right` requests the
/// alternate right-side placement used by VHV/Verovio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfNoteStemSide {
    Left,
    Right,
}

/// Octave transposition bracket interpretation.
///
/// Examples: `*8va`, `*X8va`, `*8ba`, `*15ma`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ottava {
    pub kind: OttavaKind,
    pub active: bool,
}

/// Direction and size of an [`Ottava`] marking.
///
/// Examples: `*8va` is one octave up, `*15ba` is two octaves down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OttavaKind {
    OctaveUp,
    OctaveDown,
    TwoOctavesUp,
    TwoOctavesDown,
}

/// Sustain pedal tandem interpretation.
///
/// Examples: `*ped`, `*Xped`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pedal {
    /// *ped -> true
    /// *Xped -> false
    pub active: bool,
}

/// Display the enclosed, explicitly encoded music as a repetition symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repetition {
    pub active: bool,
}

/// Tuplet-number display tandem interpretation.
///
/// Examples: `*tuplet`, `*Xtuplet`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tuplet {
    /// *tuplet -> true
    /// *Xtuplet -> false
    pub show_number: bool,
}

/// Tuplet-bracket display tandem interpretation.
///
/// Examples: `*brackettup`, `*Xbrackettup`. Verovio hides the bracket of
/// every tuplet that starts while `*Xbrackettup` is in force; `*brackettup`
/// only lifts that, so a beamed tuplet still draws no bracket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TupletBracket {
    /// *brackettup -> true
    /// *Xbrackettup -> false
    pub show_bracket: bool,
}

/// Cue-size passage toggle: `*cue` starts drawing the layer's notes at cue
/// size, `*Xcue` returns to normal size. Durations are unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cue {
    pub active: bool,
}

impl Cue {
    pub fn as_kern(self) -> &'static str {
        if self.active {
            "*cue"
        } else {
            "*Xcue"
        }
    }
}

/// Visual rhythm scale: `*rscale:N` or `*rscale:N/M` draws every following
/// note of the spine as if its duration (without dots) were multiplied by
/// N/M. Logical durations are unaffected. `*rscale:1` resets.
///
/// Invariant: numerator and denominator are positive and coprime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RhythmScale {
    numerator: u32,
    denominator: u32,
}

impl RhythmScale {
    pub const UNIT: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    pub fn new(numerator: u32, denominator: u32) -> Option<Self> {
        if numerator == 0 || denominator == 0 {
            return None;
        }
        let divisor = gcd(numerator, denominator);
        Some(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub fn numerator(self) -> u32 {
        self.numerator
    }

    pub fn denominator(self) -> u32 {
        self.denominator
    }

    pub fn is_unit(self) -> bool {
        self == Self::UNIT
    }

    pub fn as_kern(self) -> String {
        if self.denominator == 1 {
            format!("*rscale:{}", self.numerator)
        } else {
            format!("*rscale:{}/{}", self.numerator, self.denominator)
        }
    }
}

impl Default for RhythmScale {
    fn default() -> Self {
        Self::UNIT
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Visual bracket-span tandem interpretation.
///
/// Examples: `*lig`, `*Xlig`, `*col`, `*Xcol`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BracketSpan {
    pub kind: BracketSpanKind,
    pub active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BracketSpanKind {
    Ligature,
    Coloration,
}

/// Humdrum `thru` section label tandem interpretation.
///
/// Example: `*>A`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionLabel {
    pub label: String,
}

/// Humdrum `thru` expansion-list tandem interpretation.
///
/// Examples: `*>[A,A,B,B]`, `*>norep[A,B]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpansionList {
    pub version: Option<String>,
    pub items: Vec<ExpansionItem>,
}

/// Item in a Humdrum `thru` expansion list.
///
/// Empty items preserve spellings such as `*>[Minuet,Minuet,Trio,Trio,,Minuet]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpansionItem {
    Label(String),
    Empty,
}

/// Staff line count interpretation used in this repo's OMR dialect.
///
/// Example: `*stria5`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stria {
    pub count: u8,
}

/// Part label tandem interpretation used to group staves/spines.
///
/// Example: `*part1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub number: u32,
}

/// Staff label tandem interpretation used to assign staves/spines.
///
/// Examples: `*staff2`, `*staff1/2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Staff {
    pub number: u32,
    /// The further staves a `*staff1/2` declaration names. Verovio places
    /// `**dynam` content under one between them.
    pub also: Vec<u32>,
}

impl Staff {
    pub fn as_kern(&self) -> String {
        let mut kern = format!("*staff{}", self.number);
        for number in &self.also {
            kern.push_str(&format!("/{number}"));
        }
        kern
    }

    /// Whether the declaration names more than one staff.
    pub fn is_shared(&self) -> bool {
        !self.also.is_empty()
    }
}

/// Group label tandem interpretation used to group staves/spines.
///
/// Examples: `*group1`, `*group2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    pub number: u32,
}

/// Instrument-related tandem interpretation.
///
/// Examples: `*Ivioln`, `*Icello`, `*Iflt`, `*I"Cello`, `*mI"Cello`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instrument {
    pub text: String,
    pub kind: InstrumentKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentKind {
    Code,
    Name(InstrumentTextVariant),
    Abbreviation(InstrumentTextVariant),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentTextVariant {
    Regular,
    Modern,
    Original,
}

/// Exact decimal beats-per-minute value for metronome/tempo interpretations.
///
/// Examples:
/// * `96` -> units 96, scale 0
/// * `22.5` -> units 225, scale 1
/// * `618.05` -> units 61805, scale 2
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecimalTempo {
    pub units: u32,
    pub scale: u8,
}

/// Metronome/tempo tandem interpretation.
///
/// Examples: `*MM96`, `*MM22.5`, `*MM618.05`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tempo {
    pub beats_per_minute: DecimalTempo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tremolo {
    /// *tremolo -> true
    /// *Xtremolo -> false
    pub active: bool,
}
