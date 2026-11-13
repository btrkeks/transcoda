//! Meter state carried by one spine.
//!
//! Humdrum declares meter with two complementary tandems: the numeric time
//! signature (`*M3/4`) and the displayed meter symbol (`*met(c)`). A symbol on
//! its own implies a time signature, so OMR transcriptions that only carry the
//! printed sign still have a measure duration.
//!
//! Tandems on consecutive tandem interpretation lines form one declaration
//! group. Within a group, `*M` is authoritative for duration and `*met` is the
//! displayed glyph. Any other line ends the group, and the next meter tandem
//! starts a new declaration that replaces the previous one entirely.
//!
//! `*MX` declares irregular meter: bars have no fixed length until a later
//! group declares one.

use crate::{
    MeterSymbol, MusicalDuration, TandemInterpretation, TimeSignature, TimeSignatureTandem,
};

/// The meter a spine currently declares.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MeterDeclaration {
    /// `*M3/4`, `*M3/4yy` or `*MX`.
    meter: Option<DeclaredMeter>,
    symbol: Option<MeterSymbol>,
    /// The meter in force before this group. It governs a group that declares
    /// no meter of its own, such as a lone `*met()`.
    inherited: Option<Meter>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeclaredMeter {
    TimeSignature(TimeSignatureTandem),
    Free,
}

/// The meter governing bar lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Meter {
    /// Bars last as long as the time signature.
    Regular(TimeSignature),
    /// `*MX`: bars have no fixed length.
    Free,
}

/// How long the bars of a meter are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarLength {
    Fixed(MusicalDuration),
    Free,
}

impl Meter {
    pub fn bar_length(self) -> BarLength {
        match self {
            Self::Regular(time_signature) => {
                BarLength::Fixed(MusicalDuration::from_time_signature(&time_signature))
            }
            Self::Free => BarLength::Free,
        }
    }
}

impl BarLength {
    pub fn fixed(self) -> Option<MusicalDuration> {
        match self {
            Self::Fixed(duration) => Some(duration),
            Self::Free => None,
        }
    }
}

/// What the score prints for a meter declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeterDisplay {
    /// The numeric time signature.
    Numeric,
    /// A meter symbol in place of the numeric time signature.
    Symbol(MeterSymbol),
    /// Nothing: `*met()`, a `yy` time signature, or `*MX` without a symbol.
    Hidden,
}

/// What a completed declaration group does to the meter in force.
///
/// Hiding belongs to the declaring event, not to the meter: a hidden group
/// that keeps the bar length changes nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeterChange {
    /// A new bar length, or a new sign for it.
    Changed,
    /// The meter in force, printed again.
    Restated,
    /// The meter in force, printing nothing.
    Silent,
}

/// A declaration group that declares a time signature and a meter symbol
/// implying different time signatures, such as `*M3/4` with `*met(c)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeterConflict {
    pub time_signature: TimeSignature,
    pub symbol_time_signature: TimeSignature,
}

/// How a target represents a meter declaration as tandem interpretations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeterOutput {
    /// Emit the declared `*M` and `*met` tandems. A group that declares no
    /// meter of its own also re-declares the meter it inherited, so the
    /// emitted group stands alone.
    AsDeclared,
    /// Emit one tandem: a visible symbol when declared, otherwise the time
    /// signature, spelled `*M3/4yy` when hidden.
    ///
    /// Dataset normalization and the generic OMR serializer: a hidden meter
    /// keeps its bar length in one token.
    SingleSlot,
    /// Emit the effective time signature, followed by a visible symbol, or by
    /// `*met()` when hidden.
    ///
    /// Verovio needs an explicit `*M` to compute measure durations and keeps
    /// displaying the symbol instead of the numeric signature. It honors `yy`
    /// on `*M` only for the first meter of a staff, so a hidden meter change
    /// must be spelled with `*met()`.
    #[default]
    ExplicitTimeSignature,
    /// Emit only what the page prints: a visible symbol, otherwise the time
    /// signature. A hidden meter, and `*MX` without a symbol, emit nothing.
    ///
    /// Transcoda labels transcribe the printed page, and a meter that prints
    /// nothing cannot be read from it.
    Printed,
}

impl MeterDeclaration {
    pub fn time_signature(&self) -> Option<TimeSignature> {
        match self.meter? {
            DeclaredMeter::TimeSignature(time_signature) => Some(time_signature.time_signature),
            DeclaredMeter::Free => None,
        }
    }

    pub fn symbol(&self) -> Option<MeterSymbol> {
        self.symbol
    }

    pub fn is_declared(&self) -> bool {
        self.meter.is_some() || self.symbol.is_some()
    }

    pub fn display(&self) -> MeterDisplay {
        if let Some(DeclaredMeter::TimeSignature(time_signature)) = self.meter {
            if time_signature.invisible {
                return MeterDisplay::Hidden;
            }
        }
        match (self.symbol, self.meter) {
            (Some(MeterSymbol::Empty), _) => MeterDisplay::Hidden,
            (Some(symbol), _) => MeterDisplay::Symbol(symbol),
            (None, Some(DeclaredMeter::Free)) => MeterDisplay::Hidden,
            (None, _) => MeterDisplay::Numeric,
        }
    }

    /// The meter governing bar lengths: the declared `*M` or `*MX`, otherwise
    /// the time signature implied by the symbol, otherwise the meter inherited
    /// from before this group.
    pub fn effective_meter(&self) -> Option<Meter> {
        match self.meter {
            Some(DeclaredMeter::TimeSignature(time_signature)) => {
                Some(Meter::Regular(time_signature.time_signature))
            }
            Some(DeclaredMeter::Free) => Some(Meter::Free),
            None => self
                .symbol
                .and_then(MeterSymbol::implied_time_signature)
                .map(Meter::Regular)
                .or(self.inherited),
        }
    }

    pub fn effective_time_signature(&self) -> Option<TimeSignature> {
        match self.effective_meter()? {
            Meter::Regular(time_signature) => Some(time_signature),
            Meter::Free => None,
        }
    }

    pub fn bar_length(&self) -> Option<BarLength> {
        self.effective_meter().map(Meter::bar_length)
    }

    /// The fixed bar length, or `None` when it is unknown or free (`*MX`).
    pub fn measure_duration(&self) -> Option<MusicalDuration> {
        self.bar_length().and_then(BarLength::fixed)
    }

    pub fn conflict(&self) -> Option<MeterConflict> {
        let time_signature = self.time_signature()?;
        let symbol_time_signature = self.symbol?.implied_time_signature()?;
        (time_signature != symbol_time_signature).then_some(MeterConflict {
            time_signature,
            symbol_time_signature,
        })
    }

    /// Whether both declarations put the same meter in force: the same bar
    /// length, written with the same sign. Whether the declaring group printed
    /// it is not part of the meter.
    pub fn has_same_meter(&self, other: &Self) -> bool {
        self.effective_meter() == other.effective_meter() && self.sign() == other.sign()
    }

    /// What this completed group does to the meter `before` declared.
    pub fn change_from(&self, before: &Self) -> MeterChange {
        if self.display() == MeterDisplay::Hidden
            && self.effective_meter() == before.effective_meter()
        {
            MeterChange::Silent
        } else if self.has_same_meter(before) {
            MeterChange::Restated
        } else {
            MeterChange::Changed
        }
    }

    /// The symbol a restatement of this meter prints, or `None` for the
    /// numeric time signature.
    fn sign(&self) -> Option<MeterSymbol> {
        self.symbol.filter(|symbol| *symbol != MeterSymbol::Empty)
    }

    /// Tandems that re-declare this meter as one group: the time-signature
    /// slot, then the meter-symbol slot. Targets emit them on two aligned
    /// lines in that order.
    pub fn tandems(&self, output: MeterOutput) -> [Option<TandemInterpretation>; 2] {
        let (meter, symbol) = match output {
            MeterOutput::AsDeclared => self.declared_tandems(),
            MeterOutput::SingleSlot | MeterOutput::ExplicitTimeSignature => {
                self.displayed_tandems(output)
            }
            MeterOutput::Printed => self.printed_tandems(),
        };
        [
            meter.map(|meter| match meter {
                DeclaredMeter::TimeSignature(time_signature) => {
                    TandemInterpretation::TimeSignature(time_signature)
                }
                DeclaredMeter::Free => TandemInterpretation::FreeMeter,
            }),
            symbol.map(TandemInterpretation::MeterSymbol),
        ]
    }

    fn declared_tandems(&self) -> (Option<DeclaredMeter>, Option<MeterSymbol>) {
        let declares_meter = self.meter.is_some()
            || self
                .symbol
                .is_some_and(|symbol| symbol.implied_time_signature().is_some());
        let meter = if declares_meter {
            self.meter
        } else {
            self.inherited.map(|meter| match meter {
                Meter::Regular(time_signature) => {
                    DeclaredMeter::TimeSignature(TimeSignatureTandem::visible(time_signature))
                }
                Meter::Free => DeclaredMeter::Free,
            })
        };
        (meter, self.symbol)
    }

    fn printed_tandems(&self) -> (Option<DeclaredMeter>, Option<MeterSymbol>) {
        match self.display() {
            MeterDisplay::Hidden => (None, None),
            MeterDisplay::Symbol(symbol) => (None, Some(symbol)),
            MeterDisplay::Numeric => (
                self.time_signature().map(|time_signature| {
                    DeclaredMeter::TimeSignature(TimeSignatureTandem::visible(time_signature))
                }),
                None,
            ),
        }
    }

    /// One spelling per display. [`MeterOutput::SingleSlot`] hides a meter
    /// with a `yy` time signature; [`MeterOutput::ExplicitTimeSignature`] and
    /// a hidden meter without a known time signature use `*met()`. Free meter
    /// is `*MX` followed by any visible symbol.
    fn displayed_tandems(
        &self,
        output: MeterOutput,
    ) -> (Option<DeclaredMeter>, Option<MeterSymbol>) {
        let display = self.display();
        let visible_symbol = match display {
            MeterDisplay::Symbol(symbol) => Some(symbol),
            MeterDisplay::Hidden | MeterDisplay::Numeric => None,
        };
        if self.effective_meter() == Some(Meter::Free) {
            return (Some(DeclaredMeter::Free), visible_symbol);
        }
        let hidden = display == MeterDisplay::Hidden;
        let time_signature = match (output, display) {
            (MeterOutput::SingleSlot, MeterDisplay::Symbol(_)) => None,
            _ => self.effective_time_signature(),
        }
        .map(|time_signature| TimeSignatureTandem {
            time_signature,
            invisible: hidden && output == MeterOutput::SingleSlot,
        });
        let symbol = match display {
            MeterDisplay::Hidden
                if time_signature.is_none_or(|time_signature| !time_signature.invisible) =>
            {
                Some(MeterSymbol::Empty)
            }
            _ => visible_symbol,
        };
        (time_signature.map(DeclaredMeter::TimeSignature), symbol)
    }
}

/// Columns of [`MeterDeclaration::tandems`], one entry per spine, split into
/// the time-signature line and the meter-symbol line.
pub fn meter_lines<'a>(
    declarations: impl IntoIterator<Item = Option<&'a MeterDeclaration>>,
    output: MeterOutput,
) -> [Vec<Option<TandemInterpretation>>; 2] {
    let mut lines = [Vec::new(), Vec::new()];
    for declaration in declarations {
        let [time_signature, symbol] = declaration
            .map(|declaration| declaration.tandems(output))
            .unwrap_or_default();
        lines[0].push(time_signature);
        lines[1].push(symbol);
    }
    lines
}

/// Whether the tandem declares meter: a time signature, `*MX` or a meter
/// symbol.
pub fn is_meter_tandem(tandem: &TandemInterpretation) -> bool {
    matches!(
        tandem,
        TandemInterpretation::TimeSignature(_)
            | TandemInterpretation::FreeMeter
            | TandemInterpretation::MeterSymbol(_)
    )
}

/// The time signature one meter tandem implies in isolation.
///
/// This ignores the rest of its declaration group; use [`ActiveMeter`] to
/// track the meter in force.
pub fn implied_time_signature(tandem: &TandemInterpretation) -> Option<TimeSignature> {
    match tandem {
        TandemInterpretation::TimeSignature(time_signature) => Some(time_signature.time_signature),
        TandemInterpretation::MeterSymbol(symbol) => symbol.implied_time_signature(),
        _ => None,
    }
}

/// Meter declaration state for one spine, folded tandem by tandem.
///
/// Feed every meter tandem to [`ActiveMeter::apply`] and call
/// [`ActiveMeter::end_group`] for every line that is not a tandem
/// interpretation line.
#[derive(Debug, Clone, Default)]
pub struct ActiveMeter {
    declaration: MeterDeclaration,
    group_open: bool,
}

impl ActiveMeter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn declaration(&self) -> &MeterDeclaration {
        &self.declaration
    }

    /// Fold one tandem into the state, returning the updated declaration when
    /// the tandem is a meter tandem.
    pub fn apply(&mut self, tandem: &TandemInterpretation) -> Option<&MeterDeclaration> {
        if !is_meter_tandem(tandem) {
            return None;
        }
        if !self.group_open {
            self.declaration = MeterDeclaration {
                inherited: self.declaration.effective_meter(),
                ..MeterDeclaration::default()
            };
            self.group_open = true;
        }
        match *tandem {
            TandemInterpretation::TimeSignature(time_signature) => {
                self.declaration.meter = Some(DeclaredMeter::TimeSignature(time_signature));
            }
            TandemInterpretation::FreeMeter => self.declaration.meter = Some(DeclaredMeter::Free),
            TandemInterpretation::MeterSymbol(symbol) => self.declaration.symbol = Some(symbol),
            _ => unreachable!("checked by is_meter_tandem"),
        }
        Some(&self.declaration)
    }

    pub fn end_group(&mut self) {
        self.group_open = false;
    }
}

/// Equality ignores whether a declaration group is still open.
impl PartialEq for ActiveMeter {
    fn eq(&self, other: &Self) -> bool {
        self.declaration == other.declaration
    }
}

impl Eq for ActiveMeter {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MensurationSign;

    const COMMON: MeterSymbol = MeterSymbol::CommonTime;
    const CUT: MeterSymbol = MeterSymbol::CutTime;

    fn time_signature(numerator: u32, denominator: u32) -> TimeSignature {
        TimeSignature {
            numerator,
            denominator,
        }
    }

    fn numeric(numerator: u32, denominator: u32) -> TandemInterpretation {
        TandemInterpretation::TimeSignature(TimeSignatureTandem::visible(time_signature(
            numerator,
            denominator,
        )))
    }

    fn hidden_numeric(numerator: u32, denominator: u32) -> TandemInterpretation {
        TandemInterpretation::TimeSignature(TimeSignatureTandem {
            time_signature: time_signature(numerator, denominator),
            invisible: true,
        })
    }

    const EMPTY: MeterSymbol = MeterSymbol::Empty;

    fn symbol(symbol: MeterSymbol) -> TandemInterpretation {
        TandemInterpretation::MeterSymbol(symbol)
    }

    fn fold(groups: &[&[TandemInterpretation]]) -> ActiveMeter {
        let mut meter = ActiveMeter::new();
        for group in groups {
            for tandem in *group {
                meter.apply(tandem);
            }
            meter.end_group();
        }
        meter
    }

    #[test]
    fn symbol_alone_implies_time_signature() {
        let meter = fold(&[&[symbol(CUT)]]);

        assert_eq!(
            meter.declaration().effective_time_signature(),
            Some(time_signature(2, 2))
        );
    }

    #[test]
    fn numeric_meter_wins_within_a_group_regardless_of_order() {
        for group in [
            [numeric(3, 4), symbol(COMMON)],
            [symbol(COMMON), numeric(3, 4)],
        ] {
            let meter = fold(&[&group]);

            assert_eq!(
                meter.declaration().effective_time_signature(),
                Some(time_signature(3, 4))
            );
            assert_eq!(
                meter.declaration().conflict(),
                Some(MeterConflict {
                    time_signature: time_signature(3, 4),
                    symbol_time_signature: time_signature(4, 4),
                })
            );
        }
    }

    #[test]
    fn mensural_signs_imply_their_modern_equivalents() {
        let tempus = fold(&[&[symbol(MeterSymbol::Mensural(
            MensurationSign::TempusImperfectumDiminutum,
        ))]]);

        assert_eq!(
            tempus.declaration().effective_time_signature(),
            Some(time_signature(2, 2))
        );
    }

    #[test]
    fn consistent_group_has_no_conflict() {
        let meter = fold(&[&[numeric(2, 2), symbol(CUT)]]);

        assert_eq!(meter.declaration().conflict(), None);
    }

    #[test]
    fn later_group_replaces_the_whole_declaration() {
        let symbol_only = fold(&[&[numeric(4, 4), symbol(COMMON)], &[symbol(CUT)]]);
        let numeric_only = fold(&[&[symbol(COMMON)], &[numeric(4, 4)]]);

        assert_eq!(symbol_only.declaration().time_signature(), None);
        assert_eq!(
            symbol_only.declaration().effective_time_signature(),
            Some(time_signature(2, 2))
        );
        assert_eq!(numeric_only.declaration().symbol(), None);
    }

    #[test]
    fn non_meter_tandems_do_not_open_a_group() {
        let mut meter = fold(&[&[numeric(3, 4)]]);

        assert!(meter.apply(&TandemInterpretation::Null).is_none());
        meter.end_group();

        assert_eq!(
            meter.declaration().time_signature(),
            Some(time_signature(3, 4))
        );
    }

    #[test]
    fn output_policies() {
        let both = fold(&[&[numeric(4, 4), symbol(COMMON)]]);
        let symbol_only = fold(&[&[symbol(COMMON)]]);

        assert_eq!(
            both.declaration().tandems(MeterOutput::AsDeclared),
            [Some(numeric(4, 4)), Some(symbol(COMMON))]
        );
        assert_eq!(
            both.declaration().tandems(MeterOutput::SingleSlot),
            [None, Some(symbol(COMMON))]
        );
        assert_eq!(
            symbol_only
                .declaration()
                .tandems(MeterOutput::ExplicitTimeSignature),
            [Some(numeric(4, 4)), Some(symbol(COMMON))]
        );
        assert_eq!(
            symbol_only.declaration().tandems(MeterOutput::AsDeclared),
            [None, Some(symbol(COMMON))]
        );
    }

    #[test]
    fn same_meter_ignores_redundant_numeric_meter() {
        let symbol_only = fold(&[&[symbol(COMMON)]]);
        let both = fold(&[&[numeric(4, 4), symbol(COMMON)]]);
        let numeric_only = fold(&[&[numeric(4, 4)]]);

        assert!(symbol_only.declaration().has_same_meter(both.declaration()));
        assert!(!numeric_only
            .declaration()
            .has_same_meter(both.declaration()));
    }

    #[test]
    fn empty_symbol_and_yy_time_signature_both_hide_the_meter() {
        for group in [
            vec![numeric(3, 4), symbol(EMPTY)],
            vec![symbol(EMPTY), numeric(3, 4)],
            vec![hidden_numeric(3, 4)],
            vec![hidden_numeric(3, 4), symbol(COMMON)],
        ] {
            let meter = fold(&[&group]);

            assert_eq!(meter.declaration().display(), MeterDisplay::Hidden);
            assert_eq!(
                meter.declaration().effective_time_signature(),
                Some(time_signature(3, 4))
            );
        }
    }

    #[test]
    fn hide_only_group_keeps_the_time_signature_in_force() {
        let meter = fold(&[&[numeric(3, 1), symbol(COMMON)], &[symbol(EMPTY)]]);

        assert_eq!(meter.declaration().time_signature(), None);
        assert_eq!(meter.declaration().display(), MeterDisplay::Hidden);
        assert_eq!(
            meter.declaration().effective_time_signature(),
            Some(time_signature(3, 1))
        );
        assert_eq!(
            meter.declaration().tandems(MeterOutput::AsDeclared),
            [Some(numeric(3, 1)), Some(symbol(EMPTY))]
        );
    }

    #[test]
    fn hiding_is_not_part_of_the_meter() {
        let shown = fold(&[&[numeric(3, 4)]]);
        let hidden = fold(&[&[numeric(3, 4), symbol(EMPTY)]]);

        assert!(shown.declaration().has_same_meter(hidden.declaration()));
    }

    #[test]
    fn a_group_changes_the_meter_unless_it_keeps_the_bar_length_hidden() {
        let shown = fold(&[&[numeric(3, 4)]]);
        let common = fold(&[&[symbol(COMMON)]]);
        let free = fold(&[&[FREE]]);
        let change = |groups: &[&[TandemInterpretation]], before: &ActiveMeter| {
            fold(groups).declaration().change_from(before.declaration())
        };

        assert_eq!(
            change(&[&[numeric(3, 4), symbol(EMPTY)]], &shown),
            MeterChange::Silent
        );
        assert_eq!(
            change(&[&[symbol(COMMON)], &[symbol(EMPTY)]], &common),
            MeterChange::Silent
        );
        assert_eq!(
            change(&[&[hidden_numeric(3, 4)]], &shown),
            MeterChange::Silent
        );
        assert_eq!(change(&[&[numeric(3, 4)]], &shown), MeterChange::Restated);
        assert_eq!(
            change(&[&[numeric(4, 4), symbol(EMPTY)]], &shown),
            MeterChange::Changed
        );
        assert_eq!(
            change(&[&[numeric(2, 4), symbol(EMPTY)]], &free),
            MeterChange::Changed
        );
        assert_eq!(change(&[&[numeric(4, 4)]], &common), MeterChange::Changed);
    }

    #[test]
    fn empty_symbol_never_conflicts() {
        let meter = fold(&[&[numeric(3, 4), symbol(EMPTY)]]);

        assert_eq!(meter.declaration().conflict(), None);
    }

    #[test]
    fn single_slot_spells_a_hidden_meter_as_a_yy_time_signature() {
        let hidden = fold(&[&[symbol(COMMON)], &[symbol(EMPTY)]]);

        assert_eq!(
            hidden.declaration().tandems(MeterOutput::SingleSlot),
            [Some(hidden_numeric(4, 4)), None]
        );
    }

    #[test]
    fn explicit_time_signature_spells_a_hidden_meter_with_an_empty_symbol() {
        for groups in [
            vec![vec![symbol(COMMON)], vec![symbol(EMPTY)]],
            vec![vec![hidden_numeric(4, 4)]],
        ] {
            let groups = groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let hidden = fold(&groups);

            assert_eq!(
                hidden
                    .declaration()
                    .tandems(MeterOutput::ExplicitTimeSignature),
                [Some(numeric(4, 4)), Some(symbol(EMPTY))]
            );
        }
    }

    #[test]
    fn hidden_meter_without_known_time_signature_stays_empty_symbol() {
        let hidden = fold(&[&[symbol(EMPTY)]]);

        assert_eq!(hidden.declaration().effective_time_signature(), None);
        assert_eq!(
            hidden
                .declaration()
                .tandems(MeterOutput::ExplicitTimeSignature),
            [None, Some(symbol(EMPTY))]
        );
    }

    const FREE: TandemInterpretation = TandemInterpretation::FreeMeter;

    #[test]
    fn free_meter_has_no_bar_length_until_a_later_time_signature() {
        let free = fold(&[&[numeric(4, 4)], &[FREE]]);
        let restored = fold(&[&[FREE], &[numeric(4, 4), symbol(EMPTY)]]);

        assert_eq!(free.declaration().effective_meter(), Some(Meter::Free));
        assert_eq!(free.declaration().bar_length(), Some(BarLength::Free));
        assert_eq!(free.declaration().measure_duration(), None);
        assert_eq!(free.declaration().effective_time_signature(), None);
        assert_eq!(
            restored.declaration().effective_meter(),
            Some(Meter::Regular(time_signature(4, 4)))
        );
        assert_eq!(restored.declaration().display(), MeterDisplay::Hidden);
    }

    #[test]
    fn free_meter_wins_over_the_symbol_in_its_group() {
        let meter = fold(&[&[FREE, symbol(COMMON)]]);

        assert_eq!(meter.declaration().effective_meter(), Some(Meter::Free));
        assert_eq!(meter.declaration().display(), MeterDisplay::Symbol(COMMON));
    }

    #[test]
    fn hide_only_group_after_free_meter_stays_free() {
        let meter = fold(&[&[FREE], &[symbol(EMPTY)]]);

        assert_eq!(meter.declaration().effective_meter(), Some(Meter::Free));
        assert_eq!(
            meter.declaration().tandems(MeterOutput::AsDeclared),
            [Some(FREE), Some(symbol(EMPTY))]
        );
    }

    #[test]
    fn every_output_spells_free_meter_as_mx() {
        let free = fold(&[&[FREE]]);
        let free_with_symbol = fold(&[&[FREE, symbol(COMMON)]]);

        for output in [
            MeterOutput::AsDeclared,
            MeterOutput::SingleSlot,
            MeterOutput::ExplicitTimeSignature,
        ] {
            assert_eq!(free.declaration().tandems(output), [Some(FREE), None]);
            assert_eq!(
                free_with_symbol.declaration().tandems(output),
                [Some(FREE), Some(symbol(COMMON))]
            );
        }
    }

    #[test]
    fn free_and_regular_meter_differ_in_effect() {
        let free = fold(&[&[FREE]]);
        let regular = fold(&[&[numeric(4, 4)]]);

        assert!(!free.declaration().has_same_meter(regular.declaration()));
    }

    #[test]
    fn printed_output_emits_only_what_the_page_prints() {
        let cases: [(&[&[TandemInterpretation]], _); 7] = [
            (&[&[numeric(3, 4)]], [Some(numeric(3, 4)), None]),
            (
                &[&[numeric(4, 4), symbol(COMMON)]],
                [None, Some(symbol(COMMON))],
            ),
            (&[&[numeric(4, 4), symbol(EMPTY)]], [None, None]),
            (&[&[hidden_numeric(4, 4)]], [None, None]),
            (&[&[numeric(3, 4)], &[symbol(EMPTY)]], [None, None]),
            (&[&[FREE]], [None, None]),
            (&[&[FREE, symbol(COMMON)]], [None, Some(symbol(COMMON))]),
        ];
        for (groups, printed) in cases {
            assert_eq!(
                fold(groups).declaration().tandems(MeterOutput::Printed),
                printed,
                "{groups:?}"
            );
        }
    }
}
