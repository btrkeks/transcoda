//! What each output dialect does with each kind of tandem interpretation.
//!
//! [`TandemPolicy::action`] is the only place that decides it. The table has
//! one row per state slot ([`StateKey`]) and per stateless tandem kind, so a
//! new slot does not compile until every dialect has chosen. Normalization
//! reads the same table to decide which tandems an output leaves out.

use thiserror::Error;

use crate::{
    score::{Outcome, Score, StateErrors, StateKey},
    KernLine, MeterOutput, TandemInterpretation,
};

/// One dialect's decision for one kind of tandem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TandemAction {
    /// Written as in the source.
    Keep,
    /// Left out; the output does not describe it.
    Drop,
    /// Left out as written; the serializer applies its effect to the notes it
    /// governs.
    Fold,
    /// The output cannot describe the slot. A score where it leaves its
    /// default fails to serialize; tandems that keep the default are dropped.
    Reject,
}

/// A kind of tandem interpretation as output policies see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TandemKind {
    State(StateKey),
    /// `*MX` sets the meter slot but draws no time signature.
    FreeMeter,
    TupletBreak,
    SectionLabel,
    ExpansionList,
    Part,
    Staff,
    Group,
    Instrument,
    Tempo,
    Raw,
}

impl TandemKind {
    /// Every kind that is not a state slot.
    pub const STATELESS: [Self; 10] = [
        Self::FreeMeter,
        Self::TupletBreak,
        Self::SectionLabel,
        Self::ExpansionList,
        Self::Part,
        Self::Staff,
        Self::Group,
        Self::Instrument,
        Self::Tempo,
        Self::Raw,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::State(key) => key.name(),
            Self::FreeMeter => "`*MX` free meter",
            Self::TupletBreak => "`*tupbreak` tuplet break",
            Self::SectionLabel => "`*>A` section label",
            Self::ExpansionList => "`*>[A,B]` expansion list",
            Self::Part => "`*part`",
            Self::Staff => "`*staff`",
            Self::Group => "`*group`",
            Self::Instrument => "`*I` instrument",
            Self::Tempo => "`*MM` tempo",
            Self::Raw => "other tandems",
        }
    }

    /// `None` for the null tandem `*`.
    pub fn of(tandem: &TandemInterpretation) -> Option<Self> {
        if let Some(key) = StateKey::of(tandem) {
            if matches!(tandem, TandemInterpretation::FreeMeter) {
                return Some(Self::FreeMeter);
            }
            return Some(Self::State(key));
        }
        Some(match tandem {
            TandemInterpretation::Null => return None,
            TandemInterpretation::TupletBreak => Self::TupletBreak,
            TandemInterpretation::SectionLabel(_) => Self::SectionLabel,
            TandemInterpretation::ExpansionList(_) => Self::ExpansionList,
            TandemInterpretation::Part(_) => Self::Part,
            TandemInterpretation::Staff(_) => Self::Staff,
            TandemInterpretation::Group(_) => Self::Group,
            TandemInterpretation::Instrument(_) => Self::Instrument,
            TandemInterpretation::Tempo(_) => Self::Tempo,
            TandemInterpretation::Raw(_) => Self::Raw,
            _ => unreachable!("state tandem handled above: {tandem:?}"),
        })
    }
}

/// The tandem policy of one output dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TandemPolicy {
    /// Standard `**kern` and Verovio input: the HumdrumKern and KernOnly
    /// presets and render output.
    HumdrumKern,

    /// Transcoda V2 and V2 training labels.
    TranscodaV2,

    /// Structural OMR normalization (`normalize_for_omr`, the dataset
    /// `normalize` stage) and the generic OMR serializer.
    #[default]
    Omr,
}

impl TandemPolicy {
    pub fn action(self, kind: TandemKind) -> TandemAction {
        use TandemAction::{Drop, Keep, Reject};
        // Columns: HumdrumKern, TranscodaV2, Omr.
        let row: [TandemAction; 3] = match kind {
            TandemKind::State(key) => match key {
                StateKey::Clef => [Keep, Keep, Keep],
                StateKey::KeySignature => [Keep, Keep, Keep],
                StateKey::KeyDesignation => [Drop, Drop, Drop],
                StateKey::Meter => [Keep, Keep, Keep],
                StateKey::StaffLines => [Keep, Keep, Keep],
                StateKey::Ottava => [Keep, Keep, Keep],
                StateKey::Pedal => [Keep, Drop, Keep],
                StateKey::TupletNumbers => [Keep, Drop, Keep],
                StateKey::TupletBrackets => [Keep, Drop, Keep],
                StateKey::Tremolo => [Keep, Keep, Keep],
                StateKey::Repetition => [Keep, Drop, Keep],
                StateKey::Ligature => [Keep, Drop, Keep],
                StateKey::Coloration => [Keep, Drop, Keep],
                StateKey::HalfNoteStemSide => [Keep, Drop, Keep],
                StateKey::StaffPlacement => [Keep, Drop, Keep],
                // Labels keep it so they explain the naturals the rendered
                // image draws; the frozen V1 target does not change.
                StateKey::KeyCancellation => [Keep, Keep, Keep],
                StateKey::Color => [Keep, Drop, Keep],
                // D7: semantic labels cannot say which durations are drawn
                // scaled; visual labels write them as drawn. V2 labels drop a
                // rhythm scale and keep the logical durations (the beam
                // markers still show the drawn depth). V1 and dataset
                // normalization still refuse it, so synthetic sources never
                // carry one. V2 labels also drop cue size for now: the
                // Transcoda V2 tokenizer cannot spell `*cue` (no `u`); it
                // returns with the next tokenizer.
                StateKey::Cue => [Keep, Drop, Keep],
                StateKey::RhythmScale => [Keep, Drop, Reject],
            },
            TandemKind::FreeMeter => [Keep, Drop, Drop],
            TandemKind::TupletBreak => [Keep, Drop, Keep],
            TandemKind::SectionLabel => [Keep, Drop, Keep],
            TandemKind::ExpansionList => [Keep, Drop, Keep],
            // Only the `*staff` Verovio numbers staves from: normalization
            // and render output remove the rest first. Without them a
            // left-to-right numbering would flip the staves.
            TandemKind::Staff => [Keep, Drop, Drop],
            TandemKind::Part | TandemKind::Group => [Drop, Drop, Drop],
            TandemKind::Instrument | TandemKind::Tempo | TandemKind::Raw => [Keep, Drop, Drop],
        };
        row[self as usize]
    }

    /// How normalization for this output spells meter declarations. The
    /// Transcoda labels keep only meters the page prints.
    pub fn meter_output(self) -> MeterOutput {
        match self {
            Self::TranscodaV2 => MeterOutput::Printed,
            Self::HumdrumKern | Self::Omr => MeterOutput::SingleSlot,
        }
    }

    /// The action for one tandem token; the null tandem is kept.
    pub fn action_for(self, tandem: &TandemInterpretation) -> TandemAction {
        TandemKind::of(tandem).map_or(TandemAction::Keep, |kind| self.action(kind))
    }

    /// Whether output under this policy writes the tandem as it stands.
    pub fn writes(self, tandem: &TandemInterpretation) -> bool {
        self.action_for(tandem) == TandemAction::Keep
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::HumdrumKern => "HumdrumKern",

            Self::TranscodaV2 => "Transcoda V2",

            Self::Omr => "OMR",
        }
    }

    /// Fail on the first tandem that moves a slot this policy rejects off
    /// its default.
    pub fn check(self, score: &Score<'_>) -> Result<(), DialectError> {
        let document = score.document();
        for event in score.events() {
            if event.outcome != Outcome::Changed
                || self.action(TandemKind::State(event.key)) != TandemAction::Reject
                || score
                    .state_after(event.line_index, event.column)
                    .holds_default(event.key)
            {
                continue;
            }
            let KernLine::TandemInterpretationLine(tandems) = &document.lines()[event.line_index]
            else {
                unreachable!("state events come from tandem lines")
            };
            return Err(DialectError::Unexpressible {
                line: document.source_line_number(event.line_index),
                column: event.column + 1,
                written: crate::serialize::omr::spell_tandem(
                    &tandems.tokens()[event.column],
                    crate::serialize::omr::VisibilityMarkerStyle::Preserve,
                )
                .unwrap_or_default(),
                key: event.key,
                policy: self,
            });
        }
        Ok(())
    }
}

/// Why a document cannot be written in a dialect.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DialectError {
    #[error("the score does not resolve: {0}")]
    State(#[from] StateErrors),
    #[error(
        "line {line} column {column}: {} output cannot express `{written}`; its {} must stay at the default",
        .policy.name(), .key.name()
    )]
    Unexpressible {
        line: usize,
        column: usize,
        written: String,
        key: StateKey,
        policy: TandemPolicy,
    },
}

impl From<DialectError> for crate::document::DocumentError {
    fn from(error: DialectError) -> Self {
        match &error {
            DialectError::State(_) => Self::new("serialize.state", None, error.to_string()),
            DialectError::Unexpressible { line, .. } => {
                Self::new("serialize.unexpressible", Some(*line), error.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICIES: [TandemPolicy; 3] = [
        TandemPolicy::HumdrumKern,
        TandemPolicy::TranscodaV2,
        TandemPolicy::Omr,
    ];

    #[test]
    fn no_policy_keeps_a_tandem_without_a_spelling() {
        for policy in POLICIES {
            for kind in [
                TandemKind::State(StateKey::KeyDesignation),
                TandemKind::Part,
                TandemKind::Group,
            ] {
                assert_ne!(
                    policy.action(kind),
                    TandemAction::Keep,
                    "{policy:?} {kind:?}"
                );
            }
        }
    }

    #[test]
    fn free_meter_is_its_own_kind() {
        assert_eq!(
            TandemKind::of(&TandemInterpretation::FreeMeter),
            Some(TandemKind::FreeMeter)
        );
        assert_eq!(TandemKind::of(&TandemInterpretation::Null), None);
    }
}
