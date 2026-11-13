//! Canonicalization for initial pre-content tandem interpretation state.
//!
//! A leading barline before any data does not create visible musical content,
//! so tandem interpretations on both sides of it are one logical initial
//! header. This pass owns the singleton header-state invariant for clefs, key
//! signatures, and meter declarations.
//!
//! Meter follows [`ActiveMeter`] groups: a barline or comment between two meter
//! tandems ends a declaration group, so the later group replaces the earlier
//! one.

use crate::{
    initial_state::{
        default_clef, default_key_signature, initial_header_state_range, initial_spine_types,
    },
    meter::meter_lines,
    score::StateKey,
    ActiveMeter, Clef, KernDocument, KernLine, KeySignature, MeterOutput, SpineType,
    TandemInterpretation, TandemInterpretationLine,
};

use super::{NormalizationPass, NormalizeOptions};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizeInitialHeader {
    options: NormalizeOptions,
    meter_output: MeterOutput,
}

impl CanonicalizeInitialHeader {
    pub fn new(options: NormalizeOptions, meter_output: MeterOutput) -> Self {
        Self {
            options,
            meter_output,
        }
    }
}

impl NormalizationPass for CanonicalizeInitialHeader {
    fn normalize(&self, document: &mut KernDocument) {
        let range = initial_header_state_range(document);
        let initial_spines = initial_spine_types(document);
        if initial_spines.is_empty() {
            return;
        }

        let header_lines = range
            .clone()
            .map(|index| (document.lines()[index].clone(), document.source(index)))
            .collect::<Vec<_>>();
        let header = InitialHeader::collect(&header_lines, &initial_spines);
        document.splice_lines(range, header.into_lines(self.meter_output));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InitialHeader {
    spines: Vec<HeaderSpineState>,
    passthrough: Vec<(KernLine, Option<usize>)>,
}

impl InitialHeader {
    fn collect(lines: &[(KernLine, Option<usize>)], initial_spines: &[SpineType]) -> Self {
        let mut header = Self {
            spines: initial_spines
                .iter()
                .map(|spine_type| HeaderSpineState {
                    is_kern: matches!(spine_type, SpineType::Kern),
                    ..Default::default()
                })
                .collect(),
            passthrough: Vec::new(),
        };

        for (line, source) in lines {
            if !matches!(line, KernLine::TandemInterpretationLine(_)) {
                for state in &mut header.spines {
                    state.meter.end_group();
                }
            }
            match line {
                KernLine::TandemInterpretationLine(tandems) => {
                    if is_singleton_header_line(tandems) {
                        header.apply_singleton_tandems(tandems, *source);
                    } else if !tandems
                        .tokens()
                        .iter()
                        .all(|tandem| matches!(tandem, TandemInterpretation::Null))
                    {
                        header.passthrough.push((line.clone(), *source));
                    }
                }
                KernLine::LocalCommentLine(_) => header.passthrough.push((line.clone(), *source)),
                KernLine::Barline(barline)
                    if barline.has_fermata()
                        || !matches!(barline.repeat(), crate::types::RepeatStyle::None) =>
                {
                    header.passthrough.push((line.clone(), *source));
                }
                KernLine::Barline(_) => {}
                KernLine::SpineTypeLine(_)
                | KernLine::SpineManipulatorLine(_)
                | KernLine::Data(_) => {}
            }
        }

        header
    }

    fn apply_singleton_tandems(
        &mut self,
        tandems: &TandemInterpretationLine,
        source: Option<usize>,
    ) {
        for (index, (state, tandem)) in self.spines.iter_mut().zip(tandems.tokens()).enumerate() {
            match tandem {
                TandemInterpretation::Clef(clef) => {
                    state.clef = Some(clef.clone());
                    state.clef_source = source;
                    state.clef_generated = tandems.is_generated_token(index);
                }
                TandemInterpretation::KeySignature(key_signature) => {
                    // The header states the opening key, which has nothing
                    // to cancel.
                    state.key_signature = Some(key_signature.without_naturals());
                    state.key_signature_source = source;
                    state.key_signature_generated = tandems.is_generated_token(index);
                }
                TandemInterpretation::TimeSignature(_) | TandemInterpretation::FreeMeter => {
                    state.meter.apply(tandem);
                    state.time_signature_source = source;
                }
                TandemInterpretation::MeterSymbol(_) => {
                    state.meter.apply(tandem);
                    state.symbol_source = source;
                }
                _ => {}
            }
        }
    }

    fn into_lines(self, meter_output: MeterOutput) -> Vec<(KernLine, Option<usize>)> {
        let mut lines = Vec::new();

        push_tandem_line(
            &mut lines,
            self.spines
                .iter()
                .map(|state| {
                    if state.is_kern {
                        TandemInterpretation::Clef(state.clef.clone().unwrap_or_else(default_clef))
                    } else {
                        TandemInterpretation::Null
                    }
                })
                .collect(),
            self.spines
                .iter()
                .map(|state| state.is_kern && (state.clef.is_none() || state.clef_generated))
                .collect(),
            self.spines.iter().find_map(|state| state.clef_source),
        );
        push_tandem_line(
            &mut lines,
            self.spines
                .iter()
                .map(|state| {
                    if state.is_kern {
                        TandemInterpretation::KeySignature(
                            state.key_signature.unwrap_or_else(default_key_signature),
                        )
                    } else {
                        TandemInterpretation::Null
                    }
                })
                .collect(),
            self.spines
                .iter()
                .map(|state| {
                    state.is_kern
                        && (state.key_signature.is_none() || state.key_signature_generated)
                })
                .collect(),
            self.spines
                .iter()
                .find_map(|state| state.key_signature_source),
        );
        let sources = [
            self.spines
                .iter()
                .find_map(|state| state.time_signature_source),
            self.spines.iter().find_map(|state| state.symbol_source),
        ];
        let meter_tokens = meter_lines(
            self.spines
                .iter()
                .map(|state| Some(state.meter.declaration())),
            meter_output,
        );
        for (tokens, source) in meter_tokens.into_iter().zip(sources) {
            push_tandem_line(
                &mut lines,
                tokens
                    .into_iter()
                    .map(|token| token.unwrap_or(TandemInterpretation::Null))
                    .collect(),
                vec![false; self.spines.len()],
                source,
            );
        }

        lines.extend(self.passthrough);
        lines
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct HeaderSpineState {
    is_kern: bool,
    clef: Option<Clef>,
    clef_source: Option<usize>,
    clef_generated: bool,
    key_signature: Option<KeySignature>,
    key_signature_source: Option<usize>,
    key_signature_generated: bool,
    meter: ActiveMeter,
    time_signature_source: Option<usize>,
    symbol_source: Option<usize>,
}

/// A line declaring clef, key signature or meter.
fn is_singleton_header_line(tandems: &TandemInterpretationLine) -> bool {
    tandems.tokens().iter().any(defines_staff)
}

pub(crate) fn defines_staff(tandem: &TandemInterpretation) -> bool {
    StateKey::of(tandem).is_some_and(StateKey::defines_staff)
}

fn push_tandem_line(
    lines: &mut Vec<(KernLine, Option<usize>)>,
    tokens: Vec<TandemInterpretation>,
    generated: Vec<bool>,
    source: Option<usize>,
) {
    if tokens
        .iter()
        .all(|tandem| matches!(tandem, TandemInterpretation::Null))
    {
        return;
    }

    lines.push((
        KernLine::TandemInterpretationLine(
            TandemInterpretationLine::from_homogeneous_tokens_with_generated(tokens, generated),
        ),
        source,
    ));
}
