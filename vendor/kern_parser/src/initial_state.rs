//! Shared helpers for initial `**kern` state.
//!
//! Parsing and OMR normalization both need the same low-level facts about the
//! initial spine layout and the project defaults used when Humdrum leaves
//! required `**kern` state implicit.

use std::ops::Range;

use crate::{Clef, ClefSign, Fifths, KernDocument, KernLine, KeySignature, SpineType};

pub(crate) fn initial_header_start(document: &KernDocument) -> usize {
    initial_spine_type_declaration(document).map_or(0, |(index, _)| index + 1)
}

pub(crate) fn initial_header_state_range(document: &KernDocument) -> Range<usize> {
    let start = initial_header_start(document);
    let mut end = document.lines()[start..]
        .iter()
        .position(|line| {
            matches!(
                line,
                KernLine::Data(_) | KernLine::SpineManipulatorLine(_) | KernLine::SpineTypeLine(_)
            )
        })
        .map_or(document.lines().len(), |offset| start + offset);

    while end > start && matches!(document.lines().get(end - 1), Some(KernLine::Barline(_))) {
        end -= 1;
    }

    start..end
}

pub(crate) fn initial_spine_types(document: &KernDocument) -> Vec<SpineType> {
    initial_spine_type_declaration(document)
        .map(|(_, spine_types)| spine_types.to_vec())
        .unwrap_or_else(|| vec![SpineType::Kern; document.layout.initial_spine_count as usize])
}

pub(crate) fn initial_spine_type_declaration(
    document: &KernDocument,
) -> Option<(usize, &[SpineType])> {
    for (index, line) in document.lines().iter().enumerate() {
        match line {
            KernLine::SpineTypeLine(spine_types) => return Some((index, spine_types)),
            KernLine::Data(_) | KernLine::SpineManipulatorLine(_) => break,
            KernLine::TandemInterpretationLine(_)
            | KernLine::LocalCommentLine(_)
            | KernLine::Barline(_) => {}
        }
    }

    None
}

pub(crate) fn default_clef() -> Clef {
    Clef {
        sign: ClefSign::G,
        line: 2,
        octave_shift: 0,
        invisible: false,
    }
}

pub(crate) fn default_key_signature() -> KeySignature {
    KeySignature::from_fifths(Fifths::new(0).expect("0 fifths is a valid key signature"))
}
