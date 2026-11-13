//! Post-parse semantic finalization.
//!
//! Humdrum permits some leading `**kern` state to be omitted. Inside the typed
//! AST, Transcoda treats missing initial clef and key-signature state as
//! explicit defaults so downstream tree operations do not need to handle the
//! implicit case over and over.

use std::ops::Range;

use crate::{
    initial_state::{
        default_clef, default_key_signature, initial_header_state_range, initial_spine_types,
    },
    KernDocument, KernLine, SpineType, TandemInterpretation, TandemInterpretationLine,
    TandemInterpretationLineKind,
};

const GENERATED_SOURCE_LINE: usize = 0;

pub(crate) fn finalize_parsed_document(document: &mut KernDocument) {
    InitialKernStateFinalizer.finalize(document);
}

#[derive(Debug, Clone, Copy, Default)]
struct InitialKernStateFinalizer;

impl InitialKernStateFinalizer {
    fn finalize(self, document: &mut KernDocument) {
        let initial_spines = initial_spine_types(document);
        if !initial_spines
            .iter()
            .any(|spine_type| matches!(spine_type, SpineType::Kern))
        {
            return;
        }

        self.ensure_initial_state(document, &initial_spines, InitialStateKind::Clef);
        self.ensure_initial_state(document, &initial_spines, InitialStateKind::KeySignature);
    }

    fn ensure_initial_state(
        self,
        document: &mut KernDocument,
        initial_spines: &[SpineType],
        kind: InitialStateKind,
    ) {
        let range = initial_header_state_range(document);
        let missing = missing_initial_state(document, initial_spines, range.clone(), kind);
        if missing.iter().all(|is_missing| !is_missing) {
            return;
        }

        if let Some(line_index) = first_line_of_kind(document, range.clone(), kind) {
            fill_missing_tokens(document, line_index, &missing, kind);
            return;
        }

        let insert_index = default_insert_index(document, range, kind);
        document.insert_line(
            insert_index,
            default_line(initial_spines, &missing, kind),
            Some(GENERATED_SOURCE_LINE),
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitialStateKind {
    Clef,
    KeySignature,
}

fn missing_initial_state(
    document: &KernDocument,
    initial_spines: &[SpineType],
    range: Range<usize>,
    kind: InitialStateKind,
) -> Vec<bool> {
    let mut missing = initial_spines
        .iter()
        .map(|spine_type| matches!(spine_type, SpineType::Kern))
        .collect::<Vec<_>>();

    for line in &document.lines()[range] {
        let KernLine::TandemInterpretationLine(tandems) = line else {
            continue;
        };

        for (index, tandem) in tandems.tokens().iter().enumerate() {
            if index < missing.len() && kind.matches_tandem(tandem) {
                missing[index] = false;
            }
        }
    }

    missing
}

fn first_line_of_kind(
    document: &KernDocument,
    range: Range<usize>,
    kind: InitialStateKind,
) -> Option<usize> {
    document.lines()[range.clone()]
        .iter()
        .position(|line| {
            matches!(
                line,
                KernLine::TandemInterpretationLine(tandems)
                    if tandems.kind() == kind.line_kind()
            )
        })
        .map(|offset| range.start + offset)
}

fn fill_missing_tokens(
    document: &mut KernDocument,
    line_index: usize,
    missing: &[bool],
    kind: InitialStateKind,
) {
    let KernLine::TandemInterpretationLine(tandems) = &mut document.lines_mut()[line_index] else {
        return;
    };

    let mut generated_indexes = Vec::new();
    for (index, token) in tandems.tokens_mut().iter_mut().enumerate() {
        if missing.get(index).copied().unwrap_or(false) {
            *token = kind.default_tandem();
            generated_indexes.push(index);
        }
    }
    for index in generated_indexes {
        tandems.mark_generated_token(index);
    }
}

fn default_insert_index(
    document: &KernDocument,
    range: Range<usize>,
    kind: InitialStateKind,
) -> usize {
    match kind {
        InitialStateKind::Clef => range.start,
        InitialStateKind::KeySignature => document.lines()[range.clone()]
            .iter()
            .rposition(|line| {
                matches!(
                    line,
                    KernLine::TandemInterpretationLine(tandems)
                        if tandems.kind() == TandemInterpretationLineKind::Clef
                )
            })
            .map_or(range.start, |offset| range.start + offset + 1),
    }
}

fn default_line(
    initial_spines: &[SpineType],
    missing: &[bool],
    kind: InitialStateKind,
) -> KernLine {
    let mut line = TandemInterpretationLine::from_homogeneous_tokens(
        initial_spines
            .iter()
            .enumerate()
            .map(|(index, _)| {
                if missing.get(index).copied().unwrap_or(false) {
                    kind.default_tandem()
                } else {
                    TandemInterpretation::Null
                }
            })
            .collect(),
    );
    for (index, is_missing) in missing.iter().copied().enumerate() {
        if is_missing {
            line.mark_generated_token(index);
        }
    }
    KernLine::TandemInterpretationLine(line)
}

impl InitialStateKind {
    fn line_kind(self) -> TandemInterpretationLineKind {
        match self {
            Self::Clef => TandemInterpretationLineKind::Clef,
            Self::KeySignature => TandemInterpretationLineKind::KeySignature,
        }
    }

    fn matches_tandem(self, tandem: &TandemInterpretation) -> bool {
        matches!(
            (self, tandem),
            (Self::Clef, TandemInterpretation::Clef(_))
                | (Self::KeySignature, TandemInterpretation::KeySignature(_))
        )
    }

    fn default_tandem(self) -> TandemInterpretation {
        match self {
            Self::Clef => TandemInterpretation::Clef(default_clef()),
            Self::KeySignature => TandemInterpretation::KeySignature(default_key_signature()),
        }
    }
}
