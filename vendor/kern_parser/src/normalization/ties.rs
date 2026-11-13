//! Tie normalization.
//!
//! Normalizes repeated and null tie markers that occur on the same note token.
//! This pass is intentionally token-local. Document-level ordinary tie pairing
//! and balance diagnostics are owned by [`crate::analysis::tie_chains`].
//!
//! Example:
//! ```text
//! Before: 4c[] 4e[[[[ 4g]]]]
//! After:  4c   4e[[  4g]]
//! ```
//!
//! The immediate start/end tie on `4c[]` cancels, while repeated starts or ends
//! with the same resolution on the same token collapse to one marker.

use crate::{
    normalization::NormalizationPass,
    types::{NoteModifiers, Tie, TiePhase, TieResolution},
    KernContent, KernDocument, KernLine, SpineContent,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Remove same-token null ties and collapse repeated tie states.
///
/// Matching starts and ends with the same resolution on the same note token
/// cancel each other. Any remaining tie states are emitted in canonical order:
/// start, continue, end.
pub struct CanonicalizeImmediateTies {
    options: NormalizeOptions,
}

impl CanonicalizeImmediateTies {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for CanonicalizeImmediateTies {
    fn normalize(&self, document: &mut KernDocument) {
        for line in document.lines_mut() {
            let KernLine::Data(contents) = line else {
                continue;
            };

            for content in contents {
                let SpineContent::Kern(content) = content else {
                    continue;
                };

                match content {
                    KernContent::Note(note) => canonicalize_ties(&mut note.modifiers),
                    KernContent::Chord(chord) => {
                        for note in &mut chord.notes {
                            canonicalize_ties(&mut note.modifiers);
                        }
                    }
                    KernContent::Grace(grace) => {
                        for note in &mut grace.notes {
                            canonicalize_ties(&mut note.modifiers);
                        }
                    }
                    KernContent::Null
                    | KernContent::Rest(_)
                    | KernContent::InvisibleRestSlice(_)
                    | KernContent::MultiMeasureRest(_) => {}
                }
            }
        }
    }
}

fn canonicalize_ties(modifiers: &mut NoteModifiers) {
    if modifiers.ties.is_empty() {
        return;
    }

    let mut normalized = Vec::new();
    for resolution in [
        TieResolution::Ordinary,
        TieResolution::Disjunct,
        TieResolution::Linked,
    ] {
        let starts = modifiers
            .ties
            .iter()
            .filter(|tie| tie.phase() == TiePhase::Start && tie.resolution() == resolution)
            .count();
        let continues = modifiers
            .ties
            .iter()
            .any(|tie| tie.phase() == TiePhase::Continue && tie.resolution() == resolution);
        let ends = modifiers
            .ties
            .iter()
            .filter(|tie| tie.phase() == TiePhase::End && tie.resolution() == resolution)
            .count();
        let cancellations = starts.min(ends);
        let preserved = |phase| {
            let tie = Tie::new(phase, resolution);
            let placement = modifiers
                .ties
                .iter()
                .rev()
                .filter(|tie| tie.phase() == phase && tie.resolution() == resolution)
                .find_map(|tie| tie.placement());
            placement.map_or(tie, |placement| tie.with_placement(placement))
        };

        if starts > cancellations {
            normalized.push(preserved(TiePhase::Start));
        }
        if continues {
            normalized.push(preserved(TiePhase::Continue));
        }
        if ends > cancellations {
            normalized.push(preserved(TiePhase::End));
        }
    }

    modifiers.ties = normalized;
}

#[cfg(test)]
mod tests {
    use crate::{
        normalization::{ties::CanonicalizeImmediateTies, NormalizationPass},
        types::Tie,
        KernContent, KernDocument, KernLine, SpineContent,
    };

    fn normalize_ties(document: &mut KernDocument) {
        CanonicalizeImmediateTies::new(Default::default()).normalize(document);
    }

    fn first_chord_ties(input: &str) -> Vec<Tie> {
        let mut document = KernDocument::parse(input).unwrap();

        normalize_ties(&mut document);

        let KernLine::Data(contents) = document
            .lines()
            .iter()
            .find(|line| matches!(line, KernLine::Data(_)))
            .expect("expected data line")
        else {
            panic!("expected data line");
        };
        let SpineContent::Kern(KernContent::Chord(chord)) = &contents[0] else {
            panic!("expected chord");
        };
        chord.notes[0].modifiers.ties.clone()
    }

    #[test]
    fn removes_immediate_null_tie() {
        assert_eq!(first_chord_ties("**kern\n4c[]\n"), Vec::<Tie>::new());
    }

    #[test]
    fn removes_repeated_immediate_null_ties() {
        assert_eq!(first_chord_ties("**kern\n4c[[]]\n"), Vec::<Tie>::new());
    }

    #[test]
    fn collapses_repeated_identical_tie_states() {
        assert_eq!(
            first_chord_ties("**kern\n4c[[[[\n"),
            vec![Tie::DisjunctStart]
        );
        assert_eq!(first_chord_ties("**kern\n4c]]]]\n"), vec![Tie::DisjunctEnd]);
        assert_eq!(
            first_chord_ties("**kern\n4c____\n"),
            vec![Tie::DisjunctContinue]
        );
    }

    #[test]
    fn preserves_unmatched_states_in_canonical_order() {
        assert_eq!(
            first_chord_ties("**kern\n4c[_\n"),
            vec![Tie::Start, Tie::Continue]
        );
        assert_eq!(first_chord_ties("**kern\n4c[_]\n"), vec![Tie::Continue]);
    }

    #[test]
    fn keeps_placement_when_collapsing_tie_states() {
        let ties = first_chord_ties("**kern\n4c[>N_<\n");
        assert_eq!(ties.len(), 2);
        assert_eq!(
            ties[0],
            Tie::Start.with_placement(crate::types::TiePlacement::Above)
        );
        assert_eq!(
            ties[1],
            Tie::LinkedContinue.with_placement(crate::types::TiePlacement::Below)
        );
    }

    #[test]
    fn normalizes_ties_by_resolution() {
        assert_eq!(first_chord_ties("**kern\n4cN[N]\n"), Vec::<Tie>::new());
        assert_eq!(
            first_chord_ties("**kern\n4c[[N[\n"),
            vec![Tie::DisjunctStart, Tie::LinkedStart]
        );
    }

    #[test]
    fn canonicalizes_each_note_in_chords() {
        let mut document = KernDocument::parse("**kern\n4c[] 4e[ 4g]\n").unwrap();

        normalize_ties(&mut document);

        let KernLine::Data(contents) = document
            .lines()
            .iter()
            .find(|line| matches!(line, KernLine::Data(_)))
            .expect("expected data line")
        else {
            panic!("expected data line");
        };
        let SpineContent::Kern(KernContent::Chord(chord)) = &contents[0] else {
            panic!("expected chord");
        };
        assert_eq!(chord.notes[0].modifiers.ties, Vec::<Tie>::new());
        assert_eq!(chord.notes[1].modifiers.ties, vec![Tie::Start]);
        assert_eq!(chord.notes[2].modifiers.ties, vec![Tie::End]);
    }

    #[test]
    fn canonicalizes_each_grace_note() {
        let mut document = KernDocument::parse("**kern\n8qcc[] 8qdd_\n").unwrap();

        normalize_ties(&mut document);

        let KernLine::Data(contents) = document
            .lines()
            .iter()
            .find(|line| matches!(line, KernLine::Data(_)))
            .expect("expected data line")
        else {
            panic!("expected data line");
        };
        let SpineContent::Kern(KernContent::Grace(grace)) = &contents[0] else {
            panic!("expected grace chord");
        };
        assert_eq!(grace.notes[0].modifiers.ties, Vec::<Tie>::new());
        assert_eq!(grace.notes[1].modifiers.ties, vec![Tie::Continue]);
    }
}
