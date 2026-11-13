//! OMR-specific normalization preset.
//!
//! Prepares a parsed document for the lossy OMR serializer. This preset keeps
//! the AST typed, but removes or reshapes document structure that should not
//! appear in OMR training output.
//!
//! The preset:
//! 1. rewrites non-serializable tandem interpretations to `*`
//! 2. removes null-only lines
//! 3. collapses compatible sparse tandem interpretation lines
//! 4. removes superseded visible-state interpretations
//! 5. removes redundant already-active state interpretations
//! 6. completes missing leading clef and key signature header state
//! 7. reorders only the leading tandem header block as clef, key, meter
//! 8. moves selected control tandems toward the data they affect
//! 9. removes any newly-created null-only lines
//! 10. moves safe tandem interpretations across spine operations
//! 11. collapses canceling split/merge regions when projectable
//! 12. restores final split spines to the starting staff topology
//! 13. merges adjacent spine manipulator lines when legal
//! 14. removes leading barlines before musical data
//! 15. optionally writes each measure's splits and merges on its positional grid
//! 16. canonicalizes token-local ties
//! 17. removes `**dynam` hairpin marks Verovio ignores and `???` placeholders
//!
//! It does not mutate the caller when used through [`normalized_for_omr`].
//!
//! Example:
//! ```text
//! Before:
//! **kern
//! *M4/4
//! *clefG2
//! *k[]
//! =1
//! (4c)
//! 4d[]
//!
//! After:
//! **kern
//! *clefG2
//! *k[]
//! *M4/4
//! 4c()
//! 4d
//! ```
//!
//! The preset reorders the leading header for the OMR grammar, removes the
//! initial barline, and canonicalizes an immediate tie.

use std::collections::BTreeSet;

use crate::{
    analysis::leading_header_bounds,
    normalization::{
        canon::{self, CanonError, Profile},
        NormalizationPass,
    },
    score::{unread_staff_declarations, TopologyError},
    serialize::{
        tandem_policy::{TandemAction, TandemKind, TandemPolicy},
        ContentPolicy,
    },
    KernDocument, KernLine, TandemInterpretation,
};

use super::NormalizeOptions;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OmrNormalizationError {
    #[error("OMR normalization did not converge after {iterations} iterations")]
    DidNotConverge { iterations: usize },
}

impl From<CanonError> for OmrNormalizationError {
    fn from(error: CanonError) -> Self {
        match error {
            CanonError::DidNotConverge { iterations } => Self::DidNotConverge { iterations },
        }
    }
}

/// Apply the OMR normalization preset in place.
pub fn normalize_for_omr(document: &mut KernDocument) {
    normalize_for_omr_with_options(document, &NormalizeOptions::default());
}

/// Apply the OMR normalization preset in place with explicit options.
pub fn normalize_for_omr_with_options(document: &mut KernDocument, options: &NormalizeOptions) {
    normalize_for_omr_with_options_and_policy(document, options, TandemPolicy::Omr);
}

/// Apply the OMR normalization preset, leaving out the tandems `policy`
/// does not carry to its output.
pub(crate) fn normalize_for_omr_with_policy(
    document: &mut KernDocument,
    options: &NormalizeOptions,
    policy: TandemPolicy,
) {
    normalize_for_omr_with_options_and_policy(document, options, policy);
}

fn normalize_for_omr_with_options_and_policy(
    document: &mut KernDocument,
    options: &NormalizeOptions,
    policy: TandemPolicy,
) {
    try_normalize_for_omr_with_options_and_policy(document, options, policy)
        .expect("OMR normalization must converge");
}

fn try_normalize_for_omr_with_options_and_policy(
    document: &mut KernDocument,
    options: &NormalizeOptions,
    policy: TandemPolicy,
) -> Result<usize, OmrNormalizationError> {
    try_normalize_for_omr_with_options_and_policy_with_limit(
        document,
        options,
        policy,
        canon::FIXPOINT_LIMIT,
    )
}

fn try_normalize_for_omr_with_options_and_policy_with_limit(
    document: &mut KernDocument,
    options: &NormalizeOptions,
    policy: TandemPolicy,
    limit: usize,
) -> Result<usize, OmrNormalizationError> {
    canon::canonicalize_with_limit(document, &Profile::omr(options, policy), limit)
        .map_err(OmrNormalizationError::from)
}

pub(crate) fn try_canonicalize_transcoda_v2_document(
    document: &mut KernDocument,
    policy: ContentPolicy,
) -> Result<usize, OmrNormalizationError> {
    canon::canonicalize(document, &Profile::transcoda_v2(policy))
        .map_err(OmrNormalizationError::from)
}

/// Return an OMR-normalized clone.
pub fn normalized_for_omr(document: &KernDocument) -> KernDocument {
    normalized_for_omr_with_options(document, &NormalizeOptions::default())
}

/// Return an OMR-normalized clone with explicit options.
pub fn normalized_for_omr_with_options(
    document: &KernDocument,
    options: &NormalizeOptions,
) -> KernDocument {
    let mut normalized = document.clone();
    normalize_for_omr_with_options(&mut normalized, options);
    normalized
}

pub(crate) fn normalized_for_omr_with_policy(
    document: &KernDocument,
    options: &NormalizeOptions,
    policy: TandemPolicy,
) -> KernDocument {
    let mut normalized = document.clone();
    normalize_for_omr_with_policy(&mut normalized, options, policy);
    normalized
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Replace the tandem interpretations a normalization does not keep with `*`:
/// those an output's [`TandemPolicy`] leaves out. Lines that become
/// entirely null are removed later by
/// [`RemoveNullLines`](crate::normalization::null_lines::RemoveNullLines).
pub struct RewriteNonSerializableTandems {
    options: NormalizeOptions,
    column: TandemColumn,
}

impl RewriteNonSerializableTandems {
    pub fn new(options: NormalizeOptions, policy: TandemPolicy) -> Self {
        Self {
            options,
            column: TandemColumn::Output(policy),
        }
    }

    pub(crate) fn with_column(options: NormalizeOptions, column: TandemColumn) -> Self {
        Self { options, column }
    }
}

impl NormalizationPass for RewriteNonSerializableTandems {
    fn normalize(&self, document: &mut KernDocument) {
        let unread = if self.column.drops_unread_staff() {
            // A document without a topology keeps every `*staff`; whoever
            // resolves its score reports the topology error.
            unread_staff_declarations(document).unwrap_or_default()
        } else {
            BTreeSet::new()
        };
        null_tandems(document, |line, column, tandem| {
            !self.column.keeps(tandem) || unread.contains(&(line, column))
        });
    }
}

/// Replace every `*staff` Verovio does not read with `*`, leaving the staff
/// numbers as they are.
pub(crate) fn remove_unread_staff_declarations(
    document: &mut KernDocument,
) -> Result<(), TopologyError> {
    let unread = unread_staff_declarations(document)?;
    null_tandems(document, |line, column, _| unread.contains(&(line, column)));
    Ok(())
}

/// Replace the tandems `drop(line index, column, tandem)` selects with `*`.
fn null_tandems(
    document: &mut KernDocument,
    drop: impl Fn(usize, usize, &TandemInterpretation) -> bool,
) {
    for (line_index, line) in document.lines_mut().iter_mut().enumerate() {
        let KernLine::TandemInterpretationLine(tandems) = line else {
            continue;
        };

        let mut generated = tandems.generated_tokens().to_vec();
        for (index, tandem) in tandems.tokens_mut().iter_mut().enumerate() {
            if drop(line_index, index, tandem) {
                *tandem = TandemInterpretation::Null;
                if let Some(generated) = generated.get_mut(index) {
                    *generated = false;
                }
            }
        }
        *tandems = crate::TandemInterpretationLine::from_homogeneous_tokens_with_generated(
            tandems.tokens().to_vec(),
            generated,
        );
    }
}

/// Output policy for tandem normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TandemColumn {
    Output(TandemPolicy),
}
impl TandemColumn {
    fn keeps(self, tandem: &TandemInterpretation) -> bool {
        let Self::Output(policy) = self;
        carried(policy, tandem)
    }
    fn drops_unread_staff(self) -> bool {
        let Self::Output(policy) = self;
        policy.action(TandemKind::Staff) == TandemAction::Keep
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Remove barlines that occur at the very beginning of the musical content.
///
/// Verovio does not render these initial barlines, so OMR output treats
/// documents with and without them as equivalent. Only the contiguous barline
/// run after the leading tandem header block is removed; later barlines are
/// preserved.
pub struct RemoveLeadingBarlines {
    options: NormalizeOptions,
}

impl RemoveLeadingBarlines {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for RemoveLeadingBarlines {
    fn normalize(&self, document: &mut KernDocument) {
        let (_, start) = leading_header_bounds(document);
        let end = document.lines()[start..]
            .iter()
            .position(|line| !matches!(line, KernLine::Barline(barline) if !barline.has_fermata() && !matches!(barline.repeat(), crate::types::RepeatStyle::Start | crate::types::RepeatStyle::StartAndEnd)))
            .map_or(document.lines().len(), |offset| start + offset);

        document.drain_lines(start..end);
        if let Some(KernLine::Barline(barline)) = document.lines_mut().get_mut(start) {
            if matches!(
                barline.repeat(),
                crate::types::RepeatStyle::Start | crate::types::RepeatStyle::StartAndEnd
            ) {
                *barline = barline.right_face();
            }
        }
    }
}

/// Whether normalization keeps a tandem for an output under `policy`: the
/// output writes it, or folds it into the notes it governs.
fn carried(policy: TandemPolicy, tandem: &TandemInterpretation) -> bool {
    matches!(
        policy.action_for(tandem),
        TandemAction::Keep | TandemAction::Fold
    )
}

#[cfg(test)]
mod tests {
    use super::{try_normalize_for_omr_with_options_and_policy_with_limit, OmrNormalizationError};
    use crate::{
        normalization::{omr::normalized_for_omr, NormalizeOptions},
        KernDocument, KernLine, TandemInterpretation,
    };

    fn normalized_omr_string(input: &str) -> String {
        normalized_for_omr(&KernDocument::parse(input).unwrap()).to_omr_string()
    }

    fn normalized_humdrum_string(input: &str) -> String {
        normalized_for_omr(&KernDocument::parse(input).unwrap()).to_humdrum_kern_string()
    }

    #[test]
    fn reports_fixpoint_cap_exhaustion_in_release_code() {
        let mut document = KernDocument::parse("**kern\n4c\n*-").unwrap();
        let error = try_normalize_for_omr_with_options_and_policy_with_limit(
            &mut document,
            &NormalizeOptions::default(),
            crate::serialize::TandemPolicy::Omr,
            0,
        )
        .unwrap_err();
        assert_eq!(
            error,
            OmrNormalizationError::DidNotConverge { iterations: 0 }
        );
    }

    #[test]
    fn normalizes_split_after_topology_shrink_without_panicking() {
        let input = [
            "**kern\t**kern",
            "*clefF4\t*clefG2",
            "4C\t4c",
            "*-\t*",
            "*^",
            "4d\t4e",
            "*v\t*v",
            "*-",
        ]
        .join("\n");
        let document = KernDocument::parse(&input).unwrap();

        document.to_compact_omr_no_note_modifiers_string();
    }

    #[test]
    fn collapses_redundant_merge_barline_split_before_humdrum_render() {
        let input = [
            "**kern\t**kern\t**kern",
            "4a\t4b\t4cc",
            "*\t*v\t*v",
            "=\t=",
            "*\t*^",
            "4c\t4e\t4g",
            "*-\t*-\t*-",
        ]
        .join("\n");

        let normalized = normalized_humdrum_string(&input);

        assert!(normalized.contains("=\t=\t=\n"));
        assert!(!normalized.contains("*\t*v\t*v\n"));
        assert!(!normalized.contains("*\t*^\n"));
    }

    #[test]
    fn removes_leading_barline_exposed_by_redundant_merge_split_collapse() {
        let input = [
            "**kern\t**kern\t**kern",
            "*\t*v\t*v",
            "=\t=",
            "*\t*^",
            "4c\t4e\t4g",
            "*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefG2\t*clefG2\t*clefG2\n*k[]\t*k[]\t*k[]\n4c\t4e\t4g\n*-\t*-\t*-\n"
        );
    }

    #[test]
    fn rejects_a_final_merge_of_two_staves_instead_of_panicking() {
        let input = [
            "**kern\t**kern",
            "*clefF4\t*clefG2",
            "4C\t4c",
            "==\t==",
            "*v\t*v",
            "*-",
        ]
        .join("\n");
        let document = KernDocument::parse(&input).unwrap();

        let error = crate::serialize::omr::try_to_transcoda_v2_string_with_options(
            &document,
            Default::default(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("cannot merge spines from different original spines"));
    }

    #[test]
    fn rewrites_nonserializable_tandems_to_nulls_then_removes_empty_lines() {
        let document = KernDocument::parse("**kern\n*part1\n*clefG2\n4c\n").unwrap();

        let normalized = normalized_for_omr(&document);

        assert_eq!(normalized.lines().len(), 4);
        assert!(matches!(
            normalized.lines()[1],
            KernLine::TandemInterpretationLine(_)
        ));
    }

    #[test]
    fn preserves_visual_style_tandems_for_renderable_normalized_output() {
        let input = "**kern\n4c\n*2\\right\n2c\n*2\\left\n2d\n*-\n";

        let normalized = normalized_humdrum_string(input);

        assert!(normalized.contains("*2\\right\n"));
        assert!(normalized.contains("*2\\left\n"));
    }

    #[test]
    fn reorders_only_the_leading_header_block() {
        let document = KernDocument::parse("**kern\n*M3/4\n*k[f#]\n*clefG2\n4c\n*M2/4\n").unwrap();

        let normalized = normalized_for_omr(&document);

        assert!(matches!(
            &normalized.lines()[1],
            KernLine::TandemInterpretationLine(line)
                if matches!(line.tokens()[0], TandemInterpretation::Clef(_))
        ));
        assert!(matches!(
            &normalized.lines()[2],
            KernLine::TandemInterpretationLine(line)
                if matches!(line.tokens()[0], TandemInterpretation::KeySignature(_))
        ));
        assert!(matches!(
            &normalized.lines()[3],
            KernLine::TandemInterpretationLine(line)
                if matches!(line.tokens()[0], TandemInterpretation::TimeSignature(_))
        ));
        assert!(matches!(
            &normalized.lines()[5],
            KernLine::TandemInterpretationLine(line)
                if matches!(line.tokens()[0], TandemInterpretation::TimeSignature(_))
        ));
    }

    #[test]
    fn completes_missing_leading_clef() {
        let input = "**kern\n*k[]\n*M4/4\n4c\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n*M4/4\n4c\n*-\n"
        );
    }

    #[test]
    fn completes_missing_leading_key_signature() {
        let input = "**kern\n*clefF4\n*M4/4\n4C\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefF4\n*k[]\n*M4/4\n4C\n*-\n"
        );
    }

    #[test]
    fn completes_sparse_multi_spine_leading_headers() {
        let input = "**kern\t**kern\n*clefG2\t*\n*\t*k[b-]\n4c\t4E\n*-\t*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\t*clefG2\n*k[]\t*k[b-]\n4c\t4E\n*-\t*-\n"
        );
    }

    #[test]
    fn canonicalizes_initial_header_across_leading_barline() {
        let input = [
            "**kern", "*clefG2", "*k[]", "=", "*MM40", "*M4/4", "*k[]", "*clefG2", "4c", "*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefG2\n*k[]\n*M4/4\n4c\n*-\n"
        );
    }

    #[test]
    fn initial_header_conflicting_clefs_use_last_pre_data_value() {
        let input = "**kern\n*clefG2\n=\n*clefF4\n4C\n*-\n";

        assert_eq!(normalized_omr_string(input), "*clefF4\n*k[]\n4C\n*-\n");
    }

    #[test]
    fn initial_header_conflicting_key_signatures_use_last_pre_data_value() {
        let input = "**kern\n*clefG2\n*k[f#]\n=\n*k[b-]\n4c\n*-\n";

        assert_eq!(normalized_omr_string(input), "*clefG2\n*k[b-]\n4c\n*-\n");
    }

    #[test]
    fn initial_header_conflicting_meter_like_values_use_last_pre_data_value() {
        let input = "**kern\n*clefG2\n*met(C)\n=\n*M3/4\n4c\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n*M3/4\n4c\n*-\n"
        );
    }

    #[test]
    fn tandem_changes_after_data_are_not_absorbed_into_initial_header() {
        let input = "**kern\n*clefG2\n*k[]\n4c\n*clefF4\n4C\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n4c\n*clefF4\n4C\n*-\n"
        );
    }

    #[test]
    fn defaults_apply_only_to_kern_spines() {
        let input = "**kern\t**dynam\t**kern\n*clefF4\t*\t*\n4C\tmf\t4c\n*-\t*-\t*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefF4\t*clefG2\n*k[]\t*k[]\n4C\t4c\n*-\t*-\n"
        );
    }

    #[test]
    fn does_not_complete_mid_score_interpretation_changes() {
        let input = "**kern\n*clefF4\n*k[b-]\n4C\n*k[f#]\n4D\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefF4\n*k[b-]\n4C\n*k[f#]\n4D\n*-\n"
        );
    }

    #[test]
    fn removes_superseded_meter_chain_before_content() {
        let input = "**kern\n*clefG2\n*M4/4\n*M6/4\n*M8/4\n4c\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n*M8/4\n4c\n*-\n"
        );
    }

    #[test]
    fn removes_superseded_meter_symbol_chain_before_content() {
        let input = "**kern\n*clefG2\n*met(C)\n*met(C|)\n4c\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n*met(C|)\n4c\n*-\n"
        );
    }

    #[test]
    fn initial_header_meter_group_keeps_only_the_printed_symbol() {
        for input in [
            "**kern\n*clefG2\n*M4/4\n*met(C)\n4c\n*-\n",
            "**kern\n*clefG2\n*met(C)\n*M4/4\n4c\n*-\n",
        ] {
            assert_eq!(
                normalized_omr_string(input),
                "*clefG2\n*k[]\n*met(C)\n4c\n*-\n"
            );
        }
    }

    #[test]
    fn removes_superseded_key_signatures_before_content() {
        let input = "**kern\n*clefG2\n*k[]\n*k[f#c#]\n*k[b-]\n4c\n*-\n";

        assert_eq!(normalized_omr_string(input), "*clefG2\n*k[b-]\n4c\n*-\n");
    }

    #[test]
    fn removes_ottava_cancelled_before_content() {
        let input = "**kern\n*clefG2\n4c\n*8va\n*\n*X8va\n4d\n*-\n";

        assert_eq!(normalized_omr_string(input), "*clefG2\n*k[]\n4c\n4d\n*-\n");
    }

    #[test]
    fn removes_ottava_replaced_before_content_but_keeps_later_cancel() {
        let input = "**kern\n*clefG2\n4c\n*8va\n*15ma\n4d\n*X15ma\n4e\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n4c\n*15ma\n4d\n*X15ma\n4e\n*-\n"
        );
    }

    #[test]
    fn removes_pedal_cancelled_before_content() {
        let input = "**kern\n*clefG2\n4c\n*ped\n*\n*Xped\n4d\n*-\n";

        assert_eq!(normalized_omr_string(input), "*clefG2\n*k[]\n4c\n4d\n*-\n");
    }

    #[test]
    fn removes_pedal_replaced_before_content_but_keeps_later_cancel() {
        let input = "**kern\n*clefG2\n4c\n*ped\n*ped\n4d\n*Xped\n4e\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n4c\n*ped\n4d\n*Xped\n4e\n*-\n"
        );
    }

    #[test]
    fn preserves_pedal_across_content() {
        let input = "**kern\n*clefG2\n*ped\n4c\n*Xped\n4d\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n*ped\n4c\n*Xped\n4d\n*-\n"
        );
    }

    #[test]
    fn does_not_remove_superseded_interpretations_across_non_tandem_lines() {
        let input = "**kern\n*clefG2\n*M4/4\n4c\n=1\n*M6/4\n4d\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n*M4/4\n4c\n=\n*M6/4\n4d\n*-\n"
        );
    }

    #[test]
    fn removes_plain_initial_barline() {
        let input = "**kern\n*clefG2\n*k[f#c#]\n=\n4c 4e 4g\n==\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[f#c#]\n4c 4e 4g\n==\n*-\n"
        );
    }

    #[test]
    fn initial_barline_sample_matches_absent_barline_sample() {
        let with_initial_barline = [
            "**kern", "*clefG2", "*k[f#c#]", "=", "4c 4e 4g", "4c 4e 4g", "4c 4e 4g", "==", "*-",
        ]
        .join("\n");
        let without_initial_barline = [
            "**kern", "*clefG2", "*k[f#c#]", "4c 4e 4g", "4c 4e 4g", "4c 4e 4g", "==", "*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&with_initial_barline),
            normalized_omr_string(&without_initial_barline)
        );
    }

    #[test]
    fn removes_initial_barlines_except_start_repeats() {
        for barline in ["=1", "==", "=||", "=-", "=:|!"] {
            let input = format!("**kern\n*clefG2\n{kbarline}\n4c\n*-\n", kbarline = barline);

            assert_eq!(normalized_omr_string(&input), "*clefG2\n*k[]\n4c\n*-\n");
        }
    }

    #[test]
    fn preserves_barlines_after_data() {
        let input = "**kern\n*clefG2\n4c\n=1\n4d\n==\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n4c\n=\n4d\n==\n*-\n"
        );
    }

    #[test]
    fn removes_initial_barline_after_moving_split_across_it() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*^\t*",
            "=\t=\t=",
            "4c\t4e\t4g",
            "*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*^\t*\n4c\t4e\t4g\n*v\t*v\t*\n*-\t*-\n"
        );
    }

    #[test]
    fn removes_initial_barline_after_moving_right_spine_split_across_it() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*\t*^",
            "=\t=\t=",
            "4c\t4e\t4g",
            "*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*\t*^\n4c\t4e\t4g\n*\t*v\t*v\n*-\t*-\n"
        );
    }

    #[test]
    fn removes_initial_barline_after_moving_middle_spine_split_across_it() {
        let input = [
            "**kern\t**kern\t**kern",
            "*clefG2\t*clefG2\t*clefG2",
            "*\t*^\t*",
            "=\t=\t=\t=",
            "4c\t4e\t4g\t4b",
            "*-\t*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefG2\t*clefG2\t*clefG2\n*k[]\t*k[]\t*k[]\n*\t*^\t*\n4c\t4e\t4g\t4b\n*\t*v\t*v\t*\n*-\t*-\t*-\n"
        );
    }

    #[test]
    fn mixed_split_merge_after_barline_keeps_header_before_merge() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*\t*^",
            "4c\t4e\t4g",
            "=\t=\t=",
            "*^\t*v\t*v",
            "4c\t4e\t4g",
            "*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "\
*clefG2	*clefG2
*k[]	*k[]
*	*^
4c	4e	4g
*	*v	*v
=	=
*^	*
4c	4e	4g
*v	*v	*
*-	*-
"
        );
    }

    #[test]
    fn removes_multi_spine_initial_barline() {
        let input = "**kern\t**kern\n*clefG2\t*clefF4\n=\t=\n4c\t4C\n*-\t*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\t*clefF4\n*k[]\t*k[]\n4c\t4C\n*-\t*-\n"
        );
    }

    #[test]
    fn removes_initial_barline_before_split_only_line() {
        let input =
            "**kern\t**kern\n*clefF4\t*clefG2\n*k[f#]\t*k[f#]\n=\t=\n*^\t*^\n4D\t2d\t2d\t2d\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefF4\t*clefG2\n*k[f#]\t*k[f#]\n*^\t*^\n4D\t2d\t2d\t2d\n*v\t*v\t*\t*\n*\t*v\t*v\n"
        );
    }

    #[test]
    fn removes_initial_barline_after_reordered_completed_header() {
        let input = "**kern\n*M3/4\n*clefG2\n=\n4c\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n*M3/4\n4c\n*-\n"
        );
    }

    #[test]
    fn removes_superseded_interpretations_per_spine() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefF4",
            "*M4/4\t*",
            "*M6/4\t*",
            "*\t*k[]",
            "*\t*k[f#]",
            "*\t*k[b-]",
            "4c\t4C",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefG2\t*clefF4\n*k[]\t*k[b-]\n*M6/4\t*\n4c\t4C\n*-\t*-\n"
        );
    }

    #[test]
    fn removes_repeated_stria_lines() {
        let input =
            "**kern\n*clefG2\n*stria4\n4c\n*^\n*\t*stria4\n4d\t4f\n*v\t*v\n*stria4\n4e\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n*stria4\n4c\n*^\n4d\t4f\n*v\t*v\n4e\n*-\n"
        );
    }

    #[test]
    fn preserves_changed_stria_state() {
        let input = "**kern\n*clefG2\n*stria4\n4c\n*stria5\n4d\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n*stria4\n4c\n*stria5\n4d\n*-\n"
        );
    }

    #[test]
    fn removes_repeated_active_ottava_but_preserves_matching_cancel() {
        let input = "**kern\n*clefG2\n*8va\n4c\n*8va\n4d\n*X8va\n4e\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n*8va\n4c\n4d\n*X8va\n4e\n*-\n"
        );
    }

    #[test]
    fn refuses_an_ottava_cancel_with_no_ottava_open() {
        let input = "**kern\n*clefG2\n*X8va\n4c\n*8va\n4d\n*X8va\n*X8va\n4e\n*-\n";

        assert!(crate::serialize::omr::OmrSerializer::default()
            .try_serialize_document(&KernDocument::parse(input).unwrap())
            .is_err());
    }

    #[test]
    fn moves_control_tandems_toward_data_in_omr_preset() {
        let input = "**kern\n*clefG2\n4c\n*8va\n*clefC1\n4d\n*clefG2\n*X8va\n4e\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n4c\n*clefC1\n*8va\n4d\n*X8va\n*clefG2\n4e\n*-\n"
        );
    }

    #[test]
    fn preserves_tremolo_control_tandems_in_omr_preset() {
        let input = "**kern\n*clefG2\n4c\n*tremolo\n16eLL\n16eJJ\n*Xtremolo\n4f\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n4c\n*tremolo\n16eLL\n16eJJ\n*Xtremolo\n4f\n*-\n"
        );
    }

    #[test]
    fn preserves_repeated_pedal_clef_key_signature_and_meter_after_content() {
        let input = "**kern\n*clefG2\n*k[]\n*M4/4\n*ped\n4c\n*clefG2\n*k[]\n*M4/4\n*ped\n4d\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n*M4/4\n*ped\n4c\n*clefG2\n*k[]\n*M4/4\n*ped\n4d\n*-\n"
        );
    }

    #[test]
    fn removes_redundant_state_per_track() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefF4",
            "*tremolo\t*",
            "4c\t4C",
            "*tremolo\t*tremolo",
            "4d\t4D",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefG2\t*clefF4\n*k[]\t*k[]\n*tremolo\t*\n4c\t4C\n*\t*tremolo\n4d\t4D\n*-\t*-\n"
        );
    }

    #[test]
    fn preserves_state_interpretation_after_ambiguous_merge() {
        let input = [
            "**kern",
            "*clefG2",
            "*stria4",
            "*^",
            "*stria5\t*",
            "*v\t*v",
            "*stria4",
            "4c",
            "*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefG2\n*k[]\n*stria4\n*^\n*stria5\t*\n*v\t*v\n*stria4\n4c\n*-\n"
        );
    }

    #[test]
    fn merges_adjacent_spine_manipulators_in_omr_preset() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*^\t*",
            "*\t*^\t*",
            "4c\t4e\t4g\t4cc",
            "*v\t*v\t*\t*",
            "*v\t*v\t*",
            "2c\t2g",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*^\t*\n*\t*^\t*\n4c\t4e\t4g\t4cc\n*v\t*v\t*v\t*\n2c\t2g\n*-\t*-\n"
        );
    }

    #[test]
    fn collapses_trivial_split_merge_regions_in_omr_preset() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefF4",
            "*^\t*",
            "*\t*\t*",
            "=\t=\t=",
            "*v\t*v\t*",
            "4c\t4C",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefG2\t*clefF4\n*k[]\t*k[]\n4c\t4C\n*-\t*-\n"
        );
    }

    #[test]
    fn restores_equivalent_nested_splits_to_staff_count() {
        let split_right_then_right = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*\t*^",
            "*\t*\t*^",
            "*-\t*-\t*-\t*-",
        ]
        .join("\n");
        let split_right_then_left_branch = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*\t*^",
            "*\t*^\t*",
            "*-\t*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&split_right_then_right),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*\t*^\n*\t*\t*^\n*\t*v\t*v\t*v\n*-\t*-\n"
        );
        assert_eq!(
            normalized_omr_string(&split_right_then_left_branch),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*\t*^\n*\t*^\t*\n*\t*v\t*v\t*v\n*-\t*-\n"
        );
    }

    #[test]
    fn canonicalizes_equivalent_staggered_merge_spellings() {
        let merge_left_then_right = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*^\t*^",
            "4c\t4e\t4g\t4cc",
            "*v\t*v\t*\t*",
            "*\t*v\t*v",
            "*-\t*-",
        ]
        .join("\n");
        let merge_right_then_left = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*^\t*^",
            "4c\t4e\t4g\t4cc",
            "*\t*\t*v\t*v",
            "*v\t*v\t*",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&merge_left_then_right),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*^\t*^\n4c\t4e\t4g\t4cc\n*v\t*v\t*\t*\n*\t*v\t*v\n*-\t*-\n"
        );
        assert_eq!(
            normalized_omr_string(&merge_right_then_left),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*^\t*^\n4c\t4e\t4g\t4cc\n*v\t*v\t*\t*\n*\t*v\t*v\n*-\t*-\n"
        );
    }

    #[test]
    fn canonicalizes_merge_lines_added_by_final_topology_restoration() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*^\t*^",
            "4c\t4e\t4g\t4b",
            "*-\t*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            normalized_omr_string(&input),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*^\t*^\n4c\t4e\t4g\t4b\n*v\t*v\t*\t*\n*\t*v\t*v\n*-\t*-\n"
        );
    }

    #[test]
    fn canonicalizes_ties_before_omr_serialization() {
        let input = "**kern\n*clefG2\n4c[]\n4d[[\n4e[_]\n4f]]\n*-\n";

        assert_eq!(
            normalized_omr_string(input),
            "*clefG2\n*k[]\n4c\n4d[[\n4e_\n4f]]\n*-\n"
        );
    }

    #[test]
    fn preserves_immediate_slurs_before_omr_serialization() {
        let input = "**kern\n*clefG2\n(4c)\n*-\n";

        assert_eq!(normalized_omr_string(input), "*clefG2\n*k[]\n4c()\n*-\n");
    }
}
