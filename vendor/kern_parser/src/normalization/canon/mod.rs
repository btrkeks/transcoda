//! Shared normalization driver for Transcoda source and label outputs.

pub(crate) mod schedule;

use thiserror::Error;

use crate::{
    normalization::{
        barline_spine_operations::{
            CollapseRedundantMergeSplitAroundBarlines, MoveMergesBeforeBarlines,
            MoveSplitsAfterBarlines,
        },
        beam_spelling::CanonicalizeBeamSpelling,
        canceling_split_merge_regions::CollapseTrivialSplitMergeRegions,
        control_tandem_adjacency::MoveControlTandemsTowardData,
        dynamics::CanonicalizeDynamics,
        emitted_content_projection::ProjectEmittedContent,
        initial_header::CanonicalizeInitialHeader,
        interpretation_lines::CollapseInterpretationLines,
        invisible_rest_measures::RemoveInvisibleRestMeasures,
        measure_grid::CanonicalizeMeasureGrid,
        multi_measure_rests::CompactMultiMeasureRests,
        null_lines::RemoveNullLines,
        omr::{RemoveLeadingBarlines, RewriteNonSerializableTandems, TandemColumn},
        redundant_state_interpretations::RemoveRedundantStateInterpretations,
        restore_final_staff_topology::RestoreFinalStaffTopology,
        spine_manipulator_lines::MergeAdjacentSpineManipulators,
        spine_operation_interpretations::MoveInterpretationsAcrossSpineOperations,
        superseded_interpretations::RemoveSupersededInterpretations,
        ties::CanonicalizeImmediateTies,
        NormalizationPass, NormalizeOptions,
    },
    serialize::{tandem_policy::TandemPolicy, ContentKind, ContentPolicy},
    KernDocument, MeterOutput,
};

use schedule::{Gate, PassId, SCHEDULE};

/// Upper bound on schedule repetitions that change the document without
/// shortening it. Passes expose work for each other, so the schedule repeats
/// until the document stops changing; output must be a fixpoint, or every
/// parse/normalize round trip rewrites further.
pub(crate) const FIXPOINT_LIMIT: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CanonError {
    #[error("normalization did not converge after {iterations} iterations")]
    DidNotConverge { iterations: usize },
}

/// Everything the driver needs to normalize for one output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Profile {
    policy: TandemPolicy,
    /// Handed to every pass constructor, exactly as each driver did before
    /// the schedule existed. For the OMR family these are also the gates.
    options: NormalizeOptions,
}

/// The OMR family's switches, read from the profile's options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Gates<'a>(&'a NormalizeOptions);

impl Gates<'_> {
    pub(crate) fn get(&self, gate: Gate) -> bool {
        match gate {
            Gate::ProjectToEmittedKern => self.0.project_to_emitted_kern,
            Gate::CompactMultiMeasureRests => self.0.compact_multi_measure_rests,
            Gate::StripLeadingBarlines => !self.0.preserve_leading_barlines,
            Gate::CanonicalizeMeasureGrid => self.0.canonicalize_measure_grid,
            Gate::RemoveInvisibleRestMeasures => self.0.remove_invisible_rest_measures,
            Gate::CanonicalizeBeamSpelling => self.0.canonicalize_beam_spelling,
        }
    }
}

impl Profile {
    /// Structural OMR normalization for an output whose tandems `policy`
    /// decides.
    pub(crate) fn omr(options: &NormalizeOptions, policy: TandemPolicy) -> Self {
        Self {
            policy,
            options: options.clone(),
        }
    }

    /// Transcoda V2 canonicalization, after lowering. The gates follow from
    /// the content table.
    pub(crate) fn transcoda_v2(policy: ContentPolicy) -> Self {
        Self::omr(
            &NormalizeOptions {
                compact_multi_measure_rests: policy.rewrites(ContentKind::MultiMeasureRest),
                preserve_leading_barlines: !policy.drops(ContentKind::LeadingBarline),
                canonicalize_measure_grid: policy.rewrites(ContentKind::SpineSlots),
                // Lowering already projected the content and removed
                // invisible-rest measures.
                project_to_emitted_kern: false,
                remove_invisible_rest_measures: false,
                canonicalize_beam_spelling: policy.rewrites(ContentKind::BeamSpelling),
            },
            policy.tandem_policy(),
        )
    }

    /// The passes this profile runs, in order.
    fn sequence(&self) -> impl Iterator<Item = PassId> + '_ {
        SCHEDULE
            .iter()
            .filter(move |row| row.runs_in_omr(&Gates(&self.options)))
            .map(|row| row.pass)
    }

    /// The tandems this profile keeps.
    fn tandem_column(&self) -> TandemColumn {
        TandemColumn::Output(self.policy)
    }

    /// How this profile spells meter declarations. Review keeps the source's
    /// spelling.
    fn meter_output(&self) -> MeterOutput {
        self.policy.meter_output()
    }

    fn build(&self, pass: PassId) -> Box<dyn NormalizationPass> {
        let options = || self.options.clone();
        match pass {
            PassId::ProjectEmittedContent => Box::new(ProjectEmittedContent::new(options())),
            PassId::RewriteNonSerializableTandems => Box::new(
                RewriteNonSerializableTandems::with_column(options(), self.tandem_column()),
            ),
            PassId::RemoveNullLines => Box::new(RemoveNullLines::new(options())),
            PassId::CollapseInterpretationLines => {
                Box::new(CollapseInterpretationLines::new(options()))
            }
            PassId::CanonicalizeInitialHeader => Box::new(CanonicalizeInitialHeader::new(
                options(),
                self.meter_output(),
            )),
            PassId::MoveControlTandemsTowardData => {
                Box::new(MoveControlTandemsTowardData::new(options()))
            }
            PassId::RemoveSupersededInterpretations => Box::new(
                RemoveSupersededInterpretations::new(options(), self.meter_output()),
            ),
            PassId::RemoveRedundantStateInterpretations => {
                Box::new(RemoveRedundantStateInterpretations::new(options()))
            }
            PassId::MoveInterpretationsAcrossSpineOperations => {
                Box::new(MoveInterpretationsAcrossSpineOperations::new(options()))
            }
            PassId::CollapseTrivialSplitMergeRegions => {
                Box::new(CollapseTrivialSplitMergeRegions::new(options()))
            }
            PassId::RestoreFinalStaffTopology => {
                Box::new(RestoreFinalStaffTopology::new(options()))
            }
            PassId::MergeAdjacentSpineManipulators => {
                Box::new(MergeAdjacentSpineManipulators::new(options()))
            }
            PassId::CompactMultiMeasureRests => Box::new(CompactMultiMeasureRests::new(options())),
            PassId::RemoveLeadingBarlines => Box::new(RemoveLeadingBarlines::new(options())),
            PassId::MoveMergesBeforeBarlines => Box::new(MoveMergesBeforeBarlines::new(options())),
            PassId::MoveSplitsAfterBarlines => Box::new(MoveSplitsAfterBarlines::new(options())),
            PassId::CollapseRedundantMergeSplitAroundBarlines => {
                Box::new(CollapseRedundantMergeSplitAroundBarlines::new(options()))
            }
            PassId::CanonicalizeMeasureGrid => {
                let policy = self.policy;
                Box::new(CanonicalizeMeasureGrid::omitting(policy))
            }
            PassId::RemoveInvisibleRestMeasures => {
                Box::new(RemoveInvisibleRestMeasures::new(options()))
            }
            PassId::CanonicalizeImmediateTies => {
                Box::new(CanonicalizeImmediateTies::new(options()))
            }
            PassId::CanonicalizeDynamics => Box::new(CanonicalizeDynamics::default()),
            PassId::CanonicalizeBeamSpelling => Box::new(CanonicalizeBeamSpelling),
        }
    }
}

/// Run `profile`'s passes until the document stops changing, returning the
/// number of repetitions. A convergence error leaves the attempted result in
/// `document`; transactional callers normalize a clone.
pub(crate) fn canonicalize(
    document: &mut KernDocument,
    profile: &Profile,
) -> Result<usize, CanonError> {
    canonicalize_with_limit(document, profile, FIXPOINT_LIMIT)
}

pub(crate) fn canonicalize_with_limit(
    document: &mut KernDocument,
    profile: &Profile,
    limit: usize,
) -> Result<usize, CanonError> {
    if limit == 0 {
        return Err(CanonError::DidNotConverge { iterations: 0 });
    }
    let passes: Vec<_> = profile.sequence().map(|pass| profile.build(pass)).collect();
    let mut stalled = 0;
    for iteration in 1.. {
        let before = document.clone();
        for pass in &passes {
            pass.normalize(document);
        }
        if *document == before {
            return Ok(iteration);
        }
        // A round that shortens the document is progress: one empty measure
        // removed per round lets the header absorb the next measure's
        // tandems, which makes that measure removable in turn. Only rounds
        // that change the document without shortening it count toward the
        // limit; the count never resets, so passes undoing each other still
        // fail, and the line count bounds the shortening rounds.
        if document.lines().len() >= before.lines().len() {
            stalled += 1;
            if stalled >= limit {
                return Err(CanonError::DidNotConverge {
                    iterations: iteration,
                });
            }
        }
    }
    unreachable!("the rounds end by converging or stalling")
}

#[cfg(test)]
mod tests {
    /// A blank page whose measures are separated by tandem changes loses
    /// one measure per round: removing the first empty measure brings the
    /// next measure's tandems into the header, which empties that measure.
    #[test]
    fn empty_measures_peeled_one_per_round_converge() {
        let mut document = KernDocument::parse(
            "*clefG2\n*k[]\n1ryy\n=\n*clefF4\n1ryy\n=\n*k[b-]\n1ryy\n=\n*clefC3\n1ryy\n=\n*clefG2\n1ryy\n=\n",
        )
        .unwrap();
        let rounds = canonicalize(
            &mut document,
            &Profile::omr(&NormalizeOptions::default(), TandemPolicy::Omr),
        )
        .unwrap();
        assert!(rounds > FIXPOINT_LIMIT, "{rounds}");
    }

    use super::*;
    use PassId as P;

    fn sequence(profile: &Profile) -> Vec<PassId> {
        profile.sequence().collect()
    }

    /// Transcribed from `run_omr_normalization_passes` before the schedule.
    fn omr_sequence(options: &NormalizeOptions) -> Vec<PassId> {
        let mut passes = Vec::new();
        if options.project_to_emitted_kern {
            passes.push(P::ProjectEmittedContent);
        }
        passes.extend([
            P::RewriteNonSerializableTandems,
            P::RemoveNullLines,
            P::CollapseInterpretationLines,
            P::CanonicalizeInitialHeader,
            P::RemoveNullLines,
            P::MoveControlTandemsTowardData,
            P::RemoveSupersededInterpretations,
            P::RemoveRedundantStateInterpretations,
            P::RemoveNullLines,
            P::MoveInterpretationsAcrossSpineOperations,
            P::CollapseTrivialSplitMergeRegions,
            P::RemoveNullLines,
            P::CollapseInterpretationLines,
            P::RemoveRedundantStateInterpretations,
            P::RestoreFinalStaffTopology,
            P::MergeAdjacentSpineManipulators,
        ]);
        if options.compact_multi_measure_rests {
            passes.push(P::CompactMultiMeasureRests);
        }
        if !options.preserve_leading_barlines {
            passes.push(P::RemoveLeadingBarlines);
        }
        passes.extend([
            P::MoveMergesBeforeBarlines,
            P::MoveSplitsAfterBarlines,
            P::CollapseRedundantMergeSplitAroundBarlines,
        ]);
        if !options.preserve_leading_barlines {
            passes.push(P::RemoveLeadingBarlines);
        }
        if options.canonicalize_measure_grid {
            passes.extend([
                P::CanonicalizeMeasureGrid,
                P::CollapseRedundantMergeSplitAroundBarlines,
                P::MergeAdjacentSpineManipulators,
            ]);
        }
        if options.remove_invisible_rest_measures {
            passes.extend([
                P::RemoveInvisibleRestMeasures,
                P::MoveMergesBeforeBarlines,
                P::MoveSplitsAfterBarlines,
                P::CollapseRedundantMergeSplitAroundBarlines,
            ]);
            if !options.preserve_leading_barlines {
                passes.push(P::RemoveLeadingBarlines);
            }
        }
        passes.extend([P::CanonicalizeImmediateTies, P::CanonicalizeDynamics]);
        if options.canonicalize_beam_spelling {
            passes.push(P::CanonicalizeBeamSpelling);
        }
        passes
    }

    #[test]
    fn omr_family_runs_the_omr_passes_in_order_for_every_gate_combination() {
        for bits in 0..64u8 {
            let options = NormalizeOptions {
                project_to_emitted_kern: bits & 1 != 0,
                compact_multi_measure_rests: bits & 2 != 0,
                preserve_leading_barlines: bits & 4 != 0,
                canonicalize_measure_grid: bits & 8 != 0,
                remove_invisible_rest_measures: bits & 16 != 0,
                canonicalize_beam_spelling: bits & 32 != 0,
            };
            assert_eq!(
                sequence(&Profile::omr(&options, TandemPolicy::Omr)),
                omr_sequence(&options),
                "{options:?}"
            );
        }
    }
}
