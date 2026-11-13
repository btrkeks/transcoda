//! Normalization schedule for Transcoda data.
use super::Gates;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PassId {
    ProjectEmittedContent,
    RewriteNonSerializableTandems,
    RemoveNullLines,
    CollapseInterpretationLines,
    CanonicalizeInitialHeader,
    MoveControlTandemsTowardData,
    RemoveSupersededInterpretations,
    RemoveRedundantStateInterpretations,
    MoveInterpretationsAcrossSpineOperations,
    CollapseTrivialSplitMergeRegions,
    RestoreFinalStaffTopology,
    MergeAdjacentSpineManipulators,
    CompactMultiMeasureRests,
    RemoveLeadingBarlines,
    MoveMergesBeforeBarlines,
    MoveSplitsAfterBarlines,
    CollapseRedundantMergeSplitAroundBarlines,
    CanonicalizeMeasureGrid,
    RemoveInvisibleRestMeasures,
    CanonicalizeImmediateTies,
    CanonicalizeDynamics,
    CanonicalizeBeamSpelling,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gate {
    ProjectToEmittedKern,
    CompactMultiMeasureRests,
    StripLeadingBarlines,
    CanonicalizeMeasureGrid,
    RemoveInvisibleRestMeasures,
    CanonicalizeBeamSpelling,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OmrCell {
    Run,
    When(Gate),
    WhenAll(&'static [Gate]),
}
pub(crate) struct Row {
    pub(crate) pass: PassId,
    omr: OmrCell,
}
impl Row {
    pub(crate) fn runs_in_omr(&self, gates: &Gates) -> bool {
        match self.omr {
            OmrCell::Run => true,
            OmrCell::When(g) => gates.get(g),
            OmrCell::WhenAll(gs) => gs.iter().all(|g| gates.get(*g)),
        }
    }
}
use OmrCell::{Run as ORun, When, WhenAll};
use PassId as P;
pub(crate) const SCHEDULE: &[Row] = &[
    Row {
        pass: P::ProjectEmittedContent,
        omr: When(Gate::ProjectToEmittedKern),
    },
    Row {
        pass: P::RewriteNonSerializableTandems,
        omr: ORun,
    },
    Row {
        pass: P::RemoveNullLines,
        omr: ORun,
    },
    Row {
        pass: P::CollapseInterpretationLines,
        omr: ORun,
    },
    Row {
        pass: P::CanonicalizeInitialHeader,
        omr: ORun,
    },
    Row {
        pass: P::RemoveNullLines,
        omr: ORun,
    },
    Row {
        pass: P::MoveControlTandemsTowardData,
        omr: ORun,
    },
    Row {
        pass: P::RemoveSupersededInterpretations,
        omr: ORun,
    },
    Row {
        pass: P::RemoveRedundantStateInterpretations,
        omr: ORun,
    },
    Row {
        pass: P::RemoveNullLines,
        omr: ORun,
    },
    Row {
        pass: P::MoveInterpretationsAcrossSpineOperations,
        omr: ORun,
    },
    Row {
        pass: P::CollapseTrivialSplitMergeRegions,
        omr: ORun,
    },
    Row {
        pass: P::RemoveNullLines,
        omr: ORun,
    },
    Row {
        pass: P::CollapseInterpretationLines,
        omr: ORun,
    },
    Row {
        pass: P::RemoveRedundantStateInterpretations,
        omr: ORun,
    },
    Row {
        pass: P::RestoreFinalStaffTopology,
        omr: ORun,
    },
    Row {
        pass: P::MergeAdjacentSpineManipulators,
        omr: ORun,
    },
    Row {
        pass: P::CompactMultiMeasureRests,
        omr: When(Gate::CompactMultiMeasureRests),
    },
    Row {
        pass: P::RemoveLeadingBarlines,
        omr: When(Gate::StripLeadingBarlines),
    },
    Row {
        pass: P::MoveMergesBeforeBarlines,
        omr: ORun,
    },
    Row {
        pass: P::MoveSplitsAfterBarlines,
        omr: ORun,
    },
    Row {
        pass: P::CollapseRedundantMergeSplitAroundBarlines,
        omr: ORun,
    },
    Row {
        pass: P::RemoveLeadingBarlines,
        omr: When(Gate::StripLeadingBarlines),
    },
    Row {
        pass: P::CanonicalizeMeasureGrid,
        omr: When(Gate::CanonicalizeMeasureGrid),
    },
    Row {
        pass: P::CollapseRedundantMergeSplitAroundBarlines,
        omr: When(Gate::CanonicalizeMeasureGrid),
    },
    Row {
        pass: P::MergeAdjacentSpineManipulators,
        omr: When(Gate::CanonicalizeMeasureGrid),
    },
    Row {
        pass: P::RemoveInvisibleRestMeasures,
        omr: When(Gate::RemoveInvisibleRestMeasures),
    },
    Row {
        pass: P::MoveMergesBeforeBarlines,
        omr: When(Gate::RemoveInvisibleRestMeasures),
    },
    Row {
        pass: P::MoveSplitsAfterBarlines,
        omr: When(Gate::RemoveInvisibleRestMeasures),
    },
    Row {
        pass: P::CollapseRedundantMergeSplitAroundBarlines,
        omr: When(Gate::RemoveInvisibleRestMeasures),
    },
    Row {
        pass: P::RemoveLeadingBarlines,
        omr: WhenAll(&[
            Gate::RemoveInvisibleRestMeasures,
            Gate::StripLeadingBarlines,
        ]),
    },
    Row {
        pass: P::CanonicalizeImmediateTies,
        omr: ORun,
    },
    Row {
        pass: P::CanonicalizeDynamics,
        omr: ORun,
    },
    Row {
        pass: P::CanonicalizeBeamSpelling,
        omr: When(Gate::CanonicalizeBeamSpelling),
    },
];
