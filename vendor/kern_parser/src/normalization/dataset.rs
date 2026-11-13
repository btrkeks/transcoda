//! Dataset source-normalization profile used by the stage-4 `normalize` command.
//!
//! This profile starts from OMR structural normalization, then applies
//! source-corpus cleanups that should be materialized before quality filtering,
//! concatenation, rendering, and target projection.

use crate::{serialize::TandemPolicy, KernDocument};

use super::{
    invisible_rest_compaction::{CompactConsecutiveInvisibleRests, Omitted},
    invisible_rest_padding::NormalizeInvisibleRestPadding,
    null_lines::RemoveNullLines,
    omr::normalize_for_omr,
    NormalizationPass,
};

pub fn normalize_for_dataset_normalization(document: &mut KernDocument) {
    normalize_for_omr(document);
    normalize_dataset_source_cleanups(document);
}

pub fn normalize_dataset_source_cleanups(document: &mut KernDocument) {
    // Stage-4 cleanups run before the output preset is chosen, and presets
    // such as humdrum-kern print pedals, section labels and local comments:
    // only what the OMR policy drops may sit inside a compacted run.
    CompactConsecutiveInvisibleRests::omitting(Omitted {
        tandems: TandemPolicy::Omr,
        local_comments: false,
    })
    .normalize(document);
    RemoveNullLines::new(Default::default()).normalize(document);
    NormalizeInvisibleRestPadding::new().normalize(document);
}

pub fn normalized_for_dataset_normalization(document: &KernDocument) -> KernDocument {
    let mut normalized = document.clone();
    normalize_for_dataset_normalization(&mut normalized);
    normalized
}
