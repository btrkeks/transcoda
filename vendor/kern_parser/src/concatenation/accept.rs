use std::fs;

use crate::{
    transforms::{musical_bar_count, split_trailing_bars},
    KernDocument, SerializePreset,
};

use super::{
    compact_line_count,
    document::{concatenate_documents, has_musical_data},
    line_cap_for_staff_count,
    manifest::{SourceOrigin, SourceUnitKind, SourceUnitRef},
    ConcatenationError,
};

#[derive(Debug, Clone)]
pub(super) struct AcceptedCandidate {
    pub(super) unit_indices: Vec<usize>,
    pub(super) truncations: Vec<usize>,
    pub(super) contributed_line_counts: Vec<usize>,
    pub(super) origin_usage_ordinals: Vec<usize>,
    pub(super) contents: String,
    pub(super) line_count: usize,
    pub(super) truncation: AcceptedCandidateTruncation,
    pub(super) salvaged_tail: Option<SalvagedTail>,
}

#[derive(Debug, Clone)]
pub(super) struct SalvagedTail {
    pub(super) origin_index: usize,
    pub(super) document: KernDocument,
    pub(super) staff_count: usize,
    pub(super) parent_output_index: usize,
    pub(super) start_bar_index: usize,
    pub(super) salvaged_bars: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct AcceptedCandidateTruncation {
    pub(super) truncated_source_count: usize,
    pub(super) truncated_bars_total: usize,
    pub(super) truncated_chars_total: usize,
}

#[derive(Debug, Clone)]
pub(super) struct CandidateAcceptance {
    pub(super) candidate: Option<AcceptedCandidate>,
    pub(super) rejection_reason: Option<CandidateRejectionReason>,
    pub(super) rejection_detail: Option<String>,
}

impl CandidateAcceptance {
    fn accepted(candidate: AcceptedCandidate) -> Self {
        Self {
            candidate: Some(candidate),
            rejection_reason: None,
            rejection_detail: None,
        }
    }

    fn rejected(reason: CandidateRejectionReason) -> Self {
        Self {
            candidate: None,
            rejection_reason: Some(reason),
            rejection_detail: None,
        }
    }

    fn rejected_with_detail(reason: CandidateRejectionReason, detail: impl Into<String>) -> Self {
        Self {
            candidate: None,
            rejection_reason: Some(reason),
            rejection_detail: Some(detail.into()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CandidateRejectionReason {
    /// A snippet does not resolve, or the state at a seam cannot be restated
    /// on the next snippet's spines.
    SeamStateInvalid,
    GeneratedParseFailed,
    GeneratedTargetSerializationFailed,
    OverLineCapNoSalvageBars,
    OverLineCapSalvageFailed,
}

#[derive(Debug, Clone)]
enum CandidateGate {
    Accepted {
        contents: String,
        target_line_count: usize,
    },
    RetryWithSalvage,
    Rejected {
        reason: CandidateRejectionReason,
        detail: String,
    },
}

pub(super) struct CandidateAcceptor<'a> {
    origins: &'a [SourceOrigin],
    units: &'a [SourceUnitRef],
}

impl<'a> CandidateAcceptor<'a> {
    pub(super) fn new(origins: &'a [SourceOrigin], units: &'a [SourceUnitRef]) -> Self {
        Self { origins, units }
    }

    pub(super) fn accept(
        &mut self,
        unit_indices: Vec<usize>,
        origin_usage_ordinals: Vec<usize>,
        output_index: usize,
    ) -> Result<CandidateAcceptance, ConcatenationError> {
        let staff_count = self.units[unit_indices[0]].staff_count;
        line_cap_for_staff_count(staff_count)?;
        let original_documents = unit_indices
            .iter()
            .map(|unit_index| self.load_unit_document(*unit_index))
            .collect::<Result<Vec<_>, _>>()?;

        match self.try_accept_documents(&original_documents)? {
            CandidateGate::Accepted {
                contents,
                target_line_count,
                ..
            } => {
                let contributed_line_counts = match original_documents
                    .iter()
                    .map(training_line_count)
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(line_counts) => line_counts,
                    Err(error) => {
                        return Ok(CandidateAcceptance::rejected_with_detail(
                            CandidateRejectionReason::GeneratedTargetSerializationFailed,
                            format!("source target serialization failed: {error}"),
                        ));
                    }
                };
                Ok(CandidateAcceptance::accepted(AcceptedCandidate {
                    unit_indices,
                    truncations: vec![0; original_documents.len()],
                    contributed_line_counts,
                    origin_usage_ordinals,
                    line_count: target_line_count,
                    contents,
                    truncation: AcceptedCandidateTruncation::default(),
                    salvaged_tail: None,
                }))
            }
            CandidateGate::RetryWithSalvage => self.try_accept_with_salvage(
                unit_indices,
                origin_usage_ordinals,
                output_index,
                original_documents,
            ),
            CandidateGate::Rejected { reason, detail } => {
                Ok(CandidateAcceptance::rejected_with_detail(reason, detail))
            }
        }
    }

    fn load_unit_document(&self, unit_index: usize) -> Result<KernDocument, ConcatenationError> {
        let unit = &self.units[unit_index];
        if let Some(contents) = &unit.contents {
            return KernDocument::parse(contents).map_err(|source| ConcatenationError::Parse {
                path: self.origins[unit.origin_index].absolute_path.clone(),
                source,
            });
        }

        let origin = &self.origins[unit.origin_index];
        let input =
            fs::read_to_string(&origin.absolute_path).map_err(|source| ConcatenationError::Io {
                path: origin.absolute_path.clone(),
                source,
            })?;
        KernDocument::parse(&input).map_err(|source| ConcatenationError::Parse {
            path: origin.absolute_path.clone(),
            source,
        })
    }

    fn try_accept_with_salvage(
        &self,
        unit_indices: Vec<usize>,
        origin_usage_ordinals: Vec<usize>,
        output_index: usize,
        original_documents: Vec<KernDocument>,
    ) -> Result<CandidateAcceptance, ConcatenationError> {
        let last_position = original_documents.len() - 1;
        let last_document = &original_documents[last_position];
        let bar_count = musical_bar_count(last_document);
        if bar_count < 2 {
            return Ok(CandidateAcceptance::rejected(
                CandidateRejectionReason::OverLineCapNoSalvageBars,
            ));
        }

        for remove_last in 1..bar_count {
            let split = split_trailing_bars(last_document, remove_last)?;
            if !has_musical_data(&split.retained) || !has_musical_data(&split.removed) {
                continue;
            }
            let mut documents = original_documents.clone();
            documents[last_position] = split.retained.clone();
            let CandidateGate::Accepted {
                contents,
                target_line_count,
                ..
            } = self.try_accept_documents(&documents)?
            else {
                continue;
            };

            let original_len = match training_target(last_document) {
                Ok(target) => target.len(),
                Err(error) => {
                    return Ok(CandidateAcceptance::rejected_with_detail(
                        CandidateRejectionReason::GeneratedTargetSerializationFailed,
                        format!("source target serialization failed: {error}"),
                    ));
                }
            };
            let accepted_len = match training_target(&split.retained) {
                Ok(target) => target.len(),
                Err(_) => continue,
            };
            let mut truncations = vec![0; documents.len()];
            truncations[last_position] = remove_last;
            let contributed_line_counts = match documents
                .iter()
                .map(training_line_count)
                .collect::<Result<Vec<_>, _>>()
            {
                Ok(line_counts) => line_counts,
                Err(_) => continue,
            };
            let salvaged_bars = musical_bar_count(&split.removed);
            let salvaged_tail = SalvagedTail {
                origin_index: self.units[unit_indices[last_position]].origin_index,
                staff_count: self.units[unit_indices[last_position]].staff_count,
                document: split.removed,
                parent_output_index: output_index,
                start_bar_index: bar_count.saturating_sub(salvaged_bars),
                salvaged_bars,
            };

            return Ok(CandidateAcceptance::accepted(AcceptedCandidate {
                unit_indices,
                truncations,
                contributed_line_counts,
                origin_usage_ordinals,
                contents,
                line_count: target_line_count,
                truncation: AcceptedCandidateTruncation {
                    truncated_source_count: 1,
                    truncated_bars_total: remove_last,
                    truncated_chars_total: original_len.saturating_sub(accepted_len),
                },
                salvaged_tail: Some(salvaged_tail),
            }));
        }

        Ok(CandidateAcceptance::rejected(
            CandidateRejectionReason::OverLineCapSalvageFailed,
        ))
    }

    fn try_accept_documents(
        &self,
        documents: &[KernDocument],
    ) -> Result<CandidateGate, ConcatenationError> {
        let document = match documents {
            [document] => document.clone(),
            _ => match concatenate_documents(documents) {
                Ok(document) => document,
                Err(error @ (ConcatenationError::State(_) | ConcatenationError::Seam(_))) => {
                    return Ok(CandidateGate::Rejected {
                        reason: CandidateRejectionReason::SeamStateInvalid,
                        detail: error.to_string(),
                    });
                }
                Err(error) => return Err(error),
            },
        };
        let contents = match SerializePreset::HumdrumKern.try_serialize_document(&document) {
            Ok(contents) => contents,
            Err(error) => {
                return Ok(CandidateGate::Rejected {
                    reason: CandidateRejectionReason::GeneratedTargetSerializationFailed,
                    detail: format!("Humdrum serialization failed: {error}"),
                });
            }
        };
        if let Err(error) = KernDocument::parse(&contents) {
            return Ok(CandidateGate::Rejected {
                reason: CandidateRejectionReason::GeneratedParseFailed,
                detail: error.to_string(),
            });
        }
        let target_line_count = match training_line_count(&document) {
            Ok(line_count) => line_count,
            Err(error) => {
                return Ok(CandidateGate::Rejected {
                    reason: CandidateRejectionReason::GeneratedTargetSerializationFailed,
                    detail: format!("target serialization failed: {error}"),
                });
            }
        };
        if target_line_count > line_cap_for_staff_count(document.staff_count())? {
            return Ok(CandidateGate::RetryWithSalvage);
        }
        Ok(CandidateGate::Accepted {
            contents,
            target_line_count,
        })
    }
}

fn training_target(document: &KernDocument) -> Result<String, crate::SerializeError> {
    SerializePreset::TranscodaV2.try_serialize_document(document)
}

fn training_line_count(document: &KernDocument) -> Result<usize, crate::SerializeError> {
    training_target(document).map(|target| compact_line_count(&target))
}

pub(super) fn salvaged_tail_kind(tail: &SalvagedTail) -> SourceUnitKind {
    SourceUnitKind::SalvagedTail {
        parent_output_index: tail.parent_output_index,
        start_bar_index: tail.start_bar_index,
        salvaged_bars: tail.salvaged_bars,
    }
}
