use std::collections::{BTreeMap, BTreeSet};

use super::manifest::{SourceUnitKind, SourceUnitRef};

const MAX_CANDIDATE_PROPOSALS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct CandidateKey {
    pub(super) unit_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CandidateRank {
    pub(super) source_fragment_count: usize,
    pub(super) second_use_origin_count: usize,
    pub(super) split_chunk_count: usize,
    pub(super) seam_cost: usize,
    pub(super) distance_to_bucket_midpoint: usize,
    pub(super) stable_tiebreaker: String,
}

#[derive(Debug, Clone)]
pub(super) struct CandidateProposal {
    pub(super) unit_indices: Vec<usize>,
    pub(super) rank: CandidateRank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct LineBucket {
    pub(super) index: usize,
    pub(super) start: usize,
    pub(super) end: usize,
}

pub(super) fn buckets_for_range(
    min_line_count: usize,
    line_cap: usize,
    bucket_width: usize,
) -> Vec<LineBucket> {
    if min_line_count > line_cap {
        return vec![LineBucket {
            index: 0,
            start: line_cap,
            end: line_cap,
        }];
    }

    let mut buckets = Vec::new();
    let mut start = min_line_count;
    while start <= line_cap {
        let end = start.saturating_add(bucket_width - 1).min(line_cap);
        buckets.push(LineBucket {
            index: buckets.len(),
            start,
            end,
        });
        if end == line_cap {
            break;
        }
        start = end + 1;
    }
    buckets
}

pub(super) fn buckets_by_need(
    staff_count: usize,
    histogram: &BTreeMap<(usize, usize), usize>,
    buckets: &[LineBucket],
    unfillable: &BTreeSet<(usize, usize)>,
) -> Vec<LineBucket> {
    let mut buckets = buckets
        .iter()
        .copied()
        .filter(|bucket| !unfillable.contains(&(staff_count, bucket.index)))
        .collect::<Vec<_>>();
    buckets.sort_by_key(|bucket| {
        (
            histogram
                .get(&(staff_count, bucket.index))
                .copied()
                .unwrap_or_default(),
            bucket.index,
        )
    });
    buckets
}

pub(super) fn bucket_for_line_count(line_count: usize, buckets: &[LineBucket]) -> usize {
    buckets
        .iter()
        .find(|bucket| (bucket.start..=bucket.end).contains(&line_count))
        .or_else(|| buckets.last())
        .map(|bucket| bucket.index)
        .unwrap_or_default()
}

pub(super) fn candidate_source_sets(
    units: &[SourceUnitRef],
    eligible: &[usize],
    bucket: LineBucket,
    line_cap: usize,
    max_fragments: usize,
    origin_use_counts: &[usize],
    allow_second_use: bool,
) -> Vec<CandidateProposal> {
    let index = LineCountIndex::new(units, eligible, line_cap);
    for fragment_count in 1..=max_fragments {
        let mut proposals = Vec::new();
        let mut selected = Vec::new();
        collect_candidates(
            units,
            &index,
            bucket,
            line_cap,
            fragment_count,
            0,
            0,
            &mut selected,
            origin_use_counts,
            allow_second_use,
            &mut proposals,
        );
        if !proposals.is_empty() {
            proposals.sort_by(|left, right| left.rank.cmp(&right.rank));
            return proposals;
        }
    }
    Vec::new()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn anchored_candidate_source_sets(
    units: &[SourceUnitRef],
    eligible: &[usize],
    required_unit: usize,
    bucket: LineBucket,
    line_cap: usize,
    max_fragments: usize,
    origin_use_counts: &[usize],
    allow_second_use: bool,
) -> Vec<CandidateProposal> {
    let index = LineCountIndex::new(units, eligible, line_cap);
    let Some(required_position) = index.position(required_unit) else {
        return Vec::new();
    };

    for fragment_count in 1..=max_fragments {
        let mut proposals = Vec::new();
        let mut selected = vec![required_unit];
        collect_candidates(
            units,
            &index,
            bucket,
            line_cap,
            fragment_count,
            required_position + 1,
            units[required_unit].line_count,
            &mut selected,
            origin_use_counts,
            allow_second_use,
            &mut proposals,
        );
        if !proposals.is_empty() {
            proposals.sort_by(|left, right| left.rank.cmp(&right.rank));
            return proposals;
        }
    }
    Vec::new()
}

struct LineCountIndex<'a> {
    eligible: &'a [usize],
    by_line_count: Vec<Vec<usize>>,
    positions: Vec<Option<usize>>,
    min_line_count: Option<usize>,
    max_line_count: usize,
}

impl<'a> LineCountIndex<'a> {
    fn new(units: &[SourceUnitRef], eligible: &'a [usize], line_cap: usize) -> Self {
        let mut by_line_count = vec![Vec::new(); line_cap + 1];
        let mut positions = vec![None; units.len()];
        let mut min_line_count = None;
        let mut max_line_count = 0;

        for (position, unit_index) in eligible.iter().copied().enumerate() {
            positions[unit_index] = Some(position);
            let line_count = units[unit_index].line_count;
            if line_count > line_cap {
                continue;
            }
            by_line_count[line_count].push(unit_index);
            min_line_count = Some(
                min_line_count
                    .map(|current: usize| current.min(line_count))
                    .unwrap_or(line_count),
            );
            max_line_count = max_line_count.max(line_count);
        }

        Self {
            eligible,
            by_line_count,
            positions,
            min_line_count,
            max_line_count,
        }
    }

    fn position(&self, unit_index: usize) -> Option<usize> {
        self.positions.get(unit_index).copied().flatten()
    }

    fn units_with_line_count(&self, line_count: usize) -> &[usize] {
        self.by_line_count
            .get(line_count)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_candidates(
    units: &[SourceUnitRef],
    index: &LineCountIndex<'_>,
    bucket: LineBucket,
    line_cap: usize,
    target_len: usize,
    start: usize,
    current_line_count: usize,
    selected: &mut Vec<usize>,
    origin_use_counts: &[usize],
    allow_second_use: bool,
    proposals: &mut Vec<CandidateProposal>,
) {
    if selected.len() == target_len {
        if current_line_count > line_cap
            || !(bucket.start..=bucket.end).contains(&current_line_count)
        {
            return;
        }
        if duplicate_origin_in_output(units, selected) {
            return;
        }

        let rank = rank_candidate(units, selected, bucket, origin_use_counts, allow_second_use);
        insert_bounded_proposal(
            CandidateProposal {
                unit_indices: selected.clone(),
                rank,
            },
            proposals,
        );
        return;
    }

    let remaining = target_len - selected.len();
    if remaining == 1 {
        collect_final_fragment(
            units,
            index,
            bucket,
            line_cap,
            start,
            current_line_count,
            selected,
            origin_use_counts,
            allow_second_use,
            proposals,
        );
        return;
    }

    let Some(min_line_count) = index.min_line_count else {
        return;
    };
    let remaining_after_next = remaining - 1;
    let min_future_line_count = min_line_count.saturating_mul(remaining_after_next);
    let max_future_line_count = index.max_line_count.saturating_mul(remaining_after_next);

    for offset in start..index.eligible.len() {
        let unit_index = index.eligible[offset];
        let next_line_count = current_line_count + units[unit_index].line_count;
        if next_line_count > line_cap {
            continue;
        }
        if next_line_count + min_future_line_count > bucket.end {
            continue;
        }
        if next_line_count + max_future_line_count < bucket.start {
            continue;
        }
        selected.push(unit_index);
        collect_candidates(
            units,
            index,
            bucket,
            line_cap,
            target_len,
            offset + 1,
            next_line_count,
            selected,
            origin_use_counts,
            allow_second_use,
            proposals,
        );
        selected.pop();
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_final_fragment(
    units: &[SourceUnitRef],
    index: &LineCountIndex<'_>,
    bucket: LineBucket,
    line_cap: usize,
    start: usize,
    current_line_count: usize,
    selected: &mut Vec<usize>,
    origin_use_counts: &[usize],
    allow_second_use: bool,
    proposals: &mut Vec<CandidateProposal>,
) {
    if current_line_count > line_cap {
        return;
    }

    let min_line_count = bucket.start.saturating_sub(current_line_count);
    let max_line_count = bucket
        .end
        .saturating_sub(current_line_count)
        .min(line_cap - current_line_count);
    if min_line_count > max_line_count {
        return;
    }

    for line_count in min_line_count..=max_line_count {
        for unit_index in index.units_with_line_count(line_count) {
            let Some(position) = index.position(*unit_index) else {
                continue;
            };
            if position < start {
                continue;
            }
            selected.push(*unit_index);
            consider_candidate(
                units,
                selected,
                bucket,
                origin_use_counts,
                allow_second_use,
                proposals,
            );
            selected.pop();
        }
    }
}

fn consider_candidate(
    units: &[SourceUnitRef],
    selected: &[usize],
    bucket: LineBucket,
    origin_use_counts: &[usize],
    allow_second_use: bool,
    proposals: &mut Vec<CandidateProposal>,
) {
    if duplicate_origin_in_output(units, selected) {
        return;
    }

    let rank = rank_candidate(units, selected, bucket, origin_use_counts, allow_second_use);
    insert_bounded_proposal(
        CandidateProposal {
            unit_indices: selected.to_vec(),
            rank,
        },
        proposals,
    );
}

fn insert_bounded_proposal(proposal: CandidateProposal, proposals: &mut Vec<CandidateProposal>) {
    if proposals.len() < MAX_CANDIDATE_PROPOSALS {
        proposals.push(proposal);
        return;
    }

    let Some((worst_index, worst)) = proposals
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.rank.cmp(&right.rank))
    else {
        return;
    };
    if proposal.rank < worst.rank {
        proposals[worst_index] = proposal;
    }
}

fn duplicate_origin_in_output(units: &[SourceUnitRef], selected: &[usize]) -> bool {
    let mut origins = BTreeSet::new();
    selected
        .iter()
        .any(|index| !origins.insert(units[*index].origin_index))
}

fn rank_candidate(
    units: &[SourceUnitRef],
    selected: &[usize],
    bucket: LineBucket,
    origin_use_counts: &[usize],
    allow_second_use: bool,
) -> CandidateRank {
    let target_mid = (bucket.start + bucket.end) / 2;
    let line_count = selected
        .iter()
        .map(|index| units[*index].line_count)
        .sum::<usize>();
    let second_use_origin_count = selected
        .iter()
        .filter(|index| origin_use_counts[units[**index].origin_index] > 0)
        .count();
    let split_chunk_count = selected
        .iter()
        .filter(|index| matches!(units[**index].kind, SourceUnitKind::SplitChunk { .. }))
        .count();
    let stable_tiebreaker = selected
        .iter()
        .map(|index| units[*index].relative_label.as_str())
        .collect::<Vec<_>>()
        .join("|");

    CandidateRank {
        source_fragment_count: selected.len(),
        second_use_origin_count: if allow_second_use {
            second_use_origin_count
        } else {
            0
        },
        split_chunk_count,
        seam_cost: seam_cost(units, selected),
        distance_to_bucket_midpoint: line_count.abs_diff(target_mid),
        stable_tiebreaker,
    }
}

fn seam_cost(units: &[SourceUnitRef], selected: &[usize]) -> usize {
    selected
        .windows(2)
        .map(|pair| {
            let left = &units[pair[0]];
            let right = &units[pair[1]];
            if left.origin_index == right.origin_index
                && adjacent_split_chunks(&left.kind, &right.kind)
            {
                0
            } else {
                1
            }
        })
        .sum()
}

fn adjacent_split_chunks(left: &SourceUnitKind, right: &SourceUnitKind) -> bool {
    matches!(
        (left, right),
        (
            SourceUnitKind::SplitChunk {
                chunk_index: left_index,
                ..
            },
            SourceUnitKind::SplitChunk {
                chunk_index: right_index,
                ..
            }
        ) if *left_index + 1 == *right_index
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn original_unit(index: usize, label: String, line_count: usize) -> SourceUnitRef {
        original_unit_with_origin(index, index, label, line_count)
    }

    fn original_unit_with_origin(
        index: usize,
        origin_index: usize,
        label: String,
        line_count: usize,
    ) -> SourceUnitRef {
        SourceUnitRef {
            unit_index: index,
            origin_index,
            kind: SourceUnitKind::Original,
            relative_label: label,
            contents: None,
            staff_count: 1,
            line_count,
        }
    }

    fn assert_sorted_by_rank(proposals: &[CandidateProposal]) {
        assert!(proposals
            .windows(2)
            .all(|pair| pair[0].rank <= pair[1].rank));
    }

    #[test]
    fn candidate_source_sets_keeps_best_64_singletons() {
        let units = (0..80)
            .map(|index| original_unit(index, format!("unit_{index:03}"), 10))
            .collect::<Vec<_>>();
        let eligible = (0..units.len()).collect::<Vec<_>>();
        let origin_use_counts = vec![0; units.len()];

        let proposals = candidate_source_sets(
            &units,
            &eligible,
            LineBucket {
                index: 0,
                start: 10,
                end: 10,
            },
            10,
            3,
            &origin_use_counts,
            true,
        );

        assert_eq!(proposals.len(), MAX_CANDIDATE_PROPOSALS);
        assert!(proposals
            .iter()
            .all(|proposal| proposal.unit_indices.len() == 1));
        assert_sorted_by_rank(&proposals);
        assert_eq!(proposals[0].rank.stable_tiebreaker, "unit_000");
        assert_eq!(proposals[63].rank.stable_tiebreaker, "unit_063");
    }

    #[test]
    fn candidate_source_sets_keeps_best_64_pairs() {
        let units = (0..80)
            .map(|index| original_unit(index, format!("unit_{index:03}"), 10))
            .collect::<Vec<_>>();
        let eligible = (0..units.len()).collect::<Vec<_>>();
        let origin_use_counts = vec![0; units.len()];

        let proposals = candidate_source_sets(
            &units,
            &eligible,
            LineBucket {
                index: 0,
                start: 20,
                end: 20,
            },
            20,
            3,
            &origin_use_counts,
            true,
        );

        assert_eq!(proposals.len(), MAX_CANDIDATE_PROPOSALS);
        assert!(proposals
            .iter()
            .all(|proposal| proposal.unit_indices.len() == 2));
        assert_sorted_by_rank(&proposals);
        assert_eq!(proposals[0].rank.stable_tiebreaker, "unit_000|unit_001");
    }

    #[test]
    fn candidate_source_sets_finds_sparse_line_count_complements() {
        let units = vec![
            original_unit(0, "a".to_string(), 100),
            original_unit(1, "b".to_string(), 17),
            original_unit(2, "c".to_string(), 130),
            original_unit(3, "d".to_string(), 70),
            original_unit(4, "e".to_string(), 53),
        ];
        let eligible = (0..units.len()).collect::<Vec<_>>();
        let origin_use_counts = vec![0; units.len()];

        let proposals = candidate_source_sets(
            &units,
            &eligible,
            LineBucket {
                index: 0,
                start: 200,
                end: 200,
            },
            200,
            2,
            &origin_use_counts,
            true,
        );

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].unit_indices, vec![2, 3]);
        assert_eq!(proposals[0].rank.distance_to_bucket_midpoint, 0);
    }

    #[test]
    fn candidate_source_sets_rejects_same_origin_pairs() {
        let units = vec![
            original_unit_with_origin(0, 0, "same_a".to_string(), 10),
            original_unit_with_origin(1, 0, "same_b".to_string(), 10),
            original_unit_with_origin(2, 1, "other".to_string(), 10),
        ];
        let eligible = (0..units.len()).collect::<Vec<_>>();
        let origin_use_counts = vec![0, 0];

        let proposals = candidate_source_sets(
            &units,
            &eligible,
            LineBucket {
                index: 0,
                start: 20,
                end: 20,
            },
            20,
            2,
            &origin_use_counts,
            true,
        );

        assert_eq!(proposals.len(), 2);
        assert!(proposals
            .iter()
            .all(|proposal| proposal.unit_indices != vec![0, 1]));
    }

    #[test]
    fn candidate_source_sets_prefers_unused_origins_when_second_use_is_allowed() {
        let units = vec![
            original_unit(0, "aaa_used".to_string(), 10),
            original_unit(1, "zzz_unused".to_string(), 10),
        ];
        let eligible = vec![0, 1];
        let origin_use_counts = vec![1, 0];

        let proposals = candidate_source_sets(
            &units,
            &eligible,
            LineBucket {
                index: 0,
                start: 10,
                end: 10,
            },
            10,
            1,
            &origin_use_counts,
            true,
        );

        assert_eq!(proposals.len(), 2);
        assert_sorted_by_rank(&proposals);
        assert_eq!(proposals[0].unit_indices, vec![1]);
        assert_eq!(proposals[1].unit_indices, vec![0]);
    }

    #[test]
    fn anchored_candidate_source_sets_only_returns_required_unit_candidates() {
        let units = (0..12)
            .map(|index| original_unit(index, format!("unit_{index:03}"), 10))
            .collect::<Vec<_>>();
        let eligible = (0..units.len()).collect::<Vec<_>>();
        let origin_use_counts = vec![0; units.len()];

        let proposals = anchored_candidate_source_sets(
            &units,
            &eligible,
            5,
            LineBucket {
                index: 0,
                start: 30,
                end: 30,
            },
            30,
            3,
            &origin_use_counts,
            true,
        );

        assert!(!proposals.is_empty());
        assert!(proposals
            .iter()
            .all(|proposal| proposal.unit_indices.contains(&5)));
        assert!(proposals
            .iter()
            .all(|proposal| proposal.unit_indices.len() == 3));
        assert_sorted_by_rank(&proposals);
    }
}
