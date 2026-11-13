use std::{
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashSet},
    env,
    fs::{self, File},
    io::Write,
    path::Path,
    time::Instant,
};

use super::{
    accept::{salvaged_tail_kind, AcceptedCandidate, CandidateAcceptance, CandidateAcceptor},
    balancer::{
        anchored_candidate_source_sets, bucket_for_line_count, buckets_by_need, buckets_for_range,
        candidate_source_sets, CandidateKey, CandidateProposal, LineBucket,
    },
    filename::{manifest_sources, output_identity},
    line_cap_for_staff_count,
    manifest::{push_unit, SourceOrigin, SourceUnitRef},
    ConcatenationBalancingStats, ConcatenationError, ConcatenationOutputManifestRecord,
    ConcatenationProgress, ConcatenationSourceUsageReason, ConcatenationSourceUsageRecord,
    ConcatenationSourceUsageStatus, ConcatenationTruncationStats, ConcatenationUsageStats,
    OutputManifestSink,
};

const DEBUG_JSONL_ENV: &str = "KERN_PARSER_CONCAT_DEBUG_JSONL";
const DEBUG_OUTPUT_START_ENV: &str = "KERN_PARSER_CONCAT_DEBUG_OUTPUT_START";
const DEBUG_SLOW_MS_ENV: &str = "KERN_PARSER_CONCAT_DEBUG_SLOW_MS";
const DEFAULT_DEBUG_SLOW_MS: u128 = 250;
const COVERAGE_CANDIDATE_POOL_LIMIT: usize = 96;
const BALANCING_MAX_FRAGMENTS_PER_OUTPUT: usize = 2;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct ConcatenationMetrics {
    pub(super) output_count: usize,
    pub(super) truncation: ConcatenationTruncationStats,
    pub(super) usage: ConcatenationUsageStats,
    pub(super) balancing: ConcatenationBalancingStats,
    pub(super) output_manifest: Vec<ConcatenationOutputManifestRecord>,
    pub(super) source_usage: Vec<ConcatenationSourceUsageRecord>,
}

impl ConcatenationMetrics {
    fn record_output(&mut self, staff_count: usize, candidate: &AcceptedCandidate) {
        self.output_count += 1;
        *self
            .balancing
            .fragment_count_histogram
            .entry(candidate.unit_indices.len())
            .or_default() += 1;
        if candidate.truncation.truncated_source_count == 0 {
            return;
        }

        self.truncation.truncated_output_count += 1;
        self.truncation.truncated_source_count += candidate.truncation.truncated_source_count;
        self.truncation.truncated_bars_total += candidate.truncation.truncated_bars_total;
        self.truncation.truncated_chars_total += candidate.truncation.truncated_chars_total;
        *self
            .truncation
            .truncated_chars_by_staff_count
            .entry(staff_count)
            .or_default() += candidate.truncation.truncated_chars_total;
    }
}

struct GenerationState<'a, R, M> {
    origins: &'a [SourceOrigin],
    units: Vec<SourceUnitRef>,
    output_dir: &'a Path,
    target_count: Option<usize>,
    min_line_count: usize,
    bucket_width: usize,
    max_origin_uses: usize,
    max_fragments_per_output: usize,
    progress: &'a R,
    manifest_sink: &'a mut M,
    metrics: ConcatenationMetrics,
    origin_rejections: Vec<Option<ConcatenationSourceUsageReason>>,
    origin_first_uses: Vec<Option<OriginFirstUse>>,
    origin_use_counts: Vec<usize>,
    used_origin_count: usize,
    unit_use_counts: Vec<usize>,
    unit_counts_by_origin: Vec<usize>,
    histogram: BTreeMap<(usize, usize), usize>,
    unfillable: BTreeSet<(usize, usize)>,
    failed: HashSet<CandidateKey>,
    emitted: HashSet<CandidateKey>,
    debug: DebugTrace,
}

#[derive(Debug, Clone)]
struct OriginFirstUse {
    output_path: String,
    position: usize,
    source_count: usize,
    truncated_bars: usize,
}

impl<'a, R: ConcatenationProgress, M: OutputManifestSink> GenerationState<'a, R, M> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        origins: &'a [SourceOrigin],
        units: Vec<SourceUnitRef>,
        origin_rejections: Vec<Option<ConcatenationSourceUsageReason>>,
        split_chunk_count: usize,
        output_dir: &'a Path,
        target_count: Option<usize>,
        min_line_count: usize,
        bucket_width: usize,
        max_origin_uses: usize,
        max_fragments_per_output: usize,
        progress: &'a R,
        manifest_sink: &'a mut M,
    ) -> Self {
        let mut metrics = ConcatenationMetrics::default();
        metrics.balancing.split_chunk_count = split_chunk_count;
        let unit_count = units.len();
        let mut unit_counts_by_origin = vec![0; origins.len()];
        for unit in &units {
            unit_counts_by_origin[unit.origin_index] += 1;
        }
        Self {
            origins,
            units,
            output_dir,
            target_count,
            min_line_count,
            bucket_width,
            max_origin_uses,
            max_fragments_per_output,
            progress,
            manifest_sink,
            metrics,
            origin_rejections,
            origin_first_uses: vec![None; origins.len()],
            origin_use_counts: vec![0; origins.len()],
            used_origin_count: 0,
            unit_use_counts: vec![0; unit_count],
            unit_counts_by_origin,
            histogram: BTreeMap::new(),
            unfillable: BTreeSet::new(),
            failed: HashSet::new(),
            emitted: HashSet::new(),
            debug: DebugTrace::from_env(),
        }
    }

    fn target_reached(&self) -> bool {
        self.target_count
            .is_some_and(|target| self.metrics.output_count >= target)
    }

    fn accept(
        &mut self,
        unit_indices: Vec<usize>,
    ) -> Result<CandidateAcceptance, ConcatenationError> {
        let origin_usage_ordinals = unit_indices
            .iter()
            .map(|unit_index| self.origin_use_counts[self.units[*unit_index].origin_index] + 1)
            .collect::<Vec<_>>();
        CandidateAcceptor::new(self.origins, &self.units).accept(
            unit_indices,
            origin_usage_ordinals,
            self.metrics.output_count + 1,
        )
    }

    fn emit_candidate(
        &mut self,
        staff_count: usize,
        candidate: &AcceptedCandidate,
    ) -> Result<bool, ConcatenationError> {
        let output_index = self.metrics.output_count + 1;
        let identity = output_identity(output_index, &self.units, candidate);
        let path = self.output_dir.join(&identity.filename);
        let output_path = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        fs::write(&path, &candidate.contents).map_err(|source| ConcatenationError::Io {
            path: path.clone(),
            source,
        })?;

        for unit_index in &candidate.unit_indices {
            let position = candidate
                .unit_indices
                .iter()
                .position(|index| index == unit_index)
                .unwrap_or_default();
            self.unit_use_counts[*unit_index] += 1;
            let origin_index = self.units[*unit_index].origin_index;
            if self.origin_use_counts[origin_index] == 0 {
                self.used_origin_count += 1;
            }
            self.origin_use_counts[origin_index] += 1;
            if self.origin_first_uses[origin_index].is_none() {
                self.origin_first_uses[origin_index] = Some(OriginFirstUse {
                    output_path: output_path.clone(),
                    position,
                    source_count: candidate.unit_indices.len(),
                    truncated_bars: candidate.truncations[position],
                });
            }
        }

        self.metrics.record_output(staff_count, candidate);
        let line_cap = line_cap_for_staff_count(staff_count)?;
        let buckets = self.buckets(staff_count, line_cap);
        let bucket = bucket_for_line_count(candidate.line_count, &buckets);
        *self.histogram.entry((staff_count, bucket)).or_default() += 1;
        *self
            .metrics
            .balancing
            .bucket_histogram
            .entry(format!("{staff_count}:{bucket}"))
            .or_default() += 1;

        self.manifest_sink
            .record_manifest(ConcatenationOutputManifestRecord {
                schema_version: "2.0",
                output_path: output_path.clone(),
                output_index,
                output_id: identity.output_id,
                content_hash: identity.content_hash,
                provenance_hash: identity.provenance_hash,
                staff_count,
                source_count: candidate.unit_indices.len(),
                sources: manifest_sources(self.origins, &self.units, candidate),
            })?;
        self.emitted.insert(CandidateKey {
            unit_indices: candidate.unit_indices.clone(),
        });

        if let Some(tail) = &candidate.salvaged_tail {
            let relative_label = format!(
                "{}#salvage-{:06}-{}",
                self.origins[tail.origin_index].relative_path.display(),
                tail.parent_output_index,
                self.metrics.balancing.salvaged_tail_count
            );
            let pushed = push_unit(
                &mut self.units,
                tail.origin_index,
                salvaged_tail_kind(tail),
                relative_label,
                tail.document.clone(),
                tail.staff_count,
            );
            if let Err(reason) = pushed {
                self.origin_rejections[tail.origin_index] = Some(reason);
            } else {
                self.unit_use_counts.push(0);
                self.unit_counts_by_origin[tail.origin_index] += 1;
                self.metrics.balancing.salvaged_tail_count += 1;
            }
        }

        Ok(self.target_reached())
    }

    fn buckets(&self, _staff_count: usize, line_cap: usize) -> Vec<LineBucket> {
        buckets_for_range(self.min_line_count, line_cap, self.bucket_width)
    }

    fn progress(&self) {
        let remaining = self.origins.len().saturating_sub(self.used_origin_count);
        self.progress
            .generation_progress(self.metrics.output_count, remaining, None);
    }

    fn finish(mut self, reason: ConcatenationSourceUsageReason) -> ConcatenationMetrics {
        self.metrics.balancing.unfillable_bucket_count = self.unfillable.len();
        let mut usage = ConcatenationUsageStats {
            parseable_source_count: self.origins.len(),
            ..ConcatenationUsageStats::default()
        };
        let mut records = Vec::with_capacity(self.origins.len());

        for origin in self.origins {
            let use_count = self.origin_use_counts[origin.origin_index];
            let unit_count = self.unit_counts_by_origin[origin.origin_index];
            let (status, usage_reason) = if use_count > 0 {
                usage.used_source_count += 1;
                if use_count == 1 {
                    usage.origins_used_once += 1;
                } else {
                    usage.origins_used_twice += 1;
                }
                (
                    ConcatenationSourceUsageStatus::Used,
                    ConcatenationSourceUsageReason::Emitted,
                )
            } else if let Some(rejection) = self.origin_rejections[origin.origin_index] {
                usage.rejected_source_count += 1;
                (ConcatenationSourceUsageStatus::Rejected, rejection)
            } else {
                usage.unused_source_count += 1;
                usage.unused_line_count_total += origin.original_line_count;
                (ConcatenationSourceUsageStatus::Unused, reason)
            };

            let first_use = self.origin_first_uses[origin.origin_index].as_ref();
            records.push(ConcatenationSourceUsageRecord {
                source_path: origin.relative_path.display().to_string(),
                status,
                reason: usage_reason,
                staff_count: origin.staff_count,
                line_count: origin.original_line_count,
                unit_count: Some(unit_count),
                origin_usage_count: Some(use_count),
                output_path: first_use.map(|use_record| use_record.output_path.clone()),
                position: first_use.map(|use_record| use_record.position),
                source_count: first_use.map(|use_record| use_record.source_count),
                truncated_bars: first_use.map(|use_record| use_record.truncated_bars),
            });
        }

        self.metrics.usage = usage;
        self.metrics.source_usage = records;
        self.metrics
    }
}

struct DebugTrace {
    file: Option<File>,
    output_start: usize,
    slow_ms: u128,
}

impl DebugTrace {
    fn from_env() -> Self {
        let file = env::var_os(DEBUG_JSONL_ENV).and_then(|path| File::create(path).ok());
        let output_start = env::var(DEBUG_OUTPUT_START_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or_default();
        let slow_ms = env::var(DEBUG_SLOW_MS_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_DEBUG_SLOW_MS);
        Self {
            file,
            output_start,
            slow_ms,
        }
    }

    fn enabled(&self) -> bool {
        self.file.is_some()
    }

    fn should_record(&self, output_count: usize, elapsed_ms: u128) -> bool {
        self.enabled() && (output_count >= self.output_start || elapsed_ms >= self.slow_ms)
    }

    fn should_record_accept(
        &self,
        output_count: usize,
        elapsed_ms: u128,
        acceptance: &CandidateAcceptance,
    ) -> bool {
        self.enabled()
            && (acceptance.candidate.is_none()
                || output_count >= self.output_start
                || elapsed_ms >= self.slow_ms)
    }

    fn record(&mut self, value: serde_json::Value) {
        let Some(file) = &mut self.file else {
            return;
        };
        let _ = serde_json::to_writer(&mut *file, &value);
        let _ = file.write_all(b"\n");
        let _ = file.flush();
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn generate_outputs(
    origins: &[SourceOrigin],
    units: Vec<SourceUnitRef>,
    origin_rejections: Vec<Option<ConcatenationSourceUsageReason>>,
    split_chunk_count: usize,
    output_dir: &Path,
    target_count: Option<usize>,
    min_line_count: usize,
    bucket_width: usize,
    max_origin_uses: usize,
    max_fragments_per_output: usize,
    progress: &impl ConcatenationProgress,
    manifest_sink: &mut impl OutputManifestSink,
) -> Result<ConcatenationMetrics, ConcatenationError> {
    let mut state = GenerationState::new(
        origins,
        units,
        origin_rejections,
        split_chunk_count,
        output_dir,
        target_count,
        min_line_count,
        bucket_width,
        max_origin_uses,
        max_fragments_per_output,
        progress,
        manifest_sink,
    );
    state.progress();
    if matches!(target_count, Some(0)) {
        return Ok(state.finish(ConcatenationSourceUsageReason::TargetCountReached));
    }

    run_coverage_pass(&mut state)?;
    if state.target_reached() {
        return Ok(state.finish(ConcatenationSourceUsageReason::TargetCountReached));
    }
    run_balancing_pass(&mut state)?;
    Ok(state.finish(ConcatenationSourceUsageReason::NotReached))
}

fn run_coverage_pass(
    state: &mut GenerationState<'_, impl ConcatenationProgress, impl OutputManifestSink>,
) -> Result<(), ConcatenationError> {
    state.debug.record(serde_json::json!({
        "event": "pass_started",
        "pass": "coverage",
        "output_count": state.metrics.output_count,
        "used_origin_count": state.used_origin_count,
        "remaining_sources": state.origins.len().saturating_sub(state.used_origin_count),
        "failed_candidate_count": state.failed.len(),
        "unfillable_bucket_count": state.unfillable.len(),
    }));
    loop {
        if state.target_reached() {
            return Ok(());
        }
        let Some(unit_index) = next_primary_unit(state) else {
            state.debug.record(serde_json::json!({
                "event": "pass_finished",
                "pass": "coverage",
                "reason": "no_primary_unit",
                "output_count": state.metrics.output_count,
                "used_origin_count": state.used_origin_count,
                "remaining_sources": state.origins.len().saturating_sub(state.used_origin_count),
                "failed_candidate_count": state.failed.len(),
                "unfillable_bucket_count": state.unfillable.len(),
            }));
            return Ok(());
        };
        let staff_count = state.units[unit_index].staff_count;
        let line_cap = line_cap_for_staff_count(staff_count)?;
        let proposal = coverage_candidate(state, unit_index, line_cap)?;
        let Some(proposal) = proposal else {
            let origin_index = state.units[unit_index].origin_index;
            state.debug.record(serde_json::json!({
                "event": "coverage_source_rejected",
                "reason": "line_bucket_unfillable",
                "output_count": state.metrics.output_count,
                "unit_index": unit_index,
                "source_path": state.origins[origin_index].relative_path.display().to_string(),
                "staff_count": staff_count,
                "line_count": state.units[unit_index].line_count,
                "line_cap": line_cap,
            }));
            state.origin_rejections[state.units[unit_index].origin_index] =
                Some(ConcatenationSourceUsageReason::LineBucketUnfillable);
            continue;
        };
        let accept_started = Instant::now();
        let acceptance = state.accept(proposal.unit_indices.clone())?;
        let accept_elapsed_ms = accept_started.elapsed().as_millis();
        if state.debug.should_record_accept(
            state.metrics.output_count,
            accept_elapsed_ms,
            &acceptance,
        ) {
            state.debug.record(candidate_accept_event(
                state,
                "coverage",
                &proposal,
                &acceptance,
                accept_elapsed_ms,
            ));
        }
        match acceptance.candidate {
            Some(candidate) => {
                state.emit_candidate(staff_count, &candidate)?;
                state.progress();
            }
            None => {
                let source_path = state.origins[state.units[unit_index].origin_index]
                    .relative_path
                    .display()
                    .to_string();
                state.failed.insert(CandidateKey {
                    unit_indices: proposal.unit_indices,
                });
                state.debug.record(serde_json::json!({
                    "event": "coverage_source_rejected",
                    "reason": "candidate_rejected",
                    "output_count": state.metrics.output_count,
                    "unit_index": unit_index,
                    "source_path": source_path,
                    "staff_count": staff_count,
                    "line_count": state.units[unit_index].line_count,
                    "failed_candidate_count": state.failed.len(),
                }));
                state.origin_rejections[state.units[unit_index].origin_index] =
                    Some(ConcatenationSourceUsageReason::CandidateRejected);
            }
        }
    }
}

fn next_primary_unit(
    state: &GenerationState<'_, impl ConcatenationProgress, impl OutputManifestSink>,
) -> Option<usize> {
    state
        .units
        .iter()
        .filter(|unit| state.origin_rejections[unit.origin_index].is_none())
        .filter(|unit| state.unit_use_counts[unit.unit_index] == 0)
        .filter(|unit| {
            let origin = &state.origins[unit.origin_index];
            origin.oversized || state.origin_use_counts[unit.origin_index] == 0
        })
        .min_by_key(|unit| {
            (
                state.origins[unit.origin_index].origin_index,
                unit.unit_index,
            )
        })
        .map(|unit| unit.unit_index)
}

fn coverage_candidate(
    state: &mut GenerationState<'_, impl ConcatenationProgress, impl OutputManifestSink>,
    required_unit: usize,
    line_cap: usize,
) -> Result<Option<CandidateProposal>, ConcatenationError> {
    let staff_count = state.units[required_unit].staff_count;
    let buckets = state.buckets(staff_count, line_cap);
    let origin_counts = &state.origin_use_counts;
    let mut coverage_eligible = eligible_units(state, staff_count, false);
    if !coverage_eligible.contains(&required_unit) {
        coverage_eligible.push(required_unit);
    }
    let mut best: Option<RankedCoverageProposal> = None;

    for (bucket_need_rank, bucket) in
        buckets_by_need(staff_count, &state.histogram, &buckets, &state.unfillable)
            .into_iter()
            .enumerate()
    {
        let eligible =
            ranked_coverage_candidate_pool(state, &coverage_eligible, required_unit, bucket);

        let started = Instant::now();
        let proposals = anchored_candidate_source_sets(
            &state.units,
            &eligible,
            required_unit,
            bucket,
            line_cap,
            state.max_fragments_per_output,
            origin_counts,
            false,
        );
        let elapsed_ms = started.elapsed().as_millis();
        if state
            .debug
            .should_record(state.metrics.output_count, elapsed_ms)
        {
            state.debug.record(candidate_search_event(
                state,
                "coverage",
                staff_count,
                line_cap,
                bucket,
                eligible.len(),
                proposals.len(),
                elapsed_ms,
            ));
        }
        if let Some(proposal) = proposals
            .into_iter()
            .find(|proposal| proposal_is_available(state, proposal))
        {
            let ranked = RankedCoverageProposal {
                rank: coverage_proposal_rank(&proposal, bucket_need_rank),
                proposal,
            };
            if best
                .as_ref()
                .is_none_or(|current| ranked.rank < current.rank)
            {
                best = Some(ranked);
            }
        }
    }
    if let Some(best) = best {
        return Ok(Some(best.proposal));
    }

    if state.units[required_unit].line_count <= line_cap {
        return Ok(Some(CandidateProposal {
            unit_indices: vec![required_unit],
            rank: super::balancer::CandidateRank {
                source_fragment_count: 1,
                second_use_origin_count: 0,
                split_chunk_count: 0,
                seam_cost: 0,
                distance_to_bucket_midpoint: 0,
                stable_tiebreaker: state.units[required_unit].relative_label.clone(),
            },
        }));
    }

    Ok(None)
}

fn coverage_proposal_rank(
    proposal: &CandidateProposal,
    bucket_need_rank: usize,
) -> CoverageProposalRank {
    CoverageProposalRank {
        bucket_need_rank,
        source_fragment_count: proposal.rank.source_fragment_count,
        distance_to_bucket_midpoint: proposal.rank.distance_to_bucket_midpoint,
        seam_cost: proposal.rank.seam_cost,
        split_chunk_count: proposal.rank.split_chunk_count,
        unit_indices: proposal.unit_indices.clone(),
    }
}

#[derive(Debug, Clone)]
struct RankedCoverageProposal {
    rank: CoverageProposalRank,
    proposal: CandidateProposal,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CoverageProposalRank {
    bucket_need_rank: usize,
    source_fragment_count: usize,
    distance_to_bucket_midpoint: usize,
    seam_cost: usize,
    split_chunk_count: usize,
    unit_indices: Vec<usize>,
}

fn ranked_coverage_candidate_pool(
    state: &GenerationState<'_, impl ConcatenationProgress, impl OutputManifestSink>,
    eligible: &[usize],
    required_unit: usize,
    bucket: LineBucket,
) -> Vec<usize> {
    let mut retained = BinaryHeap::new();

    // Keep the required unit in the pool, then prefer companions whose line
    // counts are near this bucket's midpoint. The cap bounds combination search.
    for unit_index in eligible {
        let candidate = RankedCoverageCandidate {
            rank: coverage_candidate_pool_rank(state, *unit_index, required_unit, bucket),
            unit_index: *unit_index,
        };
        if retained.len() < COVERAGE_CANDIDATE_POOL_LIMIT {
            retained.push(candidate);
        } else if retained
            .peek()
            .is_some_and(|worst_retained| candidate < *worst_retained)
        {
            retained.pop();
            retained.push(candidate);
        }
    }

    let mut ranked = retained.into_vec();
    ranked.sort_unstable_by_key(|candidate| candidate.rank);
    ranked
        .into_iter()
        .map(|candidate| candidate.unit_index)
        .collect()
}

fn coverage_candidate_pool_rank(
    state: &GenerationState<'_, impl ConcatenationProgress, impl OutputManifestSink>,
    unit_index: usize,
    required_unit: usize,
    bucket: LineBucket,
) -> CoverageCandidatePoolRank {
    let bucket_midpoint = (bucket.start + bucket.end) / 2;
    CoverageCandidatePoolRank {
        required_unit_penalty: usize::from(unit_index != required_unit),
        distance_to_bucket_midpoint: state.units[unit_index].line_count.abs_diff(bucket_midpoint),
        unit_index,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CoverageCandidatePoolRank {
    required_unit_penalty: usize,
    distance_to_bucket_midpoint: usize,
    unit_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RankedCoverageCandidate {
    rank: CoverageCandidatePoolRank,
    unit_index: usize,
}

fn run_balancing_pass(
    state: &mut GenerationState<'_, impl ConcatenationProgress, impl OutputManifestSink>,
) -> Result<(), ConcatenationError> {
    state.debug.record(serde_json::json!({
        "event": "pass_started",
        "pass": "balancing",
        "output_count": state.metrics.output_count,
        "used_origin_count": state.used_origin_count,
        "remaining_sources": state.origins.len().saturating_sub(state.used_origin_count),
        "failed_candidate_count": state.failed.len(),
        "unfillable_bucket_count": state.unfillable.len(),
    }));
    loop {
        if state.target_reached() {
            return Ok(());
        }
        let Some((staff_count, proposal)) = balancing_candidate(state)? else {
            state.debug.record(serde_json::json!({
                "event": "pass_finished",
                "pass": "balancing",
                "reason": "no_candidate",
                "output_count": state.metrics.output_count,
                "used_origin_count": state.used_origin_count,
                "remaining_sources": state.origins.len().saturating_sub(state.used_origin_count),
                "failed_candidate_count": state.failed.len(),
                "unfillable_bucket_count": state.unfillable.len(),
            }));
            return Ok(());
        };
        let accept_started = Instant::now();
        let acceptance = state.accept(proposal.unit_indices.clone())?;
        let accept_elapsed_ms = accept_started.elapsed().as_millis();
        if state.debug.should_record_accept(
            state.metrics.output_count,
            accept_elapsed_ms,
            &acceptance,
        ) {
            state.debug.record(candidate_accept_event(
                state,
                "balancing",
                &proposal,
                &acceptance,
                accept_elapsed_ms,
            ));
        }
        match acceptance.candidate {
            Some(candidate) => {
                state.emit_candidate(staff_count, &candidate)?;
                state.progress();
            }
            None => {
                state.failed.insert(CandidateKey {
                    unit_indices: proposal.unit_indices,
                });
            }
        }
    }
}

fn balancing_candidate(
    state: &mut GenerationState<'_, impl ConcatenationProgress, impl OutputManifestSink>,
) -> Result<Option<(usize, CandidateProposal)>, ConcatenationError> {
    for staff_count in staff_counts(&state.units) {
        let line_cap = line_cap_for_staff_count(staff_count)?;
        let buckets = state.buckets(staff_count, line_cap);
        let eligible = eligible_units(state, staff_count, true);
        if eligible.is_empty() {
            continue;
        }
        for bucket in buckets_by_need(staff_count, &state.histogram, &buckets, &state.unfillable) {
            let started = Instant::now();
            let proposals = candidate_source_sets(
                &state.units,
                &eligible,
                bucket,
                line_cap,
                state
                    .max_fragments_per_output
                    .min(BALANCING_MAX_FRAGMENTS_PER_OUTPUT),
                &state.origin_use_counts,
                true,
            );
            let elapsed_ms = started.elapsed().as_millis();
            if state
                .debug
                .should_record(state.metrics.output_count, elapsed_ms)
            {
                state.debug.record(candidate_search_event(
                    state,
                    "balancing",
                    staff_count,
                    line_cap,
                    bucket,
                    eligible.len(),
                    proposals.len(),
                    elapsed_ms,
                ));
            }
            if let Some(proposal) = proposals
                .into_iter()
                .find(|proposal| proposal_is_available(state, proposal))
            {
                return Ok(Some((staff_count, proposal)));
            }
            state.unfillable.insert((staff_count, bucket.index));
            state.debug.record(serde_json::json!({
                "event": "bucket_unfillable",
                "pass": "balancing",
                "output_count": state.metrics.output_count,
                "staff_count": staff_count,
                "bucket_index": bucket.index,
                "bucket_start": bucket.start,
                "bucket_end": bucket.end,
                "eligible_count": eligible.len(),
                "failed_candidate_count": state.failed.len(),
                "unfillable_bucket_count": state.unfillable.len(),
            }));
        }
    }

    Ok(None)
}

fn proposal_is_available(
    state: &GenerationState<'_, impl ConcatenationProgress, impl OutputManifestSink>,
    proposal: &CandidateProposal,
) -> bool {
    let key = CandidateKey {
        unit_indices: proposal.unit_indices.clone(),
    };
    !state.failed.contains(&key) && !state.emitted.contains(&key)
}

fn eligible_units(
    state: &GenerationState<'_, impl ConcatenationProgress, impl OutputManifestSink>,
    staff_count: usize,
    allow_second_use: bool,
) -> Vec<usize> {
    state
        .units
        .iter()
        .filter(|unit| unit.staff_count == staff_count)
        .filter(|unit| state.origin_rejections[unit.origin_index].is_none())
        .filter(|unit| state.unit_use_counts[unit.unit_index] < state.max_origin_uses)
        .filter(|unit| {
            let origin = &state.origins[unit.origin_index];
            if origin.oversized {
                allow_second_use || state.unit_use_counts[unit.unit_index] == 0
            } else if allow_second_use {
                state.origin_use_counts[unit.origin_index] < state.max_origin_uses
            } else {
                state.origin_use_counts[unit.origin_index] == 0
            }
        })
        .map(|unit| unit.unit_index)
        .collect()
}

fn staff_counts(units: &[SourceUnitRef]) -> Vec<usize> {
    let mut staff_counts = units
        .iter()
        .map(|unit| unit.staff_count)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    staff_counts.sort();
    staff_counts
}

#[allow(clippy::too_many_arguments)]
fn candidate_search_event(
    state: &GenerationState<'_, impl ConcatenationProgress, impl OutputManifestSink>,
    pass: &'static str,
    staff_count: usize,
    line_cap: usize,
    bucket: LineBucket,
    eligible_count: usize,
    proposal_count: usize,
    elapsed_ms: u128,
) -> serde_json::Value {
    serde_json::json!({
        "event": "candidate_search",
        "pass": pass,
        "output_count": state.metrics.output_count,
        "used_origin_count": state.used_origin_count,
        "remaining_sources": state.origins.len().saturating_sub(state.used_origin_count),
        "staff_count": staff_count,
        "line_cap": line_cap,
        "bucket_index": bucket.index,
        "bucket_start": bucket.start,
        "bucket_end": bucket.end,
        "eligible_count": eligible_count,
        "proposal_count": proposal_count,
        "failed_candidate_count": state.failed.len(),
        "unfillable_bucket_count": state.unfillable.len(),
        "elapsed_ms": elapsed_ms,
    })
}

fn candidate_accept_event(
    state: &GenerationState<'_, impl ConcatenationProgress, impl OutputManifestSink>,
    pass: &'static str,
    proposal: &CandidateProposal,
    acceptance: &CandidateAcceptance,
    elapsed_ms: u128,
) -> serde_json::Value {
    let candidate = acceptance.candidate.as_ref();
    serde_json::json!({
        "event": "candidate_accept",
        "pass": pass,
        "output_count": state.metrics.output_count,
        "accepted": candidate.is_some(),
        "rejection_reason": acceptance.rejection_reason.map(|reason| format!("{reason:?}")),
        "rejection_detail": acceptance.rejection_detail.as_deref(),
        "candidate_line_count": candidate.map(|candidate| candidate.line_count),
        "candidate_source_count": proposal.unit_indices.len(),
        "candidate_unit_line_count_total": proposal
            .unit_indices
            .iter()
            .map(|unit_index| state.units[*unit_index].line_count)
            .sum::<usize>(),
        "candidate_sources": proposal
            .unit_indices
            .iter()
            .take(5)
            .map(|unit_index| {
                let unit = &state.units[*unit_index];
                let origin = &state.origins[unit.origin_index];
                serde_json::json!({
                    "unit_index": unit.unit_index,
                    "origin_index": unit.origin_index,
                    "source_path": origin.relative_path.display().to_string(),
                    "staff_count": unit.staff_count,
                    "line_count": unit.line_count,
                    "origin_use_count": state.origin_use_counts[unit.origin_index],
                    "unit_use_count": state.unit_use_counts[unit.unit_index],
                })
            })
            .collect::<Vec<_>>(),
        "elapsed_ms": elapsed_ms,
        "failed_candidate_count": state.failed.len(),
    })
}
