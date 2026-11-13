//! Dataset-oriented concatenation for normalized OMR `**kern` snippets.
//!
//! This module backs the `concatenate-dir` CLI command used in the dataset
//! pipeline after normalization. Its purpose is to turn a corpus that is biased
//! toward very short snippets into a set of training examples with a broader
//! range of compact OMR target lengths.
//!
//! Inputs are normalized `.krn` files. The manifest pass parses every file,
//! records its staff count, and measures line count using the
//! `SerializePreset::TranscodaV2` representation. Generation is
//! performed independently per staff count because the maximum useful target
//! length depends on staff count. Candidate snippets are selected by approximate
//! line-count buckets, concatenated with state-restoring seam lines, serialized,
//! and accepted only when the final compact target stays under the
//! staff-specific line cap.
//!
//! The current implementation prefers using each source at most once and falls
//! back to singleton outputs for sources that cannot be paired. If a candidate
//! is slightly too large, acceptance retries by removing trailing bars from the
//! final source in the candidate. That truncation is a last-mile fitting step,
//! not a semantic guarantee that all overlong sources can be salvaged.
//!
//! The output manifest and optional usage JSONL are part of the public behavior:
//! downstream dataset tooling relies on them to audit provenance, truncation,
//! unused sources, and rejected sources.

use std::{collections::BTreeMap, io, path::PathBuf};

use serde::Serialize;
use thiserror::Error;

use crate::ParseError;

mod accept;
mod balancer;
mod document;
mod filename;
mod generate;
mod manifest;
mod service;
mod split_plan;

pub use document::concatenate_documents;
pub(super) use service::OutputManifestSink;
pub use service::{
    concatenate_dir, concatenate_dir_streaming_manifest, concatenate_dir_with_progress,
    concatenate_dir_with_progress_streaming_manifest,
};

pub const DEFAULT_MIN_LINE_COUNT: usize = 25;
pub const DEFAULT_BUCKET_WIDTH: usize = 50;
pub const DEFAULT_MAX_ORIGIN_USES: usize = 2;
pub const DEFAULT_MAX_FRAGMENTS_PER_OUTPUT: usize = 3;
pub const DEFAULT_CONCATENATION_MANIFEST_FILENAME: &str = "concatenation_manifest.jsonl";

#[derive(Debug, Clone)]
pub struct ConcatenateDirOptions {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub target_count: Option<usize>,
    pub manifest_jsonl: Option<PathBuf>,
    pub usage_jsonl: Option<PathBuf>,
    pub min_line_count: usize,
    pub bucket_width: usize,
    pub max_origin_uses: usize,
    pub max_fragments_per_output: usize,
}

impl ConcatenateDirOptions {
    pub fn with_defaults(
        input_dir: PathBuf,
        output_dir: PathBuf,
        target_count: Option<usize>,
        manifest_jsonl: Option<PathBuf>,
        usage_jsonl: Option<PathBuf>,
    ) -> Self {
        Self {
            input_dir,
            output_dir,
            target_count,
            manifest_jsonl,
            usage_jsonl,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatenateStats {
    pub input_count: usize,
    pub skipped_parse_count: usize,
    pub output_count: usize,
    pub truncation: ConcatenationTruncationStats,
    pub usage: ConcatenationUsageStats,
    pub output_manifest: Vec<ConcatenationOutputManifestRecord>,
    pub source_usage: Vec<ConcatenationSourceUsageRecord>,
    pub balancing: ConcatenationBalancingStats,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConcatenationTruncationStats {
    pub truncated_output_count: usize,
    pub truncated_source_count: usize,
    pub truncated_bars_total: usize,
    pub truncated_chars_total: usize,
    pub truncated_chars_by_staff_count: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConcatenationUsageStats {
    pub parseable_source_count: usize,
    pub used_source_count: usize,
    pub unused_source_count: usize,
    pub rejected_source_count: usize,
    pub unused_line_count_total: usize,
    pub origins_used_once: usize,
    pub origins_used_twice: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConcatenationBalancingStats {
    pub bucket_histogram: BTreeMap<String, usize>,
    pub fragment_count_histogram: BTreeMap<usize, usize>,
    pub split_chunk_count: usize,
    pub salvaged_tail_count: usize,
    pub unfillable_bucket_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConcatenationSourceUsageStatus {
    Used,
    Unused,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConcatenationSourceUsageReason {
    Emitted,
    TargetCountReached,
    SingletonRejected,
    CandidateRejected,
    NotReached,
    OversizedSplitFailed,
    SplitChunkInvalid,
    TargetSerializationFailed,
    LineBucketUnfillable,
    MaxOriginUsesReached,
    CandidateOverLineCap,
    CandidateBelowMinimum,
    InvalidSynthesizedHeader,
    UnsupportedStaffCount,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConcatenationSourceUsageRecord {
    pub source_path: String,
    pub status: ConcatenationSourceUsageStatus,
    pub reason: ConcatenationSourceUsageReason,
    pub staff_count: usize,
    pub line_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin_usage_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncated_bars: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConcatenationOutputManifestRecord {
    pub schema_version: &'static str,
    pub output_path: String,
    pub output_index: usize,
    pub output_id: String,
    pub content_hash: String,
    pub provenance_hash: String,
    pub staff_count: usize,
    pub source_count: usize,
    pub sources: Vec<ConcatenationOutputManifestSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConcatenationOutputManifestSource {
    pub position: usize,
    pub source_path: String,
    pub unit_kind: ConcatenationManifestUnitKind,
    pub origin_usage_ordinal: usize,
    pub truncated_bars: usize,
    pub salvaged_bars: usize,
    pub staff_count: usize,
    pub line_count: usize,
    pub unit_line_count: usize,
    pub contributed_line_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_bar_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_bar_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConcatenationManifestUnitKind {
    Original,
    SplitChunk,
    SalvagedTail,
}

pub trait ConcatenationProgress {
    fn manifest_started(&self, _total: usize) {}
    fn manifest_advanced(&self) {}
    fn manifest_finished(&self, _parseable_count: usize, _skipped_parse_count: usize) {}
    fn generation_started(&self, _parseable_count: usize, _target_count: Option<usize>) {}
    fn generation_progress(
        &self,
        _output_count: usize,
        _remaining_sources: usize,
        _staff_count: Option<usize>,
    ) {
    }
    fn generation_finished(&self, _output_count: usize) {}
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoConcatenationProgress;

impl ConcatenationProgress for NoConcatenationProgress {}

#[derive(Debug, Error)]
pub enum ConcatenationError {
    #[error("input directory does not exist: {0}")]
    MissingInputDir(PathBuf),
    #[error("no .krn files found in {0}")]
    EmptyInputDir(PathBuf),
    #[error("no parseable .krn files found in {0}")]
    NoParseableInput(PathBuf),
    #[error("I/O error for {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("failed to parse {path}: {source}")]
    Parse { path: PathBuf, source: ParseError },
    #[error("cannot concatenate fewer than two snippets")]
    TooFewSnippets,
    #[error("all snippets in a concatenation must have the same staff count")]
    StaffCountMismatch,
    #[error("concatenated snippet has no musical data")]
    NoMusicalData,
    #[error("failed to truncate final snippet: {0}")]
    Truncate(#[from] crate::transforms::TruncateBarsError),
    #[error("snippet does not resolve: {0}")]
    State(#[from] crate::score::StateErrors),
    #[error("cannot restate state at a snippet seam: {0}")]
    Seam(#[from] crate::score::BridgeError),
    #[error("unsupported staff count for concatenation line cap: {0}")]
    UnsupportedStaffCount(usize),
    #[error("{name} must be greater than zero")]
    InvalidPositiveOption { name: &'static str },
}

pub(super) fn line_cap_for_staff_count(staff_count: usize) -> Result<usize, ConcatenationError> {
    match staff_count {
        1 => Ok(502),
        2 => Ok(273),
        3 => Ok(156),
        4 => Ok(94),
        5 => Ok(91),
        6 => Ok(74),
        7 => Ok(56),
        _ => Err(ConcatenationError::UnsupportedStaffCount(staff_count)),
    }
}

pub(super) fn compact_line_count(contents: &str) -> usize {
    contents.lines().count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KernDocument, SerializePreset};
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    fn parse(input: &str) -> KernDocument {
        KernDocument::parse(input).unwrap()
    }

    fn concat(parts: &[&str]) -> String {
        let documents = parts.iter().map(|part| parse(part)).collect::<Vec<_>>();
        concatenate_documents(&documents)
            .unwrap()
            .to_compact_omr_string()
    }

    fn output_target_line_count(path: &Path) -> usize {
        let contents = fs::read_to_string(path).unwrap();
        let document = KernDocument::parse(&contents).unwrap();
        compact_line_count(
            &SerializePreset::TranscodaV2
                .try_serialize_document(&document)
                .unwrap(),
        )
    }

    fn krn_paths(dir: &Path) -> Vec<PathBuf> {
        let mut paths = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "krn"))
            .collect::<Vec<_>>();
        paths.sort();
        paths
    }

    fn one_staff_sample(note: &str) -> String {
        format!("*clefG2\n*k[]\n*M4/4\n{note}\n*-\n")
    }

    fn repeated_one_staff_sample(note_count: usize) -> String {
        let mut source = String::from("*clefG2\n*k[]\n*M4/4\n");
        for _ in 0..note_count {
            source.push_str("4c\n");
        }
        source.push_str("*-\n");
        source
    }

    #[test]
    fn line_cap_lookup_uses_recommended_values() {
        assert_eq!(line_cap_for_staff_count(1).unwrap(), 502);
        assert_eq!(line_cap_for_staff_count(2).unwrap(), 273);
        assert_eq!(line_cap_for_staff_count(3).unwrap(), 156);
        assert_eq!(line_cap_for_staff_count(4).unwrap(), 94);
        assert_eq!(line_cap_for_staff_count(5).unwrap(), 91);
        assert_eq!(line_cap_for_staff_count(6).unwrap(), 74);
        assert_eq!(line_cap_for_staff_count(7).unwrap(), 56);
    }

    #[test]
    fn line_cap_lookup_rejects_unsupported_staff_counts() {
        assert!(matches!(
            line_cap_for_staff_count(8),
            Err(ConcatenationError::UnsupportedStaffCount(8))
        ));
    }

    #[test]
    fn directory_generation_skips_unsupported_staff_count() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        write(
            &input.join("eight_staff.krn"),
            "**kern\t**kern\t**kern\t**kern\t**kern\t**kern\t**kern\t**kern\n\
             4c\t4d\t4e\t4f\t4g\t4a\t4b\t4cc\n\
             *-\t*-\t*-\t*-\t*-\t*-\t*-\t*-\n",
        );

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output,
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.input_count, 1);
        assert_eq!(stats.output_count, 0);
        assert_eq!(stats.usage.parseable_source_count, 1);
        assert_eq!(stats.usage.rejected_source_count, 1);
        assert_eq!(
            stats.source_usage[0].reason,
            ConcatenationSourceUsageReason::UnsupportedStaffCount
        );
        assert_eq!(stats.source_usage[0].staff_count, 8);
    }

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    fn read_jsonl(path: &Path) -> Vec<serde_json::Value> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect()
    }

    fn concatenate_options(input_dir: PathBuf, output_dir: PathBuf) -> ConcatenateDirOptions {
        ConcatenateDirOptions {
            input_dir,
            output_dir,
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        }
    }

    #[test]
    fn concatenate_dir_updates_dataset_manifest_for_stage_six_output() {
        let temp = TempDir::new().unwrap();
        let dataset_root = temp.path().join("train/pdmx");
        let input = dataset_root.join("5_quality_filtered");
        let output = dataset_root.join("6_concatenated");
        write(&input.join("a.krn"), &one_staff_sample("4c"));
        write(&input.join("b.krn"), &one_staff_sample("4d"));

        let stats = concatenate_dir(&concatenate_options(input, output)).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(
                dataset_root.join(crate::dataset_manifest::DATASET_MANIFEST_FILE_NAME),
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(stats.output_count, 2);
        assert_eq!(
            manifest["components"]["concatenation_provenance"]["summary"]["output_count"],
            2
        );
        assert!(manifest["components"]["concatenation_provenance"]["content_id"].is_string());
    }

    #[test]
    fn streaming_manifest_writes_jsonl_without_returning_records() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        let manifest_path = temp.path().join("manifest").join("outputs.jsonl");
        write(&input.join("a.krn"), &one_staff_sample("4c"));
        write(&input.join("b.krn"), &one_staff_sample("4d"));

        let mut options = concatenate_options(input, output);
        options.manifest_jsonl = Some(manifest_path.clone());
        let stats = concatenate_dir_streaming_manifest(&options).unwrap();

        assert_eq!(stats.output_count, 2);
        assert!(stats.output_manifest.is_empty());
        let rows = read_jsonl(&manifest_path);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["schema_version"], "2.0");
        assert_eq!(rows[0]["output_index"], 1);
        assert_eq!(rows[1]["output_index"], 2);
    }

    #[test]
    fn streaming_manifest_keeps_rows_written_before_interruption() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct PanicAfterFirstOutput {
            panics: AtomicUsize,
        }

        impl ConcatenationProgress for PanicAfterFirstOutput {
            fn generation_progress(
                &self,
                output_count: usize,
                _remaining_sources: usize,
                _staff_count: Option<usize>,
            ) {
                if output_count > 0 && self.panics.fetch_add(1, Ordering::SeqCst) == 0 {
                    panic!("stop after first manifested output");
                }
            }
        }

        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        let manifest_path = temp.path().join("manifest").join("outputs.jsonl");
        write(&input.join("a.krn"), &one_staff_sample("4c"));
        write(&input.join("b.krn"), &one_staff_sample("4d"));

        let mut options = concatenate_options(input, output.clone());
        options.manifest_jsonl = Some(manifest_path.clone());
        let progress = PanicAfterFirstOutput {
            panics: AtomicUsize::new(0),
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            concatenate_dir_with_progress_streaming_manifest(&options, &progress).unwrap();
        }));

        assert!(result.is_err());
        assert_eq!(krn_paths(&output).len(), 1);
        let rows = read_jsonl(&manifest_path);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["output_index"], 1);
    }

    #[test]
    fn omits_redundant_seam_state() {
        let output = concat(&[
            "*clefG2\n*k[]\n*M4/4\n4c\n*-\n",
            "*clefG2\n*k[]\n*M4/4\n4d\n*-\n",
        ]);

        assert_eq!(output, "*clefG2\n*k[]\n*M4/4\n4c\n4%3ryy\n4d\n");
    }

    #[test]
    fn emits_only_changed_seam_state() {
        let output = concat(&[
            "*clefG2\n*k[]\n*M4/4\n4c\n*-\n",
            "*clefF4\n*k[]\n*M3/4\n4d\n*-\n",
        ]);

        assert_eq!(
            output,
            "*clefG2\n*k[]\n*M4/4\n4c\n4%3ryy\n*clefF4\n*M3/4\n4d\n"
        );
    }

    #[test]
    fn evaluates_state_per_spine() {
        let output = concat(&[
            "**kern\t**kern\n*clefG2\t*clefF4\n*k[]\t*k[]\n*M4/4\t*M4/4\n4c\t4C\n*-\t*-\n",
            "**kern\t**kern\n*clefG2\t*clefG2\n*k[]\t*k[f#]\n*M4/4\t*M3/4\n4d\t4D\n*-\t*-\n",
        ]);

        assert!(output.contains("*\t*clefG2\n"));
        assert!(output.contains("*\t*k[f#]\n"));
        assert!(output.contains("*\t*M3/4\n"));
    }

    #[test]
    fn preserves_terminal_non_repeat_double_barline() {
        let output = concat(&[
            "*clefG2\n*k[]\n*M4/4\n4c\n==\n*-\n",
            "*clefG2\n*k[]\n*M4/4\n4d\n*-\n",
        ]);

        assert_eq!(output, "*clefG2\n*k[]\n*M4/4\n4c\n4%3ryy\n==\n4d\n");
    }

    #[test]
    fn preserves_repeat_barline_at_seam() {
        let output = concat(&[
            "*clefG2\n*k[]\n*M4/4\n4c\n=:|!\n*-\n",
            "*clefG2\n*k[]\n*M4/4\n4d\n*-\n",
        ]);

        assert_eq!(output, "*clefG2\n*k[]\n*M4/4\n4c\n4%3ryy\n=:|!\n4d\n");
    }

    #[test]
    fn strips_optional_headers_metadata_comments_and_final_terminal_lines() {
        let output = concat(&[
            "!!!OTL: first\n!! comment\n**kern\n*clefG2\n*k[]\n*M4/4\n4c\n*-\n",
            "!!!OTL: second\n!! comment\n**kern\n*clefG2\n*k[]\n*M4/4\n4d\n*-\n",
        ]);

        assert_eq!(output, "*clefG2\n*k[]\n*M4/4\n4c\n4%3ryy\n4d\n");
        assert!(!output.contains("**kern"));
        assert!(!output.contains("*-"));
        assert!(!output.contains("!!!"));
    }

    #[test]
    fn directory_generation_never_pairs_different_staff_counts_and_clears_output() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        write(&input.join("a.krn"), &one_staff_sample("4c"));
        write(&input.join("b.krn"), &one_staff_sample("4d"));
        write(
            &input.join("two_staff.krn"),
            "**kern\t**kern\n*clefG2\t*clefF4\n*k[]\t*k[]\n*M4/4\t*M4/4\n4c\t4C\n*-\t*-\n",
        );
        write(&output.join("stale.krn"), "stale");

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.input_count, 3);
        assert_eq!(stats.output_count, 3);
        assert_eq!(stats.truncation, ConcatenationTruncationStats::default());
        assert_eq!(
            stats.usage,
            ConcatenationUsageStats {
                parseable_source_count: 3,
                used_source_count: 3,
                unused_source_count: 0,
                rejected_source_count: 0,
                unused_line_count_total: 0,
                origins_used_once: 3,
                origins_used_twice: 0,
            }
        );
        assert!(!output.join("stale.krn").exists());
        let files = krn_paths(&output);
        assert_eq!(files.len(), 3);
        let contents = files
            .iter()
            .map(|file| fs::read_to_string(file).unwrap())
            .collect::<Vec<_>>();
        assert!(contents.iter().any(|contents| contents.contains("4c")));
        assert!(contents.iter().any(|contents| contents.contains("4d")));
        assert!(contents.iter().any(|contents| contents.contains("4c\t4C")));
    }

    #[test]
    fn directory_generation_emits_leftover_singleton_after_pairs() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        write(&input.join("a.krn"), &one_staff_sample("4c"));
        write(&input.join("b.krn"), &one_staff_sample("4d"));
        write(&input.join("c.krn"), &one_staff_sample("4e"));

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.output_count, 3);
        assert_eq!(stats.truncation, ConcatenationTruncationStats::default());
        assert_eq!(stats.usage.parseable_source_count, 3);
        assert_eq!(stats.usage.used_source_count, 3);
        assert_eq!(stats.usage.unused_source_count, 0);
        assert_eq!(stats.usage.rejected_source_count, 0);
        let contents = krn_paths(&output)
            .iter()
            .map(|file| fs::read_to_string(file).unwrap())
            .collect::<Vec<_>>();
        assert!(contents.iter().any(|contents| contents
            == "**kern\n*clefG2\n*k[]\n*M4/4\n4e\n*-\n!!!RDF**kern: > = above\n!!!RDF**kern: < = below\n!!!RDF**kern: N = linked\n!!!RDF**kern: Z = editorial accidental, parentheses\n!!!RDF**kern: j = editorial accidental, brackets\n"));
    }

    #[test]
    fn directory_generation_does_not_emit_duplicate_second_use_candidate() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        write(&input.join("a.krn"), &one_staff_sample("4c"));
        write(&input.join("b.krn"), &one_staff_sample("4d"));

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: 8,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.output_count, 1);
        assert_eq!(krn_paths(&output).len(), 1);
        assert_eq!(stats.usage.origins_used_once, 2);
        assert_eq!(stats.usage.origins_used_twice, 0);
        assert_eq!(stats.output_manifest[0].source_count, 2);
    }

    #[test]
    fn balancing_pass_does_not_emit_second_use_triples() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        for index in 0..6 {
            write(
                &input.join(format!("{index}.krn")),
                &repeated_one_staff_sample(77),
            );
        }

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: 225,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.output_count, 2);
        assert_eq!(stats.balancing.fragment_count_histogram.get(&3), Some(&2));
        assert_eq!(stats.usage.origins_used_once, 6);
        assert_eq!(stats.usage.origins_used_twice, 0);
        assert_eq!(krn_paths(&output).len(), 2);
    }

    #[test]
    fn directory_generation_skips_parse_failures() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        write(&input.join("a.krn"), "*clefG2\n*k[]\n4c\n*-\n");
        write(&input.join("b.krn"), "*clefG2\n*k[]\n4d\n*-\n");
        write(&input.join("bad.krn"), "*unknown\n");

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.input_count, 3);
        assert_eq!(stats.skipped_parse_count, 1);
        assert_eq!(stats.output_count, 2);
        assert_eq!(stats.usage.parseable_source_count, 2);
        assert_eq!(stats.usage.used_source_count, 2);
        assert_eq!(
            krn_paths(&output).len(),
            2,
            "both valid singletons should be emitted"
        );
    }

    #[test]
    fn directory_generation_honors_target_count() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        write(&input.join("a.krn"), "*clefG2\n*k[]\n4c\n*-\n");
        write(&input.join("b.krn"), "*clefG2\n*k[]\n4d\n*-\n");
        write(&input.join("c.krn"), "*clefG2\n*k[]\n4e\n*-\n");
        write(&input.join("d.krn"), "*clefG2\n*k[]\n4f\n*-\n");

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: Some(1),
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.output_count, 1);
        assert_eq!(stats.usage.parseable_source_count, 4);
        assert_eq!(stats.usage.used_source_count, 1);
        assert_eq!(stats.usage.unused_source_count, 3);
        assert_eq!(stats.usage.rejected_source_count, 0);
        assert_eq!(stats.usage.unused_line_count_total, 9);
        assert_eq!(
            stats
                .source_usage
                .iter()
                .filter(|record| record.status == ConcatenationSourceUsageStatus::Unused)
                .map(|record| record.reason)
                .collect::<Vec<_>>(),
            vec![
                ConcatenationSourceUsageReason::TargetCountReached,
                ConcatenationSourceUsageReason::TargetCountReached,
                ConcatenationSourceUsageReason::TargetCountReached
            ]
        );
        assert_eq!(krn_paths(&output).len(), 1);
    }

    #[test]
    fn directory_generation_honors_target_count_before_singleton_fallback() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        write(&input.join("a.krn"), "*clefG2\n*k[]\n4c\n*-\n");
        write(&input.join("b.krn"), "*clefG2\n*k[]\n4d\n*-\n");
        write(&input.join("c.krn"), "*clefG2\n*k[]\n4e\n*-\n");

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: Some(1),
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.output_count, 1);
        assert_eq!(stats.usage.parseable_source_count, 3);
        assert_eq!(stats.usage.used_source_count, 1);
        assert_eq!(stats.usage.unused_source_count, 2);
        assert_eq!(stats.usage.rejected_source_count, 0);
        assert_eq!(krn_paths(&output).len(), 1);
    }

    #[test]
    fn directory_generation_target_zero_marks_all_parseable_sources_unused() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        write(&input.join("a.krn"), &one_staff_sample("4c"));
        write(&input.join("b.krn"), &one_staff_sample("4d"));

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: Some(0),
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.output_count, 0);
        assert_eq!(stats.usage.parseable_source_count, 2);
        assert_eq!(stats.usage.used_source_count, 0);
        assert_eq!(stats.usage.unused_source_count, 2);
        assert_eq!(stats.usage.rejected_source_count, 0);
        assert_eq!(stats.usage.unused_line_count_total, 8);
        assert!(stats
            .source_usage
            .iter()
            .all(|record| record.reason == ConcatenationSourceUsageReason::TargetCountReached));
        assert_eq!(krn_paths(&output).len(), 0);
    }

    #[test]
    fn directory_generation_splits_oversized_singleton() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        let mut source = String::from("*clefG2\n*k[]\n");
        for _ in 0..10 {
            for _ in 0..200 {
                source.push_str("4c\n");
            }
            source.push_str("=\n");
        }
        source.push_str("*-\n");
        write(&input.join("long.krn"), &source);

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.output_count, 5);
        assert_eq!(stats.truncation, ConcatenationTruncationStats::default());
        assert_eq!(stats.balancing.split_chunk_count, 5);
        for file in krn_paths(&output) {
            assert!(output_target_line_count(&file) <= line_cap_for_staff_count(1).unwrap());
        }
    }

    #[test]
    fn directory_generation_line_gate_uses_training_target_projection() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        let source = "*clefG2\n*k[]\n(4c#X'[ 4e])\n*-\n";
        write(&input.join("visual_tokens.krn"), source);

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        let document = KernDocument::parse(source).unwrap();
        let target_line_count = compact_line_count(
            &SerializePreset::TranscodaV2
                .try_serialize_document(&document)
                .unwrap(),
        );
        assert_eq!(stats.output_count, 1);
        assert_eq!(stats.source_usage[0].line_count, target_line_count);
        let file = krn_paths(&output).pop().unwrap();
        let contents = fs::read_to_string(file).unwrap();
        assert!(contents.contains("#X"));
        assert!(contents.contains("'"));
    }

    #[test]
    fn directory_generation_rejects_target_serialization_failures() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        let source = "**kern\t**kern\n2c\t4d\n4e\t.\n*-\t*-\n";
        write(&input.join("overlap.krn"), source);

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.output_count, 0);
        assert_eq!(stats.usage.rejected_source_count, 1);
        assert_eq!(
            stats.source_usage[0].reason,
            ConcatenationSourceUsageReason::TargetSerializationFailed
        );
        assert!(krn_paths(&output).is_empty());
    }

    #[test]
    fn failed_singleton_truncation_attempts_do_not_count_as_truncation_metrics() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        let mut source = String::from("*clefG2\n*k[]\n");
        for _ in 0..2_000 {
            source.push_str("4c\n");
        }
        source.push_str("=\n*-\n");
        write(&input.join("long.krn"), &source);

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.output_count, 0);
        assert_eq!(stats.truncation, ConcatenationTruncationStats::default());
        assert_eq!(stats.usage.parseable_source_count, 1);
        assert_eq!(stats.usage.used_source_count, 0);
        assert_eq!(stats.usage.unused_source_count, 0);
        assert_eq!(stats.usage.rejected_source_count, 1);
        assert_eq!(
            stats.source_usage[0].reason,
            ConcatenationSourceUsageReason::OversizedSplitFailed
        );
        assert_eq!(krn_paths(&output).len(), 0);
    }

    #[test]
    fn directory_generation_preserves_outputs_after_acceptance_checks() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        write(&input.join("a.krn"), &one_staff_sample("4c"));
        write(
            &input.join("b.krn"),
            "*clefG2\n*k[]\n*M4/4\n4d\n=\n4e\n=\n4f\n=\n4g\n*-\n",
        );

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output.clone(),
            target_count: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.output_count, 2);
        assert_eq!(stats.truncation, ConcatenationTruncationStats::default());
        for file in krn_paths(&output) {
            let contents = fs::read_to_string(file).unwrap();
            KernDocument::parse(&contents).unwrap();
        }
    }

    #[test]
    fn manifest_records_include_all_sources_without_semantic_filenames() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("4_normalized");
        let output = temp.path().join("5_concatenated");
        for index in 0..8 {
            write(
                &input.join(format!(
                    "source_{index:02}_abcdefghijklmnopqrstuvwxyz0123456789long.krn"
                )),
                &one_staff_sample("4c"),
            );
        }

        let stats = concatenate_dir(&ConcatenateDirOptions {
            input_dir: input,
            output_dir: output,
            target_count: Some(1),
            manifest_jsonl: None,
            usage_jsonl: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
        })
        .unwrap();

        assert_eq!(stats.output_count, 1);
        assert_eq!(stats.output_manifest.len(), 1);
        let manifest = &stats.output_manifest[0];
        assert_eq!(manifest.schema_version, "2.0");
        assert_eq!(manifest.output_index, 1);
        assert_eq!(manifest.source_count, manifest.sources.len());
        assert_eq!(manifest.source_count, 1);
        assert!(manifest.output_path.starts_with("concat_000001_"));
        assert!(manifest.output_path.ends_with(".krn"));
        assert!(!manifest.output_path.contains("source_"));
        assert_eq!(manifest.output_id.len(), 12);
        assert_eq!(manifest.provenance_hash.len(), 16);
        assert_eq!(manifest.content_hash.len(), 16);
        assert!(manifest.sources[0].source_path.contains("source_"));
        let used = stats
            .source_usage
            .iter()
            .filter(|record| record.status == ConcatenationSourceUsageStatus::Used)
            .collect::<Vec<_>>();
        assert_eq!(used.len(), 1);
        assert!(used.iter().all(|record| record.output_path.is_some()));
        assert!(used.iter().all(|record| record
            .output_path
            .as_ref()
            .unwrap()
            .starts_with("concat_000001_")));
    }
}
