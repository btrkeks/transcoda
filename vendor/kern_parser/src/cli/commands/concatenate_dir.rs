use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use crate::concatenation::{
    concatenate_dir_with_progress_streaming_manifest, ConcatenateDirOptions,
    ConcatenationBalancingStats, ConcatenationUsageStats, DEFAULT_BUCKET_WIDTH,
    DEFAULT_MAX_FRAGMENTS_PER_OUTPUT, DEFAULT_MAX_ORIGIN_USES, DEFAULT_MIN_LINE_COUNT,
};
use clap::Parser;
use serde::Serialize;

use crate::cli::progress::ConcatenateProgressDisplay;

#[derive(Debug, Parser, Clone)]
pub(in crate::cli) struct ConcatenateDirArgs {
    /// Directory containing normalized .krn snippets.
    input_dir: PathBuf,
    /// Directory where concatenated .krn files should be written.
    output_dir: PathBuf,
    /// Stop after writing this many concatenated files.
    #[arg(long)]
    target_count: Option<usize>,
    /// Minimum compact target line count for balanced outputs.
    #[arg(long, default_value_t = DEFAULT_MIN_LINE_COUNT)]
    min_line_count: usize,
    /// Compact target line-count bucket width.
    #[arg(long, default_value_t = DEFAULT_BUCKET_WIDTH)]
    bucket_width: usize,
    /// Maximum emitted uses per normal source origin.
    #[arg(long, default_value_t = DEFAULT_MAX_ORIGIN_USES)]
    max_origin_uses: usize,
    /// Maximum source units per generated output.
    #[arg(long, default_value_t = DEFAULT_MAX_FRAGMENTS_PER_OUTPUT)]
    max_fragments_per_output: usize,
    /// Optional path to write concatenation statistics as JSON.
    #[arg(long)]
    stats_json: Option<PathBuf>,
    /// Optional path to write per-output provenance records as JSONL.
    #[arg(long)]
    manifest_jsonl: Option<PathBuf>,
    /// Optional path to write per-source usage records as JSONL.
    #[arg(long)]
    usage_jsonl: Option<PathBuf>,
    /// Suppress progress output.
    #[arg(long)]
    quiet: bool,
}

#[derive(Debug, Serialize)]
struct ConcatenateStatsPayload {
    schema_version: &'static str,
    input_dir: String,
    output_dir: String,
    input_count: usize,
    skipped_parse_count: usize,
    output_count: usize,
    duration_seconds: f64,
    truncation: ConcatenateTruncationStatsPayload,
    usage: ConcatenateUsageStatsPayload,
    balancing: ConcatenateBalancingStatsPayload,
}

#[derive(Debug, Serialize)]
struct ConcatenateTruncationStatsPayload {
    truncated_output_count: usize,
    truncated_source_count: usize,
    truncated_bars_total: usize,
    truncated_chars_total: usize,
    truncated_chars_by_staff_count: BTreeMap<usize, usize>,
}

#[derive(Debug, Serialize)]
struct ConcatenateUsageStatsPayload {
    parseable_source_count: usize,
    used_source_count: usize,
    unused_source_count: usize,
    rejected_source_count: usize,
    unused_line_count_total: usize,
    origins_used_once: usize,
    origins_used_twice: usize,
}

#[derive(Debug, Serialize)]
struct ConcatenateBalancingStatsPayload {
    bucket_histogram: BTreeMap<String, usize>,
    fragment_count_histogram: BTreeMap<usize, usize>,
    split_chunk_count: usize,
    salvaged_tail_count: usize,
    unfillable_bucket_count: usize,
}

pub(in crate::cli) fn concatenate_dir_command(
    args: ConcatenateDirArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let options = ConcatenateDirOptions {
        input_dir: args.input_dir.clone(),
        output_dir: args.output_dir.clone(),
        target_count: args.target_count,
        manifest_jsonl: args.manifest_jsonl.clone(),
        usage_jsonl: args.usage_jsonl.clone(),
        min_line_count: args.min_line_count,
        bucket_width: args.bucket_width,
        max_origin_uses: args.max_origin_uses,
        max_fragments_per_output: args.max_fragments_per_output,
    };
    let progress = ConcatenateProgressDisplay::new(args.quiet);
    let stats = concatenate_dir_with_progress_streaming_manifest(&options, &progress)?;

    eprintln!(
        "Concatenated {} output files from {} input files into {}",
        stats.output_count,
        stats.input_count,
        args.output_dir.display()
    );
    eprintln!(
        "Truncation: {} outputs, {} sources, {} bars removed, {} chars removed",
        stats.truncation.truncated_output_count,
        stats.truncation.truncated_source_count,
        stats.truncation.truncated_bars_total,
        stats.truncation.truncated_chars_total,
    );
    eprintln!(
        "Usage: {} used, {} unused, {} rejected from {} parseable sources",
        stats.usage.used_source_count,
        stats.usage.unused_source_count,
        stats.usage.rejected_source_count,
        stats.usage.parseable_source_count,
    );
    if stats.skipped_parse_count > 0 {
        eprintln!(
            "Skipped {} files with parse errors",
            stats.skipped_parse_count
        );
    }

    if let Some(stats_path) = args.stats_json {
        if let Some(parent) = stats_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let payload = ConcatenateStatsPayload {
            schema_version: "1.2",
            input_dir: args.input_dir.display().to_string(),
            output_dir: args.output_dir.display().to_string(),
            input_count: stats.input_count,
            skipped_parse_count: stats.skipped_parse_count,
            output_count: stats.output_count,
            duration_seconds: start.elapsed().as_secs_f64(),
            truncation: ConcatenateTruncationStatsPayload {
                truncated_output_count: stats.truncation.truncated_output_count,
                truncated_source_count: stats.truncation.truncated_source_count,
                truncated_bars_total: stats.truncation.truncated_bars_total,
                truncated_chars_total: stats.truncation.truncated_chars_total,
                truncated_chars_by_staff_count: stats
                    .truncation
                    .truncated_chars_by_staff_count
                    .clone(),
            },
            usage: ConcatenateUsageStatsPayload::from_stats(&stats.usage),
            balancing: ConcatenateBalancingStatsPayload::from_stats(&stats.balancing),
        };
        fs::write(stats_path, serde_json::to_string_pretty(&payload)?)?;
    }

    Ok(())
}

impl ConcatenateUsageStatsPayload {
    fn from_stats(stats: &ConcatenationUsageStats) -> Self {
        Self {
            parseable_source_count: stats.parseable_source_count,
            used_source_count: stats.used_source_count,
            unused_source_count: stats.unused_source_count,
            rejected_source_count: stats.rejected_source_count,
            unused_line_count_total: stats.unused_line_count_total,
            origins_used_once: stats.origins_used_once,
            origins_used_twice: stats.origins_used_twice,
        }
    }
}

impl ConcatenateBalancingStatsPayload {
    fn from_stats(stats: &ConcatenationBalancingStats) -> Self {
        Self {
            bucket_histogram: stats.bucket_histogram.clone(),
            fragment_count_histogram: stats.fragment_count_histogram.clone(),
            split_chunk_count: stats.split_chunk_count,
            salvaged_tail_count: stats.salvaged_tail_count,
            unfillable_bucket_count: stats.unfillable_bucket_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use serde_json::Value;
    use tempfile::TempDir;

    use super::*;

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    fn krn_count(path: &Path) -> usize {
        fs::read_dir(path)
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "krn")
            })
            .count()
    }

    fn read_jsonl(path: &Path) -> Vec<Value> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect()
    }

    fn concatenate_dir_args(input_dir: PathBuf, output_dir: PathBuf) -> ConcatenateDirArgs {
        ConcatenateDirArgs {
            input_dir,
            output_dir,
            target_count: None,
            min_line_count: DEFAULT_MIN_LINE_COUNT,
            bucket_width: DEFAULT_BUCKET_WIDTH,
            max_origin_uses: DEFAULT_MAX_ORIGIN_USES,
            max_fragments_per_output: DEFAULT_MAX_FRAGMENTS_PER_OUTPUT,
            stats_json: None,
            manifest_jsonl: None,
            usage_jsonl: None,
            quiet: true,
        }
    }

    #[test]
    fn concatenate_dir_writes_stats_json() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        let stats_path = temp.path().join("stats.json");
        write(&input_path.join("a.krn"), "*clefG2\n*k[]\n4c\n*-\n");
        write(&input_path.join("b.krn"), "*clefG2\n*k[]\n4d\n*-\n");

        let mut args = concatenate_dir_args(input_path, output_dir.clone());
        args.stats_json = Some(stats_path.clone());
        concatenate_dir_command(args).unwrap();

        let stats: Value = serde_json::from_str(&fs::read_to_string(stats_path).unwrap()).unwrap();
        assert_eq!(stats["schema_version"], "1.2");
        assert_eq!(stats["input_count"], 2);
        assert_eq!(stats["skipped_parse_count"], 0);
        assert_eq!(stats["output_count"], 2);
        assert!(stats["duration_seconds"].as_f64().unwrap() >= 0.0);
        assert_eq!(stats["truncation"]["truncated_output_count"], 0);
        assert_eq!(stats["truncation"]["truncated_source_count"], 0);
        assert_eq!(stats["truncation"]["truncated_bars_total"], 0);
        assert_eq!(stats["truncation"]["truncated_chars_total"], 0);
        assert!(stats["truncation"]["truncated_chars_by_staff_count"]
            .as_object()
            .unwrap()
            .is_empty());
        assert_eq!(stats["usage"]["parseable_source_count"], 2);
        assert_eq!(stats["usage"]["used_source_count"], 2);
        assert_eq!(stats["usage"]["unused_source_count"], 0);
        assert_eq!(stats["usage"]["rejected_source_count"], 0);
        assert_eq!(stats["usage"]["unused_line_count_total"], 0);
        assert!(stats["balancing"]["bucket_histogram"].is_object());
        assert_eq!(krn_count(&output_dir), 2);
        assert!(output_dir.join("concatenation_manifest.jsonl").exists());
    }

    #[test]
    fn concatenate_dir_writes_default_manifest_jsonl() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(&input_path.join("a.krn"), "*clefG2\n*k[]\n4c\n*-\n");
        write(&input_path.join("b.krn"), "*clefG2\n*k[]\n4d\n*-\n");

        concatenate_dir_command(concatenate_dir_args(input_path, output_dir.clone())).unwrap();

        let rows = read_jsonl(&output_dir.join("concatenation_manifest.jsonl"));
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["schema_version"], "2.0");
        assert_eq!(rows[0]["output_index"], 1);
        assert!(rows[0]["output_path"]
            .as_str()
            .unwrap()
            .starts_with("concat_000001_"));
        assert_eq!(rows[0]["sources"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn concatenate_dir_writes_custom_manifest_jsonl() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        let manifest_path = temp.path().join("manifest").join("outputs.jsonl");
        write(&input_path.join("a.krn"), "*clefG2\n*k[]\n4c\n*-\n");
        write(&input_path.join("b.krn"), "*clefG2\n*k[]\n4d\n*-\n");

        let mut args = concatenate_dir_args(input_path, output_dir.clone());
        args.manifest_jsonl = Some(manifest_path.clone());
        concatenate_dir_command(args).unwrap();

        let rows = read_jsonl(&manifest_path);
        assert_eq!(rows.len(), 2);
        assert!(!output_dir.join("concatenation_manifest.jsonl").exists());
    }

    #[test]
    fn concatenate_dir_writes_usage_jsonl() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        let usage_path = temp.path().join("usage").join("sources.jsonl");
        write(&input_path.join("a.krn"), "*clefG2\n*k[]\n4c\n*-\n");
        write(&input_path.join("b.krn"), "*clefG2\n*k[]\n4d\n*-\n");

        let mut args = concatenate_dir_args(input_path, output_dir);
        args.usage_jsonl = Some(usage_path.clone());
        concatenate_dir_command(args).unwrap();

        let rows = read_jsonl(&usage_path);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| row["status"] == "used"));
        assert!(rows.iter().all(|row| row["reason"] == "emitted"));
        assert!(rows.iter().all(|row| row["output_path"].is_string()));
        assert_eq!(rows[0]["position"], 0);
        assert_eq!(rows[1]["position"], 0);
        assert_eq!(rows[0]["source_count"], 1);
        assert_eq!(rows[1]["source_count"], 1);
    }
}
