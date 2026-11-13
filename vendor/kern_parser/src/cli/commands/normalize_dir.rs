use std::path::PathBuf;

#[cfg(test)]
use crate::{batch::input::BatchInputLayout, SerializePreset};
use crate::{
    batch::input::DEFAULT_SAMPLE_TRANSCRIPTION_NAME,
    normalization::batch::{normalize_dir_with_progress, FileOutcome, StatsPayload},
    NormalizeDirOptions,
};
use clap::Parser;

use crate::cli::{commands::BatchInputLayoutArg, preset::SerializePresetArg, progress::Progress};

#[derive(Debug, Parser, Clone)]
pub(in crate::cli) struct NormalizeArgs {
    /// .krn file or directory containing cleaned .krn files.
    input_path: PathBuf,
    /// Output file for single-file input, or output directory for directory input.
    output_path: PathBuf,
    /// Number of worker threads to use.
    #[arg(long, default_value_t = 1)]
    workers: usize,
    /// Optional path to write normalization statistics as JSON.
    #[arg(long)]
    stats_json: Option<PathBuf>,
    /// Suppress progress output.
    #[arg(long)]
    quiet: bool,
    /// Stop after the first failed input and exit non-zero.
    #[arg(long)]
    fail_fast: bool,
    /// Output dialect for normalized files.
    #[arg(long, value_enum, default_value = "humdrum-kern")]
    output_dialect: SerializePresetArg,
    /// Input directory layout to use when collecting .krn files.
    #[arg(long, value_enum, default_value = "preserve-relative-paths")]
    input_layout: BatchInputLayoutArg,
    /// File name to read from each sample directory in sample-dir layout.
    #[arg(long, default_value = DEFAULT_SAMPLE_TRANSCRIPTION_NAME)]
    sample_transcription_name: String,
}

impl From<NormalizeArgs> for NormalizeDirOptions {
    fn from(args: NormalizeArgs) -> Self {
        Self {
            input_dir: args.input_path,
            output_dir: args.output_path,
            workers: args.workers,
            stats_json: args.stats_json,
            fail_fast: args.fail_fast,
            output_preset: args.output_dialect.into(),
            input_layout: args.input_layout.into(),
            sample_transcription_name: args.sample_transcription_name,
        }
    }
}

pub(in crate::cli) fn normalize_command(
    args: NormalizeArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let quiet = args.quiet;
    let input_path = args.input_path.clone();
    let options = NormalizeDirOptions::from(args);

    let total = crate::batch::input::collect_batch_tasks(
        &options.input_dir,
        &options.output_dir,
        options.input_layout,
        &options.sample_transcription_name,
        true,
    )?
    .len();
    if total == 0 {
        return Err(format!("No .krn files found in {}", input_path.display()).into());
    }

    if !quiet {
        eprintln!("Normalizing {total} files...");
    }
    let progress = Progress::new_with_prefix(total, quiet, "Normalizing");
    let report = normalize_dir_with_progress(options, || progress.inc());
    progress.finish();
    match report {
        Ok(report) => {
            if !quiet {
                print_summary(&report.stats, &report.outcomes);
            }
            if report.stats.execution_failure_count > 0 {
                return Err("batch execution failed".into());
            }
            crate::cli::io::accepted(
                report.stats.error_count == 0,
                "batch contains rejected inputs",
            )
        }
        Err(error) => Err(error),
    }
}

fn print_summary(stats: &StatsPayload, outcomes: &[FileOutcome]) {
    eprintln!();
    eprintln!("Results:");
    eprintln!("  Success: {}", stats.success_count);
    eprintln!("  Errors:  {}", stats.error_count);
    eprintln!();
    eprintln!("Normalized files saved to {}", stats.output_dir);

    let failed = outcomes
        .iter()
        .filter(|outcome| !outcome.success)
        .take(10)
        .collect::<Vec<_>>();
    if failed.is_empty() {
        return;
    }

    eprintln!();
    eprintln!("Failed files:");
    for outcome in failed {
        let prefix = outcome
            .error_type
            .as_ref()
            .map_or(String::new(), |error_type| format!("{error_type}: "));
        eprintln!(
            "  {}: {}{}",
            outcome
                .input_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<unknown>"),
            prefix,
            outcome.error_message.as_deref().unwrap_or("")
        );
    }
    if stats.error_count > 10 {
        eprintln!("  ... and {} more", stats.error_count - 10);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_output_dialect_is_humdrum_kern() {
        let args = NormalizeArgs::try_parse_from(["normalize", "in", "out"]).unwrap();

        let options = NormalizeDirOptions::from(args);
        assert_eq!(options.output_preset, SerializePreset::HumdrumKern);
    }

    #[test]
    fn parses_sample_dir_layout_defaults() {
        let args = NormalizeArgs::try_parse_from([
            "normalize",
            "in",
            "out",
            "--input-layout",
            "sample-dirs",
        ])
        .unwrap();

        let options = NormalizeDirOptions::from(args);
        assert_eq!(options.input_layout, BatchInputLayout::SampleDirs);
        assert_eq!(
            options.sample_transcription_name,
            DEFAULT_SAMPLE_TRANSCRIPTION_NAME
        );
    }

    #[test]
    fn rejects_old_output_dialect_values() {
        let error = NormalizeArgs::try_parse_from([
            "normalize",
            "in",
            "out",
            "--output-dialect",
            "transcoda-omr-training-target",
        ])
        .unwrap_err();

        assert!(error.to_string().contains("invalid value"));
    }
}
