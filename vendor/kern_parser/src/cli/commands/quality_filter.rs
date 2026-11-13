use std::path::PathBuf;

use crate::ValidationPass;
use crate::{
    validation::quality_filter::{
        collect_quality_filter_tasks, quality_filter_dir_with_progress, QualityFilterOptions,
        QualityFilterOutcome, QualityFilterStatsPayload,
    },
    CheckConfig, ValidationConfig,
};
use clap::Parser;

use crate::cli::{
    preset::{FramingArg, InputDialectArg},
    progress::Progress,
};

#[derive(Debug, Parser, Clone)]
pub(in crate::cli) struct QualityFilterArgs {
    /// Directory containing .krn files to quality-filter.
    input_dir: PathBuf,
    /// Directory where passing .krn files should be copied.
    output_dir: PathBuf,
    /// Number of worker threads to use.
    #[arg(long, default_value_t = 1)]
    workers: usize,
    /// Training snippets may omit outer wrappers; select complete for authored files.
    #[arg(long, value_enum, default_value = "excerpt")]
    framing: FramingArg,
    /// Syntax dialect of the training inputs.
    #[arg(long, value_enum, default_value = "humdrum-kern")]
    input_dialect: InputDialectArg,
    /// Suppress progress output.
    #[arg(long)]
    quiet: bool,
    /// Stop after the first rejected input and exit non-zero.
    #[arg(long)]
    fail_fast: bool,
    /// Validation pass to run. Repeat to run multiple passes.
    #[arg(long = "rule")]
    passes: Vec<ValidationPass>,
    /// Optional path to write quality-filter statistics as JSON.
    #[arg(long)]
    stats_json: Option<PathBuf>,
}

pub(in crate::cli) fn quality_filter_command(
    args: QualityFilterArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let tasks = collect_quality_filter_tasks(&args.input_dir, &args.output_dir)?;
    if tasks.is_empty() {
        return Err(format!("No .krn files found in {}", args.input_dir.display()).into());
    }

    if !args.quiet {
        eprintln!("Quality-filtering {} files...", tasks.len());
    }

    let config = CheckConfig {
        validation: quality_filter_config(&args.passes),
        framing: args.framing.into(),
        input_dialect: args.input_dialect.into(),
        ..Default::default()
    };
    let progress = Progress::new_with_prefix(tasks.len(), args.quiet, "Quality filter");
    let report = quality_filter_dir_with_progress(
        QualityFilterOptions {
            input_dir: args.input_dir,
            output_dir: args.output_dir,
            workers: args.workers,
            fail_fast: args.fail_fast,
            stats_json: args.stats_json,
            config,
        },
        || progress.inc(),
    );
    progress.finish();
    match report {
        Ok(report) => {
            if !args.quiet {
                print_quality_filter_report(&report.stats, &report.outcomes);
            }
            if report.stats.execution_failure_count > 0 {
                return Err("batch execution failed".into());
            }
            crate::cli::io::accepted(
                report.stats.rejected_count == 0,
                "batch contains rejected inputs",
            )
        }
        Err(error) => Err(error),
    }
}

fn quality_filter_config(passes: &[ValidationPass]) -> ValidationConfig {
    if passes.is_empty() {
        return ValidationConfig::with_passes(vec![
            crate::ValidationPass::State,
            crate::ValidationPass::DurationState,
            crate::ValidationPass::MeterBoundary,
            crate::ValidationPass::Beam,
            crate::ValidationPass::Rhythm,
        ]);
    }

    ValidationConfig::with_passes(passes.to_vec())
}

fn print_quality_filter_report(
    stats: &QualityFilterStatsPayload,
    outcomes: &[QualityFilterOutcome],
) {
    eprintln!(
        "Quality filter accepted {} / {} files into {}",
        stats.accepted_count, stats.input_count, stats.output_dir
    );
    if stats.rejected_count == 0 {
        return;
    }

    eprintln!("Rejected {} files:", stats.rejected_count);
    for outcome in outcomes
        .iter()
        .filter(|outcome| !outcome.accepted())
        .take(10)
    {
        eprintln!("  {}", outcome.input_path.display());
        for finding in &outcome.findings {
            eprintln!(
                "    {} {}: {}",
                finding.phase, finding.code, finding.message
            );
        }
    }
}

#[cfg(test)]
mod tests;
