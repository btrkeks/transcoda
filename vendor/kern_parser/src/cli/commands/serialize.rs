use std::io::{self, BufReader, BufWriter};
use std::path::PathBuf;

use crate::{
    batch::input::{collect_batch_tasks, DEFAULT_SAMPLE_TRANSCRIPTION_NAME},
    serialize::batch::{
        serialize_dir_with_progress, FileOutcome, SerializeDirOptions, StatsPayload,
    },
    serialize::jsonl::{self, SerializeJsonlOptions},
    BarNumberStyle, ParseOptions, SerializeOptions,
};
use clap::Parser;

use crate::cli::{
    commands::BatchInputLayoutArg,
    preset::{InputDialectArg, SerializePresetArg},
    progress::Progress,
};

#[derive(Debug, Parser, Clone)]
pub(in crate::cli) struct SerializeDirArgs {
    /// Serialization preset to apply.
    #[arg(long, value_enum)]
    preset: SerializePresetArg,
    /// Add sequential measure numbers to serialized barlines.
    #[arg(long)]
    bar_numbers: bool,
    /// Directory containing .krn files.
    input_dir: PathBuf,
    /// Directory where serialized .krn files should be written.
    output_dir: PathBuf,
    /// Number of worker threads to use.
    #[arg(long, default_value_t = 1)]
    workers: usize,
    /// Optional path to write serialization statistics as JSON.
    #[arg(long)]
    stats_json: Option<PathBuf>,
    /// Suppress progress output.
    #[arg(long)]
    quiet: bool,
    /// Stop after the first failed input and exit non-zero.
    #[arg(long)]
    fail_fast: bool,
    /// Input directory layout to use when collecting .krn files.
    #[arg(long, value_enum, default_value = "preserve-relative-paths")]
    input_layout: BatchInputLayoutArg,
    /// File name to read from each sample directory in sample-dir layout.
    #[arg(long, default_value = DEFAULT_SAMPLE_TRANSCRIPTION_NAME)]
    sample_transcription_name: String,
}

#[derive(Debug, Parser, Clone)]
pub(in crate::cli) struct SerializeJsonlArgs {
    /// Serialization preset to apply.
    #[arg(long, value_enum)]
    preset: SerializePresetArg,
    /// Number of worker threads to use.
    #[arg(long, default_value_t = 1)]
    workers: usize,
    /// JSON object field containing input and output transcription. Repeat to serialize
    /// multiple fields in one pass.
    #[arg(long = "text-field", default_value = "transcription")]
    text_fields: Vec<String>,
    /// JSON object field required to identify each row.
    #[arg(long, default_value = "row_index")]
    id_field: String,
    /// Stop after the first failed row and exit non-zero.
    #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true")]
    fail_fast: bool,
    /// Input dialect to parse before serialization.
    #[arg(long, value_enum, default_value = "humdrum-kern")]
    input_dialect: InputDialectArg,
}

impl From<SerializeDirArgs> for SerializeDirOptions {
    fn from(args: SerializeDirArgs) -> Self {
        Self {
            input_dir: args.input_dir,
            output_dir: args.output_dir,
            workers: args.workers,
            stats_json: args.stats_json,
            fail_fast: args.fail_fast,
            preset: args.preset.into(),
            serialize_options: serialize_options(args.bar_numbers),
            input_layout: args.input_layout.into(),
            sample_transcription_name: args.sample_transcription_name,
        }
    }
}

pub(in crate::cli) fn serialize_dir_command(
    args: SerializeDirArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let quiet = args.quiet;
    let input_dir = args.input_dir.clone();
    let options = SerializeDirOptions::from(args);
    let tasks = collect_batch_tasks(
        &options.input_dir,
        &options.output_dir,
        options.input_layout,
        &options.sample_transcription_name,
        false,
    )?;
    if tasks.is_empty() {
        return Err(format!("No .krn files found in {}", input_dir.display()).into());
    }

    if !quiet {
        eprintln!("Serializing {} files...", tasks.len());
    }

    let progress = Progress::new_with_prefix(tasks.len(), quiet, "Serializing");
    let report = serialize_dir_with_progress(options, || progress.inc());
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

pub(in crate::cli) fn serialize_jsonl_command(
    args: SerializeJsonlArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let reader = BufReader::new(stdin.lock());
    let writer = BufWriter::new(stdout.lock());

    let report = jsonl::process_jsonl(reader, writer, &serialize_jsonl_options(&args))?;
    if report.execution_failures > 0 {
        return Err("batch execution failed".into());
    }
    crate::cli::io::accepted(report.rejected == 0, "batch contains rejected inputs")
}

fn serialize_options(bar_numbers: bool) -> SerializeOptions {
    SerializeOptions {
        bar_number_style: if bar_numbers {
            BarNumberStyle::Sequential
        } else {
            BarNumberStyle::Omit
        },
    }
}

fn serialize_jsonl_options(args: &SerializeJsonlArgs) -> SerializeJsonlOptions {
    SerializeJsonlOptions {
        preset: args.preset.into(),
        workers: args.workers,
        text_fields: args.text_fields.clone(),
        id_field: args.id_field.clone(),
        fail_fast: args.fail_fast,
        parse_options: ParseOptions {
            dialect: args.input_dialect.into(),
        },
        serialize_options: serialize_options(false),
    }
}

fn print_summary(stats: &StatsPayload, outcomes: &[FileOutcome]) {
    eprintln!(
        "Serialized {} / {} files into {}",
        stats.success_count, stats.input_count, stats.output_dir
    );
    if stats.error_count > 0 {
        eprintln!("Failed {} files:", stats.error_count);
        for outcome in outcomes.iter().filter(|outcome| !outcome.success).take(10) {
            eprintln!(
                "  {}: {}",
                outcome.input_path.display(),
                outcome
                    .error_message
                    .as_deref()
                    .unwrap_or("unknown serialization error")
            );
        }
    }
}
