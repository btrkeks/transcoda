//! CLI entry point for the rhythm checker.

use clap::Parser;
use colored::Colorize;
use rayon::prelude::*;
use rhythm_checker::{ValidationSummary, Validator, ValidatorOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Validate rhythm correctness in Humdrum **kern files.
///
/// Checks that each measure contains the correct total duration based on
/// the time signature, using exact rational arithmetic.
#[derive(Parser, Debug)]
#[command(name = "rhythm_checker")]
#[command(version, about, long_about = None)]
struct Args {
    /// Files or directories to validate.
    ///
    /// Directories are searched recursively for .krn files.
    #[arg(required = true)]
    paths: Vec<PathBuf>,

    /// Output format.
    #[arg(short, long, value_enum, default_value = "human")]
    format: OutputFormat,

    /// Number of parallel jobs (0 = auto-detect CPU count).
    #[arg(short, long, default_value = "0")]
    jobs: usize,

    /// Only show errors (suppress progress and summary for valid files).
    #[arg(short, long)]
    quiet: bool,

    /// Show per-measure validation details.
    #[arg(short, long)]
    verbose: bool,

    /// Permit incomplete first measure (anacrusis/pickup).
    #[arg(long, default_value = "true")]
    allow_anacrusis: bool,

    /// Permit incomplete final measure (at double barline).
    #[arg(long, default_value = "true")]
    allow_incomplete_final: bool,

    /// Permit incomplete measures at repeat boundaries to pair with initial pickup.
    #[arg(long, default_value = "true")]
    allow_repeat_pairing: bool,

    /// Stop on first error encountered.
    #[arg(long)]
    fail_fast: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum OutputFormat {
    /// Human-readable colored output.
    Human,
    /// JSON output for CI integration.
    Json,
}

fn main() {
    let args = Args::parse();

    // Configure thread pool
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .ok();
    }

    // Collect all files to validate
    let files = collect_files(&args.paths);

    if files.is_empty() {
        if !args.quiet {
            eprintln!("{}: no .krn files found", "warning".yellow());
        }
        std::process::exit(0);
    }

    // Create validator options
    let options = ValidatorOptions {
        allow_anacrusis: args.allow_anacrusis,
        allow_incomplete_final: args.allow_incomplete_final,
        allow_repeat_pairing: args.allow_repeat_pairing,
        verbose: args.verbose,
    };

    // Validate files in parallel
    let is_human_format = args.format == OutputFormat::Human;
    let summary = validate_files(
        &files,
        &options,
        args.quiet,
        is_human_format,
        is_human_format,
        args.fail_fast,
    );

    // Output results
    match args.format {
        OutputFormat::Human => print_human_summary(&summary, args.quiet),
        OutputFormat::Json => print_json_summary(&summary),
    }

    // Exit with error code if any files failed
    if !summary.is_ok() {
        std::process::exit(1);
    }
}

/// Collect all .krn files from the given paths.
fn collect_files(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut files = Vec::new();

    for path in paths {
        if path.is_file() {
            if path.extension().map_or(false, |e| e == "krn") {
                files.push(path.clone());
            }
        } else if path.is_dir() {
            collect_files_recursive(path, &mut files);
        }
    }

    files.sort();
    files
}

/// Recursively collect .krn files from a directory.
fn collect_files_recursive(dir: &PathBuf, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if path.extension().map_or(false, |e| e == "krn") {
                    files.push(path);
                }
            } else if path.is_dir() {
                collect_files_recursive(&path, files);
            }
        }
    }
}

/// Validate all files and return a summary.
fn validate_files(
    files: &[PathBuf],
    options: &ValidatorOptions,
    quiet: bool,
    show_progress: bool,
    print_errors: bool,
    fail_fast: bool,
) -> ValidationSummary {
    let validator = Validator::new(options.clone());
    let summary = Mutex::new(ValidationSummary::new());
    let processed = AtomicUsize::new(0);
    let total = files.len();
    let stop_flag = AtomicBool::new(false);

    files.par_iter().for_each(|path| {
        // Check if we should stop early
        if fail_fast && stop_flag.load(Ordering::Relaxed) {
            return;
        }

        let result = validator.validate_file(path);
        let has_errors = !result.is_ok();

        // Print errors immediately (thread-safe via mutex) - only in human mode
        if has_errors && !quiet && print_errors {
            let errors_str: String = result
                .errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("\n");

            // Lock for printing to avoid interleaved output
            let _guard = summary.lock().unwrap();
            eprintln!("{}", errors_str);
        }

        // Update summary
        {
            let mut s = summary.lock().unwrap();
            s.add(result);
        }

        // Signal stop if fail-fast and we found an error
        if fail_fast && has_errors {
            stop_flag.store(true, Ordering::Relaxed);
        }

        // Progress indicator
        if show_progress && !quiet {
            let count = processed.fetch_add(1, Ordering::Relaxed) + 1;
            if count % 100 == 0 || count == total {
                eprint!("\r{}/{} files processed", count, total);
            }
        }
    });

    if show_progress && !quiet {
        eprintln!(); // newline after progress
    }

    summary.into_inner().unwrap()
}

/// Print human-readable summary.
fn print_human_summary(summary: &ValidationSummary, quiet: bool) {
    if quiet && summary.is_ok() {
        return;
    }

    println!();

    if summary.is_ok() {
        println!(
            "{} {} files, {} measures",
            "OK".green().bold(),
            summary.files_processed,
            summary.total_measures
        );
    } else {
        println!(
            "{} {} errors in {} of {} files",
            "FAIL".red().bold(),
            summary.total_errors,
            summary.files_with_errors,
            summary.files_processed
        );
    }
}

/// Print JSON summary.
fn print_json_summary(summary: &ValidationSummary) {
    // For JSON, only include files with errors to reduce output size
    let filtered = ValidationSummary {
        files_processed: summary.files_processed,
        files_with_errors: summary.files_with_errors,
        total_errors: summary.total_errors,
        total_measures: summary.total_measures,
        file_results: summary
            .file_results
            .iter()
            .filter(|r| !r.is_ok())
            .cloned()
            .collect(),
    };

    println!("{}", serde_json::to_string_pretty(&filtered).unwrap());
}
