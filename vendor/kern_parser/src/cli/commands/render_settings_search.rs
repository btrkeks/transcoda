use std::collections::BTreeMap;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use crate::analysis::{
    RenderFitSearchOptions, RenderFitSearchSeed, SvgOverlapFallbackMode, SvgOverlapGateMode,
};
use crate::dataset_manifest::{update_dataset_manifest, DatasetComponentUpdate};
use clap::{Parser, ValueEnum};

use super::collect_krn_files;
use crate::cli::progress::Progress;

mod output;
mod profile;
mod row;
mod runner;
mod service;
mod worker;

use output::{load_render_settings_search_resume, resume_file_key};
use runner::{
    process_render_settings_search_files, render_settings_search_work_items,
    RenderSettingsSearchSummary,
};
use service::{render_settings_search_file, render_settings_search_file_profiled};
use worker::worker_output;

#[derive(Debug, Parser, Clone)]
pub(in crate::cli) struct RenderSettingsSearchArgs {
    /// Directory containing .krn files to analyze.
    input_dir: PathBuf,
    /// Path to write one JSON object per line.
    #[arg(long)]
    output_jsonl: PathBuf,
    /// Directory to write selected SVG renderings.
    #[arg(long)]
    output_svg_dir: PathBuf,
    /// Number of worker threads to use.
    #[arg(long, default_value_t = 1)]
    workers: usize,
    /// SVG overlap gate behavior.
    #[arg(long, value_enum, default_value_t = SvgOverlapGateArg::Off)]
    svg_overlap_gate: SvgOverlapGateArg,
    /// Minimum overlap ratio for rejecting note/rest bounding-box overlaps.
    #[arg(long, default_value_t = 0.15)]
    svg_overlap_min_ratio: f64,
    /// Fallback strategy used after an SVG overlap rejection.
    #[arg(long, value_enum, default_value_t = SvgOverlapFallbackArg::HorizontalRelief)]
    svg_overlap_fallback: SvgOverlapFallbackArg,
    /// Suppress progress output.
    #[arg(long)]
    quiet: bool,
    /// Resume from an existing JSONL output, skipping files with success or parse-error rows.
    #[arg(long)]
    resume: bool,
    /// Optional path to write per-file profiling and phase timing JSONL.
    #[arg(long)]
    profile_jsonl: Option<PathBuf>,
}

#[derive(Debug, Parser, Clone)]
pub(in crate::cli) struct RenderSettingsSearchWorkerArgs {
    /// Single .krn file to analyze.
    file: PathBuf,
    /// Input root used to derive mirrored SVG output paths.
    #[arg(long)]
    input_root: PathBuf,
    /// Directory to write selected SVG renderings.
    #[arg(long)]
    output_svg_dir: PathBuf,
    /// SVG overlap gate behavior.
    #[arg(long, value_enum, default_value_t = SvgOverlapGateArg::Off, hide = true)]
    svg_overlap_gate: SvgOverlapGateArg,
    /// Minimum overlap ratio for rejecting note/rest bounding-box overlaps.
    #[arg(long, default_value_t = 0.15, hide = true)]
    svg_overlap_min_ratio: f64,
    /// Fallback strategy used after an SVG overlap rejection.
    #[arg(long, value_enum, default_value_t = SvgOverlapFallbackArg::HorizontalRelief, hide = true)]
    svg_overlap_fallback: SvgOverlapFallbackArg,
    /// Stable candidate sequence index assigned by the parent process.
    #[arg(long, default_value_t = 0, hide = true)]
    candidate_index: usize,
    /// Total candidate sequence count assigned by the parent process.
    #[arg(long, default_value_t = 1, hide = true)]
    candidate_count: usize,
    /// Emit a profiled worker-output wrapper instead of a bare row.
    #[arg(long, hide = true)]
    profile: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SvgOverlapGateArg {
    Off,
    Reject,
}

impl SvgOverlapGateArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Reject => "reject",
        }
    }
}

impl From<SvgOverlapGateArg> for SvgOverlapGateMode {
    fn from(mode: SvgOverlapGateArg) -> Self {
        match mode {
            SvgOverlapGateArg::Off => Self::Off,
            SvgOverlapGateArg::Reject => Self::Reject,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SvgOverlapFallbackArg {
    Off,
    HorizontalRelief,
}

impl SvgOverlapFallbackArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::HorizontalRelief => "horizontal-relief",
        }
    }
}

impl From<SvgOverlapFallbackArg> for SvgOverlapFallbackMode {
    fn from(mode: SvgOverlapFallbackArg) -> Self {
        match mode {
            SvgOverlapFallbackArg::Off => Self::Off,
            SvgOverlapFallbackArg::HorizontalRelief => Self::HorizontalRelief,
        }
    }
}

fn render_fit_search_options(
    gate: SvgOverlapGateArg,
    min_ratio: f64,
    fallback: SvgOverlapFallbackArg,
) -> RenderFitSearchOptions {
    RenderFitSearchOptions {
        svg_overlap_gate: gate.into(),
        svg_overlap_min_ratio: min_ratio,
        svg_overlap_fallback: fallback.into(),
    }
}

pub(in crate::cli) fn render_settings_search_command(
    args: RenderSettingsSearchArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let files = collect_krn_files(&args.input_dir)?;
    let mut work_items = render_settings_search_work_items(&files);

    if args.resume {
        let completed =
            load_render_settings_search_resume(&args.output_jsonl, &args.input_dir, args.quiet)?;
        let original_file_count = work_items.len();
        work_items
            .retain(|item| !completed.contains(&resume_file_key(&item.path, &args.input_dir)));
        if !args.quiet {
            eprintln!(
                "Resuming render settings search: skipping {} completed files, processing {} remaining files.",
                original_file_count - work_items.len(),
                work_items.len()
            );
        }
    }

    if !args.quiet {
        eprintln!(
            "Searching render settings for {} files...",
            work_items.len()
        );
    }

    let progress = Progress::new_with_prefix(work_items.len(), args.quiet, "Render settings");
    let summary = process_render_settings_search_files(
        &work_items,
        &args.input_dir,
        args.workers,
        render_fit_search_options(
            args.svg_overlap_gate,
            args.svg_overlap_min_ratio,
            args.svg_overlap_fallback,
        ),
        &args.output_jsonl,
        &args.output_svg_dir,
        args.resume,
        args.profile_jsonl.as_deref(),
        || progress.inc(),
    )?;
    progress.finish();

    if !args.quiet {
        print_render_settings_search_summary(&summary, start.elapsed().as_secs_f64());
    }
    update_render_settings_manifest(&args, &summary)?;

    Ok(())
}

pub(in crate::cli) fn render_settings_search_worker_command(
    args: RenderSettingsSearchWorkerArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    let search_options = render_fit_search_options(
        args.svg_overlap_gate,
        args.svg_overlap_min_ratio,
        args.svg_overlap_fallback,
    );
    let seed = RenderFitSearchSeed {
        index: args.candidate_index,
        count: args.candidate_count,
    };
    if args.profile {
        let output = render_settings_search_file_profiled(
            &args.file,
            &args.input_root,
            &args.output_svg_dir,
            search_options,
            seed,
            true,
        );
        serde_json::to_writer(&mut stdout, &worker_output(output))?;
    } else {
        let row = render_settings_search_file(
            &args.file,
            &args.input_root,
            &args.output_svg_dir,
            search_options,
            seed,
        );
        serde_json::to_writer(&mut stdout, &row)?;
    }
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn print_render_settings_search_summary(
    summary: &RenderSettingsSearchSummary,
    duration_seconds: f64,
) {
    eprintln!();
    eprintln!("Render settings search:");
    eprintln!("  Files:          {}", summary.file_count);
    eprintln!("  Search success: {}", summary.search_success_count);
    eprintln!("  Parse errors:   {}", summary.parse_error_count);
    eprintln!("  Search errors:  {}", summary.search_error_count);
    if summary.svg_overlap_gate_row_count > 0 {
        eprintln!("  SVG overlap hits: {}", summary.svg_overlap_gate_hit_count);
        eprintln!(
            "  SVG overlap resolved: {}",
            summary.svg_overlap_gate_resolved_count
        );
        eprintln!(
            "  SVG overlap unresolved: {}",
            summary.svg_overlap_gate_unresolved_count
        );
        eprintln!(
            "  SVG overlap rejected attempts: {}",
            summary.svg_overlap_rejected_attempt_count
        );
        eprintln!(
            "  SVG overlap fallback attempts: {}",
            summary.svg_overlap_fallback_attempt_count
        );
    }
    eprintln!("  Duration:       {duration_seconds:.2}s");
}

fn update_render_settings_manifest(
    args: &RenderSettingsSearchArgs,
    summary: &RenderSettingsSearchSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(dataset_root) = args.output_svg_dir.parent() else {
        return Ok(());
    };
    let mut summary_values = BTreeMap::new();
    summary_values.insert("file_count".to_owned(), summary.file_count);
    summary_values.insert(
        "search_success_count".to_owned(),
        summary.search_success_count,
    );
    summary_values.insert("parse_error_count".to_owned(), summary.parse_error_count);
    summary_values.insert("search_error_count".to_owned(), summary.search_error_count);
    summary_values.insert(
        "svg_overlap_gate_hit_count".to_owned(),
        summary.svg_overlap_gate_hit_count,
    );
    summary_values.insert(
        "svg_overlap_gate_resolved_count".to_owned(),
        summary.svg_overlap_gate_resolved_count,
    );
    summary_values.insert(
        "svg_overlap_gate_unresolved_count".to_owned(),
        summary.svg_overlap_gate_unresolved_count,
    );
    summary_values.insert(
        "svg_overlap_rejected_attempt_count".to_owned(),
        summary.svg_overlap_rejected_attempt_count,
    );
    summary_values.insert(
        "svg_overlap_fallback_attempt_count".to_owned(),
        summary.svg_overlap_fallback_attempt_count,
    );
    update_dataset_manifest(
        dataset_root,
        DatasetComponentUpdate {
            name: "svg_renders".to_owned(),
            producer: "kern_parser render-settings-search".to_owned(),
            paths: vec![args.output_svg_dir.clone(), args.output_jsonl.clone()],
            summary: summary_values,
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
