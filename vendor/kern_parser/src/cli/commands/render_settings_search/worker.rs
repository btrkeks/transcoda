use std::path::Path;
use std::process::Command as ProcessCommand;
use std::time::Instant;

use crate::analysis::{RenderFitSearchOptions, RenderFitSearchSeed};
use serde::{Deserialize, Serialize};

use super::profile::{
    ProfiledRenderSettingsSearchRow, RenderSettingsSearchPhaseTimings,
    RenderSettingsSearchProfileRow,
};
use super::row::{RenderSettingsSearchRow, RenderSettingsSearchStatus};
use super::{SvgOverlapFallbackArg, SvgOverlapGateArg};

#[cfg(test)]
pub(super) fn render_settings_search_file_via_child(
    worker_exe: &Path,
    path: &Path,
    input_root: &Path,
    output_svg_dir: &Path,
    search_options: RenderFitSearchOptions,
    seed: RenderFitSearchSeed,
) -> RenderSettingsSearchRow {
    render_settings_search_file_via_child_profiled(
        worker_exe,
        path,
        input_root,
        output_svg_dir,
        search_options,
        seed,
        false,
    )
    .row
}

pub(super) fn render_settings_search_file_via_child_profiled(
    worker_exe: &Path,
    path: &Path,
    input_root: &Path,
    output_svg_dir: &Path,
    search_options: RenderFitSearchOptions,
    seed: RenderFitSearchSeed,
    collect_profile: bool,
) -> ProfiledRenderSettingsSearchRow {
    let gate = SvgOverlapGateArg::from(search_options);
    let fallback = SvgOverlapFallbackArg::from(search_options);
    let mut command = ProcessCommand::new(worker_exe);
    command
        .arg("__render-settings-search-worker")
        .arg("--input-root")
        .arg(input_root)
        .arg("--output-svg-dir")
        .arg(output_svg_dir)
        .arg("--svg-overlap-gate")
        .arg(gate.as_str())
        .arg("--svg-overlap-min-ratio")
        .arg(search_options.svg_overlap_min_ratio.to_string())
        .arg("--svg-overlap-fallback")
        .arg(fallback.as_str())
        .arg("--candidate-index")
        .arg(seed.index.to_string())
        .arg("--candidate-count")
        .arg(seed.count.to_string());
    if collect_profile {
        command.arg("--profile");
    }
    let process_start = Instant::now();
    match command.arg(path).output() {
        Ok(output) => render_settings_search_output_from_child_output(
            path,
            output,
            collect_profile,
            process_start.elapsed().as_secs_f64() * 1000.0,
        ),
        Err(error) => {
            let row = RenderSettingsSearchRow::worker_failure(
                path,
                format!("failed to launch render settings search worker: {error}"),
                None,
            );
            worker_failure_output(
                row,
                collect_profile,
                process_start.elapsed().as_secs_f64() * 1000.0,
            )
        }
    }
}

impl From<RenderFitSearchOptions> for SvgOverlapGateArg {
    fn from(options: RenderFitSearchOptions) -> Self {
        match options.svg_overlap_gate {
            crate::analysis::SvgOverlapGateMode::Off => Self::Off,
            crate::analysis::SvgOverlapGateMode::Reject => Self::Reject,
        }
    }
}

impl From<RenderFitSearchOptions> for SvgOverlapFallbackArg {
    fn from(options: RenderFitSearchOptions) -> Self {
        match options.svg_overlap_fallback {
            crate::analysis::SvgOverlapFallbackMode::Off => Self::Off,
            crate::analysis::SvgOverlapFallbackMode::HorizontalRelief => Self::HorizontalRelief,
        }
    }
}

#[cfg(test)]
pub(super) fn render_settings_search_row_from_child_output(
    path: &Path,
    output: std::process::Output,
) -> RenderSettingsSearchRow {
    render_settings_search_output_from_child_output(path, output, false, 0.0).row
}

fn render_settings_search_output_from_child_output(
    path: &Path,
    output: std::process::Output,
    collect_profile: bool,
    worker_process_millis: f64,
) -> ProfiledRenderSettingsSearchRow {
    let stderr = stderr_field(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut json_lines = stdout.lines().filter(|line| !line.trim().is_empty());
    let first_line = json_lines.next();
    let extra_line = json_lines.next();

    let Some(first_line) = first_line else {
        let row = RenderSettingsSearchRow::worker_failure(
            path,
            worker_failure_message(
                "render settings search worker produced no JSON row",
                output.status,
            ),
            stderr,
        );
        return worker_failure_output(row, collect_profile, worker_process_millis);
    };

    if extra_line.is_some() {
        let row = RenderSettingsSearchRow::worker_failure(
            path,
            worker_failure_message(
                "render settings search worker produced multiple JSON rows",
                output.status,
            ),
            stderr,
        );
        return worker_failure_output(row, collect_profile, worker_process_millis);
    }

    let mut output_row = if collect_profile {
        match serde_json::from_str::<RenderSettingsSearchWorkerOutput>(first_line) {
            Ok(output) => ProfiledRenderSettingsSearchRow {
                row: output.row,
                profile: output
                    .profile
                    .map(|profile| profile.with_worker_process_millis(worker_process_millis)),
            },
            Err(error) => {
                let row = RenderSettingsSearchRow::worker_failure(
                    path,
                    worker_failure_message(
                        &format!(
                            "render settings search worker produced invalid profiled JSON: {error}"
                        ),
                        output.status,
                    ),
                    stderr,
                );
                return worker_failure_output(row, collect_profile, worker_process_millis);
            }
        }
    } else {
        match serde_json::from_str::<RenderSettingsSearchRow>(first_line) {
            Ok(row) => ProfiledRenderSettingsSearchRow { row, profile: None },
            Err(error) => {
                let row = RenderSettingsSearchRow::worker_failure(
                    path,
                    worker_failure_message(
                        &format!("render settings search worker produced invalid JSON: {error}"),
                        output.status,
                    ),
                    stderr,
                );
                return worker_failure_output(row, collect_profile, worker_process_millis);
            }
        }
    };

    if !output.status.success() {
        output_row.row.status = RenderSettingsSearchStatus::SearchError;
        output_row.row.search_error = Some(worker_failure_message(
            "render settings search worker exited unsuccessfully",
            output.status,
        ));
    }
    output_row.row.verovio_stderr = stderr;
    output_row
}

pub(super) fn worker_output(
    output: ProfiledRenderSettingsSearchRow,
) -> RenderSettingsSearchWorkerOutput {
    RenderSettingsSearchWorkerOutput {
        row: output.row,
        profile: output.profile,
    }
}

fn worker_failure_output(
    row: RenderSettingsSearchRow,
    collect_profile: bool,
    worker_process_millis: f64,
) -> ProfiledRenderSettingsSearchRow {
    if !collect_profile {
        return ProfiledRenderSettingsSearchRow { row, profile: None };
    }

    let profile = RenderSettingsSearchProfileRow::from_row(
        &row,
        RenderSettingsSearchPhaseTimings {
            worker_process: Some(super::profile::round_millis(worker_process_millis)),
            ..RenderSettingsSearchPhaseTimings::default()
        },
        None,
    );
    ProfiledRenderSettingsSearchRow {
        row,
        profile: Some(profile),
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RenderSettingsSearchWorkerOutput {
    row: RenderSettingsSearchRow,
    profile: Option<RenderSettingsSearchProfileRow>,
}

fn worker_failure_message(message: &str, status: std::process::ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{message} (exit code {code})"),
        None => format!("{message} (terminated by signal)"),
    }
}

fn stderr_field(stderr: &[u8]) -> Option<String> {
    if stderr.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(stderr).into_owned())
    }
}
