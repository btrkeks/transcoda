use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use crate::analysis::{RenderFitSearchOptions, RenderFitSearchSeed};
use rayon::prelude::*;

use super::output::{
    create_render_settings_search_writer, write_render_settings_search_profile_row,
    write_render_settings_search_row,
};
use super::profile::ProfiledRenderSettingsSearchRow;
use super::row::RenderSettingsSearchRow;
use super::worker::render_settings_search_file_via_child_profiled;

#[derive(Debug, Default)]
pub(super) struct RenderSettingsSearchSummary {
    pub(super) file_count: usize,
    pub(super) search_success_count: usize,
    pub(super) parse_error_count: usize,
    pub(super) search_error_count: usize,
    pub(super) svg_overlap_gate_row_count: usize,
    pub(super) svg_overlap_gate_hit_count: usize,
    pub(super) svg_overlap_gate_resolved_count: usize,
    pub(super) svg_overlap_gate_unresolved_count: usize,
    pub(super) svg_overlap_rejected_attempt_count: usize,
    pub(super) svg_overlap_fallback_attempt_count: usize,
}

impl RenderSettingsSearchSummary {
    fn add_row(&mut self, row: &RenderSettingsSearchRow) {
        self.file_count += 1;
        if row.search_success() {
            self.search_success_count += 1;
        }
        if row.parse_failed() {
            self.parse_error_count += 1;
        }
        if row.search_failed() {
            self.search_error_count += 1;
        }
        if let Some(gate) = &row.svg_overlap_gate {
            self.svg_overlap_gate_row_count += 1;
            self.svg_overlap_rejected_attempt_count += gate.rejected_attempt_count;
            self.svg_overlap_fallback_attempt_count += gate.fallback_attempt_count;
            if gate.hit {
                self.svg_overlap_gate_hit_count += 1;
                if row.search_success() {
                    self.svg_overlap_gate_resolved_count += 1;
                } else if row.search_failed() {
                    self.svg_overlap_gate_unresolved_count += 1;
                }
            }
        }
    }
}

#[derive(Debug, Default)]
struct AtomicRenderSettingsSearchSummary {
    file_count: AtomicUsize,
    search_success_count: AtomicUsize,
    parse_error_count: AtomicUsize,
    search_error_count: AtomicUsize,
    svg_overlap_gate_row_count: AtomicUsize,
    svg_overlap_gate_hit_count: AtomicUsize,
    svg_overlap_gate_resolved_count: AtomicUsize,
    svg_overlap_gate_unresolved_count: AtomicUsize,
    svg_overlap_rejected_attempt_count: AtomicUsize,
    svg_overlap_fallback_attempt_count: AtomicUsize,
}

impl AtomicRenderSettingsSearchSummary {
    fn add_row(&self, row: &RenderSettingsSearchRow) {
        self.file_count.fetch_add(1, Ordering::Relaxed);
        if row.search_success() {
            self.search_success_count.fetch_add(1, Ordering::Relaxed);
        }
        if row.parse_failed() {
            self.parse_error_count.fetch_add(1, Ordering::Relaxed);
        }
        if row.search_failed() {
            self.search_error_count.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(gate) = &row.svg_overlap_gate {
            self.svg_overlap_gate_row_count
                .fetch_add(1, Ordering::Relaxed);
            self.svg_overlap_rejected_attempt_count
                .fetch_add(gate.rejected_attempt_count, Ordering::Relaxed);
            self.svg_overlap_fallback_attempt_count
                .fetch_add(gate.fallback_attempt_count, Ordering::Relaxed);
            if gate.hit {
                self.svg_overlap_gate_hit_count
                    .fetch_add(1, Ordering::Relaxed);
                if row.search_success() {
                    self.svg_overlap_gate_resolved_count
                        .fetch_add(1, Ordering::Relaxed);
                } else if row.search_failed() {
                    self.svg_overlap_gate_unresolved_count
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    fn snapshot(&self) -> RenderSettingsSearchSummary {
        RenderSettingsSearchSummary {
            file_count: self.file_count.load(Ordering::Relaxed),
            search_success_count: self.search_success_count.load(Ordering::Relaxed),
            parse_error_count: self.parse_error_count.load(Ordering::Relaxed),
            search_error_count: self.search_error_count.load(Ordering::Relaxed),
            svg_overlap_gate_row_count: self.svg_overlap_gate_row_count.load(Ordering::Relaxed),
            svg_overlap_gate_hit_count: self.svg_overlap_gate_hit_count.load(Ordering::Relaxed),
            svg_overlap_gate_resolved_count: self
                .svg_overlap_gate_resolved_count
                .load(Ordering::Relaxed),
            svg_overlap_gate_unresolved_count: self
                .svg_overlap_gate_unresolved_count
                .load(Ordering::Relaxed),
            svg_overlap_rejected_attempt_count: self
                .svg_overlap_rejected_attempt_count
                .load(Ordering::Relaxed),
            svg_overlap_fallback_attempt_count: self
                .svg_overlap_fallback_attempt_count
                .load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct RenderSettingsSearchWorkItem {
    pub(super) path: PathBuf,
    pub(super) candidate_index: usize,
    pub(super) candidate_count: usize,
}

pub(super) fn render_settings_search_work_items(
    files: &[PathBuf],
) -> Vec<RenderSettingsSearchWorkItem> {
    let candidate_count = files.len().max(1);
    files
        .iter()
        .enumerate()
        .map(|(candidate_index, path)| RenderSettingsSearchWorkItem {
            path: path.clone(),
            candidate_index,
            candidate_count,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_render_settings_search_files<P>(
    work_items: &[RenderSettingsSearchWorkItem],
    input_root: &Path,
    workers: usize,
    search_options: RenderFitSearchOptions,
    output_path: &Path,
    output_svg_dir: &Path,
    append_output: bool,
    profile_output_path: Option<&Path>,
    on_progress: P,
) -> Result<RenderSettingsSearchSummary, Box<dyn std::error::Error>>
where
    P: Fn() + Sync,
{
    let worker_exe = env::current_exe()?;
    process_render_settings_search_files_with_worker_and_profile(
        work_items,
        workers,
        output_path,
        append_output,
        profile_output_path,
        &on_progress,
        |item| {
            render_settings_search_file_via_child_profiled(
                &worker_exe,
                &item.path,
                input_root,
                output_svg_dir,
                search_options,
                RenderFitSearchSeed {
                    index: item.candidate_index,
                    count: item.candidate_count,
                },
                profile_output_path.is_some(),
            )
        },
    )
}

#[cfg(test)]
pub(super) fn process_render_settings_search_files_with_worker<F, P, O>(
    work_items: &[RenderSettingsSearchWorkItem],
    workers: usize,
    output_path: &Path,
    append_output: bool,
    on_progress: &P,
    worker: F,
) -> Result<RenderSettingsSearchSummary, Box<dyn std::error::Error>>
where
    F: Fn(&RenderSettingsSearchWorkItem) -> O + Sync,
    O: Into<ProfiledRenderSettingsSearchRow>,
    P: Fn() + Sync,
{
    process_render_settings_search_files_with_worker_and_profile(
        work_items,
        workers,
        output_path,
        append_output,
        None,
        on_progress,
        worker,
    )
}

pub(super) fn process_render_settings_search_files_with_worker_and_profile<F, P, O>(
    work_items: &[RenderSettingsSearchWorkItem],
    workers: usize,
    output_path: &Path,
    append_output: bool,
    profile_output_path: Option<&Path>,
    on_progress: &P,
    worker: F,
) -> Result<RenderSettingsSearchSummary, Box<dyn std::error::Error>>
where
    F: Fn(&RenderSettingsSearchWorkItem) -> O + Sync,
    O: Into<ProfiledRenderSettingsSearchRow>,
    P: Fn() + Sync,
{
    let writer = create_render_settings_search_writer(output_path, append_output)?;
    let profile_writer = profile_output_path
        .map(|path| create_render_settings_search_writer(path, append_output))
        .transpose()?;

    if workers <= 1 {
        return process_render_settings_search_files_serial(
            work_items,
            writer,
            profile_writer,
            on_progress,
            &worker,
        );
    }

    process_render_settings_search_files_parallel(
        work_items,
        workers,
        writer,
        profile_writer,
        on_progress,
        &worker,
    )
}

fn process_render_settings_search_files_serial<F, P, O>(
    work_items: &[RenderSettingsSearchWorkItem],
    mut writer: BufWriter<fs::File>,
    mut profile_writer: Option<BufWriter<fs::File>>,
    on_progress: &P,
    worker: &F,
) -> Result<RenderSettingsSearchSummary, Box<dyn std::error::Error>>
where
    F: Fn(&RenderSettingsSearchWorkItem) -> O + Sync,
    O: Into<ProfiledRenderSettingsSearchRow>,
    P: Fn() + Sync,
{
    let mut summary = RenderSettingsSearchSummary::default();
    for item in work_items {
        let output = worker(item).into();
        write_render_settings_search_row(&mut writer, &output.row)?;
        writer.flush()?;
        if let (Some(profile_writer), Some(profile)) =
            (profile_writer.as_mut(), output.profile.as_ref())
        {
            write_render_settings_search_profile_row(profile_writer, profile)?;
            profile_writer.flush()?;
        }
        summary.add_row(&output.row);
        on_progress();
    }
    writer.flush()?;
    if let Some(profile_writer) = profile_writer.as_mut() {
        profile_writer.flush()?;
    }

    Ok(summary)
}

fn process_render_settings_search_files_parallel<F, P, O>(
    work_items: &[RenderSettingsSearchWorkItem],
    workers: usize,
    writer: BufWriter<fs::File>,
    profile_writer: Option<BufWriter<fs::File>>,
    on_progress: &P,
    worker: &F,
) -> Result<RenderSettingsSearchSummary, Box<dyn std::error::Error>>
where
    F: Fn(&RenderSettingsSearchWorkItem) -> O + Sync,
    O: Into<ProfiledRenderSettingsSearchRow>,
    P: Fn() + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let writer = Arc::new(Mutex::new(writer));
    let profile_writer = Arc::new(Mutex::new(profile_writer));
    let summary = AtomicRenderSettingsSearchSummary::default();

    let result: Result<(), String> = pool.install(|| {
        work_items.par_iter().try_for_each(|item| {
            let output = worker(item).into();
            {
                let mut writer = writer
                    .lock()
                    .map_err(|_| "render settings search writer mutex poisoned".to_owned())?;
                write_render_settings_search_row(&mut writer, &output.row)
                    .map_err(|error| error.to_string())?;
                writer.flush().map_err(|error| error.to_string())?;
            }
            if let Some(profile) = output.profile.as_ref() {
                let mut profile_writer = profile_writer.lock().map_err(|_| {
                    "render settings search profile writer mutex poisoned".to_owned()
                })?;
                if let Some(profile_writer) = profile_writer.as_mut() {
                    write_render_settings_search_profile_row(profile_writer, profile)
                        .map_err(|error| error.to_string())?;
                    profile_writer.flush().map_err(|error| error.to_string())?;
                }
            }
            summary.add_row(&output.row);
            on_progress();
            Ok(())
        })
    });
    result.map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;

    writer
        .lock()
        .map_err(|_| "render settings search writer mutex poisoned")?
        .flush()?;
    if let Some(profile_writer) = profile_writer
        .lock()
        .map_err(|_| "render settings search profile writer mutex poisoned")?
        .as_mut()
    {
        profile_writer.flush()?;
    }

    Ok(summary.snapshot())
}
