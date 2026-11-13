use std::sync::Mutex;
use std::time::Duration;

use crate::concatenation::ConcatenationProgress;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Clone)]
pub(super) struct Progress {
    bar: Option<ProgressBar>,
}

impl Progress {
    pub(super) fn new_with_prefix(total: usize, quiet: bool, prefix: &'static str) -> Self {
        if quiet {
            return Self { bar: None };
        }

        let bar = ProgressBar::new(total as u64);
        let style = ProgressStyle::with_template(
            "{prefix:.bold} [{bar:40.cyan/blue}] {pos}/{len} elapsed={elapsed_precise} eta={eta_precise}",
        )
        .expect("progress style template should be valid")
        .progress_chars("=> ");
        bar.set_style(style);
        bar.set_prefix(prefix);

        Self { bar: Some(bar) }
    }

    pub(super) fn inc(&self) {
        if let Some(bar) = &self.bar {
            bar.inc(1);
        }
    }

    pub(super) fn finish(&self) {
        if let Some(bar) = &self.bar {
            bar.finish_and_clear();
        }
    }
}

pub(super) struct ConcatenateProgressDisplay {
    quiet: bool,
    manifest: Mutex<Option<ProgressBar>>,
    generation: Mutex<Option<ProgressBar>>,
}

impl ConcatenateProgressDisplay {
    pub(super) fn new(quiet: bool) -> Self {
        Self {
            quiet,
            manifest: Mutex::new(None),
            generation: Mutex::new(None),
        }
    }
}

impl ConcatenationProgress for ConcatenateProgressDisplay {
    fn manifest_started(&self, total: usize) {
        if self.quiet {
            return;
        }

        let bar = ProgressBar::new(total as u64);
        let style = ProgressStyle::with_template(
            "{prefix:.bold} [{bar:40.cyan/blue}] {pos}/{len} files elapsed={elapsed_precise} eta={eta_precise}",
        )
        .expect("progress style template should be valid")
        .progress_chars("=> ");
        bar.set_style(style);
        bar.set_prefix("Manifest");
        *self
            .manifest
            .lock()
            .expect("manifest progress mutex poisoned") = Some(bar);
    }

    fn manifest_advanced(&self) {
        if let Some(bar) = self
            .manifest
            .lock()
            .expect("manifest progress mutex poisoned")
            .as_ref()
        {
            bar.inc(1);
        }
    }

    fn manifest_finished(&self, parseable_count: usize, skipped_parse_count: usize) {
        if self.quiet {
            return;
        }

        if let Some(bar) = self
            .manifest
            .lock()
            .expect("manifest progress mutex poisoned")
            .take()
        {
            bar.finish_and_clear();
        }
        eprintln!(
            "Manifest: {parseable_count} parseable files, {skipped_parse_count} skipped parse errors"
        );
    }

    fn generation_started(&self, parseable_count: usize, target_count: Option<usize>) {
        if self.quiet {
            return;
        }

        let bar = if let Some(target_count) = target_count {
            let bar = ProgressBar::new(target_count as u64);
            let style = ProgressStyle::with_template(
                "{prefix:.bold} [{bar:40.green/blue}] {pos}/{len} outputs {msg} elapsed={elapsed_precise} eta={eta_precise}",
            )
            .expect("progress style template should be valid")
            .progress_chars("=> ");
            bar.set_style(style);
            bar
        } else {
            let bar = ProgressBar::new_spinner();
            let style = ProgressStyle::with_template(
                "{prefix:.bold} {spinner} outputs={pos} {msg} {elapsed_precise}",
            )
            .expect("progress style template should be valid");
            bar.set_style(style);
            bar.enable_steady_tick(Duration::from_millis(100));
            bar
        };

        bar.set_prefix("Concatenating");
        bar.set_message(format!("sources={parseable_count}"));
        *self
            .generation
            .lock()
            .expect("generation progress mutex poisoned") = Some(bar);
    }

    fn generation_progress(
        &self,
        output_count: usize,
        remaining_sources: usize,
        staff_count: Option<usize>,
    ) {
        let Some(bar) = self
            .generation
            .lock()
            .expect("generation progress mutex poisoned")
            .as_ref()
            .cloned()
        else {
            return;
        };

        bar.set_position(output_count as u64);
        let staff = staff_count
            .map(|count| format!(" staff={count}"))
            .unwrap_or_default();
        bar.set_message(format!("remaining_sources={remaining_sources}{staff}"));
    }

    fn generation_finished(&self, output_count: usize) {
        if self.quiet {
            return;
        }

        if let Some(bar) = self
            .generation
            .lock()
            .expect("generation progress mutex poisoned")
            .take()
        {
            bar.set_position(output_count as u64);
            bar.finish_and_clear();
        }
    }
}
