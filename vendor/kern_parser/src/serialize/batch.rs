use std::collections::BTreeMap;
use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;

use crate::{
    batch::input::{collect_batch_tasks, BatchInputLayout, BatchTask},
    KernDocument, SerializeOptions, SerializePreset,
};

#[derive(Debug, Clone)]
pub struct SerializeDirOptions {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub workers: usize,
    pub stats_json: Option<PathBuf>,
    pub fail_fast: bool,
    pub preset: SerializePreset,
    pub serialize_options: SerializeOptions,
    pub input_layout: BatchInputLayout,
    pub sample_transcription_name: String,
}

#[derive(Debug)]
pub struct SerializeDirReport {
    pub stats: StatsPayload,
    pub outcomes: Vec<FileOutcome>,
}

#[derive(Debug, Clone)]
pub struct FileOutcome {
    pub input_path: PathBuf,
    pub success: bool,
    pub execution_failure: bool,
    pub error_type: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StatsPayload {
    pub schema_version: &'static str,
    pub input_dir: String,
    pub output_dir: String,
    pub input_count: usize,
    pub success_count: usize,
    pub error_count: usize,
    pub execution_failure_count: usize,
    pub pass_rate: f64,
    pub duration_seconds: f64,
    pub error_types: BTreeMap<String, usize>,
    pub failed_examples: Vec<FailedExample>,
}

#[derive(Debug, Serialize)]
pub struct FailedExample {
    pub file: String,
    pub error_type: Option<String>,
    pub error: Option<String>,
}

pub fn serialize_dir(
    options: SerializeDirOptions,
) -> Result<SerializeDirReport, Box<dyn std::error::Error>> {
    serialize_dir_with_progress(options, || {})
}

pub fn serialize_dir_with_progress<F>(
    options: SerializeDirOptions,
    progress: F,
) -> Result<SerializeDirReport, Box<dyn std::error::Error>>
where
    F: Fn() + Sync,
{
    let start = Instant::now();
    let tasks = collect_batch_tasks(
        &options.input_dir,
        &options.output_dir,
        options.input_layout,
        &options.sample_transcription_name,
        false,
    )?;
    if tasks.is_empty() {
        return Err(format!("No .krn files found in {}", options.input_dir.display()).into());
    }

    let outcomes = process_tasks(
        &tasks,
        options.workers,
        options.fail_fast,
        options.preset,
        options.serialize_options,
        progress,
    )?;
    let stats = build_stats(
        &options.input_dir,
        &options.output_dir,
        tasks.len(),
        &outcomes,
        start.elapsed().as_secs_f64(),
    );

    if let Some(stats_path) = options.stats_json {
        if let Some(parent) = stats_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(stats_path, serde_json::to_string_pretty(&stats)?)?;
    }

    Ok(SerializeDirReport { stats, outcomes })
}

pub fn process_tasks<F>(
    tasks: &[BatchTask],
    workers: usize,
    fail_fast: bool,
    preset: SerializePreset,
    serialize_options: SerializeOptions,
    progress: F,
) -> Result<Vec<FileOutcome>, rayon::ThreadPoolBuildError>
where
    F: Fn() + Sync,
{
    if fail_fast || workers <= 1 {
        let mut outcomes = Vec::new();
        for task in tasks {
            let outcome = serialize_file(task, preset, serialize_options);
            let failed = !outcome.success;
            progress();
            outcomes.push(outcome);
            if fail_fast && failed {
                break;
            }
        }
        return Ok(outcomes);
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;

    Ok(pool.install(|| {
        tasks
            .par_iter()
            .map(|task| {
                let outcome = serialize_file(task, preset, serialize_options);
                progress();
                outcome
            })
            .collect()
    }))
}

pub fn build_stats(
    input_dir: &Path,
    output_dir: &Path,
    input_count: usize,
    outcomes: &[FileOutcome],
    duration_seconds: f64,
) -> StatsPayload {
    let success_count = outcomes.iter().filter(|outcome| outcome.success).count();
    let error_count = outcomes.iter().filter(|outcome| !outcome.success).count();
    let mut error_types = BTreeMap::new();
    for outcome in outcomes.iter().filter(|outcome| !outcome.success) {
        *error_types
            .entry(
                outcome
                    .error_type
                    .clone()
                    .unwrap_or_else(|| "Unknown".to_owned()),
            )
            .or_insert(0) += 1;
    }
    let failed_examples = outcomes
        .iter()
        .filter(|outcome| !outcome.success)
        .take(20)
        .map(|outcome| FailedExample {
            file: outcome.input_path.display().to_string(),
            error_type: outcome.error_type.clone(),
            error: outcome.error_message.clone(),
        })
        .collect();

    StatsPayload {
        schema_version: "1.0",
        input_dir: input_dir.display().to_string(),
        output_dir: output_dir.display().to_string(),
        input_count,
        success_count,
        error_count,
        execution_failure_count: outcomes
            .iter()
            .filter(|outcome| outcome.execution_failure)
            .count(),
        pass_rate: if input_count == 0 {
            0.0
        } else {
            success_count as f64 / input_count as f64
        },
        duration_seconds,
        error_types,
        failed_examples,
    }
}

fn serialize_file(
    task: &BatchTask,
    preset: SerializePreset,
    serialize_options: SerializeOptions,
) -> FileOutcome {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        serialize_file_inner(task, preset, serialize_options)
    }));

    match result {
        Ok(Ok(())) => FileOutcome {
            input_path: task.input_path.clone(),
            success: true,
            execution_failure: false,
            error_type: None,
            error_message: None,
        },
        Ok(Err(error)) => FileOutcome {
            input_path: task.input_path.clone(),
            success: false,
            execution_failure: error.execution_failure,
            error_type: Some(error.kind),
            error_message: Some(error.message),
        },
        Err(payload) => FileOutcome {
            input_path: task.input_path.clone(),
            success: false,
            execution_failure: true,
            error_type: Some("Panic".to_owned()),
            error_message: Some(panic_message(payload)),
        },
    }
}

fn serialize_file_inner(
    task: &BatchTask,
    preset: SerializePreset,
    serialize_options: SerializeOptions,
) -> Result<(), FileError> {
    let input = fs::read_to_string(&task.input_path).map_err(|error| FileError {
        kind: if error.kind() == std::io::ErrorKind::InvalidData {
            "InvalidUtf8"
        } else {
            "IoError"
        }
        .to_owned(),
        execution_failure: error.kind() != std::io::ErrorKind::InvalidData,
        message: error.to_string(),
    })?;

    let document = KernDocument::parse(&input).map_err(|error| FileError {
        kind: "ParseError".to_owned(),
        execution_failure: false,
        message: error.to_string(),
    })?;
    let serialized = preset
        .try_serialize_document_with_options(&document, serialize_options)
        .map_err(|error| FileError {
            kind: "SerializeError".to_owned(),
            execution_failure: error.is_execution_failure(),
            message: error.to_string(),
        })?;

    if let Some(parent) = task
        .output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| FileError {
            kind: "IoError".to_owned(),
            execution_failure: true,
            message: error.to_string(),
        })?;
    }
    fs::write(&task.output_path, serialized).map_err(|error| FileError {
        kind: "IoError".to_owned(),
        execution_failure: true,
        message: error.to_string(),
    })?;

    Ok(())
}

#[derive(Debug)]
struct FileError {
    execution_failure: bool,
    kind: String,
    message: String,
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_owned();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "serialization panicked".to_owned()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use serde_json::Value;
    use tempfile::TempDir;

    use crate::{batch::input::DEFAULT_SAMPLE_TRANSCRIPTION_NAME, BarNumberStyle};

    use super::*;

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    fn serialize_dir_options(input_dir: PathBuf, output_dir: PathBuf) -> SerializeDirOptions {
        SerializeDirOptions {
            input_dir,
            output_dir,
            workers: 1,
            stats_json: None,
            fail_fast: false,
            preset: SerializePreset::TranscodaV2,
            serialize_options: SerializeOptions::default(),
            input_layout: BatchInputLayout::PreserveRelativePaths,
            sample_transcription_name: DEFAULT_SAMPLE_TRANSCRIPTION_NAME.to_owned(),
        }
    }

    #[test]
    fn serialize_dir_returns_typed_report_and_uses_progress_callback() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("in");
        let output = temp.path().join("out");
        write(&input.join("piece.krn"), "**kern\n4c\n*-\n");
        let progress_count = std::sync::atomic::AtomicUsize::new(0);

        let report = serialize_dir_with_progress(serialize_dir_options(input, output), || {
            progress_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        })
        .unwrap();

        assert_eq!(report.stats.input_count, 1);
        assert_eq!(report.stats.success_count, 1);
        assert_eq!(report.outcomes.len(), 1);
        assert!(report.outcomes[0].success);
        assert_eq!(progress_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn serialize_dir_writes_stats_json() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("in");
        let output = temp.path().join("out");
        let stats_path = temp.path().join("stats.json");
        write(&input.join("piece.krn"), "**kern\n4c\n*-\n");

        let mut options = serialize_dir_options(input, output);
        options.stats_json = Some(stats_path.clone());
        serialize_dir(options).unwrap();

        let stats: Value = serde_json::from_str(&fs::read_to_string(stats_path).unwrap()).unwrap();
        assert_eq!(stats["schema_version"], "1.0");
        assert_eq!(stats["input_count"], 1);
        assert_eq!(stats["success_count"], 1);
        assert_eq!(stats["error_count"], 0);
        assert_eq!(stats["pass_rate"], 1.0);
    }

    #[test]
    fn serialize_dir_fail_fast_reports_first_failure() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("in");
        let output = temp.path().join("out");
        write(&input.join("bad.krn"), "*unknown\n");
        write(&input.join("valid.krn"), "**kern\n4c\n*-\n");

        let mut options = serialize_dir_options(input, output);
        options.fail_fast = true;

        let report = serialize_dir(options).unwrap();

        assert_eq!(report.outcomes.len(), 1);
        assert_eq!(report.stats.error_count, 1);
    }

    #[test]
    fn serialize_dir_preserves_nested_relative_paths() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("in");
        let output = temp.path().join("out");
        write(
            &input.join("nested/piece.krn"),
            "**kern\n*M4/4\n4c\n=\n4d\n==\n*-\n",
        );

        let mut options = serialize_dir_options(input, output.clone());
        options.serialize_options = SerializeOptions {
            bar_number_style: BarNumberStyle::Sequential,
        };
        serialize_dir(options).unwrap();

        assert_eq!(
            fs::read_to_string(output.join("nested/piece.krn")).unwrap(),
            "*clefG2\n*k[]\n*M4/4\n=1\n4c\n=2\n4d\n==\n"
        );
    }

    #[test]
    fn serialize_dir_sample_dirs_writes_parent_named_flat_file() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("praig");
        let output = temp.path().join("out");
        write(
            &input.join("test_000000/transcription.krn"),
            "**kern\n4c\n*-\n",
        );
        write(&input.join("test_000000/sidecar.krn"), "**kern\n4d\n*-\n");

        let mut options = serialize_dir_options(input, output.clone());
        options.input_layout = BatchInputLayout::SampleDirs;
        serialize_dir(options).unwrap();

        assert_eq!(
            fs::read_to_string(output.join("test_000000.krn")).unwrap(),
            "*clefG2\n*k[]\n4c\n"
        );
        assert!(!output.join("test_000000/transcription.krn").exists());
        assert!(!output.join("test_000000/sidecar.krn").exists());
    }
}
