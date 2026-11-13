use std::collections::BTreeMap;
use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;

use crate::{
    batch::input::{collect_batch_tasks, BatchInputLayout, BatchTask},
    document::Framing,
    normalization::dataset::normalized_for_dataset_normalization,
    score::Score,
    serialize::{DialectError, SerializePreset, TandemPolicy},
    KernDocument,
};

#[derive(Debug, Clone)]
pub struct NormalizeDirOptions {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub workers: usize,
    pub stats_json: Option<PathBuf>,
    pub fail_fast: bool,
    pub output_preset: SerializePreset,
    pub input_layout: BatchInputLayout,
    pub sample_transcription_name: String,
}

#[derive(Debug)]
pub struct NormalizeDirReport {
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
    pub normalization_pass_stats: NormalizationPassStats,
}

#[derive(Debug, Serialize)]
pub struct FailedExample {
    pub file: String,
    pub error_type: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct NormalizationPassStats {
    pub canonicalize_note_order: CanonicalizeNoteOrderStats,
}

#[derive(Debug, Serialize)]
pub struct CanonicalizeNoteOrderStats {
    pub unknown_char_drops_total: usize,
    pub unknown_char_drop_counts: BTreeMap<String, usize>,
}

pub fn normalize_dir(
    options: NormalizeDirOptions,
) -> Result<NormalizeDirReport, Box<dyn std::error::Error>> {
    normalize_dir_with_progress(options, || {})
}

pub fn normalize_dir_with_progress<F>(
    options: NormalizeDirOptions,
    progress: F,
) -> Result<NormalizeDirReport, Box<dyn std::error::Error>>
where
    F: Fn() + Sync,
{
    let start = Instant::now();
    let tasks = collect_batch_tasks(
        &options.input_dir,
        &options.output_dir,
        options.input_layout,
        &options.sample_transcription_name,
        true,
    )?;
    if tasks.is_empty() {
        return Err(format!("No .krn files found in {}", options.input_dir.display()).into());
    }

    let outcomes = process_tasks(
        &tasks,
        options.workers,
        options.fail_fast,
        options.output_preset,
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

    Ok(NormalizeDirReport { stats, outcomes })
}

pub fn process_tasks<F>(
    tasks: &[BatchTask],
    workers: usize,
    fail_fast: bool,
    output_preset: SerializePreset,
    progress: F,
) -> Result<Vec<FileOutcome>, rayon::ThreadPoolBuildError>
where
    F: Fn() + Sync,
{
    if fail_fast || workers <= 1 {
        let mut outcomes = Vec::new();
        for task in tasks {
            let outcome = normalize_file(task, output_preset);
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
                let outcome = normalize_file(task, output_preset);
                progress();
                outcome
            })
            .collect()
    }))
}

fn normalize_file(task: &BatchTask, output_preset: SerializePreset) -> FileOutcome {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        normalize_file_inner(task, output_preset)
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

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_owned();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "normalization panicked".to_owned()
}

fn normalize_file_inner(task: &BatchTask, output_preset: SerializePreset) -> Result<(), FileError> {
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

    let normalized = serialize_normalized(&document, output_preset)?;
    validate_normalized_output(&normalized, output_preset).map_err(|message| FileError {
        kind: "FinalTopologyError".to_owned(),
        execution_failure: true,
        message,
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
    fs::write(&task.output_path, normalized).map_err(|error| FileError {
        kind: "IoError".to_owned(),
        execution_failure: true,
        message: error.to_string(),
    })?;

    Ok(())
}

fn serialize_normalized(
    document: &KernDocument,
    output_preset: SerializePreset,
) -> Result<String, FileError> {
    // Dataset normalization must go through the document serializers, which wire
    // the OMR-facing structural passes. Stage-4 source cleanups are materialized
    // here, independent of the eventual output dialect.
    let serialized = {
        // OMR normalization leaves out what its policy does not carry; a
        // slot it cannot express refuses the file instead of vanishing.
        let checked = Score::resolve(document, Framing::Excerpt)
            .map_err(DialectError::from)
            .and_then(|score| TandemPolicy::Omr.check(&score));
        if let Err(error) = checked {
            return Err(FileError {
                kind: "SerializeError".to_owned(),
                execution_failure: false,
                message: error.to_string(),
            });
        }
        let normalized = normalized_for_dataset_normalization(document);
        output_preset.try_serialize_document(&normalized)
    };

    serialized.map_err(|error| FileError {
        kind: "SerializeError".to_owned(),
        execution_failure: error.is_execution_failure(),
        message: error.to_string(),
    })
}

fn validate_normalized_output(
    normalized: &str,
    output_preset: SerializePreset,
) -> Result<(), String> {
    if output_preset != SerializePreset::HumdrumKern {
        return Ok(());
    }

    let lines = normalized
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with("!"))
        .collect::<Vec<_>>();
    let Some(first) = lines.first() else {
        return Ok(());
    };
    let Some(last) = lines.last() else {
        return Ok(());
    };

    let initial_width = first.split('\t').count();
    let final_tokens = last.split('\t').collect::<Vec<_>>();
    if !final_tokens.iter().all(|token| *token == "*-") {
        return Ok(());
    }
    if final_tokens.len() == initial_width {
        return Ok(());
    }

    Err(format!(
        "normalized Humdrum output final terminal width {} does not match initial width {}",
        final_tokens.len(),
        initial_width
    ))
}

#[derive(Debug)]
struct FileError {
    execution_failure: bool,
    kind: String,
    message: String,
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
        if let Some(error_type) = &outcome.error_type {
            *error_types.entry(error_type.clone()).or_insert(0) += 1;
        }
    }

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
        failed_examples: outcomes
            .iter()
            .filter(|outcome| !outcome.success)
            .take(20)
            .map(|outcome| FailedExample {
                file: outcome.input_path.display().to_string(),
                error_type: outcome.error_type.clone(),
                error: outcome.error_message.clone(),
            })
            .collect(),
        normalization_pass_stats: NormalizationPassStats {
            canonicalize_note_order: CanonicalizeNoteOrderStats {
                unknown_char_drops_total: 0,
                unknown_char_drop_counts: BTreeMap::new(),
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use serde_json::Value;
    use tempfile::TempDir;

    use crate::batch::input::DEFAULT_SAMPLE_TRANSCRIPTION_NAME;

    use super::*;

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    fn valid_kern() -> &'static str {
        "**kern\n*clefG2\n*k[]\n*M4/4\n4c\n*-\n"
    }

    fn invalid_measure_rhythm_kern() -> &'static str {
        "**kern\n*M4/4\n=\n1c\n1d\n1e\n1f\n4g\n*-\n"
    }

    fn normalize_dir_options(input_dir: PathBuf, output_dir: PathBuf) -> NormalizeDirOptions {
        NormalizeDirOptions {
            input_dir,
            output_dir,
            workers: 1,
            stats_json: None,
            fail_fast: false,
            output_preset: SerializePreset::HumdrumKern,
            input_layout: BatchInputLayout::PreserveRelativePaths,
            sample_transcription_name: DEFAULT_SAMPLE_TRANSCRIPTION_NAME.to_owned(),
        }
    }

    #[test]
    fn normalize_dir_writes_valid_normalized_file() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(
            &input_path.join("piece.krn"),
            "**kern\n*M4/4\n4c 4e 4g\n*-\n",
        );

        normalize_dir(normalize_dir_options(input_path, output_dir.clone())).unwrap();

        let normalized = fs::read_to_string(output_dir.join("piece.krn")).unwrap();
        assert_eq!(
            normalized,
            "**kern\n*clefG2\n*k[]\n*M4/4\n4c 4e 4g\n*-\n!!!RDF**kern: > = above\n!!!RDF**kern: < = below\n!!!RDF**kern: N = linked\n!!!RDF**kern: Z = editorial accidental, parentheses\n!!!RDF**kern: j = editorial accidental, brackets\n"
        );
    }

    #[test]
    fn normalize_dir_can_write_humdrum_kern_preset() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(
            &input_path.join("piece.krn"),
            "**kern\n*M4/4\n8cL\n8dJ\n*-\n",
        );

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.output_preset = SerializePreset::HumdrumKern;
        normalize_dir(options).unwrap();

        let normalized = fs::read_to_string(output_dir.join("piece.krn")).unwrap();
        assert_eq!(
            normalized,
            "**kern\n*clefG2\n*k[]\n*M4/4\n8cL\n8dJ\n*-\n!!!RDF**kern: > = above\n!!!RDF**kern: < = below\n!!!RDF**kern: N = linked\n!!!RDF**kern: Z = editorial accidental, parentheses\n!!!RDF**kern: j = editorial accidental, brackets\n"
        );
    }

    #[test]
    fn normalize_dir_materializes_invisible_rest_padding_cleanup() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(
            &input_path.join("piece.krn"),
            &[
                "**kern\t**dynam\t**kern\t**dynam",
                "*M2/4\t*\t*M2/4\t*",
                "8c\t.\t8e\t.",
                "32ryy\t.\t32ryy\t.",
                "32ryy\t.\t32ryy\t.",
                "32ryy\t.\t32ryy\t.",
                "32ryy\t.\t32ryy\t.",
                "32ryy\t.\t32ryy\t.",
                "32ryy\t.\t32ryy\t.",
                "32ryy\t.\t32ryy\t.",
                "32ryy\t.\t32ryy\t.",
                "32ryy\t.\t32ryy\t.",
                "32ryy\t.\t32ryy\t.",
                "32ryy\t.\t32ryy\t.",
                "32ryy\t.\t32ryy\t.",
                "=\t=\t=\t=",
                "4d\t.\t4f\t.",
                "*-\t*-\t*-\t*-",
            ]
            .join("\n"),
        );

        normalize_dir(normalize_dir_options(input_path, output_dir.clone())).unwrap();

        let normalized = fs::read_to_string(output_dir.join("piece.krn")).unwrap();
        assert!(!normalized.contains("32ryy"));
        assert!(normalized.contains("8c\t.\t8e\t.\n=\t=\t=\t=\n4d\t.\t4f\t."));
    }

    #[test]
    fn normalize_dir_materializes_consecutive_invisible_rest_compaction() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(
            &input_path.join("piece.krn"),
            &[
                "**kern\t**kern",
                "*M4/4\t*M4/4",
                "8c\t8ryy",
                "8d\t8ryy",
                "8e\t8ryy",
                "8f\t8ryy",
                "*-\t*-",
            ]
            .join("\n"),
        );

        normalize_dir(normalize_dir_options(input_path, output_dir.clone())).unwrap();

        let normalized = fs::read_to_string(output_dir.join("piece.krn")).unwrap();
        assert!(normalized.contains("8c\t2ryy\n8d\t.\n8e\t.\n8f\t."));
    }

    #[test]
    fn normalize_dir_humdrum_kern_restores_final_staff_topology() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(
            &input_path.join("piece.krn"),
            &[
                "**kern\t**kern",
                "*clefG2\t*clefF4",
                "*k[]\t*k[]",
                "*^\t*",
                "4c\t4e\t4g",
                "*\t*\t*^",
                "4d\t4f\t4a\t4b",
                "=\t=\t=\t=",
                "*v\t*v\t*\t*",
                "*-\t*-\t*-",
            ]
            .join("\n"),
        );

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.output_preset = SerializePreset::HumdrumKern;
        normalize_dir(options).unwrap();

        let normalized = fs::read_to_string(output_dir.join("piece.krn")).unwrap();
        let tail = normalized
            .lines()
            .rev()
            .skip(crate::serialize::render::DEFAULT_RDF_RECORDS.len())
            .take(4)
            .collect::<Vec<_>>();
        assert_eq!(tail, vec!["*-\t*-", "=\t=", "*\t*v\t*v", "*v\t*v\t*\t*"]);
    }

    #[test]
    fn normalize_dir_keeps_cue_and_refuses_a_rhythm_scale() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        let stats_path = temp.path().join("stats.json");
        write(
            &input_path.join("cue.krn"),
            "**kern\n*clefG2\n*M2/4\n*cue\n4c\n4d\n*Xcue\n=\n*-\n",
        );
        write(
            &input_path.join("scaled.krn"),
            "**kern\n*clefG2\n*M2/4\n*rscale:2\n4c\n*rscale:1\n4d\n=\n*-\n",
        );
        write(
            &input_path.join("unit.krn"),
            "**kern\n*clefG2\n*M2/4\n*rscale:1\n4c\n4d\n=\n*-\n",
        );

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.stats_json = Some(stats_path.clone());
        normalize_dir(options).unwrap();

        let cue = fs::read_to_string(output_dir.join("cue.krn")).unwrap();
        assert!(cue.contains("*cue\n") && cue.contains("*Xcue\n"), "{cue}");
        assert!(!output_dir.join("scaled.krn").exists());
        let unit = fs::read_to_string(output_dir.join("unit.krn")).unwrap();
        assert!(!unit.contains("*rscale"), "{unit}");
        let stats: Value = serde_json::from_str(&fs::read_to_string(stats_path).unwrap()).unwrap();
        assert_eq!(stats["error_types"]["SerializeError"], 1);
        assert!(stats["failed_examples"][0]["error"]
            .as_str()
            .unwrap()
            .contains("*rscale:2"));
    }

    #[test]
    fn normalize_dir_humdrum_kern_rejects_a_merge_across_staves() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        let stats_path = temp.path().join("stats.json");
        write(
            &input_path.join("piece.krn"),
            "**kern\t**kern\n4c\t4e\n*v\t*v\n=\n*-\n",
        );

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.stats_json = Some(stats_path.clone());
        normalize_dir(options).unwrap();

        assert!(!output_dir.join("piece.krn").exists());
        let stats: Value = serde_json::from_str(&fs::read_to_string(stats_path).unwrap()).unwrap();
        assert_eq!(stats["error_types"]["SerializeError"], 1);
        assert!(stats["failed_examples"][0]["error"]
            .as_str()
            .unwrap()
            .contains("cannot merge spines from different original spines"));
    }

    #[test]
    fn normalize_dir_can_write_transcoda_v2_preset() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(
            &input_path.join("piece.krn"),
            "**kern\n*M4/4\n(4c#X'[ 4e])\n*-\n",
        );

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.output_preset = SerializePreset::TranscodaV2;
        normalize_dir(options).unwrap();

        let normalized = fs::read_to_string(output_dir.join("piece.krn")).unwrap();
        assert_eq!(normalized, "*clefG2\n*k[]\n*M4/4\n4c# e\n");
    }

    #[test]
    fn normalize_dir_training_target_writes_empty_document() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(
            &input_path.join("empty.krn"),
            "!!!COM: Example\n!! only comments\n",
        );

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.output_preset = SerializePreset::TranscodaV2;
        normalize_dir(options).unwrap();

        let normalized = fs::read_to_string(output_dir.join("empty.krn")).unwrap();
        assert_eq!(normalized, "");
    }

    #[test]
    fn normalize_dir_training_target_removes_invisible_rest_measures() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(
            &input_path.join("piece.krn"),
            "**kern\n*clefG2\n*k[]\n=\n4c\n=\n4ryy\n=\n4d\n=\n*-\n",
        );

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.output_preset = SerializePreset::TranscodaV2;
        normalize_dir(options).unwrap();

        let normalized = fs::read_to_string(output_dir.join("piece.krn")).unwrap();
        assert_eq!(normalized, "*clefG2\n*k[]\n4c\n=\n4d\n=\n");
    }

    #[test]
    fn normalize_dir_preserves_nested_relative_paths() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(&input_path.join("nested/piece.krn"), valid_kern());

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.workers = 2;
        normalize_dir(options).unwrap();

        assert!(output_dir.join("nested/piece.krn").exists());
    }

    #[test]
    fn normalize_dir_sample_dirs_writes_parent_named_flat_file() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("praig");
        let output_dir = temp.path().join("out");
        write(
            &input_path.join("test_000000/transcription.krn"),
            valid_kern(),
        );
        write(&input_path.join("test_000000/sidecar.krn"), valid_kern());

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.input_layout = BatchInputLayout::SampleDirs;
        normalize_dir(options).unwrap();

        assert!(output_dir.join("test_000000.krn").exists());
        assert!(!output_dir.join("test_000000/transcription.krn").exists());
        assert!(!output_dir.join("test_000000/sidecar.krn").exists());
    }

    #[test]
    fn normalize_dir_sample_dirs_rejects_output_collisions_before_writing() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("praig");
        let output_dir = temp.path().join("out");
        write(
            &input_path.join("split_a/test_000000/transcription.krn"),
            valid_kern(),
        );
        write(
            &input_path.join("split_b/test_000000/transcription.krn"),
            valid_kern(),
        );

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.input_layout = BatchInputLayout::SampleDirs;
        let error = normalize_dir(options).unwrap_err();

        assert!(error.to_string().contains("multiple inputs map to output"));
        assert!(!output_dir.exists());
    }

    #[test]
    fn normalize_dir_sample_dirs_rejects_single_file_input() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("transcription.krn");
        let output_path = temp.path().join("out.krn");
        write(&input_path, valid_kern());

        let mut options = normalize_dir_options(input_path, output_path);
        options.input_layout = BatchInputLayout::SampleDirs;
        let error = normalize_dir(options).unwrap_err();

        assert!(error
            .to_string()
            .contains("--input-layout sample-dirs requires a directory"));
    }

    #[test]
    fn normalize_dir_writes_single_file_to_output_path() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("piece.krn");
        let output_path = temp.path().join("normalized.krn");
        write(&input_path, "**kern\n*M4/4\n4c 4e 4g\n*-\n");

        normalize_dir(normalize_dir_options(input_path, output_path.clone())).unwrap();

        let normalized = fs::read_to_string(output_path).unwrap();
        assert_eq!(
            normalized,
            "**kern\n*clefG2\n*k[]\n*M4/4\n4c 4e 4g\n*-\n!!!RDF**kern: > = above\n!!!RDF**kern: < = below\n!!!RDF**kern: N = linked\n!!!RDF**kern: Z = editorial accidental, parentheses\n!!!RDF**kern: j = editorial accidental, brackets\n"
        );
    }

    #[test]
    fn normalize_dir_returns_typed_report_and_uses_progress_callback() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(&input_path.join("piece.krn"), valid_kern());
        let progress_count = std::sync::atomic::AtomicUsize::new(0);

        let report =
            normalize_dir_with_progress(normalize_dir_options(input_path, output_dir), || {
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
    fn normalize_dir_omits_parse_errors_and_counts_them() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        let stats_path = temp.path().join("stats.json");
        write(&input_path.join("bad.krn"), "*unknown\n");

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.stats_json = Some(stats_path.clone());
        normalize_dir(options).unwrap();

        assert!(!output_dir.join("bad.krn").exists());
        let stats: Value = serde_json::from_str(&fs::read_to_string(stats_path).unwrap()).unwrap();
        assert_eq!(stats["input_count"], 1);
        assert_eq!(stats["success_count"], 0);
        assert_eq!(stats["error_count"], 1);
        assert_eq!(stats["error_types"]["ParseError"], 1);
    }

    #[test]
    fn normalize_dir_keeps_rhythm_invalid_parseable_files() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        let stats_path = temp.path().join("stats.json");
        write(&input_path.join("bad.krn"), invalid_measure_rhythm_kern());
        write(&input_path.join("good.krn"), valid_kern());

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.stats_json = Some(stats_path.clone());
        normalize_dir(options).unwrap();

        assert!(output_dir.join("bad.krn").exists());
        assert!(output_dir.join("good.krn").exists());
        let stats: Value = serde_json::from_str(&fs::read_to_string(stats_path).unwrap()).unwrap();
        assert_eq!(stats["input_count"], 2);
        assert_eq!(stats["success_count"], 2);
        assert_eq!(stats["error_count"], 0);
        assert!(stats["error_types"].as_object().unwrap().is_empty());
        assert!(stats["failed_examples"].as_array().unwrap().is_empty());
    }

    #[test]
    fn normalize_dir_fail_fast_reports_first_failure() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(&input_path.join("bad.krn"), "*unknown\n");
        write(&input_path.join("good.krn"), valid_kern());

        let mut options = normalize_dir_options(input_path, output_dir);
        options.workers = 2;
        options.fail_fast = true;
        let report = normalize_dir(options).unwrap();

        assert_eq!(report.outcomes.len(), 1);
        assert_eq!(report.stats.error_count, 1);
    }

    #[test]
    fn normalize_dir_fail_fast_ignores_rhythm_invalid_parseable_files() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(&input_path.join("bad.krn"), invalid_measure_rhythm_kern());
        write(&input_path.join("good.krn"), valid_kern());

        let mut options = normalize_dir_options(input_path, output_dir.clone());
        options.workers = 2;
        options.fail_fast = true;
        normalize_dir(options).unwrap();

        assert!(output_dir.join("bad.krn").exists());
        assert!(output_dir.join("good.krn").exists());
    }

    #[test]
    fn normalize_dir_stats_keeps_python_compatibility_shape() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        let output_dir = temp.path().join("out");
        let stats_path = temp.path().join("stats.json");
        write(&input_path.join("piece.krn"), valid_kern());

        let mut options = normalize_dir_options(input_path, output_dir);
        options.stats_json = Some(stats_path.clone());
        normalize_dir(options).unwrap();

        let stats: Value = serde_json::from_str(&fs::read_to_string(stats_path).unwrap()).unwrap();
        assert_eq!(stats["schema_version"], "1.0");
        assert_eq!(stats["input_count"], 1);
        assert_eq!(stats["pass_rate"], 1.0);
        assert_eq!(
            stats["normalization_pass_stats"]["canonicalize_note_order"]
                ["unknown_char_drops_total"],
            0
        );
        assert!(stats["normalization_pass_stats"]["canonicalize_note_order"]
            ["unknown_char_drop_counts"]
            .as_object()
            .unwrap()
            .is_empty());
    }
}
