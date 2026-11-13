use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;

use crate::{
    batch::input::collect_krn_files,
    check::{check_text, CheckConfig, CheckPhase, CheckSeverity},
};

#[derive(Debug, Clone)]
pub struct QualityFilterOptions {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub workers: usize,
    pub fail_fast: bool,
    pub stats_json: Option<PathBuf>,
    pub config: CheckConfig,
}

#[derive(Debug)]
pub struct QualityFilterReport {
    pub stats: QualityFilterStatsPayload,
    pub outcomes: Vec<QualityFilterOutcome>,
}

#[derive(Debug, Clone)]
pub struct QualityFilterOutcome {
    pub execution_failure: bool,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub findings: Vec<QualityFilterFinding>,
}

impl QualityFilterOutcome {
    pub fn accepted(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|finding| finding.severity == "Error")
    }
}

#[derive(Debug, Clone)]
pub struct QualityFilterFinding {
    pub phase: &'static str,
    pub code: String,
    pub severity: &'static str,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct QualityFilterStatsPayload {
    pub execution_failure_count: usize,
    pub schema_version: &'static str,
    pub input_dir: String,
    pub output_dir: String,
    pub input_count: usize,
    pub accepted_count: usize,
    pub rejected_count: usize,
    pub finding_count: usize,
    pub pass_rate: f64,
    pub duration_seconds: f64,
    pub rejection_types: BTreeMap<String, usize>,
    pub files: Vec<QualityFilterFilePayload>,
}

#[derive(Debug, Serialize)]
pub struct QualityFilterFilePayload {
    pub execution_failure: bool,
    pub input_file: String,
    pub output_file: String,
    pub accepted: bool,
    pub findings: Vec<QualityFilterFindingPayload>,
}

#[derive(Debug, Serialize)]
pub struct QualityFilterFindingPayload {
    pub phase: String,
    pub code: String,
    pub severity: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct QualityFilterTask {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
}

pub fn quality_filter_dir(
    options: QualityFilterOptions,
) -> Result<QualityFilterReport, Box<dyn std::error::Error>> {
    quality_filter_dir_with_progress(options, || {})
}

pub fn quality_filter_dir_with_progress<F>(
    options: QualityFilterOptions,
    progress: F,
) -> Result<QualityFilterReport, Box<dyn std::error::Error>>
where
    F: Fn() + Sync,
{
    let start = Instant::now();
    let tasks = collect_quality_filter_tasks(&options.input_dir, &options.output_dir)?;
    if tasks.is_empty() {
        return Err(format!("No .krn files found in {}", options.input_dir.display()).into());
    }

    recreate_output_dir(&options.output_dir)?;
    let outcomes = process_quality_filter_tasks(
        &tasks,
        options.workers,
        options.fail_fast,
        &options.config,
        progress,
    )?;
    let stats = build_quality_filter_stats(
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

    Ok(QualityFilterReport { stats, outcomes })
}

pub fn collect_quality_filter_tasks(
    input_dir: &Path,
    output_dir: &Path,
) -> Result<Vec<QualityFilterTask>, io::Error> {
    let files = collect_krn_files(input_dir)?;
    Ok(files
        .into_iter()
        .map(|input_path| {
            let relative_path = input_path
                .strip_prefix(input_dir)
                .expect("collect_krn_files yielded a path below input_dir");
            QualityFilterTask {
                output_path: output_dir.join(relative_path),
                input_path,
            }
        })
        .collect())
}

fn recreate_output_dir(output_dir: &Path) -> Result<(), io::Error> {
    if output_dir.exists() {
        fs::remove_dir_all(output_dir)?;
    }
    fs::create_dir_all(output_dir)
}

pub fn process_quality_filter_tasks<F>(
    tasks: &[QualityFilterTask],
    workers: usize,
    fail_fast: bool,
    config: &CheckConfig,
    progress: F,
) -> Result<Vec<QualityFilterOutcome>, rayon::ThreadPoolBuildError>
where
    F: Fn() + Sync,
{
    if fail_fast || workers <= 1 {
        let mut outcomes = Vec::new();
        for task in tasks {
            let outcome = quality_filter_file(task, config);
            let rejected = !outcome.accepted();
            progress();
            outcomes.push(outcome);
            if fail_fast && rejected {
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
                let outcome = quality_filter_file(task, config);
                progress();
                outcome
            })
            .collect()
    }))
}

fn quality_filter_file(task: &QualityFilterTask, config: &CheckConfig) -> QualityFilterOutcome {
    let result = panic::catch_unwind(AssertUnwindSafe(|| quality_filter_file_inner(task, config)));
    match result {
        Ok(outcome) => outcome,
        Err(payload) => QualityFilterOutcome {
            execution_failure: true,
            input_path: task.input_path.clone(),
            output_path: task.output_path.clone(),
            findings: vec![QualityFilterFinding {
                phase: "Panic",
                code: "Panic".to_owned(),
                severity: "Error",
                message: panic_message(payload),
            }],
        },
    }
}

fn quality_filter_file_inner(
    task: &QualityFilterTask,
    config: &CheckConfig,
) -> QualityFilterOutcome {
    let input = match fs::read_to_string(&task.input_path) {
        Ok(input) => input,
        Err(error) => {
            return QualityFilterOutcome {
                execution_failure: error.kind() != io::ErrorKind::InvalidData,
                input_path: task.input_path.clone(),
                output_path: task.output_path.clone(),
                findings: vec![QualityFilterFinding {
                    phase: if error.kind() == io::ErrorKind::InvalidData {
                        "Parse"
                    } else {
                        "Io"
                    },
                    code: if error.kind() == io::ErrorKind::InvalidData {
                        "parse.invalid-utf8"
                    } else {
                        "IoError"
                    }
                    .to_owned(),
                    severity: "Error",
                    message: error.to_string(),
                }],
            };
        }
    };

    let report = check_text(&input, config);
    let execution_failure = report.has_execution_failure();
    let findings = report
        .diagnostics
        .into_iter()
        .map(|finding| QualityFilterFinding {
            phase: match finding.phase {
                CheckPhase::Parse => "Parse",
                CheckPhase::Validation => "Validation",
            },
            code: finding.code,
            severity: match finding.severity {
                CheckSeverity::Error => "Error",
                CheckSeverity::Warning => "Warning",
                CheckSeverity::Info => "Info",
            },
            message: finding.message,
        })
        .collect();
    if !report.success {
        return QualityFilterOutcome {
            execution_failure,
            input_path: task.input_path.clone(),
            output_path: task.output_path.clone(),
            findings,
        };
    }

    if let Some(parent) = task.output_path.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            return QualityFilterOutcome {
                execution_failure: true,
                input_path: task.input_path.clone(),
                output_path: task.output_path.clone(),
                findings: vec![QualityFilterFinding {
                    phase: "Io",
                    code: "IoError".to_owned(),
                    severity: "Error",
                    message: error.to_string(),
                }],
            };
        }
    }

    match fs::copy(&task.input_path, &task.output_path) {
        Ok(_) => QualityFilterOutcome {
            execution_failure: false,
            input_path: task.input_path.clone(),
            output_path: task.output_path.clone(),
            findings,
        },
        Err(error) => QualityFilterOutcome {
            execution_failure: true,
            input_path: task.input_path.clone(),
            output_path: task.output_path.clone(),
            findings: vec![QualityFilterFinding {
                phase: "Io",
                code: "IoError".to_owned(),
                severity: "Error",
                message: error.to_string(),
            }],
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
    "quality filter panicked".to_owned()
}

pub fn build_quality_filter_stats(
    input_dir: &Path,
    output_dir: &Path,
    input_count: usize,
    outcomes: &[QualityFilterOutcome],
    duration_seconds: f64,
) -> QualityFilterStatsPayload {
    let accepted_count = outcomes.iter().filter(|outcome| outcome.accepted()).count();
    let rejected_count = outcomes
        .iter()
        .filter(|outcome| !outcome.accepted())
        .count();
    let finding_count = outcomes
        .iter()
        .map(|outcome| outcome.findings.len())
        .sum::<usize>();
    let mut rejection_types = BTreeMap::new();
    for finding in outcomes
        .iter()
        .flat_map(|outcome| outcome.findings.iter())
        .filter(|finding| finding.severity == "Error")
    {
        let key = format!("{}:{}", finding.phase, finding.code);
        *rejection_types.entry(key).or_insert(0) += 1;
    }

    QualityFilterStatsPayload {
        execution_failure_count: outcomes
            .iter()
            .filter(|outcome| outcome.execution_failure)
            .count(),
        schema_version: "1.0",
        input_dir: input_dir.display().to_string(),
        output_dir: output_dir.display().to_string(),
        input_count,
        accepted_count,
        rejected_count,
        finding_count,
        pass_rate: if input_count == 0 {
            0.0
        } else {
            accepted_count as f64 / input_count as f64
        },
        duration_seconds,
        rejection_types,
        files: outcomes
            .iter()
            .map(|outcome| QualityFilterFilePayload {
                execution_failure: outcome.execution_failure,
                input_file: outcome.input_path.display().to_string(),
                output_file: outcome.output_path.display().to_string(),
                accepted: outcome.accepted(),
                findings: outcome
                    .findings
                    .iter()
                    .map(|finding| QualityFilterFindingPayload {
                        phase: finding.phase.to_owned(),
                        code: finding.code.clone(),
                        severity: finding.severity.to_owned(),
                        message: finding.message.clone(),
                    })
                    .collect(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;
    use crate::ValidationConfig;

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    #[test]
    fn input_rejection_and_output_io_failure_keep_distinct_categories() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("piece.krn");
        let output_parent = temp.path().join("output-is-a-file");
        write(&input_path, "**kern\n4c\n*-\n");
        write(&output_parent, "occupied");
        let task = QualityFilterTask {
            input_path: input_path.clone(),
            output_path: output_parent.join("piece.krn"),
        };
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![crate::ValidationPass::State]),
            ..Default::default()
        };
        let failed = quality_filter_file(&task, &config);
        assert!(!failed.accepted());
        assert!(failed.execution_failure);
        assert_eq!(failed.findings[0].phase, "Io");
        write(&input_path, "**kern\ninvalid-note\n*-\n");
        let rejected = quality_filter_file(&task, &config);
        assert!(!rejected.accepted());
        assert!(!rejected.execution_failure);
        assert_eq!(rejected.findings[0].phase, "Parse");
        let stats =
            build_quality_filter_stats(temp.path(), temp.path(), 2, &[failed, rejected], 0.0);
        assert_eq!(stats.rejected_count, 2);
        assert_eq!(stats.execution_failure_count, 1);
    }

    #[test]
    #[cfg(not(feature = "engraving"))]
    fn missing_engine_is_counted_as_execution_failure() {
        let temp = TempDir::new().unwrap();
        let task = QualityFilterTask {
            input_path: temp.path().join("input.krn"),
            output_path: temp.path().join("output.krn"),
        };
        write(&task.input_path, "**kern\n4c\n*-\n");
        let config = CheckConfig {
            validation: ValidationConfig::with_passes(vec![
                crate::ValidationPass::MediumSinglePage,
            ]),
            ..Default::default()
        };
        let outcomes = process_quality_filter_tasks(&[task], 1, false, &config, || {}).unwrap();
        assert!(outcomes[0].execution_failure);
        assert!(!outcomes[0].accepted());
    }

    #[test]
    fn quality_filter_service_returns_report_and_uses_progress_callback() {
        let temp = TempDir::new().unwrap();
        let input_dir = temp.path().join("in");
        let output_dir = temp.path().join("out");
        write(
            &input_dir.join("piece.krn"),
            "**kern\n*clefG2\n*k[]\n4c\n*-\n",
        );
        let progress_count = std::sync::atomic::AtomicUsize::new(0);

        let report = quality_filter_dir_with_progress(
            QualityFilterOptions {
                input_dir,
                output_dir: output_dir.clone(),
                workers: 1,
                fail_fast: false,
                stats_json: None,
                config: CheckConfig {
                    validation: ValidationConfig::with_passes(vec![crate::ValidationPass::State]),
                    ..Default::default()
                },
            },
            || {
                progress_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            },
        )
        .unwrap();

        assert_eq!(report.stats.input_count, 1);
        assert_eq!(report.stats.accepted_count, 1);
        assert_eq!(report.outcomes.len(), 1);
        assert!(output_dir.join("piece.krn").exists());
        assert_eq!(progress_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    }
}
