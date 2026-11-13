use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

use serde::Serialize;

use super::{
    generate, manifest, ConcatenateDirOptions, ConcatenateStats, ConcatenationError,
    ConcatenationOutputManifestRecord, ConcatenationProgress, ConcatenationSourceUsageRecord,
    NoConcatenationProgress, DEFAULT_CONCATENATION_MANIFEST_FILENAME,
};
use crate::dataset_manifest::{update_dataset_manifest, DatasetComponentUpdate};

pub fn concatenate_dir(
    options: &ConcatenateDirOptions,
) -> Result<ConcatenateStats, ConcatenationError> {
    concatenate_dir_with_progress(options, &NoConcatenationProgress)
}

pub fn concatenate_dir_streaming_manifest(
    options: &ConcatenateDirOptions,
) -> Result<ConcatenateStats, ConcatenationError> {
    concatenate_dir_with_progress_streaming_manifest(options, &NoConcatenationProgress)
}

pub fn concatenate_dir_with_progress(
    options: &ConcatenateDirOptions,
    progress: &impl ConcatenationProgress,
) -> Result<ConcatenateStats, ConcatenationError> {
    let manifest = prepare_concatenation(options, progress)?;
    let mut manifest_sink = CollectingOutputManifestSink::default();
    let mut stats = run_concatenation_generation(options, progress, manifest, &mut manifest_sink)?;
    stats.output_manifest = manifest_sink.into_records();

    let manifest_path = resolved_manifest_path(options);
    write_output_manifest_jsonl(&manifest_path, &stats.output_manifest)?;
    write_usage_jsonl_if_requested(options, &stats.source_usage)?;
    update_concatenation_manifest_component(options, &manifest_path, &stats)?;

    Ok(stats)
}

pub fn concatenate_dir_with_progress_streaming_manifest(
    options: &ConcatenateDirOptions,
    progress: &impl ConcatenationProgress,
) -> Result<ConcatenateStats, ConcatenationError> {
    let manifest = prepare_concatenation(options, progress)?;
    let manifest_path = resolved_manifest_path(options);
    let mut manifest_sink = JsonlOutputManifestSink::create(&manifest_path)?;
    let stats = run_concatenation_generation(options, progress, manifest, &mut manifest_sink)?;
    write_usage_jsonl_if_requested(options, &stats.source_usage)?;
    update_concatenation_manifest_component(options, &manifest_path, &stats)?;
    Ok(stats)
}

fn prepare_concatenation(
    options: &ConcatenateDirOptions,
    progress: &impl ConcatenationProgress,
) -> Result<manifest::SourceManifest, ConcatenationError> {
    validate_options(options)?;
    if !options.input_dir.is_dir() {
        return Err(ConcatenationError::MissingInputDir(
            options.input_dir.clone(),
        ));
    }

    let manifest = manifest::build_manifest(&options.input_dir, progress)?;
    if manifest.input_count == 0 {
        return Err(ConcatenationError::EmptyInputDir(options.input_dir.clone()));
    }
    if manifest.origins.is_empty() {
        return Err(ConcatenationError::NoParseableInput(
            options.input_dir.clone(),
        ));
    }

    recreate_output_dir(&options.output_dir)?;
    Ok(manifest)
}

fn run_concatenation_generation(
    options: &ConcatenateDirOptions,
    progress: &impl ConcatenationProgress,
    manifest: manifest::SourceManifest,
    manifest_sink: &mut impl OutputManifestSink,
) -> Result<ConcatenateStats, ConcatenationError> {
    progress.generation_started(manifest.origins.len(), options.target_count);

    let output_metrics = generate::generate_outputs(
        &manifest.origins,
        manifest.units,
        manifest.origin_rejections,
        manifest.split_chunk_count,
        &options.output_dir,
        options.target_count,
        options.min_line_count,
        options.bucket_width,
        options.max_origin_uses,
        options.max_fragments_per_output,
        progress,
        manifest_sink,
    )?;
    progress.generation_finished(output_metrics.output_count);

    Ok(ConcatenateStats {
        input_count: manifest.input_count,
        skipped_parse_count: manifest.skipped_parse_count,
        output_count: output_metrics.output_count,
        truncation: output_metrics.truncation,
        usage: output_metrics.usage,
        output_manifest: output_metrics.output_manifest,
        source_usage: output_metrics.source_usage,
        balancing: output_metrics.balancing,
    })
}

fn write_usage_jsonl_if_requested(
    options: &ConcatenateDirOptions,
    records: &[ConcatenationSourceUsageRecord],
) -> Result<(), ConcatenationError> {
    if let Some(path) = &options.usage_jsonl {
        write_usage_jsonl(path, records)?;
    }
    Ok(())
}

fn resolved_manifest_path(options: &ConcatenateDirOptions) -> PathBuf {
    options.manifest_jsonl.clone().unwrap_or_else(|| {
        options
            .output_dir
            .join(DEFAULT_CONCATENATION_MANIFEST_FILENAME)
    })
}

fn update_concatenation_manifest_component(
    options: &ConcatenateDirOptions,
    manifest_path: &Path,
    stats: &ConcatenateStats,
) -> Result<(), ConcatenationError> {
    if options
        .output_dir
        .file_name()
        .and_then(|name| name.to_str())
        != Some("6_concatenated")
    {
        return Ok(());
    }
    let Some(dataset_root) = options.output_dir.parent() else {
        return Ok(());
    };
    update_dataset_manifest(
        dataset_root,
        DatasetComponentUpdate {
            name: "concatenation_provenance".to_owned(),
            producer: "kern_parser concatenate-dir".to_owned(),
            paths: vec![manifest_path.to_owned()],
            summary: BTreeMap::from([
                ("input_count".to_owned(), stats.input_count),
                ("skipped_parse_count".to_owned(), stats.skipped_parse_count),
                ("output_count".to_owned(), stats.output_count),
            ]),
        },
    )
    .map_err(|source| ConcatenationError::Io {
        path: dataset_root.join(crate::dataset_manifest::DATASET_MANIFEST_FILE_NAME),
        source,
    })?;
    Ok(())
}

pub(crate) trait OutputManifestSink {
    fn record_manifest(
        &mut self,
        record: ConcatenationOutputManifestRecord,
    ) -> Result<(), ConcatenationError>;
}

#[derive(Debug, Default)]
struct CollectingOutputManifestSink {
    records: Vec<ConcatenationOutputManifestRecord>,
}

impl CollectingOutputManifestSink {
    fn into_records(self) -> Vec<ConcatenationOutputManifestRecord> {
        self.records
    }
}

impl OutputManifestSink for CollectingOutputManifestSink {
    fn record_manifest(
        &mut self,
        record: ConcatenationOutputManifestRecord,
    ) -> Result<(), ConcatenationError> {
        self.records.push(record);
        Ok(())
    }
}

struct JsonlOutputManifestSink {
    path: PathBuf,
    file: File,
}

impl JsonlOutputManifestSink {
    fn create(path: &Path) -> Result<Self, ConcatenationError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| ConcatenationError::Io {
                path: parent.to_owned(),
                source,
            })?;
        }

        let file = File::create(path).map_err(|source| ConcatenationError::Io {
            path: path.to_owned(),
            source,
        })?;

        Ok(Self {
            path: path.to_owned(),
            file,
        })
    }
}

impl OutputManifestSink for JsonlOutputManifestSink {
    fn record_manifest(
        &mut self,
        record: ConcatenationOutputManifestRecord,
    ) -> Result<(), ConcatenationError> {
        serde_json::to_writer(&mut self.file, &record).map_err(|source| {
            ConcatenationError::Io {
                path: self.path.clone(),
                source: io::Error::new(io::ErrorKind::InvalidData, source),
            }
        })?;
        self.file
            .write_all(b"\n")
            .map_err(|source| ConcatenationError::Io {
                path: self.path.clone(),
                source,
            })?;
        self.file.flush().map_err(|source| ConcatenationError::Io {
            path: self.path.clone(),
            source,
        })
    }
}

fn validate_options(options: &ConcatenateDirOptions) -> Result<(), ConcatenationError> {
    for (name, value) in [
        ("min_line_count", options.min_line_count),
        ("bucket_width", options.bucket_width),
        ("max_origin_uses", options.max_origin_uses),
        ("max_fragments_per_output", options.max_fragments_per_output),
    ] {
        if value == 0 {
            return Err(ConcatenationError::InvalidPositiveOption { name });
        }
    }
    Ok(())
}

fn write_output_manifest_jsonl(
    path: &Path,
    records: &[ConcatenationOutputManifestRecord],
) -> Result<(), ConcatenationError> {
    write_jsonl(path, records)
}

fn write_usage_jsonl(
    path: &Path,
    records: &[ConcatenationSourceUsageRecord],
) -> Result<(), ConcatenationError> {
    write_jsonl(path, records)
}

fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> Result<(), ConcatenationError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| ConcatenationError::Io {
            path: parent.to_owned(),
            source,
        })?;
    }

    let mut file = File::create(path).map_err(|source| ConcatenationError::Io {
        path: path.to_owned(),
        source,
    })?;
    for record in records {
        serde_json::to_writer(&mut file, record).map_err(|source| ConcatenationError::Io {
            path: path.to_owned(),
            source: io::Error::new(io::ErrorKind::InvalidData, source),
        })?;
        file.write_all(b"\n")
            .map_err(|source| ConcatenationError::Io {
                path: path.to_owned(),
                source,
            })?;
    }

    Ok(())
}

fn recreate_output_dir(output_dir: &Path) -> Result<(), ConcatenationError> {
    if output_dir.exists() {
        fs::remove_dir_all(output_dir).map_err(|source| ConcatenationError::Io {
            path: output_dir.to_owned(),
            source,
        })?;
    }
    fs::create_dir_all(output_dir).map_err(|source| ConcatenationError::Io {
        path: output_dir.to_owned(),
        source,
    })
}
