use std::{
    fs,
    path::{Path, PathBuf},
};

use walkdir::WalkDir;

use crate::{
    transforms::{musical_bar_count, split_trailing_bars},
    KernDocument, SerializePreset,
};

use super::{
    compact_line_count, document::has_musical_data, line_cap_for_staff_count, split_plan,
    ConcatenationError, ConcatenationManifestUnitKind, ConcatenationProgress,
    ConcatenationSourceUsageReason,
};

#[derive(Debug, Clone)]
pub(super) struct SourceOrigin {
    pub(super) origin_index: usize,
    pub(super) relative_path: PathBuf,
    #[allow(dead_code)]
    pub(super) absolute_path: PathBuf,
    pub(super) staff_count: usize,
    pub(super) original_line_count: usize,
    pub(super) oversized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SourceUnitKind {
    Original,
    SplitChunk {
        chunk_index: usize,
        chunk_count: usize,
        start_bar_index: usize,
        end_bar_index: usize,
    },
    SalvagedTail {
        parent_output_index: usize,
        start_bar_index: usize,
        salvaged_bars: usize,
    },
}

impl SourceUnitKind {
    pub(super) fn manifest_kind(&self) -> ConcatenationManifestUnitKind {
        match self {
            Self::Original => ConcatenationManifestUnitKind::Original,
            Self::SplitChunk { .. } => ConcatenationManifestUnitKind::SplitChunk,
            Self::SalvagedTail { .. } => ConcatenationManifestUnitKind::SalvagedTail,
        }
    }

    pub(super) fn split_metadata(
        &self,
    ) -> (Option<usize>, Option<usize>, Option<usize>, Option<usize>) {
        match self {
            Self::SplitChunk {
                chunk_index,
                chunk_count,
                start_bar_index,
                end_bar_index,
            } => (
                Some(*chunk_index),
                Some(*chunk_count),
                Some(*start_bar_index),
                Some(*end_bar_index),
            ),
            _ => (None, None, None, None),
        }
    }

    pub(super) fn salvaged_bars(&self) -> usize {
        match self {
            Self::SalvagedTail { salvaged_bars, .. } => *salvaged_bars,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct SourceUnitRef {
    pub(super) unit_index: usize,
    pub(super) origin_index: usize,
    pub(super) kind: SourceUnitKind,
    pub(super) relative_label: String,
    pub(super) contents: Option<String>,
    pub(super) staff_count: usize,
    pub(super) line_count: usize,
}

#[derive(Debug, Clone)]
pub(super) struct SourceManifest {
    pub(super) origins: Vec<SourceOrigin>,
    pub(super) units: Vec<SourceUnitRef>,
    pub(super) origin_rejections: Vec<Option<ConcatenationSourceUsageReason>>,
    pub(super) input_count: usize,
    pub(super) skipped_parse_count: usize,
    pub(super) split_chunk_count: usize,
}

pub(super) fn build_manifest(
    input_dir: &Path,
    progress: &impl ConcatenationProgress,
) -> Result<SourceManifest, ConcatenationError> {
    let mut paths = Vec::new();
    for entry in WalkDir::new(input_dir) {
        let entry = entry.map_err(|source| ConcatenationError::Io {
            path: input_dir.to_owned(),
            source: source.into(),
        })?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|ext| ext == "krn") {
            paths.push(entry.path().to_owned());
        }
    }
    paths.sort();

    let input_count = paths.len();
    progress.manifest_started(input_count);

    let mut skipped_parse_count = 0;
    let mut origins = Vec::new();
    let mut units = Vec::new();
    let mut origin_rejections = Vec::new();
    let mut split_chunk_count = 0;

    for path in paths {
        let relative_path = path
            .strip_prefix(input_dir)
            .expect("walkdir yielded a path below input_dir")
            .to_owned();
        let input = fs::read_to_string(&path).map_err(|source| ConcatenationError::Io {
            path: path.clone(),
            source,
        })?;
        let document = match KernDocument::parse(&input) {
            Ok(document) => document,
            Err(_) => {
                skipped_parse_count += 1;
                progress.manifest_advanced();
                continue;
            }
        };

        let staff_count = document.staff_count();
        let original_line_count = match training_line_count(&document) {
            Ok(line_count) => line_count,
            Err(reason) => {
                origins.push(SourceOrigin {
                    origin_index: origins.len(),
                    relative_path: relative_path.clone(),
                    absolute_path: path,
                    staff_count,
                    original_line_count: 0,
                    oversized: false,
                });
                origin_rejections.push(Some(reason));
                progress.manifest_advanced();
                continue;
            }
        };
        let origin_index = origins.len();
        let line_cap = match line_cap_for_staff_count(staff_count) {
            Ok(line_cap) => line_cap,
            Err(ConcatenationError::UnsupportedStaffCount(_)) => {
                origins.push(SourceOrigin {
                    origin_index,
                    relative_path: relative_path.clone(),
                    absolute_path: path,
                    staff_count,
                    original_line_count,
                    oversized: false,
                });
                origin_rejections.push(Some(ConcatenationSourceUsageReason::UnsupportedStaffCount));
                progress.manifest_advanced();
                continue;
            }
            Err(error) => return Err(error),
        };
        origins.push(SourceOrigin {
            origin_index,
            relative_path: relative_path.clone(),
            absolute_path: path,
            staff_count,
            original_line_count,
            oversized: original_line_count > line_cap,
        });
        origin_rejections.push(None);

        if original_line_count <= line_cap {
            push_original_unit(
                &mut units,
                origin_index,
                relative_path.display().to_string(),
                staff_count,
                original_line_count,
            );
        } else {
            match split_oversized_source(
                origin_index,
                &relative_path,
                document,
                staff_count,
                line_cap,
            ) {
                Ok(mut split_units) if !split_units.is_empty() => {
                    split_chunk_count += split_units.len();
                    for unit in &mut split_units {
                        unit.unit_index = units.len();
                        units.push(unit.clone());
                    }
                }
                Ok(_) | Err(ConcatenationSourceUsageReason::OversizedSplitFailed) => {
                    origin_rejections[origin_index] =
                        Some(ConcatenationSourceUsageReason::OversizedSplitFailed);
                }
                Err(reason) => {
                    origin_rejections[origin_index] = Some(reason);
                }
            }
        }

        progress.manifest_advanced();
    }
    progress.manifest_finished(origins.len(), skipped_parse_count);

    Ok(SourceManifest {
        origins,
        units,
        origin_rejections,
        input_count,
        skipped_parse_count,
        split_chunk_count,
    })
}

pub(super) fn push_unit(
    units: &mut Vec<SourceUnitRef>,
    origin_index: usize,
    kind: SourceUnitKind,
    relative_label: String,
    document: KernDocument,
    staff_count: usize,
) -> Result<(), ConcatenationSourceUsageReason> {
    let line_count = training_line_count(&document)?;
    let contents = SerializePreset::HumdrumKern
        .try_serialize_document(&document)
        .map_err(|_| ConcatenationSourceUsageReason::SplitChunkInvalid)?;
    units.push(SourceUnitRef {
        unit_index: units.len(),
        origin_index,
        kind,
        relative_label,
        contents: Some(contents),
        staff_count,
        line_count,
    });
    Ok(())
}

pub(super) fn push_original_unit(
    units: &mut Vec<SourceUnitRef>,
    origin_index: usize,
    relative_label: String,
    staff_count: usize,
    line_count: usize,
) {
    units.push(SourceUnitRef {
        unit_index: units.len(),
        origin_index,
        kind: SourceUnitKind::Original,
        relative_label,
        contents: None,
        staff_count,
        line_count,
    });
}

fn split_oversized_source(
    origin_index: usize,
    relative_path: &Path,
    document: KernDocument,
    staff_count: usize,
    line_cap: usize,
) -> Result<Vec<SourceUnitRef>, ConcatenationSourceUsageReason> {
    let mut chunks = Vec::new();
    let mut start_bar_index = 0;
    let mut remaining = split_plan::compact_for_line_planning(&document);

    loop {
        let planned = split_plan::planned_transcoda_v2_line_count(&remaining)
            .map_err(|_| ConcatenationSourceUsageReason::TargetSerializationFailed)?;
        if planned <= line_cap {
            if has_musical_data(&remaining) {
                let line_count = training_line_count(&remaining)?;
                if line_count <= line_cap {
                    chunks.push(pending_split_unit(
                        origin_index,
                        relative_path,
                        chunks.len(),
                        0,
                        start_bar_index,
                        start_bar_index + musical_bar_count(&remaining),
                        remaining,
                        staff_count,
                        Some(line_count),
                    )?);
                    break;
                }
            } else {
                break;
            }
        }

        let bar_count = musical_bar_count(&remaining);
        if bar_count < 2 {
            return Err(ConcatenationSourceUsageReason::OversizedSplitFailed);
        }

        let mut accepted_split = None;
        let candidates = split_plan::retained_bar_candidates_under_cap(&remaining, line_cap)
            .map_err(|_| ConcatenationSourceUsageReason::TargetSerializationFailed)?;
        for retained_bars in candidates {
            let split = split_trailing_bars(&remaining, bar_count - retained_bars)
                .map_err(|_| ConcatenationSourceUsageReason::OversizedSplitFailed)?;
            if !has_musical_data(&split.retained)
                || !has_musical_data(&split.removed)
                || !parse_round_trips(&split.removed)
            {
                continue;
            }
            let Ok(retained_line_count) = training_line_count(&split.retained) else {
                continue;
            };
            if retained_line_count > line_cap {
                continue;
            }
            accepted_split = Some((retained_bars, retained_line_count, split));
            break;
        }

        let Some((retained_bars, retained_line_count, split)) = accepted_split else {
            return Err(ConcatenationSourceUsageReason::SplitChunkInvalid);
        };
        chunks.push(pending_split_unit(
            origin_index,
            relative_path,
            chunks.len(),
            0,
            start_bar_index,
            start_bar_index + retained_bars,
            split.retained,
            staff_count,
            Some(retained_line_count),
        )?);
        start_bar_index += retained_bars;
        remaining = split.removed;
    }

    let chunk_count = chunks.len();
    for (index, chunk) in chunks.iter_mut().enumerate() {
        if let SourceUnitKind::SplitChunk {
            chunk_index,
            chunk_count: stored_count,
            ..
        } = &mut chunk.kind
        {
            *chunk_index = index;
            *stored_count = chunk_count;
        }
        chunk.relative_label = format!(
            "{}#chunk-{index:03}-of-{chunk_count:03}",
            relative_path.display()
        );
    }

    Ok(chunks)
}

#[allow(clippy::too_many_arguments)]
fn pending_split_unit(
    origin_index: usize,
    relative_path: &Path,
    chunk_index: usize,
    chunk_count: usize,
    start_bar_index: usize,
    end_bar_index: usize,
    document: KernDocument,
    staff_count: usize,
    line_count: Option<usize>,
) -> Result<SourceUnitRef, ConcatenationSourceUsageReason> {
    if !has_musical_data(&document) || !parse_round_trips(&document) {
        return Err(ConcatenationSourceUsageReason::SplitChunkInvalid);
    }

    let line_count = match line_count {
        Some(line_count) => line_count,
        None => training_line_count(&document)?,
    };
    let contents = SerializePreset::HumdrumKern
        .try_serialize_document(&document)
        .map_err(|_| ConcatenationSourceUsageReason::SplitChunkInvalid)?;
    Ok(SourceUnitRef {
        unit_index: 0,
        origin_index,
        kind: SourceUnitKind::SplitChunk {
            chunk_index,
            chunk_count,
            start_bar_index,
            end_bar_index,
        },
        relative_label: relative_path.display().to_string(),
        contents: Some(contents),
        staff_count,
        line_count,
    })
}

fn parse_round_trips(document: &KernDocument) -> bool {
    let Ok(contents) = SerializePreset::HumdrumKern.try_serialize_document(document) else {
        return false;
    };
    KernDocument::parse(&contents).is_ok()
}

fn training_line_count(document: &KernDocument) -> Result<usize, ConcatenationSourceUsageReason> {
    SerializePreset::TranscodaV2
        .try_serialize_document(document)
        .map(|target| compact_line_count(&target))
        .map_err(|_| ConcatenationSourceUsageReason::TargetSerializationFailed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn long_one_staff_document(bar_count: usize, lines_per_bar: usize) -> KernDocument {
        let mut source = String::from("*clefG2\n*k[]\n*M4/4\n");
        for index in 0..bar_count {
            source.push_str(&format!("={}\n", index + 1));
            for _ in 0..lines_per_bar {
                source.push_str("4c\n");
            }
        }
        source.push_str("*-\n");
        KernDocument::parse(&source).unwrap()
    }

    #[test]
    fn oversized_split_selects_largest_prefix_under_line_cap() {
        let document = long_one_staff_document(12, 20);
        let line_cap = 75;

        let units =
            split_oversized_source(0, Path::new("piece.krn"), document.clone(), 1, line_cap)
                .unwrap();
        let first_unit = &units[0];
        assert!(first_unit.line_count <= line_cap);

        let SourceUnitKind::SplitChunk {
            end_bar_index: retained_bars,
            ..
        } = first_unit.kind
        else {
            panic!("expected split chunk");
        };
        let compacted = split_plan::compact_for_line_planning(&document);
        let bar_count = musical_bar_count(&compacted);
        let next_split = split_trailing_bars(&compacted, bar_count - retained_bars - 1).unwrap();
        let next_line_count = compact_line_count(
            &SerializePreset::TranscodaV2
                .try_serialize_document(&next_split.retained)
                .unwrap(),
        );
        assert!(next_line_count > line_cap);
    }
}
