use super::{
    accept::AcceptedCandidate,
    manifest::{SourceOrigin, SourceUnitRef},
};

const HASH_SCHEMA_SALT: &str = "transcoda-concat-output-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OutputIdentity {
    pub(super) filename: String,
    pub(super) output_id: String,
    pub(super) content_hash: String,
    pub(super) provenance_hash: String,
}

pub(super) fn output_identity(
    output_index: usize,
    units: &[SourceUnitRef],
    candidate: &AcceptedCandidate,
) -> OutputIdentity {
    let content_hash = stable_hash_hex(|hasher| {
        hasher.write_str(HASH_SCHEMA_SALT);
        hasher.write_str("content");
        hasher.write_str(&candidate.contents);
    });
    let provenance_hash = stable_hash_hex(|hasher| {
        hasher.write_str(HASH_SCHEMA_SALT);
        hasher.write_str("provenance");
        hasher.write_usize(candidate.unit_indices.len());
        for (position, unit_index) in candidate.unit_indices.iter().enumerate() {
            let unit = &units[*unit_index];
            hasher.write_usize(position);
            hasher.write_str(&unit.relative_label);
            hasher.write_usize(candidate.truncations[position]);
        }
        hasher.write_str(&candidate.contents);
    });
    let output_id = provenance_hash.chars().take(12).collect::<String>();
    OutputIdentity {
        filename: format!("concat_{output_index:06}_{output_id}.krn"),
        output_id,
        content_hash,
        provenance_hash,
    }
}

#[derive(Debug, Clone, Copy)]
struct StableHasher {
    value: u64,
}

impl StableHasher {
    fn new() -> Self {
        Self {
            value: 0xcbf29ce484222325,
        }
    }

    fn write_str(&mut self, value: &str) {
        self.write_bytes(value.as_bytes());
    }

    fn write_usize(&mut self, value: usize) {
        self.write_bytes(&(value as u64).to_le_bytes());
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        self.write_raw_bytes(&(bytes.len() as u64).to_le_bytes());
        self.write_raw_bytes(bytes);
    }

    fn write_raw_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.value ^= u64::from(*byte);
            self.value = self.value.wrapping_mul(0x100000001b3);
        }
    }

    fn finish_hex(self) -> String {
        format!("{:016x}", self.value)
    }
}

fn stable_hash_hex(write: impl FnOnce(&mut StableHasher)) -> String {
    let mut hasher = StableHasher::new();
    write(&mut hasher);
    hasher.finish_hex()
}

pub(super) fn manifest_sources(
    origins: &[SourceOrigin],
    units: &[SourceUnitRef],
    candidate: &AcceptedCandidate,
) -> Vec<super::ConcatenationOutputManifestSource> {
    candidate
        .unit_indices
        .iter()
        .enumerate()
        .map(|(position, unit_index)| {
            let unit = &units[*unit_index];
            let origin = &origins[unit.origin_index];
            let (chunk_index, chunk_count, start_bar_index, end_bar_index) =
                unit.kind.split_metadata();
            super::ConcatenationOutputManifestSource {
                position,
                source_path: origin.relative_path.display().to_string(),
                unit_kind: unit.kind.manifest_kind(),
                origin_usage_ordinal: candidate.origin_usage_ordinals[position],
                truncated_bars: candidate.truncations[position],
                salvaged_bars: unit.kind.salvaged_bars(),
                staff_count: unit.staff_count,
                line_count: origin.original_line_count,
                unit_line_count: unit.line_count,
                contributed_line_count: candidate.contributed_line_counts[position],
                chunk_index,
                chunk_count,
                start_bar_index,
                end_bar_index,
            }
        })
        .collect()
}
