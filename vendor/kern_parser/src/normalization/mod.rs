//! Normalization passes for typed Humdrum `**kern` documents.
//!
//! These passes edit the typed AST into a more canonical shape without
//! reparsing raw text. The OMR-specific preset in [`omr`] composes the passes
//! needed for serialization-facing filtering and header ordering.
//!
//! Example:
//! ```text
//! **kern
//! *
//! 4c[]
//! ```
//!
//! Individual passes can remove the null tandem line, canonicalize the
//! same-token tie, or compose both operations through an OMR-facing preset.

use crate::{serialize::TandemPolicy, KernDocument, TandemInterpretation};

pub mod barline_spine_operations;
pub mod batch;
pub mod beam_spelling;
pub mod canceling_split_merge_regions;
pub(crate) mod canon;
pub mod control_tandem_adjacency;
pub mod dataset;
pub mod dynamics;
pub mod emitted_content_projection;
pub mod initial_header;
pub mod interpretation_lines;
pub mod invisible_rest_compaction;
pub mod invisible_rest_measures;
pub mod invisible_rest_padding;
pub mod invisible_rest_slices;
pub(crate) mod invisible_rest_tokens;
pub mod measure_grid;
pub mod multi_measure_rests;
pub mod null_lines;
pub mod omr;
pub mod redundant_state_interpretations;
pub mod restore_final_staff_topology;
pub mod slurs;
pub mod spine_manipulator_lines;
pub mod spine_operation_interpretations;
pub(crate) mod spine_topology;
pub mod superseded_interpretations;
pub mod ties;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizeOptions {
    pub compact_multi_measure_rests: bool,
    pub preserve_leading_barlines: bool,
    pub remove_invisible_rest_measures: bool,
    pub canonicalize_measure_grid: bool,
    /// Project the document onto the measure-canonical emitted content model
    /// (kern-only spines, no dropped comment or tandem lines) before the
    /// other passes run, so normalization sees the same shape a re-parse of
    /// the emitted text would produce.
    pub project_to_emitted_kern: bool,
    /// Respell beam marks from the beams Verovio draws; see [`beam_spelling`].
    pub canonicalize_beam_spelling: bool,
}

impl Default for NormalizeOptions {
    fn default() -> Self {
        Self {
            compact_multi_measure_rests: false,
            preserve_leading_barlines: false,
            remove_invisible_rest_measures: true,
            canonicalize_measure_grid: false,
            project_to_emitted_kern: false,
            canonicalize_beam_spelling: false,
        }
    }
}

/// Whether an output under `policy` leaves this tandem out.
///
/// Passes that decide layout from line adjacency must treat lines made only
/// of these tokens as transparent: the emitted text has no trace of them, so
/// letting them block or split a rewrite makes the next parse/serialize
/// round trip normalize further and the output drifts. A tandem the output
/// writes must keep blocking.
pub(crate) fn tandem_omitted_by(policy: TandemPolicy, tandem: &TandemInterpretation) -> bool {
    !matches!(tandem, TandemInterpretation::Null) && !policy.writes(tandem)
}

pub trait NormalizationPass {
    fn normalize(&self, document: &mut KernDocument);
}
