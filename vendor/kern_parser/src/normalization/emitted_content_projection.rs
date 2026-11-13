//! Projection onto the content model that measure-canonical serialization
//! emits.
//!
//! [`ContentPolicy`] and [`crate::serialize::TandemPolicy`] decide what the
//! Transcoda V2 presets leave out or rewrite. Normalization passes
//! that key on line adjacency or measure contents (multi-measure rest
//! compaction, invisible-rest measure removal, measure-slot
//! canonicalization, spine-operation cleanups, invisible-rest compaction)
//! must see the same document shape that a re-parse of the emitted text
//! would produce, otherwise every parse/serialize round trip keeps
//! normalizing further and the output drifts instead of being a fixpoint.
//! This pass applies that projection up front so in-memory normalization and
//! re-parsed normalization agree by construction.

use crate::serialize::TandemPolicy;
use crate::{
    normalization::{tandem_omitted_by, NormalizationPass, NormalizeOptions},
    score::Topology,
    serialize::{ContentKind, ContentPolicy},
    KernDocument, KernLine, LocalComment, SpineManipulator, SpineManipulatorLine, SpineType,
    TandemInterpretation, TandemInterpretationLine,
};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EmittedContentProjectionError {
    #[error("could not project document onto a valid kern-only spine grid")]
    InvalidKernOnlyGrid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Rewrite the document to the measure-canonical emitted content model.
pub struct ProjectEmittedContent {
    #[allow(dead_code)]
    options: NormalizeOptions,
}

impl ProjectEmittedContent {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for ProjectEmittedContent {
    fn normalize(&self, document: &mut KernDocument) {
        // The legacy normalization pass keeps its historical bail-safe
        // behavior. The staged Transcoda V2 pipeline calls the fallible owner
        // below and never relies on the serializer as a projection fallback.
        let _ = try_project_transcoda_v2_content(document, ContentPolicy::TranscodaV2);
    }
}

/// Project all content decisions owned by the Transcoda V2 dialects.
pub(crate) fn try_project_transcoda_v2_content(
    document: &mut KernDocument,
    policy: ContentPolicy,
) -> Result<(), EmittedContentProjectionError> {
    drop_non_emitted_local_comment_lines(document, policy);
    null_omitted_tandems(document, policy.tandem_policy());
    for line in document.lines_mut() {
        let KernLine::Data(contents) = line else {
            continue;
        };
        for cell in contents {
            if let crate::SpineContent::Kern(content) = cell {
                policy.project_content(content);
            }
        }
    }
    if policy.drops(ContentKind::NonKernSpine) {
        project_kern_spines_only(document)
            .ok_or(EmittedContentProjectionError::InvalidKernOnlyGrid)?;
    }
    Ok(())
}

/// Remove local comment lines that measure-canonical output omits entirely
/// and null the comments it omits on the lines that stay.
fn drop_non_emitted_local_comment_lines(document: &mut KernDocument, policy: ContentPolicy) {
    document.retain_lines(|line| {
        let KernLine::LocalCommentLine(comments) = line else {
            return true;
        };
        comments.iter().any(|comment| {
            !matches!(comment, LocalComment::Null) && policy.keeps_local_comment(comment)
        })
    });
    for line in document.lines_mut() {
        let KernLine::LocalCommentLine(comments) = line else {
            continue;
        };
        for comment in comments {
            if !policy.keeps_local_comment(comment) {
                *comment = LocalComment::Null;
            }
        }
    }
}

/// Null out tandem tokens that measure-canonical output omits (pedal marks,
/// tuplet visibility, section labels, ...). Lines that become entirely null
/// are removed later by `RemoveNullLines`.
fn null_omitted_tandems(document: &mut KernDocument, omitted: TandemPolicy) {
    for line in document.lines_mut() {
        let KernLine::TandemInterpretationLine(tandems) = line else {
            continue;
        };
        if !tandems
            .tokens()
            .iter()
            .any(|tandem| tandem_omitted_by(omitted, tandem))
        {
            continue;
        }
        let projected = tandems
            .tokens()
            .iter()
            .map(|tandem| {
                if tandem_omitted_by(omitted, tandem) {
                    TandemInterpretation::Null
                } else {
                    tandem.clone()
                }
            })
            .collect::<Vec<_>>();
        let Ok(projected) = TandemInterpretationLine::new(projected) else {
            continue;
        };
        *line = KernLine::TandemInterpretationLine(projected);
    }
}

/// Remove non-kern spines from the whole document, mirroring the per-line
/// column filtering the OMR serializer applies at emission time.
///
/// Documents with irregular widths or spine operations this projection
/// cannot express on the kern-only grid are left unchanged.
fn project_kern_spines_only(document: &mut KernDocument) -> Option<()> {
    if !document_has_non_kern_spines(document) {
        return Some(());
    }

    let topology = Topology::build(document).ok()?;
    let spine_types = |layers: &[crate::score::LayerId]| {
        layers
            .iter()
            .map(|&layer| topology.spine_type(layer).clone())
            .collect::<Vec<_>>()
    };
    let mut replacement: Vec<(KernLine, Option<usize>)> = Vec::new();

    for (index, line) in document.lines().iter().enumerate() {
        let source = document.source(index);
        let active = spine_types(topology.layers_at(index));
        match line {
            KernLine::SpineTypeLine(types) => {
                let kern_types = kern_cells(types, types);
                if kern_types.is_empty() {
                    return None;
                }
                replacement.push((KernLine::SpineTypeLine(kern_types), source));
            }
            KernLine::TandemInterpretationLine(tandems) => {
                let cells = kern_cells(tandems.tokens(), &active);
                let projected = TandemInterpretationLine::new(cells).ok()?;
                replacement.push((KernLine::TandemInterpretationLine(projected), source));
            }
            KernLine::LocalCommentLine(comments) => {
                replacement.push((
                    KernLine::LocalCommentLine(kern_cells(comments, &active)),
                    source,
                ));
            }
            KernLine::Data(contents) => {
                let cells = kern_cells(contents, &active);
                if cells.is_empty() {
                    return None;
                }
                replacement.push((KernLine::Data(cells), source));
            }
            KernLine::Barline(barline) => {
                let source_indices = active
                    .iter()
                    .enumerate()
                    .filter_map(|(index, spine_type)| {
                        matches!(spine_type, SpineType::Kern).then_some(index)
                    })
                    .collect::<Vec<_>>();
                replacement.push((
                    KernLine::Barline(barline.projected_to_spines(&source_indices)?),
                    source,
                ));
            }
            KernLine::SpineManipulatorLine(manipulators) => {
                let is_kern = |spine: &SpineType| matches!(spine, SpineType::Kern);
                let retained: Vec<_> = active.iter().map(is_kern).collect();
                let kern_after = spine_types(topology.layers_after(index))
                    .iter()
                    .filter(|spine| is_kern(spine))
                    .count();
                for projected in
                    project_manipulator_lines(manipulators.tokens(), &retained, kern_after)?
                {
                    replacement.push((KernLine::SpineManipulatorLine(projected), source));
                }
            }
        }
    }

    let kern_initial = document
        .layout
        .initial_spine_count
        .min(u32::try_from(replacement_initial_width(&replacement)).ok()?);
    document.splice_lines(0..document.lines().len(), replacement);
    document.layout.initial_spine_count = kern_initial;
    Some(())
}

fn replacement_initial_width(replacement: &[(KernLine, Option<usize>)]) -> usize {
    for (line, _) in replacement {
        match line {
            KernLine::SpineTypeLine(types) => return types.len(),
            KernLine::TandemInterpretationLine(tandems) => return tandems.len(),
            KernLine::LocalCommentLine(comments) => return comments.len(),
            KernLine::Data(contents) => return contents.len(),
            KernLine::SpineManipulatorLine(manipulators) => return manipulators.len(),
            KernLine::Barline(_) => {}
        }
    }
    0
}

/// Express retained columns as legal manipulator lines, keeping separate
/// merge groups distinct when removing their separator columns.
///
/// `retained[column]` says whether the input column survives, and
/// `retained_after` is how many columns survive once the line has applied.
/// A merge group must be retained whole or not at all; a caller dropping
/// some of a group's inputs rewrites the survivors' tokens first.
///
/// Merge groups are computed from the original column runs (a removed
/// separator column keeps two retained groups distinct even when filtering makes
/// them adjacent) and packed greedily onto shared lines, left to right,
/// matching the canonical packing `MergeAdjacentSpineManipulators` settles
/// on when the emitted text is re-parsed.
pub(crate) fn project_manipulator_lines(
    manipulators: &[SpineManipulator],
    retained: &[bool],
    retained_after: usize,
) -> Option<Vec<SpineManipulatorLine>> {
    let retained_tokens: Vec<_> = manipulators
        .iter()
        .zip(retained)
        .filter(|(_, retained)| **retained)
        .map(|(token, _)| token.clone())
        .collect();
    if retained_tokens
        .iter()
        .all(|token| matches!(token, SpineManipulator::Null))
    {
        return Some(Vec::new());
    }

    // Merge groups in retained coordinates, derived from the original runs.
    let mut retained_positions = vec![None; manipulators.len()];
    let mut retained_width = 0usize;
    for (index, position) in retained_positions.iter_mut().enumerate() {
        if retained.get(index).copied().unwrap_or(false) {
            *position = Some(retained_width);
            retained_width += 1;
        }
    }
    let mut merge_groups: Vec<(usize, usize)> = Vec::new();
    let mut index = 0;
    while index < manipulators.len() {
        if !matches!(manipulators[index], SpineManipulator::Merge) {
            index += 1;
            continue;
        }
        let start = index;
        while index < manipulators.len() && matches!(manipulators[index], SpineManipulator::Merge) {
            index += 1;
        }
        let positions = (start..index)
            .filter_map(|position| retained_positions[position])
            .collect::<Vec<_>>();
        if positions.len() > 1 {
            merge_groups.push((positions[0], positions.len()));
        }
    }

    // The retained tokens form one line unless removing a separator column
    // made two merge groups touch.
    if rendered_merge_runs(&retained_tokens) == merge_groups.len()
        && rendered_width(&retained_tokens) == retained_after
    {
        if let Ok(line) = SpineManipulatorLine::new(retained_tokens.clone()) {
            return Some(vec![line]);
        }
    }

    if !retained_tokens
        .iter()
        .all(|token| matches!(token, SpineManipulator::Null | SpineManipulator::Merge))
    {
        return None;
    }

    // Greedy first-fit packing: place each group on the earliest line where
    // it does not touch an already placed group.
    let mut lines: Vec<Vec<(usize, usize)>> = Vec::new();
    for group in merge_groups {
        let slot = lines.iter_mut().find(|line| {
            line.iter()
                .all(|(start, len)| group.0 + group.1 < *start || start + len < group.0)
        });
        match slot {
            Some(line) => line.push(group),
            None => lines.push(vec![group]),
        }
    }

    // Render each line, shifting group positions by the columns collapsed on
    // earlier lines to their left, and verify the rendered runs still match
    // the intended groups.
    let mut output = Vec::new();
    let mut current_width = retained_width;
    let mut collapsed: Vec<(usize, usize)> = Vec::new();
    for line_groups in lines {
        let mut tokens = vec![SpineManipulator::Null; current_width];
        for (start, len) in &line_groups {
            let shift: usize = collapsed
                .iter()
                .filter(|(collapsed_start, _)| collapsed_start < start)
                .map(|(_, collapsed_len)| collapsed_len - 1)
                .sum();
            let start = start.checked_sub(shift)?;
            if start + len > current_width {
                return None;
            }
            for token in tokens.iter_mut().skip(start).take(*len) {
                *token = SpineManipulator::Merge;
            }
        }
        if rendered_merge_runs(&tokens) != line_groups.len() {
            return None;
        }
        output.push(SpineManipulatorLine::new(tokens).ok()?);
        current_width -= line_groups.iter().map(|(_, len)| len - 1).sum::<usize>();
        collapsed.extend(line_groups);
    }

    (current_width == retained_after).then_some(output)
}

/// Columns a manipulator line leaves active.
fn rendered_width(tokens: &[SpineManipulator]) -> usize {
    let mut width = 0;
    let mut in_run = false;
    for token in tokens {
        match token {
            SpineManipulator::Merge => {
                if !in_run {
                    width += 1;
                }
                in_run = true;
                continue;
            }
            SpineManipulator::Null => width += 1,
            SpineManipulator::Split => width += 2,
            SpineManipulator::Terminate => {}
        }
        in_run = false;
    }
    width
}

fn rendered_merge_runs(tokens: &[SpineManipulator]) -> usize {
    let mut runs = 0;
    let mut in_run = false;
    for token in tokens {
        match token {
            SpineManipulator::Merge if !in_run => {
                runs += 1;
                in_run = true;
            }
            SpineManipulator::Merge => {}
            _ => in_run = false,
        }
    }
    runs
}

fn document_has_non_kern_spines(document: &KernDocument) -> bool {
    document.lines().iter().any(|line| {
        matches!(
            line,
            KernLine::SpineTypeLine(types)
                if types.iter().any(|spine| !matches!(spine, SpineType::Kern))
        )
    })
}

fn kern_cells<T: Clone>(cells: &[T], active: &[SpineType]) -> Vec<T> {
    cells
        .iter()
        .zip(active)
        .filter(|(_, spine)| matches!(spine, SpineType::Kern))
        .map(|(cell, _)| cell.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{try_project_transcoda_v2_content, EmittedContentProjectionError};
    use crate::{KernDocument, KernLine};

    #[test]
    fn fallible_projection_reports_an_irregular_non_kern_grid() {
        let mut document = KernDocument::parse("**kern\t**dynam\n4c\t.\n*-\t*-").unwrap();
        let cells = document
            .lines_mut()
            .iter_mut()
            .find_map(|line| match line {
                KernLine::Data(cells) => Some(cells),
                _ => None,
            })
            .unwrap();
        cells.pop();

        assert_eq!(
            try_project_transcoda_v2_content(
                &mut document,
                crate::serialize::ContentPolicy::TranscodaV2
            ),
            Err(EmittedContentProjectionError::InvalidKernOnlyGrid)
        );
    }

    #[test]
    fn projection_handles_non_kern_spines_across_a_kern_split_merge() {
        let mut document =
            KernDocument::parse("**kern\t**dynam\n*^\t*\n4c\t4e\t.\n*v\t*v\t*\n4g\t.\n*-\t*-")
                .unwrap();

        try_project_transcoda_v2_content(
            &mut document,
            crate::serialize::ContentPolicy::TranscodaV2,
        )
        .unwrap();
        assert!(document.lines().iter().all(|line| !matches!(
            line,
            KernLine::SpineTypeLine(types)
                if types.iter().any(|spine| !matches!(spine, crate::SpineType::Kern))
        )));
    }

    #[test]
    fn projection_keeps_mixed_barlines_attached_to_their_kern_spines() {
        let mut document = KernDocument::parse(
            "**kern\t**dynam\t**kern\n4C\t.\t4c\n=\t=\t=-\n4D\t.\t4d\n*-\t*-\t*-",
        )
        .unwrap();

        try_project_transcoda_v2_content(
            &mut document,
            crate::serialize::ContentPolicy::TranscodaV2,
        )
        .unwrap();
        let barline = document
            .lines()
            .iter()
            .find_map(|line| match line {
                KernLine::Barline(barline) => Some(barline),
                _ => None,
            })
            .unwrap();

        assert_eq!(barline.spine_count(), Some(2));
        assert!(!barline.is_invisible_for_spine(0));
        assert!(barline.is_invisible_for_spine(1));
    }
}
