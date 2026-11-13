//! Transcoda v2 durationless invisible rest slice projection.
//!
//! Canonical `**kern` invisible rests are duration-bearing (`4ryy`).  The
//! Transcoda v2 training dialect can instead encode an invisible rest that is
//! active on a data line as `ryy`; the line duration supplies the missing
//! rhythmic value when the target is rehydrated.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::{
    analysis::{
        data_line_durations, line_durations::ScoreClock, score_line_durations,
        DataLineDurationError,
    },
    document::Framing,
    measure::MusicalDuration,
    score::{LayerId, LayerOrigin, Score},
    InvisibleRestSlice, KernContent, KernDocument, KernLine, Rest, SpineContent, SpineType,
};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InvisibleRestSliceError {
    #[error(transparent)]
    DataLineDurations(#[from] DataLineDurationError),
    #[error(
        "line {line}: cannot merge invisible rest slices at different remaining durations; first {first}, found {found}"
    )]
    UnequalMergedRemaining {
        line: usize,
        first: MusicalDuration,
        found: MusicalDuration,
    },
    #[error("line {line}, column {column}: durationless invisible rest slice has zero duration")]
    ZeroDurationSlice { line: usize, column: usize },
    #[error("line {line}, column {column}: cannot spell invisible rest slice duration {duration}")]
    UnspellableSliceDuration {
        line: usize,
        column: usize,
        duration: MusicalDuration,
    },
    #[error(
        "line {line}, column {column}: invisible rest slice exceeds tracked remaining duration; slice {slice}, remaining {remaining}"
    )]
    RemainingDurationUnderflow {
        line: usize,
        column: usize,
        slice: MusicalDuration,
        remaining: MusicalDuration,
    },
    #[error("Transcoda V2 lowering failed: {message}")]
    TranscodaV2Lowering { message: String },
    #[error("Transcoda V2 projected invariant failed: {message}")]
    TranscodaV2Invariant { message: String },
    #[error("Transcoda V2 canonicalization did not converge after {iterations} iterations")]
    TranscodaV2DidNotConverge { iterations: usize },
    #[error("Transcoda V2 semantic event projection failed: {message}")]
    TranscodaV2SemanticEvents { message: String },
    #[error(transparent)]
    Dialect(#[from] crate::serialize::DialectError),
    #[error(
        "line {line} column {column}: this merge closes a split the page never opens; \
         a label renders the split's columns as separate spines, which cannot merge"
    )]
    UnopenedSplitMerge { line: usize, column: usize },
    #[error(transparent)]
    TimelineBreak(#[from] crate::analysis::TimelineBreak),
}

/// Return whether the document contains any Transcoda v2 durationless rest slices.
pub fn contains_durationless_invisible_rest_slices(document: &KernDocument) -> bool {
    document.lines().iter().any(|line| {
        matches!(
            line,
            KernLine::Data(contents)
                if contents.iter().any(|content| {
                    matches!(content, SpineContent::Kern(KernContent::InvisibleRestSlice(_)))
                })
        )
    })
}

/// Project duration-bearing invisible rests to Transcoda v2 `ryy` slices.
pub fn project_durationless_invisible_rest_slices(
    document: &mut KernDocument,
) -> Result<(), InvisibleRestSliceError> {
    let original = document.clone();
    let score =
        Score::resolve(&original, Framing::Excerpt).map_err(DataLineDurationError::State)?;
    let durations = score_line_durations(&score)?
        .into_iter()
        .map(|duration| (duration.line_index, duration.duration))
        .collect::<BTreeMap<_, _>>();
    let topology = score.topology();
    // Remaining duration of the invisible rest each layer is projecting.
    let mut remaining = vec![MusicalDuration::zero(); topology.layer_count()];
    let mut projected_clock = ScoreClock::new(&score);

    for (line_index, line) in document.lines_mut().iter_mut().enumerate() {
        let source_line = original.source_line_number(line_index);
        match line {
            KernLine::SpineManipulatorLine(_) => {
                carry_into_new_layers(&score, line_index, &mut remaining, source_line)?;
                projected_clock.observe(line_index)?;
            }
            KernLine::Data(contents) => {
                let layers = topology.layers_at(line_index);
                let kern = layers
                    .iter()
                    .map(|&layer| *topology.spine_type(layer) == SpineType::Kern)
                    .collect::<Vec<_>>();
                let mut columns = layers
                    .iter()
                    .map(|layer| remaining[layer.index()])
                    .collect::<Vec<_>>();
                let slice = durations
                    .get(&line_index)
                    .copied()
                    .unwrap_or_else(MusicalDuration::zero);
                project_data_line(
                    contents,
                    &kern,
                    &mut columns,
                    slice,
                    (line_index, source_line),
                    &mut projected_clock,
                )?;
                for (layer, column) in layers.iter().zip(columns) {
                    remaining[layer.index()] = column;
                }
            }
            _ => {}
        }
    }

    remove_zero_duration_slice_only_data_lines(document)?;

    Ok(())
}

/// A split child keeps projecting its parent's rest; merged layers must be
/// at the same point of theirs.
fn carry_into_new_layers(
    score: &Score<'_>,
    line_index: usize,
    remaining: &mut [MusicalDuration],
    source_line: usize,
) -> Result<(), InvisibleRestSliceError> {
    let topology = score.topology();
    let before = topology.layers_at(line_index);
    for &layer in topology.layers_after(line_index) {
        if before.contains(&layer) {
            continue;
        }
        remaining[layer.index()] = match &topology.layer(layer).origin {
            LayerOrigin::SplitOf(parent) => remaining[parent.index()],
            LayerOrigin::MergeOf(inputs) => merged_remaining(inputs, remaining, source_line)?,
            LayerOrigin::Exclusive => MusicalDuration::zero(),
        };
    }
    Ok(())
}

fn merged_remaining(
    inputs: &[LayerId],
    remaining: &[MusicalDuration],
    source_line: usize,
) -> Result<MusicalDuration, InvisibleRestSliceError> {
    let first = remaining[inputs[0].index()];
    match inputs
        .iter()
        .map(|input| remaining[input.index()])
        .find(|found| *found != first)
    {
        Some(found) => Err(InvisibleRestSliceError::UnequalMergedRemaining {
            line: source_line,
            first,
            found,
        }),
        None => Ok(first),
    }
}

/// Rehydrate durationless rest slices only when the document actually contains them.
pub fn rehydrate_durationless_invisible_rest_slices_if_present(
    document: &mut KernDocument,
) -> Result<(), InvisibleRestSliceError> {
    if contains_durationless_invisible_rest_slices(document) {
        rehydrate_durationless_invisible_rest_slices(document)?;
    }

    Ok(())
}

/// Rehydrate Transcoda v2 `ryy` slices into ordinary invisible rests.
pub fn rehydrate_durationless_invisible_rest_slices(
    document: &mut KernDocument,
) -> Result<(), InvisibleRestSliceError> {
    let durations = data_line_durations(document)?
        .into_iter()
        .map(|duration| (duration.line_index, duration.duration))
        .collect::<BTreeMap<_, _>>();
    let source_lines = source_lines(document);

    for (line_index, line) in document.lines_mut().iter_mut().enumerate() {
        let KernLine::Data(contents) = line else {
            continue;
        };
        let slice = durations
            .get(&line_index)
            .copied()
            .unwrap_or_else(MusicalDuration::zero);
        let source_line = source_lines[line_index];

        for (column_index, content) in contents.iter_mut().enumerate() {
            let SpineContent::Kern(KernContent::InvisibleRestSlice(slice_rest)) = content else {
                continue;
            };
            if slice.is_zero() {
                return Err(InvisibleRestSliceError::ZeroDurationSlice {
                    line: source_line,
                    column: column_index + 1,
                });
            }
            let duration =
                slice
                    .into_duration()
                    .ok_or(InvisibleRestSliceError::UnspellableSliceDuration {
                        line: source_line,
                        column: column_index + 1,
                        duration: slice,
                    })?;
            *content = SpineContent::Kern(KernContent::Rest(Rest {
                duration,
                cross_staff: None,
                placement: None,
                invisible: true,
                breath: None,
                fermata: slice_rest.fermata,
                slurs: Vec::new(),
                phrases: Vec::new(),
            }));
        }
    }

    data_line_durations(document)?;

    Ok(())
}

/// Project one data line, keeping invisible rests that start here as slices
/// only when the projected line still advances by `slice`.
fn project_data_line(
    contents: &mut [SpineContent],
    kern: &[bool],
    remaining: &mut [MusicalDuration],
    slice: MusicalDuration,
    (line_index, source_line): (usize, usize),
    projected_clock: &mut ScoreClock<'_, '_>,
) -> Result<(), InvisibleRestSliceError> {
    let mut candidate_contents = contents.to_vec();
    let mut candidate_remaining = remaining.to_vec();
    apply_data_line_projection(
        &mut candidate_contents,
        kern,
        &mut candidate_remaining,
        slice,
        source_line,
        true,
    )?;
    let mut candidate_clock = projected_clock.clone();
    if candidate_clock.advance(line_index, &candidate_contents)? == slice {
        contents.clone_from_slice(&candidate_contents);
        remaining.clone_from_slice(&candidate_remaining);
        *projected_clock = candidate_clock;
        return Ok(());
    }

    let mut fallback_contents = contents.to_vec();
    apply_data_line_projection(
        &mut fallback_contents,
        kern,
        remaining,
        slice,
        source_line,
        false,
    )?;
    projected_clock.advance(line_index, &fallback_contents)?;
    contents.clone_from_slice(&fallback_contents);

    Ok(())
}

fn apply_data_line_projection(
    contents: &mut [SpineContent],
    kern: &[bool],
    remaining: &mut [MusicalDuration],
    slice: MusicalDuration,
    source_line: usize,
    convert_new_rests: bool,
) -> Result<(), InvisibleRestSliceError> {
    if slice.is_zero() {
        return Ok(());
    }

    for (column_index, ((content, &kern), remaining)) in contents
        .iter_mut()
        .zip(kern)
        .zip(remaining.iter_mut())
        .enumerate()
    {
        if !kern {
            continue;
        }

        match content {
            SpineContent::Kern(KernContent::Rest(rest)) if rest.invisible && convert_new_rests => {
                *remaining = MusicalDuration::from_duration(&rest.duration);
                *content =
                    SpineContent::Kern(KernContent::InvisibleRestSlice(InvisibleRestSlice {
                        fermata: rest.fermata,
                    }));
            }
            SpineContent::Kern(KernContent::Null) if !remaining.is_zero() => {
                *content =
                    SpineContent::Kern(KernContent::InvisibleRestSlice(InvisibleRestSlice {
                        fermata: None,
                    }));
            }
            _ => {}
        }

        if !remaining.is_zero() {
            *remaining = remaining.checked_sub(slice).ok_or(
                InvisibleRestSliceError::RemainingDurationUnderflow {
                    line: source_line,
                    column: column_index + 1,
                    slice,
                    remaining: *remaining,
                },
            )?;
        }
    }

    Ok(())
}

fn remove_zero_duration_slice_only_data_lines(
    document: &mut KernDocument,
) -> Result<(), InvisibleRestSliceError> {
    let durations = data_line_durations(document)?
        .into_iter()
        .map(|duration| (duration.line_index, duration.duration))
        .collect::<BTreeMap<_, _>>();
    let lines = document
        .take_lines()
        .into_iter()
        .enumerate()
        .filter_map(|(line_index, (line, source))| {
            let is_zero_duration_slice_line = matches!(
                line,
                KernLine::Data(ref contents) if data_line_is_only_invisible_rest_slices(contents)
            ) && durations
                .get(&line_index)
                .copied()
                .unwrap_or_else(MusicalDuration::zero)
                .is_zero();
            (!is_zero_duration_slice_line).then_some((line, source))
        })
        .collect();
    document.replace_lines(lines);

    Ok(())
}

fn data_line_is_only_invisible_rest_slices(contents: &[SpineContent]) -> bool {
    let mut saw_slice = false;

    for content in contents {
        match content {
            SpineContent::Kern(KernContent::Null) => {}
            SpineContent::Kern(KernContent::InvisibleRestSlice(_)) => saw_slice = true,
            SpineContent::Dynam(crate::DynamContent::Null)
            | SpineContent::Fing(crate::FingContent::Null)
            | SpineContent::Text(crate::TextContent::Null)
            | SpineContent::MxmlHarmony(crate::MxmlHarmonyContent::Null) => {}
            SpineContent::Kern(
                KernContent::Rest(_)
                | KernContent::MultiMeasureRest(_)
                | KernContent::Note(_)
                | KernContent::Chord(_)
                | KernContent::Grace(_),
            )
            | SpineContent::Dynam(_)
            | SpineContent::Fing(_)
            | SpineContent::Text(_)
            | SpineContent::MxmlHarmony(_)
            | SpineContent::Opaque { .. } => return false,
        }
    }

    saw_slice
}

fn source_lines(document: &KernDocument) -> Vec<usize> {
    (0..document.lines().len())
        .map(|line_index| document.source_line_number(line_index))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParseDialect, ParseError, ParseOptions};

    fn project(input: &str) -> String {
        let document = KernDocument::parse(input).unwrap();
        crate::serialize::omr::to_transcoda_v2_durationless_slices_string(&document)
    }

    #[test]
    fn projects_invisible_rest_across_line_duration_slices() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*k[]\t*k[]",
            "8DD\t4.ryy",
            "8c\t.",
            "8d\t.",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            project(&input),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n8DD\tryy\n8c\tryy\n8d\tryy\n"
        );
    }

    #[test]
    fn carries_invisible_rest_across_zero_duration_grace_lines_as_nulls() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*k[]\t*k[]",
            "4c\t2ryy",
            "16qbb\t.",
            "4d\t.",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            project(&input),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n4c\tryy\n16qbb\t\n4d\tryy\n"
        );
    }

    #[test]
    fn keeps_duration_when_invisible_rest_defines_the_time_slice() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*k[]\t*k[]",
            "2c\t4ryy",
            ".\t4ryy",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            project(&input),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n2c\t4ryy\n\tryy\n"
        );
    }

    #[test]
    fn plain_parse_rejects_durationless_invisible_rest_slices() {
        assert!(KernDocument::parse("**kern\t**kern\n4c\tryy\n").is_err());
    }

    #[test]
    fn transcoda_v2_parse_accepts_and_rehydrates_slices() {
        let mut document = KernDocument::parse_with_options(
            "**kern\t**kern\n4c\tryy\n4d\t.\n",
            ParseOptions {
                dialect: ParseDialect::TranscodaV2,
            },
        )
        .unwrap();

        rehydrate_durationless_invisible_rest_slices(&mut document).unwrap();

        assert_eq!(
            document.to_omr_string(),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n4c\t4ryy\n4d\t.\n"
        );
    }

    #[test]
    fn rehydrate_slice_borrows_carried_visible_duration() {
        let mut document = KernDocument::parse_with_options(
            "**kern\t**kern\n2c\t4d\n.\tryy\n",
            ParseOptions {
                dialect: ParseDialect::TranscodaV2,
            },
        )
        .unwrap();

        rehydrate_durationless_invisible_rest_slices(&mut document).unwrap();

        assert_eq!(
            document.to_omr_string(),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n2c\t4d\n.\t4ryy\n"
        );
    }

    #[test]
    fn rehydrate_mixed_duration_bearing_and_durationless_rests() {
        let mut document = KernDocument::parse_with_options(
            "**kern\t**kern\n2c\t4ryy\n.\tryy\n",
            ParseOptions {
                dialect: ParseDialect::TranscodaV2,
            },
        )
        .unwrap();

        rehydrate_durationless_invisible_rest_slices(&mut document).unwrap();

        assert_eq!(
            document.to_omr_string(),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n2c\t4ryy\n.\t4ryy\n"
        );
    }

    #[test]
    fn transcoda_v2_parse_rejects_slice_only_lines() {
        let error = KernDocument::parse_with_options(
            "**kern\t**kern\n2c\t.\n.\tryy\n",
            ParseOptions {
                dialect: ParseDialect::TranscodaV2,
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ParseError::InvalidDurationlessInvisibleRestSlice {
                line: 3,
                column: 2,
                ..
            }
        ));
    }
}
