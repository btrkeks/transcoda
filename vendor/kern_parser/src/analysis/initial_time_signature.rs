//! Initial meter resolution from declarations or rhythmic content.

use thiserror::Error;

use crate::{
    measure::{kern_content_duration, MeasureDuration, MeasureLocation},
    score::{LayerOrigin, Score},
    InitialTimeSignatureError, KernDocument, KernLine, SpineContent, SpineType, TimeSignature,
};

use super::measure_spans;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum InitialTimeSignatureResolutionError {
    #[error("{0}")]
    InitialDeclaration(#[from] InitialTimeSignatureError),
    #[error("cannot infer initial time signature: document has no second measure")]
    NoSecondMeasure,
    #[error("cannot infer initial time signature from second measure: no usable rhythmic content")]
    NoUsableRhythmicContent,
    #[error("cannot infer initial time signature from second measure: {location}: {message}")]
    InvalidMeasureContent { location: String, message: String },
    #[error(
        "cannot infer initial time signature from second measure: mixed kern spine durations; expected {expected}, found {found}"
    )]
    MixedKernSpineDurations { expected: String, found: String },
    #[error(
        "cannot infer initial time signature from second measure: duration {duration} cannot be represented as a time signature"
    )]
    UnrepresentableDuration { duration: String },
}

pub fn resolve_initial_time_signature(
    score: &Score<'_>,
) -> Result<TimeSignature, InitialTimeSignatureResolutionError> {
    resolve_initial_time_signature_in(score, score.document())
}

/// `document` is the score's own document or a copy that changed data tokens
/// only, so the score's layout still applies.
pub(crate) fn resolve_initial_time_signature_in(
    score: &Score<'_>,
    document: &KernDocument,
) -> Result<TimeSignature, InitialTimeSignatureResolutionError> {
    if let Some(time_signature) = document.declared_initial_time_signature()? {
        return Ok(time_signature);
    }

    infer_initial_time_signature_from_second_measure(score, document)
}

/// Every `**kern` layer that sounds in the second measure must fill the same
/// duration. A split copies its parent's elapsed time; merged layers must
/// agree.
fn infer_initial_time_signature_from_second_measure(
    score: &Score<'_>,
    document: &KernDocument,
) -> Result<TimeSignature, InitialTimeSignatureResolutionError> {
    let topology = score.topology();
    let spans = measure_spans(document);
    let span = spans
        .get(1)
        .or_else(|| (spans.len() == 1).then(|| &spans[0]))
        .cloned()
        .ok_or(InitialTimeSignatureResolutionError::NoSecondMeasure)?;
    let is_kern = |layer| *topology.spine_type(layer) == SpineType::Kern;
    let mut elapsed = vec![MeasureDuration::zero(); topology.layer_count()];

    for (line_index, line) in document.lines().iter().enumerate().take(span.content.end) {
        let source_line = document.source_line_number(line_index);
        match line {
            KernLine::SpineManipulatorLine(_) => {
                let before = topology.layers_at(line_index);
                for &layer in topology.layers_after(line_index) {
                    if before.contains(&layer) {
                        continue;
                    }
                    elapsed[layer.index()] = match &topology.layer(layer).origin {
                        LayerOrigin::SplitOf(parent) => elapsed[parent.index()],
                        LayerOrigin::MergeOf(inputs) => {
                            let first = elapsed[inputs[0].index()];
                            if let Some(input) = inputs
                                .iter()
                                .find(|input| is_kern(**input) && elapsed[input.index()] != first)
                            {
                                return Err(
                                    InitialTimeSignatureResolutionError::MixedKernSpineDurations {
                                        expected: first.display(),
                                        found: elapsed[input.index()].display(),
                                    },
                                );
                            }
                            first
                        }
                        LayerOrigin::Exclusive => MeasureDuration::zero(),
                    };
                }
            }
            KernLine::Data(contents) if span.content.contains(&line_index) => {
                for (column_index, (&layer, content)) in topology
                    .layers_at(line_index)
                    .iter()
                    .zip(contents)
                    .enumerate()
                {
                    if !is_kern(layer) {
                        continue;
                    }
                    let location = MeasureLocation {
                        source_line,
                        column: Some(column_index + 1),
                    };
                    let SpineContent::Kern(content) = content else {
                        return invalid_measure_content(
                            location,
                            "kern spine contains non-kern data content",
                        );
                    };
                    let duration = kern_content_duration(content).map_err(|error| {
                        InitialTimeSignatureResolutionError::InvalidMeasureContent {
                            location: location.to_string(),
                            message: error.to_string(),
                        }
                    })?;
                    elapsed[layer.index()] += duration;
                }
            }
            _ => {}
        }
    }

    let last = span.content.end.saturating_sub(1);
    let inferred = topology
        .layers_after(last)
        .iter()
        .filter(|layer| is_kern(**layer))
        .map(|layer| elapsed[layer.index()])
        .filter(|elapsed| !elapsed.is_zero())
        .try_fold(None::<MeasureDuration>, |expected, total| match expected {
            Some(expected) if expected != total => Err(
                InitialTimeSignatureResolutionError::MixedKernSpineDurations {
                    expected: expected.display(),
                    found: total.display(),
                },
            ),
            Some(expected) => Ok(Some(expected)),
            None => Ok(Some(total)),
        })?
        .ok_or(InitialTimeSignatureResolutionError::NoUsableRhythmicContent)?;

    inferred.to_time_signature_prefer_quarter().ok_or_else(|| {
        InitialTimeSignatureResolutionError::UnrepresentableDuration {
            duration: inferred.display(),
        }
    })
}

fn invalid_measure_content<T>(
    location: MeasureLocation,
    message: impl Into<String>,
) -> Result<T, InitialTimeSignatureResolutionError> {
    Err(InitialTimeSignatureResolutionError::InvalidMeasureContent {
        location: location.to_string(),
        message: message.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> KernDocument {
        KernDocument::parse(input).expect("test input parses")
    }

    fn resolve(
        document: &KernDocument,
    ) -> Result<TimeSignature, InitialTimeSignatureResolutionError> {
        let score = Score::resolve(document, crate::document::Framing::Excerpt).unwrap();
        resolve_initial_time_signature(&score)
    }

    fn time_signature(numerator: u32, denominator: u32) -> TimeSignature {
        TimeSignature {
            numerator,
            denominator,
        }
    }

    #[test]
    fn declared_initial_time_signature_wins_over_second_measure() {
        let document = parse("**kern\n*M3/4\n4c\n=\n4d\n4e\n4f\n4g\n=\n*-\n");

        assert_eq!(resolve(&document).unwrap(), time_signature(3, 4));
    }

    #[test]
    fn infers_four_four_from_second_measure_after_pickup() {
        let document = parse("**kern\n4c\n=\n4d\n4e\n4f\n4g\n=\n*-\n");

        assert_eq!(resolve(&document).unwrap(), time_signature(4, 4));
    }

    #[test]
    fn infers_three_four_from_second_measure_after_pickup() {
        let document = parse("**kern\n4c\n=\n4d\n4e\n4f\n=\n*-\n");

        assert_eq!(resolve(&document).unwrap(), time_signature(3, 4));
    }

    #[test]
    fn infers_second_measure_through_interior_spine_split() {
        let document = parse(
            "**kern\t**kern\n\
4c\t4e\n\
=\t=\n\
16cLL\t4e\n\
16d\t.\n\
16e\t.\n\
16fJJ\t.\n\
*\t*^\n\
16gLL\t4a\t16bLL\n\
16a\t.\t16cc\n\
16b\t.\t16dd\n\
16ccJJ\t.\t16eeJJ\n\
=\t=\t=\n\
*-\t*-\t*-\n",
        );

        assert_eq!(resolve(&document).unwrap(), time_signature(2, 4));
    }

    #[test]
    fn infers_from_only_measure_without_second_measure() {
        let document = parse("**kern\n4c\n4d\n4e\n4f\n=\n*-\n");

        assert_eq!(resolve(&document).unwrap(), time_signature(4, 4));
    }

    #[test]
    fn errors_on_mixed_second_measure_spine_durations() {
        let document = parse("**kern\t**kern\n4c\t4e\n=\t=\n4d\t4f\n4e\t2g\n=\t=\n*-\t*-\n");
        let error = resolve(&document).unwrap_err();

        assert!(matches!(
            error,
            InitialTimeSignatureResolutionError::MixedKernSpineDurations { .. }
        ));
    }
}
