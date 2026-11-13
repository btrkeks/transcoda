//! Pickup-aware measure numbering used by the renderer.
use crate::{
    analysis::{measure_spans, score_line_durations, DataLineDurationError, MeasureSpan},
    document::Framing,
    score::{Score, StateErrors},
    KernContent, KernDocument, KernLine, MusicalDuration, SpineContent, SpineType,
};
use thiserror::Error;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrittenMeasure(u32);
impl WrittenMeasure {
    pub fn new(n: u32) -> Self {
        Self(n)
    }
    pub fn number(self) -> u32 {
        self.0
    }
}
#[derive(Debug, Error)]
pub enum WrittenMeasureError {
    #[error("the score does not resolve: {0}")]
    State(StateErrors),
    #[error(transparent)]
    Timing(#[from] DataLineDurationError),
    #[error("musical addresses require two initial kern spines, found {found}")]
    HandRoots { found: usize },
    #[error("line {line}: {message}")]
    Structure { line: usize, message: String },
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenMeasureSpan {
    pub measure: WrittenMeasure,
    pub span: MeasureSpan,
}
pub fn written_measure_numbers(
    document: &KernDocument,
) -> Result<Vec<WrittenMeasureSpan>, WrittenMeasureError> {
    let score = resolve(document)?;
    let durations = score_line_durations(&score)?;
    let contexts = contexts(&score)?;
    number_measures(document, &durations, &contexts)
}
fn resolve(document: &KernDocument) -> Result<Score<'_>, WrittenMeasureError> {
    Score::resolve(document, Framing::Excerpt).map_err(WrittenMeasureError::State)
}
fn number_measures(
    document: &KernDocument,
    durations: &[crate::analysis::DataLineDuration],
    contexts: &[Vec<SpineContext>],
) -> Result<Vec<WrittenMeasureSpan>, WrittenMeasureError> {
    let spans = measure_spans(document);
    let Some(first) = spans.first() else {
        return Ok(Vec::new());
    };
    let first_lines: Vec<_> = durations
        .iter()
        .filter(|d| first.content.contains(&d.line_index))
        .collect();
    let mut meters = first_lines.iter().flat_map(|d| {
        contexts[d.line_index]
            .iter()
            .filter(|c| c.spine_type == SpineType::Kern)
            .map(|c| c.meter)
    });
    let meter = meters.next().flatten();
    let consistent_meter = meter.filter(|meter| meters.all(|other| other == Some(*meter)));
    let duration = first_lines
        .iter()
        .fold(MusicalDuration::zero(), |total, d| total + d.duration);
    let mut number: u32 = if consistent_meter.is_some_and(|meter| duration < meter) {
        0
    } else {
        1
    };
    spans
        .into_iter()
        .map(|span| {
            let measure = WrittenMeasure::new(number);
            number = number
                .checked_add(bar_count(document, &span))
                .ok_or_else(|| WrittenMeasureError::Structure {
                    line: document.source_line_number(span.content.start),
                    message: "written bar number overflow".into(),
                })?;
            Ok(WrittenMeasureSpan { measure, span })
        })
        .collect()
}
fn bar_count(document: &KernDocument, span: &MeasureSpan) -> u32 {
    document.lines()[span.content.clone()]
        .iter()
        .filter_map(|line| {
            let KernLine::Data(contents) = line else {
                return None;
            };
            contents
                .iter()
                .filter_map(|content| match content {
                    SpineContent::Kern(KernContent::MultiMeasureRest(rest)) => {
                        Some(rest.measure_count)
                    }
                    _ => None,
                })
                .max()
        })
        .max()
        .unwrap_or(1)
}
struct SpineContext {
    spine_type: SpineType,
    meter: Option<MusicalDuration>,
}
fn contexts(score: &Score<'_>) -> Result<Vec<Vec<SpineContext>>, WrittenMeasureError> {
    let topology = score.topology();
    let count = topology
        .track_ids()
        .filter(|&t| topology.track(t).spine_type == SpineType::Kern)
        .count();
    if count != 2 {
        return Err(WrittenMeasureError::HandRoots { found: count });
    }
    Ok((0..score.document().lines().len())
        .map(|line_index| {
            topology
                .layers_at(line_index)
                .iter()
                .enumerate()
                .map(|(column, &layer)| SpineContext {
                    spine_type: topology.spine_type(layer).clone(),
                    meter: score
                        .state_at(line_index, column)
                        .track
                        .meter
                        .measure_duration(),
                })
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pickup_starts_at_zero_and_a_full_measure_starts_at_one() {
        for (first, expected) in [("4C\t4c", vec![0, 1]), ("1C\t1c", vec![1, 2])] {
            let document = KernDocument::parse(&format!(
                "**kern\t**kern\n*M4/4\t*M4/4\n{first}\n=\t=\n1D\t1d\n*-\t*-\n"
            ))
            .unwrap();
            let numbers: Vec<_> = written_measure_numbers(&document)
                .unwrap()
                .iter()
                .map(|entry| entry.measure.number())
                .collect();
            assert_eq!(numbers, expected);
        }
    }
}
