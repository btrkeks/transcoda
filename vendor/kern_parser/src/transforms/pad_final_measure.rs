//! Invisible structural padding for incomplete final measures.
//! Rational deficits remain exact without asking the engraver to approximate a rest glyph.

use std::collections::BTreeSet;

use crate::{
    analysis::resolve_initial_time_signature,
    document::Framing,
    measure::MeasureDuration,
    score::{LayerOrigin, Score, StateKey},
    BarLength, KernContent, KernDocument, KernLine, Rest, SpineContent, SpineManipulator,
    SpineType,
};

pub fn pad_final_measure_with_rests(document: &KernDocument) -> KernDocument {
    let Some(padding) = final_measure_padding(document) else {
        return document.clone();
    };

    let mut document = document.clone();
    let source = padding
        .insert_index
        .checked_sub(1)
        .and_then(|index| document.source(index));
    document.insert_line(
        padding.insert_index,
        KernLine::Data(padding.contents),
        source,
    );
    document
}

#[derive(Debug)]
struct FinalMeasurePadding {
    insert_index: usize,
    contents: Vec<SpineContent>,
}

#[derive(Debug, Clone)]
struct MeasureSpine {
    spine_type: SpineType,
    meter: BarLength,
    elapsed: MeasureDuration,
}

/// Time elapsed in the final measure per layer, read from the resolved
/// score. Meter is the track's (D5); a meter declaration in any subspine
/// restarts the measure of every layer of its track unless that layer has
/// just filled the new bar.
fn final_measure_padding(document: &KernDocument) -> Option<FinalMeasurePadding> {
    let score = Score::resolve(document, Framing::Excerpt).ok()?;
    let topology = score.topology();
    let scan_end = trailing_terminal_start(document);
    let initial_meter = initial_meter(&score)?;
    let meter_at = |line: usize, column: usize| {
        let track = if line < document.lines().len() {
            score.state_at(line, column).track
        } else {
            score.state_after_last(column).track
        };
        track.meter.bar_length().unwrap_or(initial_meter)
    };
    let meter_events = score
        .events()
        .iter()
        .filter(|event| event.key == StateKey::Meter)
        .map(|event| {
            let layer = topology.layer_at(event.line_index, event.column);
            (event.line_index, topology.track_of(layer))
        })
        .collect::<BTreeSet<_>>();

    let mut elapsed = vec![MeasureDuration::zero(); topology.layer_count()];
    let mut final_barline = None;
    let mut last_data_index = None;
    for (line_index, line) in document.lines()[..scan_end].iter().enumerate() {
        let layers = topology.layers_at(line_index);
        match line {
            KernLine::TandemInterpretationLine(_) => {
                for (column, &layer) in layers.iter().enumerate() {
                    if !meter_events.contains(&(line_index, topology.track_of(layer))) {
                        continue;
                    }
                    let meter = meter_at(line_index + 1, column);
                    if BarLength::Fixed(elapsed[layer.index()]) != meter {
                        elapsed[layer.index()] = MeasureDuration::zero();
                    }
                }
            }
            KernLine::SpineManipulatorLine(_) => {
                for &layer in topology.layers_after(line_index) {
                    match &topology.layer(layer).origin {
                        LayerOrigin::SplitOf(parent) => {
                            elapsed[layer.index()] = elapsed[parent.index()];
                        }
                        LayerOrigin::MergeOf(inputs) => {
                            let first = elapsed[inputs[0].index()];
                            if inputs.iter().any(|input| elapsed[input.index()] != first) {
                                return None;
                            }
                            elapsed[layer.index()] = first;
                        }
                        LayerOrigin::Exclusive => {}
                    }
                }
            }
            KernLine::Barline(_) => {
                final_barline = Some((line_index, elapsed.clone()));
                elapsed.fill(MeasureDuration::zero());
            }
            KernLine::Data(contents) => {
                last_data_index = Some(line_index);
                for (&layer, content) in layers.iter().zip(contents) {
                    if let SpineContent::Kern(content) = content {
                        if let Ok(duration) = content.duration() {
                            elapsed[layer.index()] += duration;
                        }
                    }
                }
            }
            KernLine::SpineTypeLine(_) | KernLine::LocalCommentLine(_) => {}
        }
    }

    let (insert_index, elapsed) = match final_barline {
        Some((line_index, at_barline))
            if last_data_index.is_some_and(|data_index| line_index > data_index) =>
        {
            (line_index, at_barline)
        }
        _ => (scan_end, elapsed),
    };
    let layers = if insert_index < document.lines().len() {
        topology.layers_at(insert_index)
    } else {
        topology.layers_after_last()
    };
    let spines = layers
        .iter()
        .enumerate()
        .map(|(column, &layer)| MeasureSpine {
            spine_type: topology.spine_type(layer).clone(),
            meter: meter_at(insert_index, column),
            elapsed: elapsed[layer.index()],
        })
        .collect::<Vec<_>>();

    padding_contents(insert_index, &spines)
}

fn initial_meter(score: &Score) -> Option<BarLength> {
    let time_signature = resolve_initial_time_signature(score).ok()?;
    Some(BarLength::Fixed(MeasureDuration::from_time_signature(
        &time_signature,
    )))
}

fn trailing_terminal_start(document: &KernDocument) -> usize {
    let mut index = document.lines().len();
    while index > 0 && is_terminal_line(&document.lines()[index - 1]) {
        index -= 1;
    }
    index
}

fn is_terminal_line(line: &KernLine) -> bool {
    matches!(
        line,
        KernLine::SpineManipulatorLine(manipulators)
            if manipulators
                .tokens()
                .iter()
                .all(|token| matches!(token, SpineManipulator::Terminate))
    )
}

fn padding_contents(insert_index: usize, spines: &[MeasureSpine]) -> Option<FinalMeasurePadding> {
    let mut saw_padding = false;
    let mut contents = Vec::with_capacity(spines.len());

    for spine in spines {
        let content = match spine.spine_type {
            SpineType::Kern => {
                // A free-meter (`*MX`) measure has no length to fill.
                if let Some(missing) = spine
                    .meter
                    .fixed()
                    .and_then(|meter| spine.elapsed.missing_from(meter))
                {
                    saw_padding = true;
                    SpineContent::Kern(KernContent::Rest(Rest {
                        duration: missing.into_duration()?,
                        cross_staff: None,
                        placement: None,
                        invisible: true,
                        breath: None,
                        fermata: None,
                        slurs: Vec::new(),
                        phrases: Vec::new(),
                    }))
                } else {
                    SpineContent::Kern(KernContent::Null)
                }
            }
            SpineType::Dynam => SpineContent::Dynam(crate::DynamContent::Null),
            SpineType::Fing => SpineContent::Fing(crate::FingContent::Null),
            SpineType::Text => SpineContent::Text(crate::TextContent::Null),
            SpineType::MxmlChords => SpineContent::MxmlHarmony(crate::MxmlHarmonyContent::Null),
            SpineType::Other(ref raw) => SpineContent::Opaque {
                spine_type: SpineType::Other(raw.clone()),
                raw: crate::RawToken(".".to_owned()),
            },
        };
        contents.push(content);
    }

    saw_padding.then_some(FinalMeasurePadding {
        insert_index,
        contents,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pad(input: &str) -> String {
        pad_final_measure_with_rests(&KernDocument::parse(input).unwrap()).to_compact_omr_string()
    }

    #[test]
    fn complete_final_measure_is_unchanged() {
        let input = "**kern\n*M4/4\n=1\n1c\n==\n*-\n";
        let document = KernDocument::parse(input).unwrap();
        assert_eq!(pad_final_measure_with_rests(&document), document);
    }

    #[test]
    fn meter_written_in_one_subspine_sets_the_bar_of_its_whole_staff() {
        assert_eq!(
            pad("**kern\n*M4/4\n*^\n*M3/4\t*\n=1\t=1\n4c\t4e\n*v\t*v\n==\n*-\n"),
            "*clefG2\n*k[]\n*M4/4\n*^\n*M3/4\t*\n=\t=\n4c\t4e\n*v\t*v\n2ryy\n==\n"
        );
    }

    #[test]
    fn pads_incomplete_final_measure_before_final_barline() {
        assert_eq!(
            pad("**kern\n*M4/4\n=1\n4c\n==\n*-\n"),
            "*clefG2\n*k[]\n*M4/4\n4c\n4%3ryy\n==\n"
        );
    }

    #[test]
    fn pads_incomplete_tail_before_terminal_line_without_final_barline() {
        assert_eq!(
            pad("**kern\n*M4/4\n=1\n2c\n*-\n"),
            "*clefG2\n*k[]\n*M4/4\n2c\n2ryy\n"
        );
    }

    #[test]
    fn pads_multiple_kern_spines_and_preserves_width() {
        assert_eq!(
            pad("**kern\t**kern\n*M4/4\t*M4/4\n=1\t=1\n2c\t4e\n==\t==\n*-\t*-\n"),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n2c\t4e\n2ryy\t4%3ryy\n==\t==\n"
        );
    }

    #[test]
    fn pads_fingering_spines_with_null_data() {
        let padded = pad_final_measure_with_rests(
            &KernDocument::parse("**kern\t**fing\n*M4/4\t*\n=1\t=1\n2c\t1\n*-\t*-\n").unwrap(),
        );

        assert!(padded.lines().iter().any(|line| matches!(
            line,
            KernLine::Data(contents)
                if matches!(contents.as_slice(), [
                    SpineContent::Kern(KernContent::Rest(_)),
                    SpineContent::Fing(crate::FingContent::Null)
                ])
        )));
    }

    #[test]
    fn final_meter_change_controls_padding_duration() {
        assert_eq!(
            pad("**kern\n*M4/4\n=1\n1c\n=\n*M3/4\n4d\n==\n*-\n"),
            "*clefG2\n*k[]\n*M4/4\n1c\n=\n*M3/4\n4d\n2ryy\n==\n"
        );
    }

    #[test]
    fn final_meter_symbol_change_controls_padding_duration() {
        assert_eq!(
            pad("**kern\n*M3/4\n=1\n2.c\n=\n*met(c)\n4d\n==\n*-\n"),
            "*clefG2\n*k[]\n*M3/4\n2.c\n=\n*met(c)\n4d\n4%3ryy\n==\n"
        );
    }

    #[test]
    fn does_not_pad_a_free_meter_final_measure() {
        let document = KernDocument::parse("**kern\n*M4/4\n=1\n1c\n=\n*MX\n4d\n==\n*-\n").unwrap();

        assert_eq!(pad_final_measure_with_rests(&document), document);
    }

    #[test]
    fn keeps_non_binary_deficit_invisible() {
        assert_eq!(
            pad("**kern\n*M4/4\n=1\n5%3c\n==\n*-\n"),
            "*clefG2\n*k[]\n*M4/4\n5%3c\n5%2ryy\n==\n"
        );
    }
}
