use crate::{
    document::Framing,
    normalization::{multi_measure_rests::CompactMultiMeasureRests, NormalizationPass},
    score::{LayerId, Score, StateErrors},
    serialize::omr::{serialize_spine_manipulator_lines, serialize_tandem, OmrSerializeOptions},
    KernContent, KernDocument, KernLine, SpineContent, SpineManipulator, SpineType,
    TandemInterpretation,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LineCapPlan {
    bar_prefix_line_counts: Vec<usize>,
    total_line_count: usize,
}

pub(super) fn compact_for_line_planning(document: &KernDocument) -> KernDocument {
    let mut compacted = document.clone();
    CompactMultiMeasureRests::new(Default::default()).normalize(&mut compacted);
    compacted
}

pub(super) fn planned_transcoda_v2_line_count(
    document: &KernDocument,
) -> Result<usize, StateErrors> {
    Ok(build_line_cap_plan(document)?.total_line_count)
}

pub(super) fn retained_bar_candidates_under_cap(
    document: &KernDocument,
    line_cap: usize,
) -> Result<Vec<usize>, StateErrors> {
    let plan = build_line_cap_plan(document)?;
    Ok(plan
        .bar_prefix_line_counts
        .iter()
        .enumerate()
        .take(plan.bar_prefix_line_counts.len().saturating_sub(1))
        .filter_map(|(index, line_count)| (*line_count <= line_cap).then_some(index + 1))
        .rev()
        .collect())
}

fn build_line_cap_plan(document: &KernDocument) -> Result<LineCapPlan, StateErrors> {
    let line_counts = planned_line_counts(&Score::resolve(document, Framing::Excerpt)?);
    let total_line_count = line_counts.iter().sum();
    let bar_starts = bar_start_indices(document);
    let mut prefix_counts = Vec::with_capacity(bar_starts.len());

    for (bar_index, _) in bar_starts.iter().enumerate() {
        let end = bar_starts
            .get(bar_index + 1)
            .copied()
            .unwrap_or(document.lines().len());
        prefix_counts.push(line_counts[..end].iter().sum());
    }

    Ok(LineCapPlan {
        bar_prefix_line_counts: prefix_counts,
        total_line_count,
    })
}

/// Lines each source line contributes to a Transcoda V2 target, with spine
/// types read from the resolved topology.
fn planned_line_counts(score: &Score) -> Vec<usize> {
    let document = score.document();
    let topology = score.topology();
    let spine_types = |layers: &[LayerId]| {
        layers
            .iter()
            .map(|layer| topology.spine_type(*layer).clone())
            .collect::<Vec<_>>()
    };
    let final_terminal_start = trailing_terminal_start(document);

    document
        .lines()
        .iter()
        .enumerate()
        .map(|(line_index, line)| {
            let active_spines = spine_types(topology.layers_at(line_index));
            match line {
                KernLine::SpineTypeLine(_) | KernLine::LocalCommentLine(_) => 0,
                KernLine::TandemInterpretationLine(tandems) => {
                    usize::from(tandem_line_emits(tandems.tokens(), &active_spines))
                }
                KernLine::SpineManipulatorLine(manipulators) => {
                    if final_terminal_start.is_some_and(|start| line_index >= start) {
                        0
                    } else {
                        serialize_spine_manipulator_lines(
                            manipulators.tokens(),
                            &active_spines,
                            &spine_types(topology.layers_after(line_index)),
                        )
                        .len()
                    }
                }
                KernLine::Barline(barline) => {
                    usize::from(barline.system_break_before.is_some())
                        + usize::from(has_kern_spine(&active_spines))
                }
                KernLine::Data(contents) => usize::from(data_line_emits(contents, &active_spines)),
            }
        })
        .collect()
}

fn bar_start_indices(document: &KernDocument) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut pending_boundary = None;
    let mut active_start = None;

    for (index, line) in document.lines().iter().enumerate() {
        match line {
            KernLine::Barline(_) => {
                pending_boundary = Some(index);
                active_start = None;
            }
            KernLine::Data(_) if active_start.is_none() => {
                let start = pending_boundary.take().unwrap_or(index);
                active_start = Some(start);
                starts.push(start);
            }
            KernLine::SpineManipulatorLine(manipulators)
                if is_terminal_line(manipulators.tokens()) =>
            {
                active_start = None;
            }
            _ => {}
        }
    }

    starts
}

fn trailing_terminal_start(document: &KernDocument) -> Option<usize> {
    let mut start = document.lines().len();
    while start > 0 {
        match &document.lines()[start - 1] {
            KernLine::SpineManipulatorLine(manipulators)
                if is_terminal_line(manipulators.tokens()) =>
            {
                start -= 1;
            }
            _ => break,
        }
    }
    (start < document.lines().len()).then_some(start)
}

fn tandem_line_emits(tandems: &[TandemInterpretation], active_spines: &[SpineType]) -> bool {
    tandems
        .iter()
        .enumerate()
        .filter(|(index, _)| is_kern_spine(active_spines, *index))
        .any(|(_, tandem)| {
            serialize_tandem(tandem, &OmrSerializeOptions::default())
                .as_deref()
                .is_some_and(|token| token != "*")
        })
}

fn data_line_emits(contents: &[SpineContent], active_spines: &[SpineType]) -> bool {
    contents
        .iter()
        .enumerate()
        .filter(|(index, _)| is_kern_spine(active_spines, *index))
        .any(|(_, content)| {
            matches!(
                content,
                SpineContent::Kern(
                    KernContent::Rest(_)
                        | KernContent::MultiMeasureRest(_)
                        | KernContent::Note(_)
                        | KernContent::Chord(_)
                        | KernContent::Grace(_)
                )
            )
        })
}

fn has_kern_spine(active_spines: &[SpineType]) -> bool {
    active_spines
        .iter()
        .any(|spine_type| matches!(spine_type, SpineType::Kern))
}

fn is_kern_spine(active_spines: &[SpineType], index: usize) -> bool {
    active_spines
        .get(index)
        .is_some_and(|spine_type| matches!(spine_type, SpineType::Kern))
}

fn is_terminal_line(manipulators: &[SpineManipulator]) -> bool {
    manipulators
        .iter()
        .all(|manipulator| matches!(manipulator, SpineManipulator::Terminate))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> KernDocument {
        KernDocument::parse(input).unwrap()
    }

    #[test]
    fn planned_count_omits_spine_wrappers_comments_and_final_terminal() {
        let document = parse("!!!COM: Example\n**kern\n*clefG2\n*k[]\n!! comment\n4c\n*-\n");

        assert_eq!(planned_transcoda_v2_line_count(&document).unwrap(), 3);
    }

    #[test]
    fn planned_count_includes_system_breaks() {
        let document = parse("**kern\n*clefG2\n*k[]\n4c\n!!LO:LB:g=original\n=\n4d\n*-\n");

        assert_eq!(planned_transcoda_v2_line_count(&document).unwrap(), 6);
    }

    #[test]
    fn line_planning_compacts_multi_measure_rests() {
        let document = parse("**kern\n*M3/4\n=\n2.r\n=\n2.r\n=\n4c\n=\n*-\n");
        let compacted = compact_for_line_planning(&document);

        assert_eq!(planned_transcoda_v2_line_count(&compacted).unwrap(), 8);
        assert_eq!(
            retained_bar_candidates_under_cap(&compacted, 6).unwrap(),
            vec![1]
        );
    }

    #[test]
    fn retained_candidates_are_largest_first_and_leave_a_suffix_bar() {
        let document = parse("**kern\n*clefG2\n*k[]\n=\n4c\n=\n4d\n=\n4e\n=\n*-\n");

        assert_eq!(
            retained_bar_candidates_under_cap(&document, 6).unwrap(),
            vec![2, 1]
        );
    }
}
