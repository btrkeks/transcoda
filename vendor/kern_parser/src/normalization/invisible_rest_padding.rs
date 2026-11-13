//! Normalize unobservable invisible-rest padding.
//!
//! Some sources encode pickup or page-boundary padding as many tiny invisible
//! rests. Those lines do not correspond to rendered notation and make model
//! labels unnecessarily hard. This pass deletes only outer-boundary padding and
//! compacts remaining consecutive invisible-only runs inside measures.

use std::ops::Range;

use crate::{
    analysis::{measure_spans, MeasureSpan},
    KernContent, KernDocument, KernLine, SpineContent,
};

use super::{
    invisible_rest_tokens::{is_non_kern_data_null, is_plain_invisible_rest},
    NormalizationPass,
};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NormalizeInvisibleRestPadding;

impl NormalizeInvisibleRestPadding {
    pub fn new() -> Self {
        Self
    }
}

impl NormalizationPass for NormalizeInvisibleRestPadding {
    fn normalize(&self, document: &mut KernDocument) {
        normalize_invisible_rest_padding(document);
    }
}

pub fn normalize_invisible_rest_padding(document: &mut KernDocument) {
    remove_boundary_padding(document);
    compact_interior_padding(document);
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InvisibleRestRun {
    range: Range<usize>,
    duration: crate::MusicalDuration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PaddingLine {
    duration: crate::MusicalDuration,
}

fn remove_boundary_padding(document: &mut KernDocument) {
    let spans = measure_spans(document);
    let mut removals = Vec::new();

    if let Some(first) = spans.first() {
        removals.extend(boundary_padding_runs(document, first));
    }
    if spans.len() > 1 {
        if let Some(last) = spans.last() {
            removals.extend(boundary_padding_runs(document, last));
        }
    }

    remove_ranges(document, removals);
}

fn boundary_padding_runs(document: &KernDocument, span: &MeasureSpan) -> Vec<Range<usize>> {
    padding_runs(document, span)
        .into_iter()
        .filter(|run| {
            (run.range.start == span.content.start || run.range.end == span.content.end)
                && run.range != span.content
                && span_has_visible_data_outside_run(document, span, &run.range)
        })
        .map(|run| run.range)
        .collect()
}

fn compact_interior_padding(document: &mut KernDocument) {
    let spans = measure_spans(document);
    let runs = spans
        .iter()
        .flat_map(|span| padding_runs(document, span))
        .filter(|run| run.range.len() > 1)
        .collect::<Vec<_>>();

    for run in runs.into_iter().rev() {
        compact_run(document, run);
    }
}

fn compact_run(document: &mut KernDocument, run: InvisibleRestRun) {
    let Some(duration) = run.duration.into_duration_prefer_dots() else {
        return;
    };

    let KernLine::Data(contents) = &mut document.lines_mut()[run.range.start] else {
        return;
    };
    for content in contents {
        let SpineContent::Kern(KernContent::Rest(rest)) = content else {
            continue;
        };
        if is_plain_invisible_rest(rest) {
            rest.duration = duration.clone();
        }
    }

    document.drain_lines((run.range.start + 1)..run.range.end);
}

fn remove_ranges(document: &mut KernDocument, ranges: Vec<Range<usize>>) {
    for range in merged_ranges(ranges).into_iter().rev() {
        document.drain_lines(range);
    }
}

fn merged_ranges(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.retain(|range| range.start < range.end);
    ranges.sort_by_key(|range| (range.start, range.end));

    let mut merged: Vec<Range<usize>> = Vec::new();
    for range in ranges {
        let Some(last) = merged.last_mut() else {
            merged.push(range);
            continue;
        };
        if range.start <= last.end {
            last.end = last.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    merged
}

fn padding_runs(document: &KernDocument, span: &MeasureSpan) -> Vec<InvisibleRestRun> {
    let mut runs = Vec::new();
    let mut index = span.content.start;

    while index < span.content.end {
        let Some(first) = padding_line(document.lines().get(index)) else {
            index += 1;
            continue;
        };

        let start = index;
        let mut duration = first.duration;
        index += 1;

        while index < span.content.end {
            let Some(next) = padding_line(document.lines().get(index)) else {
                break;
            };
            duration += next.duration;
            index += 1;
        }

        runs.push(InvisibleRestRun {
            range: start..index,
            duration,
        });
    }

    runs
}

fn padding_line(line: Option<&KernLine>) -> Option<PaddingLine> {
    let KernLine::Data(contents) = line? else {
        return None;
    };

    let mut duration = None;
    for content in contents {
        match content {
            SpineContent::Kern(KernContent::Rest(rest)) if is_plain_invisible_rest(rest) => {
                let rest_duration = crate::MusicalDuration::from_duration(&rest.duration);
                match duration {
                    Some(duration) if duration != rest_duration => return None,
                    Some(_) => {}
                    None => duration = Some(rest_duration),
                }
            }
            content if is_non_kern_data_null(content) => {}
            SpineContent::Kern(_)
            | SpineContent::Dynam(_)
            | SpineContent::Fing(_)
            | SpineContent::Text(_)
            | SpineContent::MxmlHarmony(_)
            | SpineContent::Opaque { .. } => return None,
        }
    }

    duration.map(|duration| PaddingLine { duration })
}

fn span_has_visible_data_outside_run(
    document: &KernDocument,
    span: &MeasureSpan,
    run: &Range<usize>,
) -> bool {
    (span.content.start..span.content.end)
        .filter(|index| !run.contains(index))
        .any(|index| {
            matches!(document.lines().get(index), Some(KernLine::Data(_)))
                && padding_line(document.lines().get(index)).is_none()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{types::DurationValue, KernDocument};

    fn normalize(input: &str) -> KernDocument {
        let mut document = KernDocument::parse(input).unwrap();
        NormalizeInvisibleRestPadding::new().normalize(&mut document);
        document
    }

    fn invisible_rest_durations(document: &KernDocument) -> Vec<(u32, u8)> {
        document
            .lines()
            .iter()
            .flat_map(|line| match line {
                KernLine::Data(contents) => contents.iter().collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .filter_map(|content| match content {
                SpineContent::Kern(KernContent::Rest(rest)) if rest.invisible => {
                    match rest.duration.value {
                        DurationValue::Reciprocal(value) => {
                            Some((value, rest.duration.augmentation_dots))
                        }
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn drops_trailing_padding_in_initial_pickup_measure() {
        let input = [
            "**kern\t**dynam\t**kern\t**dynam",
            "*M2/4\t*\t*M2/4\t*",
            "8c\t.\t8e\t.",
            "32ryy\t.\t32ryy\t.",
            "32ryy\t.\t32ryy\t.",
            "32ryy\t.\t32ryy\t.",
            "32ryy\t.\t32ryy\t.",
            "32ryy\t.\t32ryy\t.",
            "32ryy\t.\t32ryy\t.",
            "32ryy\t.\t32ryy\t.",
            "32ryy\t.\t32ryy\t.",
            "32ryy\t.\t32ryy\t.",
            "32ryy\t.\t32ryy\t.",
            "32ryy\t.\t32ryy\t.",
            "32ryy\t.\t32ryy\t.",
            "=\t=\t=\t=",
            "4d\t.\t4f\t.",
            "*-\t*-\t*-\t*-",
        ]
        .join("\n");

        let normalized = normalize(&input);

        assert!(!normalized.lines().iter().any(|line| {
            matches!(
                line,
                KernLine::Data(contents)
                    if contents.iter().any(|content| matches!(
                        content,
                        SpineContent::Kern(KernContent::Rest(rest)) if rest.invisible
                    ))
            )
        }));
    }

    #[test]
    fn compacts_trailing_padding_in_interior_measure() {
        let input = [
            "**kern", "*M2/4", "4c", "=", "8d", "32ryy", "32ryy", "32ryy", "32ryy", "32ryy",
            "32ryy", "32ryy", "32ryy", "32ryy", "32ryy", "32ryy", "32ryy", "=", "4e", "*-",
        ]
        .join("\n");

        let normalized = normalize(&input);

        assert_eq!(invisible_rest_durations(&normalized), vec![(4, 1)]);
    }

    #[test]
    fn drops_trailing_padding_in_final_measure() {
        let input = [
            "**kern", "*M2/4", "4c", "=", "8d", "32ryy", "32ryy", "32ryy", "32ryy", "32ryy",
            "32ryy", "32ryy", "32ryy", "32ryy", "32ryy", "32ryy", "32ryy", "*-",
        ]
        .join("\n");

        let normalized = normalize(&input);

        assert!(invisible_rest_durations(&normalized).is_empty());
    }

    #[test]
    fn drops_trailing_padding_in_final_measure_with_in_measure_tandem() {
        let input = [
            "**kern", "*M2/4", "4c", "=", "4d", "*clefF4", "8e", "8ryy", "==", "*-",
        ]
        .join("\n");

        let normalized = normalize(&input);

        assert!(invisible_rest_durations(&normalized).is_empty());
        assert!(normalized
            .lines()
            .iter()
            .any(|line| matches!(line, KernLine::TandemInterpretationLine(_))));
    }

    #[test]
    fn does_not_compact_across_non_data_lines() {
        let input = [
            "**kern", "*M2/4", "4c", "=", "32ryy", "*clefG2", "32ryy", "=", "4d", "*-",
        ]
        .join("\n");

        let normalized = normalize(&input);

        assert_eq!(
            invisible_rest_durations(&normalized),
            vec![(32, 0), (32, 0)]
        );
    }

    #[test]
    fn leaves_kern_null_continuations_unchanged() {
        let input = [
            "**kern\t**kern",
            "*M2/4\t*M2/4",
            "2c\t4e",
            ".\t32ryy",
            ".\t32ryy",
            "*-\t*-",
        ]
        .join("\n");

        let normalized = normalize(&input);

        assert_eq!(normalized, KernDocument::parse(&input).unwrap());
    }
}
