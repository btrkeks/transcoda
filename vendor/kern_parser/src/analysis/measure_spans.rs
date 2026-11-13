//! Logical measure spans and conservative visual-empty classification.

use std::ops::Range;

use crate::{
    DynamContent, FingContent, KernContent, KernDocument, KernLine, MxmlHarmonyContent,
    SpineContent, TextContent,
};

/// Half-open line span for one logical measure body and its adjacent barlines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasureSpan {
    pub leading_barline: Option<usize>,
    pub content: Range<usize>,
    pub trailing_barline: Option<usize>,
}

impl MeasureSpan {
    /// Compact multi-measure rests occupy several logical bars in one span.
    pub fn bar_count(&self, document: &KernDocument) -> u32 {
        document.lines()[self.content.clone()]
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
}

/// Return logical measure spans discovered from barline-delimited data regions.
pub fn measure_spans(document: &KernDocument) -> Vec<MeasureSpan> {
    let mut spans = Vec::new();
    let mut leading_barline = None;
    let mut content_start = None;
    let mut saw_data = false;

    for (index, line) in document.lines().iter().enumerate() {
        match line {
            KernLine::Barline(_) => {
                if saw_data {
                    spans.push(MeasureSpan {
                        leading_barline,
                        content: content_start.unwrap_or(index)..index,
                        trailing_barline: Some(index),
                    });
                }
                leading_barline = Some(index);
                content_start = None;
                saw_data = false;
            }
            KernLine::Data(_) => {
                if content_start.is_none() {
                    content_start = Some(
                        leading_barline
                            .map(|barline_index| barline_index + 1)
                            .unwrap_or(index),
                    );
                }
                saw_data = true;
            }
            _ if leading_barline.is_some() && content_start.is_none() => {
                content_start = Some(index);
            }
            _ => {}
        }
    }

    if saw_data {
        let content_end = trailing_terminal_start(document);
        spans.push(MeasureSpan {
            leading_barline,
            content: content_start.unwrap_or(content_end)..content_end,
            trailing_barline: None,
        });
    }

    spans
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
                .all(|token| matches!(token, crate::SpineManipulator::Terminate))
    )
}

/// Return true only for measures whose typed content is visually empty.
///
/// This is intentionally conservative: any non-data line, opaque token, visible
/// musical content, or non-null non-`**kern` content makes the measure ineligible.
pub fn measure_renders_empty(document: &KernDocument, span: &MeasureSpan) -> bool {
    if [span.leading_barline, span.trailing_barline].into_iter().flatten().any(|index| {
        matches!(&document.lines()[index], KernLine::Barline(barline) if barline.has_fermata())
    }) {
        return false;
    }
    let mut saw_invisible_rest = false;

    for line in &document.lines()[span.content.clone()] {
        let KernLine::Data(contents) = line else {
            return false;
        };

        for content in contents {
            match content {
                SpineContent::Kern(KernContent::Null) => {}
                SpineContent::Kern(KernContent::Rest(rest)) if rest.invisible => {
                    saw_invisible_rest = true;
                }
                SpineContent::Kern(KernContent::InvisibleRestSlice(_)) => {
                    saw_invisible_rest = true;
                }
                SpineContent::Dynam(DynamContent::Null)
                | SpineContent::Fing(FingContent::Null)
                | SpineContent::Text(TextContent::Null)
                | SpineContent::MxmlHarmony(MxmlHarmonyContent::Null) => {}
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
    }

    saw_invisible_rest
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(input: &str) -> Vec<MeasureSpan> {
        measure_spans(&KernDocument::parse(input).unwrap())
    }

    fn first_span(input: &str) -> (KernDocument, MeasureSpan) {
        let document = KernDocument::parse(input).unwrap();
        let span = measure_spans(&document).remove(0);
        (document, span)
    }

    #[test]
    fn discovers_first_measure_with_initial_barline() {
        assert_eq!(
            spans("**kern\n*clefG2\n=\n4c\n=\n*-\n"),
            vec![MeasureSpan {
                leading_barline: Some(3),
                content: 4..5,
                trailing_barline: Some(5),
            }]
        );
    }

    #[test]
    fn discovers_first_measure_without_initial_barline() {
        assert_eq!(
            spans("**kern\n*clefG2\n4c\n=\n4d\n=\n*-\n"),
            vec![
                MeasureSpan {
                    leading_barline: None,
                    content: 3..4,
                    trailing_barline: Some(4),
                },
                MeasureSpan {
                    leading_barline: Some(4),
                    content: 5..6,
                    trailing_barline: Some(6),
                },
            ]
        );
    }

    #[test]
    fn discovers_final_measure_without_trailing_barline() {
        assert_eq!(
            spans("**kern\n=\n4c\n*-\n"),
            vec![MeasureSpan {
                leading_barline: Some(3),
                content: 4..5,
                trailing_barline: None,
            }]
        );
    }

    #[test]
    fn invisible_rest_measure_renders_empty() {
        let (document, span) = first_span("**kern\n=\n4ryy\n4ryy\n=\n*-\n");

        assert!(measure_renders_empty(&document, &span));
    }

    #[test]
    fn visible_rest_measure_does_not_render_empty() {
        let (document, span) = first_span("**kern\n=\n4r\n=\n*-\n");

        assert!(!measure_renders_empty(&document, &span));
    }

    #[test]
    fn note_measure_does_not_render_empty() {
        let (document, span) = first_span("**kern\n=\n4c\n=\n*-\n");

        assert!(!measure_renders_empty(&document, &span));
    }

    #[test]
    fn chord_measure_does_not_render_empty() {
        let (document, span) = first_span("**kern\n=\n4c 4e\n=\n*-\n");

        assert!(!measure_renders_empty(&document, &span));
    }

    #[test]
    fn mixed_visible_and_invisible_measure_does_not_render_empty() {
        let (document, span) = first_span("**kern\n=\n4ryy\n4c\n=\n*-\n");

        assert!(!measure_renders_empty(&document, &span));
    }

    #[test]
    fn non_null_non_kern_content_does_not_render_empty() {
        let (document, span) = first_span("**kern\t**text\n=\t=\n4ryy\tword\n=\t=\n*-\t*-\n");

        assert!(!measure_renders_empty(&document, &span));
    }

    #[test]
    fn opaque_dot_content_does_not_render_empty() {
        let (document, span) = first_span("**kern\t**foo\n=\t=\n4ryy\t.\n=\t=\n*-\t*-\n");

        assert!(!measure_renders_empty(&document, &span));
    }

    #[test]
    fn tandem_line_inside_measure_does_not_render_empty() {
        let (document, span) = first_span("**kern\n=\n*M3/4\n4ryy\n=\n*-\n");

        assert!(!measure_renders_empty(&document, &span));
    }
}
