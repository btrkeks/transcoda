//! Remove measures that contain only invisible rests and typed nulls.

use crate::{
    analysis::{measure_renders_empty, measure_spans, MeasureSpan},
    KernDocument,
};

use super::{NormalizationPass, NormalizeOptions};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoveInvisibleRestMeasures {
    options: NormalizeOptions,
}

impl RemoveInvisibleRestMeasures {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for RemoveInvisibleRestMeasures {
    fn normalize(&self, document: &mut KernDocument) {
        let _ = &self.options;
        remove_invisible_rest_measures(document);
    }
}

pub fn remove_invisible_rest_measures(document: &mut KernDocument) {
    // Decide every measure on the unmodified document: removing a measure
    // also removes its leading barline, which is the previous measure's
    // trailing barline, so spans read after a removal point at the wrong
    // lines (past the end when the last measure goes). Removing back to
    // front keeps the earlier spans' indices valid.
    let empty: Vec<_> = measure_spans(document)
        .into_iter()
        .filter(|span| measure_renders_empty(document, span))
        .collect();
    for span in empty.iter().rev() {
        remove_measure(document, span);
    }
}

fn remove_measure(document: &mut KernDocument, span: &MeasureSpan) {
    let start = span.leading_barline.unwrap_or(span.content.start);
    let end = span.trailing_barline.unwrap_or(span.content.end);

    if start < end && end <= document.lines().len() {
        document.drain_lines(start..end);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        normalization::omr::normalized_for_omr_with_options,
        serialize::omr::{OmrSerializeOptions, OmrSerializer},
        KernDocument,
    };

    fn normalized_omr(input: &str) -> String {
        let document = KernDocument::parse(input).unwrap();
        let normalized = normalized_for_omr_with_options(
            &document,
            &NormalizeOptions {
                remove_invisible_rest_measures: true,
                preserve_leading_barlines: true,
                ..Default::default()
            },
        );
        OmrSerializer::new(OmrSerializeOptions {
            preserve_leading_barlines: true,
            ..Default::default()
        })
        .serialize_document(&normalized)
    }

    #[test]
    fn removes_a_final_invisible_rest_measure_of_an_excerpt_without_terminator() {
        let mut document =
            KernDocument::parse("**kern\n*clefG2\n*k[]\n=1\n4c\n=2\n4ryy\n").unwrap();
        remove_invisible_rest_measures(&mut document);
        assert_eq!(document.lines().len(), 5);
    }

    #[test]
    fn removes_one_invisible_rest_measure() {
        assert_eq!(
            normalized_omr("**kern\n*clefG2\n*k[]\n=\n4c\n=\n4ryy\n4ryy\n=\n4d\n=\n*-\n"),
            "*clefG2\n*k[]\n=\n4c\n=\n4d\n=\n*-\n"
        );
    }

    #[test]
    fn removes_consecutive_invisible_rest_measures() {
        assert_eq!(
            normalized_omr("**kern\n*clefG2\n*k[]\n=\n4c\n=\n4ryy\n=\n4ryy\n=\n4d\n=\n*-\n"),
            "*clefG2\n*k[]\n=\n4c\n=\n4d\n=\n*-\n"
        );
    }

    #[test]
    fn removes_final_invisible_rest_measure_without_deleting_terminal() {
        assert_eq!(
            normalized_omr("**kern\n*clefG2\n*k[]\n=\n4c\n=\n4ryy\n*-\n"),
            "*clefG2\n*k[]\n=\n4c\n*-\n"
        );
    }

    #[test]
    fn preserves_visible_rest_measure() {
        assert!(normalized_omr("**kern\n*clefG2\n=\n4r\n=\n*-\n").contains("4r"));
    }

    #[test]
    fn leaves_document_unchanged_when_omr_option_is_disabled() {
        let document = KernDocument::parse("**kern\n*clefG2\n=\n4ryy\n=\n*-\n").unwrap();

        let normalized = normalized_for_omr_with_options(
            &document,
            &NormalizeOptions {
                preserve_leading_barlines: true,
                remove_invisible_rest_measures: false,
                ..Default::default()
            },
        );

        assert!(crate::analysis::measure_spans(&normalized)
            .iter()
            .any(|span| crate::analysis::measure_renders_empty(&normalized, span)));
    }

    #[test]
    fn compact_omr_no_note_modifiers_removes_invisible_rest_measures_by_default() {
        let document =
            KernDocument::parse("**kern\n*clefG2\n*k[]\n=\n4c\n=\n4ryy\n=\n4d\n=\n*-\n").unwrap();

        assert_eq!(
            document.to_compact_omr_no_note_modifiers_string(),
            "*clefG2\n*k[]\n4c\n=\n4d\n=\n"
        );
    }

    #[test]
    fn preserves_measure_with_spine_operation_inside() {
        let input = "**kern\n*clefG2\n=\n*^\n4ryy\t.\n.\t4ryy\n*v\t*v\n=\n*-\n";
        let document = KernDocument::parse(input).unwrap();
        let mut normalized = document.clone();

        RemoveInvisibleRestMeasures::new(Default::default()).normalize(&mut normalized);

        assert_eq!(normalized, document);
    }

    #[test]
    fn long_invisible_rest_runs_leave_no_empty_measures_when_enabled() {
        let input = [
            "**kern", "*clefG2", "*k[]", "=", "4c", "=", "4ryy", "=", "4ryy", "=", "4ryy", "=",
            "4d", "=", "*-",
        ]
        .join("\n");
        let document = KernDocument::parse(&input).unwrap();
        let normalized = normalized_for_omr_with_options(
            &document,
            &NormalizeOptions {
                remove_invisible_rest_measures: true,
                preserve_leading_barlines: true,
                ..Default::default()
            },
        );
        let remaining_empty_measures = crate::analysis::measure_spans(&normalized)
            .iter()
            .filter(|span| crate::analysis::measure_renders_empty(&normalized, span))
            .count();

        assert_eq!(remaining_empty_measures, 0);
    }
}
