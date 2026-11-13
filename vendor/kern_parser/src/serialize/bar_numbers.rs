use crate::{
    analysis::{measure_spans, MeasureSpan},
    KernContent, KernDocument, KernLine, SpineContent,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BarNumberStyle {
    #[default]
    Omit,
    Sequential,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BarNumberPlan {
    barline_numbers: Vec<Option<u32>>,
    insertions: Vec<Option<u32>>,
}

impl BarNumberPlan {
    pub(crate) fn new(document: &KernDocument, style: BarNumberStyle) -> Self {
        Self::with_first_number(document, style, 1)
    }

    pub(crate) fn with_first_number(
        document: &KernDocument,
        style: BarNumberStyle,
        first: u32,
    ) -> Self {
        let mut plan = Self {
            barline_numbers: vec![None; document.lines().len()],
            insertions: vec![None; document.lines().len()],
        };

        if !matches!(style, BarNumberStyle::Sequential) {
            return plan;
        }

        let mut number = first;
        for span in measure_spans(document) {
            if let Some(line_index) = span.leading_barline {
                if let Some(slot) = plan.barline_numbers.get_mut(line_index) {
                    *slot = Some(number);
                }
            } else if let Some(slot) = plan.insertions.get_mut(span.content.start) {
                *slot = Some(number);
            }
            number = number.saturating_add(measure_count_for_span(document, &span));
        }

        plan
    }

    pub(crate) fn barline_number(&self, line_index: usize) -> Option<u32> {
        self.barline_numbers.get(line_index).copied().flatten()
    }

    pub(crate) fn insertion_before(&self, line_index: usize) -> Option<u32> {
        self.insertions.get(line_index).copied().flatten()
    }
}

fn measure_count_for_span(document: &KernDocument, span: &MeasureSpan) -> u32 {
    document.lines()[span.content.clone()]
        .iter()
        .filter_map(|line| {
            let KernLine::Data(contents) = line else {
                return None;
            };

            contents
                .iter()
                .filter_map(|content| {
                    let SpineContent::Kern(KernContent::MultiMeasureRest(rest)) = content else {
                        return None;
                    };

                    Some(rest.measure_count)
                })
                .max()
        })
        .max()
        .unwrap_or(1)
}
