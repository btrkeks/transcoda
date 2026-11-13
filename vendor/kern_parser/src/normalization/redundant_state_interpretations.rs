//! Normalization for state-setting tandem interpretations that restate the
//! value already in force.
//!
//! The resolved score decides: every tandem it reports as
//! [`Outcome::Redundant`] becomes a null tandem. A restatement Verovio draws
//! again ([`Outcome::Restated`]: a clef, key or time signature, a pedal
//! mark, a ligature or coloration bracket) is kept, as is a sibling copy of
//! a change (`*clefF4\t*clefF4yy`) and a `*cue`/`*Xcue` that changes
//! Verovio's cue flag for its layer number. A document whose score does not
//! resolve is left unchanged.
//!
//! Example:
//! ```text
//! Before:
//! *8va
//! 4c
//! *8va
//! 4d
//!
//! After:
//! *8va
//! 4c
//! *
//! 4d
//! ```
//!
//! The second `*8va` repeats an already active ottava, so it becomes a null
//! tandem.

use std::collections::BTreeMap;

use crate::{
    document::Framing,
    normalization::NormalizationPass,
    score::{Outcome, Score},
    KernDocument, KernLine, TandemInterpretation, TandemInterpretationLine,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Remove redundant state-setting interpretations.
pub struct RemoveRedundantStateInterpretations {
    options: NormalizeOptions,
}

impl RemoveRedundantStateInterpretations {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for RemoveRedundantStateInterpretations {
    fn normalize(&self, document: &mut KernDocument) {
        let _ = &self.options;
        let Ok(score) = Score::resolve(document, Framing::Excerpt) else {
            return;
        };
        let redundant = redundant_tandems(&score);
        for (line_index, columns) in redundant {
            let KernLine::TandemInterpretationLine(tandems) = &mut document.lines_mut()[line_index]
            else {
                unreachable!("state events come from tandem lines");
            };
            null_tandems(tandems, &columns);
        }
    }
}

/// Columns of the tandems the score reports as redundant, by line index.
pub(crate) fn redundant_tandems(score: &Score<'_>) -> BTreeMap<usize, Vec<usize>> {
    let mut redundant: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for event in score.events() {
        if event.outcome == Outcome::Redundant {
            redundant
                .entry(event.line_index)
                .or_default()
                .push(event.column);
        }
    }
    redundant
}

/// Replace `columns` of a tandem line with null tandems.
pub(crate) fn null_tandems(tandems: &mut TandemInterpretationLine, columns: &[usize]) {
    let mut generated = tandems.generated_tokens().to_vec();
    let mut tokens = tandems.tokens().to_vec();
    for &column in columns {
        tokens[column] = TandemInterpretation::Null;
        if let Some(generated) = generated.get_mut(column) {
            *generated = false;
        }
    }
    *tandems = TandemInterpretationLine::from_homogeneous_tokens_with_generated(tokens, generated);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized(input: &str) -> String {
        let mut document = KernDocument::parse(input).unwrap();
        RemoveRedundantStateInterpretations::new(NormalizeOptions::default())
            .normalize(&mut document);
        document.to_humdrum_kern_string()
    }

    #[test]
    fn nulls_what_the_score_reports_as_redundant() {
        let output = normalized("**kern\n*clefG2\n*8va\n4c\n*8va\n*clefG2\n4d\n*X8va\n*-\n");
        assert_eq!(output.matches("*8va").count(), 1, "{output}");
        // Verovio draws a restated clef.
        assert_eq!(output.matches("*clefG2").count(), 2, "{output}");
    }

    #[test]
    fn keeps_a_sibling_copy_of_a_change() {
        let output = normalized("**kern\n*clefG2\n*^\n*clefF4\t*clefF4yy\n4C\t4E\n*v\t*v\n*-\n");
        assert!(output.contains("*clefF4yy\t*clefF4"), "{output}");
    }

    #[test]
    fn keeps_a_cue_tandem_that_clears_a_stale_verovio_flag() {
        // The resplit's right child is at normal size, but Verovio still
        // holds the cue flag of layer 2 from the first split.
        let input = "**kern\n*^\n*\t*cue\n4c\t4e\n*v\t*v\n4c\n*^\n*\t*Xcue\n4c\t4e\n*v\t*v\n*-\n";
        assert_eq!(normalized(input).matches("*Xcue").count(), 1);
    }

    #[test]
    fn leaves_a_document_that_does_not_resolve_unchanged() {
        let input = "**kern\n4c\n*X8va\n*X8va\n4d\n*-\n";
        let mut document = KernDocument::parse(input).unwrap();
        let before = document.clone();
        RemoveRedundantStateInterpretations::new(NormalizeOptions::default())
            .normalize(&mut document);
        assert_eq!(document, before);
    }
}
