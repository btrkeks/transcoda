use crate::{
    document::Framing,
    document_boundaries::{
        closing_state, has_musical_data as document_has_musical_data, leading_state_header_end,
        opening_state, seam_state_lines, strip_final_terminal, strip_optional_spine_type,
    },
    score::Score,
    transforms::pad_final_measure_with_rests,
    DocumentLayout, KernDocument, KernLine, KernMetadata,
};

use super::ConcatenationError;

pub fn concatenate_documents(parts: &[KernDocument]) -> Result<KernDocument, ConcatenationError> {
    if parts.len() < 2 {
        return Err(ConcatenationError::TooFewSnippets);
    }

    let staff_count = parts[0].staff_count();
    if parts.iter().any(|part| part.staff_count() != staff_count) {
        return Err(ConcatenationError::StaffCountMismatch);
    }

    let parts = parts_prepared_for_boundaries(parts);

    let mut lines = body_lines_for_first(&parts[0]);
    if lines.is_empty() {
        return Err(ConcatenationError::NoMusicalData);
    }

    let scores = parts
        .iter()
        .map(|part| Score::resolve(part, Framing::Excerpt))
        .collect::<Result<Vec<_>, _>>()?;
    for pair in scores.windows(2) {
        doubleize_terminal_basic_barline(&mut lines);
        let mut body = body_lines_for_appended(pair[1].document());
        if matches!(body.first(), Some(KernLine::Barline(barline)) if matches!(barline.repeat(), crate::types::RepeatStyle::Start | crate::types::RepeatStyle::StartAndEnd))
        {
            // Verovio discards meter declarations before an opening repeat.
            lines.push(body.remove(0));
        }
        lines.extend(seam_state_lines(
            &closing_state(&pair[0]),
            &opening_state(&pair[1]),
        )?);
        lines.extend(body);
    }

    let document = KernDocument::from_parts(
        KernMetadata::default(),
        DocumentLayout {
            initial_spine_count: staff_count as u32,
        },
        lines.into_iter().map(|line| (line, None)).collect(),
    );
    if !has_musical_data(&document) {
        return Err(ConcatenationError::NoMusicalData);
    }
    Ok(document)
}

fn parts_prepared_for_boundaries(parts: &[KernDocument]) -> Vec<KernDocument> {
    parts
        .iter()
        .enumerate()
        .map(|(index, part)| {
            if index + 1 == parts.len() {
                part.clone()
            } else {
                pad_final_measure_with_rests(part)
            }
        })
        .collect()
}

fn body_lines_for_first(document: &KernDocument) -> Vec<KernLine> {
    strip_final_terminal(strip_optional_spine_type(document.lines()).to_vec())
}

fn body_lines_for_appended(document: &KernDocument) -> Vec<KernLine> {
    let lines = strip_optional_spine_type(document.lines());
    let start = leading_state_header_end(lines);
    strip_final_terminal(lines[start..].to_vec())
}

fn doubleize_terminal_basic_barline(lines: &mut [KernLine]) {
    let Some(KernLine::Barline(barline)) = lines
        .iter_mut()
        .rev()
        .find(|line| matches!(line, KernLine::Barline(_)))
    else {
        return;
    };

    // The appended snippet may begin with an anacrusis. A double barline keeps
    // that pickup from reading like a broken internal measure after the seam.
    barline.doubleize_basic();
}

pub(super) fn has_musical_data(document: &KernDocument) -> bool {
    document_has_musical_data(document)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> KernDocument {
        KernDocument::parse(input).unwrap()
    }

    fn concatenate(inputs: &[&str]) -> String {
        let documents = inputs.iter().map(|input| parse(input)).collect::<Vec<_>>();
        let document = concatenate_documents(&documents).unwrap();
        let text = crate::SerializePreset::HumdrumKern.serialize_document(&document);
        Score::resolve(&parse(&text), Framing::Complete)
            .unwrap_or_else(|errors| panic!("{errors}\n{text}"));
        document.to_omr_string()
    }

    #[test]
    fn pads_incomplete_non_final_part_before_concatenating() {
        assert_eq!(
            concatenate(&[
                "**kern\n*M4/4\n=1\n4c\n==\n*-\n",
                "**kern\n*M4/4\n=1\n1d\n==\n*-\n",
            ]),
            "*clefG2\n*k[]\n*M4/4\n4c\n4%3ryy\n==\n=\n1d\n==\n"
        );
    }

    #[test]
    fn leaves_incomplete_final_part_unpadded() {
        assert_eq!(
            concatenate(&[
                "**kern\n*M4/4\n=1\n1c\n==\n*-\n",
                "**kern\n*M4/4\n=1\n4d\n==\n*-\n",
            ]),
            "*clefG2\n*k[]\n*M4/4\n1c\n==\n=\n4d\n==\n"
        );
    }

    #[test]
    fn preserves_terminal_thin_barline_before_appending() {
        assert_eq!(
            concatenate(&[
                "**kern\n*M4/4\n=1\n1c\n=||\n*-\n",
                "**kern\n*M4/4\n=1\n1d\n==\n*-\n"
            ]),
            "*clefG2\n*k[]\n*M4/4\n1c\n=||\n=\n1d\n==\n"
        );
    }

    #[test]
    fn doubleizes_terminal_basic_barline_before_appending() {
        assert_eq!(
            concatenate(&[
                "**kern\n*M4/4\n=1\n1c\n=\n*-\n",
                "**kern\n*M4/4\n=1\n1d\n==\n*-\n"
            ]),
            "*clefG2\n*k[]\n*M4/4\n1c\n==\n=\n1d\n==\n"
        );
    }

    #[test]
    fn doubleizes_only_visible_faces_of_mixed_terminal_barline() {
        let output = concatenate(&[
            "**kern\t**kern\n*M4/4\t*M4/4\n1C\t1c\n=\t=-\n*-\t*-\n",
            "**kern\t**kern\n*M4/4\t*M4/4\n1D\t1d\n==\t==\n*-\t*-\n",
        ]);

        assert!(output.contains("==\t=-\n"));
    }

    #[test]
    fn restates_only_the_staff_definition_that_changes() {
        assert_eq!(
            concatenate(&[
                "**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n=1\t=1\n1C\t1c\n==\t==\n*-\t*-\n",
                "**kern\t**kern\n*clefF4\t*clefG2\n*k[b-]\t*k[b-]\n*M3/4\t*M3/4\n=1\t=1\n2.D\t2.d\n==\t==\n*-\t*-\n",
            ]),
            "*clefF4\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n1C\t1c\n==\t==\n*k[b-]\t*k[b-]\n*M3/4\t*M3/4\n=\t=\n2.D\t2.d\n==\t==\n"
        );
    }

    #[test]
    fn restates_a_meter_whose_header_runs_into_another_tandem() {
        let output = concatenate(&[
            "**kern\n*M3/4\n=1\n2.c\n==\n*-\n",
            "**kern\n*M4/4\n*Xtuplet\n=1\n1d\n==\n*-\n",
        ]);

        assert!(output.contains("==\n*M4/4\n"), "{output}");
    }

    #[test]
    fn refuses_a_snippet_that_does_not_resolve() {
        let documents = [
            parse("**kern\n*M4/4\n=1\n1c\n==\n*-\n"),
            parse("**kern\n*M4/4\n=1\n1d\n*X8va\n==\n*-\n"),
        ];
        assert!(matches!(
            concatenate_documents(&documents),
            Err(ConcatenationError::State(_))
        ));
    }

    #[test]
    fn refuses_snippets_whose_spines_differ_at_the_seam() {
        let documents = [
            parse("**kern\t**kern\t**dynam\n*M4/4\t*M4/4\t*\n=1\t=1\t=1\n1C\t1c\tp\n==\t==\t==\n*-\t*-\t*-\n"),
            parse("**kern\t**kern\n*M3/4\t*M3/4\n=1\t=1\n2.D\t2.d\n==\t==\n*-\t*-\n"),
        ];
        assert!(matches!(
            concatenate_documents(&documents),
            Err(ConcatenationError::Seam(_))
        ));
    }

    #[test]
    fn doubleizes_seam_before_appended_anacrusis() {
        assert_eq!(
            concatenate(&[
                "**kern\n*M4/4\n=1\n1c\n=\n*-\n",
                "**kern\n*M4/4\n4d\n=1\n1e\n==\n*-\n"
            ]),
            "*clefG2\n*k[]\n*M4/4\n1c\n==\n4d\n=\n1e\n==\n"
        );
    }
}
