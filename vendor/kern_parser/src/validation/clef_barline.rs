//! Warnings for clef changes placed after visible measure boundaries.

use std::collections::BTreeSet;

use crate::{
    types::BarlineStyle,
    validation::{SourceLocation, ValidationSeverity, ValidationWarning},
    DocumentEventKind, KernDocument, TandemInterpretation,
};

pub(crate) fn check(document: &KernDocument) -> Vec<ValidationWarning> {
    // The parser splits a source tandem line into one line per interpretation
    // kind, so a clef's companions live on sibling events with its line number.
    let mut mixed_lines = BTreeSet::new();
    for event in document.events() {
        if let DocumentEventKind::TandemInterpretationLine(tandems) = event.kind {
            let mixed = tandems.tokens().iter().enumerate().any(|(index, tandem)| {
                !tandems.is_generated_token(index)
                    && !matches!(
                        tandem,
                        TandemInterpretation::Clef(_) | TandemInterpretation::Null
                    )
            });
            if mixed {
                mixed_lines.insert(event.line_number);
            }
        }
    }

    let mut seen_data = false;
    let mut boundary = None;
    let mut warnings = Vec::new();

    for event in document.events() {
        match event.kind {
            DocumentEventKind::Data(_) => {
                seen_data = true;
                boundary = None;
            }
            DocumentEventKind::Barline(barline) => {
                boundary = (seen_data && !barline.styles().contains(&BarlineStyle::Invisible))
                    .then_some(event.line_number);
            }
            DocumentEventKind::TandemInterpretationLine(tandems) => {
                let Some(barline) = boundary else {
                    continue;
                };
                for (field_index, tandem) in tandems.tokens().iter().enumerate() {
                    if !matches!(tandem, TandemInterpretation::Clef(_))
                        || tandems.is_generated_token(field_index)
                    {
                        continue;
                    }
                    let line = event.line_number;
                    let field = field_index + 1;
                    let mut message = format!(
                        "clef change at line {line}, field {field} comes after the barline at line {barline}; printed editions place a clef change at a measure boundary before the barline. Move the clef above line {barline}, or acknowledge it with `!ack:` if this edition prints it after the barline."
                    );
                    if mixed_lines.contains(&line) {
                        message.push_str(" Move only the clef onto its own interpretation line above the barline; key and meter changes stay after it.");
                    }
                    warnings.push(ValidationWarning {
                        span: None,
                        severity: ValidationSeverity::Warning,
                        code: "clef-barline.after-barline",
                        location: SourceLocation {
                            line,
                            column: Some(field),
                        },
                        message,
                        loose_end: None,
                    });
                }
            }
            _ => {}
        }
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParseDialect, ParseOptions};

    fn warnings(input: &str) -> Vec<ValidationWarning> {
        check(&KernDocument::parse(input).unwrap())
    }

    #[test]
    fn warns_for_each_clef_in_any_spine_with_source_locations() {
        let warnings =
            warnings("**kern\t**text\n4c\tword\n=\t=\n!! page\n*clefF4\t*clefG2yy\n*-\t*-\n");
        assert_eq!(warnings.len(), 2);
        for (warning, field) in warnings.iter().zip([1, 2]) {
            assert_eq!(warning.code, "clef-barline.after-barline");
            assert_eq!(
                warning.location,
                SourceLocation {
                    line: 5,
                    column: Some(field)
                }
            );
            assert!(warning.message.contains("barline at line 3"));
            assert!(!warning.message.contains("Move only the clef"));
        }
    }

    #[test]
    fn comments_tandems_and_manipulators_preserve_the_boundary() {
        let warnings =
            warnings("**kern\n4c\n=\n! comment\n*k[b-]\n*^\n*\t*clefF4\n4c\t4e\n*-\t*-\n");
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings[0].location,
            SourceLocation {
                line: 7,
                column: Some(2)
            }
        );
    }

    #[test]
    fn mixed_interpretations_need_a_clef_only_move() {
        for other in ["*k[b-]", "*M3/4", "*met(c)"] {
            let input = format!("**kern\t**kern\n4c\t4e\n=\t=\n*clefF4\t{other}\n*-\t*-\n");
            let warnings = warnings(&input);
            assert_eq!(warnings.len(), 1);
            assert!(warnings[0].message.ends_with("Move only the clef onto its own interpretation line above the barline; key and meter changes stay after it."));
        }
    }

    #[test]
    fn ignores_headers_opening_barlines_and_mid_measure_clefs() {
        for input in [
            "**kern\n*clefG2\n4c\n*-\n",
            "**kern\n=1-\n*clefG2\n4c\n*-\n",
            "**kern\n=1\n*clefG2\n4c\n*-\n",
            "**kern\n4c\n*clefF4\n=\n*-\n",
            "**kern\n4c\n=\n4e\n*clefF4\n*-\n",
        ] {
            assert!(warnings(input).is_empty(), "{input}");
        }
    }

    #[test]
    fn skips_only_entirely_invisible_nearest_barlines() {
        for barline in ["=-", "=5-"] {
            let input = format!("**kern\n4c\n=\n{barline}\n*clefF4\n*-\n");
            assert!(warnings(&input).is_empty());
        }
        assert!(warnings("**kern\t**kern\n4c\t4e\n=-\t=5-\n*clefF4\t*\n*-\t*-\n").is_empty());
        assert_eq!(
            warnings("**kern\t**kern\n4c\t4e\n=-\t=\n*clefF4\t*\n*-\t*-\n").len(),
            1
        );
    }

    #[test]
    fn warns_after_double_bars_repeats_and_section_ends() {
        for barline in ["==", "=|:", "=:|", "==||"] {
            let input = format!("**kern\n4c\n{barline}\n*clefF4\n*-\n");
            assert_eq!(warnings(&input).len(), 1, "{input}");
        }
    }

    #[test]
    fn supports_both_dialects_and_free_meter() {
        for dialect in [ParseDialect::HumdrumKern, ParseDialect::TranscodaV2] {
            let document = KernDocument::parse_with_options(
                "**kern\n*MX\n4c\n=\n*clefF4\n*-\n",
                ParseOptions { dialect },
            )
            .unwrap();
            assert_eq!(check(&document).len(), 1);
        }
    }
}
