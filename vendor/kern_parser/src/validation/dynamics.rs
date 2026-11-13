//! `**dynam` marks Verovio draws differently from what the text suggests and
//! normalization leaves alone.

use crate::document::Framing;
use crate::normalization::dynamics::{HairpinLanes, Site, StepEvent};
use crate::validation::{SourceLocation, ValidationSeverity, ValidationWarning};
use crate::{
    DynamContent, DynamMarks, DynamicHairpinDirection, KernDocument, KernLine, SpineContent,
};

pub(crate) fn check(document: &KernDocument, framing: Framing) -> Vec<ValidationWarning> {
    let location = |site: Site| SourceLocation {
        line: document.source_line_number(site.line),
        column: Some(site.column + 1),
    };
    let mut warnings = Vec::new();
    let mut lanes = HairpinLanes::default();
    for (line, record) in document.lines().iter().enumerate() {
        lanes.follow(record);
        let KernLine::Data(contents) = record else {
            continue;
        };
        for (column, content) in contents.iter().enumerate() {
            let (Some(lane), SpineContent::Dynam(content)) = (lanes.lane(column), content) else {
                continue;
            };
            let site = Site { line, column };
            match content {
                DynamContent::Marks(_) => {}
                DynamContent::Raw { text, marks } => {
                    if let Some(message) = raw_conflict(&text.0, marks) {
                        warnings.push(warning("dynam.conflicting-token", location(site), message));
                    }
                }
                DynamContent::Null => continue,
            };
            for event in lanes.step(lane, site, content) {
                match event {
                    StepEvent::LeadingEnd(direction) if framing == Framing::Complete => warnings.push(warning(
                        "dynam.unpaired-hairpin",
                        location(site),
                        format!(
                            "this {} end has no start in its spine",
                            direction_name(direction)
                        ),
                    )),
                    StepEvent::IgnoredEnd(direction, start, interrupted) => warnings.push(warning(
                        "dynam.ignored-hairpin-end",
                        location(site),
                        format!(
                            "this {} end draws no hairpin: the start on line {} renders as text because line {} interrupts endpoint lookup",
                            direction_name(direction),
                            location(start).line,
                            location(interrupted).line
                        ),
                    )),
                    StepEvent::StrayEnd(direction) => warnings.push(warning(
                        "dynam.ignored-hairpin-end",
                        location(site),
                        format!("this {} end draws no hairpin: the earlier hairpin already ended", direction_name(direction)),
                    )),
                    StepEvent::LeadingEnd(_) | StepEvent::StrayContinue(_) => {}
                }
            }
        }
    }
    warnings
}

fn raw_conflict(text: &str, marks: &DynamMarks) -> Option<String> {
    if marks.crescendo.begin.is_some() && marks.diminuendo.begin.is_some() {
        return Some(format!("Verovio ignores the diminuendo start `>` in `{text}` because the token also contains `<`"));
    }
    let letters: String = text.chars().filter(|ch| ch.is_ascii_alphabetic()).collect();
    if !letters.is_empty() && !verovio_level(&letters) {
        return Some(format!(
            "Verovio does not draw the dynamic letters `{letters}` in `{text}`"
        ));
    }
    let unsupported: String = text
        .chars()
        .filter(|ch| {
            !ch.is_ascii_alphabetic() && !ch.is_ascii_whitespace() && !"<>()[].".contains(*ch)
        })
        .collect();
    if unsupported.is_empty() {
        None
    } else {
        Some(format!(
            "Verovio ignores the unsupported symbols `{unsupported}` in `{text}`"
        ))
    }
}

// These are addDynamicsMark's accepted letter patterns in Verovio's iohumdrum.cpp.
fn verovio_level(letters: &str) -> bool {
    let all = |text: &str, ch| !text.is_empty() && text.chars().all(|part| part == ch);
    let forte = letters.strip_prefix(['s', 'r']).unwrap_or(letters);
    let forte = forte.strip_suffix('z').unwrap_or(forte);
    let piano = letters.strip_prefix('s').unwrap_or(letters);
    let before_piano = piano.trim_end_matches('p');
    let before_piano = before_piano.strip_suffix('z').unwrap_or(before_piano);
    all(forte, 'f')
        || all(letters, 'p')
        || matches!(letters, "mp" | "mf" | "rz")
        || (piano.ends_with('p') && all(before_piano, 'f'))
}

fn direction_name(direction: DynamicHairpinDirection) -> &'static str {
    match direction {
        DynamicHairpinDirection::Crescendo => "crescendo",
        DynamicHairpinDirection::Diminuendo => "diminuendo",
    }
}

fn warning(code: &'static str, location: SourceLocation, message: String) -> ValidationWarning {
    ValidationWarning {
        code,
        severity: ValidationSeverity::Warning,
        location,
        message,
        loose_end: None,
        span: None,
    }
}

#[cfg(test)]
mod tests {
    use crate::check::{check_text, CheckConfig};
    use crate::validation::{ValidationConfig, ValidationPass};

    fn findings(input: &str) -> Vec<(String, Option<usize>, String)> {
        let report = check_text(
            input,
            &CheckConfig {
                validation: ValidationConfig::with_passes(vec![ValidationPass::Dynam]),
                ..Default::default()
            },
        );
        report
            .diagnostics
            .into_iter()
            .map(|diagnostic| (diagnostic.code, diagnostic.line, diagnostic.message))
            .collect()
    }

    #[test]
    fn a_level_at_the_endpoint_is_valid() {
        assert_eq!(
            findings("**kern\t**dynam\n4c\tp <\n4d\t.\n4e\t[ f\n*-\t*-\n"),
            []
        );
    }

    #[test]
    fn a_conflicting_token_is_reported() {
        assert_eq!(
            findings("**kern\t**dynam\n4c\tp f\n*-\t*-\n"),
            [(
                "dynam.conflicting-token".to_owned(),
                Some(2),
                "Verovio does not draw the dynamic letters `pf` in `p f`".to_owned()
            )]
        );
    }

    #[test]
    fn supported_compound_dynamics_are_valid() {
        for token in ["fz p", "fzp", "f p", "s fz p", "ffz pp", "r z", "fz p <"] {
            assert_eq!(
                findings(&format!("**kern\t**dynam\n4c\t{token}\n4d\t.\n*-\t*-\n")),
                [],
                "{token} is supported by Verovio"
            );
        }
    }

    #[cfg(feature = "engraving")]
    #[test]
    fn compound_dynamics_match_verovio_output() {
        for (token, expected) in [
            ("fz p", "fzp"),
            ("fzp", "fzp"),
            ("s fz p", "sfzp"),
            ("r z", "rz"),
        ] {
            let source = format!("**kern\t**dynam\n*clefG2\t*\n4c\t{token}\n*-\t*-\n");
            let mut toolkit = verovioxide::Toolkit::new().unwrap();
            toolkit.render_data(&source, None).unwrap();
            let mei = toolkit.get_mei().unwrap();
            let xml = roxmltree::Document::parse(&mei).unwrap();
            let levels: Vec<_> = xml
                .descendants()
                .filter(|node| node.has_tag_name("dynam"))
                .map(|node| {
                    node.descendants()
                        .filter(|child| child.is_text())
                        .filter_map(|child| child.text())
                        .collect::<String>()
                })
                .collect();
            assert_eq!(levels, [expected], "{token}");
            assert_eq!(findings(&source), [], "{token}");
        }
    }

    #[test]
    fn a_supported_level_does_not_hide_unsupported_symbols() {
        assert_eq!(
            findings("**kern\t**dynam\n4c\tfz p @\n*-\t*-\n"),
            [(
                "dynam.conflicting-token".to_owned(),
                Some(2),
                "Verovio ignores the unsupported symbols `@` in `fz p @`".to_owned()
            )]
        );
    }

    #[test]
    fn lone_starts_are_text_dynamics() {
        for token in ["<", ">", "p <", "p >"] {
            assert_eq!(
                findings(&format!("**kern\t**dynam\n4c\t{token}\n4d\t.\n*-\t*-\n")),
                [],
                "{token} is a text dynamic when it has no endpoint"
            );
        }
    }

    #[test]
    fn text_dynamics_before_levels_are_valid() {
        for token in ["<", ">", "p <", "p >"] {
            assert_eq!(
                findings(&format!("**kern\t**dynam\n4c\t{token}\n4d\tff\n*-\t*-\n")),
                []
            );
        }
    }

    #[test]
    fn an_endpoint_after_a_level_is_reported_at_the_endpoint() {
        assert_eq!(
            findings("**kern\t**dynam\n4c\t<\n4d\tff\n4e\t[\n*-\t*-\n"),
            [(
                "dynam.ignored-hairpin-end".to_owned(),
                Some(4),
                "this crescendo end draws no hairpin: the start on line 2 renders as text because line 3 interrupts endpoint lookup".to_owned()
            )]
        );
    }

    #[test]
    fn opposite_starts_interrupt_endpoint_lookup_too() {
        assert_eq!(
            findings("**kern\t**dynam\n4c\t<\n4d\t>\n4e\t]\n4f\t[\n*-\t*-\n"),
            [(
                "dynam.ignored-hairpin-end".to_owned(),
                Some(5),
                "this crescendo end draws no hairpin: the start on line 2 renders as text because line 3 interrupts endpoint lookup".to_owned()
            )]
        );
    }

    #[test]
    fn raw_letters_interrupt_endpoint_lookup() {
        let report = findings("**kern\t**dynam\n4c\t<\n4d\tp f\n4e\t[\n*-\t*-\n");
        assert_eq!(
            report
                .iter()
                .map(|(code, line, _)| (code.as_str(), *line))
                .collect::<Vec<_>>(),
            [
                ("dynam.conflicting-token", Some(3)),
                ("dynam.ignored-hairpin-end", Some(4)),
            ]
        );
    }

    #[test]
    fn a_leading_endpoint_in_an_excerpt_is_not_a_proven_error() {
        let report = check_text(
            "**kern\t**dynam\n4c\t[\n*-\t*-\n",
            &CheckConfig {
                framing: crate::document::Framing::Excerpt,
                validation: ValidationConfig::with_passes(vec![ValidationPass::Dynam]),
                ..Default::default()
            },
        );
        assert!(report.diagnostics.is_empty(), "{:?}", report.diagnostics);
    }

    #[test]
    fn conflicting_directions_explain_which_mark_is_ignored() {
        assert_eq!(findings("**kern\t**dynam\n4c\t< >\n4d\t[\n*-\t*-\n"), [(
            "dynam.conflicting-token".to_owned(), Some(2),
            "Verovio ignores the diminuendo start `>` in `< >` because the token also contains `<`".to_owned(),
        )]);
    }

    #[test]
    fn a_single_note_hairpin_after_a_text_dynamic_is_valid() {
        assert_eq!(
            findings("**kern\t**dynam\n4c\t<\n4d\tff\n4e\t<[\n*-\t*-\n"),
            []
        );
    }

    #[test]
    fn a_leading_end_is_reported_but_a_trailing_start_is_not() {
        assert_eq!(
            findings("**kern\t**dynam\n4c\t]\n4d\t<\n4e\t[\n4f\t>\n*-\t*-\n"),
            [(
                "dynam.unpaired-hairpin".to_owned(),
                Some(2),
                "this diminuendo end has no start in its spine".to_owned()
            )]
        );
        assert_eq!(
            findings("**kern\t**dynam\n4c\t[\n*-\t*-\n"),
            [(
                "dynam.unpaired-hairpin".to_owned(),
                Some(2),
                "this crescendo end has no start in its spine".to_owned()
            )]
        );
    }

    #[test]
    fn a_start_left_open_in_a_half_that_merges_is_a_text_dynamic() {
        assert_eq!(
            findings("**kern\t**dynam\n*\t*^\n4c\t<\t<\n4d\t[\t.\n*\t*v\t*v\n*-\t*-\n"),
            []
        );
    }
}
