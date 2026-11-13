use crate::types::{BarlineStyle, RepeatStyle};
use crate::*;

#[test]
fn parses_barline_tokens() {
    let cases = [
        (
            "=",
            Barline::new(vec![BarlineStyle::Basic], RepeatStyle::None),
        ),
        (
            "=1",
            Barline::new(vec![BarlineStyle::Basic], RepeatStyle::None),
        ),
        (
            "=||",
            Barline::new(vec![BarlineStyle::Thin], RepeatStyle::None),
        ),
        (
            "==",
            Barline::new(vec![BarlineStyle::Double], RepeatStyle::None),
        ),
        (
            "=:|!",
            Barline::new(vec![BarlineStyle::Double], RepeatStyle::End),
        ),
        (
            "=!|:",
            Barline::new(vec![BarlineStyle::Double], RepeatStyle::Start),
        ),
        (
            "=:|!|:",
            Barline::new(vec![BarlineStyle::Double], RepeatStyle::StartAndEnd),
        ),
        (
            "=-",
            Barline::new(vec![BarlineStyle::Invisible], RepeatStyle::None),
        ),
        (
            "=1-",
            Barline::new(vec![BarlineStyle::Invisible], RepeatStyle::None),
        ),
        (
            "=:!",
            Barline::new(vec![BarlineStyle::Double], RepeatStyle::End),
        ),
        (
            "=!:",
            Barline::new(vec![BarlineStyle::Double], RepeatStyle::Start),
        ),
        (
            "=12:!:",
            Barline::new(vec![BarlineStyle::Double], RepeatStyle::StartAndEnd),
        ),
        (
            "=:!!:",
            Barline::new(vec![BarlineStyle::Double], RepeatStyle::StartAndEnd),
        ),
        (
            "=:",
            Barline::new(vec![BarlineStyle::Dashed], RepeatStyle::None),
        ),
        (
            "=12:",
            Barline::new(vec![BarlineStyle::Dashed], RepeatStyle::None),
        ),
    ];

    for (raw, expected) in cases {
        assert_eq!(Barline::try_from(raw).unwrap(), expected);
    }
}

#[test]
fn parses_equivalent_barline_spellings() {
    assert_eq!(
        Barline::try_from("==:|!").unwrap(),
        Barline::try_from("=:|!").unwrap()
    );
    assert_eq!(
        Barline::try_from("=|!").unwrap(),
        Barline::try_from("=").unwrap()
    );
    assert_eq!(
        Barline::try_from("==||").unwrap(),
        Barline::try_from("==").unwrap()
    );
}

#[test]
fn kern_line_parses_barline_tokens() {
    let line = KernLine::parse(3, "==:|!").unwrap();

    assert_eq!(line, KernLine::Barline(Barline::try_from("=:|!").unwrap()));
}

#[test]
fn document_folds_equivalent_repeated_barline_tokens() {
    let document = KernDocument::parse("**kern\t**kern\n==:|!\t=:|!\n4c\t4e\n").unwrap();
    let barline = document
        .lines()
        .iter()
        .find_map(|line| match line {
            KernLine::Barline(barline) => Some(barline),
            _ => None,
        })
        .expect("expected parsed barline");

    assert_eq!(barline, &Barline::try_from("=:|!").unwrap());
}

#[test]
fn repeated_barlines_require_active_spine_count() {
    assert_eq!(
        KernDocument::parse("=\t=\n4c\t4e\n").unwrap_err(),
        ParseError::RepeatedBarlineWithoutActiveSpines { line: 1, found: 2 }
    );
}

#[test]
fn repeated_barlines_must_match_active_spine_count() {
    assert_eq!(
        KernDocument::parse("**kern\t**kern\n=\t=\t=\n4c\t4e\n").unwrap_err(),
        ParseError::SpineCountMismatch {
            line: 2,
            expected: 2,
            found: 3,
        }
    );
}

#[test]
fn preserves_mixed_barline_tokens_per_spine() {
    let document = KernDocument::parse("**kern\t**kern\n=2\t=2-\n4c\t4e\n").unwrap();
    let barline = document
        .lines()
        .iter()
        .find_map(|line| match line {
            KernLine::Barline(barline) => Some(barline),
            _ => None,
        })
        .unwrap();

    assert!(barline.is_spine_specific());
    assert_eq!(barline.spine_count(), Some(2));
    assert_eq!(barline.styles(), vec![BarlineStyle::Basic]);
    assert_eq!(barline.styles_for_spine(0), [BarlineStyle::Basic]);
    assert_eq!(barline.styles_for_spine(1), [BarlineStyle::Invisible]);
}

#[test]
fn aggregates_mixed_visible_styles_without_flattening_them() {
    let document = KernDocument::parse("**kern\t**kern\n=\t==\n4c\t4e\n").unwrap();
    let barline = document
        .lines()
        .iter()
        .find_map(|line| match line {
            KernLine::Barline(barline) => Some(barline),
            _ => None,
        })
        .unwrap();

    assert_eq!(barline.styles(), vec![BarlineStyle::Double]);
    assert_eq!(barline.styles_for_spine(0), [BarlineStyle::Basic]);
    assert_eq!(barline.styles_for_spine(1), [BarlineStyle::Double]);
}

#[test]
fn aggregates_dashed_spines_below_solid_ones() {
    let barline = |input: &str| {
        KernDocument::parse(input)
            .unwrap()
            .lines()
            .iter()
            .find_map(|line| match line {
                KernLine::Barline(barline) => Some(barline.clone()),
                _ => None,
            })
            .unwrap()
    };

    let mixed = barline("**kern\t**kern\n=:\t=\n4c\t4e\n");
    assert_eq!(mixed.styles(), vec![BarlineStyle::Basic]);
    assert_eq!(mixed.styles_for_spine(0), [BarlineStyle::Dashed]);

    let partly_hidden = barline("**kern\t**kern\n=:\t=-\n4c\t4e\n");
    assert_eq!(partly_hidden.styles(), vec![BarlineStyle::Dashed]);
}

#[test]
fn compact_single_barline_does_not_establish_document_width() {
    let document = KernDocument::parse("=\n4c\t4e\n").unwrap();

    assert_eq!(document.layout.initial_spine_count, 2);
    assert!(document
        .lines()
        .iter()
        .any(|line| matches!(line, KernLine::Barline(_))));
}

#[test]
fn compact_multi_spine_barline_serializes_per_active_spine() {
    let document =
        KernDocument::parse("**kern\t**kern\n*M4/4\t*M4/4\n4c\t4e\n=\n4d\t4f\n").unwrap();

    assert!(document.to_omr_string().contains("=\t=\n"));
}

#[test]
fn compact_repeat_barlines_parse_after_active_spine_count_is_known() {
    let document = KernDocument::parse(
        "**kern\t**kern\n4c\t4e\n=:|!\n4d\t4f\n=!|:\n4e\t4g\n=:|!|:\n4f\t4a\n==\n",
    )
    .unwrap();

    let barlines = document
        .lines()
        .iter()
        .filter(|line| matches!(line, KernLine::Barline(_)))
        .count();

    assert_eq!(barlines, 4);
}

#[test]
fn barline_parser_rejects_unknown_tokens() {
    for raw in ["=abc", "=1?", "=1-foo", "=-1", "=1--", "=A-", "=::", "==:"] {
        assert_eq!(
            Barline::try_from(raw).unwrap_err(),
            ParseError::InvalidBarline {
                raw: raw.to_owned(),
            }
        );
    }

    assert_eq!(
        KernLine::parse(3, "=abc").unwrap_err(),
        ParseError::InvalidSpineToken {
            line: 3,
            column: 1,
            raw: "=abc".to_owned(),
            source: Box::new(ParseError::InvalidBarline {
                raw: "=abc".to_owned(),
            }),
        }
    );
}
