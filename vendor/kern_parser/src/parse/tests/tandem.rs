use crate::types::{
    BracketSpan, BracketSpanKind, ExpansionItem, ExpansionList, Ottava, OttavaKind, Pedal,
    SectionLabel, TimeSignature, TimeSignatureTandem, Tremolo, Tuplet,
};
use crate::*;

fn tandem_line(tokens: Vec<TandemInterpretation>) -> KernLine {
    KernLine::TandemInterpretationLine(TandemInterpretationLine::new(tokens).unwrap())
}

#[test]
fn parse_record_groups_mixed_tandems_in_first_occurrence_order() {
    let lines = KernLine::parse_record(3, "*\t*Xtuplet\t*ped\t*tuplet\t*Xped").unwrap();

    assert_eq!(
        lines,
        vec![
            tandem_line(vec![
                TandemInterpretation::Null,
                TandemInterpretation::Tuplet(Tuplet { show_number: false }),
                TandemInterpretation::Null,
                TandemInterpretation::Tuplet(Tuplet { show_number: true }),
                TandemInterpretation::Null,
            ]),
            tandem_line(vec![
                TandemInterpretation::Null,
                TandemInterpretation::Null,
                TandemInterpretation::Pedal(Pedal { active: true }),
                TandemInterpretation::Null,
                TandemInterpretation::Pedal(Pedal { active: false }),
            ]),
        ]
    );
}

#[test]
fn parse_record_preserves_single_lines() {
    for raw in ["*", "*\t*", "*ped\t*", "4c\t4e", "*^\t*", "**kern\t**kern"] {
        assert_eq!(
            KernLine::parse_record(3, raw).unwrap(),
            vec![KernLine::parse(3, raw).unwrap()],
            "{raw}"
        );
    }
}

#[test]
fn parse_record_validates_all_tokens_before_lowering_mixed_tandems() {
    assert_eq!(
        KernLine::parse_record(7, "*ped\t*Xtuplet\t*clefINVALID").unwrap_err(),
        ParseError::InvalidSpineToken {
            line: 7,
            column: 3,
            raw: "*clefINVALID".to_owned(),
            source: Box::new(ParseError::InvalidClef {
                raw: "*clefINVALID".to_owned(),
            }),
        }
    );
    for (raw, found) in [
        ("*ped\t*Xtuplet\t*^", "spine manipulator"),
        ("*ped\t*Xtuplet\t4c", "data"),
    ] {
        assert_eq!(
            KernLine::parse_record(7, raw).unwrap_err(),
            ParseError::MixedKernLineTypes {
                line: 7,
                column: 3,
                expected: "tandem interpretation",
                found,
            }
        );
    }
}

#[test]
fn kern_line_parses_single_null_interpretation_token_line() {
    let line = KernLine::parse(3, "*").unwrap();

    assert_eq!(line, tandem_line(vec![TandemInterpretation::Null]));
}

#[test]
fn kern_line_parses_all_null_interpretation_token_line() {
    let line = KernLine::parse(3, "*\t*\t*").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::Null,
            TandemInterpretation::Null,
        ])
    );
}

#[test]
fn kern_line_parses_nulls_with_ottava_tandem() {
    let line = KernLine::parse(3, "*\t*\t*\t*8va").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::Null,
            TandemInterpretation::Null,
            TandemInterpretation::Ottava(Ottava {
                kind: OttavaKind::OctaveUp,
                active: true,
            }),
        ])
    );
}

#[test]
fn kern_line_parses_nulls_with_tremolo_tandem() {
    let line = KernLine::parse(3, "*\t*\t*tremolo\t*").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::Null,
            TandemInterpretation::Tremolo(Tremolo { active: true }),
            TandemInterpretation::Null,
        ])
    );
}

#[test]
fn kern_line_parses_nulls_with_pedal_tandem() {
    let line = KernLine::parse(3, "*\t*\t*ped\t*").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::Null,
            TandemInterpretation::Pedal(Pedal { active: true }),
            TandemInterpretation::Null,
        ])
    );
}

#[test]
fn kern_line_parses_nulls_with_tuplet_tandem() {
    let line = KernLine::parse(3, "*\t*\t*tuplet\t*").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::Null,
            TandemInterpretation::Tuplet(Tuplet { show_number: true }),
            TandemInterpretation::Null,
        ])
    );
}

#[test]
fn kern_line_parses_nulls_with_bracket_span_tandem() {
    let line = KernLine::parse(3, "*\t*\t*lig\t*").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::Null,
            TandemInterpretation::BracketSpan(BracketSpan {
                kind: BracketSpanKind::Ligature,
                active: true,
            }),
            TandemInterpretation::Null,
        ])
    );
}

#[test]
fn kern_line_parses_tremolo_stop_tandem() {
    let line = KernLine::parse(3, "*Xtremolo\t*").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Tremolo(Tremolo { active: false }),
            TandemInterpretation::Null,
        ])
    );
}

#[test]
fn kern_line_parses_pedal_stop_tandem() {
    let line = KernLine::parse(3, "*Xped\t*").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Pedal(Pedal { active: false }),
            TandemInterpretation::Null,
        ])
    );
}

#[test]
fn kern_line_parses_tuplet_stop_tandem() {
    let line = KernLine::parse(3, "*Xtuplet\t*").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Tuplet(Tuplet { show_number: false }),
            TandemInterpretation::Null,
        ])
    );
}

#[test]
fn kern_line_parses_bracket_span_stop_tandems() {
    let cases = [
        ("*Xlig\t*", BracketSpanKind::Ligature),
        ("*Xcol\t*", BracketSpanKind::Coloration),
    ];

    for (raw, kind) in cases {
        assert_eq!(
            KernLine::parse(3, raw).unwrap(),
            tandem_line(vec![
                TandemInterpretation::BracketSpan(BracketSpan {
                    kind,
                    active: false,
                }),
                TandemInterpretation::Null,
            ])
        );
    }
}

#[test]
fn kern_line_parses_coloration_start_tandem() {
    let line = KernLine::parse(3, "*\t*col").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::BracketSpan(BracketSpan {
                kind: BracketSpanKind::Coloration,
                active: true,
            }),
        ])
    );
}

#[test]
fn kern_line_parses_section_label_tandems() {
    let line = KernLine::parse(3, "*>A\t*>2nd ending\t*>Form>>1st Theme").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::SectionLabel(SectionLabel {
                label: "A".to_owned(),
            }),
            TandemInterpretation::SectionLabel(SectionLabel {
                label: "2nd ending".to_owned(),
            }),
            TandemInterpretation::SectionLabel(SectionLabel {
                label: "Form>>1st Theme".to_owned(),
            }),
        ])
    );
}

#[test]
fn kern_line_parses_expansion_list_tandems() {
    let line = KernLine::parse(3, "*>[A,A,B,B]\t*>norep[Smith>A,Jones>B]").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::ExpansionList(ExpansionList {
                version: None,
                items: vec![
                    ExpansionItem::Label("A".to_owned()),
                    ExpansionItem::Label("A".to_owned()),
                    ExpansionItem::Label("B".to_owned()),
                    ExpansionItem::Label("B".to_owned()),
                ],
            }),
            TandemInterpretation::ExpansionList(ExpansionList {
                version: Some("norep".to_owned()),
                items: vec![
                    ExpansionItem::Label("Smith>A".to_owned()),
                    ExpansionItem::Label("Jones>B".to_owned()),
                ],
            }),
        ])
    );
}

#[test]
fn expansion_list_parser_preserves_empty_items() {
    assert_eq!(
        ExpansionList::try_from("*>ossia[Minuet,Minuet,Trio,Trio,,Minuet]").unwrap(),
        ExpansionList {
            version: Some("ossia".to_owned()),
            items: vec![
                ExpansionItem::Label("Minuet".to_owned()),
                ExpansionItem::Label("Minuet".to_owned()),
                ExpansionItem::Label("Trio".to_owned()),
                ExpansionItem::Label("Trio".to_owned()),
                ExpansionItem::Empty,
                ExpansionItem::Label("Minuet".to_owned()),
            ],
        }
    );
}

#[test]
fn kern_line_parses_nulls_with_meter_tandem() {
    let line = KernLine::parse(3, "*\t*M6/8\t*\t*").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::TimeSignature(TimeSignatureTandem::visible(TimeSignature {
                numerator: 6,
                denominator: 8,
            })),
            TandemInterpretation::Null,
            TandemInterpretation::Null,
        ])
    );
}

#[test]
fn kern_line_rejects_mixed_tandem_interpretation_kinds() {
    assert_eq!(
        KernLine::parse(3, "*\t*M4/4\t*clefG2").unwrap_err(),
        ParseError::MixedTandemInterpretationTypes {
            line: 3,
            column: 3,
            expected: "time signature",
            found: "clef",
        }
    );
}

#[test]
fn kern_line_rejects_mixed_tuplet_tandem_interpretation_kinds() {
    assert_eq!(
        KernLine::parse(3, "*\t*tuplet\t*clefG2").unwrap_err(),
        ParseError::MixedTandemInterpretationTypes {
            line: 3,
            column: 3,
            expected: "tuplet",
            found: "clef",
        }
    );
}

#[test]
fn bracket_span_parser_rejects_invalid_tokens() {
    for raw in ["*lig2", "*Xligature", "*color", "*Xcolor", "*ligature"] {
        assert_eq!(
            BracketSpan::try_from(raw).unwrap_err(),
            ParseError::InvalidBracketSpan {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn section_label_parser_rejects_invalid_tokens() {
    for raw in ["*>", "*>[]", "*>A[B]", "*>A,B"] {
        assert_eq!(
            SectionLabel::try_from(raw).unwrap_err(),
            ParseError::InvalidSectionLabel {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn expansion_list_parser_rejects_invalid_tokens() {
    for raw in [
        "*>[A",
        "*>A]",
        "*>long[A]tail",
        "*>bad/version[A]",
        "*>[A[B]]",
    ] {
        assert_eq!(
            ExpansionList::try_from(raw).unwrap_err(),
            ParseError::InvalidExpansionList {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn pedal_parser_rejects_invalid_tokens() {
    for raw in ["*Ped", "*xped", "*pedal", "*XPed"] {
        assert_eq!(
            Pedal::try_from(raw).unwrap_err(),
            ParseError::InvalidPedal {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn tuplet_parser_rejects_invalid_tokens() {
    for raw in ["*Tuplet", "*xtuplet", "*tuplet2", "*XTuplet"] {
        assert_eq!(
            Tuplet::try_from(raw).unwrap_err(),
            ParseError::InvalidTuplet {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn meter_parser_rejects_invalid_time_signature() {
    let error = TimeSignature::try_from("*M6/0").unwrap_err();

    assert_eq!(
        error,
        ParseError::InvalidTimeSignature {
            raw: "*M6/0".to_owned(),
        }
    );
}
