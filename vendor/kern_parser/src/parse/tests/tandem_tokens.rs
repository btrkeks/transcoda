use crate::*;

fn tandem_line(tokens: Vec<TandemInterpretation>) -> KernLine {
    KernLine::TandemInterpretationLine(TandemInterpretationLine::new(tokens).unwrap())
}

#[test]
fn parses_clef() {
    assert_eq!(
        Clef::try_from("*clefG2").unwrap(),
        Clef {
            sign: ClefSign::G,
            line: 2,
            octave_shift: 0,
            invisible: false,
        }
    );
    assert_eq!(
        Clef::try_from("*clefGvv2").unwrap(),
        Clef {
            sign: ClefSign::G,
            line: 2,
            octave_shift: -2,
            invisible: false,
        }
    );
    assert_eq!(
        Clef::try_from("*clefG^^2").unwrap(),
        Clef {
            sign: ClefSign::G,
            line: 2,
            octave_shift: 2,
            invisible: false,
        }
    );
    assert_eq!(
        Clef::try_from("*clefX").unwrap(),
        Clef {
            sign: ClefSign::Percussion,
            line: 0,
            octave_shift: 0,
            invisible: false,
        }
    );
    assert_eq!(
        Clef::try_from("*clefX3").unwrap(),
        Clef {
            sign: ClefSign::Percussion,
            line: 3,
            octave_shift: 0,
            invisible: false,
        }
    );
    assert_eq!(
        Clef::try_from("*clefTAB").unwrap(),
        Clef {
            sign: ClefSign::Tab,
            line: 0,
            octave_shift: 0,
            invisible: false,
        }
    );
    assert_eq!(
        Clef::try_from("*clefTAB5").unwrap(),
        Clef {
            sign: ClefSign::Tab,
            line: 5,
            octave_shift: 0,
            invisible: false,
        }
    );
}

#[test]
fn parses_invisible_clefs() {
    for (raw, expected) in [
        (
            "*clefG2yy",
            Clef {
                sign: ClefSign::G,
                line: 2,
                octave_shift: 0,
                invisible: true,
            },
        ),
        (
            "*clefGvv2yy",
            Clef {
                sign: ClefSign::G,
                line: 2,
                octave_shift: -2,
                invisible: true,
            },
        ),
        (
            "*clefXyy",
            Clef {
                sign: ClefSign::Percussion,
                line: 0,
                octave_shift: 0,
                invisible: true,
            },
        ),
        (
            "*clefTAB5yy",
            Clef {
                sign: ClefSign::Tab,
                line: 5,
                octave_shift: 0,
                invisible: true,
            },
        ),
    ] {
        assert_eq!(Clef::try_from(raw).unwrap(), expected, "{raw}");
    }
}

#[test]
fn rejects_invalid_clef() {
    let error = Clef::try_from("*clefGv^2").unwrap_err();

    assert_eq!(
        error,
        ParseError::InvalidClef {
            raw: "*clefGv^2".to_owned()
        }
    );

    for raw in [
        "*clefXv2",
        "*clefX^2",
        "*clefTABv5",
        "*clefTAB^5",
        "*clefG2y",
        "*clefG2yyyy",
        "*clefGyy2",
    ] {
        assert_eq!(
            Clef::try_from(raw).unwrap_err(),
            ParseError::InvalidClef {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn clef_parser_rejects_full_humdrum_line() {
    let error = Clef::try_from("*clefG2\t*clefF4").unwrap_err();

    assert_eq!(
        error,
        ParseError::InvalidClef {
            raw: "*clefG2\t*clefF4".to_owned()
        }
    );
}

#[test]
fn parses_meter_symbols_and_round_trips_them() {
    let cases = [
        ("*met(c)", MeterSymbol::CommonTime),
        ("*met(c|)", MeterSymbol::CutTime),
        (
            "*met(C)",
            MeterSymbol::Mensural(MensurationSign::TempusImperfectum),
        ),
        (
            "*met(C|)",
            MeterSymbol::Mensural(MensurationSign::TempusImperfectumDiminutum),
        ),
        ("*met()", MeterSymbol::Empty),
    ];

    for (raw, symbol) in cases {
        assert_eq!(MeterSymbol::try_from(raw).unwrap(), symbol);
        assert_eq!(symbol.as_kern(), raw);
    }
}

#[test]
fn meter_symbol_parser_rejects_invalid_tokens() {
    for raw in ["*metC", "*met( )", "*met(O)", "*met(C", "*metC)"] {
        assert_eq!(
            MeterSymbol::try_from(raw).unwrap_err(),
            ParseError::InvalidMeterSymbol {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn parses_stria() {
    for (raw, count) in [("*stria0", 0), ("*stria1", 1), ("*stria5", 5)] {
        assert_eq!(Stria::try_from(raw).unwrap(), Stria { count });
    }
}

#[test]
fn stria_parser_rejects_invalid_tokens() {
    for raw in ["*stria", "*stria-1", "*striax", "*stria256"] {
        assert_eq!(
            Stria::try_from(raw).unwrap_err(),
            ParseError::InvalidStria {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn parses_part_labels() {
    for (raw, number) in [("*part1", 1), ("*part42", 42)] {
        assert_eq!(Part::try_from(raw).unwrap(), Part { number });
    }
}

#[test]
fn part_parser_rejects_invalid_tokens() {
    for raw in ["*part", "*part0", "*part-1", "*partA", "*part1a"] {
        assert_eq!(
            Part::try_from(raw).unwrap_err(),
            ParseError::InvalidPart {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn kern_line_parses_part_labels() {
    let line = KernLine::parse(3, "*part2\t*part1").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Part(Part { number: 2 }),
            TandemInterpretation::Part(Part { number: 1 }),
        ])
    );
}

#[test]
fn kern_line_parses_repeated_part_labels() {
    let line = KernLine::parse(3, "*part2\t*part2\t*part1\t*part1").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Part(Part { number: 2 }),
            TandemInterpretation::Part(Part { number: 2 }),
            TandemInterpretation::Part(Part { number: 1 }),
            TandemInterpretation::Part(Part { number: 1 }),
        ])
    );
}

#[test]
fn kern_line_parses_part_labels_with_nulls() {
    let line = KernLine::parse(3, "*part2\t*\t*part1").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Part(Part { number: 2 }),
            TandemInterpretation::Null,
            TandemInterpretation::Part(Part { number: 1 }),
        ])
    );
}

#[test]
fn parses_staff_labels() {
    for (raw, number) in [("*staff2", 2), ("*staff4", 4), ("*staff10", 10)] {
        assert_eq!(
            Staff::try_from(raw).unwrap(),
            Staff {
                number,
                also: vec![]
            }
        );
    }
}

#[test]
fn parses_composite_staff_labels() {
    for (raw, number, also) in [
        ("*staff1/2", 1, vec![2]),
        ("*staff2/1", 2, vec![1]),
        ("*staff10/11/12", 10, vec![11, 12]),
    ] {
        let staff = Staff::try_from(raw).unwrap();
        assert_eq!(staff, Staff { number, also });
        assert_eq!(staff.as_kern(), raw);
    }
}

#[test]
fn staff_parser_rejects_invalid_tokens() {
    for raw in [
        "*staff",
        "*staff0",
        "*staff-1",
        "*staffA",
        "*staff1a",
        "*staff1/",
        "*staff/1",
        "*staff1/0",
        "*staff1//2",
        "*staff1/A",
    ] {
        assert_eq!(
            Staff::try_from(raw).unwrap_err(),
            ParseError::InvalidStaff {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn kern_line_parses_staff_labels() {
    let line = KernLine::parse(3, "*staff2\t*staff4\t*\t*staff10").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Staff(Staff {
                number: 2,
                also: vec![]
            }),
            TandemInterpretation::Staff(Staff {
                number: 4,
                also: vec![]
            }),
            TandemInterpretation::Null,
            TandemInterpretation::Staff(Staff {
                number: 10,
                also: vec![]
            }),
        ])
    );
}

#[test]
fn parses_group_labels() {
    for (raw, number) in [("*group1", 1), ("*group42", 42)] {
        assert_eq!(Group::try_from(raw).unwrap(), Group { number });
    }
}

#[test]
fn parses_composite_group_labels() {
    for (raw, number) in [("*group1/2", 1), ("*group2/1", 2), ("*group10/11/12", 10)] {
        assert_eq!(Group::try_from(raw).unwrap(), Group { number });
    }
}

#[test]
fn group_parser_rejects_invalid_tokens() {
    for raw in [
        "*group",
        "*group0",
        "*group-1",
        "*groupA",
        "*group1a",
        "*group1/",
        "*group/1",
        "*group1/0",
        "*group1//2",
        "*group1/A",
    ] {
        assert_eq!(
            Group::try_from(raw).unwrap_err(),
            ParseError::InvalidGroup {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn kern_line_parses_group_labels() {
    let line = KernLine::parse(3, "*group2\t*group4\t*\t*group10").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Group(Group { number: 2 }),
            TandemInterpretation::Group(Group { number: 4 }),
            TandemInterpretation::Null,
            TandemInterpretation::Group(Group { number: 10 }),
        ])
    );
}

#[test]
fn parses_tempo() {
    assert_eq!(
        Tempo::try_from("*MM96").unwrap(),
        Tempo {
            beats_per_minute: DecimalTempo {
                units: 96,
                scale: 0,
            },
        }
    );
    assert_eq!(
        Tempo::try_from("*MM22.5").unwrap(),
        Tempo {
            beats_per_minute: DecimalTempo {
                units: 225,
                scale: 1,
            },
        }
    );
    assert_eq!(
        Tempo::try_from("*MM618.05").unwrap(),
        Tempo {
            beats_per_minute: DecimalTempo {
                units: 61805,
                scale: 2,
            },
        }
    );
}

#[test]
fn tempo_parser_rejects_invalid_tokens() {
    for raw in [
        "*MM",
        "*MM0",
        "*MM0.0",
        "*MM.5",
        "*MM96.",
        "*MM96..3",
        "*MMfast",
        "*MM1e2",
        "*MM4294967296",
    ] {
        assert_eq!(
            Tempo::try_from(raw).unwrap_err(),
            ParseError::InvalidTempo {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn kern_line_parses_new_tandem_interpretations() {
    let cases = [
        (
            "*\t*met(c|)",
            TandemInterpretation::MeterSymbol(MeterSymbol::CutTime),
        ),
        (
            "*\t*stria5",
            TandemInterpretation::Stria(Stria { count: 5 }),
        ),
        (
            "*\t*MM96",
            TandemInterpretation::Tempo(Tempo {
                beats_per_minute: DecimalTempo {
                    units: 96,
                    scale: 0,
                },
            }),
        ),
        (
            "*\t*clefX",
            TandemInterpretation::Clef(Clef {
                sign: ClefSign::Percussion,
                line: 0,
                octave_shift: 0,
                invisible: false,
            }),
        ),
        (
            "*\t*clefTAB",
            TandemInterpretation::Clef(Clef {
                sign: ClefSign::Tab,
                line: 0,
                octave_shift: 0,
                invisible: false,
            }),
        ),
        (
            "*\t*2\\right",
            TandemInterpretation::VisualStyle(VisualStyle::HalfNoteStemSide(
                HalfNoteStemSide::Right,
            )),
        ),
        (
            "*\t*2\\left",
            TandemInterpretation::VisualStyle(VisualStyle::HalfNoteStemSide(
                HalfNoteStemSide::Left,
            )),
        ),
    ];

    for (raw, tandem) in cases {
        assert_eq!(
            KernLine::parse(3, raw).unwrap(),
            tandem_line(vec![TandemInterpretation::Null, tandem])
        );
    }
}

#[test]
fn parses_hidden_time_signature_and_round_trips_it() {
    for (raw, invisible) in [("*M3/4", false), ("*M3/4yy", true)] {
        let tandem = TimeSignatureTandem::try_from(raw).unwrap();

        assert_eq!(
            tandem,
            TimeSignatureTandem {
                time_signature: TimeSignature {
                    numerator: 3,
                    denominator: 4,
                },
                invisible,
            }
        );
        assert_eq!(tandem.as_kern(), raw);
    }
}

#[test]
fn time_signature_parser_rejects_other_suffixes() {
    for raw in ["*M3/4y", "*M3/4yyy", "*M3/4x"] {
        assert_eq!(
            TimeSignatureTandem::try_from(raw).unwrap_err(),
            ParseError::InvalidTimeSignature {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn parses_free_meter() {
    assert_eq!(
        TandemInterpretation::try_from("*MX").unwrap(),
        TandemInterpretation::FreeMeter
    );
    for raw in ["*MXyy", "*Mx", "*MX/4"] {
        assert!(TandemInterpretation::try_from(raw).is_err(), "{raw}");
    }
}
