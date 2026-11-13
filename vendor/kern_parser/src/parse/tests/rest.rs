use crate::types::{
    Breath, CrossStaffPlacement, DiatonicStep, DurationValue, Elision, Fermata, FermataPlacement,
    KernContent, Phrase, Slur, SlurPlacement,
};
use crate::*;

#[test]
fn parses_rest_type_definition_examples() {
    let cases = [
        (
            "4r",
            Rest {
                duration: Duration {
                    value: DurationValue::Reciprocal(4),
                    augmentation_dots: 0,
                },
                cross_staff: None,
                placement: None,
                invisible: false,
                breath: None,
                fermata: None,
                slurs: Vec::new(),
                phrases: Vec::new(),
            },
        ),
        (
            "2.r",
            Rest {
                duration: Duration {
                    value: DurationValue::Reciprocal(2),
                    augmentation_dots: 1,
                },
                cross_staff: None,
                placement: None,
                invisible: false,
                breath: None,
                fermata: None,
                slurs: Vec::new(),
                phrases: Vec::new(),
            },
        ),
        (
            "2.rr",
            Rest {
                duration: Duration {
                    value: DurationValue::Reciprocal(2),
                    augmentation_dots: 1,
                },
                cross_staff: None,
                placement: None,
                invisible: false,
                breath: None,
                fermata: None,
                slurs: Vec::new(),
                phrases: Vec::new(),
            },
        ),
        (
            "4ryy",
            Rest {
                duration: Duration {
                    value: DurationValue::Reciprocal(4),
                    augmentation_dots: 0,
                },
                cross_staff: None,
                placement: None,
                invisible: true,
                breath: None,
                fermata: None,
                slurs: Vec::new(),
                phrases: Vec::new(),
            },
        ),
    ];

    for (raw, expected) in cases {
        assert_eq!(Rest::try_from(raw).unwrap(), expected);
    }
}

#[test]
fn parses_long_rest_durations() {
    let cases = [
        ("0r", DurationValue::Breve, 0),
        ("00.r", DurationValue::Long, 1),
        ("000..r", DurationValue::Maxima, 2),
    ];

    for (raw, value, augmentation_dots) in cases {
        assert_eq!(
            Rest::try_from(raw).unwrap().duration,
            Duration {
                value,
                augmentation_dots,
            }
        );
    }
}

#[test]
fn parses_rest_phrase_markers() {
    assert_eq!(
        Rest::try_from("4r}").unwrap().phrases,
        vec![Phrase::Close {
            elision: Elision { depth: 0 },
        }]
    );
    assert_eq!(
        Rest::try_from("{8.r").unwrap().phrases,
        vec![Phrase::Open {
            elision: Elision { depth: 0 },
        }]
    );
    assert_eq!(
        Rest::try_from("&{4r&}").unwrap().phrases,
        vec![
            Phrase::Open {
                elision: Elision { depth: 1 },
            },
            Phrase::Close {
                elision: Elision { depth: 1 },
            },
        ]
    );
}

#[test]
fn parses_rdf_slur_placement_on_rest_slurs() {
    assert_eq!(
        Rest::try_from("&(<4r").unwrap().slurs,
        vec![Slur::Open {
            elided: true,
            depth: 1,
            count: 1,
            placement: Some(SlurPlacement::Below),
            linked: false,
        }]
    );
}

#[test]
fn parses_phrase_markers_with_other_rest_modifiers() {
    let rest = Rest::try_from("{4re}yy,;").unwrap();

    assert!(rest.invisible);
    assert_eq!(rest.breath, Some(Breath));
    assert_eq!(rest.fermata, Some(Fermata::default()));
    assert!(rest.placement.is_some());
    assert_eq!(
        rest.phrases,
        vec![
            Phrase::Open {
                elision: Elision { depth: 0 },
            },
            Phrase::Close {
                elision: Elision { depth: 0 },
            },
        ]
    );
}

#[test]
fn parses_rest_placement_examples() {
    let cases = [
        (
            "4re",
            RestPlacement {
                position: Pitch {
                    step: DiatonicStep::E,
                    octave: 4,
                },
            },
        ),
        (
            "4rg",
            RestPlacement {
                position: Pitch {
                    step: DiatonicStep::G,
                    octave: 4,
                },
            },
        ),
        (
            "16rF",
            RestPlacement {
                position: Pitch {
                    step: DiatonicStep::F,
                    octave: 3,
                },
            },
        ),
        (
            "8rGG",
            RestPlacement {
                position: Pitch {
                    step: DiatonicStep::G,
                    octave: 2,
                },
            },
        ),
        (
            "4rdd",
            RestPlacement {
                position: Pitch {
                    step: DiatonicStep::D,
                    octave: 5,
                },
            },
        ),
    ];

    for (raw, placement) in cases {
        assert_eq!(Rest::try_from(raw).unwrap().placement, Some(placement));
    }
}

#[test]
fn parses_rest_cross_staff_markers() {
    let cases = [
        ("4.r>yy", CrossStaffPlacement::Above, None),
        ("8r<", CrossStaffPlacement::Below, None),
        (
            "8r>ff",
            CrossStaffPlacement::Above,
            Some(RestPlacement {
                position: Pitch {
                    step: DiatonicStep::F,
                    octave: 5,
                },
            }),
        ),
        (
            "16r<Gyy",
            CrossStaffPlacement::Below,
            Some(RestPlacement {
                position: Pitch {
                    step: DiatonicStep::G,
                    octave: 3,
                },
            }),
        ),
        (
            "4rdd<yy",
            CrossStaffPlacement::Below,
            Some(RestPlacement {
                position: Pitch {
                    step: DiatonicStep::D,
                    octave: 5,
                },
            }),
        ),
    ];

    for (raw, cross_staff, placement) in cases {
        let rest = Rest::try_from(raw).unwrap();
        assert_eq!(rest.cross_staff, Some(cross_staff), "{raw}");
        assert_eq!(rest.placement, placement, "{raw}");
    }
}

#[test]
fn rejects_invalid_rest_cross_staff_markers() {
    for raw in ["4r><", "4r>d<", "4r>d>", "4rd>e", "4r<>yy"] {
        assert!(Rest::try_from(raw).is_err(), "{raw} should be invalid");
    }
}

#[test]
fn rest_parser_rejects_placements_outside_supported_pitch_range() {
    for raw in ["4rCCCCCC", "4rBBBBBB", "4rcccccc", "4rbbbbbb"] {
        assert!(Rest::try_from(raw).is_ok(), "{raw} should be valid");
    }

    for raw in ["4rCCCCCCC", "4rGGGGGGG", "4rccccccc", "4rggggggg"] {
        assert!(matches!(
            Rest::try_from(raw).unwrap_err(),
            ParseError::InvalidRest { .. }
        ));
    }
}

#[test]
fn parses_rest_breath_marks() {
    assert_eq!(Rest::try_from("0r,").unwrap().breath, Some(Breath));

    for raw in ["8re,", "4r,yy", "2r;,", "16rGGyy,;"] {
        assert_eq!(Rest::try_from(raw).unwrap().breath, Some(Breath));
    }
}

#[test]
fn rejects_duplicate_rest_breath_marks() {
    for raw in ["4r,,", "4r,yy,", "4r;,,"] {
        assert!(Rest::try_from(raw).is_err(), "{raw} should be invalid");
    }
}

#[test]
fn rejects_embedded_rest_breath_marks() {
    for raw in ["4r,e", "4re,y"] {
        assert!(Rest::try_from(raw).is_err(), "{raw} should be invalid");
    }
}

#[test]
fn parses_rest_invisible_and_fermata_examples() {
    for raw in ["2ryy", "4.ryy", "8reyy", "4rGGyy"] {
        assert!(Rest::try_from(raw).unwrap().invisible);
    }

    for raw in ["2r;", "8re;", "2.r;"] {
        assert_eq!(
            Rest::try_from(raw).unwrap().fermata,
            Some(Fermata::default())
        );
    }

    let rest = Rest::try_from("8r;yy").unwrap();
    assert!(rest.invisible);
    assert_eq!(rest.fermata, Some(Fermata::default()));
}

#[test]
fn kern_line_parses_rest_data_tokens() {
    let line = KernLine::parse(3, ".\t4r\t8reyy").unwrap();

    assert_eq!(
        line,
        KernLine::Data(vec![
            SpineContent::Kern(KernContent::Null),
            SpineContent::Kern(KernContent::Rest(Rest::try_from("4r").unwrap())),
            SpineContent::Kern(KernContent::Rest(Rest::try_from("8reyy").unwrap())),
        ])
    );
}

#[test]
fn kern_line_parses_multi_measure_rest_data_tokens() {
    let line = KernLine::parse(3, ".\tMM55r").unwrap();

    assert_eq!(
        line,
        KernLine::Data(vec![
            SpineContent::Kern(KernContent::Null),
            SpineContent::Kern(KernContent::MultiMeasureRest(MultiMeasureRest {
                measure_count: 55,
            })),
        ])
    );
}

#[test]
fn kern_line_rejects_invalid_multi_measure_rest_data_tokens() {
    for raw in ["MM0r", "MM1r", "MM005r", "MMr", "MM-4r", "MM3.5r", "MM2ryy"] {
        assert!(KernLine::parse(3, raw).is_err(), "{raw}");
    }
}

#[test]
fn parses_rdf_fermata_placement_on_rests() {
    let below = Rest::try_from("4r;<").unwrap();
    assert_eq!(
        below.fermata,
        Some(Fermata {
            placement: Some(FermataPlacement::Below)
        })
    );
    assert_eq!(below.cross_staff, None);

    let above = Rest::try_from("8reyy;>").unwrap();
    assert_eq!(
        above.fermata,
        Some(Fermata {
            placement: Some(FermataPlacement::Above)
        })
    );
    assert!(above.invisible);

    let cross_staff = Rest::try_from("4r<;").unwrap();
    assert_eq!(cross_staff.fermata, Some(Fermata { placement: None }));
    assert_eq!(cross_staff.cross_staff, Some(CrossStaffPlacement::Below));
}
