use crate::types::DiatonicStep;
use crate::*;

#[test]
fn mixed_tandem_records_lower_to_homogeneous_lines_with_source_locations() {
    let document = KernDocument::parse(
        "**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n4C\t4e\n*ped\t*Xtuplet\n4D\t4f\n*Xped\t*tuplet\n*-\t*-\n",
    )
    .unwrap();

    assert_eq!(document.lines().len(), 10);
    for (index, kind, tokens, source) in [
        (
            4,
            TandemInterpretationLineKind::Pedal,
            vec![
                TandemInterpretation::Pedal(Pedal { active: true }),
                TandemInterpretation::Null,
            ],
            5,
        ),
        (
            5,
            TandemInterpretationLineKind::Tuplet,
            vec![
                TandemInterpretation::Null,
                TandemInterpretation::Tuplet(Tuplet { show_number: false }),
            ],
            5,
        ),
        (
            7,
            TandemInterpretationLineKind::Pedal,
            vec![
                TandemInterpretation::Pedal(Pedal { active: false }),
                TandemInterpretation::Null,
            ],
            7,
        ),
        (
            8,
            TandemInterpretationLineKind::Tuplet,
            vec![
                TandemInterpretation::Null,
                TandemInterpretation::Tuplet(Tuplet { show_number: true }),
            ],
            7,
        ),
    ] {
        let KernLine::TandemInterpretationLine(line) = &document.lines()[index] else {
            panic!("expected a tandem line at index {index}");
        };
        assert_eq!(line.kind(), kind);
        assert_eq!(line.tokens(), tokens);
        assert_eq!(document.source(index), Some(source));
    }
    assert_eq!(document.source(6), Some(6));
    assert_eq!(document.source(9), Some(8));
}

#[test]
fn pitch_order_follows_diatonic_staff_position() {
    let mut pitches = vec![
        Pitch {
            step: DiatonicStep::B,
            octave: 3,
        },
        Pitch {
            step: DiatonicStep::C,
            octave: 4,
        },
        Pitch {
            step: DiatonicStep::A,
            octave: 3,
        },
        Pitch {
            step: DiatonicStep::C,
            octave: 3,
        },
    ];

    pitches.sort();

    assert_eq!(
        pitches,
        vec![
            Pitch {
                step: DiatonicStep::C,
                octave: 3,
            },
            Pitch {
                step: DiatonicStep::A,
                octave: 3,
            },
            Pitch {
                step: DiatonicStep::B,
                octave: 3,
            },
            Pitch {
                step: DiatonicStep::C,
                octave: 4,
            },
        ]
    );
}

#[test]
fn parses_formal_typed_kern_document() {
    let input = [
        "!!!OTL: Prelude",
        "!!!COM: Example Composer",
        "!! ignored global comment",
        "**kern\t**kern",
        "*M4/4\t*M4/4",
        "! ignored\t! local comment",
        "=1\t=1",
        "4c\t4e",
        "*-\t*-",
    ]
    .join("\n");

    let document = KernDocument::parse(&input).unwrap();

    assert_eq!(
        document.metadata,
        KernMetadata {
            title: Some("Prelude".to_owned()),
            composer: Some("Example Composer".to_owned()),
            ..Default::default()
        }
    );
    assert_eq!(document.layout.initial_spine_count, 2);
    assert_eq!(document.lines().len(), 7);
    assert!(matches!(document.lines()[0], KernLine::SpineTypeLine(_)));
    assert!(matches!(
        document.lines()[1],
        KernLine::TandemInterpretationLine(_)
    ));
    assert!(matches!(
        document.lines()[2],
        KernLine::TandemInterpretationLine(_)
    ));
    assert!(matches!(
        document.lines()[3],
        KernLine::TandemInterpretationLine(_)
    ));
    assert!(matches!(document.lines()[4], KernLine::Barline(_)));
    assert!(matches!(document.lines()[5], KernLine::Data(_)));
    assert!(matches!(
        document.lines()[6],
        KernLine::SpineManipulatorLine(_)
    ));
    assert_eq!(document.source(1), None);
    assert_eq!(document.source(2), None);
}

#[test]
fn parses_normalized_slice_without_document_wrappers() {
    let input = "*clefF4\t*clefG2\n*k[]\t*k[]\n*M3/4\t*M3/4\n4r\t4g\n.\t4e\n==\t==\n";

    let document = KernDocument::parse(input).unwrap();

    assert_eq!(document.metadata, KernMetadata::default());
    assert_eq!(document.layout.initial_spine_count, 2);
    assert_eq!(document.lines().len(), 6);
    assert!(matches!(
        document.lines()[0],
        KernLine::TandemInterpretationLine(_)
    ));
    assert!(matches!(document.lines()[3], KernLine::Data(_)));
    assert!(matches!(document.lines()[5], KernLine::Barline(_)));
}

#[test]
fn parse_finalizes_missing_initial_clef_and_key_state() {
    let document = KernDocument::parse("4a\t4c\t4cc\n4b\t4d\t4dd\n").unwrap();

    assert_eq!(document.layout.initial_spine_count, 3);
    assert!(matches!(
        &document.lines()[0],
        KernLine::TandemInterpretationLine(tandems)
            if tandems
                .tokens()
                .iter()
                .all(|token| matches!(token, TandemInterpretation::Clef(_)))
    ));
    assert!(matches!(
        &document.lines()[1],
        KernLine::TandemInterpretationLine(tandems)
            if tandems
                .tokens()
                .iter()
                .all(|token| matches!(token, TandemInterpretation::KeySignature(_)))
    ));
    assert!(matches!(document.lines()[2], KernLine::Data(_)));
    assert_eq!(document.source(0), None);
    assert_eq!(document.source(1), None);
}

#[test]
fn parse_finalizes_single_spine_missing_initial_state() {
    let document = KernDocument::parse("4a\n4b\n").unwrap();

    assert!(matches!(
        &document.lines()[0],
        KernLine::TandemInterpretationLine(tandems)
            if tandems.len() == 1 && matches!(tandems.tokens()[0], TandemInterpretation::Clef(_))
    ));
    assert!(matches!(
        &document.lines()[1],
        KernLine::TandemInterpretationLine(tandems)
            if tandems.len() == 1
                && matches!(tandems.tokens()[0], TandemInterpretation::KeySignature(_))
    ));
    assert!(matches!(document.lines()[2], KernLine::Data(_)));
}

#[test]
fn parse_finalizes_only_kern_spines() {
    let document = KernDocument::parse("**kern\t**dynam\t**kern\n4a\tp\t4cc\n").unwrap();

    assert!(matches!(
        &document.lines()[1],
        KernLine::TandemInterpretationLine(tandems)
            if matches!(tandems.tokens()[0], TandemInterpretation::Clef(_))
                && matches!(tandems.tokens()[1], TandemInterpretation::Null)
                && matches!(tandems.tokens()[2], TandemInterpretation::Clef(_))
    ));
    assert!(matches!(
        &document.lines()[2],
        KernLine::TandemInterpretationLine(tandems)
            if matches!(tandems.tokens()[0], TandemInterpretation::KeySignature(_))
                && matches!(tandems.tokens()[1], TandemInterpretation::Null)
                && matches!(tandems.tokens()[2], TandemInterpretation::KeySignature(_))
    ));
}

#[test]
fn parse_finalizer_fills_sparse_initial_state_lines() {
    let document = KernDocument::parse("**kern\t**kern\n*clefF4\t*\n4a\t4c\n").unwrap();

    assert!(matches!(
        &document.lines()[1],
        KernLine::TandemInterpretationLine(tandems)
            if matches!(tandems.tokens()[0], TandemInterpretation::Clef(_))
                && matches!(tandems.tokens()[1], TandemInterpretation::Clef(_))
    ));
    assert!(matches!(
        &document.lines()[2],
        KernLine::TandemInterpretationLine(tandems)
            if tandems
                .tokens()
                .iter()
                .all(|token| matches!(token, TandemInterpretation::KeySignature(_)))
    ));
}

#[test]
fn parse_finalizer_scans_across_leading_barline_before_data() {
    let document = KernDocument::parse("**kern\n=1\n*clefF4\n4a\n").unwrap();

    assert!(matches!(document.lines()[1], KernLine::Barline(_)));
    assert!(matches!(
        &document.lines()[2],
        KernLine::TandemInterpretationLine(tandems)
            if matches!(tandems.tokens()[0], TandemInterpretation::Clef(_))
    ));
    assert!(matches!(
        &document.lines()[3],
        KernLine::TandemInterpretationLine(tandems)
            if matches!(tandems.tokens()[0], TandemInterpretation::KeySignature(_))
    ));
}

#[test]
fn kern_document_extracts_supported_metadata_and_ignores_unsupported_records() {
    let input = [
        "!!!OTL: First title",
        "!!!OTL: Final title",
        "!!!OPS: Op. 1",
        "!!!OMV: II",
        "!!!OMD: Allegro",
        "!!!ODT: 1722",
        "!!!URL: https://example.test/source.krn",
        "!!!SCT: shelfmark",
        "!!!XYZ: ignored",
        "!! ignored",
        "! ignored",
        "*clefG2",
        "4c",
    ]
    .join("\n");

    let document = KernDocument::parse(&input).unwrap();

    assert_eq!(
        document.metadata,
        KernMetadata {
            title: Some("Final title".to_owned()),
            composer: None,
            opus: Some("Op. 1".to_owned()),
            movement: Some("II".to_owned()),
            movement_designation: Some("Allegro".to_owned()),
            date: Some("1722".to_owned()),
            source: Some("shelfmark".to_owned()),
            ..Default::default()
        }
    );
    assert_eq!(document.layout.initial_spine_count, 1);
    assert_eq!(document.lines().len(), 3);
}

#[test]
fn kern_document_prefers_spine_type_line_for_initial_spine_count() {
    let document = KernDocument::parse("**kern\t**kern\t**kern\n4c\t4e\t4g\n").unwrap();

    assert_eq!(document.layout.initial_spine_count, 3);
}

#[test]
fn kern_document_counts_initial_kern_spines_as_staves() {
    let document = KernDocument::parse("**kern\t**kern\t**kern\n4c\t4e\t4g\n").unwrap();

    assert_eq!(document.staff_count(), 3);
}

#[test]
fn kern_document_staff_count_ignores_initial_non_kern_spines() {
    let document =
        KernDocument::parse("**kern\t**dynam\t**text\t**mxhm\t**kern\n4c\tmf\tDo-\tG\t4e\n")
            .unwrap();

    assert_eq!(document.layout.initial_spine_count, 5);
    assert_eq!(document.staff_count(), 2);
}

#[test]
fn kern_document_staff_count_treats_wrapperless_slices_as_kern_spines() {
    let document = KernDocument::parse("4c\t4e\n").unwrap();

    assert_eq!(document.staff_count(), 2);
}

#[test]
fn parses_ornament_convention_reference_record() {
    let document = KernDocument::parse(
        "!!!COM: Bach, Johann Sebastian\n!!!ornament-convention: baroque\n**kern\n4c\n*-\n",
    )
    .unwrap();

    assert_eq!(
        document.metadata.ornament_convention,
        Some(OrnamentConvention::Baroque)
    );
    assert_eq!(
        KernDocument::parse("**kern\n4c\n*-\n")
            .unwrap()
            .metadata
            .ornament_convention,
        None
    );
}

#[test]
fn parses_every_ornament_convention_record_value() {
    for (value, expected) in [
        ("baroque", OrnamentConvention::Baroque),
        ("classical", OrnamentConvention::Classical),
        ("modern", OrnamentConvention::Modern),
        ("baroque-v2", OrnamentConvention::BaroqueV2),
        ("classical-v2", OrnamentConvention::ClassicalV2),
        ("modern-v2", OrnamentConvention::ModernV2),
    ] {
        let document = KernDocument::parse(&format!(
            "!!!ornament-convention: {value}\n**kern\n4c\n*-\n"
        ))
        .unwrap();
        assert_eq!(
            document.metadata.ornament_convention,
            Some(expected),
            "{value}"
        );
    }
}

#[test]
fn unknown_or_repeated_ornament_convention_is_a_parse_error() {
    for value in [
        "barock",
        "baroque-v3",
        "baroquev2",
        "baroque_v2",
        "Baroque-V2",
    ] {
        let unknown = KernDocument::parse(&format!(
            "!!!ornament-convention: {value}\n**kern\n4c\n*-\n"
        ))
        .unwrap_err();
        assert!(
            matches!(
                unknown,
                ParseError::InvalidOrnamentConvention { line: 1, .. }
            ),
            "{value}: {unknown:?}"
        );
    }

    let changed = KernDocument::parse(
        "!!!ornament-convention: baroque\n!!!ornament-convention: baroque-v2\n**kern\n4c\n*-\n",
    )
    .unwrap_err();
    assert!(
        matches!(
            changed,
            ParseError::InvalidOrnamentConvention { line: 2, .. }
        ),
        "{changed:?}"
    );

    let repeated = KernDocument::parse(
        "!!!ornament-convention: baroque\n!!!ornament-convention: baroque\n**kern\n4c\n*-\n",
    )
    .unwrap_err();
    assert!(
        matches!(
            repeated,
            ParseError::InvalidOrnamentConvention { line: 2, .. }
        ),
        "{repeated:?}"
    );
}
