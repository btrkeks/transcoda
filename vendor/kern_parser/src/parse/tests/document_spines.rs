use crate::types::{Chord, KernContent};
use crate::*;

fn data_lines(document: &KernDocument) -> Vec<&Vec<SpineContent>> {
    document
        .lines()
        .iter()
        .filter_map(|line| match line {
            KernLine::Data(data) => Some(data),
            _ => None,
        })
        .collect()
}

fn original_tandem_lines(document: &KernDocument) -> Vec<&TandemInterpretationLine> {
    document
        .lines()
        .iter()
        .enumerate()
        .filter(|(index, _)| document.source(*index).is_some())
        .filter_map(|(_, line)| match line {
            KernLine::TandemInterpretationLine(tandems) => Some(tandems),
            _ => None,
        })
        .collect()
}

fn finger_number(digit: u8) -> FingerNumber {
    FingerNumber::try_from(digit).unwrap()
}

fn single_fingering(digit: u8) -> Fingering {
    Fingering::single(finger_number(digit))
}

#[test]
fn kern_document_parses_mixed_spine_data_with_active_spine_types() {
    let document =
        KernDocument::parse("**kern\t**dynam\t**text\t**mxhm\n4c\tmf\tDo-\tG major\n").unwrap();

    let data = data_lines(&document).remove(0);

    assert_eq!(data[0].spine_type(), SpineType::Kern);
    assert_eq!(data[1].spine_type(), SpineType::Dynam);
    assert_eq!(data[2].spine_type(), SpineType::Text);
    assert_eq!(data[3].spine_type(), SpineType::MxmlChords);
    assert_eq!(
        data[0],
        SpineContent::Kern(KernContent::Chord(Chord::try_from("4c").unwrap()))
    );
    assert_eq!(
        data[1],
        SpineContent::Dynam(DynamContent::Marks(DynamMarks {
            level: Some(DynamicLevel {
                value: DynamicLevelValue::MezzoForte,
                subito: false,
                sforzando: false,
                accent: false,
                provenance: None,
            }),
            ..DynamMarks::default()
        }))
    );
    assert_eq!(
        data[2],
        SpineContent::Text(TextContent::Text(TextUnderlay {
            text: "Do-".to_owned(),
        }))
    );
    assert_eq!(
        data[3],
        SpineContent::MxmlHarmony(MxmlHarmonyContent::Label(MxmlHarmonyLabel {
            text: "G major".to_owned(),
        }))
    );
}

#[test]
fn kern_document_parses_fingering_spine_data() {
    let document = KernDocument::parse("**kern\t**fing\n4c\t1\n4d 4f 4a\t1 3 5\n.\t.\n").unwrap();

    let KernLine::SpineTypeLine(spine_types) = &document.lines()[0] else {
        panic!("expected spine type line");
    };
    assert_eq!(spine_types, &vec![SpineType::Kern, SpineType::Fing]);

    let data = data_lines(&document);
    let single = data[0];
    assert_eq!(
        single[1],
        SpineContent::Fing(FingContent::Fingerings(vec![Some(single_fingering(1))]))
    );

    let chord = data[1];
    assert_eq!(
        chord[1],
        SpineContent::Fing(FingContent::Fingerings(vec![
            Some(single_fingering(1)),
            Some(single_fingering(3)),
            Some(single_fingering(5)),
        ]))
    );

    let null = data[2];
    assert_eq!(null[1], SpineContent::Fing(FingContent::Null));
}

#[test]
fn kern_document_parses_finger_substitutions_as_single_entries() {
    let document = KernDocument::parse("**kern\t**fing\n2c\t2-1\n4d 4f\t2-1 4\n*-\t*-\n").unwrap();

    let data = data_lines(&document);
    assert_eq!(
        data[0][1],
        SpineContent::Fing(FingContent::Fingerings(vec![Some(
            Fingering::substitution(finger_number(2), finger_number(1),)
        )]))
    );
    assert_eq!(
        data[1][1],
        SpineContent::Fing(FingContent::Fingerings(vec![
            Some(Fingering::substitution(finger_number(2), finger_number(1))),
            Some(single_fingering(4)),
        ]))
    );

    assert_eq!(
        SerializePreset::HumdrumKern.serialize_document(&document),
        "**kern\t**fing\n*clefG2\t*\n*k[]\t*\n2c\t2-1\n4d 4f\t2-1 4\n*-\t*-\n!!!RDF**kern: > = above\n!!!RDF**kern: < = below\n!!!RDF**kern: N = linked\n!!!RDF**kern: Z = editorial accidental, parentheses\n!!!RDF**kern: j = editorial accidental, brackets\n"
    );
}

#[test]
fn kern_document_parses_unfingered_chord_slots_and_stacked_digits() {
    let document = KernDocument::parse("**kern\t**fing\n4d 4f 4a\t. 4/3/2 .\n*-\t*-\n").unwrap();

    assert_eq!(
        data_lines(&document)[0][1],
        SpineContent::Fing(FingContent::Fingerings(vec![
            None,
            Some(Fingering::Stacked {
                first: finger_number(4),
                second: finger_number(3),
                rest: vec![finger_number(2)],
            }),
            None,
        ]))
    );
}

#[test]
fn kern_document_rejects_invalid_fingering_tokens() {
    for token in [
        "L1", "R2", "x", "1,x", "1  2", "2-", "-1", "2--1", "2-1-2", "6-1", "2-6", "2–1", "2-1  4",
        "2 -1", ". .", ". . .", "..", "4/", "/4", "4//3", "4/6", "4/3-2", "4-3/2", "4/.",
    ] {
        assert!(matches!(
            KernDocument::parse(format!("**kern\t**fing\n4c\t{token}\n").as_str()).unwrap_err(),
            ParseError::InvalidSpineToken {
                line: 2,
                column: 2,
                source,
                ..
            } if matches!(*source, ParseError::InvalidFingering { .. })
        ));
    }
}

#[test]
fn kern_document_preserves_raw_fingering_tandems() {
    let document =
        KernDocument::parse("**kern\t**fing\t**fing\n*\t*left\t*right\n4c\t1\t2\n").unwrap();

    let tandems = original_tandem_lines(&document).remove(0);
    assert_eq!(
        tandems.tokens(),
        &[
            TandemInterpretation::Null,
            TandemInterpretation::Raw(RawToken("*left".to_owned())),
            TandemInterpretation::Raw(RawToken("*right".to_owned())),
        ]
    );
}

#[test]
fn kern_document_rejects_dynamic_token_in_kern_spine() {
    assert!(matches!(
        KernDocument::parse("**kern\nmf\n").unwrap_err(),
        ParseError::InvalidSpineToken {
            line: 2,
            column: 1,
            source,
            ..
        } if matches!(*source, ParseError::InvalidNote { .. })
    ));
}

#[test]
fn kern_document_parses_null_as_active_spine_type() {
    let document = KernDocument::parse("**kern\t**dynam\t**text\t**mxhm\n.\t.\t.\t.\n").unwrap();

    let data = data_lines(&document).remove(0);

    assert_eq!(
        data,
        &vec![
            SpineContent::Kern(KernContent::Null),
            SpineContent::Dynam(DynamContent::Null),
            SpineContent::Text(TextContent::Null),
            SpineContent::MxmlHarmony(MxmlHarmonyContent::Null),
        ]
    );
}

#[test]
fn kern_document_parses_empty_kern_data_fields_as_nulls() {
    let document = KernDocument::parse("**kern\t**kern\n\t4e\n4c\t\n").unwrap();

    let data = data_lines(&document);

    assert_eq!(data.len(), 2);
    assert_eq!(
        data[0],
        &vec![
            SpineContent::Kern(KernContent::Null),
            SpineContent::Kern(KernContent::Chord(Chord::try_from("4e").unwrap())),
        ]
    );
    assert_eq!(
        data[1],
        &vec![
            SpineContent::Kern(KernContent::Chord(Chord::try_from("4c").unwrap())),
            SpineContent::Kern(KernContent::Null),
        ]
    );
}

#[test]
fn kern_document_parses_empty_kern_data_fields_without_spine_type_line() {
    let document = KernDocument::parse("*clefG2\t*clefG2\n*k[]\t*k[]\n\t4e\n4c\t\n").unwrap();

    let data = data_lines(&document);

    assert_eq!(document.layout.initial_spine_count, 2);
    assert_eq!(data.len(), 2);
    assert_eq!(data[0][0], SpineContent::Kern(KernContent::Null));
    assert_eq!(data[1][1], SpineContent::Kern(KernContent::Null));
}

#[test]
fn kern_document_parses_empty_kern_data_field_with_text_exclamation() {
    let document = KernDocument::parse("**kern\t**text\n\tfoo!\n").unwrap();

    let data = data_lines(&document);

    assert_eq!(data.len(), 1);
    assert_eq!(data[0][0], SpineContent::Kern(KernContent::Null));
    assert_eq!(
        data[0][1],
        SpineContent::Text(TextContent::Text(TextUnderlay {
            text: "foo!".to_owned()
        }))
    );
}

#[test]
fn kern_line_parses_empty_fields_as_kern_data_nulls_by_default() {
    let line = KernLine::parse(3, "\t4e").unwrap();

    assert_eq!(
        line,
        KernLine::Data(vec![
            SpineContent::Kern(KernContent::Null),
            SpineContent::Kern(KernContent::Chord(Chord::try_from("4e").unwrap())),
        ])
    );
}

#[test]
fn kern_document_rejects_empty_fields_in_non_kern_spines() {
    assert_eq!(
        KernDocument::parse("**kern\t**dynam\n4c\t\n").unwrap_err(),
        ParseError::EmptySpineToken { line: 2, column: 2 }
    );
}

#[test]
fn kern_document_parses_opaque_non_kern_spines() {
    let document =
        KernDocument::parse("**kern\t**fb\t**fba\t**recip\t**foo\n4c\t6 5\tx 4 2\t4\tpayload\n")
            .unwrap();

    let KernLine::SpineTypeLine(spine_types) = &document.lines()[0] else {
        panic!("expected spine type line");
    };
    assert_eq!(
        spine_types,
        &vec![
            SpineType::Kern,
            SpineType::Other("**fb".to_owned()),
            SpineType::Other("**fba".to_owned()),
            SpineType::Other("**recip".to_owned()),
            SpineType::Other("**foo".to_owned()),
        ]
    );

    let data = data_lines(&document).remove(0);
    assert_eq!(
        data[1],
        SpineContent::Opaque {
            spine_type: SpineType::Other("**fb".to_owned()),
            raw: RawToken("6 5".to_owned()),
        }
    );
    assert_eq!(
        data[2],
        SpineContent::Opaque {
            spine_type: SpineType::Other("**fba".to_owned()),
            raw: RawToken("x 4 2".to_owned()),
        }
    );
    assert_eq!(
        data[3],
        SpineContent::Opaque {
            spine_type: SpineType::Other("**recip".to_owned()),
            raw: RawToken("4".to_owned()),
        }
    );
    assert_eq!(
        data[4],
        SpineContent::Opaque {
            spine_type: SpineType::Other("**foo".to_owned()),
            raw: RawToken("payload".to_owned()),
        }
    );
}

#[test]
fn kern_document_parses_san_with_recip_text_and_mxhm_spines() {
    let input = [
        "!!!OTL: San",
        "**recip\t**kern\t**text\t**text\t**mxhm",
        "*\t*I\"P1\t*\t*\t*",
        "*\t*clefF4\t*\t*\t*",
        "*\t*k[b-]\t*\t*\t*",
        "*\t*M4/4\t*\t*\t*",
        "=1\t=1\t=1\t=1\t=1",
        "55\t!LO:TX:a:i:t=Bass Intro\t!\t!\t!",
        "4\t4DD\t.\t.\t.",
        "8\t8AAL\t.\t.\t.",
        "8\t8AAJ\t.\t.\t.",
        "4\t4DD\t.\t.\t.",
        "4\t4AA\t.\t.\t.",
        "=\t=\t=\t=\t=",
        "4\t4DD\t.\t.\t.",
        "8\t8AAL\t.\t.\t.",
        "8\t8AAJ\t.\t.\t.",
        "4\t4DD\t.\t.\t.",
        "4\t4AA\t.\t.\t.",
        "=\t=\t=\t=\t=",
        "4\t4DD\t.\t.\t.",
        "8\t8AAL\t.\t.\t.",
        "8\t8AAJ\t.\t.\t.",
        "4\t4DD\t.\t.\t.",
        "4\t4AA\t.\t.\t.",
        "=\t=\t=\t=\t=",
        "*-\t*-\t*-\t*-\t*-",
    ]
    .join("\n");

    let document = KernDocument::parse(&input).unwrap();

    assert_eq!(document.metadata.title, Some("San".to_owned()));
    assert_eq!(document.layout.initial_spine_count, 5);
    assert_eq!(document.staff_count(), 1);
}

#[test]
fn kern_document_preserves_unsupported_tandems_in_opaque_spines() {
    let document = KernDocument::parse("**kern\t**fb\n*\t*fb:layout\n4c\t6\n").unwrap();

    let line = original_tandem_lines(&document).remove(0);

    assert_eq!(
        line.tokens(),
        &[
            TandemInterpretation::Null,
            TandemInterpretation::Raw(RawToken("*fb:layout".to_owned())),
        ]
    );
}

#[test]
fn kern_document_tracks_opaque_spine_types_through_topology() {
    let document = KernDocument::parse("**fb\n*^\n6\t5\n*v\t*v\n7\n").unwrap();

    let KernLine::Data(split_data) = &document.lines()[2] else {
        panic!("expected split data line");
    };
    assert_eq!(
        split_data
            .iter()
            .map(SpineContent::spine_type)
            .collect::<Vec<_>>(),
        vec![
            SpineType::Other("**fb".to_owned()),
            SpineType::Other("**fb".to_owned())
        ]
    );

    let KernLine::Data(merged_data) = &document.lines()[4] else {
        panic!("expected merged data line");
    };
    assert_eq!(
        merged_data[0],
        SpineContent::Opaque {
            spine_type: SpineType::Other("**fb".to_owned()),
            raw: RawToken("7".to_owned()),
        }
    );
}

#[test]
fn kern_document_rejects_unsupported_tandem_in_kern_spine() {
    let error = KernDocument::parse("**kern\n*fb:layout\n").unwrap_err();

    assert!(matches!(
        error,
        ParseError::InvalidSpineToken {
            line: 2,
            column: 1,
            source,
            ..
        } if matches!(*source, ParseError::UnsupportedTandemInterpretation { .. })
    ));
}

#[test]
fn kern_document_tracks_spine_types_through_split_merge_and_termination() {
    let input = [
        "**kern\t**dynam\t**text",
        "*^\t*\t*",
        "4c\t4e\tmf\tword",
        "*v\t*v\t*\t*",
        "*\t*\t*-",
        "4g\t>",
    ]
    .join("\n");
    let document = KernDocument::parse(&input).unwrap();

    let data = data_lines(&document);
    let first_data = data[0];
    assert_eq!(
        first_data
            .iter()
            .map(SpineContent::spine_type)
            .collect::<Vec<_>>(),
        vec![
            SpineType::Kern,
            SpineType::Kern,
            SpineType::Dynam,
            SpineType::Text
        ]
    );

    let second_data = data[1];
    assert_eq!(
        second_data
            .iter()
            .map(SpineContent::spine_type)
            .collect::<Vec<_>>(),
        vec![SpineType::Kern, SpineType::Dynam]
    );
}

#[test]
fn kern_document_rejects_merging_different_spine_types() {
    assert_eq!(
        KernDocument::parse("**kern\t**dynam\n*v\t*v\n").unwrap_err(),
        ParseError::InvalidSpineOperation {
            message: "cannot merge spines with different spine types".to_owned(),
        }
    );
}

#[test]
fn kern_document_rejects_merging_kern_and_opaque_spines() {
    assert_eq!(
        KernDocument::parse("**kern\t**fb\n*v\t*v\n").unwrap_err(),
        ParseError::InvalidSpineOperation {
            message: "cannot merge spines with different spine types".to_owned(),
        }
    );
}

#[test]
fn kern_document_accepts_empty_input() {
    let document = KernDocument::parse("").unwrap();

    assert_eq!(document.layout.initial_spine_count, 0);
    assert_eq!(document.lines(), Vec::new());
}

#[test]
fn kern_document_accepts_metadata_only_input() {
    let document =
        KernDocument::parse("!!!COM: Example\n!! only comments\n! local only\n").unwrap();

    assert_eq!(document.metadata.composer.as_deref(), Some("Example"));
    assert_eq!(document.layout.initial_spine_count, 0);
    assert_eq!(document.lines(), Vec::new());
}
