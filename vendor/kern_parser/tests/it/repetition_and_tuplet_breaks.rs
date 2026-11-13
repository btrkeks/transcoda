use kern_parser::{
    serialize::{omr, SerializeOptions},
    KernDocument, KernLine, ParseError, Repetition, SerializePreset, TandemInterpretation,
    TandemInterpretationLineKind,
};

const KERN: &str = "**kern\n*clefG2\n*k[]\n*M4/4\n4c\n*rep\n12d\n12e\n12f\n*Xrep\n*tupbreak\n12g\n12a\n12b\n4cc\n*-\n";

#[test]
fn parses_exact_markers_as_typed_interpretations() {
    for (raw, expected, kind) in [
        (
            "*rep",
            TandemInterpretation::Repetition(Repetition { active: true }),
            TandemInterpretationLineKind::Repetition,
        ),
        (
            "*Xrep",
            TandemInterpretation::Repetition(Repetition { active: false }),
            TandemInterpretationLineKind::Repetition,
        ),
        (
            "*tupbreak",
            TandemInterpretation::TupletBreak,
            TandemInterpretationLineKind::TupletBreak,
        ),
    ] {
        assert_eq!(TandemInterpretation::try_from(raw).unwrap(), expected);
        let KernLine::TandemInterpretationLine(line) = KernLine::parse(1, raw).unwrap() else {
            panic!("expected a tandem line");
        };
        assert_eq!(line.kind(), kind);
        assert_eq!(line.tokens(), &[expected]);
    }

    for raw in [
        "*rep2",
        "*Xrep2",
        "*Rep",
        "*xrep",
        "*Xtupbreak",
        "*tupbreak2",
    ] {
        assert_eq!(
            TandemInterpretation::try_from(raw).unwrap_err(),
            ParseError::UnsupportedTandemInterpretation { raw: raw.into() }
        );
        assert!(KernDocument::parse(&format!("**kern\n{raw}\n4c\n*-\n")).is_err());
    }
}

#[test]
fn all_v2_syntax_variants_omit_markers() {
    let document = KernDocument::parse(KERN).unwrap();
    let expected = "*clefG2\n*k[]\n*M4/4\n4c\n12d\n12e\n12f\n12g\n12a\n12b\n4cc\n";
    for serialize in [
        omr::try_to_transcoda_v2_string_with_options,
        omr::try_to_transcoda_v2_empty_nulls_string_with_options,
        omr::try_to_transcoda_v2_compact_barlines_string_with_options,
        omr::try_to_transcoda_v2_compact_barlines_empty_nulls_string_with_options,
    ] {
        assert_eq!(
            serialize(&document, SerializeOptions::default()).unwrap(),
            expected
        );
    }
}

#[test]
fn omitted_markers_do_not_block_spine_canonicalization() {
    let input = "**kern\t**kern\n*clefG2\t*clefG2\n=\n*rep\t*\n*\t*^\n4c\t4e\t4g\n*\t*v\t*v\n*Xrep\t*\n*tupbreak\t*\n=\n2d\t4f\n*\t*^\n.\t4a\t4b\n*\t*v\t*v\n=\n*-\t*-\n";
    let document = KernDocument::parse(input).unwrap();
    let expected = "*clefG2\t*clefG2\n*k[]\t*k[]\n*\t*^\n4c\t4e\t4g\n=\n2d\t4f\t4ryy\n\t4a\t4b\n*\t*v\t*v\n=\n";
    let first = SerializePreset::TranscodaV2
        .try_serialize_document(&document)
        .unwrap();
    assert_eq!(first, expected);
    let reparsed = KernDocument::parse_with_options(
        &first,
        kern_parser::ParseOptions {
            dialect: kern_parser::ParseDialect::TranscodaV2,
        },
    )
    .unwrap();
    assert_eq!(
        SerializePreset::TranscodaV2
            .try_serialize_document(&reparsed)
            .unwrap(),
        expected
    );
}
