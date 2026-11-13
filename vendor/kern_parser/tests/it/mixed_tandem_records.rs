use kern_parser::{
    serialize::{render, SerializePreset},
    KernDocument, KernLine, ParseDialect, ParseError, ParseOptions, TandemInterpretation,
};

const MIXED: &str = "**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*ped\t*Xtuplet\n4C\t4e\n*Xped\t*tuplet\n4D\t4f\n*-\t*-\n";
const SEPARATED: &str = "**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*ped\t*\n*\t*Xtuplet\n4C\t4e\n*Xped\t*\n*\t*tuplet\n4D\t4f\n*-\t*-\n";

#[test]
fn mixed_tandem_records_serialize_like_separated_records_and_reach_a_fixpoint() {
    let mixed = KernDocument::parse(MIXED).unwrap();
    let separated = KernDocument::parse(SEPARATED).unwrap();
    assert_eq!(mixed, separated);

    for (preset, dialect, expected) in [
        (
            SerializePreset::HumdrumKern,
            ParseDialect::HumdrumKern,
            format!("{SEPARATED}!!!RDF**kern: > = above\n!!!RDF**kern: < = below\n!!!RDF**kern: N = linked\n!!!RDF**kern: Z = editorial accidental, parentheses\n!!!RDF**kern: j = editorial accidental, brackets\n"),
        ),
        (
            SerializePreset::TranscodaV2,
            ParseDialect::TranscodaV2,
            "*clefF4\t*clefG2\n*k[]\t*k[]\n4C\t4e\n4D\t4f\n".to_owned(),
        ),
    ] {
        let output = preset.try_serialize_document(&mixed).unwrap();
        assert_eq!(output, expected);
        assert_eq!(output, preset.try_serialize_document(&separated).unwrap());
        let reparsed = KernDocument::parse_with_options(&output, ParseOptions { dialect }).unwrap();
        assert_eq!(preset.try_serialize_document(&reparsed).unwrap(), output);
    }
}

#[test]
fn mixed_header_tandems_finalize_missing_state_after_lowering() {
    let document =
        KernDocument::parse("**kern\t**kern\n*clefF4\t*k[f#]\n4C\t4f#\n*-\t*-\n").unwrap();
    assert_eq!(
        &document.lines()[1..3],
        &[
            KernLine::parse(2, "*clefF4\t*clefG2").unwrap(),
            KernLine::parse(2, "*k[]\t*k[f#]").unwrap(),
        ]
    );
    assert_eq!(document.source(1), Some(2));
    assert_eq!(document.source(2), Some(2));
}

#[test]
fn mixed_tandem_records_respect_split_spine_widths() {
    let document = KernDocument::parse("**kern\t**kern\n*^\t*\n*ped\t*\t*Xtuplet\n4c\t4e\t4g\n*Xped\t*\t*tuplet\n*v\t*v\t*\n*-\t*-\n").unwrap();
    let expanded = document
        .lines()
        .iter()
        .enumerate()
        .filter(|(index, _)| document.source(*index) == Some(3))
        .map(|(_, line)| line.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        expanded,
        vec![
            KernLine::parse(3, "*ped\t*\t*").unwrap(),
            KernLine::parse(3, "*\t*\t*Xtuplet").unwrap()
        ]
    );
    assert_eq!(document.layout.initial_spine_count, 2);

    assert_eq!(
        KernDocument::parse("**kern\t**kern\n*^\t*\n*ped\t*Xtuplet\n").unwrap_err(),
        ParseError::SpineCountMismatch {
            line: 3,
            expected: 3,
            found: 2
        }
    );
}

#[test]
fn mixed_tandem_records_preserve_opaque_spine_interpretations() {
    let document =
        KernDocument::parse("**kern\t**foo\n*ped\t*custom\n4c\tdata\n*Xped\t*\n*-\t*-\n").unwrap();
    let expanded = document
        .lines()
        .iter()
        .enumerate()
        .filter(|(index, _)| document.source(*index) == Some(2))
        .map(|(_, line)| match line {
            KernLine::TandemInterpretationLine(line) => line.tokens().to_vec(),
            _ => panic!("expected a tandem line"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        expanded,
        vec![
            vec![
                TandemInterpretation::Pedal(kern_parser::Pedal { active: true }),
                TandemInterpretation::Null
            ],
            vec![
                TandemInterpretation::Null,
                TandemInterpretation::Raw(kern_parser::RawToken("*custom".to_owned()))
            ],
        ]
    );
}

#[test]
fn mixed_tandem_records_preserve_render_provenance() {
    let document = KernDocument::parse(MIXED).unwrap();
    let output = render::to_render_output(&document, Default::default()).unwrap();
    let text = output.text();
    let expanded = text
        .lines()
        .zip(output.sources())
        .enumerate()
        .filter(|(_, (_, source))| **source == Some(4))
        .collect::<Vec<_>>();
    assert_eq!(
        expanded
            .iter()
            .map(|(_, (text, source))| (*text, **source))
            .collect::<Vec<_>>(),
        vec![("*ped\t*", Some(4)), ("*\t*Xtuplet", Some(4))]
    );
    for (row, (index, _)) in expanded.iter().enumerate() {
        let tokens = output
            .token_provenance()
            .iter()
            .filter(|token| token.rendered.line == index + 1)
            .collect::<Vec<_>>();
        assert_eq!(tokens.len(), 2);
        for (column, token) in tokens.iter().enumerate() {
            assert_eq!(
                token.source,
                if column == row {
                    Some(render::RenderAddress::field(4, column + 1))
                } else {
                    None
                }
            );
        }
    }
}
