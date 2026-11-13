use kern_parser::{
    document::Framing,
    score::{Outcome, Score, StateKey},
    serialize::render,
    KernDocument, KernLine, ParseError, SerializePreset, TandemInterpretation,
    TandemInterpretationLineKind, TupletBracket,
};

const KERN: &str =
    "**kern\n*clefG2\n*k[]\n*M2/4\n*Xbrackettup\n12c\n12d\n12e\n*brackettup\n12f\n12g\n12a\n*-\n";

#[test]
fn parses_exact_markers_as_typed_interpretations() {
    for (raw, show_bracket) in [("*brackettup", true), ("*Xbrackettup", false)] {
        let expected = TandemInterpretation::TupletBracket(TupletBracket { show_bracket });
        assert_eq!(TandemInterpretation::try_from(raw).unwrap(), expected);
        let KernLine::TandemInterpretationLine(line) = KernLine::parse(1, raw).unwrap() else {
            panic!("expected a tandem line");
        };
        assert_eq!(line.kind(), TandemInterpretationLineKind::TupletBracket);
        assert_eq!(line.tokens(), &[expected]);
    }

    for raw in [
        "*Brackettup",
        "*xbrackettup",
        "*XBrackettup",
        "*brackettup2",
        "*bracket",
        "*Xbeamtup",
    ] {
        assert_eq!(
            TandemInterpretation::try_from(raw).unwrap_err(),
            ParseError::UnsupportedTandemInterpretation { raw: raw.into() }
        );
        assert!(KernDocument::parse(&format!("**kern\n{raw}\n4c\n*-\n")).is_err());
    }
}

#[test]
fn source_and_render_output_keep_markers() {
    let document = KernDocument::parse(KERN).unwrap();
    assert_eq!(
        SerializePreset::KernOnly.serialize_document(&document),
        KERN
    );
    for output in [
        SerializePreset::HumdrumKern.serialize_document(&document),
        render::to_string(&document).unwrap(),
    ] {
        assert!(output.starts_with(KERN), "{output}");
        let reparsed = KernDocument::parse(&output).unwrap();
        assert_eq!(render::to_string(&reparsed).unwrap(), output);
    }
}

#[test]
fn bracket_visibility_is_track_state_shown_by_default() {
    let source = "**kern\n12c\n*^\n*Xbrackettup\t*\n12c\t12e\n*\t*Xbrackettup\n*v\t*v\n*brackettup\n12c\n*-\n";
    let document = KernDocument::parse(source).unwrap();
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let data_lines = (0..document.lines().len())
        .filter(|&index| matches!(document.lines()[index], KernLine::Data(_)))
        .collect::<Vec<_>>();

    assert!(score.state_at(data_lines[0], 0).track.tuplet_brackets);
    for column in 0..2 {
        assert!(!score.state_at(data_lines[1], column).track.tuplet_brackets);
    }
    assert!(score.state_at(data_lines[2], 0).track.tuplet_brackets);

    let outcomes = score
        .events()
        .iter()
        .filter(|event| event.key == StateKey::TupletBrackets)
        .map(|event| event.outcome)
        .collect::<Vec<_>>();
    assert_eq!(
        outcomes,
        vec![Outcome::Changed, Outcome::Redundant, Outcome::Changed]
    );
}

#[cfg(feature = "engraving")]
#[test]
fn verovio_hides_brackets_while_xbrackettup_is_in_force() {
    let document = KernDocument::parse(KERN).unwrap();
    let rendered = render::to_string(&document).unwrap();
    let mut toolkit = if let Some(path) = std::env::var_os("KERN_PARSER_VEROVIO_RESOURCE_DIR") {
        verovioxide::Toolkit::with_resource_path(std::path::Path::new(&path))
    } else {
        verovioxide::Toolkit::new()
    }
    .expect("Verovio toolkit should start");
    toolkit.render_data(&rendered, None).unwrap();
    let mei = toolkit.get_mei().unwrap();
    let xml = roxmltree::Document::parse(&mei).unwrap();
    let brackets = xml
        .descendants()
        .filter(|node| node.has_tag_name("tuplet"))
        .map(|node| node.attribute("bracket.visible"))
        .collect::<Vec<_>>();
    assert_eq!(brackets, vec![Some("false"), None]);
}
