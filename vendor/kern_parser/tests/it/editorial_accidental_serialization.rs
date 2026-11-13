use kern_parser::{serialize::SerializePreset, KernDocument};

#[test]
fn humdrum_kern_writes_every_user_signifier_and_transcoda_v2_only_editorial_ones() {
    let score = KernDocument::parse("**kern\n4f#ZiIl\n*-\n").unwrap();
    for (preset, expected) in [
        (SerializePreset::HumdrumKern, "4f#ZiIl"),
        (SerializePreset::TranscodaV2, "4f#Z"),
    ] {
        let text = preset.serialize_document(&score);
        assert!(text.lines().any(|line| line == expected), "{text}");
    }
}

#[test]
fn training_labels_strip_both_enclosures_without_changing_pitch_or_source() {
    let source = "**kern\n*clefG2\n*k[]\n4f#Z 4gnj\n8qb-j\n4cnXZ\n4d#yj\n*-\n";
    let score = KernDocument::parse(source).unwrap();
    let before = score.clone();
    let labels = SerializePreset::TranscodaV2Training.serialize_document(&score);
    assert_eq!(labels, "*clefG2\n*k[]\n4f# gn\n8qb-\n4cn\n4d#\n");
    let reparsed = KernDocument::parse(&labels).unwrap();
    assert_eq!(
        semantic_events(&score).unwrap(),
        semantic_events(&reparsed).unwrap()
    );
    assert_eq!(
        SerializePreset::TranscodaV2Training.serialize_document(&reparsed),
        labels
    );
    assert_eq!(score, before);
}

fn semantic_events(
    document: &kern_parser::KernDocument,
) -> Result<Vec<kern_parser::analysis::SemanticEvent>, String> {
    let score =
        kern_parser::score::Score::resolve(document, kern_parser::document::Framing::Excerpt)
            .map_err(|error| error.to_string())?;
    kern_parser::analysis::semantic_events(&score).map_err(|error| error.to_string())
}
