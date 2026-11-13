use kern_parser::{serialize::SerializePreset, KernDocument, ParseDialect, ParseOptions};
use proptest::prelude::*;

fn valid_document_for_seed(mut seed: u64) -> String {
    let lengths = ["4", "8", "16"];
    let pitches = ["c", "d", "e", "f", "g", "a", "b", "cc"];
    let count = 1 + (seed as usize % 24);
    let mut lines = vec!["**kern".to_owned(), "*M4/4".to_owned()];
    for index in 0..count {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let duration = lengths[(seed as usize >> 5) % lengths.len()];
        let pitch = pitches[(seed as usize >> 13) % pitches.len()];
        let accidental = match (seed >> 21) % 5 {
            0 => "#",
            1 => "-",
            _ => "",
        };
        lines.push(format!("{duration}{pitch}{accidental}"));
        if index % 4 == 3 {
            lines.push("=".to_owned());
        }
    }
    lines.push("*-".to_owned());
    lines.join("\n")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn transcoda_v2_is_a_textual_and_semantic_fixpoint(seed in any::<u64>()) {
        // The seed-to-document mapping is deterministic; proptest reports a replay seed
        // and shrinks the generated u64 on failure.
        let source = valid_document_for_seed(seed);
        let document = KernDocument::parse(&source).unwrap();
        let first = SerializePreset::TranscodaV2.serialize_document(&document);
        let reparsed = KernDocument::parse_with_options(
            &first,
            ParseOptions { dialect: ParseDialect::TranscodaV2 },
        ).unwrap();
        let second = SerializePreset::TranscodaV2.serialize_document(&reparsed);

        prop_assert_eq!(&second, &first, "seed={}\nsource={}", seed, source);
        prop_assert_eq!(
            semantic_events(&document).unwrap(),
            semantic_events(&reparsed).unwrap(),
            "seed={}\nsource={}\nserialized={}", seed, source, first,
        );
    }
}

fn semantic_events(
    document: &kern_parser::KernDocument,
) -> Result<Vec<kern_parser::analysis::SemanticEvent>, String> {
    let score =
        kern_parser::score::Score::resolve(document, kern_parser::document::Framing::Excerpt)
            .map_err(|error| error.to_string())?;
    kern_parser::analysis::semantic_events(&score).map_err(|error| error.to_string())
}
