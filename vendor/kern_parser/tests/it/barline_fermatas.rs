use kern_parser::{
    normalization::{multi_measure_rests::CompactMultiMeasureRests, NormalizationPass},
    serialize::{BarNumberStyle, SerializeOptions, SerializePreset},
    Barline, BarlineFermata, KernDocument,
};

#[test]
fn parses_single_and_paired_boundary_annotations_without_losing_style() {
    for (token, expected, output) in [
        ("=4;", BarlineFermata::Single, "=;"),
        ("=29c;", BarlineFermata::Single, "=;"),
        ("==;", BarlineFermata::Single, "==;"),
        ("====;", BarlineFermata::Single, "==;"),
        ("=6;;", BarlineFermata::Pair, "=;;"),
        ("=3||;", BarlineFermata::Single, "=||;"),
        ("=3;||", BarlineFermata::Single, "=||;"),
        ("=3-;", BarlineFermata::Single, "=-;"),
        ("=:|!|:;;", BarlineFermata::Pair, "=:|!|:;;"),
    ] {
        let barline = Barline::try_from(token).unwrap();
        assert_eq!(barline.fermata_for_spine(0), Some(expected), "{token}");
        let document = KernDocument::parse(&format!("**kern\n1c\n{token}\n*-\n")).unwrap();
        let rendered = SerializePreset::HumdrumKern.serialize_document(&document);
        assert!(
            rendered.lines().any(|line| line == output),
            "{token}: {rendered}"
        );
    }
    for token in ["=;;;", "=;|;", "=;2", "=29;?", "=29-;|", "=1--;"] {
        assert!(Barline::try_from(token).is_err(), "must reject {token}");
    }
}

#[test]
fn uniform_fermatas_remain_compact_and_bar_numbers_precede_marks() {
    let document =
        KernDocument::parse("**kern\t**kern\n1c\t1e\n=2;;\t=2;;\n1d\t1f\n*-\t*-\n").unwrap();
    let compact = SerializePreset::TranscodaV2.serialize_document(&document);
    assert!(compact.contains("\n=;;\n"), "{compact}");
    let numbered = SerializePreset::HumdrumKern.serialize_document_with_options(
        &document,
        SerializeOptions {
            bar_number_style: BarNumberStyle::Sequential,
        },
    );
    assert!(numbered.contains("\n=2;;\t=2;;\n"), "{numbered}");
}

#[test]
fn leading_fermatas_survive_header_canonicalization_and_barline_removal() {
    let document = KernDocument::parse("**kern\n=1;\n*clefG2\n*M4/4\n1c\n==\n*-\n").unwrap();
    for preset in [SerializePreset::HumdrumKern, SerializePreset::TranscodaV2] {
        let output = preset.try_serialize_document(&document).unwrap();
        assert!(output.contains("\n=;\n1c\n"), "{preset:?}: {output}");
    }
}

#[test]
fn rest_compaction_retains_the_annotated_internal_boundary() {
    let mut document =
        KernDocument::parse("**kern\n*M4/4\n=1\n1r\n=2;\n1r\n=3\n1r\n==\n*-\n").unwrap();
    CompactMultiMeasureRests::new(Default::default()).normalize(&mut document);
    let output = SerializePreset::HumdrumKern.serialize_document(&document);
    assert!(output.contains("1r\n=;\n1r\n=\n1r\n=="), "{output}");
    let training = SerializePreset::TranscodaV2.serialize_document(&document);
    assert!(training.contains("\n=;\n"), "{training}");
}

#[test]
fn invisible_measure_cleanup_retains_adjacent_fermatas_and_their_time() {
    let document = KernDocument::parse("**kern\n*M4/4\n1c\n=;\n1ryy\n=\n1d\n==\n*-\n").unwrap();
    let output = SerializePreset::TranscodaV2
        .try_serialize_document(&document)
        .unwrap();
    assert!(output.contains("1c\n=;\n1ryy\n=\n1d"), "{output}");
}

#[test]
fn non_kern_spine_projection_preserves_the_authored_kern_fermatas() {
    let document = KernDocument::parse(
        "**kern\t**dynam\t**kern\n1c\t.\t1e\n=;\t=;;\t=\n1d\t.\t1f\n*-\t*-\t*-\n",
    )
    .unwrap();
    let output = SerializePreset::TranscodaV2
        .try_serialize_document(&document)
        .unwrap();
    assert!(output.contains("\n=;\t=\n"), "{output}");
    assert!(
        !output.contains(";;"),
        "must project away the dynam field: {output}"
    );
}

#[cfg(feature = "engraving")]
#[test]
fn serialized_barline_fermatas_render_as_one_or_two_glyphs() {
    let mut toolkit = verovioxide::Toolkit::new().unwrap();
    for (barline, expected) in [("=;", 1), ("=;;", 2), ("==;", 1)] {
        let document =
            KernDocument::parse(&format!("**kern\n*clefG2\n*M4/4\n1c\n{barline}\n*-\n")).unwrap();
        let humdrum = SerializePreset::HumdrumKern.serialize_document(&document);
        let svg = toolkit.render_data(&humdrum, None).unwrap();
        let tree = roxmltree::Document::parse(&svg).unwrap();
        let count = tree
            .descendants()
            .filter(|node| {
                node.attribute("class").is_some_and(|classes| {
                    classes.split_whitespace().any(|class| class == "fermata")
                })
            })
            .count();
        assert_eq!(count, expected, "{barline}");
    }
}

#[test]
fn topology_normalization_preserves_fermatas_on_both_split_layers() {
    let document =
        KernDocument::parse("**kern\n*M4/4\n*^\n1c\t1e\n=;\t=;\n*v\t*v\n1d\n==\n*-\n").unwrap();
    let rendered = SerializePreset::HumdrumKern.serialize_document(&document);
    assert!(
        rendered.contains("1c\t1e\n=;\t=;\n*v\t*v\n1d"),
        "{rendered}"
    );
    let training = SerializePreset::TranscodaV2
        .try_serialize_document(&document)
        .unwrap();
    // Compact syntax is uniform here, but the two active layers must remain
    // active at the annotated boundary when it is expanded for rendering.
    assert!(training.contains("1c\t1e\n=;\n*v\t*v\n1d"), "{training}");
    let reparsed = KernDocument::parse(&training).unwrap();
    let round_trip = SerializePreset::HumdrumKern.serialize_document(&reparsed);
    assert!(
        round_trip.contains("1c\t1e\n=;\t=;\n*v\t*v\n1d"),
        "{round_trip}"
    );
}

#[test]
fn rejects_misplaced_or_conflicting_barline_placement_markers() {
    for token in [
        "=<", "=>", "=<;", "=;<>", "=;><", "=;<<", "=;>>", "=;;<", "=;;>", "=;<;", "=;|<", "=;<2",
        "=;<x",
    ] {
        assert!(Barline::try_from(token).is_err(), "must reject {token}");
    }
}

#[cfg(feature = "engraving")]
#[test]
fn serialized_grand_staff_fermatas_render_on_the_outer_sides() {
    let document = KernDocument::parse("**kern\t**kern\n*staff2\t*staff1\n*clefF4\t*clefG2\n*M4/4\t*M4/4\n1C\t1c\n=;<\t=;>\n1D\t1d\n==\t==\n*-\t*-\n").unwrap();
    // KernOnly has no default RDF footer, so the barline marks must request it.
    let humdrum = SerializePreset::KernOnly.serialize_document(&document);
    let mut toolkit = verovioxide::Toolkit::new().unwrap();
    let svg = toolkit.render_data(&humdrum, None).unwrap();
    let tree = roxmltree::Document::parse(&svg).unwrap();
    assert_eq!(
        tree.descendants()
            .filter(|node| node
                .attribute("class")
                .is_some_and(|classes| classes.split_whitespace().any(|class| class == "fermata")))
            .count(),
        2
    );
    let mei = toolkit.get_mei().unwrap();
    let tree = roxmltree::Document::parse(&mei).unwrap();
    let mut positions = tree
        .descendants()
        .filter(|node| node.has_tag_name("fermata"))
        .map(|node| {
            (
                node.attribute("staff").unwrap(),
                node.attribute("place").unwrap(),
            )
        })
        .collect::<Vec<_>>();
    positions.sort();
    assert_eq!(positions, [("1", "above"), ("2", "below")]);
}
