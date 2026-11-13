use kern_parser::{
    augment_for_render_with_report,
    layout_targets::{clear_system_breaks, project_meter_visibility, project_staff_visibility},
    serialize::{render::to_render_output, SerializePreset},
    transforms::{
        pad_final_measure_with_rests,
        split::{split_trailing_bars_with_prefix_policy, SlicePrefixPolicy},
    },
    KernDocument, RenderAugmentationConfig, RenderAugmentationSeed,
};
fn render(doc: &KernDocument) -> (String, Vec<Option<usize>>) {
    let out = to_render_output(doc, Default::default()).unwrap();
    let svg = verovioxide::Toolkit::new()
        .unwrap()
        .render_data(&out.text(), None)
        .unwrap();
    (svg, out.sources().to_vec())
}
fn count(svg: &str, class: &str) -> usize {
    let xml = roxmltree::Document::parse(svg).unwrap();
    xml.descendants()
        .filter(|n| {
            n.attribute("class")
                .is_some_and(|c| c.split_whitespace().any(|c| c == class))
                && !n
                    .ancestors()
                    .any(|a| a.attribute("visibility") == Some("hidden"))
                && n.descendants()
                    .any(|c| c.is_element() && c.tag_name().name() == "use")
        })
        .count()
}
fn target(doc: &KernDocument) -> String {
    SerializePreset::TranscodaV2Training
        .try_serialize_document(doc)
        .unwrap()
}
fn project(doc: &mut KernDocument, svg: &str, sources: &[Option<usize>]) {
    project_staff_visibility(doc, svg, sources).unwrap();
    project_meter_visibility(doc, svg, sources).unwrap();
}
#[test]
fn labelled_injection_replays_recipe_and_matches_svg_marks() {
    let doc = KernDocument::parse(
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c 4e 4g\n4d 4f 4a\n4e 4g 4b\n4f 4a 4cc\n==\n*-\n",
    )
    .unwrap();
    let config = RenderAugmentationConfig {
        fermata_page_probability: 1.0,
        arpeggio_page_probability: 1.0,
        ..Default::default()
    };
    let (augmented, report) =
        augment_for_render_with_report(&doc, &config, RenderAugmentationSeed(42)).unwrap();
    let receipt = serde_json::to_string(&report).unwrap();
    let report: kern_parser::RenderAugmentationReport = serde_json::from_str(&receipt).unwrap();
    let mut replay = augment_for_render_with_report(
        &doc,
        report.config.as_ref().unwrap(),
        RenderAugmentationSeed(report.seed),
    )
    .unwrap()
    .0;
    assert_eq!(augmented, replay);
    let (svg, sources) = render(&augmented);
    project(&mut replay, &svg, &sources);
    let label = target(&replay);
    assert_eq!(count(&svg, "fermata"), 1);
    assert_eq!(count(&svg, "arpeg"), 1);
    assert_eq!(
        label.matches(';').count(),
        count(&svg, "fermata"),
        "{label}"
    );
    assert_eq!(label.matches(':').count(), count(&svg, "arpeg"), "{label}");
}
#[test]
fn zero_page_rates_disable_injection_but_keep_authored_marks() {
    let doc = KernDocument::parse("**kern\n*M4/4\n4c; 4e 4g:\n4d 4f 4a\n4e 4g 4b\n4f 4a 4cc\n*-\n")
        .unwrap();
    let config = RenderAugmentationConfig {
        fermata_page_probability: 0.0,
        arpeggio_page_probability: 0.0,
        ..Default::default()
    };
    let augmented = augment_for_render_with_report(&doc, &config, RenderAugmentationSeed(42))
        .unwrap()
        .0;
    let svg = render(&augmented).0;
    assert_eq!(count(&svg, "fermata"), 1);
    assert_eq!(count(&svg, "arpeg"), 1);
    assert_eq!(target(&augmented).matches(';').count(), 1);
    assert_eq!(target(&augmented).matches(':').count(), 1);
}

#[test]
fn fermata_injection_does_not_target_rests_that_verovio_can_compact() {
    let doc =
        KernDocument::parse("**kern\n*clefG2\n*M4/4\n=1\n1r\n=2\n1r\n=3\n1r\n==\n*-\n").unwrap();
    let config = RenderAugmentationConfig {
        fermata_page_probability: 1.0,
        ..Default::default()
    };
    let (augmented, _) =
        augment_for_render_with_report(&doc, &config, RenderAugmentationSeed(42)).unwrap();
    let (svg, _) = render(&augmented);
    assert_eq!(count(&svg, "fermata"), 0);
    assert_eq!(count(&svg, "multiRest"), 1);
    assert!(!target(&augmented).contains(';'));
}

#[test]
fn continuation_cut_keeps_only_start_repeat_face() {
    for (boundary, repeat) in [
        ("=:|:", true),
        ("=!|:", true),
        ("=:|!", false),
        ("=||", false),
        ("==", false),
    ] {
        let doc = KernDocument::parse(&format!(
            "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n1c\n{boundary}\n1d\n==\n*-\n"
        ))
        .unwrap();
        for policy in [
            SlicePrefixPolicy::StandaloneExcerpt,
            SlicePrefixPolicy::PageRendering {
                is_first_page: false,
            },
        ] {
            let cut = split_trailing_bars_with_prefix_policy(&doc, 1, policy)
                .unwrap()
                .removed;
            let label = target(&cut);
            assert_eq!(label.contains("=!|:"), repeat, "{boundary}: {label}");
            assert!(!label.contains("=:|"), "{boundary}: {label}");
            let before_note = label.split("1d").next().unwrap();
            assert!(
                !before_note.contains("=||") && !before_note.contains("=="),
                "{label}"
            );
            let svg = render(&cut).0;
            // A repeat is rendered as an MEI barLine group at the opening.
            assert!(svg.contains("class=\"note\""));
        }
    }
}
#[test]
fn rational_padding_is_invisible_in_svg_and_label() {
    for note in ["4c", "5%3c"] {
        let doc = KernDocument::parse(&format!("**kern\n*M4/4\n=1\n{note}\n==\n*-\n")).unwrap();
        let padded = pad_final_measure_with_rests(&doc);
        let svg = render(&padded).0;
        assert_eq!(count(&svg, "rest"), 0, "{svg}");
        let label = target(&padded);
        assert!(label.contains("ryy"), "{label}");
        assert!(
            !label.lines().any(|l| l.contains('r') && !l.contains("ryy")),
            "{label}"
        );
    }
}
#[test]
fn source_breaks_are_cleared_before_svg_annotation() {
    let mut doc = KernDocument::parse("**kern\n*M4/4\n1c\n!!LO:LB:g=z\n=2\n1d\n*-\n").unwrap();
    assert!(target(&doc).contains("!!LO:LB"));
    clear_system_breaks(&mut doc);
    assert!(!target(&doc).contains("!!LO:LB"));
}
#[test]
fn unprinted_restatements_collapse_but_printed_restatements_survive() {
    for (header, repeated) in [
        ("*clefG2\n*k[]\n*M4/4", "*clefG2"),
        ("*clefG2\n*k[]\n*M4/4", "*k[]"),
        ("*clefG2\n*k[f#]\n*M4/4", "*k[f#]"),
        ("*clefG2\n*k[]\n*M4/4\n*met(c)", "*met(c)"),
        ("*clefG2\n*k[]\n*M4/4", "*M4/4"),
    ] {
        let mut doc = KernDocument::parse(&format!(
            "**kern\n{header}\n=1\n1c\n=2\n{repeated}\n1d\n==\n*-\n"
        ))
        .unwrap();
        let (svg, sources) = render(&doc);
        let class = if repeated.starts_with("*clef") {
            "clef"
        } else if repeated.starts_with("*k") {
            "keySig"
        } else {
            "meterSig"
        };
        let before = target(&doc).matches(repeated).count();
        project(&mut doc, &svg, &sources);
        let label = target(&doc);
        let expected = if count(&svg, class) > 1 { before } else { 1 };
        assert_eq!(
            label.matches(repeated).count(),
            expected,
            "{repeated}, SVG count {}: {label}",
            count(&svg, class)
        );
    }
}
#[test]
fn authored_changes_survive_visibility_projection() {
    let mut doc = KernDocument::parse(
        "**kern\n*clefG2\n*k[]\n*M4/4\n1c\n=2\n*clefF4\n*k[b-]\n*M3/4\n2.C\n==\n*-\n",
    )
    .unwrap();
    let (svg, sources) = render(&doc);
    project(&mut doc, &svg, &sources);
    let label = target(&doc);
    for declaration in ["*clefF4", "*k[b-]", "*M3/4"] {
        assert!(label.contains(declaration), "{label}");
    }
}

#[test]
fn opening_header_cannot_lend_ink_to_a_terminal_clef() {
    let mut doc =
        KernDocument::parse("**kern\n*clefG2\n*k[]\n*M4/4\n1c\n=\n*clefF4\n*-\n").unwrap();
    let (svg, sources) = render(&doc);
    assert_eq!(count(&svg, "clef"), 1);
    project(&mut doc, &svg, &sources);
    assert!(!target(&doc).contains("*clefF4"));
}

#[test]
fn printed_first_measure_restatements_survive_header_reservation() {
    let mut doc =
        KernDocument::parse("**kern\n*clefG2\n*k[f#]\n*M4/4\n2c\n*clefG2\n*k[f#]\n2d\n==\n*-\n")
            .unwrap();
    let (svg, sources) = render(&doc);
    assert_eq!(count(&svg, "clef"), 2);
    assert_eq!(count(&svg, "keySig"), 2);
    project(&mut doc, &svg, &sources);
    let label = target(&doc);
    assert_eq!(label.matches("*clefG2").count(), count(&svg, "clef"));
    assert_eq!(label.matches("*k[f#]").count(), count(&svg, "keySig"));
}

#[test]
fn arpeggio_injection_excludes_tremolo_spans_but_resumes_after_them() {
    let config = RenderAugmentationConfig {
        fermata_page_probability: 0.0,
        arpeggio_page_probability: 1.0,
        ..Default::default()
    };
    for suffix in ["", "*Xtremolo\n=2\n1d 1f 1a\n"] {
        let doc = KernDocument::parse(&format!("**kern\n*clefG2\n*M4/4\n*tremolo\n=1\n8cL 8e 8g\n8cJ 8e 8g\n2.c 2.e 2.g\n{suffix}==\n*-\n")).unwrap();
        let (augmented, report) =
            augment_for_render_with_report(&doc, &config, RenderAugmentationSeed(42)).unwrap();
        let (svg, _) = render(&augmented);
        let expected = usize::from(!suffix.is_empty());
        assert_eq!(report.injected_arpeggio_count, expected);
        assert_eq!(count(&svg, "arpeg"), expected);
        assert_eq!(target(&augmented).matches(':').count(), expected);
    }
}

#[test]
fn source_initial_end_start_repeat_keeps_only_its_right_face() {
    let doc = KernDocument::parse("**kern\n*clefG2\n*M3/4\n=:|!|:\n2.c\n==\n*-\n").unwrap();
    let label = target(&doc);
    assert!(label.contains("=!|:"), "{label}");
    assert!(!label.contains("=:|"), "{label}");
}

#[test]
fn changed_meter_at_opening_repeat_seam_is_drawn_and_labelled() {
    let parts = [
        "**kern\n*clefG2\n*M4/4\n=1\n1c\n==\n*-\n",
        "**kern\n*clefG2\n*M3/4\n=!|:\n2.d\n==\n*-\n",
    ]
    .map(|text| KernDocument::parse(text).unwrap());
    let merged = kern_parser::concatenation::concatenate_documents(&parts).unwrap();
    let mut doc = KernDocument::parse(
        &SerializePreset::HumdrumKern
            .try_serialize_document(&merged)
            .unwrap(),
    )
    .unwrap();
    let (svg, sources) = render(&doc);
    assert_eq!(count(&svg, "meterSig"), 2);
    project(&mut doc, &svg, &sources);
    assert!(target(&doc).contains("*M3/4"));
}
