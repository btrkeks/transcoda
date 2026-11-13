use kern_parser::{
    document::Framing,
    layout_targets::project_meter_visibility,
    score::Score,
    serialize::{render::to_render_output, SerializePreset},
    transforms::split::{
        hide_opening_meter, split_trailing_bars_with_prefix_policy, SlicePrefixPolicy,
    },
    KernDocument,
};

fn render(document: &KernDocument) -> (String, Vec<Option<usize>>) {
    let output = to_render_output(document, Default::default()).unwrap();
    let mut toolkit = verovioxide::Toolkit::new().unwrap();
    let svg = toolkit.render_data(&output.text(), None).unwrap();
    (svg, output.sources().to_vec())
}
fn glyphs(svg: &str) -> usize {
    let xml = roxmltree::Document::parse(svg).unwrap();
    xml.descendants()
        .filter(|node| {
            node.attribute("class").is_some_and(|class| {
                class
                    .split_whitespace()
                    .any(|class| class == "meterSig" || class == "mensur")
            }) && !node
                .ancestors()
                .any(|ancestor| ancestor.attribute("visibility") == Some("hidden"))
                && node
                    .descendants()
                    .any(|child| child.is_element() && child.tag_name().name() == "use")
        })
        .count()
}
fn target(document: &KernDocument) -> String {
    SerializePreset::TranscodaV2Training
        .try_serialize_document(document)
        .unwrap()
}
fn non_meter(text: &str) -> String {
    text.lines()
        .filter(|line| !line.starts_with("*M") && !line.starts_with("*met("))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn carried_meter_is_hidden_but_remains_active_and_other_labels_are_identical() {
    for meter in ["*M3/4", "*M4/4\n*met(c)", "*met(c|)", "*met(C)"] {
        let document = KernDocument::parse(&format!(
            "**kern\n*clefG2\n*k[f#]\n{meter}\n=1\n2.c\n=2\n2.d\n=3\n2.e\n*-\n"
        ))
        .unwrap();
        let standalone = split_trailing_bars_with_prefix_policy(
            &document,
            2,
            SlicePrefixPolicy::StandaloneExcerpt,
        )
        .unwrap()
        .removed;
        let hidden = split_trailing_bars_with_prefix_policy(
            &document,
            2,
            SlicePrefixPolicy::PageRendering {
                is_first_page: false,
            },
        )
        .unwrap()
        .removed;
        assert_eq!(glyphs(&render(&hidden).0), 0, "{meter}");
        assert_eq!(glyphs(&render(&standalone).0), 1, "{meter}");
        assert!(!target(&hidden).contains("*M"));
        assert!(!target(&hidden).contains("*met("));
        assert_eq!(non_meter(&target(&hidden)), non_meter(&target(&standalone)));
        let state = |doc: &KernDocument| {
            let score = Score::resolve(doc, Framing::Excerpt).unwrap();
            let data = doc
                .lines()
                .iter()
                .position(|line| matches!(line, kern_parser::KernLine::Data(_)))
                .unwrap();
            score.state_at(data, 0).track.meter.effective_meter()
        };
        assert_eq!(state(&hidden), state(&standalone), "{meter}");
    }
}

#[test]
fn opening_page_is_unchanged_and_authored_boundary_change_prints() {
    let document =
        KernDocument::parse("**kern\n*clefG2\n*k[]\n*M3/4\n=1\n2.c\n=2\n*M2/4\n2d\n=3\n2e\n*-\n")
            .unwrap();
    let standard =
        split_trailing_bars_with_prefix_policy(&document, 2, SlicePrefixPolicy::StandaloneExcerpt)
            .unwrap();
    let page = split_trailing_bars_with_prefix_policy(
        &document,
        2,
        SlicePrefixPolicy::PageRendering {
            is_first_page: false,
        },
    )
    .unwrap();
    assert_eq!(target(&standard.retained), target(&page.retained));
    assert_eq!(target(&standard.removed), target(&page.removed));
    assert_eq!(glyphs(&render(&page.removed).0), 1);
    assert!(target(&page.removed).contains("*M2/4"));
}

#[test]
fn in_page_meter_change_survives_hidden_header() {
    let mut page =
        KernDocument::parse("**kern\n*clefG2\n*k[]\n*M3/4\n=1\n2.c\n=2\n*M2/4\n2d\n*-\n").unwrap();
    let before = target(&page);
    hide_opening_meter(&mut page).unwrap();
    let (svg, sources) = render(&page);
    assert_eq!(glyphs(&svg), 1);
    project_meter_visibility(&mut page, &svg, &sources).unwrap();
    assert!(!target(&page).contains("*M3/4"));
    assert!(target(&page).contains("*M2/4"));
    assert_eq!(non_meter(&target(&page)), non_meter(&before));
}

#[test]
fn repeated_symbols_follow_glyphs_and_printed_numeric_repeats_stay() {
    for (meter, repeated, count) in [
        ("*M4/4\n*met(c)", "*met(c)", 1),
        ("*M2/2\n*met(c|)", "*met(c|)", 1),
        ("*M4/4", "*M4/4", 2),
        ("*M4/4\n*met(c)", "*M4/4\n*met(c)", 2),
        ("*met(C)", "*met(C)", 1),
    ] {
        let mut page = KernDocument::parse(&format!(
            "**kern\n*clefG2\n*k[]\n{meter}\n=1\n1c\n=2\n{repeated}\n1d\n*-\n"
        ))
        .unwrap();
        let (svg, sources) = render(&page);
        assert_eq!(glyphs(&svg), count, "{meter}");
        project_meter_visibility(&mut page, &svg, &sources).unwrap();
        let events = target(&page)
            .lines()
            .filter(|line| line.starts_with("*M") || line.starts_with("*met("))
            .count();
        assert_eq!(events, count, "{}", target(&page));
    }
}

#[test]
fn a_boundary_change_on_one_staff_does_not_expose_other_carried_meters() {
    let document = KernDocument::parse("**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*M3/4\t*M3/4\n=1\t=1\n2.C\t2.c\n=2\t=2\n*\t*M2/4\n2D\t2d\n*-\t*-\n").unwrap();
    let page = split_trailing_bars_with_prefix_policy(
        &document,
        1,
        SlicePrefixPolicy::PageRendering {
            is_first_page: false,
        },
    )
    .unwrap()
    .removed;
    assert_eq!(glyphs(&render(&page).0), 1);
    assert!(!target(&page).contains("*M3/4"));
    assert!(target(&page).contains("*M2/4"));
}

#[test]
fn opening_glyph_cannot_cover_an_unprinted_restatement_in_the_same_measure() {
    let mut page =
        KernDocument::parse("**kern\n*clefG2\n*k[]\n*M4/4\n*met(c)\n4c\n*met(c)\n4d\n*-\n")
            .unwrap();
    let (svg, sources) = render(&page);
    assert_eq!(glyphs(&svg), 1);
    project_meter_visibility(&mut page, &svg, &sources).unwrap();
    assert_eq!(target(&page).matches("*met(c)").count(), 1);
}

#[test]
fn a_printed_restatement_keeps_its_position_after_an_unprinted_symbol() {
    let mut page = KernDocument::parse(
        "**kern\n*clefG2\n*k[]\n*M4/4\n*met(c)\n4c\n*met(c)\n4d\n*M4/4\n*met(c)\n2e\n*-\n",
    )
    .unwrap();
    let (svg, sources) = render(&page);
    assert_eq!(glyphs(&svg), 2);
    project_meter_visibility(&mut page, &svg, &sources).unwrap();
    let label = target(&page);
    assert_eq!(label.matches("*met(c)").count(), 2, "{label}");
    assert!(label.contains("4c\n4d\n*met(c)\n2e"), "{label}");
}

#[test]
fn a_generated_opening_measure_does_not_require_a_source_owner() {
    let mut page =
        KernDocument::parse("**kern\n*clefG2\n*k[]\n*M4/4\n*met(c)\n1c\n=\n*met(c)\n1d\n*-\n")
            .unwrap();
    let (svg, mut sources) = render(&page);
    sources[0] = None;
    project_meter_visibility(&mut page, &svg, &sources).unwrap();
    assert_eq!(target(&page).matches("*met(c)").count(), 1);
}

#[test]
fn an_unprinted_symbol_after_a_hidden_header_is_not_a_target_change() {
    let mut page =
        KernDocument::parse("**kern\n*clefG2\n*k[]\n*M3/2\n*met()\n1.c\n=\n*met(c)\n1d\n*-\n")
            .unwrap();
    let active_meter = |document: &KernDocument| {
        let score = Score::resolve(document, Framing::Excerpt).unwrap();
        let last_data = document
            .lines()
            .iter()
            .rposition(|line| matches!(line, kern_parser::KernLine::Data(_)))
            .unwrap();
        score.state_at(last_data, 0).track.meter.effective_meter()
    };
    let active_before = active_meter(&page);
    let before = target(&page);
    let (svg, sources) = render(&page);
    assert_eq!(glyphs(&svg), 0);
    project_meter_visibility(&mut page, &svg, &sources).unwrap();
    let after = target(&page);
    assert_eq!(active_before, active_meter(&page));
    assert!(!after.contains("*met(c)"), "{after}");
    assert_eq!(non_meter(&before), non_meter(&after));
}
