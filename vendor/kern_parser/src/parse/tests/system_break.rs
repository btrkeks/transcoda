use crate::*;

#[test]
fn system_break_parses_with_group() {
    let document = KernDocument::parse("**kern\n4c\n!!LO:LB:g=original\n=\n4d\n*-\n").unwrap();
    let barline = document
        .lines()
        .iter()
        .find_map(|line| match line {
            KernLine::Barline(barline) => Some(barline),
            _ => None,
        })
        .expect("expected barline");

    let system_break = barline.system_break_before.as_ref().unwrap();
    assert_eq!(system_break.group, Some("original".to_owned()));
    assert_eq!(system_break.source_line, Some(3));
}

#[test]
fn system_break_canonicalizes_tabbed_overlay_suffix() {
    let document =
        KernDocument::parse("**kern\t**kern\t**kern\n4c\t4e\t4g\n!!LO:LB:g=original\t!\t!\n=\t=\t=\n4d\t4f\t4a\n*-\t*-\t*-\n")
            .unwrap();
    let barline = document
        .lines()
        .iter()
        .find_map(|line| match line {
            KernLine::Barline(barline) => Some(barline),
            _ => None,
        })
        .expect("expected barline");

    let system_break = barline.system_break_before.as_ref().unwrap();
    assert_eq!(system_break.group, Some("original".to_owned()));
    assert_eq!(system_break.source_line, Some(3));
}

#[test]
fn ungrouped_system_break_canonicalizes_tabbed_overlay_suffix() {
    let document =
        KernDocument::parse("**kern\t**kern\n4c\t4e\n!!LO:LB\t!\n=\t=\n4d\t4f\n*-\t*-\n").unwrap();
    let barline = document
        .lines()
        .iter()
        .find_map(|line| match line {
            KernLine::Barline(barline) => Some(barline),
            _ => None,
        })
        .expect("expected barline");

    let system_break = barline.system_break_before.as_ref().unwrap();
    assert_eq!(system_break.group, None);
    assert_eq!(system_break.source_line, Some(3));
}

#[test]
fn system_break_parses_without_group() {
    let document = KernDocument::parse("**kern\n4c\n!!LO:LB\n=\n4d\n*-\n").unwrap();
    let barline = document
        .lines()
        .iter()
        .find_map(|line| match line {
            KernLine::Barline(barline) => Some(barline),
            _ => None,
        })
        .expect("expected barline");

    let system_break = barline.system_break_before.as_ref().unwrap();
    assert_eq!(system_break.group, None);
    assert_eq!(system_break.source_line, Some(3));
}

#[test]
fn system_break_rejects_malformed_parameters() {
    for input in [
        "**kern\n4c\n!!LO:LB:x=1\n4d\n*-\n",
        "**kern\n4c\n!!LO:LB:g=\n4d\n*-\n",
        "**kern\n4c\n!!LO:LB:g=original:i\n4d\n*-\n",
    ] {
        let error = KernDocument::parse(input).unwrap_err();
        assert!(
            matches!(error, ParseError::InvalidSystemBreak { line: 3, .. }),
            "expected invalid system break for {input:?}, got {error:?}"
        );
    }
}

#[test]
fn longer_layout_codes_remain_skipped_global_comments() {
    let document = KernDocument::parse("**kern\n4c\n!!LO:LBX:g=original\n4d\n*-\n").unwrap();

    assert!(document.lines().iter().all(|line| match line {
        KernLine::Barline(barline) => barline.system_break_before.is_none(),
        _ => true,
    }));
}

#[test]
fn system_break_rejects_non_barline_target() {
    let error = KernDocument::parse("**kern\n4c\n!!LO:LB:g=original\n4d\n*-\n").unwrap_err();

    assert!(matches!(
        error,
        ParseError::InvalidSystemBreak { line: 3, .. }
    ));
}
