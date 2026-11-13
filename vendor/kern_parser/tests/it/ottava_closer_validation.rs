use kern_parser::{
    check_text, document::Framing, CheckConfig, CheckReport, ParseDialect, ValidationConfig,
    ValidationPass,
};

fn check(source: &str) -> CheckReport {
    check_text(
        source,
        &CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::State]),
            ..CheckConfig::default()
        },
    )
}

fn unmatched(source: &str, line: usize, field: usize) {
    rejected(source, "state.unmatched-close", line, field);
}

fn rejected(source: &str, code: &str, line: usize, field: usize) {
    let report = check(source);
    assert!(!report.success, "{report:?}");
    assert_eq!(report.diagnostics.len(), 1);
    let diagnostic = &report.diagnostics[0];
    assert_eq!(diagnostic.code, code);
    assert_eq!(diagnostic.line, Some(line));
    assert_eq!(diagnostic.column, Some(field));
    assert_eq!(diagnostic.rule, Some(ValidationPass::State));
}

#[test]
fn rejects_bare_closers_for_each_ottava_kind() {
    for kind in ["8va", "8ba", "15ma", "15ba"] {
        unmatched(&format!("**kern\n4cc\n*X{kind}\n*-\n"), 3, 0);
    }
}

#[test]
fn rejects_a_second_closer_on_a_later_row() {
    for middle in ["", "4dd\n", "=\n"] {
        let text = format!("**kern\n*8va\n4cc\n*X8va\n{middle}*X8va\n*-\n");
        unmatched(&text, if middle.is_empty() { 5 } else { 6 }, 0);
    }
}

#[test]
fn rejects_a_closer_on_a_different_original_spine() {
    unmatched(
        "**kern\t**kern\n*8va\t*\n4cc\t4ee\n*X8va\t*X8va\n*-\t*-\n",
        4,
        1,
    );
}

#[test]
fn reports_the_original_row_for_a_mixed_tandem_line() {
    unmatched("**kern\t**kern\n4cc\t4ee\n*clefG2\t*X8va\n*-\t*-\n", 3, 1);
}

#[test]
fn default_checks_reject_unmatched_closers() {
    let report = check_text("**kern\n4cc\n*X8va\n*-\n", &CheckConfig::default());
    assert!(!report.success);
    assert!(report
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "state.unmatched-close"));
}

#[test]
fn rejects_a_closer_of_a_different_kind() {
    rejected(
        "**kern\n*8va\n4cc\n*X15ma\n*X8va\n*-\n",
        "state.span-mismatch",
        4,
        0,
    );
}

#[test]
fn permits_shared_closers_only_on_their_matching_row() {
    let prefix = "**kern\n*8va\n*^\n4cc\t4ee\n*X8va\t*X8va\n";
    assert!(check(&format!("{prefix}*v\t*v\n*-\n")).success);
    unmatched(&format!("{prefix}*\t*X8va\n*v\t*v\n*-\n"), 6, 1);
}

#[test]
fn accepts_balanced_controls_sibling_changes_and_auxiliary_spines() {
    for kind in ["8va", "8ba", "15ma", "15ba"] {
        let text = format!("**kern\n*{kind}\n4cc\n*X{kind}\n*-\n");
        assert!(check(&text).success);
    }
    for source in [
        "**kern\n*^\n*8va\t*\n4cc\t4ee\n*\t*X8va\n*v\t*v\n*-\n",
        "**kern\n*8va\n*^\n4cc\t4ee\n*v\t*v\n*X8va\n*-\n",
        "**kern\n*8va\n4cc\n*8va\n4dd\n*X8va\n*-\n",
    ] {
        assert!(check(source).success, "{source}");
    }
}

#[test]
fn pairs_ottavas_in_auxiliary_spines_too() {
    unmatched("**kern\t**fing\n4cc\t1\n*\t*X8va\n*-\t*-\n", 3, 1);
}

#[test]
fn rejects_open_and_close_of_one_spine_on_a_shared_row() {
    for row in ["*8va\t*X8va", "*X8va\t*8va"] {
        rejected(
            &format!("**kern\n*^\n{row}\n4cc\t4ee\n*v\t*v\n*-\n"),
            "state.conflicting-values",
            3,
            1,
        );
    }
}

#[test]
fn rejects_orphan_closers_in_wrapperless_fragments() {
    let report = check_text(
        "4cc\n*X8va\n",
        &CheckConfig {
            input_dialect: ParseDialect::TranscodaV2,
            framing: Framing::Excerpt,
            validation: ValidationConfig::with_passes(vec![ValidationPass::State]),
            ..CheckConfig::default()
        },
    );
    assert!(!report.success, "{report:?}");
    assert_eq!(report.diagnostics[0].code, "state.unmatched-close");
    assert_eq!(report.diagnostics[0].line, Some(2));
    assert_eq!(report.diagnostics[0].column, Some(0));
}
