use kern_parser::{
    check::{CheckConfig, CheckSeverity},
    check_text,
    validation::{self, RuleStatus, ValidationCode, ValidationConfig, ValidationSeverity},
    KernDocument, ValidationPass,
};

fn cross_staff_config() -> CheckConfig {
    CheckConfig {
        validation: ValidationConfig::with_passes(vec![ValidationPass::CrossStaff]),
        ..Default::default()
    }
}

fn findings(input: &str) -> Vec<(String, usize, usize)> {
    let report = check_text(input, &cross_staff_config());
    assert!(report.success, "{report:?}");
    assert!(report.diagnostics.iter().all(|diagnostic| {
        diagnostic.rule == Some(ValidationPass::CrossStaff)
            && diagnostic.severity == CheckSeverity::Warning
    }));
    report
        .diagnostics
        .into_iter()
        .map(|diagnostic| {
            (
                diagnostic.code,
                diagnostic.line.unwrap(),
                diagnostic.column.unwrap(),
            )
        })
        .collect()
}

#[test]
fn default_public_checks_warn_and_deny_warnings_rejects() {
    let input = "**kern\t**kern\n4C<\t4c>\n*-\t*-\n";
    let report = check_text(input, &CheckConfig::default());
    assert!(report.success);
    assert_eq!(report.diagnostics.len(), 2);
    assert_eq!(report.diagnostics[0].code, "cross-staff.no-staff-below");
    assert_eq!(report.diagnostics[0].line, Some(2));
    assert_eq!(report.diagnostics[0].column, Some(0));
    assert_eq!(report.diagnostics[1].code, "cross-staff.no-staff-above");
    assert_eq!(report.diagnostics[1].column, Some(1));
    assert_eq!(
        report.diagnostics[0].message,
        "pitch-attached `<` moves a note below the lowest staff 2, but no destination staff exists below it"
    );
    assert_eq!(
        report.diagnostics[1].message,
        "pitch-attached `>` moves a note above the uppermost staff 1, but no destination staff exists above it"
    );
    assert!(
        !check_text(
            input,
            &CheckConfig {
                deny_warnings: true,
                ..Default::default()
            }
        )
        .success
    );

    let document = KernDocument::parse(input).unwrap();
    let report = validation::validate_report(&document);
    assert!(!report.has_errors());
    assert_eq!(report.findings.len(), 2);
    assert!(report.findings.iter().all(|finding| {
        finding.code == ValidationCode::CrossStaff
            && finding.severity == ValidationSeverity::Warning
    }));
    validation::validate(&document).unwrap();
}

#[test]
fn rule_selection_and_prerequisites_use_the_registry() {
    let rule = ValidationPass::from_id("cross-staff").unwrap();
    assert_eq!(rule, ValidationPass::CrossStaff);
    assert!(rule.definition().default);
    assert_eq!(rule.definition().prerequisites, &[ValidationPass::State]);

    let input = "**kern\n4c>\n*-\n";
    let report = check_text(input, &cross_staff_config());
    assert_eq!(report.rules.len(), 1);
    assert_eq!(report.rules[0].rule, rule);
    assert_eq!(report.rules[0].status, RuleStatus::Findings);
    let report = check_text(
        input,
        &CheckConfig {
            validation: ValidationConfig::with_passes(vec![ValidationPass::State]),
            ..Default::default()
        },
    );
    assert!(report.diagnostics.is_empty());

    let report = check_text(
        "**kern\t**kern\n4C<\t4c>\n*v\t*v\n4c>\n*-\n",
        &cross_staff_config(),
    );
    assert!(!report.success);
    assert_eq!(
        report.rules[0].status,
        RuleStatus::Blocked {
            causes: vec!["state".into()]
        }
    );
}

#[test]
fn accepts_inward_and_middle_staff_moves() {
    assert!(findings("**kern\t**kern\n4C>\t4c<\n*-\t*-\n").is_empty());
    assert!(findings("**kern\t**kern\t**kern\n4C>\t4e> 4g<\t4cc<\n*-\t*-\t*-\n").is_empty());
}

#[test]
fn single_staff_warns_for_each_field_and_direction_once() {
    assert_eq!(
        findings("**kern\n4c> 4e> 4g< 4b<\n*-\n"),
        [
            ("cross-staff.no-staff-above".into(), 2, 0),
            ("cross-staff.no-staff-below".into(), 2, 0)
        ]
    );
}

/// Verovio numbers staves by position; declared numbers only choose whether
/// to count from the left or the right (MEI `staff/@n` of Verovio 6.3).
#[test]
fn declared_ids_choose_the_direction_and_duplicates_are_separate_staves() {
    // 7, 2, 7 decreases, so staves count from the right: 3, 2, 1.
    assert_eq!(
        findings("**kern\t**kern\t**kern\n*staff7\t*staff2\t*staff7\n4C<\t4c>\t4E<\n4C>\t4c<\t4E>\n*-\t*-\t*-\n"),
        [
            ("cross-staff.no-staff-below".into(), 3, 0),
            ("cross-staff.no-staff-above".into(), 4, 2)
        ]
    );
    // 5, 5 never decreases, so staves count from the left: 1, 2.
    assert_eq!(
        findings("**kern\t**kern\n*staff5\t*staff5\n4c>\t4e<\n*-\t*-\n"),
        [
            ("cross-staff.no-staff-above".into(), 3, 0),
            ("cross-staff.no-staff-below".into(), 3, 1)
        ]
    );
}

#[test]
fn implicit_order_counts_kern_roots_and_ignores_auxiliary_spines() {
    assert_eq!(
        findings("**text\t**kern\t**text\t**kern\t**dynam\nleft\t4C<\tmiddle\t4c>\tp\nleft\t4C>\tmiddle\t4c<\tp\n*-\t*-\t*-\t*-\t*-\n"),
        [
            ("cross-staff.no-staff-below".into(), 2, 1),
            ("cross-staff.no-staff-above".into(), 2, 3)
        ]
    );
}

#[test]
fn split_and_merged_layers_inherit_their_root_staff() {
    assert_eq!(
        findings("**kern\t**kern\n*^\t*\n4C<\t4E>\t4c>\n*v\t*v\t*\n4C<\t4c<\n*-\t*-\n"),
        [
            ("cross-staff.no-staff-below".into(), 3, 0),
            ("cross-staff.no-staff-above".into(), 3, 2),
            ("cross-staff.no-staff-below".into(), 5, 0)
        ]
    );
}

#[test]
fn staff_termination_does_not_remove_a_destination_from_the_score() {
    assert_eq!(
        findings("**kern\t**kern\t**kern\n4C\t4e\t4cc\n*-\t*\t*\n4e<\t4cc>\n*\t*-\n4e>\n*-\n"),
        [("cross-staff.no-staff-above".into(), 4, 1)]
    );
}

#[test]
fn partial_and_late_declarations_still_number_every_staff() {
    // One declaration counts from the left: staves 1 and 2.
    assert_eq!(
        findings("**kern\t**kern\n*staff1\t*\n4c>\t4e<\n*-\t*-\n"),
        [
            ("cross-staff.no-staff-above".into(), 3, 0),
            ("cross-staff.no-staff-below".into(), 3, 1)
        ]
    );
    // A declaration after the first data line does not renumber.
    assert_eq!(
        findings("**kern\n*staff1\n4c>\n*staff2\n4d<\n*-\n"),
        [
            ("cross-staff.no-staff-above".into(), 3, 0),
            ("cross-staff.no-staff-below".into(), 5, 0)
        ]
    );
}

#[test]
fn zero_staff_declaration_remains_a_parse_error() {
    let report = check_text(
        "**kern\t**kern\n*staff1\t*staff0\n4c>\t4e<\n*-\t*-\n",
        &cross_staff_config(),
    );
    assert!(!report.success);
    assert_eq!(report.diagnostics[0].code, "parse.invalid-staff");
}

#[test]
fn chord_members_grace_notes_and_grace_chords_are_checked() {
    assert_eq!(
        findings("**kern\n4c 4e> 4g>\n8qe#>\n8qcc< dd#<\n4c\n*-\n"),
        [
            ("cross-staff.no-staff-above".into(), 2, 0),
            ("cross-staff.no-staff-above".into(), 3, 0),
            ("cross-staff.no-staff-below".into(), 4, 0)
        ]
    );
}

#[test]
fn ignores_rest_positions_and_tie_or_slur_angle_brackets() {
    assert!(
        findings("**kern\n4r>\n4r<\n4c[>\n4c_>\n4c]>\n4d(<\n4e)<\n8qf[>\n8qf]>\n4c\n*-\n")
            .is_empty()
    );
    let report = check_text("**kern\n4r>\n4r<\n*-\n", &cross_staff_config());
    assert_eq!(report.rules[0].status, RuleStatus::NotApplicable);
}
