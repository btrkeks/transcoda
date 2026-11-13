#![cfg(feature = "cli")]

use kern_parser::{
    check_text, validation::RuleStatus, CheckConfig, CheckSeverity, ValidationConfig,
    ValidationPass,
};

#[test]
fn timing_and_state_errors_block_the_selected_rule() {
    for (input, cause, code) in [
        (
            "**kern\n*^\n*^\t*\n2r\t4C\t4G\n4r\t4D\t4A\n*-\t*-\t*-\n",
            "duration-state",
            "duration-state.overlap",
        ),
        (
            "**kern\t**kern\n*v\t*v\n4r\n*-\n",
            "state",
            "state.cross-track-merge",
        ),
    ] {
        let report = check_text(
            input,
            &CheckConfig {
                validation: ValidationConfig::with_passes(vec![ValidationPass::RestPlacement]),
                ..Default::default()
            },
        );
        assert!(!report.success);
        assert_eq!(
            report.rules[0].status,
            RuleStatus::Blocked {
                causes: vec![cause.into()]
            }
        );
        assert_eq!(report.diagnostics.len(), 1);
        assert_eq!(report.diagnostics[0].code, code);
        assert_eq!(report.diagnostics[0].severity, CheckSeverity::Error);
    }
}
