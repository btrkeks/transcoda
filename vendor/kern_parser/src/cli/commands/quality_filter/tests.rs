use std::fs;
use std::path::Path;

use clap::Parser;
use serde_json::Value;
use tempfile::TempDir;

use super::*;

fn write(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

fn valid_kern() -> &'static str {
    "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n=2\n4d\n*-\n"
}

fn invalid_duration_state_kern() -> &'static str {
    "**kern\t**kern\n\
*M4/4\t*M4/4\n\
2c\t4e\n\
*^\t*\n\
.\t4d\t4f\n\
*-\t*-\t*-\n"
}

#[test]
fn quality_filter_default_passes_do_not_render() {
    assert_eq!(
        quality_filter_config(&[]).passes(),
        &[
            crate::ValidationPass::State,
            crate::ValidationPass::DurationState,
            crate::ValidationPass::MeterBoundary,
            crate::ValidationPass::Beam,
            crate::ValidationPass::Rhythm,
        ]
    );
}

#[test]
fn quality_filter_preserves_nested_relative_paths_and_copies_exact_input() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("4_normalized");
    let output = temp.path().join("5_quality_filtered");
    let source = valid_kern();
    write(&input.join("nested/piece.krn"), source);

    quality_filter_command(QualityFilterArgs {
        input_dir: input,
        output_dir: output.clone(),
        workers: 1,
        framing: FramingArg::Excerpt,
        input_dialect: InputDialectArg::HumdrumKern,
        quiet: true,
        fail_fast: false,
        passes: vec![ValidationPass::State],
        stats_json: None,
    })
    .unwrap();

    assert_eq!(
        fs::read_to_string(output.join("nested/piece.krn")).unwrap(),
        source
    );
}

#[test]
fn quality_filter_omits_invalid_files_and_writes_stats() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("4_normalized");
    let output = temp.path().join("5_quality_filtered");
    let stats_path = temp.path().join("stats.json");
    write(&input.join("valid.krn"), valid_kern());
    write(&input.join("bad.krn"), "*unknown\n");

    quality_filter_command(QualityFilterArgs {
        input_dir: input,
        output_dir: output.clone(),
        workers: 2,
        framing: FramingArg::Excerpt,
        input_dialect: InputDialectArg::HumdrumKern,
        quiet: true,
        fail_fast: false,
        passes: vec![ValidationPass::State],
        stats_json: Some(stats_path.clone()),
    })
    .unwrap_err();

    assert!(output.join("valid.krn").exists());
    assert!(!output.join("bad.krn").exists());
    let stats: Value = serde_json::from_str(&fs::read_to_string(stats_path).unwrap()).unwrap();
    assert_eq!(stats["input_count"], 2);
    assert_eq!(stats["accepted_count"], 1);
    assert_eq!(stats["rejected_count"], 1);
    assert_eq!(
        stats["rejection_types"]["Parse:parse.unsupported-interpretation"],
        1
    );
}

#[test]
fn quality_filter_default_rejects_invalid_duration_state() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("4_normalized");
    let output = temp.path().join("5_quality_filtered");
    let stats_path = temp.path().join("stats.json");
    write(
        &input.join("bad_duration_state.krn"),
        invalid_duration_state_kern(),
    );

    quality_filter_command(QualityFilterArgs {
        input_dir: input,
        output_dir: output.clone(),
        workers: 1,
        framing: FramingArg::Excerpt,
        input_dialect: InputDialectArg::HumdrumKern,
        quiet: true,
        fail_fast: false,
        passes: vec![],
        stats_json: Some(stats_path.clone()),
    })
    .unwrap_err();

    assert!(!output.join("bad_duration_state.krn").exists());
    let stats: Value = serde_json::from_str(&fs::read_to_string(stats_path).unwrap()).unwrap();
    assert_eq!(stats["rejected_count"], 1);
    assert_eq!(
        stats["rejection_types"]["Validation:duration-state.overlap"],
        1
    );
}

#[test]
fn quality_filter_clears_stale_output() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("4_normalized");
    let output = temp.path().join("5_quality_filtered");
    write(&input.join("valid.krn"), valid_kern());
    write(&output.join("stale.krn"), "stale");

    quality_filter_command(QualityFilterArgs {
        input_dir: input,
        output_dir: output.clone(),
        workers: 1,
        framing: FramingArg::Excerpt,
        input_dialect: InputDialectArg::HumdrumKern,
        quiet: true,
        fail_fast: false,
        passes: vec![ValidationPass::State],
        stats_json: None,
    })
    .unwrap();

    assert!(output.join("valid.krn").exists());
    assert!(!output.join("stale.krn").exists());
}

#[test]
fn quality_filter_fail_fast_exits_nonzero_on_rejection() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("4_normalized");
    let output = temp.path().join("5_quality_filtered");
    write(&input.join("bad.krn"), "*unknown\n");
    write(&input.join("valid.krn"), valid_kern());

    let error = quality_filter_command(QualityFilterArgs {
        input_dir: input,
        output_dir: output,
        workers: 1,
        framing: FramingArg::Excerpt,
        input_dialect: InputDialectArg::HumdrumKern,
        quiet: true,
        fail_fast: true,
        passes: vec![ValidationPass::State],
        stats_json: None,
    })
    .unwrap_err();

    assert_eq!(error.to_string(), "batch contains rejected inputs");
}

#[test]
fn quality_filter_parses_selected_passes() {
    let args = QualityFilterArgs::try_parse_from([
        "quality-filter-dir",
        "--rule",
        "state",
        "--rule",
        "duration-state",
        "--rule",
        "rhythm",
        "in",
        "out",
    ])
    .unwrap();

    assert_eq!(
        args.passes,
        vec![
            ValidationPass::State,
            ValidationPass::DurationState,
            ValidationPass::Rhythm
        ]
    );
}

#[test]
fn quality_filter_defaults_to_training_excerpts_and_complete_is_opt_in() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("4_normalized");
    let output = temp.path().join("5_quality_filtered");
    let snippet = "*clefG2\n*k[]\n*M4/4\n4c\n4d\n4e\n4f\n=\n";
    write(&input.join("piece.krn"), snippet);
    let args = QualityFilterArgs::try_parse_from([
        "quality-filter-dir",
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--quiet",
    ])
    .unwrap();
    assert!(matches!(args.framing, FramingArg::Excerpt));
    quality_filter_command(args).unwrap();
    assert_eq!(
        fs::read_to_string(output.join("piece.krn")).unwrap(),
        snippet
    );

    let args = QualityFilterArgs::try_parse_from([
        "quality-filter-dir",
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--quiet",
        "--framing",
        "complete",
    ])
    .unwrap();
    let error = quality_filter_command(args).unwrap_err();
    assert_eq!(crate::cli::exit_code(error.as_ref()), 1);
    assert!(!output.join("piece.krn").exists());
}

#[test]
fn quality_filter_accepts_explicit_transcoda_dialect() {
    let args = QualityFilterArgs::try_parse_from([
        "quality-filter-dir",
        "in",
        "out",
        "--input-dialect",
        "transcoda-v2",
        "--framing",
        "excerpt",
    ])
    .unwrap();
    assert!(matches!(args.input_dialect, InputDialectArg::TranscodaV2));
    assert!(matches!(args.framing, FramingArg::Excerpt));
}
