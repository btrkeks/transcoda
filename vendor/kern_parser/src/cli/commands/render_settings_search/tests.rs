use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use crate::analysis::{
    find_seeded_render_settings, RenderFitAttempt, RenderFitConfig, RenderFitQualityReport,
    RenderFitResult, RenderFitSearchSeed, RenderedPageArtifact, SvgContentBox, SvgPageWhitespace,
    SvgWhitespace, VerovioRenderSettings,
};
use crate::dataset_manifest::DATASET_MANIFEST_FILE_NAME;
use crate::{
    augment_for_render_with_report, KernDocument, RenderAugmentationConfig,
    RenderAugmentationReport, TupletVisibilitySpanReport,
};
use serde_json::{json, Value};
use tempfile::TempDir;

use super::output::{
    create_render_settings_search_writer, load_render_settings_search_resume, resume_file_key,
    selected_svg_path, write_render_settings_search_row,
};
use super::row::{
    FileStats, GeometricAugmentationStatusPayload, RenderSettingsSearchRow,
    RenderSettingsSearchStatus, SvgOverlapGatePayload,
};
use super::runner::{
    process_render_settings_search_files_with_worker,
    process_render_settings_search_files_with_worker_and_profile,
    render_settings_search_work_items,
};
use super::service::{
    kern_render_augmentation_config, kern_render_augmentation_seed, render_settings_search_file,
    render_settings_search_file_profiled, row_from_successful_search,
};
use super::worker::{
    render_settings_search_file_via_child, render_settings_search_file_via_child_profiled,
    render_settings_search_row_from_child_output,
};
use super::*;
use crate::cli::commands::collect_krn_files;

fn write(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

#[cfg(unix)]
fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}

fn valid_kern() -> &'static str {
    "**kern\n*clefG2\n*k[]\n*M4/4\n4c\n*-\n"
}

fn mixed_spine_kern() -> &'static str {
    "**kern\t**dynam\t**text\t**kern\n*clefG2\t*\t*\t*clefG2\n*k[]\t*\t*\t*k[]\n*M4/4\t*\t*\t*M4/4\n4c\tmf\tla\t4e\n*-\t*-\t*-\t*-\n"
}

fn render_settings_search_args(
    input_dir: PathBuf,
    output_jsonl: PathBuf,
) -> RenderSettingsSearchArgs {
    let output_svg_dir = output_jsonl
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("7_svg_renderings");
    RenderSettingsSearchArgs {
        input_dir,
        output_jsonl,
        output_svg_dir,
        workers: 1,
        svg_overlap_gate: SvgOverlapGateArg::Off,
        svg_overlap_min_ratio: 0.15,
        svg_overlap_fallback: SvgOverlapFallbackArg::HorizontalRelief,
        quiet: true,
        resume: false,
        profile_jsonl: None,
    }
}

fn run_render_settings_search_with_in_process_worker(args: RenderSettingsSearchArgs) {
    let files = collect_krn_files(&args.input_dir).unwrap();
    let mut work_items = render_settings_search_work_items(&files);
    if args.resume {
        let completed =
            load_render_settings_search_resume(&args.output_jsonl, &args.input_dir, true).unwrap();
        work_items
            .retain(|item| !completed.contains(&resume_file_key(&item.path, &args.input_dir)));
    }
    process_render_settings_search_files_with_worker_and_profile(
        &work_items,
        args.workers,
        &args.output_jsonl,
        args.resume,
        args.profile_jsonl.as_deref(),
        &|| {},
        |item| {
            render_settings_search_file_profiled(
                &item.path,
                &args.input_dir,
                &args.output_svg_dir,
                render_fit_search_options(
                    args.svg_overlap_gate,
                    args.svg_overlap_min_ratio,
                    args.svg_overlap_fallback,
                ),
                RenderFitSearchSeed {
                    index: item.candidate_index,
                    count: item.candidate_count,
                },
                args.profile_jsonl.is_some(),
            )
        },
    )
    .unwrap();
}

fn read_jsonl(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn jsonl_row(file: &Path, status: &str) -> String {
    let mut row = json!({
        "schema_version": "1.0",
        "file": file.display().to_string(),
        "line_count": 1,
        "character_count": 12,
        "initial_spine_count": 1,
        "staff_count": 1,
        "barline_count": 0,
        "status": status,
        "selected_settings": null,
        "selected_score": null,
        "selected_page_count": null,
        "selected_system_count": null,
        "selected_whitespace": null,
        "selected_content_box": null,
        "geometric_augmentation": null,
        "selected_svg_file": null,
        "attempt_count": 0,
        "accepted_attempt_count": 0,
        "parse_error": null,
        "search_error": null,
        "verovio_stderr": null
    });
    if status == "success" {
        row["selected_render_source_lines"] = json!([1, null, 3]);
        row["render_augmentation"] = serde_json::to_value(
            augment_for_render_with_report(
                &KernDocument::parse(valid_kern()).unwrap(),
                &RenderAugmentationConfig::default(),
                Default::default(),
            )
            .unwrap()
            .1,
        )
        .unwrap();
    }
    row.to_string()
}

fn legacy_success_jsonl_row_without_source_map(file: &Path) -> String {
    let mut row = serde_json::from_str::<Value>(&jsonl_row(file, "success")).unwrap();
    row.as_object_mut()
        .unwrap()
        .remove("selected_render_source_lines");
    row.to_string()
}

fn svg_overlap_gate_payload(
    hit: bool,
    rejected_attempt_count: usize,
    fallback_attempt_count: usize,
) -> SvgOverlapGatePayload {
    SvgOverlapGatePayload {
        enabled: true,
        hit,
        rejected_attempt_count,
        fallback_attempt_count,
        worst_overlap_count: if hit { 1 } else { 0 },
        worst_overlap_ratio: hit.then_some(0.25),
        worst_pairs: Vec::new(),
    }
}

#[test]
#[cfg(feature = "engraving")]
fn render_settings_search_writes_successful_search_row() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let output_path = temp.path().join("settings.jsonl");
    let piece_path = input_path.join("piece.krn");
    write(&piece_path, valid_kern());

    run_render_settings_search_with_in_process_worker(render_settings_search_args(
        input_path,
        output_path.clone(),
    ));

    let rows = read_jsonl(&output_path);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["schema_version"], "1.7");
    assert_eq!(rows[0]["file"], piece_path.display().to_string());
    assert_eq!(rows[0]["line_count"], 6);
    assert_eq!(rows[0]["character_count"], valid_kern().chars().count());
    assert_eq!(rows[0]["initial_spine_count"], 1);
    assert_eq!(rows[0]["staff_count"], 1);
    assert_eq!(rows[0]["barline_count"], 0);
    assert_eq!(rows[0]["status"], "success");
    assert!(rows[0]["selected_settings"].is_object());
    assert!(rows[0]["selected_score"].is_number());
    assert_eq!(rows[0]["selected_page_count"], 1);
    assert_eq!(rows[0]["selected_system_count"], 1);
    assert!(rows[0]["selected_whitespace"].is_object());
    assert!(rows[0]["selected_content_box"].is_object());
    assert!(rows[0]["geometric_augmentation"].is_object());
    assert!(rows[0]["selected_svg_file"].is_string());
    assert_eq!(rows[0]["selected_svg_has_red_color"], false);
    assert!(rows[0]["render_augmentation"].is_object());
    assert_eq!(rows[0]["render_augmentation"]["tuplet_visibility_count"], 0);
    let source_lines = rows[0]["selected_render_source_lines"].as_array().unwrap();
    assert!(!source_lines.is_empty());
    assert!(source_lines.iter().any(|line| line.is_null()));
    assert!(rows[0]["svg_overlap_gate"].is_null());
    assert_eq!(rows[0]["attempt_count"].as_u64().unwrap(), 1);
    assert_eq!(rows[0]["accepted_attempt_count"].as_u64().unwrap(), 1);
    assert!(rows[0]["parse_error"].is_null());
    assert!(rows[0]["search_error"].is_null());
    assert!(rows[0]["verovio_stderr"].is_null());
    let selected_svg = fs::read_to_string(temp.path().join("7_svg_renderings/piece.svg")).unwrap();
    assert!(!selected_svg.contains("bounding-box"));
    assert!(!selected_svg.contains("content-bounding-box"));
}

#[test]
#[cfg(feature = "engraving")]
fn render_settings_search_tuplet_report_matches_selected_svg() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let output_path = temp.path().join("settings.jsonl");
    let output_svg_dir = temp.path().join("7_svg_renderings");
    let piece_path = input_path.join("piece.krn");
    let input = "**kern\n*clefG2\n*k[]\n*M4/4\n12c\n12d\n12e\n4f\n*-\n";
    write(&piece_path, input);
    let document = KernDocument::parse(input).unwrap();

    let seed = (0..256)
        .find_map(|index| {
            let seed = RenderFitSearchSeed { index, count: 256 };
            let (_, report) = augment_for_render_with_report(
                &document,
                &kern_render_augmentation_config(),
                kern_render_augmentation_seed(seed),
            )
            .unwrap();
            (report.tuplet_visibility_count == 1).then_some(seed)
        })
        .expect("bounded seed search must select the only tuplet run");

    let baseline = find_seeded_render_settings(
        &document,
        VerovioRenderSettings::medium(),
        RenderFitConfig::single_page_balanced_for_document(&document),
        seed,
    )
    .unwrap();
    assert!(
        baseline.selected_pages[0]
            .svg
            .matches("class=\"tupletNum\"")
            .count()
            > 0
    );

    let row = render_settings_search_file(
        &piece_path,
        &input_path,
        &output_svg_dir,
        render_fit_search_options(
            SvgOverlapGateArg::Off,
            0.15,
            SvgOverlapFallbackArg::HorizontalRelief,
        ),
        seed,
    );
    assert!(row.search_success(), "{:?}", row.search_error);
    let mut writer = create_render_settings_search_writer(&output_path, false).unwrap();
    write_render_settings_search_row(&mut writer, &row).unwrap();
    drop(writer);

    let rows = read_jsonl(&output_path);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["schema_version"], "1.7");
    assert_eq!(rows[0]["render_augmentation"]["tuplet_visibility_count"], 1);
    assert_eq!(
        rows[0]["render_augmentation"]["tuplet_visibility_spans"],
        json!([{
            "staff_index": 0,
            "spine_index": 0,
            "start_source_line": 5,
            "end_source_line": 7,
        }])
    );
    let selected_svg = fs::read_to_string(row.selected_svg_file.unwrap()).unwrap();
    assert_eq!(
        selected_svg.matches("class=\"tupletNum\"").count(),
        0,
        "{selected_svg}"
    );
}

#[test]
#[cfg(feature = "engraving")]
fn render_settings_search_writes_separate_profile_jsonl_when_requested() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let output_path = temp.path().join("settings.jsonl");
    let profile_path = temp.path().join("profile.jsonl");
    let piece_path = input_path.join("piece.krn");
    write(&piece_path, valid_kern());

    let mut args = render_settings_search_args(input_path, output_path.clone());
    args.profile_jsonl = Some(profile_path.clone());
    run_render_settings_search_with_in_process_worker(args);

    let rows = read_jsonl(&output_path);
    assert_eq!(rows.len(), 1);
    assert!(!rows[0].as_object().unwrap().contains_key("timings_ms"));

    let profile_rows = read_jsonl(&profile_path);
    assert_eq!(profile_rows.len(), 1);
    let profile = &profile_rows[0];
    assert_eq!(profile["schema_version"], "1.0");
    assert_eq!(profile["file"], piece_path.display().to_string());
    assert_eq!(profile["status"], "success");
    assert_eq!(profile["attempt_count"], rows[0]["attempt_count"]);
    assert!(profile["timings_ms"]["total"].as_f64().unwrap() >= 0.0);
    assert!(profile["timings_ms"]["render_fit_search"].as_f64().unwrap() >= 0.0);
    assert!(profile["render_fit"]["candidate_count"].as_u64().unwrap() >= 1);
    assert_eq!(
        profile["render_fit"]["attempts"].as_array().unwrap().len() as u64,
        profile["attempt_count"].as_u64().unwrap()
    );
    assert_eq!(profile["render_fit"]["attempts"][0]["origin"], "primary");
    assert!(
        profile["render_fit"]["attempts"][0]["render_ms"]
            .as_f64()
            .unwrap()
            >= 0.0
    );
}

#[test]
#[cfg(feature = "engraving")]
fn render_settings_search_counts_only_initial_kern_spines_as_staffs() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let piece_path = input_path.join("piece.krn");
    write(&piece_path, mixed_spine_kern());

    let row = render_settings_search_file(
        &piece_path,
        temp.path().join("other-root").as_path(),
        &temp.path().join("svg"),
        render_fit_search_options(
            SvgOverlapGateArg::Off,
            0.15,
            SvgOverlapFallbackArg::HorizontalRelief,
        ),
        RenderFitSearchSeed { index: 0, count: 1 },
    );

    assert_eq!(row.initial_spine_count, Some(4));
    assert_eq!(row.staff_count, Some(2));
    assert!(row.search_failed());
}

#[test]
#[cfg(feature = "engraving")]
fn render_settings_search_stops_after_first_fit() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let output_path = temp.path().join("settings.jsonl");
    write(&input_path.join("piece.krn"), valid_kern());

    run_render_settings_search_with_in_process_worker(render_settings_search_args(
        input_path,
        output_path.clone(),
    ));

    let rows = read_jsonl(&output_path);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["status"], "success");
    assert!(rows[0]["attempt_count"].as_u64().unwrap() <= 8);
    assert!(rows[0]["accepted_attempt_count"].as_u64().unwrap() >= 1);
    assert_ne!(rows[0]["selected_settings"]["page_width"], 3000);
}

#[test]
#[cfg(feature = "engraving")]
fn render_settings_search_records_parse_errors_without_aborting() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let output_path = temp.path().join("settings.jsonl");
    write(&input_path.join("bad.krn"), "*unknown\n");
    write(&input_path.join("good.krn"), valid_kern());

    run_render_settings_search_with_in_process_worker(render_settings_search_args(
        input_path,
        output_path.clone(),
    ));

    let rows = read_jsonl(&output_path);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["status"], "parse_error");
    assert!(!rows[0]["parse_error"].as_str().unwrap().is_empty());
    assert!(rows[0]["search_error"].is_null());
    assert!(rows[0]["render_augmentation"].is_null());
    assert_eq!(rows[1]["status"], "success");
    assert!(rows[1]["parse_error"].is_null());
    assert!(rows[1]["search_error"].is_null());
    assert!(rows[1]["render_augmentation"].is_object());
}

#[test]
#[cfg(feature = "engraving")]
fn render_settings_search_creates_output_parent_directory() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let output_path = temp.path().join("nested/settings/render_settings.jsonl");
    write(&input_path.join("piece.krn"), valid_kern());

    run_render_settings_search_with_in_process_worker(render_settings_search_args(
        input_path,
        output_path.clone(),
    ));

    assert!(output_path.exists());
    assert_eq!(read_jsonl(&output_path).len(), 1);
}

#[test]
fn render_settings_search_allows_empty_input_directory() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let output_path = temp.path().join("settings.jsonl");
    fs::create_dir_all(&input_path).unwrap();

    run_render_settings_search_with_in_process_worker(render_settings_search_args(
        input_path,
        output_path.clone(),
    ));

    assert_eq!(fs::read_to_string(output_path).unwrap(), "");
}

#[test]
fn render_settings_search_manifest_update_preserves_existing_components() {
    let temp = TempDir::new().unwrap();
    let dataset_root = temp.path().join("train/pdmx");
    let output_svg_dir = dataset_root.join("7_svg_renderings");
    let output_jsonl = output_svg_dir.join("render_settings.jsonl");
    write(&output_svg_dir.join("piece.svg"), "<svg/>");
    write(&output_jsonl, "{}\n");
    write(
        &dataset_root.join(DATASET_MANIFEST_FILE_NAME),
        r#"{
            "schema_version": "transcoda.dataset_manifest.v2",
            "dataset": {"split": "train", "name": "pdmx", "root": "unused"},
            "components": {
                "layout_targets": {
                    "content_id": "target-v1",
                    "producer": "test",
                    "paths": ["7_targets"],
                    "summary": {}
                }
            }
        }"#,
    );
    let args = RenderSettingsSearchArgs {
        input_dir: dataset_root.join("6_concatenated"),
        output_jsonl,
        output_svg_dir,
        workers: 1,
        svg_overlap_gate: SvgOverlapGateArg::Off,
        svg_overlap_min_ratio: 0.15,
        svg_overlap_fallback: SvgOverlapFallbackArg::HorizontalRelief,
        quiet: true,
        resume: false,
        profile_jsonl: None,
    };
    let summary = RenderSettingsSearchSummary {
        file_count: 1,
        search_success_count: 1,
        parse_error_count: 0,
        search_error_count: 0,
        ..Default::default()
    };

    update_render_settings_manifest(&args, &summary).unwrap();

    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(dataset_root.join(DATASET_MANIFEST_FILE_NAME)).unwrap(),
    )
    .unwrap();
    assert_eq!(
        manifest["components"]["layout_targets"]["content_id"],
        "target-v1"
    );
    assert!(manifest["components"]["svg_renders"]["content_id"].is_string());
    assert_eq!(
        manifest["components"]["svg_renders"]["summary"]["svg_overlap_gate_hit_count"],
        0
    );
    assert_eq!(
        manifest["components"]["svg_renders"]["summary"]["svg_overlap_rejected_attempt_count"],
        0
    );
    assert_eq!(
        manifest["components"]["svg_renders"]["paths"],
        json!(["7_svg_renderings", "7_svg_renderings/render_settings.jsonl"])
    );
}

#[test]
#[cfg(feature = "engraving")]
fn render_settings_search_parallel_output_contains_all_files_without_ordering_requirement() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let output_path = temp.path().join("settings.jsonl");
    let a_path = input_path.join("a/piece.krn");
    let z_path = input_path.join("z/piece.krn");
    write(&z_path, valid_kern());
    write(&a_path, valid_kern());

    let mut args = render_settings_search_args(input_path, output_path.clone());
    args.workers = 2;
    run_render_settings_search_with_in_process_worker(args);

    let rows = read_jsonl(&output_path);
    let mut files = rows
        .iter()
        .map(|row| row["file"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(
        files,
        vec![a_path.display().to_string(), z_path.display().to_string()]
    );
}

#[test]
fn render_settings_search_summary_counts_unique_svg_overlap_gate_hits() {
    let temp = TempDir::new().unwrap();
    let output_path = temp.path().join("settings.jsonl");
    let a_path = temp.path().join("a.krn");
    let b_path = temp.path().join("b.krn");
    let c_path = temp.path().join("c.krn");
    let files = vec![a_path.clone(), b_path.clone(), c_path.clone()];
    let work_items = render_settings_search_work_items(&files);

    let summary = process_render_settings_search_files_with_worker(
        &work_items,
        1,
        &output_path,
        false,
        &|| {},
        |item| {
            let stats = FileStats {
                line_count: 1,
                character_count: 1,
                initial_spine_count: Some(1),
                staff_count: Some(1),
                barline_count: Some(0),
            };
            if item.path == a_path {
                let mut row = RenderSettingsSearchRow::search_error(
                    &item.path,
                    stats,
                    "placeholder".to_owned(),
                );
                row.status = RenderSettingsSearchStatus::Success;
                row.search_error = None;
                row.with_svg_overlap_gate(Some(svg_overlap_gate_payload(true, 2, 1)))
            } else if item.path == b_path {
                RenderSettingsSearchRow::search_error(&item.path, stats, "no fit".to_owned())
                    .with_svg_overlap_gate(Some(svg_overlap_gate_payload(true, 3, 2)))
            } else {
                RenderSettingsSearchRow::search_error(&item.path, stats, "other".to_owned())
                    .with_svg_overlap_gate(Some(svg_overlap_gate_payload(false, 0, 0)))
            }
        },
    )
    .unwrap();

    assert_eq!(summary.file_count, 3);
    assert_eq!(summary.svg_overlap_gate_row_count, 3);
    assert_eq!(summary.svg_overlap_gate_hit_count, 2);
    assert_eq!(summary.svg_overlap_gate_resolved_count, 1);
    assert_eq!(summary.svg_overlap_gate_unresolved_count, 1);
    assert_eq!(summary.svg_overlap_rejected_attempt_count, 5);
    assert_eq!(summary.svg_overlap_fallback_attempt_count, 3);

    let rows = read_jsonl(&output_path);
    assert_eq!(rows[0]["svg_overlap_gate"]["enabled"], true);
    assert_eq!(rows[0]["svg_overlap_gate"]["hit"], true);
}

#[test]
#[cfg(feature = "engraving")]
fn render_settings_search_resume_skips_completed_rows_and_retries_search_errors() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let output_path = temp.path().join("settings.jsonl");
    let completed_path = input_path.join("completed.krn");
    let parse_error_path = input_path.join("parse_error.krn");
    let retry_path = input_path.join("retry.krn");
    let new_path = input_path.join("new.krn");
    write(&completed_path, valid_kern());
    write(&parse_error_path, "*unknown\n");
    write(&retry_path, valid_kern());
    write(&new_path, valid_kern());
    write(
        &output_path,
        &format!(
            "{}\n{}\n{}\n",
            jsonl_row(&completed_path, "success"),
            jsonl_row(&parse_error_path, "parse_error"),
            jsonl_row(&retry_path, "search_error")
        ),
    );

    let mut args = render_settings_search_args(input_path, output_path.clone());
    args.resume = true;
    run_render_settings_search_with_in_process_worker(args);

    let rows = read_jsonl(&output_path);
    assert_eq!(rows.len(), 5);
    let new_rows = &rows[3..];
    let mut files = new_rows
        .iter()
        .map(|row| row["file"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(
        files,
        vec![
            new_path.display().to_string(),
            retry_path.display().to_string()
        ]
    );
    assert!(new_rows.iter().all(|row| row["status"] == "success"));
}

#[test]
#[cfg(feature = "engraving")]
fn render_settings_search_resume_regenerates_legacy_success_without_source_map() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let output_path = temp.path().join("settings.jsonl");
    let legacy_path = input_path.join("legacy.krn");
    write(&legacy_path, valid_kern());
    write(
        &output_path,
        &format!(
            "{}\n",
            legacy_success_jsonl_row_without_source_map(&legacy_path)
        ),
    );

    let mut args = render_settings_search_args(input_path, output_path.clone());
    args.resume = true;
    run_render_settings_search_with_in_process_worker(args);

    let rows = read_jsonl(&output_path);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["schema_version"], "1.7");
    assert_eq!(rows[0]["file"], legacy_path.display().to_string());
    assert_eq!(rows[0]["status"], "success");
    assert!(rows[0]["selected_render_source_lines"].is_array());
}

#[test]
fn render_settings_search_resume_preserves_original_candidate_indices() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let completed_path = input_path.join("a_completed.krn");
    let retry_path = input_path.join("b_retry.krn");
    let new_path = input_path.join("c_new.krn");
    write(&completed_path, valid_kern());
    write(&retry_path, valid_kern());
    write(&new_path, valid_kern());

    let files = collect_krn_files(&input_path).unwrap();
    let completed = HashSet::from([resume_file_key(&completed_path, &input_path)]);
    let mut work_items = render_settings_search_work_items(&files);
    work_items.retain(|item| !completed.contains(&resume_file_key(&item.path, &input_path)));

    assert_eq!(work_items.len(), 2);
    assert_eq!(work_items[0].path, retry_path);
    assert_eq!(work_items[0].candidate_index, 1);
    assert_eq!(work_items[0].candidate_count, 3);
    assert_eq!(work_items[1].path, new_path);
    assert_eq!(work_items[1].candidate_index, 2);
    assert_eq!(work_items[1].candidate_count, 3);
}

#[test]
fn render_settings_search_derives_stable_kern_augmentation_seeds() {
    let first = kern_render_augmentation_seed(RenderFitSearchSeed { index: 1, count: 3 });
    let same = kern_render_augmentation_seed(RenderFitSearchSeed { index: 1, count: 3 });
    let different = kern_render_augmentation_seed(RenderFitSearchSeed { index: 2, count: 3 });

    assert_eq!(first, same);
    assert_ne!(first, different);
}

#[test]
#[cfg(feature = "engraving")]
fn render_settings_search_resume_repairs_truncated_jsonl_before_appending() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("in");
    let output_path = temp.path().join("settings.jsonl");
    let completed_path = input_path.join("completed.krn");
    let new_path = input_path.join("new.krn");
    write(&completed_path, valid_kern());
    write(&new_path, valid_kern());
    write(
        &output_path,
        &format!(
            "{}\n{{\"schema_version\":\"1.0\"",
            jsonl_row(&completed_path, "success")
        ),
    );

    let mut args = render_settings_search_args(input_path, output_path.clone());
    args.resume = true;
    run_render_settings_search_with_in_process_worker(args);

    let rows = read_jsonl(&output_path);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["file"], completed_path.display().to_string());
    assert_eq!(rows[1]["file"], new_path.display().to_string());
    assert!(fs::read_to_string(output_path).unwrap().ends_with('\n'));
}

#[test]
fn selected_svg_path_mirrors_nested_input_path_with_svg_extension() {
    let input_root = Path::new("/dataset/5_concatenated");
    let output_svg_dir = Path::new("/dataset/7_svg_renderings");
    let input_path = input_root.join("composer/piece.krn");

    let path = selected_svg_path(&input_path, input_root, output_svg_dir).unwrap();

    assert_eq!(path, output_svg_dir.join("composer/piece.svg"));
}

#[test]
fn selected_svg_path_rejects_paths_outside_input_root() {
    let error = selected_svg_path(
        Path::new("/other/piece.krn"),
        Path::new("/dataset/5_concatenated"),
        Path::new("/dataset/7_svg_renderings"),
    )
    .unwrap_err();

    assert!(error.contains("is not inside input root"));
}

#[test]
fn successful_search_row_writes_svg_and_records_path_and_whitespace() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("5_concatenated/nested/piece.krn");
    let output_svg_dir = temp.path().join("7_svg_renderings");
    let selected_svg_path = selected_svg_path(
        &input_path,
        &temp.path().join("5_concatenated"),
        &output_svg_dir,
    )
    .unwrap();
    let whitespace = SvgPageWhitespace {
        content_box: SvgContentBox {
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 40.0,
        },
        whitespace: SvgWhitespace {
            left: 10.124,
            right: 60.125,
            top: 20.126,
            bottom: 140.129,
        },
    };
    let settings = VerovioRenderSettings {
        bar_line_width: 0.1234,
        unit: 9.8765,
        spacing_staff: 3.4567,
        spacing_system: 4.5678,
        measure_min_width: 12.3456,
        spacing_linear: 0.9876,
        spacing_non_linear: 1.2345,
        ..VerovioRenderSettings::medium()
    };
    let result = RenderFitResult {
        settings,
        page_count: 1,
        whitespace,
        score: 0.25,
        attempts: vec![RenderFitAttempt {
            settings,
            page_count: 1,
            whitespace: Some(whitespace),
            score: Some(0.25),
        }],
        selected_pages: vec![RenderedPageArtifact {
            page_number: 1,
            svg: r##"<svg viewBox="0 0 100 200">
    <g class="system">
        <g class="system content-bounding-box">
            <rect x="0" y="0" width="10" height="10" />
        </g>
        <path color="#c41414" d="M0 0 L10 0" />
        <g class="system bounding-box" />
    </g>
</svg>
"##
            .to_owned(),
            whitespace,
        }],
        selected_render_source_lines: vec![Some(1), None, Some(3)],
        quality_report: RenderFitQualityReport::default(),
    };

    let row = row_from_successful_search(
        &input_path,
        FileStats {
            line_count: 6,
            character_count: 42,
            initial_spine_count: Some(1),
            staff_count: Some(1),
            barline_count: Some(0),
        },
        result,
        &selected_svg_path,
        RenderFitSearchSeed { index: 0, count: 1 },
        RenderAugmentationReport {
            config: Some(RenderAugmentationConfig::default()),
            seed: 12,
            injected_fermata_count: 0,
            injected_arpeggio_count: 0,
            tuplet_visibility_count: 1,
            tuplet_visibility_spans: vec![TupletVisibilitySpanReport {
                staff_index: 0,
                spine_index: 0,
                start_source_line: 2,
                end_source_line: 4,
            }],
        },
    )
    .unwrap();

    assert!(row.search_success());
    assert_eq!(
        row.selected_svg_file.as_deref(),
        Some(selected_svg_path.display().to_string().as_str())
    );
    assert_eq!(
        row.selected_render_source_lines,
        Some(vec![Some(1), None, Some(3)])
    );
    let selected_whitespace = row.selected_whitespace.unwrap();
    assert_eq!(selected_whitespace.left, 10.12);
    assert_eq!(selected_whitespace.right, 60.13);
    assert_eq!(selected_whitespace.top, 20.13);
    assert_eq!(selected_whitespace.bottom, 140.13);
    assert_eq!(row.selected_system_count, Some(1));
    assert!(row.selected_svg_has_red_color);
    assert_eq!(
        row.render_augmentation
            .as_ref()
            .unwrap()
            .tuplet_visibility_count,
        1
    );
    let selected_content_box = row.selected_content_box.unwrap();
    assert_eq!(selected_content_box.x, 10.0);
    assert_eq!(selected_content_box.y, 20.0);
    assert_eq!(selected_content_box.width, 30.0);
    assert_eq!(selected_content_box.height, 40.0);
    let geometric_augmentation = row.geometric_augmentation.unwrap();
    assert_eq!(
        geometric_augmentation.status,
        GeometricAugmentationStatusPayload::Applied
    );
    assert!(geometric_augmentation.attempt_count >= 1);
    assert!(geometric_augmentation.transform.is_some());
    assert_eq!(row.accepted_attempt_count, 1);
    let selected_settings = row.selected_settings.unwrap();
    assert_eq!(selected_settings.font, settings.font.as_str());
    assert_eq!(selected_settings.bar_line_width, 0.12);
    assert_eq!(selected_settings.unit, 9.88);
    assert_eq!(selected_settings.spacing_staff, 3.46);
    assert_eq!(selected_settings.spacing_system, 4.57);
    assert_eq!(selected_settings.measure_min_width, 12.35);
    assert_eq!(selected_settings.spacing_linear, 0.99);
    assert_eq!(selected_settings.spacing_non_linear, 1.23);
    let selected_svg = fs::read_to_string(selected_svg_path).unwrap();
    assert!(selected_svg.starts_with(r#"<svg viewBox="0 0 100 200">"#));
    assert!(selected_svg.contains(r#"<g class="geometric-augmentation" transform=""#));
    assert!(selected_svg.contains(
        "\n    <g class=\"system\">\n        \n        <path color=\"#c41414\" d=\"M0 0 L10 0\" />\n        \n    </g>\n"
    ));
    assert!(!selected_svg.contains("bounding-box"));
    assert!(!selected_svg.contains("content-bounding-box"));
}

#[test]
fn render_settings_search_child_worker_attaches_stderr_to_row() {
    let temp = TempDir::new().unwrap();
    let file_path = temp.path().join("piece.krn");
    write(&file_path, valid_kern());
    let worker_path = temp.path().join("worker.sh");
    write(
        &worker_path,
        "#!/usr/bin/env sh\nprintf '%s\\n' '{\"schema_version\":\"1.0\",\"file\":\"child.krn\",\"line_count\":1,\"character_count\":12,\"initial_spine_count\":1,\"barline_count\":0,\"status\":\"search_error\",\"selected_settings\":null,\"selected_score\":null,\"selected_page_count\":null,\"selected_whitespace\":null,\"selected_svg_file\":null,\"attempt_count\":0,\"accepted_attempt_count\":0,\"parse_error\":null,\"search_error\":\"not implemented\",\"verovio_stderr\":null}'\nprintf 'args:%s\\n' \"$*\" >&2\nprintf 'hum warning\\n' >&2\n",
    );
    make_executable(&worker_path);

    let row = render_settings_search_file_via_child(
        &worker_path,
        &file_path,
        temp.path(),
        &temp.path().join("svg"),
        render_fit_search_options(
            SvgOverlapGateArg::Off,
            0.15,
            SvgOverlapFallbackArg::HorizontalRelief,
        ),
        RenderFitSearchSeed {
            index: 7,
            count: 11,
        },
    );

    assert_eq!(row.file, "child.krn");
    assert_eq!(row.character_count, 12);
    assert_eq!(row.barline_count, Some(0));
    let stderr = row.verovio_stderr.as_deref().unwrap();
    let removed_mode_arg = ["--search", "mode"].join("-");
    assert!(!stderr.contains(&removed_mode_arg));
    assert!(stderr.contains("--candidate-index 7"));
    assert!(stderr.contains("--candidate-count 11"));
    assert!(stderr.contains("hum warning\n"));
}

#[test]
fn render_settings_search_child_worker_parses_profile_wrapper() {
    let temp = TempDir::new().unwrap();
    let file_path = temp.path().join("piece.krn");
    write(&file_path, valid_kern());
    let worker_path = temp.path().join("worker.sh");
    write(
        &worker_path,
        r#"#!/usr/bin/env sh
printf '%s\n' '{"row":{"schema_version":"1.6","file":"child.krn","line_count":1,"character_count":12,"initial_spine_count":1,"staff_count":1,"barline_count":0,"status":"search_error","selected_settings":null,"selected_score":null,"selected_page_count":null,"selected_system_count":null,"selected_whitespace":null,"selected_content_box":null,"geometric_augmentation":null,"selected_svg_file":null,"selected_svg_has_red_color":false,"selected_render_source_lines":null,"svg_overlap_gate":null,"attempt_count":0,"accepted_attempt_count":0,"parse_error":null,"search_error":"not implemented","verovio_stderr":null},"profile":{"schema_version":"1.0","file":"child.krn","status":"search_error","line_count":1,"character_count":12,"initial_spine_count":1,"staff_count":1,"barline_count":0,"attempt_count":0,"accepted_attempt_count":0,"timings_ms":{"total":1.0,"read_input":0.1,"parse_document":0.2,"original_file_stats":0.0,"derive_svg_path":0.0,"augment_kern":0.0,"render_fit_search":0.0,"selected_svg_postprocess":0.0},"search_error":"not implemented"}}'
printf 'args:%s\n' "$*" >&2
"#,
    );
    make_executable(&worker_path);

    let output = render_settings_search_file_via_child_profiled(
        &worker_path,
        &file_path,
        temp.path(),
        &temp.path().join("svg"),
        render_fit_search_options(
            SvgOverlapGateArg::Off,
            0.15,
            SvgOverlapFallbackArg::HorizontalRelief,
        ),
        RenderFitSearchSeed {
            index: 7,
            count: 11,
        },
        true,
    );

    assert_eq!(output.row.file, "child.krn");
    let stderr = output.row.verovio_stderr.as_deref().unwrap();
    assert!(stderr.contains("--profile"));
    let profile = output.profile.unwrap();
    assert_eq!(profile.file, "child.krn");
    assert_eq!(profile.status, "search_error");
    assert!(profile.timings_ms.worker_process.unwrap() >= 0.0);
}

#[test]
fn render_settings_search_worker_protocol_failure_becomes_row() {
    let temp = TempDir::new().unwrap();
    let file_path = temp.path().join("piece.krn");
    write(&file_path, valid_kern());
    let output = ProcessCommand::new("sh")
        .arg("-c")
        .arg("printf 'not json\\n'; printf 'worker stderr\\n' >&2")
        .output()
        .unwrap();

    let row = render_settings_search_row_from_child_output(&file_path, output);

    assert_eq!(row.character_count, 0);
    assert!(row.barline_count.is_none());
    assert!(row
        .search_error
        .as_deref()
        .unwrap()
        .contains("invalid JSON"));
    assert_eq!(row.verovio_stderr.as_deref(), Some("worker stderr\n"));
}

#[test]
fn resume_drops_success_without_replay_config_but_keeps_current_receipt() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("in");
    let output = temp.path().join("settings.jsonl");
    let legacy = root.join("legacy.krn");
    let current = root.join("current.krn");
    let mut stale: Value = serde_json::from_str(&jsonl_row(&legacy, "success")).unwrap();
    stale["render_augmentation"]
        .as_object_mut()
        .unwrap()
        .remove("config");
    write(
        &output,
        &format!("{}\n{}\n", stale, jsonl_row(&current, "success")),
    );
    let completed = load_render_settings_search_resume(&output, &root, true).unwrap();
    assert!(!completed.contains("legacy.krn"));
    assert!(completed.contains("current.krn"));
    assert_eq!(read_jsonl(&output).len(), 1);
    assert_eq!(
        read_jsonl(&output)[0]["file"],
        current.display().to_string()
    );
}
