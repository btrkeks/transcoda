use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::{
    analysis::{
        augment_svg_geometry, count_svg_systems, find_seeded_render_settings_with_options,
        find_seeded_render_settings_with_options_and_timing, strip_svg_bounding_boxes,
        GeometricAugmentationConfig, GeometricAugmentationResult, GeometricAugmentationSeed,
        RenderFitConfig, RenderFitResult, RenderFitSearchOptions, RenderFitSearchSeed,
        RenderFitTimingReport, SvgContentBox, VerovioRenderSettings,
    },
    augment_for_render_with_report, KernDocument, KernLine, RenderAugmentationConfig,
    RenderAugmentationReport, RenderAugmentationSeed,
};

use super::output::selected_svg_path;
use super::profile::{
    round_millis, ProfiledRenderSettingsSearchRow, RenderSettingsSearchPhaseTimings,
    RenderSettingsSearchProfileRow,
};
use super::row::{
    blank_file_stats, FileStats, GeometricAugmentationPayload, RenderSettingsSearchRow,
    SvgOverlapGatePayload,
};

pub(super) fn render_settings_search_file(
    path: &Path,
    input_root: &Path,
    output_svg_dir: &Path,
    search_options: RenderFitSearchOptions,
    seed: RenderFitSearchSeed,
) -> RenderSettingsSearchRow {
    render_settings_search_file_profiled(
        path,
        input_root,
        output_svg_dir,
        search_options,
        seed,
        false,
    )
    .row
}

pub(super) fn render_settings_search_file_profiled(
    path: &Path,
    input_root: &Path,
    output_svg_dir: &Path,
    search_options: RenderFitSearchOptions,
    seed: RenderFitSearchSeed,
    collect_profile: bool,
) -> ProfiledRenderSettingsSearchRow {
    let total_start = Instant::now();
    let mut timings = RenderSettingsSearchPhaseTimings::default();
    let empty_gate = SvgOverlapGatePayload::from_options(search_options);
    let read_start = Instant::now();
    let input = match fs::read_to_string(path) {
        Ok(input) => input,
        Err(error) => {
            timings.read_input = elapsed_millis(read_start);
            let row =
                RenderSettingsSearchRow::parse_error(path, blank_file_stats(), error.to_string())
                    .with_svg_overlap_gate(empty_gate.clone());
            return profiled_row(row, collect_profile, timings, total_start, None);
        }
    };
    timings.read_input = elapsed_millis(read_start);

    let stats_start = Instant::now();
    let line_count = input.lines().count();
    let character_count = input.chars().count();
    timings.original_file_stats += elapsed_millis(stats_start);

    let parse_start = Instant::now();
    let document = match KernDocument::parse(&input) {
        Ok(document) => document,
        Err(error) => {
            timings.parse_document = elapsed_millis(parse_start);
            let row = RenderSettingsSearchRow::parse_error(
                path,
                FileStats {
                    line_count,
                    character_count,
                    initial_spine_count: None,
                    staff_count: None,
                    barline_count: None,
                },
                error.to_string(),
            )
            .with_svg_overlap_gate(empty_gate.clone());
            return profiled_row(row, collect_profile, timings, total_start, None);
        }
    };
    timings.parse_document = elapsed_millis(parse_start);

    let stats_start = Instant::now();
    let file_stats = original_file_stats(&document, line_count, character_count);
    timings.original_file_stats += elapsed_millis(stats_start);
    let svg_path_start = Instant::now();
    let selected_svg_path = match selected_svg_path(path, input_root, output_svg_dir) {
        Ok(path) => path,
        Err(error) => {
            timings.derive_svg_path = elapsed_millis(svg_path_start);
            let row = RenderSettingsSearchRow::search_error(path, file_stats, error)
                .with_svg_overlap_gate(empty_gate.clone());
            return profiled_row(row, collect_profile, timings, total_start, None);
        }
    };
    timings.derive_svg_path = elapsed_millis(svg_path_start);

    let config = match kern_render_augmentation_config_from_env() {
        Ok(config) => config,
        Err(error) => {
            return profiled_row(
                RenderSettingsSearchRow::search_error(path, file_stats, error),
                collect_profile,
                timings,
                total_start,
                None,
            )
        }
    };
    let augment_start = Instant::now();
    let (augmented_document, render_augmentation) = match augment_for_render_with_report(
        &document,
        &config,
        kern_render_augmentation_seed(seed),
    ) {
        Ok(document) => document,
        Err(error) => {
            timings.augment_kern = elapsed_millis(augment_start);
            let row = RenderSettingsSearchRow::search_error(
                path,
                file_stats,
                format!("failed to augment kern document for render: {error}"),
            )
            .with_svg_overlap_gate(empty_gate.clone());
            return profiled_row(row, collect_profile, timings, total_start, None);
        }
    };
    timings.augment_kern = elapsed_millis(augment_start);

    let mut render_fit_timing = RenderFitTimingReport::default();
    let search_start = Instant::now();
    let search_result = if collect_profile {
        find_seeded_render_settings_with_options_and_timing(
            &augmented_document,
            VerovioRenderSettings::medium(),
            RenderFitConfig::single_page_balanced_for_document(&augmented_document),
            seed,
            search_options,
            &mut render_fit_timing,
        )
    } else {
        find_seeded_render_settings_with_options(
            &augmented_document,
            VerovioRenderSettings::medium(),
            RenderFitConfig::single_page_balanced_for_document(&augmented_document),
            seed,
            search_options,
        )
    };
    timings.render_fit_search = elapsed_millis(search_start);
    let render_fit_profile = collect_profile.then_some(render_fit_timing);

    match search_result {
        Ok(result) => {
            let gate_payload = SvgOverlapGatePayload::from_report(&result.quality_report);
            let postprocess_start = Instant::now();
            let row = row_from_successful_search(
                path,
                file_stats,
                result,
                &selected_svg_path,
                seed,
                render_augmentation,
            )
            .unwrap_or_else(|error| {
                RenderSettingsSearchRow::search_error(path, file_stats, error)
                    .with_svg_overlap_gate(gate_payload)
            });
            timings.selected_svg_postprocess = elapsed_millis(postprocess_start);
            profiled_row(
                row,
                collect_profile,
                timings,
                total_start,
                render_fit_profile,
            )
        }
        Err(error) => {
            let row = RenderSettingsSearchRow::search_error(path, file_stats, error.to_string())
                .with_svg_overlap_gate(SvgOverlapGatePayload::from_report(&error.quality_report));
            profiled_row(
                row,
                collect_profile,
                timings,
                total_start,
                render_fit_profile,
            )
        }
    }
}

fn profiled_row(
    row: RenderSettingsSearchRow,
    collect_profile: bool,
    mut timings: RenderSettingsSearchPhaseTimings,
    total_start: Instant,
    render_fit: Option<RenderFitTimingReport>,
) -> ProfiledRenderSettingsSearchRow {
    if !collect_profile {
        return ProfiledRenderSettingsSearchRow { row, profile: None };
    }

    timings.total = elapsed_millis(total_start);
    let profile = RenderSettingsSearchProfileRow::from_row(&row, timings, render_fit);
    ProfiledRenderSettingsSearchRow {
        row,
        profile: Some(profile),
    }
}

fn elapsed_millis(start: Instant) -> f64 {
    round_millis(start.elapsed().as_secs_f64() * 1000.0)
}

fn original_file_stats(
    document: &KernDocument,
    line_count: usize,
    character_count: usize,
) -> FileStats {
    FileStats {
        line_count,
        character_count,
        initial_spine_count: Some(document.layout.initial_spine_count),
        staff_count: Some(document.staff_count()),
        barline_count: Some(
            document
                .lines()
                .iter()
                .filter(|line| matches!(line, KernLine::Barline(_)))
                .count(),
        ),
    }
}

pub(super) fn kern_render_augmentation_config() -> RenderAugmentationConfig {
    RenderAugmentationConfig::default()
}

// A single setting switches the labelled injections off: `0,0`.
fn kern_render_augmentation_config_from_env() -> Result<RenderAugmentationConfig, String> {
    let mut config = kern_render_augmentation_config();
    if let Some(value) = std::env::var_os("KERN_LABELLED_MARK_PAGE_RATES") {
        let value = value
            .to_str()
            .ok_or("KERN_LABELLED_MARK_PAGE_RATES must be UTF-8")?;
        let rates = value
            .split(',')
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "KERN_LABELLED_MARK_PAGE_RATES must be fermata,arpeggio probabilities")?;
        if rates.len() != 2
            || rates
                .iter()
                .any(|rate| !rate.is_finite() || !(0.0..=1.0).contains(rate))
        {
            return Err(
                "KERN_LABELLED_MARK_PAGE_RATES must contain two probabilities in [0,1]".into(),
            );
        }
        config.fermata_page_probability = rates[0];
        config.arpeggio_page_probability = rates[1];
    }
    Ok(config)
}

pub(super) fn kern_render_augmentation_seed(seed: RenderFitSearchSeed) -> RenderAugmentationSeed {
    let index = seed.index as u64;
    let count = seed.count as u64;
    RenderAugmentationSeed(
        0x6a09_e667_f3bc_c909 ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ count.rotate_left(32),
    )
}

pub(super) fn row_from_successful_search(
    path: &Path,
    stats: FileStats,
    result: RenderFitResult,
    selected_svg_path: &Path,
    seed: RenderFitSearchSeed,
    render_augmentation: RenderAugmentationReport,
) -> Result<RenderSettingsSearchRow, String> {
    let [selected_page] = result.selected_pages.as_slice() else {
        return Err(format!(
            "render fit search returned {} selected pages, expected 1",
            result.selected_pages.len()
        ));
    };

    if let Some(parent) = selected_svg_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create SVG output directory: {error}"))?;
    }
    let selected_system_count = count_svg_systems(&selected_page.svg)
        .map_err(|error| format!("failed to count selected SVG systems: {error}"))?;
    let svg_geometric_result = apply_svg_geometric_augmentation(
        &selected_page.svg,
        selected_page.whitespace.content_box,
        seed,
    )?;
    let stripped_svg = strip_svg_bounding_boxes(&svg_geometric_result.svg)
        .map_err(|error| format!("failed to strip selected SVG bounding boxes: {error}"))?;
    let selected_svg_has_red_color = stripped_svg.contains("#c41414");
    fs::write(selected_svg_path, stripped_svg)
        .map_err(|error| format!("failed to write selected SVG: {error}"))?;

    let original_content_box = svg_geometric_result.original_content_box;
    let geometric_augmentation = GeometricAugmentationPayload {
        status: svg_geometric_result.status.into(),
        attempt_count: svg_geometric_result.attempt_count,
        transform: svg_geometric_result.transform.map(Into::into),
        transformed_content_box: svg_geometric_result.transformed_content_box.into(),
    };

    Ok(RenderSettingsSearchRow::success(
        path,
        stats,
        result,
        selected_system_count,
        original_content_box,
        geometric_augmentation,
        selected_svg_path,
        selected_svg_has_red_color,
        render_augmentation,
    ))
}

fn apply_svg_geometric_augmentation(
    svg: &str,
    content_box: SvgContentBox,
    seed: RenderFitSearchSeed,
) -> Result<GeometricAugmentationResult, String> {
    augment_svg_geometry(
        svg,
        content_box,
        GeometricAugmentationConfig::conservative_page_geometry(),
        GeometricAugmentationSeed {
            file_index: seed.index,
            file_count: seed.count,
            variant_index: 0,
        },
    )
    .map_err(|error| format!("failed to apply selected SVG geometric augmentation: {error}"))
}
