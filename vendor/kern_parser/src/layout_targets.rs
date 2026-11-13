//! Layout-derived OMR training target generation.

mod meter_visibility;
mod staff_visibility;
pub use meter_visibility::project_meter_visibility;
pub use staff_visibility::project_staff_visibility;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, Write},
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
};

use crate::dataset_manifest::{update_dataset_manifest, DatasetComponentUpdate};
use roxmltree::Node;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use walkdir::WalkDir;

use crate::{serialize::SerializePreset, KernDocument, KernLine, SerializeError, SystemBreak};

const SYSTEM_BREAK_GROUP: &str = "original";
const REPORT_SCHEMA_VERSION: &str = "layout_targets.v1";

#[derive(Debug, Clone, PartialEq)]
pub struct RenderedLayout {
    pub system_count: usize,
    pub systems: Vec<RenderedSystem>,
    pub measures: Vec<RenderedMeasure>,
    pub barlines: Vec<RenderedBarline>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderedSystem {
    pub index: usize,
    pub first_measure_line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderedMeasure {
    pub rendered_line: usize,
    pub system_index: usize,
    pub bbox: Option<RenderedBBox>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderedBarline {
    pub rendered_line: usize,
    pub system_index: usize,
    pub bbox: Option<RenderedBBox>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderedBBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemBreakPlan {
    pub rendered_start_lines: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AnnotationSummary {
    pub requested_break_count: usize,
    pub inserted_break_count: usize,
    pub existing_break_count: usize,
    pub insertion_source_lines: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct LayoutTargetOptions {
    pub input_dir: PathBuf,
    pub render_settings_jsonl: PathBuf,
    pub output_dir: PathBuf,
    pub report_jsonl: PathBuf,
    pub debug_output_dir: Option<PathBuf>,
    pub skip_render_failures: bool,
    pub skip_target_failures: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LayoutTargetReportRow {
    pub schema_version: String,
    pub file: String,
    pub output_file: String,
    pub selected_svg_file: String,
    pub selected_system_count: usize,
    pub parsed_system_count: usize,
    pub rendered_break_lines: Vec<usize>,
    pub source_break_lines: Vec<usize>,
    pub requested_break_count: usize,
    pub inserted_break_count: usize,
    pub existing_break_count: usize,
    pub status: String,
}

#[derive(Debug, Error)]
pub enum LayoutTargetError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse render settings JSONL line {line}: {source}")]
    RenderSettingsJson {
        line: usize,
        source: serde_json::Error,
    },
    #[error("render settings row line {line} is missing required field `{field}`")]
    MissingRenderSettingsField { line: usize, field: &'static str },
    #[error("duplicate successful render settings row for {file}")]
    DuplicateRenderSettingsRow { file: PathBuf },
    #[error("missing successful render settings row for {file}")]
    MissingRenderSettingsRow { file: PathBuf },
    #[error("failed to parse SVG: {message}")]
    MalformedSvg { message: String },
    #[error("SVG contains a system without any measure-L# measure")]
    MissingSystemMeasure,
    #[error("SVG measure node has malformed id `{id}`")]
    MalformedMeasureId { id: String },
    #[error("SVG system count mismatch for {file}: row selected {selected}, parsed {parsed}")]
    SystemCountMismatch {
        file: PathBuf,
        selected: usize,
        parsed: usize,
    },
    #[error("rendered line {rendered_line} does not map to a source line")]
    MissingSourceForRenderedLine { rendered_line: usize },
    #[error("rendered line {rendered_line} is outside rendered source map")]
    RenderedLineOutOfRange { rendered_line: usize },
    #[error("duplicate requested system break at source line {line}")]
    DuplicateBreakLine { line: usize },
    #[error("source line {line} was not found in the parsed document")]
    SourceLineNotFound { line: usize },
    #[error("source line {line} is not a barline/measure-start line")]
    SourceLineNotBarline { line: usize },
    #[error("source line {line} already has a conflicting system break")]
    ConflictingSystemBreak { line: usize },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        source: crate::ParseError,
    },
    #[error("meter visibility projection failed for {path}: {message}")]
    MeterProjection { path: PathBuf, message: String },
    #[error("failed to serialize {path}: {source}")]
    Serialize {
        path: PathBuf,
        source: SerializeError,
    },
    #[error("failed to augment {path} for render: {message}")]
    Augment { path: PathBuf, message: String },
    #[error("injected {mark} glyph count mismatch for {path}: SVG {glyphs}, label {labelled}")]
    InjectedMarkMismatch {
        path: PathBuf,
        mark: &'static str,
        glyphs: usize,
        labelled: usize,
    },
    #[error("{path} is not below input directory {input_dir}")]
    PathOutsideInputDir { path: PathBuf, input_dir: PathBuf },
    #[error("target generation panicked for {path}: {message}")]
    Panic { path: PathBuf, message: String },
}

#[derive(Debug, Deserialize)]
struct RenderSettingsRow {
    file: String,
    status: String,
    #[serde(default)]
    selected_svg_file: Option<String>,
    #[serde(default)]
    selected_system_count: Option<usize>,
    #[serde(default)]
    selected_render_source_lines: Option<Vec<Option<usize>>>,
    #[serde(default)]
    render_augmentation: Option<crate::RenderAugmentationReport>,
}

#[derive(Debug, Clone)]
struct SuccessfulRenderSettings {
    selected_svg_file: PathBuf,
    selected_system_count: Option<usize>,
    selected_render_source_lines: Vec<Option<usize>>,
    render_augmentation: Option<crate::RenderAugmentationReport>,
}

#[derive(Debug, Clone)]
enum RenderSettingsLookup {
    Success(SuccessfulRenderSettings),
    Failure,
}

pub fn parse_rendered_layout(svg: &str) -> Result<RenderedLayout, LayoutTargetError> {
    let document =
        roxmltree::Document::parse(svg).map_err(|error| LayoutTargetError::MalformedSvg {
            message: error.to_string(),
        })?;

    let mut systems = Vec::new();
    let mut measures = Vec::new();
    let mut barlines = Vec::new();

    for (system_index, system) in document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "g")
        .filter(|node| {
            has_class(*node, "system")
                && !has_class(*node, "bounding-box")
                && !has_class(*node, "content-bounding-box")
        })
        .enumerate()
    {
        let system_measures = system
            .descendants()
            .filter(|node| node.is_element() && node.tag_name().name() == "g")
            .filter(|node| has_class(*node, "measure"))
            .map(|node| {
                let id = node.attribute("id").unwrap_or_default();
                let rendered_line = parse_measure_id(id)?;
                Ok(RenderedMeasure {
                    rendered_line,
                    system_index,
                    bbox: None,
                })
            })
            .collect::<Result<Vec<_>, LayoutTargetError>>()?;

        let first_measure_line = system_measures
            .first()
            .map(|measure| measure.rendered_line)
            .ok_or(LayoutTargetError::MissingSystemMeasure)?;

        systems.push(RenderedSystem {
            index: system_index,
            first_measure_line,
        });
        measures.extend(system_measures);

        for barline in system
            .descendants()
            .filter(|node| node.is_element() && node.tag_name().name() == "g")
            .filter(|node| has_class(*node, "barLine"))
        {
            if let Some(id) = barline.attribute("id") {
                if let Some(rendered_line) = parse_line_id(id) {
                    barlines.push(RenderedBarline {
                        rendered_line,
                        system_index,
                        bbox: None,
                    });
                }
            }
        }
    }

    Ok(RenderedLayout {
        system_count: systems.len(),
        systems,
        measures,
        barlines,
    })
}

pub fn system_break_plan_from_layout(layout: &RenderedLayout) -> SystemBreakPlan {
    SystemBreakPlan {
        rendered_start_lines: layout
            .systems
            .iter()
            .skip(1)
            .map(|system| system.first_measure_line)
            .collect(),
    }
}

pub fn source_break_lines_from_layout(
    layout: &RenderedLayout,
    document: &KernDocument,
) -> Result<Vec<usize>, LayoutTargetError> {
    let source_starts = source_measure_start_lines(document);
    let has_leading_start = source_starts
        .first()
        .is_some_and(|(_, source_line)| *source_line < first_data_source_line(document));

    layout
        .systems
        .iter()
        .skip(1)
        .map(|system| {
            let ordinal = layout
                .measures
                .iter()
                .position(|measure| {
                    measure.system_index == system.index
                        && measure.rendered_line == system.first_measure_line
                })
                .ok_or(LayoutTargetError::SourceLineNotFound {
                    line: system.first_measure_line,
                })?;

            let source_start_index = if has_leading_start {
                ordinal
            } else {
                ordinal
                    .checked_sub(1)
                    .ok_or(LayoutTargetError::SourceLineNotBarline {
                        line: system.first_measure_line,
                    })?
            };

            source_starts
                .get(source_start_index)
                .map(|(_, source_line)| *source_line)
                .ok_or(LayoutTargetError::SourceLineNotFound {
                    line: system.first_measure_line,
                })
        })
        .collect()
}

fn source_measure_start_lines(document: &KernDocument) -> Vec<(usize, usize)> {
    document
        .lines()
        .iter()
        .enumerate()
        .filter(|(index, line)| {
            matches!(line, KernLine::Barline(_)) && barline_starts_measure(document, *index)
        })
        .map(|(index, _)| (index, document.source(index).unwrap_or(index + 1)))
        .collect()
}

fn barline_starts_measure(document: &KernDocument, barline_index: usize) -> bool {
    document.lines()[barline_index + 1..]
        .iter()
        .take_while(|line| !matches!(line, KernLine::Barline(_)))
        .any(|line| matches!(line, KernLine::Data(_)))
}

fn first_data_source_line(document: &KernDocument) -> usize {
    document
        .lines()
        .iter()
        .enumerate()
        .find(|(_, line)| matches!(line, KernLine::Data(_)))
        .map(|(index, _)| document.source(index).unwrap_or(index + 1))
        .unwrap_or(usize::MAX)
}

pub fn resolve_break_source_lines(
    plan: &SystemBreakPlan,
    rendered_sources: &[Option<usize>],
) -> Result<Vec<usize>, LayoutTargetError> {
    plan.rendered_start_lines
        .iter()
        .map(|rendered_line| resolve_rendered_source_line(*rendered_line, rendered_sources))
        .collect()
}

fn resolve_rendered_source_line(
    rendered_line: usize,
    rendered_sources: &[Option<usize>],
) -> Result<usize, LayoutTargetError> {
    let index = rendered_line
        .checked_sub(1)
        .ok_or(LayoutTargetError::RenderedLineOutOfRange { rendered_line })?;
    if index >= rendered_sources.len() {
        return Err(LayoutTargetError::RenderedLineOutOfRange { rendered_line });
    }
    if let Some(source_line) = rendered_sources[index] {
        return Ok(source_line);
    }

    rendered_sources[index + 1..]
        .iter()
        .chain(rendered_sources[..index].iter().rev())
        .find_map(|source| *source)
        .ok_or(LayoutTargetError::MissingSourceForRenderedLine { rendered_line })
}

/// Source breaks are requests; only the selected SVG establishes target breaks.
pub fn clear_system_breaks(document: &mut KernDocument) {
    for line in document.lines_mut() {
        if let KernLine::Barline(barline) = line {
            barline.system_break_before = None;
        }
    }
}

pub fn annotate_system_breaks(
    document: &mut KernDocument,
    source_lines: &[usize],
) -> Result<AnnotationSummary, LayoutTargetError> {
    let mut seen_insertions = BTreeSet::new();
    let mut targets: Vec<(usize, usize)> = Vec::new();
    let mut insertion_source_lines = Vec::new();
    let mut existing_break_count = 0;
    for source_line in source_lines {
        let (index, insertion_source_line) =
            resolve_system_break_insertion(document, *source_line)?;
        if !seen_insertions.insert(insertion_source_line) {
            return Err(LayoutTargetError::DuplicateBreakLine {
                line: insertion_source_line,
            });
        }
        insertion_source_lines.push(insertion_source_line);

        if let KernLine::Barline(barline) = &mut document.lines_mut()[index] {
            if let Some(system_break) = &mut barline.system_break_before {
                system_break.group = Some(SYSTEM_BREAK_GROUP.to_owned());
                existing_break_count += 1;
                continue;
            }
        }

        targets.push((index, insertion_source_line));
    }

    targets.sort_by_key(|(index, _)| *index);
    let inserted_break_count = targets.len();
    for (index, source_line) in targets {
        if let KernLine::Barline(barline) = &mut document.lines_mut()[index] {
            barline.system_break_before = Some(SystemBreak {
                group: Some(SYSTEM_BREAK_GROUP.to_owned()),
                source_line: Some(source_line),
            });
        }
    }

    Ok(AnnotationSummary {
        requested_break_count: source_lines.len(),
        inserted_break_count,
        existing_break_count,
        insertion_source_lines,
    })
}

fn resolve_system_break_insertion(
    document: &KernDocument,
    source_line: usize,
) -> Result<(usize, usize), LayoutTargetError> {
    let index = document
        .lines()
        .iter()
        .enumerate()
        .find(|(index, _)| document.source(*index) == Some(source_line))
        .map(|(index, _)| index)
        .ok_or(LayoutTargetError::SourceLineNotFound { line: source_line })?;

    match &document.lines()[index] {
        KernLine::Barline(_) => Ok((index, source_line)),
        KernLine::Data(_) => {
            for barline_index in (0..index).rev() {
                if matches!(document.lines()[barline_index], KernLine::Barline(_)) {
                    let insertion_source_line =
                        document.source(barline_index).unwrap_or(barline_index + 1);
                    return Ok((barline_index, insertion_source_line));
                }
            }
            for barline_index in index + 1..document.lines().len() {
                if matches!(document.lines()[barline_index], KernLine::Barline(_)) {
                    let insertion_source_line =
                        document.source(barline_index).unwrap_or(barline_index + 1);
                    return Ok((barline_index, insertion_source_line));
                }
            }
            Err(LayoutTargetError::SourceLineNotBarline { line: source_line })
        }
        _ => {
            for barline_index in (0..index).rev() {
                if matches!(document.lines()[barline_index], KernLine::Barline(_)) {
                    let insertion_source_line =
                        document.source(barline_index).unwrap_or(barline_index + 1);
                    return Ok((barline_index, insertion_source_line));
                }
            }
            Err(LayoutTargetError::SourceLineNotBarline { line: source_line })
        }
    }
}

pub fn generate_layout_targets(
    options: &LayoutTargetOptions,
) -> Result<Vec<LayoutTargetReportRow>, LayoutTargetError> {
    let files = collect_krn_files(&options.input_dir)?;
    let settings = load_render_settings(&options.render_settings_jsonl, &options.input_dir)?;
    fs::create_dir_all(&options.output_dir).map_err(|source| LayoutTargetError::Write {
        path: options.output_dir.clone(),
        source,
    })?;
    if let Some(debug_dir) = &options.debug_output_dir {
        fs::create_dir_all(debug_dir).map_err(|source| LayoutTargetError::Write {
            path: debug_dir.clone(),
            source,
        })?;
    }
    if let Some(parent) = options.report_jsonl.parent() {
        fs::create_dir_all(parent).map_err(|source| LayoutTargetError::Write {
            path: parent.to_owned(),
            source,
        })?;
    }

    let mut report = Vec::new();
    let mut skipped_render_failure_count = 0usize;
    let mut skipped_target_failure_count = 0usize;
    let mut report_file =
        fs::File::create(&options.report_jsonl).map_err(|source| LayoutTargetError::Write {
            path: options.report_jsonl.clone(),
            source,
        })?;

    for path in files.iter() {
        let relative = relative_to_input(path, &options.input_dir)?;
        let settings = match settings.get(&relative) {
            Some(RenderSettingsLookup::Success(settings)) => settings,
            Some(RenderSettingsLookup::Failure) if options.skip_render_failures => {
                skipped_render_failure_count += 1;
                continue;
            }
            _ => {
                return Err(LayoutTargetError::MissingRenderSettingsRow {
                    file: relative.clone(),
                });
            }
        };
        let row =
            match generate_one_layout_target_catching_panics(path, &relative, settings, options) {
                Ok(row) => row,
                Err(error) if options.skip_target_failures => {
                    skipped_target_failure_count += 1;
                    eprintln!("Skipping layout target {}: {}", relative.display(), error);
                    continue;
                }
                Err(error) => return Err(error),
            };
        serde_json::to_writer(&mut report_file, &row).map_err(|source| {
            LayoutTargetError::Write {
                path: options.report_jsonl.clone(),
                source: std::io::Error::other(source),
            }
        })?;
        writeln!(&mut report_file).map_err(|source| LayoutTargetError::Write {
            path: options.report_jsonl.clone(),
            source,
        })?;
        report.push(row);
    }

    update_layout_targets_manifest(
        options,
        report.len(),
        skipped_render_failure_count,
        skipped_target_failure_count,
    )?;

    Ok(report)
}

fn update_layout_targets_manifest(
    options: &LayoutTargetOptions,
    row_count: usize,
    skipped_render_failure_count: usize,
    skipped_target_failure_count: usize,
) -> Result<(), LayoutTargetError> {
    let Some(dataset_root) = options.output_dir.parent() else {
        return Ok(());
    };
    let summary = BTreeMap::from([
        ("row_count".to_owned(), row_count),
        (
            "skipped_render_failure_count".to_owned(),
            skipped_render_failure_count,
        ),
        (
            "skipped_target_failure_count".to_owned(),
            skipped_target_failure_count,
        ),
    ]);
    update_dataset_manifest(
        dataset_root,
        DatasetComponentUpdate {
            name: "layout_targets".to_owned(),
            producer: "kern_parser generate-layout-targets".to_owned(),
            paths: vec![options.output_dir.clone(), options.report_jsonl.clone()],
            summary,
        },
    )
    .map_err(|source| LayoutTargetError::Write {
        path: dataset_root.join(crate::dataset_manifest::DATASET_MANIFEST_FILE_NAME),
        source,
    })?;
    Ok(())
}

fn generate_one_layout_target_catching_panics(
    path: &Path,
    relative: &Path,
    settings: &SuccessfulRenderSettings,
    options: &LayoutTargetOptions,
) -> Result<LayoutTargetReportRow, LayoutTargetError> {
    match panic::catch_unwind(AssertUnwindSafe(|| {
        generate_one_layout_target(path, relative, settings, options)
    })) {
        Ok(result) => result,
        Err(payload) => Err(LayoutTargetError::Panic {
            path: path.to_owned(),
            message: panic_message(payload),
        }),
    }
}

fn generate_one_layout_target(
    path: &Path,
    relative: &Path,
    settings: &SuccessfulRenderSettings,
    options: &LayoutTargetOptions,
) -> Result<LayoutTargetReportRow, LayoutTargetError> {
    let input = fs::read_to_string(path).map_err(|source| LayoutTargetError::Read {
        path: path.to_owned(),
        source,
    })?;
    let mut document = KernDocument::parse(&input).map_err(|source| LayoutTargetError::Parse {
        path: path.to_owned(),
        source,
    })?;
    let mut injected_counts = [0, 0];
    if let Some(report) = &settings.render_augmentation {
        let config = report.config.as_ref().ok_or_else(|| LayoutTargetError::Augment {
            path: path.to_owned(), message: "legacy render receipt has no augmentation recipe; rerender before generating targets".into(),
        })?;
        let (augmented, replay) = crate::augment_for_render_with_report(
            &document,
            config,
            crate::RenderAugmentationSeed(report.seed),
        )
        .map_err(|error| LayoutTargetError::Augment {
            path: path.to_owned(),
            message: error.to_string(),
        })?;
        injected_counts = [
            replay.injected_fermata_count,
            replay.injected_arpeggio_count,
        ];
        document = augmented;
    }
    let svg = fs::read_to_string(&settings.selected_svg_file).map_err(|source| {
        LayoutTargetError::Read {
            path: settings.selected_svg_file.clone(),
            source,
        }
    })?;
    let layout = parse_rendered_layout(&svg)?;
    if let Some(selected_system_count) = settings.selected_system_count {
        if layout.system_count != selected_system_count {
            return Err(LayoutTargetError::SystemCountMismatch {
                file: relative.to_owned(),
                selected: selected_system_count,
                parsed: layout.system_count,
            });
        }
    }

    let selected_system_count = settings
        .selected_system_count
        .unwrap_or(layout.system_count);

    let plan = system_break_plan_from_layout(&layout);
    let source_break_lines =
        resolve_break_source_lines(&plan, &settings.selected_render_source_lines)?;
    project_meter_visibility(&mut document, &svg, &settings.selected_render_source_lines).map_err(
        |error| LayoutTargetError::MeterProjection {
            path: path.to_owned(),
            message: error.to_string(),
        },
    )?;
    project_staff_visibility(&mut document, &svg, &settings.selected_render_source_lines)?;
    clear_system_breaks(&mut document);
    let annotation = annotate_system_breaks(&mut document, &source_break_lines)?;
    let target = SerializePreset::TranscodaV2Training
        .try_serialize_document(&document)
        .map_err(|source| LayoutTargetError::Serialize {
            path: path.to_owned(),
            source,
        })?;
    validate_injected_mark_counts(path, &svg, &target, injected_counts)?;
    let output_path = options.output_dir.join(relative);
    write_text(&output_path, &target)?;

    if let Some(debug_dir) = &options.debug_output_dir {
        let debug_path = debug_dir.join(relative);
        let debug = SerializePreset::HumdrumKern
            .try_serialize_document(&document)
            .map_err(|source| LayoutTargetError::Serialize {
                path: path.to_owned(),
                source,
            })?;
        write_text(&debug_path, &debug)?;
    }

    Ok(LayoutTargetReportRow {
        schema_version: REPORT_SCHEMA_VERSION.to_owned(),
        file: relative.display().to_string(),
        output_file: output_path.display().to_string(),
        selected_svg_file: settings.selected_svg_file.display().to_string(),
        selected_system_count,
        parsed_system_count: layout.system_count,
        rendered_break_lines: plan.rendered_start_lines,
        source_break_lines: annotation.insertion_source_lines,
        requested_break_count: annotation.requested_break_count,
        inserted_break_count: annotation.inserted_break_count,
        existing_break_count: annotation.existing_break_count,
        status: "success".to_owned(),
    })
}

fn validate_injected_mark_counts(
    path: &Path,
    svg: &str,
    target: &str,
    injected_counts: [usize; 2],
) -> Result<(), LayoutTargetError> {
    if injected_counts == [0, 0] {
        return Ok(());
    }
    let xml = roxmltree::Document::parse(svg).map_err(|error| LayoutTargetError::MalformedSvg {
        message: error.to_string(),
    })?;
    for ((class, symbol), injected) in [("fermata", ';'), ("arpeg", ':')]
        .into_iter()
        .zip(injected_counts)
    {
        if injected == 0 {
            continue;
        }
        let glyphs = xml
            .descendants()
            .filter(|&node| {
                has_class(node, class)
                    && !node.ancestors().any(|ancestor| {
                        ancestor.attribute("visibility") == Some("hidden")
                            || ancestor.attribute("display") == Some("none")
                            || ancestor.attribute("style").is_some_and(|style| {
                                style.replace(' ', "").contains("display:none")
                            })
                    })
                    && node
                        .descendants()
                        .any(|child| child.is_element() && child.tag_name().name() == "use")
            })
            .count();
        let labelled = target
            .lines()
            .filter(|line| {
                !line.starts_with(['*', '!']) && (symbol != ':' || !line.starts_with('='))
            })
            .map(|line| line.matches(symbol).count())
            .sum();
        if glyphs != labelled {
            return Err(LayoutTargetError::InjectedMarkMismatch {
                path: path.to_owned(),
                mark: class,
                glyphs,
                labelled,
            });
        }
    }
    Ok(())
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_owned();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "layout target generation panicked".to_owned()
}

fn load_render_settings(
    path: &Path,
    input_dir: &Path,
) -> Result<BTreeMap<PathBuf, RenderSettingsLookup>, LayoutTargetError> {
    let file = fs::File::open(path).map_err(|source| LayoutTargetError::Read {
        path: path.to_owned(),
        source,
    })?;
    let mut rows = BTreeMap::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|source| LayoutTargetError::Read {
            path: path.to_owned(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str::<Value>(&line).map_err(|source| {
            LayoutTargetError::RenderSettingsJson {
                line: line_number,
                source,
            }
        })?;
        let row = serde_json::from_value::<RenderSettingsRow>(value).map_err(|source| {
            LayoutTargetError::RenderSettingsJson {
                line: line_number,
                source,
            }
        })?;
        let file = normalize_settings_file_key(&row.file, input_dir);
        if row.status != "success" {
            rows.entry(file).or_insert(RenderSettingsLookup::Failure);
            continue;
        }
        let selected_svg_file =
            row.selected_svg_file
                .ok_or(LayoutTargetError::MissingRenderSettingsField {
                    line: line_number,
                    field: "selected_svg_file",
                })?;
        let selected_system_count = row.selected_system_count;
        let selected_render_source_lines = row.selected_render_source_lines.ok_or(
            LayoutTargetError::MissingRenderSettingsField {
                line: line_number,
                field: "selected_render_source_lines",
            },
        )?;
        let previous = rows.insert(
            file.clone(),
            RenderSettingsLookup::Success(SuccessfulRenderSettings {
                selected_svg_file: PathBuf::from(selected_svg_file),
                selected_system_count,
                selected_render_source_lines,
                render_augmentation: row.render_augmentation,
            }),
        );
        if matches!(previous, Some(RenderSettingsLookup::Success(_))) {
            return Err(LayoutTargetError::DuplicateRenderSettingsRow { file });
        }
    }
    Ok(rows)
}

fn collect_krn_files(input_dir: &Path) -> Result<Vec<PathBuf>, LayoutTargetError> {
    let mut files = Vec::new();
    for entry in WalkDir::new(input_dir) {
        let entry = entry.map_err(|source| LayoutTargetError::Read {
            path: input_dir.to_owned(),
            source: std::io::Error::other(source),
        })?;
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "krn") {
            continue;
        }
        files.push(entry.path().to_owned());
    }
    files.sort();
    Ok(files)
}

fn write_text(path: &Path, text: &str) -> Result<(), LayoutTargetError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| LayoutTargetError::Write {
            path: parent.to_owned(),
            source,
        })?;
    }
    fs::write(path, text).map_err(|source| LayoutTargetError::Write {
        path: path.to_owned(),
        source,
    })
}

fn normalize_settings_file_key(file: &str, input_dir: &Path) -> PathBuf {
    let path = PathBuf::from(file);
    path.strip_prefix(input_dir)
        .map(Path::to_owned)
        .unwrap_or(path)
}

fn relative_to_input(path: &Path, input_dir: &Path) -> Result<PathBuf, LayoutTargetError> {
    path.strip_prefix(input_dir)
        .map(Path::to_owned)
        .map_err(|_| LayoutTargetError::PathOutsideInputDir {
            path: path.to_owned(),
            input_dir: input_dir.to_owned(),
        })
}

fn parse_measure_id(id: &str) -> Result<usize, LayoutTargetError> {
    parse_line_id(id).ok_or_else(|| LayoutTargetError::MalformedMeasureId { id: id.to_owned() })
}

fn parse_line_id(id: &str) -> Option<usize> {
    let (_, suffix) = id.split_once("-L")?;
    let digits = suffix
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() || digits.len() != suffix.len() {
        return None;
    }
    let line = digits.parse::<usize>().ok()?;
    (line > 0).then_some(line)
}

fn has_class(node: Node<'_, '_>, class_name: &str) -> bool {
    node.attribute("class")
        .unwrap_or_default()
        .split_ascii_whitespace()
        .any(|class| class == class_name)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    fn svg_with_systems(system_measure_lines: &[&[usize]]) -> String {
        let systems = system_measure_lines
            .iter()
            .map(|measure_lines| {
                let measures = measure_lines
                    .iter()
                    .map(|line| format!(r#"<g id="measure-L{line}" class="measure"></g>"#))
                    .collect::<String>();
                format!(r#"<g class="system">{measures}</g>"#)
            })
            .collect::<String>();
        format!(r#"<svg>{systems}</svg>"#)
    }

    #[test]
    fn injected_marks_require_drawn_glyphs_and_ignore_repeat_colons() {
        let path = Path::new("page.krn");
        let svg = r#"<svg><g class="arpeg"><use/></g><g visibility="hidden"><g class="fermata"><use/></g></g></svg>"#;
        assert!(validate_injected_mark_counts(path, svg, "=!|:\n4c 4e:\n", [0, 1]).is_ok());
        assert!(matches!(
            validate_injected_mark_counts(path, svg, "4c;\n", [1, 0]),
            Err(LayoutTargetError::InjectedMarkMismatch {
                mark: "fermata",
                glyphs: 0,
                labelled: 1,
                ..
            })
        ));
        assert!(validate_injected_mark_counts(
            path,
            "<svg><g class=\"fermata\"><use/></g></svg>",
            "=;\n4c\n",
            [1, 0]
        )
        .is_ok());
        assert!(validate_injected_mark_counts(path, "<svg/>", "4c; 4e:\n", [0, 0]).is_ok());
    }

    #[test]
    fn mismatched_injected_marks_fail_the_row_and_are_counted_when_skipped() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();
        let input_dir = root.join("6_concatenated");
        fs::create_dir_all(&input_dir).unwrap();
        let path = input_dir.join("page.krn");
        let input = "**kern\n*clefG2\n*M4/4\n=1\n1c 1e 1g\n==\n*-\n";
        fs::write(&path, input).unwrap();
        let config = crate::RenderAugmentationConfig {
            fermata_page_probability: 0.0,
            arpeggio_page_probability: 1.0,
            ..Default::default()
        };
        let doc = KernDocument::parse(input).unwrap();
        let (augmented, receipt) =
            crate::augment_for_render_with_report(&doc, &config, crate::RenderAugmentationSeed(42))
                .unwrap();
        assert_eq!(receipt.injected_arpeggio_count, 1);
        let render =
            crate::serialize::render::to_render_output(&augmented, Default::default()).unwrap();
        let svg = verovioxide::Toolkit::new()
            .unwrap()
            .render_data(&render.text(), None)
            .unwrap();
        assert!(svg.contains("class=\"arpeg\""));
        let svg = svg.replace("class=\"arpeg\"", "class=\"removed-arpeg\"");
        let svg_path = root.join("page.svg");
        fs::write(&svg_path, &svg).unwrap();
        let settings = SuccessfulRenderSettings {
            selected_svg_file: svg_path.clone(),
            selected_system_count: Some(parse_rendered_layout(&svg).unwrap().system_count),
            selected_render_source_lines: render.sources().to_vec(),
            render_augmentation: Some(receipt.clone()),
        };
        let options = LayoutTargetOptions {
            input_dir: input_dir.clone(),
            render_settings_jsonl: root.join("settings.jsonl"),
            output_dir: root.join("7_targets"),
            report_jsonl: root.join("report.jsonl"),
            debug_output_dir: None,
            skip_render_failures: false,
            skip_target_failures: true,
        };
        assert!(matches!(
            generate_one_layout_target(&path, Path::new("page.krn"), &settings, &options),
            Err(LayoutTargetError::InjectedMarkMismatch { mark: "arpeg", .. })
        ));
        fs::write(&options.render_settings_jsonl, serde_json::json!({"file": path, "status": "success", "selected_svg_file": svg_path, "selected_system_count": settings.selected_system_count, "selected_render_source_lines": render.sources(), "render_augmentation": receipt}).to_string()).unwrap();
        assert!(generate_layout_targets(&options).unwrap().is_empty());
        assert!(!options.output_dir.join("page.krn").exists());
        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(root.join("dataset_manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            manifest["components"]["layout_targets"]["summary"]["skipped_target_failure_count"],
            1
        );
    }

    #[test]
    fn one_system_produces_zero_break_starts() {
        let layout = parse_rendered_layout(&svg_with_systems(&[&[2, 4]])).unwrap();
        let plan = system_break_plan_from_layout(&layout);
        assert_eq!(layout.system_count, 1);
        assert!(plan.rendered_start_lines.is_empty());
    }

    #[test]
    fn multiple_systems_produce_lower_system_first_measure_lines() {
        let layout = parse_rendered_layout(&svg_with_systems(&[&[2, 4], &[8, 12], &[16]])).unwrap();
        let plan = system_break_plan_from_layout(&layout);
        assert_eq!(plan.rendered_start_lines, vec![8, 16]);
    }

    #[test]
    fn source_resolution_scans_past_render_only_measure_anchor() {
        let plan = SystemBreakPlan {
            rendered_start_lines: vec![3],
        };
        let sources = vec![Some(1), Some(2), None, Some(8), Some(9)];

        assert_eq!(
            resolve_break_source_lines(&plan, &sources).unwrap(),
            vec![8]
        );
    }

    #[test]
    fn source_map_resolution_handles_collapsed_opening_multi_rest() {
        let layout =
            parse_rendered_layout(&svg_with_systems(&[&[1, 8, 46, 51, 55, 59], &[63]])).unwrap();
        let plan = system_break_plan_from_layout(&layout);
        let rendered_sources = (1..=63).map(Some).collect::<Vec<_>>();
        let source_lines = resolve_break_source_lines(&plan, &rendered_sources).unwrap();
        let mut document = KernDocument::parse(&collapsed_opening_rest_input()).unwrap();

        let summary = annotate_system_breaks(&mut document, &source_lines).unwrap();
        let target = SerializePreset::TranscodaV2.serialize_document(&document);

        assert_eq!(source_lines, vec![63]);
        assert_eq!(summary.insertion_source_lines, vec![59]);
        assert!(target.contains("2.r\t2.r\n=\nMM19r\tMM19r\n=\n4ff\t4aa"));
        assert!(target.contains("4gg\t4aa\n2ff\t2aa\n!!LO:LB:g=original\n=\n4ff\t4ff"));
        assert!(!target.contains("MM4r"));
    }

    #[test]
    fn layout_ordinals_map_lower_systems_to_source_barlines() {
        let layout = parse_rendered_layout(&svg_with_systems(&[&[1], &[11], &[20]])).unwrap();
        let document = KernDocument::parse("**kern\n4c\n=2\n4d\n=3\n4e\n*-\n").unwrap();

        assert_eq!(
            source_break_lines_from_layout(&layout, &document).unwrap(),
            vec![3, 5]
        );
    }

    #[test]
    fn layout_ordinals_account_for_leading_barline() {
        let layout = parse_rendered_layout(&svg_with_systems(&[&[2], &[8]])).unwrap();
        let document = KernDocument::parse("**kern\n=1\n4c\n=2\n4d\n*-\n").unwrap();

        assert_eq!(
            source_break_lines_from_layout(&layout, &document).unwrap(),
            vec![4]
        );
    }

    #[test]
    fn malformed_measure_ids_fail_clearly() {
        let error = parse_rendered_layout(
            r#"<svg><g class="system"><g id="measure-L3F1" class="measure"/></g></svg>"#,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            LayoutTargetError::MalformedMeasureId { .. }
        ));
    }

    #[test]
    fn missing_measure_ids_fail_clearly() {
        let error =
            parse_rendered_layout(r#"<svg><g class="system"><g class="measure"/></g></svg>"#)
                .unwrap_err();
        assert!(matches!(
            error,
            LayoutTargetError::MalformedMeasureId { .. }
        ));
    }

    #[test]
    fn detects_selected_system_count_disagreement() {
        let temp = TempDir::new().unwrap();
        let input_dir = temp.path().join("in");
        let svg_path = temp.path().join("page.svg");
        let settings_path = temp.path().join("settings.jsonl");
        let output_dir = temp.path().join("out");
        let report_path = temp.path().join("report.jsonl");
        fs::create_dir_all(&input_dir).unwrap();
        fs::write(input_dir.join("a.krn"), "**kern\n=1\n4c\n*-\n").unwrap();
        fs::write(&svg_path, svg_with_systems(&[&[2]])).unwrap();
        fs::write(
            &settings_path,
            serde_json::json!({
                "file": input_dir.join("a.krn"),
                "status": "success",
                "selected_svg_file": svg_path,
                "selected_system_count": 2,
                "selected_render_source_lines": [1, 2]
            })
            .to_string(),
        )
        .unwrap();

        let error = generate_layout_targets(&LayoutTargetOptions {
            input_dir,
            render_settings_jsonl: settings_path,
            output_dir,
            report_jsonl: report_path,
            debug_output_dir: None,
            skip_render_failures: false,
            skip_target_failures: false,
        })
        .unwrap_err();

        assert!(matches!(
            error,
            LayoutTargetError::SystemCountMismatch { .. }
        ));
    }

    #[test]
    fn annotation_inserts_system_break_before_barline() {
        let mut document = KernDocument::parse("**kern\n=1\n4c\n=2\n4d\n*-\n").unwrap();
        let summary = annotate_system_breaks(&mut document, &[4]).unwrap();

        assert_eq!(summary.inserted_break_count, 1);
        assert_eq!(summary.insertion_source_lines, vec![4]);
        assert!(SerializePreset::HumdrumKern
            .serialize_document(&document)
            .contains("4c\n!!LO:LB:g=original\n="));
    }

    #[test]
    fn annotation_resolves_data_anchor_to_preceding_barline() {
        let mut document = KernDocument::parse("**kern\n=1\n4c\n=2\n4d\n4e\n*-\n").unwrap();
        let summary = annotate_system_breaks(&mut document, &[6]).unwrap();

        assert_eq!(summary.insertion_source_lines, vec![4]);
        assert!(SerializePreset::HumdrumKern
            .serialize_document(&document)
            .contains("4c\n!!LO:LB:g=original\n=\n4d"));
    }

    #[test]
    fn annotation_resolves_opening_measure_data_anchor_to_following_barline() {
        let mut document = KernDocument::parse("**kern\n4c\n4d\n=\n4e\n*-\n").unwrap();
        let summary = annotate_system_breaks(&mut document, &[3]).unwrap();

        assert_eq!(summary.insertion_source_lines, vec![4]);
        assert!(SerializePreset::HumdrumKern
            .serialize_document(&document)
            .contains("4d\n!!LO:LB:g=original\n=\n4e"));
    }

    #[test]
    fn annotation_resolves_tandem_anchor_to_preceding_barline() {
        let mut document =
            KernDocument::parse("**kern\n=1\n4c\n==\n*clefG2\n*k[]\n4d\n*-\n").unwrap();
        let summary = annotate_system_breaks(&mut document, &[5]).unwrap();

        assert_eq!(summary.insertion_source_lines, vec![4]);
        assert!(SerializePreset::HumdrumKern
            .serialize_document(&document)
            .contains("4c\n!!LO:LB:g=original\n==\n*clefG2"));
    }

    #[test]
    fn annotation_refuses_invalid_source_lines() {
        let mut document = KernDocument::parse("**kern\n=1\n4c\n*-\n").unwrap();
        let error = annotate_system_breaks(&mut document, &[99]).unwrap_err();
        assert!(matches!(
            error,
            LayoutTargetError::SourceLineNotFound { line: 99 }
        ));
    }

    #[test]
    fn annotation_refuses_non_measure_anchor_source_lines() {
        let mut document = KernDocument::parse("**kern\n=1\n4c\n*-\n").unwrap();
        let error = annotate_system_breaks(&mut document, &[1]).unwrap_err();
        assert!(matches!(
            error,
            LayoutTargetError::SourceLineNotBarline { line: 1 }
        ));
    }

    #[test]
    fn annotation_handles_no_breaks() {
        let mut document = KernDocument::parse("**kern\n=1\n4c\n*-\n").unwrap();
        let summary = annotate_system_breaks(&mut document, &[]).unwrap();
        assert_eq!(summary.inserted_break_count, 0);
        assert!(!SerializePreset::HumdrumKern
            .serialize_document(&document)
            .contains("!!LO:LB"));
    }

    #[test]
    fn annotation_avoids_duplicate_existing_break() {
        let mut document =
            KernDocument::parse("**kern\n=1\n4c\n!!LO:LB:g=original\n=2\n4d\n*-\n").unwrap();
        let summary = annotate_system_breaks(&mut document, &[5]).unwrap();
        assert_eq!(summary.inserted_break_count, 0);
        assert_eq!(summary.existing_break_count, 1);
        assert_eq!(
            SerializePreset::HumdrumKern
                .serialize_document(&document)
                .matches("!!LO:LB:g=original")
                .count(),
            1
        );
    }

    #[test]
    fn annotation_normalizes_conflicting_existing_break() {
        let mut document =
            KernDocument::parse("**kern\n=1\n4c\n!!LO:LB:g=manual\n=2\n4d\n*-\n").unwrap();
        let summary = annotate_system_breaks(&mut document, &[5]).unwrap();
        assert_eq!(summary.inserted_break_count, 0);
        assert_eq!(summary.existing_break_count, 1);
        let existing_break = document
            .lines()
            .iter()
            .find_map(|line| match line {
                KernLine::Barline(barline) => barline.system_break_before.as_ref(),
                _ => None,
            })
            .unwrap();
        assert_eq!(existing_break.group.as_deref(), Some(SYSTEM_BREAK_GROUP));
    }

    #[test]
    fn skip_render_failures_omits_inputs_with_failed_render_settings_rows() {
        let temp = TempDir::new().unwrap();
        let dataset_root = temp.path().join("train/pdmx");
        let input_dir = dataset_root.join("6_concatenated");
        let svg_path = dataset_root.join("7_svg_renderings/a.svg");
        let settings_path = dataset_root.join("7_svg_renderings/render_settings.jsonl");
        let output_dir = dataset_root.join("7_targets");
        let report_path = output_dir.join("layout_targets_report.jsonl");
        fs::create_dir_all(&input_dir).unwrap();
        fs::create_dir_all(svg_path.parent().unwrap()).unwrap();
        let success_path = input_dir.join("a.krn");
        let failed_path = input_dir.join("b.krn");
        fs::write(&success_path, "**kern\n=1\n4c\n*-\n").unwrap();
        fs::write(&failed_path, "**kern\n=1\n4d\n*-\n").unwrap();
        fs::write(&svg_path, svg_with_systems(&[&[2]])).unwrap();
        fs::write(
            &settings_path,
            format!(
                "{}\n{}\n",
                serde_json::json!({
                    "file": success_path,
                    "status": "success",
                    "selected_svg_file": svg_path,
                    "selected_system_count": 1,
                    "selected_render_source_lines": [1, 2, 3]
                }),
                serde_json::json!({
                    "file": failed_path,
                    "status": "search_error",
                    "search_error": "render settings search worker produced no JSON row"
                })
            ),
        )
        .unwrap();

        let rows = generate_layout_targets(&LayoutTargetOptions {
            input_dir,
            render_settings_jsonl: settings_path,
            output_dir: output_dir.clone(),
            report_jsonl: report_path.clone(),
            debug_output_dir: None,
            skip_render_failures: true,
            skip_target_failures: false,
        })
        .unwrap();

        assert_eq!(rows.len(), 1);
        assert!(output_dir.join("a.krn").exists());
        assert!(!output_dir.join("b.krn").exists());
        let report = fs::read_to_string(report_path).unwrap();
        assert_eq!(report.lines().count(), 1);
        let manifest: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(
                dataset_root.join(crate::dataset_manifest::DATASET_MANIFEST_FILE_NAME),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            manifest["components"]["layout_targets"]["summary"]["row_count"],
            1
        );
        assert_eq!(
            manifest["components"]["layout_targets"]["summary"]["skipped_render_failure_count"],
            1
        );
        assert_eq!(
            manifest["components"]["layout_targets"]["summary"]["skipped_target_failure_count"],
            0
        );
    }

    #[test]
    fn skip_render_failures_keeps_missing_render_settings_rows_strict() {
        let temp = TempDir::new().unwrap();
        let input_dir = temp.path().join("in");
        let settings_path = temp.path().join("settings.jsonl");
        let output_dir = temp.path().join("out");
        let report_path = temp.path().join("report.jsonl");
        fs::create_dir_all(&input_dir).unwrap();
        fs::write(input_dir.join("missing.krn"), "**kern\n=1\n4c\n*-\n").unwrap();
        fs::write(&settings_path, "").unwrap();

        let error = generate_layout_targets(&LayoutTargetOptions {
            input_dir,
            render_settings_jsonl: settings_path,
            output_dir,
            report_jsonl: report_path,
            debug_output_dir: None,
            skip_render_failures: true,
            skip_target_failures: false,
        })
        .unwrap_err();

        assert!(matches!(
            error,
            LayoutTargetError::MissingRenderSettingsRow { .. }
        ));
    }

    #[test]
    fn skip_target_failures_omits_inputs_with_serialization_errors() {
        let temp = TempDir::new().unwrap();
        let dataset_root = temp.path().join("train/pdmx");
        let input_dir = dataset_root.join("6_concatenated");
        let svg_path = dataset_root.join("7_svg_renderings/bad.svg");
        let settings_path = dataset_root.join("7_svg_renderings/render_settings.jsonl");
        let output_dir = dataset_root.join("7_targets");
        let report_path = output_dir.join("layout_targets_report.jsonl");
        fs::create_dir_all(&input_dir).unwrap();
        fs::create_dir_all(svg_path.parent().unwrap()).unwrap();
        let bad_path = input_dir.join("bad.krn");
        fs::write(
            &bad_path,
            "**kern\t**kern\n*M4/4\t*M4/4\n2c\t4e\n*^\t*\n.\t4d\t4f\n*-\t*-\t*-\n",
        )
        .unwrap();
        fs::write(&svg_path, svg_with_systems(&[&[2]])).unwrap();
        fs::write(
            &settings_path,
            format!(
                "{}\n",
                serde_json::json!({
                    "file": bad_path,
                    "status": "success",
                    "selected_svg_file": svg_path,
                    "selected_system_count": 1,
                    "selected_render_source_lines": [1, 2, 3, 4, 5, 6]
                })
            ),
        )
        .unwrap();

        let rows = generate_layout_targets(&LayoutTargetOptions {
            input_dir,
            render_settings_jsonl: settings_path,
            output_dir: output_dir.clone(),
            report_jsonl: report_path.clone(),
            debug_output_dir: None,
            skip_render_failures: false,
            skip_target_failures: true,
        })
        .unwrap();

        assert!(rows.is_empty());
        assert!(!output_dir.join("bad.krn").exists());
        assert_eq!(fs::read_to_string(report_path).unwrap().lines().count(), 0);
        let manifest: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(
                dataset_root.join(crate::dataset_manifest::DATASET_MANIFEST_FILE_NAME),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            manifest["components"]["layout_targets"]["summary"]["row_count"],
            0
        );
        assert_eq!(
            manifest["components"]["layout_targets"]["summary"]["skipped_target_failure_count"],
            1
        );
    }

    #[test]
    fn successful_render_settings_rows_require_render_source_map() {
        let temp = TempDir::new().unwrap();
        let input_dir = temp.path().join("in");
        let svg_path = temp.path().join("page.svg");
        let settings_path = temp.path().join("settings.jsonl");
        let output_dir = temp.path().join("out");
        let report_path = temp.path().join("report.jsonl");
        fs::create_dir_all(&input_dir).unwrap();
        fs::write(input_dir.join("a.krn"), "**kern\n=1\n4c\n*-\n").unwrap();
        fs::write(&svg_path, svg_with_systems(&[&[2]])).unwrap();
        fs::write(
            &settings_path,
            serde_json::json!({
                "file": input_dir.join("a.krn"),
                "status": "success",
                "selected_svg_file": svg_path,
                "selected_system_count": 1
            })
            .to_string(),
        )
        .unwrap();

        let error = generate_layout_targets(&LayoutTargetOptions {
            input_dir,
            render_settings_jsonl: settings_path,
            output_dir,
            report_jsonl: report_path,
            debug_output_dir: None,
            skip_render_failures: false,
            skip_target_failures: false,
        })
        .unwrap_err();

        assert!(matches!(
            error,
            LayoutTargetError::MissingRenderSettingsField {
                field: "selected_render_source_lines",
                ..
            }
        ));
    }

    #[test]
    fn integration_generates_target_and_report() {
        let temp = TempDir::new().unwrap();
        let dataset_root = temp.path().join("train/pdmx");
        let input_dir = dataset_root.join("6_concatenated");
        let svg_path = dataset_root.join("7_svg_renderings/page.svg");
        let settings_path = dataset_root.join("7_svg_renderings/render_settings.jsonl");
        let output_dir = dataset_root.join("7_targets");
        let debug_dir = dataset_root.join("debug_targets");
        let report_path = dataset_root.join("7_targets/layout_targets_report.jsonl");
        fs::create_dir_all(&input_dir).unwrap();
        fs::create_dir_all(svg_path.parent().unwrap()).unwrap();
        fs::create_dir_all(settings_path.parent().unwrap()).unwrap();
        let input_path = input_dir.join("a.krn");
        fs::write(&input_path, "**kern\n=1\n4cnZ\n=2\n4d#j\n*-\n").unwrap();
        fs::write(&svg_path, svg_with_systems(&[&[100], &[200]])).unwrap();
        let mut render_source_lines = vec![None; 200];
        render_source_lines[199] = Some(5);
        fs::write(
            dataset_root.join(crate::dataset_manifest::DATASET_MANIFEST_FILE_NAME),
            r#"{
                "schema_version": "transcoda.dataset_manifest.v2",
                "dataset": {"split": "train", "name": "pdmx", "root": "unused"},
                "components": {
                    "svg_renders": {
                        "content_id": "svg-v1",
                        "producer": "test",
                        "paths": ["7_svg_renderings"],
                        "summary": {}
                    }
                }
            }"#,
        )
        .unwrap();
        fs::write(
            &settings_path,
            format!(
                "{}\n",
                serde_json::json!({
                    "file": input_path,
                    "status": "success",
                    "selected_svg_file": svg_path,
                    "selected_system_count": 2,
                    "selected_render_source_lines": render_source_lines
                })
            ),
        )
        .unwrap();

        let rows = generate_layout_targets(&LayoutTargetOptions {
            input_dir,
            render_settings_jsonl: settings_path,
            output_dir: output_dir.clone(),
            report_jsonl: report_path.clone(),
            debug_output_dir: Some(debug_dir.clone()),
            skip_render_failures: false,
            skip_target_failures: false,
        })
        .unwrap();

        let target = fs::read_to_string(output_dir.join("a.krn")).unwrap();
        assert_eq!(target, "*clefG2\n*k[]\n4cn\n!!LO:LB:g=original\n=\n4d#\n");
        assert!(fs::read_to_string(&input_path).unwrap().contains("4cnZ"));
        assert!(fs::read_to_string(debug_dir.join("a.krn"))
            .unwrap()
            .contains("4d#j"));
        assert!(fs::read_to_string(debug_dir.join("a.krn"))
            .unwrap()
            .contains("!!LO:LB:g=original"));
        assert_eq!(rows[0].source_break_lines, vec![4]);
        let report = fs::read_to_string(report_path).unwrap();
        assert!(report.contains(r#""inserted_break_count":1"#));
        let manifest: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(
                dataset_root.join(crate::dataset_manifest::DATASET_MANIFEST_FILE_NAME),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            manifest["components"]["svg_renders"]["content_id"],
            "svg-v1"
        );
        assert!(manifest["components"]["layout_targets"]["content_id"].is_string());
    }

    fn collapsed_opening_rest_input() -> String {
        let mut lines = vec![
            "**kern\t**dynam\t**kern\t**dynam".to_owned(),
            "*clefG2\t*\t*clefG^2\t*".to_owned(),
            "*k[b-]\t*\t*k[b-]\t*".to_owned(),
            "*M3/4\t*\t*M3/4\t*".to_owned(),
        ];
        for _ in 0..20 {
            lines.push("2.r\t.\t2.r\t.".to_owned());
            lines.push("=\t=\t=\t=".to_owned());
        }
        lines.extend(
            [
                "4ff\tpp\t4aa\tpp",
                "4.ff\t.\t4.aa\t.",
                "8aa'\t.\t8ccc'\t.",
                "=\t=\t=\t=",
                "4gg\t.\t4bb-\t.",
                "4ggv\t.\t2bb-\t.",
                "4ff\t.\t.\t.",
                "=\t=\t=\t=",
                "4ee\t.\t4gg\t.",
                "4.ee\t.\t4.gg\t.",
                "8gg'\t.\t8bb-'\t.",
                "=\t=\t=\t=",
                "4gg\t.\t4aa\t.",
                "2ffv\t.\t2aa\t.",
                "=\t=\t=\t=",
                "4ff\t.\t4ff\t.",
                "4ee\t.\t4gg\t.",
                "8ffL\t.\t4aa\t.",
                "8eeJ\t.\t.\t.",
                "=\t=\t=\t=",
                "*-\t*-\t*-\t*-",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        format!("{}\n", lines.join("\n"))
    }
}
