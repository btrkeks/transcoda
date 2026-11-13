use std::path::PathBuf;

use crate::layout_targets::{generate_layout_targets, LayoutTargetOptions};
use clap::{ArgAction, Args};

#[derive(Debug, Args)]
pub(in crate::cli) struct GenerateLayoutTargetsArgs {
    /// Directory of concatenated render-input Humdrum .krn files.
    input_dir: PathBuf,
    /// Output directory, or legacy render_settings_jsonl output_dir report_jsonl paths.
    #[arg(value_name = "PATH", num_args = 1..=3, action = ArgAction::Append)]
    paths: Vec<PathBuf>,
    /// Optional directory for annotated Humdrum debug files.
    #[arg(long)]
    debug_output_dir: Option<PathBuf>,
    /// Skip .krn files with explicit non-success render settings rows.
    #[arg(long)]
    skip_render_failures: bool,
    /// Skip .krn files that fail target annotation or serialization.
    #[arg(long)]
    skip_target_failures: bool,
}

struct ResolvedLayoutTargetArgs {
    input_dir: PathBuf,
    render_settings_jsonl: PathBuf,
    output_dir: PathBuf,
    report_jsonl: PathBuf,
    debug_output_dir: Option<PathBuf>,
    skip_render_failures: bool,
    skip_target_failures: bool,
}

pub(in crate::cli) fn generate_layout_targets_command(
    args: GenerateLayoutTargetsArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let args = resolve_args(args)?;
    let rows = generate_layout_targets(&LayoutTargetOptions {
        input_dir: args.input_dir,
        render_settings_jsonl: args.render_settings_jsonl,
        output_dir: args.output_dir,
        report_jsonl: args.report_jsonl,
        debug_output_dir: args.debug_output_dir,
        skip_render_failures: args.skip_render_failures,
        skip_target_failures: args.skip_target_failures,
    })?;

    eprintln!("Generated {} layout-derived targets.", rows.len());
    Ok(())
}

fn resolve_args(
    args: GenerateLayoutTargetsArgs,
) -> Result<ResolvedLayoutTargetArgs, Box<dyn std::error::Error>> {
    match args.paths.as_slice() {
        [output_dir] => {
            let dataset_root = args.input_dir.parent().ok_or_else(|| {
                format!(
                    "cannot infer dataset root from input dir: {}",
                    args.input_dir.display()
                )
            })?;
            let render_settings_jsonl = dataset_root
                .join("7_svg_renderings")
                .join("render_settings.jsonl");
            Ok(ResolvedLayoutTargetArgs {
                input_dir: args.input_dir,
                render_settings_jsonl,
                output_dir: output_dir.clone(),
                report_jsonl: output_dir.join("layout_targets_report.jsonl"),
                debug_output_dir: args.debug_output_dir,
                skip_render_failures: args.skip_render_failures,
                skip_target_failures: args.skip_target_failures,
            })
        }
        [render_settings_jsonl, output_dir, report_jsonl] => Ok(ResolvedLayoutTargetArgs {
            input_dir: args.input_dir,
            render_settings_jsonl: render_settings_jsonl.clone(),
            output_dir: output_dir.clone(),
            report_jsonl: report_jsonl.clone(),
            debug_output_dir: args.debug_output_dir,
            skip_render_failures: args.skip_render_failures,
            skip_target_failures: args.skip_target_failures,
        }),
        _ => Err(
            "expected either <input_dir> <output_dir> or legacy <input_dir> <render_settings_jsonl> <output_dir> <report_jsonl>"
                .into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(input_dir: PathBuf, paths: Vec<PathBuf>) -> GenerateLayoutTargetsArgs {
        GenerateLayoutTargetsArgs {
            input_dir,
            paths,
            debug_output_dir: None,
            skip_render_failures: false,
            skip_target_failures: false,
        }
    }

    #[test]
    fn infers_stage_local_jsonl_paths_from_short_form() {
        let resolved = resolve_args(args(
            PathBuf::from("data/interim/train/pdmx/6_concatenated"),
            vec![PathBuf::from("data/interim/train/pdmx/7_targets")],
        ))
        .unwrap();

        assert_eq!(
            resolved.render_settings_jsonl,
            PathBuf::from("data/interim/train/pdmx/7_svg_renderings/render_settings.jsonl")
        );
        assert_eq!(
            resolved.report_jsonl,
            PathBuf::from("data/interim/train/pdmx/7_targets/layout_targets_report.jsonl")
        );
    }

    #[test]
    fn preserves_legacy_explicit_jsonl_paths() {
        let resolved = resolve_args(args(
            PathBuf::from("in"),
            vec![
                PathBuf::from("settings.jsonl"),
                PathBuf::from("targets"),
                PathBuf::from("report.jsonl"),
            ],
        ))
        .unwrap();

        assert_eq!(
            resolved.render_settings_jsonl,
            PathBuf::from("settings.jsonl")
        );
        assert_eq!(resolved.output_dir, PathBuf::from("targets"));
        assert_eq!(resolved.report_jsonl, PathBuf::from("report.jsonl"));
    }
}
