use std::collections::HashSet;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::profile::RenderSettingsSearchProfileRow;
use super::row::RenderSettingsSearchRow;

pub(super) fn selected_svg_path(
    input_path: &Path,
    input_root: &Path,
    output_svg_dir: &Path,
) -> Result<PathBuf, String> {
    let relative = input_path.strip_prefix(input_root).map_err(|_| {
        format!(
            "{} is not inside input root {}",
            input_path.display(),
            input_root.display()
        )
    })?;
    let mut output_path = output_svg_dir.join(relative);
    output_path.set_extension("svg");
    Ok(output_path)
}

pub(super) fn load_render_settings_search_resume(
    path: &Path,
    input_root: &Path,
    quiet: bool,
) -> Result<HashSet<String>, Box<dyn std::error::Error>> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(HashSet::new()),
        Err(error) => return Err(error.into()),
    };

    let mut completed = HashSet::new();
    let mut valid_lines = Vec::new();
    let mut dropped_line_count = 0usize;
    let mut stale_success_count = 0usize;
    let mut needs_repair = !contents.is_empty() && !contents.ends_with('\n');

    for line in contents.lines() {
        if line.trim().is_empty() {
            needs_repair = true;
            continue;
        }

        match serde_json::from_str::<RenderSettingsSearchRow>(line) {
            Ok(row) => {
                if row.search_success()
                    && (row.selected_render_source_lines.is_none()
                        || row
                            .render_augmentation
                            .as_ref()
                            .and_then(|report| report.config.as_ref())
                            .is_none())
                {
                    stale_success_count += 1;
                    needs_repair = true;
                    continue;
                }
                if !row.search_failed() {
                    completed.insert(resume_file_key(Path::new(&row.file), input_root));
                }
                valid_lines.push(line.to_owned());
            }
            Err(_) => {
                dropped_line_count += 1;
                needs_repair = true;
            }
        }
    }

    if needs_repair {
        let mut repaired = String::new();
        for line in &valid_lines {
            repaired.push_str(line);
            repaired.push('\n');
        }
        fs::write(path, repaired)?;
        if !quiet && dropped_line_count > 0 {
            eprintln!(
                "Resume repaired {} malformed JSONL line(s) in {}.",
                dropped_line_count,
                path.display()
            );
        }
        if !quiet && stale_success_count > 0 {
            eprintln!(
                "Resume dropped {} stale success row(s) without a source map or augmentation config in {}.",
                stale_success_count,
                path.display()
            );
        }
    }

    Ok(completed)
}

pub(super) fn resume_file_key(path: &Path, input_root: &Path) -> String {
    if let Ok(relative) = path.strip_prefix(input_root) {
        return relative.to_string_lossy().into_owned();
    }

    if let (Ok(path), Ok(input_root)) = (path.canonicalize(), input_root.canonicalize()) {
        if let Ok(relative) = path.strip_prefix(input_root) {
            return relative.to_string_lossy().into_owned();
        }
        return path.to_string_lossy().into_owned();
    }

    path.to_string_lossy().into_owned()
}

pub(super) fn create_render_settings_search_writer(
    path: &Path,
    append: bool,
) -> Result<BufWriter<fs::File>, io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = if append {
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?
    } else {
        fs::File::create(path)?
    };
    Ok(BufWriter::new(file))
}

pub(super) fn write_render_settings_search_row(
    writer: &mut BufWriter<fs::File>,
    row: &RenderSettingsSearchRow,
) -> Result<(), io::Error> {
    serde_json::to_writer(&mut *writer, row)?;
    writer.write_all(b"\n")
}

pub(super) fn write_render_settings_search_profile_row(
    writer: &mut BufWriter<fs::File>,
    row: &RenderSettingsSearchProfileRow,
) -> Result<(), io::Error> {
    serde_json::to_writer(&mut *writer, row)?;
    writer.write_all(b"\n")
}
