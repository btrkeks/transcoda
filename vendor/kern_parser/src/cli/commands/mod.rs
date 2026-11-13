use std::io;
use std::path::{Path, PathBuf};

use crate::batch::input::BatchInputLayout;
use clap::ValueEnum;
use walkdir::WalkDir;

pub(super) mod concatenate_dir;
pub(super) mod generate_layout_targets;
pub(super) mod normalize_dir;
pub(super) mod quality_filter;
pub(super) mod render_settings_search;
pub(super) mod serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(super) enum BatchInputLayoutArg {
    /// Recursively preserve paths below the input directory.
    PreserveRelativePaths,
    /// Treat each parent directory as a sample id and read transcription.krn files.
    SampleDirs,
}

impl From<BatchInputLayoutArg> for BatchInputLayout {
    fn from(value: BatchInputLayoutArg) -> Self {
        match value {
            BatchInputLayoutArg::PreserveRelativePaths => BatchInputLayout::PreserveRelativePaths,
            BatchInputLayoutArg::SampleDirs => BatchInputLayout::SampleDirs,
        }
    }
}

pub(super) fn collect_krn_files(input_path: &Path) -> Result<Vec<PathBuf>, io::Error> {
    if !input_path.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} is not a directory", input_path.display()),
        ));
    }

    let mut files = Vec::new();
    for entry in WalkDir::new(input_path) {
        let entry = entry?;
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "krn") {
            continue;
        }
        files.push(entry.path().to_owned());
    }
    files.sort();

    Ok(files)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use tempfile::TempDir;

    use super::*;

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    fn valid_kern() -> &'static str {
        "**kern\n*clefG2\n*k[]\n*M4/4\n4c\n*-\n"
    }

    #[test]
    fn collect_krn_files_preserves_nested_sorted_paths() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("in");
        write(&input_path.join("z/piece.krn"), valid_kern());
        write(&input_path.join("a/piece.krn"), valid_kern());
        write(&input_path.join("ignore.txt"), "");

        let files = collect_krn_files(&input_path).unwrap();
        let relative = files
            .iter()
            .map(|path| path.strip_prefix(&input_path).unwrap().to_owned())
            .collect::<Vec<_>>();

        assert_eq!(
            relative,
            vec![PathBuf::from("a/piece.krn"), PathBuf::from("z/piece.krn")]
        );
    }
}
