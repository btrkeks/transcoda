use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

pub const DEFAULT_SAMPLE_TRANSCRIPTION_NAME: &str = "transcription.krn";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchInputLayout {
    PreserveRelativePaths,
    SampleDirs,
}

#[derive(Debug, Clone)]
pub struct BatchTask {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
}

pub fn collect_batch_tasks(
    input_path: &Path,
    output_path: &Path,
    layout: BatchInputLayout,
    sample_transcription_name: &str,
    allow_single_file: bool,
) -> Result<Vec<BatchTask>, io::Error> {
    if input_path.is_file() {
        if allow_single_file && layout == BatchInputLayout::PreserveRelativePaths {
            return Ok(vec![BatchTask {
                input_path: input_path.to_owned(),
                output_path: output_path.to_owned(),
            }]);
        }

        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{} is a file, but --input-layout sample-dirs requires a directory",
                input_path.display()
            ),
        ));
    }

    match layout {
        BatchInputLayout::PreserveRelativePaths => {
            collect_preserve_relative_tasks(input_path, output_path)
        }
        BatchInputLayout::SampleDirs => {
            collect_sample_dir_tasks(input_path, output_path, sample_transcription_name)
        }
    }
}

#[derive(Debug, Clone)]
pub struct SampleTranscription {
    pub sample_id: String,
    pub input_path: PathBuf,
}

pub fn collect_sample_transcriptions(
    input_path: &Path,
    sample_transcription_name: &str,
) -> Result<Vec<SampleTranscription>, io::Error> {
    let files = collect_krn_files(input_path)?;
    let mut samples = files
        .into_iter()
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name == sample_transcription_name)
        })
        .map(|input_path| {
            let sample_id = input_path
                .parent()
                .and_then(Path::file_name)
                .expect("sample transcription path should have a parent directory")
                .to_string_lossy()
                .into_owned();
            SampleTranscription {
                sample_id,
                input_path,
            }
        })
        .collect::<Vec<_>>();
    samples.sort_by(|left, right| {
        (left.sample_id.as_str(), left.input_path.as_path())
            .cmp(&(right.sample_id.as_str(), right.input_path.as_path()))
    });
    reject_duplicate_sample_ids(&samples)?;
    Ok(samples)
}

fn reject_duplicate_sample_ids(samples: &[SampleTranscription]) -> Result<(), io::Error> {
    let mut by_sample_id = BTreeMap::<&str, Vec<&Path>>::new();
    for sample in samples {
        by_sample_id
            .entry(sample.sample_id.as_str())
            .or_default()
            .push(sample.input_path.as_path());
    }

    if let Some((sample_id, input_paths)) = by_sample_id
        .into_iter()
        .find(|(_, input_paths)| input_paths.len() > 1)
    {
        let inputs = input_paths
            .into_iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("multiple inputs map to sample id {sample_id}: {inputs}"),
        ));
    }

    Ok(())
}

pub fn collect_krn_files(input_path: &Path) -> Result<Vec<PathBuf>, io::Error> {
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

fn collect_preserve_relative_tasks(
    input_path: &Path,
    output_path: &Path,
) -> Result<Vec<BatchTask>, io::Error> {
    let files = collect_krn_files(input_path)?;
    Ok(files
        .into_iter()
        .map(|input_file| {
            let relative_path = input_file
                .strip_prefix(input_path)
                .expect("collect_krn_files yielded a path below input_dir");
            BatchTask {
                output_path: output_path.join(relative_path),
                input_path: input_file,
            }
        })
        .collect())
}

fn collect_sample_dir_tasks(
    input_path: &Path,
    output_path: &Path,
    sample_transcription_name: &str,
) -> Result<Vec<BatchTask>, io::Error> {
    let files = collect_krn_files(input_path)?;
    let mut tasks = files
        .into_iter()
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name == sample_transcription_name)
        })
        .map(|input_file| {
            let sample_name = input_file
                .parent()
                .and_then(Path::file_name)
                .expect("sample transcription path should have a parent directory");
            let mut output_file_name = sample_name.to_os_string();
            output_file_name.push(".krn");
            BatchTask {
                output_path: output_path.join(output_file_name),
                input_path: input_file,
            }
        })
        .collect::<Vec<_>>();
    tasks.sort_by(|left, right| left.input_path.cmp(&right.input_path));
    reject_output_collisions(&tasks)?;
    Ok(tasks)
}

fn reject_output_collisions(tasks: &[BatchTask]) -> Result<(), io::Error> {
    let mut by_output = BTreeMap::<&Path, Vec<&Path>>::new();
    for task in tasks {
        by_output
            .entry(task.output_path.as_path())
            .or_default()
            .push(task.input_path.as_path());
    }

    if let Some((output_path, input_paths)) = by_output
        .into_iter()
        .find(|(_, input_paths)| input_paths.len() > 1)
    {
        let inputs = input_paths
            .into_iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "multiple inputs map to output {}: {}",
                output_path.display(),
                inputs
            ),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

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

    #[test]
    fn sample_dirs_maps_parent_directory_to_flat_output_name() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("praig");
        let output = temp.path().join("out");
        write(&input.join("test_000000/transcription.krn"), valid_kern());
        write(&input.join("test_000000/sidecar.krn"), valid_kern());

        let tasks = collect_batch_tasks(
            &input,
            &output,
            BatchInputLayout::SampleDirs,
            DEFAULT_SAMPLE_TRANSCRIPTION_NAME,
            false,
        )
        .unwrap();

        assert_eq!(tasks.len(), 1);
        assert_eq!(
            tasks[0].input_path,
            input.join("test_000000/transcription.krn")
        );
        assert_eq!(tasks[0].output_path, output.join("test_000000.krn"));
    }

    #[test]
    fn sample_dirs_appends_krn_without_truncating_dotted_sample_names() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("imslp");
        let output = temp.path().join("out");
        let sample_name = "sample_0001_Example__Piece_B.16__p0001";
        write(
            &input.join(sample_name).join("transcription.krn"),
            valid_kern(),
        );

        let tasks = collect_batch_tasks(
            &input,
            &output,
            BatchInputLayout::SampleDirs,
            DEFAULT_SAMPLE_TRANSCRIPTION_NAME,
            false,
        )
        .unwrap();

        assert_eq!(tasks.len(), 1);
        assert_eq!(
            tasks[0].output_path,
            output.join(format!("{sample_name}.krn"))
        );
    }

    #[test]
    fn collect_sample_transcriptions_maps_parent_directory_to_sample_id() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("samples");
        write(&input.join("sample_b/transcription.krn"), valid_kern());
        write(&input.join("sample_a/transcription.krn"), valid_kern());
        write(&input.join("sample_a/sidecar.krn"), valid_kern());

        let samples =
            collect_sample_transcriptions(&input, DEFAULT_SAMPLE_TRANSCRIPTION_NAME).unwrap();

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].sample_id, "sample_a");
        assert_eq!(
            samples[0].input_path,
            input.join("sample_a/transcription.krn")
        );
        assert_eq!(samples[1].sample_id, "sample_b");
    }

    #[test]
    fn collect_sample_transcriptions_rejects_duplicate_sample_ids() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("samples");
        write(
            &input.join("split_a/sample_x/transcription.krn"),
            valid_kern(),
        );
        write(
            &input.join("split_b/sample_x/transcription.krn"),
            valid_kern(),
        );

        let error =
            collect_sample_transcriptions(&input, DEFAULT_SAMPLE_TRANSCRIPTION_NAME).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error
            .to_string()
            .contains("multiple inputs map to sample id sample_x"));
    }

    #[test]
    fn collect_sample_transcriptions_returns_empty_for_directory_without_samples() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("samples");
        fs::create_dir_all(&input).unwrap();

        let samples =
            collect_sample_transcriptions(&input, DEFAULT_SAMPLE_TRANSCRIPTION_NAME).unwrap();

        assert!(samples.is_empty());
    }

    #[test]
    fn sample_dirs_rejects_output_collisions() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("praig");
        let output = temp.path().join("out");
        write(
            &input.join("split_a/test_000000/transcription.krn"),
            valid_kern(),
        );
        write(
            &input.join("split_b/test_000000/transcription.krn"),
            valid_kern(),
        );

        let error = collect_batch_tasks(
            &input,
            &output,
            BatchInputLayout::SampleDirs,
            DEFAULT_SAMPLE_TRANSCRIPTION_NAME,
            false,
        )
        .unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("multiple inputs map to output"));
        assert!(error.to_string().contains("test_000000.krn"));
    }
}
