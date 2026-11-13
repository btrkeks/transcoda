//! Dataset artifact manifest shared by dataset-generation commands.

use std::{
    collections::BTreeMap,
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const DATASET_MANIFEST_FILE_NAME: &str = "dataset_manifest.json";
pub const DATASET_MANIFEST_SCHEMA_VERSION: &str = "transcoda.dataset_manifest.v2";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatasetManifest {
    pub schema_version: String,
    pub dataset: DatasetIdentity,
    pub components: BTreeMap<String, DatasetManifestComponent>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatasetIdentity {
    pub split: Option<String>,
    pub name: String,
    pub root: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatasetManifestComponent {
    pub content_id: String,
    pub producer: String,
    pub paths: Vec<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub summary: BTreeMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct DatasetComponentUpdate {
    pub name: String,
    pub producer: String,
    pub paths: Vec<PathBuf>,
    pub summary: BTreeMap<String, usize>,
}

pub fn update_dataset_manifest(
    dataset_root: &Path,
    update: DatasetComponentUpdate,
) -> Result<DatasetManifest, io::Error> {
    fs::create_dir_all(dataset_root)?;
    let manifest_path = dataset_root.join(DATASET_MANIFEST_FILE_NAME);
    let mut manifest = load_or_create_manifest(dataset_root, &manifest_path)?;
    let content_id = component_content_id(&update, dataset_root)?;
    let component = DatasetManifestComponent {
        content_id,
        producer: update.producer,
        paths: update
            .paths
            .iter()
            .map(|path| manifest_path_string(path, dataset_root))
            .collect(),
        summary: update.summary,
    };
    manifest
        .components
        .insert(update.name.to_owned(), component);
    write_manifest_atomic(&manifest_path, &manifest)?;
    Ok(manifest)
}

fn load_or_create_manifest(
    dataset_root: &Path,
    manifest_path: &Path,
) -> Result<DatasetManifest, io::Error> {
    match fs::read_to_string(manifest_path) {
        Ok(contents) => match serde_json::from_str::<DatasetManifest>(&contents) {
            Ok(manifest)
                if manifest.schema_version == DATASET_MANIFEST_SCHEMA_VERSION
                    && manifest.dataset.name == dataset_name(dataset_root) =>
            {
                Ok(manifest)
            }
            _ => Ok(new_manifest(dataset_root)),
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(new_manifest(dataset_root)),
        Err(error) => Err(error),
    }
}

fn new_manifest(dataset_root: &Path) -> DatasetManifest {
    DatasetManifest {
        schema_version: DATASET_MANIFEST_SCHEMA_VERSION.to_owned(),
        dataset: DatasetIdentity {
            split: dataset_split(dataset_root),
            name: dataset_name(dataset_root),
            root: dataset_root.display().to_string(),
        },
        components: BTreeMap::new(),
    }
}

fn dataset_name(dataset_root: &Path) -> String {
    dataset_root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn dataset_split(dataset_root: &Path) -> Option<String> {
    dataset_root
        .parent()
        .and_then(Path::file_name)
        .map(|name| name.to_string_lossy().into_owned())
}

fn manifest_path_string(path: &Path, dataset_root: &Path) -> String {
    path.strip_prefix(dataset_root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn component_content_id(
    update: &DatasetComponentUpdate,
    dataset_root: &Path,
) -> Result<String, io::Error> {
    let mut hasher = Sha256::new();
    hasher.update(update.name.as_bytes());
    let mut paths = update.paths.clone();
    paths.sort();
    for path in paths {
        hash_path(&mut hasher, &path, dataset_root)?;
    }
    for (key, value) in &update.summary {
        hasher.update(key.as_bytes());
        hasher.update(value.to_string().as_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_path(hasher: &mut Sha256, path: &Path, dataset_root: &Path) -> Result<(), io::Error> {
    hasher.update(manifest_path_string(path, dataset_root).as_bytes());
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => {
            hasher.update(b"\0dir\0");
            let mut children = fs::read_dir(path)?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<Result<Vec<_>, _>>()?;
            children.sort();
            for child in children {
                hash_path(hasher, &child, dataset_root)?;
            }
        }
        Ok(metadata) if metadata.is_file() => {
            hasher.update(b"\0file\0");
            hasher.update(metadata.len().to_string().as_bytes());
            let mut file = fs::File::open(path)?;
            let mut buffer = vec![0_u8; 1024 * 1024];
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                hasher.update(&buffer[..count]);
            }
        }
        Ok(_) => hasher.update(b"\0other\0"),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            hasher.update(b"\0missing\0");
        }
        Err(error) => return Err(error),
    }
    Ok(())
}

fn write_manifest_atomic(path: &Path, manifest: &DatasetManifest) -> Result<(), io::Error> {
    let tmp_path = path.with_extension(format!(
        "json.{}.tmp",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let bytes = serde_json::to_vec_pretty(manifest).map_err(io::Error::other)?;
    let mut file = fs::File::create(&tmp_path)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(tmp_path, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn update(name: &str, path: PathBuf, count: usize) -> DatasetComponentUpdate {
        DatasetComponentUpdate {
            name: name.to_owned(),
            producer: format!("producer:{name}"),
            paths: vec![path],
            summary: BTreeMap::from([("count".to_owned(), count)]),
        }
    }

    #[test]
    fn updates_one_manifest_without_dropping_existing_components() {
        let temp = TempDir::new().unwrap();
        let dataset_root = temp.path().join("train/pdmx");
        let svg_path = dataset_root.join("7_svg_renderings/a.svg");
        let target_path = dataset_root.join("7_targets/a.krn");
        fs::create_dir_all(svg_path.parent().unwrap()).unwrap();
        fs::create_dir_all(target_path.parent().unwrap()).unwrap();
        fs::write(&svg_path, "<svg/>").unwrap();
        fs::write(&target_path, "**kern\n*-\n").unwrap();

        update_dataset_manifest(&dataset_root, update("svg_renders", svg_path, 1)).unwrap();
        let manifest =
            update_dataset_manifest(&dataset_root, update("layout_targets", target_path, 1))
                .unwrap();

        assert_eq!(manifest.schema_version, DATASET_MANIFEST_SCHEMA_VERSION);
        assert_eq!(manifest.dataset.split.as_deref(), Some("train"));
        assert_eq!(manifest.dataset.name, "pdmx");
        assert!(manifest.components.contains_key("svg_renders"));
        assert!(manifest.components.contains_key("layout_targets"));
        assert!(manifest.components["svg_renders"].content_id.len() == 64);
        assert!(dataset_root.join(DATASET_MANIFEST_FILE_NAME).is_file());
    }

    #[test]
    fn component_content_id_is_stable_across_updates_until_bytes_change() {
        let temp = TempDir::new().unwrap();
        let dataset_root = temp.path().join("train/pdmx");
        let svg_path = dataset_root.join("7_svg_renderings/a.svg");
        fs::create_dir_all(svg_path.parent().unwrap()).unwrap();
        fs::write(&svg_path, "<svg/>").unwrap();

        let first =
            update_dataset_manifest(&dataset_root, update("svg_renders", svg_path.clone(), 1))
                .unwrap()
                .components["svg_renders"]
                .content_id
                .clone();
        let second =
            update_dataset_manifest(&dataset_root, update("svg_renders", svg_path.clone(), 1))
                .unwrap()
                .components["svg_renders"]
                .content_id
                .clone();
        fs::write(&svg_path, "<svg><path/></svg>").unwrap();
        let changed = update_dataset_manifest(&dataset_root, update("svg_renders", svg_path, 1))
            .unwrap()
            .components["svg_renders"]
            .content_id
            .clone();

        assert_eq!(first, second);
        assert_ne!(first, changed);
    }
}
