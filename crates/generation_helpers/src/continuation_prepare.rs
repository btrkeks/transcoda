//! Prepare one renderable continuation from a production concatenation row.
use kern_parser::{document::Framing, meter::MeterDisplay, score::Score, KernLine};
use kern_parser::{
    normalization::{multi_measure_rests::CompactMultiMeasureRests, NormalizationPass},
    serialize::SerializePreset,
    transforms::split::{
        hide_opening_meter_except_staffs, musical_bar_count,
        split_trailing_bars_with_prefix_policy, SlicePrefixPolicy,
    },
    KernDocument,
};
use serde::Deserialize;
use std::collections::BTreeSet;

use serde_json::json;
use std::{
    fs,
    io::{self, Read},
};

fn printed_staffs(document: &KernDocument) -> Result<BTreeSet<u32>, Box<dyn std::error::Error>> {
    let score = Score::resolve(document, Framing::Excerpt)?;
    let data = document
        .lines()
        .iter()
        .position(|line| matches!(line, KernLine::Data(_)))
        .ok_or("page has no data")?;
    Ok(score
        .topology()
        .layers_at(data)
        .iter()
        .enumerate()
        .filter_map(|(column, &layer)| {
            let declaration = &score.state_at(data, column).track.meter;
            (declaration.is_declared() && declaration.display() != MeterDisplay::Hidden)
                .then(|| {
                    score
                        .topology()
                        .track(score.topology().track_of(layer))
                        .staff
                })
                .flatten()
        })
        .collect())
}

#[derive(Deserialize)]
struct PageRequest {
    input: String,
    output: Option<String>,
    #[serde(default)]
    preflight: bool,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    start_bar_index: usize,
    #[serde(default)]
    end_bar_index: Option<usize>,
}

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum Request {
    Page(PageRequest),
    Manifest {
        output_dir: std::path::PathBuf,
        input_count: usize,
        output_count: usize,
        #[serde(default)]
        failure_count: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    let request: Request = serde_json::from_str(&text)?;
    let request = match request {
        Request::Page(page) => page,
        Request::Manifest {
            output_dir,
            input_count,
            output_count,
            failure_count,
        } => {
            if output_dir.file_name().and_then(|name| name.to_str()) == Some("6_concatenated") {
                let root = output_dir
                    .parent()
                    .ok_or("output directory has no dataset parent")?;
                kern_parser::dataset_manifest::update_dataset_manifest(
                    root,
                    kern_parser::dataset_manifest::DatasetComponentUpdate {
                        name: "concatenation_provenance".to_owned(),
                        producer: "pipelines.datasets.generation.continuations".to_owned(),
                        paths: vec![
                            output_dir.join("concatenation_manifest.jsonl"),
                            output_dir.join("continuations.json"),
                        ],
                        summary: std::collections::BTreeMap::from([
                            ("input_count".to_owned(), input_count),
                            ("output_count".to_owned(), output_count),
                            ("failure_count".to_owned(), failure_count),
                        ]),
                    },
                )?;
            }
            println!("{}", json!({"status":"success"}));
            return Ok(());
        }
    };
    let input_text = fs::read_to_string(&request.input)?;
    let mut page = KernDocument::parse(&input_text)?;
    let mut preserved = BTreeSet::new();
    let before = printed_staffs(&page)?;
    if request.start_bar_index > 0 {
        let source = request
            .source
            .ok_or("natural continuations require their unsplit source")?;
        let mut original = KernDocument::parse(&fs::read_to_string(source)?)?;
        CompactMultiMeasureRests::new(Default::default()).normalize(&mut original);
        let count = musical_bar_count(&original);
        let end = request.end_bar_index.unwrap_or(count);
        if end > count || request.start_bar_index >= end {
            return Err("invalid continuation bar range".into());
        }
        if end < count {
            original = split_trailing_bars_with_prefix_policy(
                &original,
                count - end,
                SlicePrefixPolicy::StandaloneExcerpt,
            )?
            .retained;
        }
        let remove_last = end - request.start_bar_index;
        let hidden = split_trailing_bars_with_prefix_policy(
            &original,
            remove_last,
            SlicePrefixPolicy::PageRendering {
                is_first_page: false,
            },
        )?
        .removed;
        // The prefix policy leaves a source-authored declaration at a cut visible.
        preserved = printed_staffs(&hidden)?;
    }
    let hidden_staffs: Vec<_> = before.difference(&preserved).copied().collect();
    if !hidden_staffs.is_empty() {
        hide_opening_meter_except_staffs(&mut page, &preserved)?;
    }
    let output_text = if hidden_staffs.is_empty() {
        input_text.clone()
    } else {
        SerializePreset::HumdrumKern.try_serialize_document(&page)?
    };
    let changed = output_text != input_text;
    if !request.preflight {
        fs::write(
            request.output.ok_or("page output is required")?,
            output_text,
        )?;
    }
    println!(
        "{}",
        json!({"status":"success", "carried_meter_hidden":!hidden_staffs.is_empty(), "hidden_staffs":hidden_staffs, "changed":changed})
    );
    Ok(())
}
