use kern_parser::{
    concatenation::concatenate_documents,
    normalization::{multi_measure_rests::CompactMultiMeasureRests, NormalizationPass},
    serialize::SerializePreset,
    transforms::{musical_bar_count, split_trailing_bars},
    KernDocument,
};
use serde_json::Value;
use std::{
    fs,
    io::{self, Read},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let request: Value = serde_json::from_str(&input)?;
    let parts = request["parts"]
        .as_array()
        .ok_or("missing parts")?
        .iter()
        .map(|part| {
            let path = part
                .as_str()
                .or_else(|| part["path"].as_str())
                .ok_or("invalid path")?;
            let mut document = KernDocument::parse(&fs::read_to_string(path)?)?;
            if let Some(start) = part["start"].as_u64() {
                CompactMultiMeasureRests::new(Default::default()).normalize(&mut document);
                let end = part["end"].as_u64().ok_or("missing end")? as usize;
                let count = musical_bar_count(&document);
                if end > count {
                    return Err(format!("Frozen range {start}..{end} exceeds {count} bars").into());
                }
                if end < count {
                    document = split_trailing_bars(&document, count - end)?.retained;
                }
                if start > 0 {
                    document = split_trailing_bars(&document, end - start as usize)?.removed;
                }
            }
            Ok(document)
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let document = match parts.as_slice() {
        [document] => document.clone(),
        _ => concatenate_documents(&parts)?,
    };
    let text = SerializePreset::HumdrumKern.try_serialize_document(&document)?;
    fs::write(request["output"].as_str().ok_or("missing output")?, text)?;
    Ok(())
}
