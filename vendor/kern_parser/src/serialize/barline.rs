use crate::types::{BarlineFermata, BarlineStyle, FermataPlacement, RepeatStyle};

use super::omr::MarkPlacementStyle;

#[derive(Clone, Copy)]
pub(super) enum BarlineSpelling {
    Omr,
    Humdrum,
}

pub(super) fn serialize_barline_parts(
    styles: &[BarlineStyle],
    repeat: RepeatStyle,
    fermata: Option<BarlineFermata>,
    bar_number: Option<u32>,
    spelling: BarlineSpelling,
    placement_style: MarkPlacementStyle,
) -> String {
    let token = match repeat {
        RepeatStyle::StartAndEnd => match spelling {
            BarlineSpelling::Omr => "=:|:",
            BarlineSpelling::Humdrum => "=:|!|:",
        },
        RepeatStyle::Start => "=!|:",
        RepeatStyle::End => "=:|!",
        RepeatStyle::None => {
            if styles.contains(&BarlineStyle::Invisible) {
                "=-"
            } else if styles.contains(&BarlineStyle::Double) {
                "=="
            } else if styles.contains(&BarlineStyle::Thin) {
                "=||"
            } else if styles.contains(&BarlineStyle::Dashed) {
                "=:"
            } else {
                "="
            }
        }
    };
    let prefix = if token.starts_with("==") { "==" } else { "=" };
    let mut output = prefix.to_owned();
    if let Some(number) = bar_number {
        output.push_str(&number.to_string());
    }
    output.push_str(&token[prefix.len()..]);
    match fermata {
        Some(BarlineFermata::Single) => output.push(';'),
        Some(BarlineFermata::PlacedSingle(placement)) => {
            output.push(';');
            if matches!(placement_style, MarkPlacementStyle::Preserve) {
                output.push(match placement {
                    FermataPlacement::Above => '>',
                    FermataPlacement::Below => '<',
                });
            }
        }
        Some(BarlineFermata::Pair) => output.push_str(";;"),
        None => {}
    }
    output
}
