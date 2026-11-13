use crate::serialize::barline::{serialize_barline_parts, BarlineSpelling};
use crate::types::*;

use crate::serialize::tandem_policy::{TandemAction, TandemPolicy};

use super::{
    serialize_kern_content, BarlineEmissionPolicy, MarkPlacementStyle, OmrSerializeOptions,
    VisibilityMarkerStyle,
};

fn is_kern_spine(active_spines: &[SpineType], index: usize) -> bool {
    active_spines
        .get(index)
        .is_some_and(|spine_type| matches!(spine_type, SpineType::Kern))
}

pub(crate) fn serialize_tandem_line(
    tandems: &[TandemInterpretation],
    active_spines: &[SpineType],
    options: &OmrSerializeOptions,
) -> Option<String> {
    let tokens = tandems
        .iter()
        .enumerate()
        .filter(|(index, _)| is_kern_spine(active_spines, *index))
        .map(|(_, tandem)| serialize_tandem(tandem, options).unwrap_or("*".to_owned()))
        .collect::<Vec<_>>();

    non_null_line(tokens, "*")
}

pub(crate) fn serialize_tandem(
    tandem: &TandemInterpretation,
    options: &OmrSerializeOptions,
) -> Option<String> {
    if options.tandem_policy.action_for(tandem) != TandemAction::Keep {
        return None;
    }
    // The frozen V1 target predates cancellation naturals, and the
    // Transcoda V2 tokenizer only spells whole keys like `*k[f#c#]`.
    if let (TandemPolicy::TranscodaV2, TandemInterpretation::KeySignature(key_signature)) =
        (options.tandem_policy, tandem)
    {
        return Some(key_signature.without_naturals().as_kern());
    }
    spell_tandem(tandem, options.visibility_marker_style)
}

/// `**kern` spelling of a tandem. `None` for kinds that have no spelling
/// (key designation, part, group; no policy keeps them) and for a
/// meter symbol that only hides a time signature when visibility markers
/// are omitted.
pub(crate) fn spell_tandem(
    tandem: &TandemInterpretation,
    visibility_marker_style: VisibilityMarkerStyle,
) -> Option<String> {
    Some(match tandem {
        TandemInterpretation::Null => "*".to_owned(),
        TandemInterpretation::Clef(clef) => serialize_clef(clef, visibility_marker_style),
        TandemInterpretation::KeySignature(key_signature) => key_signature.as_kern(),
        TandemInterpretation::TimeSignature(time_signature) => {
            serialize_time_signature(*time_signature, visibility_marker_style)
        }
        TandemInterpretation::FreeMeter => "*MX".to_owned(),
        TandemInterpretation::MeterSymbol(MeterSymbol::Empty)
            if matches!(visibility_marker_style, VisibilityMarkerStyle::Omit) =>
        {
            return None
        }
        TandemInterpretation::MeterSymbol(symbol) => symbol.as_kern().to_owned(),
        TandemInterpretation::Ottava(ottava) => format!(
            "*{}{}",
            if ottava.active { "" } else { "X" },
            match ottava.kind {
                OttavaKind::OctaveUp => "8va",
                OttavaKind::OctaveDown => "8ba",
                OttavaKind::TwoOctavesUp => "15ma",
                OttavaKind::TwoOctavesDown => "15ba",
            }
        ),
        TandemInterpretation::Pedal(pedal) => {
            if pedal.active { "*ped" } else { "*Xped" }.to_owned()
        }
        TandemInterpretation::Tremolo(tremolo) => if tremolo.active {
            "*tremolo"
        } else {
            "*Xtremolo"
        }
        .to_owned(),
        TandemInterpretation::Tuplet(tuplet) => if tuplet.show_number {
            "*tuplet"
        } else {
            "*Xtuplet"
        }
        .to_owned(),
        TandemInterpretation::TupletBracket(bracket) => if bracket.show_bracket {
            "*brackettup"
        } else {
            "*Xbrackettup"
        }
        .to_owned(),
        TandemInterpretation::Repetition(repetition) => {
            if repetition.active { "*rep" } else { "*Xrep" }.to_owned()
        }
        TandemInterpretation::TupletBreak => "*tupbreak".to_owned(),
        TandemInterpretation::Stria(stria) => format!("*stria{}", stria.count),
        TandemInterpretation::BracketSpan(bracket_span) => format!(
            "*{}{}",
            if bracket_span.active { "" } else { "X" },
            match bracket_span.kind {
                BracketSpanKind::Ligature => "lig",
                BracketSpanKind::Coloration => "col",
            }
        ),
        TandemInterpretation::SectionLabel(section_label) => format!("*>{}", section_label.label),
        TandemInterpretation::ExpansionList(expansion_list) => {
            serialize_expansion_list(expansion_list)
        }
        TandemInterpretation::VisualStyle(visual_style) => match visual_style {
            VisualStyle::Color([r, g, b]) => format!("*color:#{r:02X}{g:02X}{b:02X}"),
            VisualStyle::StaffPlacement(StaffPlacement::Above) => "*above".to_owned(),
            VisualStyle::StaffPlacement(StaffPlacement::Below) => "*below".to_owned(),
            VisualStyle::KeyCancellation(cancellation) => cancellation.as_kern().to_owned(),
            VisualStyle::HalfNoteStemSide(HalfNoteStemSide::Left) => "*2\\left".to_owned(),
            VisualStyle::HalfNoteStemSide(HalfNoteStemSide::Right) => "*2\\right".to_owned(),
        },
        TandemInterpretation::Cue(cue) => cue.as_kern().to_owned(),
        TandemInterpretation::RhythmScale(scale) => scale.as_kern(),
        TandemInterpretation::Instrument(instrument) => {
            let prefix = match instrument.kind {
                InstrumentKind::Code => "*I",
                InstrumentKind::Name(InstrumentTextVariant::Regular) => "*I\"",
                InstrumentKind::Name(InstrumentTextVariant::Modern) => "*mI\"",
                InstrumentKind::Name(InstrumentTextVariant::Original) => "*oI\"",
                InstrumentKind::Abbreviation(InstrumentTextVariant::Regular) => "*I'",
                InstrumentKind::Abbreviation(InstrumentTextVariant::Modern) => "*mI'",
                InstrumentKind::Abbreviation(InstrumentTextVariant::Original) => "*oI'",
            };
            format!("{prefix}{}", instrument.text)
        }
        TandemInterpretation::Tempo(tempo) => format!(
            "*MM{}",
            serialize_decimal_tempo(tempo.beats_per_minute.units, tempo.beats_per_minute.scale)
        ),
        TandemInterpretation::Raw(raw) => raw.0.clone(),
        TandemInterpretation::Staff(staff) => staff.as_kern(),
        TandemInterpretation::KeyDesignation(_)
        | TandemInterpretation::Part(_)
        | TandemInterpretation::Group(_) => return None,
    })
}

fn serialize_decimal_tempo(units: u32, scale: u8) -> String {
    if scale == 0 {
        return units.to_string();
    }
    let mut digits = units.to_string();
    let scale = scale as usize;
    if digits.len() <= scale {
        digits = format!("{}{}", "0".repeat(scale + 1 - digits.len()), digits);
    }
    let split = digits.len() - scale;
    format!("{}.{}", &digits[..split], &digits[split..])
}

fn serialize_expansion_list(expansion_list: &ExpansionList) -> String {
    let version = expansion_list.version.as_deref().unwrap_or("");
    let items = expansion_list
        .items
        .iter()
        .map(|item| match item {
            ExpansionItem::Label(label) => label.as_str(),
            ExpansionItem::Empty => "",
        })
        .collect::<Vec<_>>()
        .join(",");

    format!("*>{version}[{items}]")
}

pub(crate) fn serialize_system_break(system_break: &SystemBreak) -> String {
    match &system_break.group {
        Some(group) => format!("!!LO:LB:g={group}"),
        None => "!!LO:LB".to_owned(),
    }
}

fn serialize_time_signature(
    mut time_signature: TimeSignatureTandem,
    visibility_marker_style: VisibilityMarkerStyle,
) -> String {
    if matches!(visibility_marker_style, VisibilityMarkerStyle::Omit) {
        time_signature.invisible = false;
    }
    time_signature.as_kern()
}

fn serialize_clef(clef: &Clef, visibility_marker_style: VisibilityMarkerStyle) -> String {
    let mut serialized = clef.clone();
    if matches!(visibility_marker_style, VisibilityMarkerStyle::Omit) {
        serialized.invisible = false;
    }
    serialized.as_kern()
}

/// `active_spines` and `spines_after` are the spine types the line
/// manipulates and the ones it leaves.
pub(crate) fn serialize_spine_manipulator_lines(
    manipulators: &[SpineManipulator],
    active_spines: &[SpineType],
    spines_after: &[SpineType],
) -> Vec<String> {
    let direct_tokens = manipulators
        .iter()
        .enumerate()
        .filter(|(index, _)| is_kern_spine(active_spines, *index))
        .map(|(_, manipulator)| manipulator.clone())
        .collect::<Vec<_>>();
    let kern_after = spines_after
        .iter()
        .filter(|spine_type| matches!(spine_type, SpineType::Kern))
        .count();

    if SpineManipulatorLine::new(direct_tokens.clone()).is_ok()
        && manipulated_width(&direct_tokens) == kern_after
    {
        return non_null_line(
            direct_tokens
                .iter()
                .map(serialize_spine_manipulator)
                .collect(),
            "*",
        )
        .into_iter()
        .collect();
    }

    if manipulators
        .iter()
        .any(|manipulator| matches!(manipulator, SpineManipulator::Merge))
    {
        return serialize_merge_manipulator_lines(manipulators, active_spines);
    }

    let tokens = manipulators
        .iter()
        .enumerate()
        .filter(|(index, _)| is_kern_spine(active_spines, *index))
        .map(|(_, manipulator)| serialize_spine_manipulator(manipulator))
        .collect::<Vec<_>>();

    non_null_line(tokens, "*").into_iter().collect()
}

/// Width a valid manipulator line leaves: a run of merges becomes one spine.
fn manipulated_width(manipulators: &[SpineManipulator]) -> usize {
    manipulators
        .iter()
        .enumerate()
        .map(|(index, manipulator)| match manipulator {
            SpineManipulator::Null => 1,
            SpineManipulator::Split => 2,
            SpineManipulator::Terminate => 0,
            SpineManipulator::Merge => {
                usize::from(index == 0 || manipulators[index - 1] != SpineManipulator::Merge)
            }
        })
        .sum()
}

pub(crate) fn serialize_spine_manipulator(manipulator: &SpineManipulator) -> String {
    match manipulator {
        SpineManipulator::Null => "*",
        SpineManipulator::Terminate => "*-",
        SpineManipulator::Split => "*^",
        SpineManipulator::Merge => "*v",
    }
    .to_owned()
}

fn serialize_merge_manipulator_lines(
    manipulators: &[SpineManipulator],
    active_spines: &[SpineType],
) -> Vec<String> {
    let mut kern_positions = vec![None; manipulators.len()];
    let mut kern_width = 0;
    for (index, position) in kern_positions.iter_mut().enumerate() {
        if is_kern_spine(active_spines, index) {
            *position = Some(kern_width);
            kern_width += 1;
        }
    }

    let mut merge_groups = Vec::new();
    let mut index = 0;
    while index < manipulators.len() {
        if !matches!(manipulators[index], SpineManipulator::Merge) {
            index += 1;
            continue;
        }

        let start = index;
        while index < manipulators.len() && matches!(manipulators[index], SpineManipulator::Merge) {
            index += 1;
        }
        let positions = (start..index)
            .filter_map(|position| kern_positions[position])
            .collect::<Vec<_>>();
        if positions.len() > 1 {
            merge_groups.push((positions[0], positions.len()));
        }
    }

    // Emit merge groups left to right so the multi-line form matches the
    // canonical order `MergeAdjacentSpineManipulators` produces when the
    // emitted text is re-parsed; emitting right-first made the next round
    // trip swap the lines. Collapsing a left group shifts the kern positions
    // of the remaining groups, so track that shift explicitly.
    let mut output = Vec::new();
    let mut current_width = kern_width;
    let mut collapsed = 0usize;
    for (start, len) in merge_groups {
        let start: usize = start;
        let Some(start) = start.checked_sub(collapsed) else {
            continue;
        };
        if start + len > current_width {
            continue;
        }
        let mut tokens = vec!["*".to_owned(); current_width];
        for token in tokens.iter_mut().skip(start).take(len) {
            *token = "*v".to_owned();
        }
        if let Some(line) = non_null_line(tokens, "*") {
            output.push(line);
        }
        current_width -= len - 1;
        collapsed += len - 1;
    }

    output
}

pub(crate) fn serialize_barline_line_with_policy(
    barline: &Barline,
    active_spines: &[SpineType],
    bar_number: Option<u32>,
    policy: BarlineEmissionPolicy,
    placement_style: MarkPlacementStyle,
) -> Option<String> {
    let mut tokens = active_spines
        .iter()
        .enumerate()
        .filter(|(_, spine_type)| matches!(spine_type, SpineType::Kern))
        .map(|(index, _)| serialize_barline_for_spine(barline, index, bar_number, placement_style))
        .collect::<Vec<_>>();
    if matches!(policy, BarlineEmissionPolicy::Compact)
        && tokens.windows(2).all(|pair| pair[0] == pair[1])
    {
        tokens.truncate(1);
    }

    data_line(tokens)
}

fn serialize_barline_for_spine(
    barline: &Barline,
    spine_index: usize,
    bar_number: Option<u32>,
    placement_style: MarkPlacementStyle,
) -> String {
    serialize_barline_parts(
        barline.styles_for_spine(spine_index),
        barline.repeat_for_spine(spine_index),
        barline.fermata_for_spine(spine_index),
        bar_number,
        BarlineSpelling::Omr,
        placement_style,
    )
}

fn has_kern_spine(active_spines: &[SpineType]) -> bool {
    active_spines
        .iter()
        .any(|spine_type| matches!(spine_type, SpineType::Kern))
}

pub(crate) fn serialize_synthetic_barline_line(
    active_spines: &[SpineType],
    bar_number: Option<u32>,
    policy: BarlineEmissionPolicy,
) -> Option<String> {
    let bar_number = bar_number?;
    let token = format!("={bar_number}");
    let tokens = match policy {
        BarlineEmissionPolicy::PerActiveSpine => active_spines
            .iter()
            .filter(|spine_type| matches!(spine_type, SpineType::Kern))
            .map(|_| token.clone())
            .collect::<Vec<_>>(),
        BarlineEmissionPolicy::Compact => {
            if has_kern_spine(active_spines) {
                vec![token]
            } else {
                Vec::new()
            }
        }
    };

    data_line(tokens)
}

pub(crate) fn serialize_synthetic_render_barline_line(
    active_spines: &[SpineType],
    bar_number: Option<u32>,
    include_dynamics: bool,
    include_fingering: bool,
) -> Option<String> {
    let bar_number = bar_number?;
    let tokens = active_spines
        .iter()
        .filter(|spine_type| {
            matches!(spine_type, SpineType::Kern)
                || (include_dynamics && matches!(spine_type, SpineType::Dynam))
                || (include_fingering && matches!(spine_type, SpineType::Fing))
        })
        .map(|_| format!("={bar_number}"))
        .collect::<Vec<_>>();

    data_line(tokens)
}

pub(crate) fn serialize_data_line(
    contents: &[SpineContent],
    active_spines: &[SpineType],
    options: &OmrSerializeOptions,
) -> Option<String> {
    let tokens = contents
        .iter()
        .enumerate()
        .filter(|(index, _)| is_kern_spine(active_spines, *index))
        .filter_map(|(_, content)| match content {
            SpineContent::Kern(content) => Some(serialize_kern_content(content, options)),
            SpineContent::Dynam(_)
            | SpineContent::Fing(_)
            | SpineContent::Text(_)
            | SpineContent::MxmlHarmony(_)
            | SpineContent::Opaque { .. } => None,
        })
        .collect::<Vec<_>>();

    data_line(tokens)
}

fn non_null_line(tokens: Vec<String>, null_token: &str) -> Option<String> {
    if tokens.is_empty() || tokens.iter().all(|token| token == null_token) {
        None
    } else {
        Some(tokens.join("\t"))
    }
}

fn data_line(tokens: Vec<String>) -> Option<String> {
    if tokens.is_empty() || tokens.iter().all(String::is_empty) {
        None
    } else {
        Some(tokens.join("\t"))
    }
}
