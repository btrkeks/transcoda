use crate::normalization::{invisible_rest_slices::InvisibleRestSliceError, NormalizeOptions};
use crate::serialize::bar_numbers::BarNumberPlan;
use crate::serialize::tandem_policy::{DialectError, TandemPolicy};
use crate::serialize::{ContentKind, ContentPolicy, SerializeOptions};
use crate::types::*;

use crate::{
    document::Framing,
    score::{LayerId, Score},
};

use super::{
    serialize_barline_line_with_policy, serialize_data_line, serialize_spine_manipulator_lines,
    serialize_synthetic_barline_line, serialize_system_break, serialize_tandem_line,
    ArticulationStyle, BarlineEmissionPolicy, BeamStyle, BowingStyle, ChordDurationStyle,
    ChordNoteOrder, CrossStaffStyle, InvisibleNoteStyle, InvisibleRestModifierStyle,
    InvisibleRestStyle, NoteModifierStyle, NullDataTokenStyle, OmrSerializeOptions, OrnamentStyle,
    RestPlacementStyle, SlurPlacementStyle, SpineTopologyStyle, StemDirectionStyle,
    SystemBreakStyle, TerminalLineStyle, TiePlacementStyle, TrillWavyLineStyle,
    VisibilityMarkerStyle,
};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OmrSerializer {
    options: OmrSerializeOptions,
}

impl OmrSerializer {
    pub fn new(options: OmrSerializeOptions) -> Self {
        Self { options }
    }

    /// Serialize to the strict OMR-facing `**kern` subset.
    ///
    /// Slur serialization should be stricter than parsing. In Humdrum/VHV,
    /// multiple-stop slurs are chord-level, not notehead-local: VHV's chord
    /// docs say rhythms and articulations are duplicated per note, but slurs,
    /// fermatas, and beams are not. Therefore duplicate ordinary `(` or `)`
    /// markers in a chord are not visually distinguishable by themselves.
    ///
    /// Preserve `Slur::count` while resolving document-level spans. Multiple
    /// same-onset slurs are meaningful when they close at distinct later
    /// events, or otherwise produce distinct start/end pairs. Collapse only
    /// duplicate resolved spans with the same start and same end; strict OMR
    /// serialization should emit those redundant spans only once.
    pub fn serialize_document(&self, document: &KernDocument) -> String {
        self.try_serialize_document(document)
            .expect("OMR serialization failed")
    }

    pub fn try_serialize_document(
        &self,
        document: &KernDocument,
    ) -> Result<String, InvisibleRestSliceError> {
        // Normalization leaves out the tandems the policy does not carry; a
        // slot the policy rejects must hold its default first.
        self.options
            .tandem_policy
            .check(&Score::resolve(document, Framing::Excerpt).map_err(DialectError::from)?)?;
        let mut document = crate::normalization::omr::normalized_for_omr_with_policy(
            document,
            &NormalizeOptions {
                compact_multi_measure_rests: self.options.compact_multi_measure_rests,
                preserve_leading_barlines: self.options.preserve_leading_barlines,
                // Keep this in sync with every normalization pass. `normalize`
                // is the dataset path, and all OMR normalizations should run there.
                canonicalize_measure_grid: matches!(
                    self.options.spine_topology_style,
                    SpineTopologyStyle::MeasureGrid
                ),
                project_to_emitted_kern: matches!(
                    self.options.spine_topology_style,
                    SpineTopologyStyle::MeasureGrid
                ),
                ..Default::default()
            },
            self.options.tandem_policy,
        );
        match self.options.invisible_rest_style {
            InvisibleRestStyle::DurationBearing => {
                crate::normalization::invisible_rest_slices::rehydrate_durationless_invisible_rest_slices_if_present(
                    &mut document,
                )?;
                if self.options.compact_consecutive_invisible_rests {
                    // Compaction silently skips documents with inconsistent
                    // duration state; presets that request it must keep
                    // failing fast on such documents instead.
                    crate::analysis::score_line_durations(&resolve(&document)?)?;
                    crate::normalization::invisible_rest_compaction::compact_consecutive_invisible_rests(
                        &mut document,
                    );
                }
            }
            InvisibleRestStyle::DurationlessSlices => {
                crate::normalization::invisible_rest_slices::project_durationless_invisible_rest_slices(
                    &mut document,
                )?;
            }
        }
        Ok(self.serialize_prepared_document(&resolve(&document)?))
    }

    pub(crate) fn serialize_prepared_document(&self, score: &Score<'_>) -> String {
        let document = score.document();
        let topology = score.topology();
        let spine_types = |layers: &[LayerId]| {
            layers
                .iter()
                .map(|&layer| topology.spine_type(layer).clone())
                .collect::<Vec<_>>()
        };
        let bar_number_plan = BarNumberPlan::new(document, self.options.bar_number_style);
        let mut output = Vec::new();

        for (line_index, line) in document.lines().iter().enumerate() {
            let active_spines = spine_types(topology.layers_at(line_index));
            match line {
                KernLine::SpineTypeLine(_) => {}
                KernLine::TandemInterpretationLine(tandems) => {
                    let Some(serialized) =
                        serialize_tandem_line(tandems.tokens(), &active_spines, &self.options)
                    else {
                        continue;
                    };

                    output.push(serialized);
                }
                KernLine::SpineManipulatorLine(manipulators) => {
                    output.extend(serialize_spine_manipulator_lines(
                        manipulators.tokens(),
                        &active_spines,
                        &spine_types(topology.layers_after(line_index)),
                    ));
                }
                KernLine::LocalCommentLine(_) => {}
                KernLine::Barline(barline) => {
                    if matches!(self.options.system_break_style, SystemBreakStyle::Preserve) {
                        if let Some(system_break) = &barline.system_break_before {
                            output.push(serialize_system_break(system_break));
                        }
                    }
                    if let Some(serialized) = serialize_barline_line_with_policy(
                        barline,
                        &active_spines,
                        bar_number_plan.barline_number(line_index),
                        self.options.barline_emission_policy,
                        self.options.mark_placement_style,
                    ) {
                        output.push(serialized);
                    }
                }
                KernLine::Data(contents) => {
                    if let Some(serialized) = serialize_synthetic_barline_line(
                        &active_spines,
                        bar_number_plan.insertion_before(line_index),
                        self.options.barline_emission_policy,
                    ) {
                        output.push(serialized);
                    }
                    if let Some(serialized) =
                        serialize_data_line(contents, &active_spines, &self.options)
                    {
                        output.push(serialized);
                    }
                }
            }
        }

        finalize_output(&mut output, &self.options);

        if output.is_empty() {
            String::new()
        } else {
            format!("{}\n", output.join("\n"))
        }
    }
}

/// The score of a document the serializer prepared. Normalization keeps a
/// resolvable document resolvable, so a failure here names the input's
/// state error.
fn resolve(document: &KernDocument) -> Result<Score<'_>, InvisibleRestSliceError> {
    Score::resolve(document, Framing::Excerpt)
        .map_err(|errors| InvisibleRestSliceError::Dialect(errors.into()))
}

pub fn to_string(document: &KernDocument) -> String {
    OmrSerializer::default().serialize_document(document)
}

#[allow(dead_code)]
pub(crate) fn to_compact_full_visual_string(document: &KernDocument) -> String {
    to_compact_full_visual_string_with_options(document, SerializeOptions::default())
}

#[allow(dead_code)]
pub(crate) fn to_compact_full_visual_string_with_options(
    document: &KernDocument,
    options: SerializeOptions,
) -> String {
    OmrSerializer::new(OmrSerializeOptions {
        chord_duration_style: ChordDurationStyle::FirstOnly,
        chord_note_order: ChordNoteOrder::Pitch,
        terminal_line_style: TerminalLineStyle::OmitFinal,
        note_modifier_style: NoteModifierStyle::Preserve,
        slur_placement_style: SlurPlacementStyle::Omit,
        tie_placement_style: TiePlacementStyle::Omit,
        mark_placement_style: super::MarkPlacementStyle::Omit,
        rest_placement_style: RestPlacementStyle::Omit,
        beam_style: BeamStyle::Preserve,
        articulation_style: ArticulationStyle::Preserve,
        ornament_style: OrnamentStyle::Preserve,
        trill_wavy_line_style: TrillWavyLineStyle::Omit,
        bowing_style: BowingStyle::Preserve,
        stem_direction_style: StemDirectionStyle::Omit,
        cross_staff_style: CrossStaffStyle::Preserve,
        visibility_marker_style: VisibilityMarkerStyle::Preserve,
        invisible_note_style: InvisibleNoteStyle::Preserve,
        invisible_rest_style: InvisibleRestStyle::DurationBearing,
        invisible_rest_modifier_style: InvisibleRestModifierStyle::Preserve,
        spine_topology_style: SpineTopologyStyle::AsWritten,
        tandem_policy: TandemPolicy::Omr,
        system_break_style: SystemBreakStyle::Omit,
        compact_multi_measure_rests: true,
        compact_consecutive_invisible_rests: false,
        preserve_leading_barlines: false,
        barline_emission_policy: BarlineEmissionPolicy::PerActiveSpine,
        null_data_token_style: NullDataTokenStyle::Dot,
        bar_number_style: options.bar_number_style,
    })
    .serialize_document(document)
}

pub fn to_transcoda_v2_string(document: &KernDocument) -> String {
    to_transcoda_v2_string_with_options(document, SerializeOptions::default())
}

pub fn to_transcoda_v2_string_with_options(
    document: &KernDocument,
    options: SerializeOptions,
) -> String {
    try_to_transcoda_v2_string_with_options(document, options)
        .expect("Transcoda V2 serialization should succeed")
}

pub fn try_to_transcoda_v2_string_with_options(
    document: &KernDocument,
    options: SerializeOptions,
) -> Result<String, InvisibleRestSliceError> {
    crate::serialize::transcoda_v2::try_serialize_transcoda_v2(
        document,
        crate::serialize::transcoda_v2::TranscodaV2Syntax::PerSpineBarlinesDotNulls,
        ContentPolicy::TranscodaV2,
        options,
    )
}

pub fn to_transcoda_v2_empty_nulls_string(document: &KernDocument) -> String {
    to_transcoda_v2_empty_nulls_string_with_options(document, SerializeOptions::default())
}

pub fn to_transcoda_v2_empty_nulls_string_with_options(
    document: &KernDocument,
    options: SerializeOptions,
) -> String {
    try_to_transcoda_v2_empty_nulls_string_with_options(document, options)
        .expect("Transcoda V2 serialization should succeed")
}

pub fn try_to_transcoda_v2_empty_nulls_string_with_options(
    document: &KernDocument,
    options: SerializeOptions,
) -> Result<String, InvisibleRestSliceError> {
    crate::serialize::transcoda_v2::try_serialize_transcoda_v2(
        document,
        crate::serialize::transcoda_v2::TranscodaV2Syntax::PerSpineBarlinesEmptyNulls,
        ContentPolicy::TranscodaV2,
        options,
    )
}

pub fn to_transcoda_v2_compact_barlines_string(document: &KernDocument) -> String {
    to_transcoda_v2_compact_barlines_string_with_options(document, SerializeOptions::default())
}

pub fn to_transcoda_v2_compact_barlines_string_with_options(
    document: &KernDocument,
    options: SerializeOptions,
) -> String {
    try_to_transcoda_v2_compact_barlines_string_with_options(document, options)
        .expect("Transcoda V2 serialization should succeed")
}

pub fn try_to_transcoda_v2_compact_barlines_string_with_options(
    document: &KernDocument,
    options: SerializeOptions,
) -> Result<String, InvisibleRestSliceError> {
    crate::serialize::transcoda_v2::try_serialize_transcoda_v2(
        document,
        crate::serialize::transcoda_v2::TranscodaV2Syntax::CompactBarlinesDotNulls,
        ContentPolicy::TranscodaV2,
        options,
    )
}

pub fn to_transcoda_v2_compact_barlines_empty_nulls_string(document: &KernDocument) -> String {
    to_transcoda_v2_compact_barlines_empty_nulls_string_with_options(
        document,
        SerializeOptions::default(),
    )
}

pub fn to_transcoda_v2_compact_barlines_empty_nulls_string_with_options(
    document: &KernDocument,
    options: SerializeOptions,
) -> String {
    try_to_transcoda_v2_compact_barlines_empty_nulls_string_with_options(document, options)
        .expect("Transcoda V2 serialization should succeed")
}

pub fn try_to_transcoda_v2_compact_barlines_empty_nulls_string_with_options(
    document: &KernDocument,
    options: SerializeOptions,
) -> Result<String, InvisibleRestSliceError> {
    crate::serialize::transcoda_v2::try_serialize_transcoda_v2(
        document,
        crate::serialize::transcoda_v2::TranscodaV2Syntax::CompactBarlinesEmptyNulls,
        ContentPolicy::TranscodaV2,
        options,
    )
}

/// Transcoda v2 shape with the retired durationless slice projection,
/// kept so projection coverage stays independent of the current preset.
#[cfg(test)]
pub(crate) fn to_transcoda_v2_durationless_slices_string(document: &KernDocument) -> String {
    OmrSerializer::new(OmrSerializeOptions {
        invisible_rest_style: InvisibleRestStyle::DurationlessSlices,
        compact_consecutive_invisible_rests: false,
        ..transcoda_v2_options(
            ContentPolicy::TranscodaV2,
            SerializeOptions::default(),
            BarlineEmissionPolicy::Compact,
            NullDataTokenStyle::Empty,
        )
    })
    .serialize_document(document)
}

/// Printer options for a Transcoda V2 dialect.
///
/// Lowering projects the document through `policy` before printing, so every
/// content style writes what is left; the table only decides the layout
/// rewrites here.
pub(crate) fn transcoda_v2_options(
    policy: ContentPolicy,
    options: SerializeOptions,
    barline_emission_policy: BarlineEmissionPolicy,
    null_data_token_style: NullDataTokenStyle,
) -> OmrSerializeOptions {
    let rewrites = |kind| policy.rewrites(kind);
    let drops = |kind| policy.drops(kind);
    OmrSerializeOptions {
        chord_duration_style: if rewrites(ContentKind::ChordDuration) {
            ChordDurationStyle::FirstOnly
        } else {
            ChordDurationStyle::Repeated
        },
        chord_note_order: if rewrites(ContentKind::ChordNoteOrder) {
            ChordNoteOrder::Pitch
        } else {
            ChordNoteOrder::Source
        },
        terminal_line_style: if drops(ContentKind::TerminalLine) {
            TerminalLineStyle::OmitFinal
        } else {
            TerminalLineStyle::Emit
        },
        note_modifier_style: NoteModifierStyle::Preserve,
        slur_placement_style: SlurPlacementStyle::Preserve,
        tie_placement_style: TiePlacementStyle::Preserve,
        // `Omit` writes the default duration on grace notes written without
        // one; the projection has already removed every other marker it
        // would leave out.
        mark_placement_style: if rewrites(ContentKind::GraceDuration) {
            super::MarkPlacementStyle::Omit
        } else {
            super::MarkPlacementStyle::Preserve
        },
        rest_placement_style: RestPlacementStyle::Preserve,
        beam_style: BeamStyle::Preserve,
        articulation_style: ArticulationStyle::Preserve,
        ornament_style: OrnamentStyle::Preserve,
        trill_wavy_line_style: TrillWavyLineStyle::Preserve,
        bowing_style: BowingStyle::Preserve,
        stem_direction_style: StemDirectionStyle::Preserve,
        cross_staff_style: CrossStaffStyle::Preserve,
        visibility_marker_style: VisibilityMarkerStyle::Preserve,
        invisible_note_style: InvisibleNoteStyle::Preserve,
        invisible_rest_style: InvisibleRestStyle::DurationBearing,
        invisible_rest_modifier_style: InvisibleRestModifierStyle::Preserve,
        spine_topology_style: if rewrites(ContentKind::SpineSlots) {
            SpineTopologyStyle::MeasureGrid
        } else {
            SpineTopologyStyle::AsWritten
        },
        tandem_policy: policy.tandem_policy(),
        system_break_style: if drops(ContentKind::SystemBreak) {
            SystemBreakStyle::Omit
        } else {
            SystemBreakStyle::Preserve
        },
        compact_multi_measure_rests: rewrites(ContentKind::MultiMeasureRest),
        compact_consecutive_invisible_rests: rewrites(ContentKind::InvisibleRestRun),
        preserve_leading_barlines: !drops(ContentKind::LeadingBarline),
        barline_emission_policy,
        null_data_token_style,
        bar_number_style: options.bar_number_style,
    }
}

#[allow(dead_code)]
pub(crate) fn to_compact_string(document: &KernDocument) -> String {
    to_compact_full_visual_string(document)
}

#[allow(dead_code)]
pub(crate) fn to_compact_no_note_modifiers_string(document: &KernDocument) -> String {
    to_transcoda_v2_string(document)
}

fn finalize_output(output: &mut Vec<String>, options: &OmrSerializeOptions) {
    if matches!(options.terminal_line_style, TerminalLineStyle::OmitFinal) {
        while output
            .last()
            .is_some_and(|line| is_all_terminal_manipulator_line(line))
        {
            output.pop();
        }
    }
}

fn is_all_terminal_manipulator_line(line: &str) -> bool {
    line.split('\t').all(|token| token == "*-")
}
