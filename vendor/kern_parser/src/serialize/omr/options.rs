use crate::serialize::{bar_numbers::BarNumberStyle, tandem_policy::TandemPolicy};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChordDurationStyle {
    #[default]
    Repeated,
    FirstOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChordNoteOrder {
    Source,
    #[default]
    Pitch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalLineStyle {
    #[default]
    Emit,
    OmitFinal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NoteModifierStyle {
    #[default]
    Preserve,
    OmitSlursTiesAndForcedAccidentals,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlurPlacementStyle {
    #[default]
    Omit,
    Preserve,
}

/// RDF `<`/`>` after fermatas, articulations, ornaments, bowing, breath and
/// harmonic marks.
///
/// The standard Humdrum targets are exactly the targets that preserve mark
/// placement, so `Preserve` also writes what only they carry: breath marks
/// (`,`), harmonics (`o`), phrases (`{`, `}`), editorial `x`, every user
/// signifier, grace notes without a duration, and ornaments in source order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarkPlacementStyle {
    #[default]
    Omit,
    Preserve,
}

impl MarkPlacementStyle {
    pub(crate) fn writes_every_marker(self) -> bool {
        matches!(self, Self::Preserve)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TiePlacementStyle {
    #[default]
    Omit,
    Preserve,
}

/// Whether rests keep the pitch letters that place them on the staff (`4rB`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RestPlacementStyle {
    #[default]
    Omit,
    Preserve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BeamStyle {
    #[default]
    Preserve,
    Omit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArticulationStyle {
    #[default]
    Omit,
    Preserve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrnamentStyle {
    #[default]
    Preserve,
    OmitConcludingTurn,
    OmitAll,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrillWavyLineStyle {
    #[default]
    Omit,
    Preserve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BowingStyle {
    #[default]
    Preserve,
    Omit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StemDirectionStyle {
    Preserve,
    #[default]
    Omit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrossStaffStyle {
    #[default]
    Preserve,
    Omit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisibilityMarkerStyle {
    #[default]
    Preserve,
    Omit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InvisibleRestStyle {
    #[default]
    DurationBearing,
    DurationlessSlices,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InvisibleNoteStyle {
    Preserve,
    #[default]
    StripMarker,
    Omit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InvisibleRestModifierStyle {
    #[default]
    Preserve,
    Omit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpineTopologyStyle {
    #[default]
    AsWritten,
    MeasureGrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SystemBreakStyle {
    #[default]
    Omit,
    Preserve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BarlineEmissionPolicy {
    #[default]
    PerActiveSpine,
    Compact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NullDataTokenStyle {
    #[default]
    Dot,
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OmrSerializeOptions {
    pub chord_duration_style: ChordDurationStyle,
    pub chord_note_order: ChordNoteOrder,
    pub terminal_line_style: TerminalLineStyle,
    pub note_modifier_style: NoteModifierStyle,
    pub slur_placement_style: SlurPlacementStyle,
    pub tie_placement_style: TiePlacementStyle,
    pub mark_placement_style: MarkPlacementStyle,
    pub rest_placement_style: RestPlacementStyle,
    pub beam_style: BeamStyle,
    pub articulation_style: ArticulationStyle,
    pub ornament_style: OrnamentStyle,
    pub trill_wavy_line_style: TrillWavyLineStyle,
    pub bowing_style: BowingStyle,
    pub stem_direction_style: StemDirectionStyle,
    pub cross_staff_style: CrossStaffStyle,
    pub visibility_marker_style: VisibilityMarkerStyle,
    pub invisible_note_style: InvisibleNoteStyle,
    pub invisible_rest_style: InvisibleRestStyle,
    pub invisible_rest_modifier_style: InvisibleRestModifierStyle,
    pub spine_topology_style: SpineTopologyStyle,
    pub tandem_policy: TandemPolicy,
    pub system_break_style: SystemBreakStyle,
    pub compact_multi_measure_rests: bool,
    pub compact_consecutive_invisible_rests: bool,
    pub preserve_leading_barlines: bool,
    pub barline_emission_policy: BarlineEmissionPolicy,
    pub null_data_token_style: NullDataTokenStyle,
    pub bar_number_style: BarNumberStyle,
}
