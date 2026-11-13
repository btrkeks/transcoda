//! Canonical single-token serialization.

use crate::{
    serialize::omr::{
        serialize_kern_content_preserving_arpeggio_scope, ArticulationStyle, BarlineEmissionPolicy,
        BeamStyle, BowingStyle, ChordDurationStyle, ChordNoteOrder, CrossStaffStyle,
        InvisibleNoteStyle, InvisibleRestModifierStyle, InvisibleRestStyle, NoteModifierStyle,
        NullDataTokenStyle, OmrSerializeOptions, OrnamentStyle, RestPlacementStyle,
        SlurPlacementStyle, SpineTopologyStyle, StemDirectionStyle, SystemBreakStyle,
        TerminalLineStyle, TiePlacementStyle, VisibilityMarkerStyle,
    },
    BarNumberStyle, KernContent,
};

/// Canonical standard-kern spelling of a single data token.
pub fn kern_token_string(content: &KernContent) -> String {
    serialize_kern_content_preserving_arpeggio_scope(content, &token_options())
}

fn token_options() -> OmrSerializeOptions {
    OmrSerializeOptions {
        chord_duration_style: ChordDurationStyle::Repeated,
        chord_note_order: ChordNoteOrder::Pitch,
        terminal_line_style: TerminalLineStyle::Emit,
        note_modifier_style: NoteModifierStyle::Preserve,
        slur_placement_style: SlurPlacementStyle::Preserve,
        tie_placement_style: TiePlacementStyle::Preserve,
        mark_placement_style: crate::serialize::omr::MarkPlacementStyle::Preserve,
        rest_placement_style: RestPlacementStyle::Preserve,
        beam_style: BeamStyle::Preserve,
        articulation_style: ArticulationStyle::Preserve,
        ornament_style: OrnamentStyle::Preserve,
        trill_wavy_line_style: crate::serialize::omr::TrillWavyLineStyle::Preserve,
        bowing_style: BowingStyle::Preserve,
        stem_direction_style: StemDirectionStyle::Preserve,
        cross_staff_style: CrossStaffStyle::Preserve,
        visibility_marker_style: VisibilityMarkerStyle::Preserve,
        invisible_note_style: InvisibleNoteStyle::Preserve,
        invisible_rest_style: InvisibleRestStyle::DurationlessSlices,
        invisible_rest_modifier_style: InvisibleRestModifierStyle::Preserve,
        spine_topology_style: SpineTopologyStyle::AsWritten,
        tandem_policy: crate::serialize::TandemPolicy::Omr,
        system_break_style: SystemBreakStyle::Omit,
        compact_multi_measure_rests: false,
        compact_consecutive_invisible_rests: false,
        preserve_leading_barlines: true,
        barline_emission_policy: BarlineEmissionPolicy::PerActiveSpine,
        null_data_token_style: NullDataTokenStyle::Dot,
        bar_number_style: BarNumberStyle::Omit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraceChord, InvisibleRestSlice};

    fn token(raw: &str) -> String {
        kern_token_string(&KernContent::Chord(raw.try_into().unwrap()))
    }

    #[test]
    fn serializes_note_level_modifiers_in_canonical_order() {
        assert_eq!(token("64cc##X>'LLL"), "64cc##X>'LLL");
        assert_eq!(token("4c#[yy"), "4c#[yy");
        assert_eq!(token("4d-y<"), "4d-y<");
    }

    #[test]
    fn writes_breath_and_harmonic_marks_with_their_placement() {
        assert_eq!(token("4c,<"), "4c,<");
        assert_eq!(token("4co>"), "4co>");
        assert_eq!(token("4c,o"), "4c,o");
        assert_eq!(token("4co>;<'>,"), "4c'>;<,o>");
        assert_eq!(
            kern_token_string(&KernContent::Grace("8qc,>o<".try_into().unwrap())),
            "8qc,>o<"
        );
        assert_eq!(
            kern_token_string(&KernContent::Rest("4r(,;".try_into().unwrap())),
            "4r;,("
        );
    }

    #[test]
    fn writes_phrases_with_their_elision_and_rest_phrase_opens_first() {
        assert_eq!(token("&{4c"), "4c&{");
        assert_eq!(token("4c{ 4e&&}"), "4c 4e{&&}");
        assert_eq!(token("4c({L"), "4c({L");
        assert_eq!(
            kern_token_string(&KernContent::Rest("({4r&}".try_into().unwrap())),
            "{4r(&}"
        );
    }

    #[test]
    fn writes_editorial_x_after_the_cross_staff_signifier() {
        assert_eq!(token("4cx#<"), "4c#<x");
        assert_eq!(token("4c#Xx<"), "4c#X<x");
        assert_eq!(token("4cx[<"), "4cx[<");
    }

    #[test]
    fn writes_every_user_signifier_last_in_its_subtoken() {
        assert_eq!(token("(8c#N/L]"), "8c#]/(LN");
        assert_eq!(token("4c[<Z"), "4c[<Z");
        assert_eq!(token("4cNZ;"), "4c;NZ");
        assert_eq!(token("4ci 4e[<j"), "4ci 4e[<j");
    }

    #[test]
    fn writes_grace_notes_without_a_duration() {
        let grace = |raw: &str| kern_token_string(&KernContent::Grace(raw.try_into().unwrap()));
        assert_eq!(grace("qe"), "qe");
        assert_eq!(grace("qc d"), "qc qd");
        assert_eq!(grace("8qc d"), "8qc 8qd");
    }

    #[test]
    fn writes_ornaments_in_source_order_and_one_copy_per_chord() {
        assert_eq!(token("4cMT"), "4cMT");
        assert_eq!(token("4cTM"), "4cTM");
        assert_eq!(token("4cT 4eT"), "4c 4eT");
    }

    #[test]
    fn serializes_chord_level_modifier_classes() {
        assert_eq!(token("8cSssT;'v/:((LL"), "8c'SssT;v/:((LL");
        assert_eq!(token("8cTT"), "8cTT");
        assert_eq!(token("8ctt"), "8ctt");
        assert_eq!(token("8cTTT"), "8cTTT");
        assert_eq!(token("8cttt"), "8cttt");
        assert_eq!(token("16e`~^zMkJ"), "16e`~^zMJk");
        assert_eq!(token("4c 4e 4g:"), "4c 4e 4g:");
        assert_eq!(token("4c 4e 4g::"), "4c 4e 4g::");
        assert_eq!(token("4c 4e 4g:::"), "4c 4e 4g::");
    }

    #[test]
    fn canonical_chord_serialization_orders_pitch() {
        let chord = KernContent::Chord("4g 4c 4e".try_into().unwrap());

        assert_eq!(kern_token_string(&chord), "4c 4e 4g");
    }

    #[test]
    fn serializes_mordent_spellings_canonically() {
        for raw in ["4cM", "4cm", "4cW", "4cw", "4cMM", "4cmm", "4cWW", "4cww"] {
            assert_eq!(token(raw), raw);
        }
        assert_eq!(token("4cMm"), "4cMm");
        assert_eq!(token("4cmM"), "4cMm");
        assert_eq!(token("4cWw"), "4cWw");
        assert_eq!(token("4cwW"), "4cWw");
    }

    #[test]
    fn serializes_grace_chords_and_rational_durations() {
        let grace = KernContent::Grace(GraceChord::try_from("16qqcc#X> 16qqdd").unwrap());
        assert_eq!(kern_token_string(&grace), "16qqcc#X> 16qqdd");
        let descending = KernContent::Grace(GraceChord::try_from("8qgg 8qcc").unwrap());
        assert_eq!(kern_token_string(&descending), "8qcc 8qgg");
        assert_eq!(token("3%5ff#"), "3%5ff#");
    }

    #[test]
    fn serializes_non_note_data_tokens() {
        assert_eq!(kern_token_string(&KernContent::Null), ".");
        assert_eq!(
            kern_token_string(&KernContent::Rest("8reyy;".try_into().unwrap())),
            "8reyy;"
        );
        assert_eq!(
            kern_token_string(&KernContent::InvisibleRestSlice(InvisibleRestSlice {
                fermata: Some(crate::types::Fermata::default()),
            })),
            "ryy;"
        );
    }
}
