//! OMR-specific serialization for typed `**kern` documents.

mod document;
mod lines;
mod options;
mod tokens;

#[cfg(test)]
pub(crate) use document::to_compact_full_visual_string_with_options;
#[cfg(test)]
pub(crate) use document::to_transcoda_v2_durationless_slices_string;
pub(crate) use document::transcoda_v2_options;
#[cfg(test)]
pub(crate) use document::{to_compact_no_note_modifiers_string, to_compact_string};
pub use document::{
    to_string, to_transcoda_v2_compact_barlines_empty_nulls_string,
    to_transcoda_v2_compact_barlines_empty_nulls_string_with_options,
    to_transcoda_v2_compact_barlines_string, to_transcoda_v2_compact_barlines_string_with_options,
    to_transcoda_v2_empty_nulls_string, to_transcoda_v2_empty_nulls_string_with_options,
    to_transcoda_v2_string, to_transcoda_v2_string_with_options,
    try_to_transcoda_v2_compact_barlines_empty_nulls_string_with_options,
    try_to_transcoda_v2_compact_barlines_string_with_options,
    try_to_transcoda_v2_empty_nulls_string_with_options, try_to_transcoda_v2_string_with_options,
    OmrSerializer,
};
pub(crate) use lines::{
    serialize_barline_line_with_policy, serialize_data_line, serialize_spine_manipulator,
    serialize_spine_manipulator_lines, serialize_synthetic_barline_line,
    serialize_synthetic_render_barline_line, serialize_system_break, serialize_tandem,
    serialize_tandem_line, spell_tandem,
};
pub use options::{
    ArticulationStyle, BarlineEmissionPolicy, BeamStyle, BowingStyle, ChordDurationStyle,
    ChordNoteOrder, CrossStaffStyle, InvisibleNoteStyle, InvisibleRestModifierStyle,
    InvisibleRestStyle, MarkPlacementStyle, NoteModifierStyle, NullDataTokenStyle,
    OmrSerializeOptions, OrnamentStyle, RestPlacementStyle, SlurPlacementStyle, SpineTopologyStyle,
    StemDirectionStyle, SystemBreakStyle, TerminalLineStyle, TiePlacementStyle, TrillWavyLineStyle,
    VisibilityMarkerStyle,
};
pub(crate) use tokens::{
    serialize_kern_content, serialize_kern_content_preserving_arpeggio_scope, ArpeggioStyle,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::{BarNumberStyle, SerializeOptions};
    use crate::types::{Barline, ForcedAccidental, KernContent, RepeatStyle, SpineContent};
    use crate::{KernDocument, KernLine, ParseDialect, ParseOptions};

    fn serialize(input: &str) -> String {
        KernDocument::parse(input).unwrap().to_omr_string()
    }

    fn serialize_compact(input: &str) -> String {
        KernDocument::parse(input).unwrap().to_compact_omr_string()
    }

    fn serialize_compact_no_note_modifiers(input: &str) -> String {
        KernDocument::parse(input).unwrap().to_transcoda_v2_string()
    }

    fn serialize_compact_with_bar_numbers(input: &str) -> String {
        let document = KernDocument::parse(input).unwrap();

        to_compact_full_visual_string_with_options(
            &document,
            SerializeOptions {
                bar_number_style: BarNumberStyle::Sequential,
            },
        )
    }

    #[test]
    fn serializes_empty_document_as_empty_string() {
        assert_eq!(serialize(""), "");
        assert_eq!(serialize_compact("!!!COM: Example\n"), "");
        assert_eq!(
            serialize_compact_no_note_modifiers("!! only comments\n"),
            ""
        );
    }

    #[test]
    fn serializes_only_kern_spines() {
        let input = [
            "**kern\t**dynam\t**text\t**mxhm\t**kern",
            "*clefG2\t*\t*\t*\t*clefF4",
            "*k[]\t*\t*\t*\t*k[b-]",
            "4c\tmf\tla\tC major\t4E",
            "*-\t*-\t*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            serialize(&input),
            "*clefG2\t*clefF4\n*k[]\t*k[b-]\n4c\t4E\n*-\t*-\n"
        );
    }

    #[test]
    fn duration_bearing_omr_rehydrates_parsed_transcoda_v2_slices() {
        let document = KernDocument::parse_with_options(
            "**kern\t**kern\n4c\tryy;\n*-\t*-\n",
            ParseOptions {
                dialect: ParseDialect::TranscodaV2,
            },
        )
        .unwrap();

        assert_eq!(
            document.to_omr_string(),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n4c\t4ryy;\n*-\t*-\n"
        );
    }

    #[test]
    fn serializes_disjunct_and_linked_ties() {
        let document = KernDocument::parse("**kern\n4c[[\n4c]]\nN[8f<\n2.fN]\n*-\n").unwrap();

        assert_eq!(
            document.to_omr_string(),
            "*clefG2\n*k[]\n4c[[\n4c]]\n8f<N[\n2.fN]\n*-\n"
        );
    }

    #[test]
    fn omits_fingering_spines() {
        let input = ["**kern\t**fing", "4c\t1", "4e 4g\t2 4", "*-\t*-"].join("\n");

        assert_eq!(serialize_compact(&input), "*clefG2\n*k[]\n4c\n4e g\n");
    }

    #[test]
    fn transcoda_v2_writes_an_invisible_note_as_an_invisible_rest() {
        let input = "**kern\n[3dd-yy\n4c\n*-\n";

        assert_eq!(
            serialize_compact_no_note_modifiers(input),
            "*clefG2\n*k[]\n3ryy\n4c\n"
        );
    }

    #[test]
    fn transcoda_v2_omits_invisible_chord_noteheads() {
        let input = "**kern\n4cyy 4e 4g\n4c 4eyy 4g\n4cyy 4eyy\n*-\n";

        assert_eq!(
            serialize_compact_no_note_modifiers(input),
            "*clefG2\n*k[]\n4e g\n4c g\n4ryy\n"
        );
    }

    #[test]
    fn transcoda_v2_keeps_other_spines_when_invisible_note_field_is_omitted() {
        let input = "**kern\t**kern\n4cyy\t4e\n4d\t4fyy\n*-\t*-\n";

        assert_eq!(
            serialize_compact_no_note_modifiers(input),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n4ryy\t4e\n4d\t4ryy\n"
        );
    }

    #[test]
    fn serializes_start_and_end_repeat_barline() {
        let input = ["**kern", "*clefG2", "*k[]", "4c", "=:|!|:", "*-"].join("\n");

        assert_eq!(
            Barline::try_from("=:|:").unwrap().repeat(),
            RepeatStyle::StartAndEnd
        );
        assert_eq!(serialize(&input), "*clefG2\n*k[]\n4c\n=:|:\n*-\n");
    }

    #[test]
    fn transcoda_v2_presets_preserve_invisible_barlines() {
        let document = KernDocument::parse(
            [
                "**kern\t**kern",
                "*M4/4\t*M4/4",
                "4c\t4e",
                "=-\t=-",
                "4d\t4f",
                "*-\t*-",
            ]
            .join("\n")
            .as_str(),
        )
        .unwrap();

        let expected_per_spine =
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n4c\t4e\n=-\t=-\n4d\t4f\n";
        let expected_compact = "*clefG2\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n4c\t4e\n=-\n4d\t4f\n";

        assert_eq!(to_transcoda_v2_string(&document), expected_per_spine);
        assert_eq!(
            to_transcoda_v2_empty_nulls_string(&document),
            expected_per_spine
        );
        assert_eq!(
            to_transcoda_v2_compact_barlines_string(&document),
            expected_compact
        );
        assert_eq!(
            to_transcoda_v2_compact_barlines_empty_nulls_string(&document),
            expected_compact
        );
    }

    #[test]
    fn transcoda_v2_bar_numbers_preserve_invisible_barlines() {
        let document = KernDocument::parse("**kern\n*M4/4\n4c\n=-\n4d\n*-\n").unwrap();

        assert_eq!(
            to_transcoda_v2_compact_barlines_empty_nulls_string_with_options(
                &document,
                SerializeOptions {
                    bar_number_style: BarNumberStyle::Sequential,
                },
            ),
            "*clefG2\n*k[]\n*M4/4\n=1\n4c\n=2-\n4d\n"
        );
    }

    #[test]
    fn sequential_bar_numbers_number_measure_starts_not_final_barlines() {
        let input = ["**kern", "*M4/4", "=9", "4c", "=", "4d", "==", "*-"].join("\n");

        assert_eq!(
            serialize_compact_with_bar_numbers(&input),
            "*clefG2\n*k[]\n*M4/4\n=1\n4c\n=2\n4d\n==\n"
        );
    }

    #[test]
    fn sequential_bar_numbers_preserve_barline_styles() {
        let input = [
            "**kern", "4c", "=:|!", "4d", "=!|:", "4e", "=:|!|:", "4f", "==", "*-",
        ]
        .join("\n");

        assert_eq!(
            serialize_compact_with_bar_numbers(&input),
            "*clefG2\n*k[]\n=1\n4c\n=2:|!\n4d\n=3!|:\n4e\n=4:|:\n4f\n==\n"
        );
    }

    #[test]
    fn sequential_bar_numbers_are_emitted_for_each_kern_spine() {
        let input = ["**kern\t**kern", "4c\t4e", "=\t=", "4d\t4f", "*-\t*-"].join("\n");

        assert_eq!(
            serialize_compact_with_bar_numbers(&input),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n=1\t=1\n4c\t4e\n=2\t=2\n4d\t4f\n"
        );
    }

    #[test]
    fn sequential_bar_numbers_advance_over_multi_measure_rests() {
        let input = ["**kern", "*M3/4", "MM2r", "=", "2.r", "=", "*-"].join("\n");

        assert_eq!(
            serialize_compact_with_bar_numbers(&input),
            "*clefG2\n*k[]\n*M3/4\n=1\nMM2r\n=3\n2.r\n=\n"
        );
    }

    #[test]
    fn omits_comments_metadata_and_nonserializable_fields() {
        let input = [
            "!!!OTL: Hidden title",
            "!! hidden global comment",
            "**kern\t**dynam",
            "! hidden local comment\t!",
            "*part1\t*",
            "*staff1\t*",
            "*Ivioln\t*",
            "*MM96\t*",
            "*clefG2\t*",
            "*k[]\t*",
            "*M4/4\t*",
            "4c#Xiyy\tmf",
            "4rG,yy;\t.",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(serialize(&input), "*clefG2\n*k[]\n*M4/4\n4c#X\n4ryy;\n*-\n");
    }

    #[test]
    fn reorders_only_the_initial_header_block() {
        let input = [
            "**kern", "*M3/4", "*k[f#]", "*clefG2", "4c", "*M2/4", "4d", "*-",
        ]
        .join("\n");

        assert_eq!(
            serialize(&input),
            "*clefG2\n*k[f#]\n*M3/4\n4c\n*M2/4\n4d\n*-\n"
        );
    }

    #[test]
    fn canonicalizes_kintail_leading_barline_header_normalization() {
        let input = [
            "!!!COM: Traditional",
            "!!!OTL: I Will Go Home to Kintail",
            "**kern",
            "*clefG2",
            "*k[]",
            "=",
            "*MM40",
            "*M4/4",
            "*k[]",
            "*clefG2",
            "8ggqq",
            "16bLK",
            "8gqqL",
            "8ddqq",
            "8ccqqJ",
            "8.dd",
            "16eek",
            "8ggqq",
            "8.a",
            "8ggqqL",
            "8eeqq",
            "8ffqq",
            "8eeqqJ",
            "8.ff",
            "16ddL",
            "16eeJ",
            "8ggqq",
            "8.aJ",
            "=",
        ]
        .join("\n");

        assert_eq!(
            serialize_compact(&input),
            "*clefG2\n*k[]\n*M4/4\n8qqgg\n16bLK\n8qqgL\n8qqdd\n8qqccJ\n8.dd\n16eek\n8qqgg\n8.a\n8qqggL\n8qqee\n8qqff\n8qqeeJ\n8.ff\n16ddL\n16eeJ\n8qqgg\n8.aJ\n=\n"
        );
    }

    #[test]
    fn serializes_pedal_tandems() {
        let input = ["**kern", "*clefG2", "4c", "*ped", "4d", "*Xped", "4e", "*-"].join("\n");

        assert_eq!(
            serialize(&input),
            "*clefG2\n*k[]\n4c\n*ped\n4d\n*Xped\n4e\n*-\n"
        );
    }

    #[test]
    fn transcoda_v2_omits_pedal_tandems() {
        let input = ["**kern", "*clefG2", "4c", "*ped", "4d", "*Xped", "4e", "*-"].join("\n");

        assert_eq!(
            serialize_compact_no_note_modifiers(&input),
            "*clefG2\n*k[]\n4c\n4d\n4e\n"
        );
    }

    #[test]
    fn serializes_tuplet_tandems() {
        let input = [
            "**kern", "*clefG2", "4c", "*Xtuplet", "12d", "12e", "12f", "*tuplet", "4g", "*-",
        ]
        .join("\n");

        assert_eq!(
            serialize(&input),
            "*clefG2\n*k[]\n4c\n*Xtuplet\n12d\n12e\n12f\n*tuplet\n4g\n*-\n"
        );
    }

    #[test]
    fn transcoda_v2_omits_tuplet_tandems() {
        let input = [
            "**kern", "*clefG2", "4c", "*Xtuplet", "12d", "12e", "12f", "*tuplet", "4g", "*-",
        ]
        .join("\n");

        assert_eq!(
            serialize_compact_no_note_modifiers(&input),
            "*clefG2\n*k[]\n4c\n12d\n12e\n12f\n4g\n"
        );
    }

    #[test]
    fn serializes_manual_fix_tuplet_tandem_regression() {
        let input = [
            "**kern\t**kern",
            "*clefF4\t*clefG2",
            "*k[f#c#g#d#a#]\t*k[f#c#g#d#a#]",
            "*Xtuplet\t*Xtuplet",
            "2BBB 2FF#\t12r",
            ".\t12f#L",
            ".\t12bJ",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            serialize(&input),
            "*clefF4\t*clefG2\n*k[f#c#g#d#a#]\t*k[f#c#g#d#a#]\n*Xtuplet\t*Xtuplet\n2BBB 2FF#\t12r\n.\t12f#L\n.\t12bJ\n*-\t*-\n"
        );
    }

    #[test]
    fn serializes_stria_tandems() {
        let input = [
            "**kern", "*stria5", "*clefG2", "*k[]", "4c", "*stria4", "4d", "*-",
        ]
        .join("\n");

        assert_eq!(
            serialize(&input),
            "*clefG2\n*k[]\n*stria5\n4c\n*stria4\n4d\n*-\n"
        );
    }

    #[test]
    fn serializes_bracket_span_tandems() {
        let input = [
            "**kern", "*clefC3", "*lig", "1d", "*col", "1B", "*Xcol", "0c", "*Xlig", "*-",
        ]
        .join("\n");

        assert_eq!(
            serialize(&input),
            "*clefC3\n*k[]\n*lig\n1d\n*col\n1B\n*Xcol\n0c\n*Xlig\n*-\n"
        );
    }

    #[test]
    fn serializes_thru_section_and_expansion_tandems() {
        let input = [
            "**kern",
            "*>[A,A,B,B]",
            "*>A",
            "4c",
            "*>ossia[Minuet,Minuet,Trio,Trio,,Minuet]",
            "*>B",
            "4d",
            "*-",
        ]
        .join("\n");

        assert_eq!(
            serialize(&input),
            "*clefG2\n*k[]\n*>[A,A,B,B]\n*>A\n4c\n*>ossia[Minuet,Minuet,Trio,Trio,,Minuet]\n*>B\n4d\n*-\n"
        );
    }

    #[test]
    fn transcoda_v2_omits_thru_tandems() {
        let input = ["**kern", "*>[A,A,B,B]", "*>A", "4c", "*>B", "4d", "*-"].join("\n");

        assert_eq!(
            serialize_compact_no_note_modifiers(&input),
            "*clefG2\n*k[]\n4c\n4d\n"
        );
    }

    #[test]
    fn serializes_percussion_clef_without_zero_line_suffix() {
        let input = ["**kern", "*clefX", "*k[]", "4c", "*-"].join("\n");
        let output = serialize(&input);

        assert_eq!(output, "*clefX\n*k[]\n4c\n*-\n");
        KernDocument::parse(&output).unwrap();
    }

    #[test]
    fn preserves_invisible_clef_markers() {
        let input = [
            "**kern\t**kern",
            "*clefGv2yy\t*clefXyy",
            "*k[]\t*k[]",
            "4c\t4e",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            serialize(&input),
            "*clefGv2yy\t*clefXyy\n*k[]\t*k[]\n4c\t4e\n*-\t*-\n"
        );
    }

    #[test]
    fn synthesizes_missing_header_lines() {
        let input = ["**kern", "*M6/8", "8c", "*-"].join("\n");

        assert_eq!(serialize(&input), "*clefG2\n*k[]\n*M6/8\n8c\n*-\n");
    }

    #[test]
    fn serializes_data_tokens_in_canonical_order() {
        let input = [
            "**kern\t**kern\t**kern",
            "*clefG2\t*clefG2\t*clefG2",
            "*k[]\t*k[]\t*k[]",
            "4cL[\t4c~'v:\tccq",
            "*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            serialize(&input),
            "*clefG2\t*clefG2\t*clefG2\n*k[]\t*k[]\t*k[]\n4c[L\t4cv:\t8qcc\n*-\t*-\t*-\n"
        );
    }

    #[test]
    fn compact_serialization_omits_repeated_chord_durations() {
        let input = ["**kern", "4c 4e 4g", "*-"].join("\n");

        assert_eq!(serialize_compact(&input), "*clefG2\n*k[]\n4c e g\n");
    }

    #[test]
    fn serializes_normalized_chord_beams() {
        let input = ["**kern", "4cLJ", "4dLLJ", "*-"].join("\n");

        assert_eq!(serialize(&input), "*clefG2\n*k[]\n4c\n4dL\n*-\n");
    }

    #[test]
    fn compact_serialization_is_opt_in() {
        let input = ["**kern", "4c 4e 4g", "*-"].join("\n");

        assert_eq!(serialize(&input), "*clefG2\n*k[]\n4c 4e 4g\n*-\n");
    }

    #[test]
    fn compact_serialization_keeps_single_notes_unchanged() {
        let input = ["**kern", "4c", "*-"].join("\n");

        assert_eq!(serialize_compact(&input), "*clefG2\n*k[]\n4c\n");
    }

    #[test]
    fn serializes_literal_multi_measure_rest() {
        let input = ["**kern", "*M3/4", "MM55r", "=", "*-"].join("\n");

        assert_eq!(
            serialize_compact(&input),
            "*clefG2\n*k[]\n*M3/4\nMM55r\n=\n"
        );
    }

    #[test]
    fn compact_serialization_collapses_barline_bounded_full_measure_rests() {
        let input = ["**kern", "*M3/4", "=1", "2.r", "=", "2.r", "=", "*-"].join("\n");

        assert_eq!(serialize_compact(&input), "*clefG2\n*k[]\n*M3/4\nMM2r\n=\n");
    }

    #[test]
    fn compact_serialization_collapses_rests_with_null_dynamic_spine() {
        let input = [
            "**kern\t**dynam",
            "*M3/4\t*",
            "=1\t=1",
            "2.r\t.",
            "=\t=",
            "2.r\t.",
            "=\t=",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            serialize_compact_no_note_modifiers(&input),
            "*clefG2\n*k[]\n*M3/4\nMM2r\n=\n"
        );
    }

    #[test]
    fn compact_serialization_does_not_include_unbarlined_leading_rest() {
        let input = ["**kern", "*M3/4", "2.r", "=", "2.r", "=", "*-"].join("\n");

        assert_eq!(
            serialize_compact(&input),
            "*clefG2\n*k[]\n*M3/4\n2.r\n=\n2.r\n=\n"
        );
    }

    #[test]
    fn compact_serialization_collapses_final_rest_run_before_terminal_barline() {
        let input = [
            "**kern", "*M2/2", "=", "1r", "=", "1r", "=", "1r", "==", "*-",
        ]
        .join("\n");

        assert_eq!(
            serialize_compact(&input),
            "*clefG2\n*k[]\n*M2/2\nMM3r\n==\n"
        );
    }

    #[test]
    fn compact_serialization_tracks_meter_symbol_meter_changes() {
        let input = [
            "**kern", "*M3/4", "=1", "2.r", "=||", "*met(c|)", "1r", "=", "1r", "==", "*-",
        ]
        .join("\n");

        assert_eq!(
            serialize_compact(&input),
            "*clefG2\n*k[]\n*M3/4\n2.r\n=||\n*met(c|)\nMM2r\n==\n"
        );
    }

    #[test]
    fn compact_no_note_modifiers_collapses_multi_measure_rests() {
        let input = ["**kern", "*M3/4", "=1", "2.r", "=", "2.r", "=", "*-"].join("\n");

        assert_eq!(
            serialize_compact_no_note_modifiers(&input),
            "*clefG2\n*k[]\n*M3/4\nMM2r\n=\n"
        );
    }

    #[test]
    fn compact_serialization_does_not_cross_meter_changes_or_decorated_rests() {
        let meter_change = [
            "**kern", "*M3/4", "=1", "2.r", "=", "*M4/4", "1r", "=", "*-",
        ]
        .join("\n");
        let decorated = ["**kern", "*M3/4", "=1", "2.r;", "=", "2.r", "=", "*-"].join("\n");

        assert_eq!(
            serialize_compact(&meter_change),
            "*clefG2\n*k[]\n*M3/4\n2.r\n=\n*M4/4\n1r\n=\n"
        );
        assert_eq!(
            serialize_compact(&decorated),
            "*clefG2\n*k[]\n*M3/4\n2.r;\n=\n2.r\n=\n"
        );
    }

    #[test]
    fn compact_serialization_does_not_cross_repeat_barlines_or_non_rest_data() {
        let repeat = ["**kern", "*M3/4", "=1", "2.r", "=:|", "2.r", "=", "*-"].join("\n");
        let note = [
            "**kern", "*M3/4", "=1", "2.r", "=", "4c", "=", "2.r", "=", "*-",
        ]
        .join("\n");

        assert_eq!(
            serialize_compact(&repeat),
            "*clefG2\n*k[]\n*M3/4\n2.r\n=:|!\n2.r\n=\n"
        );
        assert_eq!(
            serialize_compact(&note),
            "*clefG2\n*k[]\n*M3/4\n2.r\n=\n4c\n=\n2.r\n=\n"
        );
    }

    #[test]
    fn compact_serialization_preserves_note_local_marks() {
        let input = ["**kern", "[4f#X 4a-y]u", "4ccn_v", "*-"].join("\n");

        assert_eq!(
            serialize_compact(&input),
            "*clefG2\n*k[]\n4f#X[ a-y]u\n4ccn_v\n"
        );
    }

    #[test]
    fn compact_serialization_keeps_cross_staff_at_pitch_boundary() {
        let mut document = KernDocument::parse("**kern\n4c#>\n4d#<\n4e<[\n*-\n").unwrap();
        let mut data_index = 0;
        for line in document.lines_mut() {
            let KernLine::Data(contents) = line else {
                continue;
            };
            let SpineContent::Kern(KernContent::Chord(chord)) = &mut contents[0] else {
                panic!("expected chord content");
            };
            match data_index {
                0 => chord.notes[0].modifiers.forced_accidental = Some(ForcedAccidental),
                1 => chord.notes[0].modifiers.invisible_accidental = true,
                _ => {}
            }
            data_index += 1;
        }
        assert_eq!(data_index, 3);

        let output = document.to_compact_omr_string();

        assert_eq!(output, "*clefG2\n*k[]\n4c#X>\n4d#y<\n4e<[\n");
        KernDocument::parse(&output).unwrap();
    }

    #[test]
    fn compact_no_note_modifiers_omits_slurs_ties_accidentals_and_bowings() {
        let input = ["**kern", "(4f#X[ 4a-y]u)", "4ccn_v", "*-"].join("\n");

        assert_eq!(
            serialize_compact_no_note_modifiers(&input),
            "*clefG2\n*k[]\n4f# a-\n4ccn\n"
        );
    }

    #[test]
    fn compact_no_note_modifiers_applies_to_grace_chords() {
        let input = ["**kern", "(8gg#Xq[ 8aa-y]q)", "*-"].join("\n");

        assert_eq!(
            serialize_compact_no_note_modifiers(&input),
            "*clefG2\n*k[]\n8qgg# aa-\n"
        );
    }

    #[test]
    fn compact_serialization_omits_repeated_grace_chord_durations_and_kinds() {
        let input = ["**kern", "16qqCC# 16qqC#LL", "16qqDD 16qqD#JJ", "*-"].join("\n");
        let output = serialize_compact_no_note_modifiers(&input);

        assert_eq!(output, "*clefG2\n*k[]\n16qqCC# C#LL\n16qqDD D#JJ\n");
        KernDocument::parse(&output).unwrap();
    }

    #[test]
    fn compact_serialization_keeps_divergent_grace_kinds_on_secondary_notes() {
        let input = ["**kern", "8qcc 8Qee", "*-"].join("\n");
        let output = serialize_compact_no_note_modifiers(&input);

        assert_eq!(output, "*clefG2\n*k[]\n8qcc Qee\n");
        KernDocument::parse(&output).unwrap();
    }

    #[test]
    fn compact_serialization_places_grace_kind_before_pitch() {
        let input = ["**kern", "16ccc#yqqJJ", "*-"].join("\n");
        let output = serialize_compact(&input);

        assert_eq!(output, "*clefG2\n*k[]\n16qqccc#yJJ\n");
        KernDocument::parse(&output).unwrap();
    }

    #[test]
    fn compact_serialization_keeps_chord_modifiers_at_chord_level() {
        let input = ["**kern", "4c: 4e", "*-"].join("\n");

        assert_eq!(serialize_compact(&input), "*clefG2\n*k[]\n4c e:\n");
    }

    #[test]
    fn full_visual_preserves_articulations_and_training_target_omits_them() {
        let input = ["**kern", "4c' 4d\" 4e` 4f~ 4g^ 4az", "*-"].join("\n");

        assert_eq!(serialize(&input), "*clefG2\n*k[]\n4c 4d 4e 4f 4g 4a\n*-\n");
        assert_eq!(
            serialize_compact(&input),
            "*clefG2\n*k[]\n4c d e f g a'\"`~^z\n"
        );
        assert_eq!(
            serialize_compact_no_note_modifiers(&input),
            "*clefG2\n*k[]\n4c d e f g a\n"
        );
    }

    #[test]
    fn compact_serialization_omits_final_terminal_line() {
        let input = ["**kern", "4c", "*-"].join("\n");

        assert_eq!(serialize_compact(&input), "*clefG2\n*k[]\n4c\n");
    }

    #[test]
    fn compact_serialization_omits_multi_spine_final_terminal_line() {
        let input = ["**kern\t**kern", "4c\t4e", "*-\t*-"].join("\n");

        assert_eq!(
            serialize_compact(&input),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n4c\t4e\n"
        );
    }

    #[test]
    fn compact_serialization_keeps_internal_topology_lines() {
        let input = ["**kern", "*^", "4c\t4e", "*v\t*v", "2g", "*-"].join("\n");

        assert_eq!(
            serialize_compact(&input),
            "*clefG2\n*k[]\n*^\n4c\t4e\n*v\t*v\n2g\n"
        );
    }

    #[test]
    fn compact_serialization_keeps_independent_merge_groups_after_filtering_non_kern_spines() {
        let input = [
            "**kern\t**text\t**kern\t**text",
            "*clefG2\t*\t*clefG2\t*",
            "*k[]\t*\t*k[]\t*",
            "*^\t*\t*^\t*",
            "4c\t4d\ta\t4e\t4f\tb",
            "*v\t*v\t*\t*v\t*v\t*",
            "2c\ta\t2e\tb",
            "*-\t*-\t*-\t*",
        ]
        .join("\n");
        let output = serialize_compact(&input);

        // Merge groups come out left to right, matching the canonical order
        // `MergeAdjacentSpineManipulators` produces when re-parsing.
        assert_eq!(
            output,
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*^\t*^\n4c\t4d\t4e\t4f\n*v\t*v\t*\t*\n*\t*v\t*v\n2c\t2e\n"
        );
        KernDocument::parse(&output).unwrap();
    }

    #[test]
    fn compact_serialization_canonicalizes_valid_mixed_split_merge_topology() {
        let input = [
            "**kern\t**kern",
            "*clefG2\t*clefG2",
            "*k[]\t*k[]",
            "*\t*^",
            "*^\t*v\t*v",
            "4c\t4e\t4g",
            "*v\t*v\t*",
            "2c\t2g",
            "*-\t*-",
        ]
        .join("\n");
        let output = serialize_compact(&input);

        assert_eq!(
            output,
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*^\t*\n4c\t4e\t4g\n*v\t*v\t*\n2c\t2g\n"
        );
        KernDocument::parse(&output).unwrap();
    }

    #[test]
    fn serialization_restores_open_final_split_before_final_barline() {
        let input = ["**kern", "*^", "4c\t4e", "=\t=", "*-\t*-"].join("\n");

        assert_eq!(
            serialize(&input),
            "*clefG2\n*k[]\n*^\n4c\t4e\n*v\t*v\n=\n*-\n"
        );
    }

    #[test]
    fn compact_serialization_restores_open_final_split_before_terminal() {
        let input = ["**kern", "*^", "4c\t4e", "*-\t*-"].join("\n");

        assert_eq!(
            serialize_compact(&input),
            "*clefG2\n*k[]\n*^\n4c\t4e\n*v\t*v\n"
        );
    }

    #[test]
    fn compact_serialization_keeps_non_final_terminator_lines() {
        let input = ["**kern\t**kern", "4c\t4e", "*-\t*", "4g", "*-"].join("\n");

        assert_eq!(
            serialize_compact(&input),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n4c\t4e\n*-\t*\n4g\n"
        );
    }

    #[test]
    fn keeps_kern_spine_topology_width_correct_after_filtering() {
        let input = [
            "**kern\t**dynam",
            "*^\t*",
            "4c\t4e\tmf",
            "*v\t*v\t*",
            "2g\tff",
            "*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            serialize(&input),
            "*clefG2\n*k[]\n*^\n4c\t4e\n*v\t*v\n2g\n*-\n"
        );
    }

    #[test]
    fn applies_omr_normalization_before_serializing() {
        let input = ["**kern", "*part1", "*M3/4", "*k[]", "*clefG2", "4c[]", "*-"].join("\n");

        assert_eq!(serialize(&input), "*clefG2\n*k[]\n*M3/4\n4c\n*-\n");
    }

    #[test]
    fn serialization_does_not_mutate_original_document() {
        let input = ["**kern", "*part1", "*M3/4", "*k[]", "*clefG2", "(4c)", "*-"].join("\n");
        let document = KernDocument::parse(&input).unwrap();
        let original = document.clone();

        let _ = document.to_omr_string();

        assert_eq!(document, original);
    }
}
