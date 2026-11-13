//! Serializers for parsed Humdrum data.

mod bar_numbers;
mod barline;
pub mod batch;
pub mod content_policy;
pub mod jsonl;
pub mod omr;
pub mod render;
pub mod staff_signatures;
pub mod tandem_policy;
pub mod token;
pub mod transcoda_v2;

use crate::KernDocument;
use thiserror::Error;

pub use bar_numbers::BarNumberStyle;
pub use content_policy::{ContentAction, ContentGroup, ContentKind, ContentPolicy};
pub use tandem_policy::{DialectError, TandemAction, TandemKind, TandemPolicy};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernDialect {
    HumdrumKern,
    TranscodaOmr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentProfile {
    Renderable,
    TrainingTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializePreset {
    HumdrumKern,
    KernOnly,

    TranscodaV2,
    /// Transcoda V2 labels without display-only editorial accidentals.
    TranscodaV2Training,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SerializeOptions {
    pub bar_number_style: BarNumberStyle,
}

#[derive(Debug, Error)]
pub enum SerializeError {
    #[error(transparent)]
    InvisibleRestSlice(crate::normalization::invisible_rest_slices::InvisibleRestSliceError),
    #[error("Transcoda V2 canonicalization did not converge after {iterations} iterations")]
    CanonicalizationDidNotConverge { iterations: usize },

    #[error(transparent)]
    Dialect(#[from] DialectError),
}

impl SerializeError {
    /// Distinguish rejected input from a failure of the conversion itself.
    pub fn is_execution_failure(&self) -> bool {
        match self {
            Self::CanonicalizationDidNotConverge { .. } => true,
            Self::InvisibleRestSlice(error) => rest_slice_execution_failure(error),

            Self::Dialect(_) => false,
        }
    }
}

fn rest_slice_execution_failure(
    error: &crate::normalization::invisible_rest_slices::InvisibleRestSliceError,
) -> bool {
    use crate::normalization::invisible_rest_slices::InvisibleRestSliceError;
    match error {
        InvisibleRestSliceError::TranscodaV2DidNotConverge { .. }
        | InvisibleRestSliceError::TranscodaV2Invariant { .. }
        | InvisibleRestSliceError::TranscodaV2SemanticEvents { .. }
        | InvisibleRestSliceError::RemainingDurationUnderflow { .. } => true,
        InvisibleRestSliceError::DataLineDurations(
            crate::analysis::DataLineDurationError::DurationUnderflow { .. },
        ) => true,
        InvisibleRestSliceError::DataLineDurations(_)
        | InvisibleRestSliceError::Dialect(_)
        | InvisibleRestSliceError::UnequalMergedRemaining { .. }
        | InvisibleRestSliceError::ZeroDurationSlice { .. }
        | InvisibleRestSliceError::UnspellableSliceDuration { .. }
        | InvisibleRestSliceError::TranscodaV2Lowering { .. }
        | InvisibleRestSliceError::UnopenedSplitMerge { .. }
        | InvisibleRestSliceError::TimelineBreak(_) => false,
    }
}

impl From<crate::normalization::invisible_rest_slices::InvisibleRestSliceError> for SerializeError {
    fn from(error: crate::normalization::invisible_rest_slices::InvisibleRestSliceError) -> Self {
        match error {
            crate::normalization::invisible_rest_slices::InvisibleRestSliceError::TranscodaV2DidNotConverge {
                iterations,
            } => Self::CanonicalizationDidNotConverge { iterations },
            other => Self::InvisibleRestSlice(other),
        }
    }
}

impl SerializePreset {
    pub fn dialect(self) -> KernDialect {
        match self {
            Self::HumdrumKern | Self::KernOnly => KernDialect::HumdrumKern,
            Self::TranscodaV2 | Self::TranscodaV2Training => KernDialect::TranscodaOmr,
        }
    }

    pub fn content_profile(self) -> ContentProfile {
        match self {
            Self::HumdrumKern | Self::KernOnly => ContentProfile::Renderable,
            Self::TranscodaV2 | Self::TranscodaV2Training => ContentProfile::TrainingTarget,
        }
    }

    pub fn serialize_document(self, document: &KernDocument) -> String {
        self.serialize_document_with_options(document, SerializeOptions::default())
    }

    pub fn serialize_document_with_options(
        self,
        document: &KernDocument,
        options: SerializeOptions,
    ) -> String {
        self.try_serialize_document_with_options(document, options)
            .expect("serialization failed")
    }

    pub fn try_serialize_document(self, document: &KernDocument) -> Result<String, SerializeError> {
        self.try_serialize_document_with_options(document, SerializeOptions::default())
    }

    pub fn try_serialize_document_with_options(
        self,
        document: &KernDocument,
        options: SerializeOptions,
    ) -> Result<String, SerializeError> {
        let humdrum_kern = |ornament_style,
                            trill_wavy_line_style,
                            visibility_marker_style,
                            include_dynamics,
                            include_fingering,
                            tandem_policy,
                            include_metadata,
                            extended_metadata_records,
                            include_local_comments,
                            include_system_breaks,
                            include_cross_staff_rdf_footer,
                            cross_staff_style,
                            arpeggio_style| {
            render::to_string_with_options_and_arpeggio_style(
                document,
                render::RenderSerializeOptions {
                    bar_number_style: options.bar_number_style,
                    accidental_suppression_style: render::AccidentalSuppressionStyle::HonorSource,
                    forced_accidental_style: render::ForcedAccidentalStyle::Source,
                    ornament_style,
                    trill_wavy_line_style,
                    visibility_marker_style,
                    include_dynamics,
                    include_fingering,
                    tandem_policy,
                    include_metadata,
                    extended_metadata_records,
                    include_local_comments,
                    // Frozen and engine-facing presets keep collapsing
                    // comments the typed AST cannot model.
                    preserve_unsupported_local_comments: false,
                    meter_output: crate::MeterOutput::ExplicitTimeSignature,
                    include_system_breaks,
                    include_cross_staff_rdf_footer,
                    cross_staff_style,
                    slur_placement_style: if matches!(self, Self::HumdrumKern | Self::KernOnly) {
                        omr::SlurPlacementStyle::Preserve
                    } else {
                        omr::SlurPlacementStyle::Omit
                    },
                    tie_placement_style: if matches!(self, Self::HumdrumKern | Self::KernOnly) {
                        omr::TiePlacementStyle::Preserve
                    } else {
                        omr::TiePlacementStyle::Omit
                    },
                    mark_placement_style: if matches!(self, Self::HumdrumKern | Self::KernOnly) {
                        omr::MarkPlacementStyle::Preserve
                    } else {
                        omr::MarkPlacementStyle::Omit
                    },
                    rest_placement_style: if matches!(self, Self::HumdrumKern | Self::KernOnly) {
                        omr::RestPlacementStyle::Preserve
                    } else {
                        omr::RestPlacementStyle::Omit
                    },
                },
                arpeggio_style,
            )
        };

        Ok(match self {
            Self::HumdrumKern => {
                render::to_render_output(document, humdrum_kern_render_options(options))?
                    .canonical_text()
            }
            Self::KernOnly => humdrum_kern(
                omr::OrnamentStyle::Preserve,
                omr::TrillWavyLineStyle::Preserve,
                omr::VisibilityMarkerStyle::Preserve,
                false,
                false,
                TandemPolicy::HumdrumKern,
                false,
                false,
                false,
                false,
                false,
                omr::CrossStaffStyle::Preserve,
                omr::ArpeggioStyle::Preserve,
            )?,

            Self::TranscodaV2 => {
                omr::try_to_transcoda_v2_compact_barlines_empty_nulls_string_with_options(
                    document, options,
                )?
            }
            Self::TranscodaV2Training => transcoda_v2::try_serialize_transcoda_v2(
                document,
                transcoda_v2::TranscodaV2Syntax::CompactBarlinesEmptyNulls,
                ContentPolicy::TranscodaV2Training,
                options,
            )?,
        })
    }
}

/// Render options behind [`SerializePreset::HumdrumKern`], exposed so callers
/// that need the per-line [`render::RenderOutput`] serialize identically.
pub(crate) fn humdrum_kern_render_options(
    options: SerializeOptions,
) -> render::RenderSerializeOptions {
    render::RenderSerializeOptions {
        bar_number_style: options.bar_number_style,
        extended_metadata_records: true,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernDocument;
    use crate::{ParseDialect, ParseOptions};

    #[test]
    fn conversion_invariant_failures_are_execution_failures() {
        assert!(
            SerializeError::CanonicalizationDidNotConverge { iterations: 12 }
                .is_execution_failure()
        );
        assert!(SerializeError::InvisibleRestSlice(crate::normalization::invisible_rest_slices::InvisibleRestSliceError::TranscodaV2Invariant { message: "invalid projected model".into() }).is_execution_failure());
    }

    #[test]
    fn free_meter_is_kept_by_humdrum_kern_and_omitted_from_transcoda_v2() {
        let document = KernDocument::parse(
            "**kern\n*M4/4\n=1\n1c\n=2\n*MX\n2.d\n16e\n=3\n*M4/4\n1f\n==\n*-\n",
        )
        .unwrap();

        assert!(SerializePreset::HumdrumKern
            .serialize_document(&document)
            .contains("=\n*MX\n2.d\n"));
        assert!(!SerializePreset::TranscodaV2
            .serialize_document(&document)
            .contains("*MX"));
    }

    #[test]
    fn transcoda_v2_keeps_only_printed_meter_changes() {
        let document = KernDocument::parse(
            "**kern\n*M3/4\n=1\n2.c\n=2\n*M4/4\n*met()\n1d\n=3\n*M3/4\n2.e\n=4\n*-\n",
        )
        .unwrap();

        assert_eq!(
            SerializePreset::TranscodaV2.serialize_document(&document),
            "*clefG2\n*k[]\n*M3/4\n2.c\n=\n1d\n=\n*M3/4\n2.e\n=\n"
        );
    }

    #[test]
    fn kern_only_omits_auxiliary_spines_comments_and_metadata() {
        let document = KernDocument::parse(
            [
                "!!!COM: Adler, Clara",
                "!!!OTL: Prelude",
                "**kern\t**fing\t**dynam\t**text\t**foo",
                "*M4/4\t*\t*\t*\t*",
                "!LO:TX:a:t=Allegro\t!\t!\t!\t!",
                "=1\t=1\t=1\t=1\t=1",
                "4c\t1\tmf\tla\tx",
                "*-\t*-\t*-\t*-\t*-",
            ]
            .join("\n")
            .as_str(),
        )
        .unwrap();

        assert_eq!(
            SerializePreset::KernOnly.serialize_document(&document),
            "**kern\n*clefG2\n*k[]\n*M4/4\n=\n4c\n*-\n"
        );
    }

    #[test]
    fn transcoda_v2_omits_concluding_turn() {
        let document = KernDocument::parse("**kern\n4cR\n*-\n").unwrap();

        assert_eq!(
            SerializePreset::TranscodaV2.serialize_document(&document),
            "*clefG2\n*k[]\n4c\n"
        );
    }

    #[test]
    fn transcoda_v2_omits_visual_style_tandems() {
        let document = KernDocument::parse("**kern\n*2\\right\n4c\n*2\\left\n4d\n*-\n").unwrap();

        assert_eq!(
            SerializePreset::TranscodaV2.serialize_document(&document),
            "*clefG2\n*k[]\n4c\n4d\n"
        );
    }

    #[test]
    fn transcoda_v2_omits_group_tandems() {
        let document = KernDocument::parse("**kern\n*group1\n*M4/4\n4c\n*-\n").unwrap();

        assert_eq!(
            SerializePreset::TranscodaV2.serialize_document(&document),
            "*clefG2\n*k[]\n*M4/4\n4c\n"
        );
    }

    #[test]
    fn transcoda_v2_omits_bracket_span_tandems() {
        let document =
            KernDocument::parse("**kern\n*lig\n4c\n*col\n4d\n*Xcol\n4e\n*Xlig\n*-\n").unwrap();

        assert_eq!(
            SerializePreset::TranscodaV2.serialize_document(&document),
            "*clefG2\n*k[]\n4c\n4d\n4e\n"
        );
    }

    #[test]
    fn transcoda_v2_serializes_null_data_as_empty_fields() {
        let single_spine = KernDocument::parse("**kern\n.\n4c\n*-\n").unwrap();

        assert_eq!(
            SerializePreset::TranscodaV2.serialize_document(&single_spine),
            "*clefG2\n*k[]\n4c\n"
        );

        let multi_spine = KernDocument::parse(
            ["**kern\t**kern", ".\t4e", "4c\t.", ".\t.", "*-\t*-"]
                .join("\n")
                .as_str(),
        )
        .unwrap();

        assert_eq!(
            SerializePreset::TranscodaV2.serialize_document(&multi_spine),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n\t4e\n4c\t\n"
        );
    }

    #[test]
    fn transcoda_v2_serializes_compact_barlines() {
        let document = KernDocument::parse(
            [
                "**kern\t**kern",
                "*M4/4\t*M4/4",
                "4c\t4e",
                "=\t=",
                "4d\t4f",
                "==\t==",
                "*-\t*-",
            ]
            .join("\n")
            .as_str(),
        )
        .unwrap();

        assert_eq!(
            SerializePreset::TranscodaV2.serialize_document(&document),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n4c\t4e\n=\n4d\t4f\n==\n"
        );
    }

    #[test]
    fn mixed_barlines_remain_per_spine_in_all_renderable_presets() {
        let document =
            KernDocument::parse("**kern\t**kern\n8C\t8cL\n=2\t=2-\n8D\t8dJ\n*-\t*-\n").unwrap();

        assert!(SerializePreset::HumdrumKern
            .serialize_document(&document)
            .contains("=\t=-\n"));
        assert!(SerializePreset::TranscodaV2
            .serialize_document(&document)
            .contains("=\t=-\n"));
    }

    #[test]
    fn transcoda_v2_compact_barlines_emits_synthetic_bar_numbers_once() {
        let document = KernDocument::parse(
            [
                "**kern\t**kern",
                "*M4/4\t*M4/4",
                "4c\t4e",
                "=\t=",
                "4d\t4f",
                "==\t==",
                "*-\t*-",
            ]
            .join("\n")
            .as_str(),
        )
        .unwrap();

        let serialized = SerializePreset::TranscodaV2.serialize_document_with_options(
            &document,
            SerializeOptions {
                bar_number_style: BarNumberStyle::Sequential,
            },
        );

        assert_eq!(
            serialized,
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n=1\n4c\t4e\n=2\n4d\t4f\n==\n"
        );
    }

    #[test]
    fn transcoda_v2_combines_compact_barlines_and_empty_nulls() {
        let document = KernDocument::parse(
            [
                "**kern\t**kern",
                "*M4/4\t*M4/4",
                ".\t4e",
                "=\t=",
                "4c\t.",
                "==\t==",
                "*-\t*-",
            ]
            .join("\n")
            .as_str(),
        )
        .unwrap();

        assert_eq!(
            SerializePreset::TranscodaV2.serialize_document(&document),
            "*clefG2\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n\t4e\n=\n4c\t\n==\n"
        );
    }

    #[test]
    fn try_serialize_transcoda_v2_reports_duration_state_errors() {
        let document = KernDocument::parse(
            [
                "**kern\t**kern",
                "*M4/4\t*M4/4",
                "2c\t4e",
                "*^\t*",
                ".\t4d\t4f",
                "*-\t*-\t*-",
            ]
            .join("\n")
            .as_str(),
        )
        .unwrap();

        let error = SerializePreset::TranscodaV2
            .try_serialize_document(&document)
            .unwrap_err();
        assert!(!error.is_execution_failure());

        assert!(error
            .to_string()
            .contains("duration-bearing token starts before previous token has finished"));
    }

    #[test]
    fn humdrum_kern_preserves_system_breaks_in_position() {
        let document =
            KernDocument::parse("**kern\n*M4/4\n4c\n4d\n4e\n4f\n!!LO:LB:g=original\n=\n4g\n*-\n")
                .unwrap();

        let serialized = SerializePreset::HumdrumKern.serialize_document(&document);

        assert!(
            serialized.contains("4f\n!!LO:LB:g=original\n="),
            "expected break before barline in {serialized:?}"
        );
    }

    #[test]
    fn humdrum_kern_appends_cross_staff_rdf_footer() {
        let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();

        assert!(SerializePreset::HumdrumKern
            .serialize_document(&document)
            .ends_with(
                "*-\n!!!RDF**kern: > = above\n!!!RDF**kern: < = below\n!!!RDF**kern: N = linked\n!!!RDF**kern: Z = editorial accidental, parentheses\n!!!RDF**kern: j = editorial accidental, brackets\n"
            ));
    }

    #[test]
    fn humdrum_kern_rehydrates_parsed_transcoda_v2_slices() {
        let document = KernDocument::parse_with_options(
            "**kern\t**kern\n4c\tryy\n*-\t*-\n",
            ParseOptions {
                dialect: ParseDialect::TranscodaV2,
            },
        )
        .unwrap();

        let serialized = SerializePreset::HumdrumKern.serialize_document(&document);

        assert!(
            serialized.contains("4c\t4ryy\n"),
            "expected duration-bearing invisible rest, got {serialized:?}"
        );
        assert!(
            !serialized.contains("4c\tryy\n"),
            "unexpected durationless rest slice in {serialized:?}"
        );
    }

    #[test]
    fn mark_placement_follows_its_mark_in_humdrum_kern_and_transcoda_v2_drops_the_marks() {
        let document =
            KernDocument::parse("**kern\n4c'<\n4c^^>T<u>\n8qcSsS>\n4c()(>\n*-\n").unwrap();

        let humdrum = SerializePreset::HumdrumKern.serialize_document(&document);
        assert!(
            humdrum.contains("\n4c'<\n4c^^>T<u>\n8qcSsS>\n4c()(>\n"),
            "{humdrum:?}"
        );
        assert_eq!(
            SerializePreset::HumdrumKern
                .serialize_document(&KernDocument::parse(&humdrum).unwrap()),
            humdrum
        );

        let v2 = SerializePreset::TranscodaV2.serialize_document(&document);
        assert_eq!(v2, "*clefG2\n*k[]\n4c\n4c\n8qc\n4c\n");
    }

    #[test]
    fn fermata_placement_survives_humdrum_kern_and_is_omitted_from_transcoda_v2() {
        let document =
            KernDocument::parse("**kern\t**kern\n4c;<\t4r;>\n4c 4e;>\t4r\n*-\t*-\n").unwrap();

        let humdrum = SerializePreset::HumdrumKern.serialize_document(&document);
        assert!(humdrum.contains("4c;<\t4r;>\n"), "{humdrum:?}");
        assert!(humdrum.contains("4c 4e;>\t"), "{humdrum:?}");
        assert!(humdrum.contains("!!!RDF**kern: < = below"), "{humdrum:?}");
        assert_eq!(
            KernDocument::parse(&humdrum)
                .unwrap()
                .lines()
                .iter()
                .filter(|line| matches!(line, crate::KernLine::Data(_)))
                .count(),
            2
        );

        let v2 = SerializePreset::TranscodaV2.serialize_document(&document);
        assert!(!v2.contains(";<") && !v2.contains(";>"), "{v2:?}");
    }

    #[test]
    fn humdrum_kern_preserves_invisible_rest_fermatas() {
        let document = KernDocument::parse("**kern\n4ryy;\n*-\n").unwrap();

        let serialized = SerializePreset::HumdrumKern.serialize_document(&document);

        assert!(
            serialized.contains("4ryy;"),
            "expected HumdrumKern to preserve invisible-rest fermata, got {serialized:?}"
        );
    }

    #[test]
    fn transcoda_v2_rehydrates_parsed_slices_and_omits_invisible_rest_fermatas() {
        let document = KernDocument::parse_with_options(
            "**kern\t**kern\n4c\tryy;\n*-\t*-\n",
            ParseOptions {
                dialect: ParseDialect::TranscodaV2,
            },
        )
        .unwrap();

        let serialized = SerializePreset::TranscodaV2.serialize_document(&document);

        assert!(
            serialized.contains("4c\t4ryy\n"),
            "expected duration-bearing invisible rest without fermata, got {serialized:?}"
        );
        assert!(
            !serialized.contains("ryy;"),
            "expected TranscodaV2 to omit invisible-rest fermata, got {serialized:?}"
        );
    }

    #[test]
    fn transcoda_v2_omits_duration_bearing_invisible_rest_fermatas() {
        let document = KernDocument::parse("**kern\t**kern\n2c\t4ryy;\n.\t4ryy\n*-\t*-\n").unwrap();

        let serialized = SerializePreset::TranscodaV2.serialize_document(&document);

        // The fermata is dropped from emitted output, so it must not block
        // compaction either: both invisible rests merge into one.
        assert!(
            serialized.contains("2c\t2ryy\n"),
            "expected compacted invisible rest without fermata, got {serialized:?}"
        );
        assert!(
            !serialized.contains("ryy;"),
            "expected TranscodaV2 to omit invisible-rest fermata, got {serialized:?}"
        );
    }

    #[test]
    fn transcoda_v2_preserves_visible_rest_fermatas() {
        let document = KernDocument::parse("**kern\n4r;\n*-\n").unwrap();

        let serialized = SerializePreset::TranscodaV2.serialize_document(&document);

        assert!(
            serialized.contains("4r;"),
            "expected visible-rest fermata to remain observable, got {serialized:?}"
        );
    }

    #[test]
    fn humdrum_kern_preserves_cross_staff_placement_markers() {
        let document = KernDocument::parse("**kern\n4c>\n4d<\n*-\n").unwrap();

        let serialized = SerializePreset::HumdrumKern.serialize_document(&document);

        assert!(serialized.contains("4c>\n4d<"));
    }

    #[test]
    fn humdrum_kern_preserves_visual_style_tandems() {
        let document = KernDocument::parse("**kern\n*2\\right\n4c\n*2\\left\n4d\n*-\n").unwrap();

        let serialized = SerializePreset::HumdrumKern.serialize_document(&document);

        assert!(serialized.contains("*2\\right\n4c\n*2\\left\n4d"));
    }

    #[test]
    fn transcoda_v2_preserves_system_breaks_in_position() {
        let document =
            KernDocument::parse("**kern\n*M4/4\n4c\n4d\n4e\n4f\n!!LO:LB:g=original\n=\n4g\n*-\n")
                .unwrap();

        let serialized = SerializePreset::TranscodaV2.serialize_document(&document);

        assert!(
            serialized.contains("4f\n!!LO:LB:g=original\n="),
            "expected break before barline in {serialized:?}"
        );
    }

    #[test]
    fn transcoda_v2_serializes_canonical_system_break_after_tabbed_overlay_suffix() {
        let document = KernDocument::parse(
            "**kern\t**kern\n*M4/4\t*M4/4\n4c\t4e\n!!LO:LB:g=original\t!\n=\t=\n4d\t4f\n*-\t*-\n",
        )
        .unwrap();

        let serialized = SerializePreset::TranscodaV2.serialize_document(&document);

        assert!(
            serialized.contains("4c\t4e\n!!LO:LB:g=original\n=\n"),
            "expected canonical break before barline in {serialized:?}"
        );
        assert!(
            !serialized.contains("!!LO:LB:g=original\t!"),
            "expected tabbed overlay suffix to be omitted in {serialized:?}"
        );
    }

    #[test]
    fn transcoda_v2_presets_omit_stem_direction_markers() {
        let document = KernDocument::parse("**kern\n4c/\n4d\\\n4e 4g/\n*-\n").unwrap();

        {
            let preset = SerializePreset::TranscodaV2;
            let serialized = preset.serialize_document(&document);
            assert!(
                !serialized.contains('/') && !serialized.contains('\\'),
                "expected {preset:?} to omit stem directions, got {serialized:?}"
            );
        }
    }

    #[test]
    fn transcoda_v2_preserves_cross_staff_placement_markers() {
        let document = KernDocument::parse("**kern\n4c>\n4d<\n*-\n").unwrap();

        assert_eq!(
            SerializePreset::TranscodaV2.serialize_document(&document),
            "*clefG2\n*k[]\n4c>\n4d<\n"
        );
    }

    const CUE_SPLIT: &str = "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n*cue\n4c\n4d\n*^\n4e\t4g\n4f\t4a\n*v\t*v\n=2\n*Xcue\n1c\n=3\n*-\n";

    #[test]
    fn transcoda_v2_labels_draw_cue_notes_at_normal_size() {
        // The Transcoda V2 tokenizer cannot spell `*cue` (D7, amended
        // 2026-10-13): V2 labels keep the notes and drop their cue size.
        let document = KernDocument::parse(CUE_SPLIT).unwrap();
        let events = |document: &KernDocument| {
            let score =
                crate::score::Score::resolve(document, crate::document::Framing::Excerpt).unwrap();
            crate::analysis::semantic_events(&score)
                .unwrap()
                .into_iter()
                .map(|event| (event.onset, event.pitch))
                .collect::<Vec<_>>()
        };
        for preset in [
            SerializePreset::TranscodaV2,
            SerializePreset::TranscodaV2Training,
        ] {
            let labels = preset.try_serialize_document(&document).unwrap();
            assert!(!labels.contains("cue"), "{preset:?}: {labels}");
            let reparsed = KernDocument::parse_with_options(
                &labels,
                ParseOptions {
                    dialect: ParseDialect::TranscodaV2,
                },
            )
            .unwrap();
            assert_eq!(events(&reparsed), events(&document), "{preset:?}");
        }
    }

    #[test]
    fn v2_labels_drop_a_rhythm_scale_and_keep_logical_durations() {
        let document = KernDocument::parse(
            "**kern\n*M4/4\n4c\n*cue\n*rscale:2\n8d\n8e\n*rscale:1\n*Xcue\n4f\n4g\n*-\n",
        )
        .unwrap();
        for preset in [
            SerializePreset::TranscodaV2,
            SerializePreset::TranscodaV2Training,
        ] {
            let labels = preset.try_serialize_document(&document).unwrap();
            assert!(!labels.contains("rscale"), "{preset:?}: {labels}");
            assert!(!labels.contains("cue"), "{preset:?}: {labels}");
            assert!(labels.contains("4c\n8d\n8e\n4f\n"), "{preset:?}: {labels}");
        }
    }

    #[test]
    fn humdrum_kern_passes_cue_and_rhythm_scale_through() {
        let document =
            KernDocument::parse("**kern\n*cue\n4c\n*Xcue\n*rscale:2\n8d\n8e\n*-\n").unwrap();
        for preset in [SerializePreset::HumdrumKern, SerializePreset::KernOnly] {
            let text = preset.serialize_document(&document);
            assert!(
                text.contains("*cue\n4c\n*Xcue\n*rscale:2\n8d\n8e\n"),
                "{preset:?}: {text}"
            );
        }
    }
}
