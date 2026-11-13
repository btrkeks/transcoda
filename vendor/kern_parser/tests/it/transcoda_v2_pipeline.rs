use kern_parser::{
    normalization::invisible_rest_slices::InvisibleRestSliceError,
    serialize::{
        omr::{
            try_to_transcoda_v2_compact_barlines_empty_nulls_string_with_options,
            try_to_transcoda_v2_compact_barlines_string_with_options,
            try_to_transcoda_v2_empty_nulls_string_with_options,
            try_to_transcoda_v2_string_with_options,
        },
        transcoda_v2::{
            canonicalize_transcoda_v2, print_transcoda_v2, try_lower_transcoda_v2,
            TranscodaV2Syntax,
        },
        BarNumberStyle, ContentPolicy, SerializeError, SerializeOptions, SerializePreset,
    },
    KernDocument,
};

type V2FallibleSignature =
    fn(&KernDocument, SerializeOptions) -> Result<String, InvisibleRestSliceError>;

#[test]
fn public_fallible_v2_signatures_and_error_mapping_remain_compatible() {
    let _: V2FallibleSignature = try_to_transcoda_v2_string_with_options;
    let _: V2FallibleSignature = try_to_transcoda_v2_empty_nulls_string_with_options;
    let _: V2FallibleSignature = try_to_transcoda_v2_compact_barlines_string_with_options;
    let _: V2FallibleSignature =
        try_to_transcoda_v2_compact_barlines_empty_nulls_string_with_options;

    let mapped =
        SerializeError::from(InvisibleRestSliceError::TranscodaV2DidNotConverge { iterations: 4 });
    assert!(matches!(
        mapped,
        SerializeError::CanonicalizationDidNotConverge { iterations: 4 }
    ));
}

#[test]
fn staged_pipeline_matches_the_public_v2_preset() {
    let document =
        KernDocument::parse("**kern\t**dynam\n*clefG2\t*\n*k[]\t*\n4c[\t.\n4c]\t.\n=\t=\n*-\t*-")
            .expect("fixture should parse");
    let expected = SerializePreset::TranscodaV2.serialize_document_with_options(
        &document,
        SerializeOptions {
            bar_number_style: BarNumberStyle::Sequential,
        },
    );

    let projected = try_lower_transcoda_v2(&document, ContentPolicy::TranscodaV2)
        .expect("lowering should succeed");
    let canonical = canonicalize_transcoda_v2(projected).expect("canonicalization should converge");
    let actual = print_transcoda_v2(
        &canonical,
        TranscodaV2Syntax::CompactBarlinesEmptyNulls,
        SerializeOptions {
            bar_number_style: BarNumberStyle::Sequential,
        },
    );

    assert_eq!(actual, expected);
}

#[test]
fn lowering_owns_invisible_note_and_modifier_omission() {
    let document = KernDocument::parse("**kern\n*ped\n4c#X[\n4dyy\n4e'\n*Xped\n*-")
        .expect("fixture should parse");
    let projected = try_lower_transcoda_v2(&document, ContentPolicy::TranscodaV2)
        .expect("lowering should succeed");
    let canonical = canonicalize_transcoda_v2(projected).expect("canonicalization should converge");
    let output = print_transcoda_v2(
        &canonical,
        TranscodaV2Syntax::CompactBarlinesEmptyNulls,
        SerializeOptions::default(),
    );

    assert!(!output.contains("*ped"));
    assert!(!output.contains("*Xped"));
    assert!(!output.contains("4d"));
    assert!(!output.contains('X'));
    assert!(!output.contains("4c#["), "{output:?}");
    assert!(!output.contains('\''));
    assert!(output.contains("4c"));
    assert!(output.contains("4e"));
}
