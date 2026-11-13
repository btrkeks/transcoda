//! Serialize/parse round trips must reach a fixpoint after one pass.
//!
//! `SerializePreset::TranscodaV2` output is re-parsed by downstream tooling
//! and serialized again; if a second pass produces different text the
//! normalization pipeline is drifting instead of canonicalizing.

use kern_parser::{serialize::SerializePreset, KernDocument};

fn serialize_v2(text: &str) -> String {
    let document = KernDocument::parse(text).expect("input should parse");
    SerializePreset::TranscodaV2.serialize_document(&document)
}

#[test]
fn transcoda_v2_canonicalized_adjacent_closed_splits_reach_fixpoint_in_one_pass() {
    // Measure 1 carries an already-canonical closed split episode; measure 2
    // carries a mid-measure closed split. Canonicalizing measure 2 emits a
    // split right after the barline, directly below measure 1's merge, and
    // that redundant merge/barline/split window must collapse in the same
    // serialization pass instead of on the next round trip.
    let input = [
        "**kern\t**kern",
        "*clefG2\t*clefG2",
        "=",
        "*\t*^",
        "4c\t4e\t4g",
        "*\t*v\t*v",
        "=",
        "2d\t4f",
        "*\t*^",
        ".\t4a\t4b",
        "*\t*v\t*v",
        "=",
        "*-\t*-",
    ]
    .join("\n");

    let first = serialize_v2(&input);
    assert_eq!(
        first,
        [
            "*clefG2\t*clefG2",
            "*k[]\t*k[]",
            "*\t*^",
            "4c\t4e\t4g",
            "=",
            "2d\t4f\t4ryy",
            "\t4a\t4b",
            "*\t*v\t*v",
            "=",
        ]
        .join("\n")
            + "\n"
    );

    let second = serialize_v2(&first);
    assert_eq!(
        second, first,
        "second serialization pass must be a fixpoint"
    );
}

#[test]
fn transcoda_v2_removed_leading_invisible_rest_measure_leaves_no_leading_barline() {
    // Removing a leading all-invisible-rest measure must not leave its
    // closing barline behind as a new leading barline that only the next
    // round trip removes.
    let input = [
        "**kern", "*clefG2", "*k[]", "*M4/4", "1ryy", "=", "4c", "4d", "4e", "4f", "=", "*-",
    ]
    .join("\n");

    let first = serialize_v2(&input);
    assert!(
        !first.contains("ryy"),
        "leading invisible-rest measure should be removed, got {first:?}"
    );
    assert!(
        first.contains("4c"),
        "sanity: music content must survive, got {first:?}"
    );

    let second = serialize_v2(&first);
    assert_eq!(
        second, first,
        "second serialization pass must be a fixpoint"
    );
}

#[test]
fn transcoda_v2_canonicalizes_closed_splits_despite_non_kern_spines() {
    // A **fing spine used to block measure-slot canonicalization on the
    // first pass while the serialized output drops the spine, so only the
    // second round trip canonicalized the measure.
    let input = [
        "**kern\t**fing\t**kern",
        "*clefG2\t*\t*clefG2",
        "=\t=\t=",
        "2d\t1\t4f",
        "*\t*\t*^",
        ".\t.\t4a\t4b",
        "*\t*\t*v\t*v",
        "=\t=\t=",
        "*-\t*-\t*-",
    ]
    .join("\n");

    let first = serialize_v2(&input);
    assert!(
        first.contains("ryy"),
        "closed split should canonicalize with invisible-rest fill on the first pass, got {first:?}"
    );
    let second = serialize_v2(&first);
    assert_eq!(
        second, first,
        "second serialization pass must be a fixpoint"
    );
}

#[test]
fn transcoda_v2_canonicalizes_closed_splits_despite_local_comments() {
    // A local comment inside the measure used to block canonicalization on
    // the first pass while the serialized output omits the comment, so only
    // the second round trip canonicalized the measure.
    let input = [
        "**kern\t**kern",
        "*clefG2\t*clefG2",
        "=\t=",
        "2d\t4f",
        "!\t!LO:TX:a:t=x",
        "*\t*^",
        ".\t4a\t4b",
        "*\t*v\t*v",
        "=\t=",
        "*-\t*-",
    ]
    .join("\n");

    let first = serialize_v2(&input);
    assert!(
        first.contains("ryy"),
        "closed split should canonicalize with invisible-rest fill on the first pass, got {first:?}"
    );
    let second = serialize_v2(&first);
    assert_eq!(
        second, first,
        "second serialization pass must be a fixpoint"
    );
}

#[test]
fn transcoda_v2_compacts_adjacent_dormant_fills_in_the_same_pass() {
    // Canonicalizing a mid-measure merge back-fills the dormant slot with
    // one invisible-rest slice per remaining data line. Those fills must be
    // compacted into a single rest by the same serialize call; an earlier
    // regression emitted `4ryy` + `8ryy` + `8ryy` and only the next
    // parse/serialize round trip produced the compacted `2ryy`.
    let input = [
        "**kern\t**kern",
        "*clefG2\t*clefG2",
        "=",
        "2.g\t2.b",
        "=",
        "*\t*^",
        "4c\t4e\t4g",
        "*\t*v\t*v",
        "4d\t4f",
        "8e\t8g",
        "8f\t8a",
        "=",
        "*-\t*-",
    ]
    .join("\n");

    let first = serialize_v2(&input);
    assert_eq!(
        first,
        [
            "*clefG2\t*clefG2",
            "*k[]\t*k[]",
            "2.g\t2.b",
            "=",
            "*\t*^",
            "4c\t4e\t4g",
            "4d\t4f\t2ryy",
            "8e\t8g\t",
            "8f\t8a\t",
            "*\t*v\t*v",
            "=",
        ]
        .join("\n")
            + "\n"
    );

    let second = serialize_v2(&first);
    assert_eq!(
        second, first,
        "second serialization pass must be a fixpoint"
    );
}

#[test]
fn transcoda_v2_splits_combined_merge_lines_left_to_right() {
    // With a **fing spine between two kern staves, both staves can merge on
    // a single manipulator line. Dropping the fing column forces the
    // serializer to emit one merge line per group; the multi-line form must
    // come out left to right, matching the canonical order the normalizer
    // produces on re-parse. Emitting right-first made every round trip swap
    // the two lines.
    let input = [
        "**kern\t**fing\t**kern",
        "*clefF4\t*\t*clefG2",
        "=\t=\t=",
        "*^\t*\t*^",
        "4c\t4c\t1\t4e\t4g",
        "4d\t4d\t.\t4f\t4a",
        "*v\t*v\t*\t*v\t*v",
        "=\t=\t=",
        "*-\t*-\t*-",
    ]
    .join("\n");

    let first = serialize_v2(&input);
    assert!(
        first.contains("*v\t*v\t*\t*\n*\t*v\t*v\n"),
        "combined merge line should serialize left group first, got {first:?}"
    );
    let second = serialize_v2(&first);
    assert_eq!(
        second, first,
        "second serialization pass must be a fixpoint"
    );
}

#[test]
fn transcoda_v2_compacts_dormant_fills_across_dropped_non_kern_data_lines() {
    // A zero-duration data line kept alive only by a **dynam token is
    // dropped from the output, so it must not split the invisible-rest run
    // around it; splitting in memory while the emitted text drops the line
    // meant only the next round trip compacted the fills.
    let input = [
        "**kern\t**kern\t**dynam",
        "*clefF4\t*clefG2\t*",
        "=\t=\t=",
        "2.g\t2.b\t.",
        "=\t=\t=",
        "*\t*^\t*",
        "4c\t4e\t4g\t.",
        "*\t*v\t*v\t*",
        "4d\t4f\tp",
        ".\t.\t[",
        "8e\t8g\t.",
        "8f\t8a\t.",
        "=\t=\t=",
        "*-\t*-\t*-",
    ]
    .join("\n");

    let first = serialize_v2(&input);
    assert!(
        first.contains("2ryy"),
        "dormant fills should compact across the dropped dynam-only line, got {first:?}"
    );
    let second = serialize_v2(&first);
    assert_eq!(
        second, first,
        "second serialization pass must be a fixpoint"
    );
}

#[test]
fn transcoda_v2_compacts_invisible_rests_across_dropped_local_comments() {
    // Layout-text local comments are dropped from the output and consume no
    // time, so they must not split invisible-rest runs.
    let input = [
        "**kern\t**kern",
        "*clefF4\t*clefG2",
        "=\t=",
        "2.g\t2.b",
        "=\t=",
        "8ryy\t8c",
        "!\t!LO:TX:a:i:t=cresc.",
        "8ryy\t8d",
        "4ryy\t4e",
        "=\t=",
        "*-\t*-",
    ]
    .join("\n");

    let first = serialize_v2(&input);
    assert!(
        first.contains("2ryy"),
        "invisible rests should compact across the dropped comment, got {first:?}"
    );
    let second = serialize_v2(&first);
    assert_eq!(
        second, first,
        "second serialization pass must be a fixpoint"
    );
}

#[test]
fn transcoda_v2_canonicalizes_closed_splits_despite_section_labels() {
    // Section-label tandem lines are dropped from the output, so they must
    // not block measure-slot canonicalization; blocking in memory meant the
    // split only moved to the measure start on the next round trip.
    let input = [
        "**kern\t**kern",
        "*clefG2\t*clefG2",
        "=\t=",
        "2.g\t2.b",
        "=\t=",
        "*>B\t*>B",
        "2d\t4f",
        "*\t*^",
        ".\t4a\t4b",
        "*\t*v\t*v",
        "=\t=",
        "*-\t*-",
    ]
    .join("\n");

    let first = serialize_v2(&input);
    assert!(
        first.contains("ryy"),
        "closed split should canonicalize with invisible-rest fill on the first pass, got {first:?}"
    );
    let second = serialize_v2(&first);
    assert_eq!(
        second, first,
        "second serialization pass must be a fixpoint"
    );
}

#[test]
fn transcoda_v2_compacts_invisible_rests_with_dropped_placement_pitches() {
    // Placement pitches on invisible rests (`4reyy`, `4rgyy`) are never
    // emitted, so they must not stop the rests from counting as plain
    // invisible rests during compaction; blocking in memory meant only the
    // re-parsed bare `4ryy` forms compacted on the next round trip.
    let input = [
        "**kern\t**kern\t**dynam",
        "*clefF4\t*clefG2\t*",
        "=\t=\t=",
        "2g\t2b\t.",
        "4reyy\t4c\t.",
        "4rgyy\t4d\t.",
        "=\t=\t=",
        "*-\t*-\t*-",
    ]
    .join("\n");

    let first = serialize_v2(&input);
    assert!(
        first.contains("2ryy"),
        "placed invisible rests should compact to one bare rest, got {first:?}"
    );
    let second = serialize_v2(&first);
    assert_eq!(
        second, first,
        "second serialization pass must be a fixpoint"
    );
}

#[test]
fn transcoda_v2_packs_merge_groups_split_by_non_kern_columns() {
    // Three kern merge groups separated only by **dynam columns collide when
    // the non-kern columns are dropped. The projected multi-line form must
    // use the same greedy packing `MergeAdjacentSpineManipulators` settles
    // on after a re-parse (groups one and three share a line), otherwise the
    // next round trip repacks the lines and the output drifts.
    let input = [
        "**kern\t**dynam\t**kern\t**dynam\t**kern",
        "*clefF4\t*\t*clefG2\t*\t*clefG2",
        "=\t=\t=\t=\t=",
        "*^\t*\t*^\t*\t*^",
        "4c\t4C\t.\t4e\t4E\t.\t4g\t4G",
        "*v\t*v\t*\t*v\t*v\t*\t*v\t*v",
        "=\t=\t=\t=\t=",
        "*-\t*-\t*-\t*-\t*-",
    ]
    .join("\n");

    let first = serialize_v2(&input);
    assert!(
        first.contains("*v\t*v\t*\t*\t*v\t*v\n*\t*v\t*v\t*\n"),
        "merge groups should pack one and three onto a shared line, got {first:?}"
    );
    let second = serialize_v2(&first);
    assert_eq!(
        second, first,
        "second serialization pass must be a fixpoint"
    );
}
