//! Staff and part numbers of every track, as Verovio assigns them.
//!
//! Verovio numbers the `**kern` spines of an exclusive interpretation line by
//! position; that number is MEI `staff/@n`. It counts from the right unless
//! the `*staffN` that `**kern` spines declare before the first data line never
//! decrease from left to right; then it counts from the left. A declared
//! number only chooses the direction: `*staff5`, `*staff3` gives staves 2 and 1.
//! Declarations after the first data line, in subspines other than the
//! leftmost, or in other spine types do not number staves.
//!
//! A track of another spine type belongs to the staff of the nearest `**kern`
//! track on its left, as Verovio attaches `**dynam` spines. Tracks implied by a
//! document without exclusive interpretations are numbered the same way, with
//! the lines before the first data line as their header: render output gives
//! each implied column its own `**kern` spine.

use std::collections::BTreeSet;

use crate::{KernDocument, KernLine, SpineType, TandemInterpretation};

use super::topology::{Topology, TopologyError, TrackId};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TrackNumbers {
    pub(super) staff: Option<u32>,
    pub(super) part: Option<u32>,
}

/// The `*staffN` a track declares in its header, and where.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StaffDeclaration {
    number: u32,
    line: usize,
    column: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct HeaderDeclarations {
    staff: Option<StaffDeclaration>,
    part: Option<u32>,
}

/// Staff and part number of every track, indexed by [`TrackId`].
pub(super) fn number(document: &KernDocument, topology: &Topology) -> Vec<TrackNumbers> {
    let declared = header_declarations(document, topology);
    let mut numbers = declared
        .iter()
        .map(|declared| TrackNumbers {
            staff: None,
            part: declared.part,
        })
        .collect::<Vec<_>>();

    for group in exclusive_groups(topology) {
        let kern = kern_tracks(topology, &group);
        let from_left = counts_from_left(&declared, &kern);
        for (ordinal, track) in kern.iter().enumerate() {
            numbers[track.index()].staff = Some(if from_left {
                ordinal as u32 + 1
            } else {
                (kern.len() - ordinal) as u32
            });
        }
        let mut owner = None;
        for track in group {
            if topology.track(track).spine_type == SpineType::Kern {
                owner = numbers[track.index()].staff;
            } else {
                numbers[track.index()].staff = owner;
            }
        }
    }
    numbers
}

/// Every `*staff` token Verovio does not read, as (line index, column).
///
/// Verovio numbers staves from the header declarations of a group of two or
/// more `**kern` tracks that count from the left; without them the group
/// counts from the right. Every other `*staff` is unread: a group that already
/// counts from the right, a lone `**kern` track, other spine types, later
/// declarations and other subspines.
///
/// A shared declaration (`*staff1/2`) places `**dynam` content between the
/// staves the `**kern` spines declare with those numbers, so a document with
/// one has no unread `*staff`.
pub(crate) fn unread_staff_declarations(
    document: &KernDocument,
) -> Result<BTreeSet<(usize, usize)>, TopologyError> {
    let topology = Topology::build(document)?;
    let staff_tokens = || {
        document
            .lines()
            .iter()
            .enumerate()
            .filter_map(|(index, line)| match line {
                KernLine::TandemInterpretationLine(tandems) => Some((index, tandems)),
                _ => None,
            })
            .flat_map(|(index, tandems)| {
                tandems
                    .tokens()
                    .iter()
                    .enumerate()
                    .filter_map(move |(column, tandem)| match tandem {
                        TandemInterpretation::Staff(staff) => Some(((index, column), staff)),
                        _ => None,
                    })
            })
    };
    if staff_tokens().any(|(_, staff)| staff.is_shared()) {
        return Ok(BTreeSet::new());
    }
    let declared = header_declarations(document, &topology);
    let mut read = BTreeSet::new();
    for group in exclusive_groups(&topology) {
        let kern = kern_tracks(&topology, &group);
        if kern.len() > 1 && counts_from_left(&declared, &kern) {
            read.extend(
                kern.iter()
                    .filter_map(|track| declared[track.index()].staff)
                    .map(|declaration| (declaration.line, declaration.column)),
            );
        }
    }
    Ok(staff_tokens()
        .map(|(position, _)| position)
        .filter(|position| !read.contains(position))
        .collect())
}

/// Tracks grouped by the exclusive interpretation that opened them, in
/// track order.
fn exclusive_groups(topology: &Topology) -> Vec<Vec<TrackId>> {
    let tracks = topology.track_ids().collect::<Vec<_>>();
    tracks
        .chunk_by(|a, b| topology.track(*a).born == topology.track(*b).born)
        .map(<[TrackId]>::to_vec)
        .collect()
}

fn kern_tracks(topology: &Topology, group: &[TrackId]) -> Vec<TrackId> {
    group
        .iter()
        .copied()
        .filter(|track| topology.track(*track).spine_type == SpineType::Kern)
        .collect()
}

/// Whether Verovio numbers these `**kern` tracks from the left: some declare
/// a staff and the declared numbers never decrease.
fn counts_from_left(declared: &[HeaderDeclarations], kern: &[TrackId]) -> bool {
    let staves = kern
        .iter()
        .filter_map(|track| declared[track.index()].staff)
        .map(|declaration| declaration.number)
        .collect::<Vec<_>>();
    !staves.is_empty() && staves.windows(2).all(|pair| pair[0] <= pair[1])
}

/// The first `*staffN` and `*partN` of each track before its first data
/// line, read in the track's leftmost column: Verovio follows a spine's first
/// branch from its exclusive interpretation.
fn header_declarations(document: &KernDocument, topology: &Topology) -> Vec<HeaderDeclarations> {
    let mut declared = vec![HeaderDeclarations::default(); topology.track_count()];
    let mut in_header = vec![true; topology.track_count()];
    for (index, line) in document.lines().iter().enumerate() {
        let layers = topology.layers_at(index);
        match line {
            KernLine::Data(_) => {
                for &layer in layers {
                    in_header[topology.track_of(layer).index()] = false;
                }
            }
            KernLine::TandemInterpretationLine(tandems) => {
                for (column, (tandem, &layer)) in tandems.tokens().iter().zip(layers).enumerate() {
                    let track = topology.track_of(layer);
                    if !in_header[track.index()]
                        || topology.track_columns(index, track).start != column
                    {
                        continue;
                    }
                    let declared = &mut declared[track.index()];
                    match tandem {
                        TandemInterpretation::Staff(staff) => {
                            declared.staff.get_or_insert(StaffDeclaration {
                                number: staff.number,
                                line: index,
                                column,
                            });
                        }
                        TandemInterpretation::Part(part) => {
                            declared.part.get_or_insert(part.number);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    declared
}

#[cfg(test)]
mod tests {
    use crate::{
        normalization::omr::remove_unread_staff_declarations, score::Topology, KernDocument,
    };

    /// Staff of each track, in track order.
    fn staves(input: &str) -> Vec<Option<u32>> {
        staves_of(&KernDocument::parse(input).unwrap())
    }

    fn staves_of(document: &KernDocument) -> Vec<Option<u32>> {
        let topology = Topology::build(document).unwrap();
        topology
            .track_ids()
            .map(|track| topology.track(track).staff)
            .collect()
    }

    fn parts(input: &str) -> Vec<Option<u32>> {
        let document = KernDocument::parse(input).unwrap();
        let topology = Topology::build(&document).unwrap();
        topology
            .track_ids()
            .map(|track| topology.track(track).part)
            .collect()
    }

    // Each expectation below is Verovio 6.3's MEI `staff/@n` for the input.

    #[test]
    fn counts_from_the_right_without_declarations() {
        assert_eq!(
            staves("**kern\t**kern\n4C\t4c\n*-\t*-\n"),
            [Some(2), Some(1)]
        );
    }

    #[test]
    fn increasing_declarations_count_from_the_left() {
        assert_eq!(
            staves("**kern\t**kern\n*staff1\t*staff2\n4C\t4c\n*-\t*-\n"),
            [Some(1), Some(2)]
        );
        // One declaration cannot decrease, so it counts from the left too.
        assert_eq!(
            staves("**kern\t**kern\n*staff2\t*\n4C\t4c\n*-\t*-\n"),
            [Some(1), Some(2)]
        );
    }

    #[test]
    fn declared_numbers_choose_the_direction_only() {
        assert_eq!(
            staves("**kern\t**kern\n*staff5\t*staff3\n4C\t4c\n*-\t*-\n"),
            [Some(2), Some(1)]
        );
        assert_eq!(
            staves("**kern\t**kern\t**kern\n*staff3\t*\t*staff1\n4C\t4e\t4c\n*-\t*-\t*-\n"),
            [Some(3), Some(2), Some(1)]
        );
    }

    #[test]
    fn declarations_after_the_first_data_line_are_ignored() {
        assert_eq!(
            staves("**kern\t**kern\n*staff2\t*staff1\n4C\t4c\n*staff1\t*staff2\n4D\t4d\n*-\t*-\n"),
            [Some(2), Some(1)]
        );
    }

    #[test]
    fn only_the_leftmost_subspine_declares() {
        assert_eq!(
            staves(
                "**kern\t**kern\n*^\t*\n*staff1\t*staff2\t*staff2\n4C\t4E\t4c\n*v\t*v\t*\n*-\t*-\n"
            ),
            [Some(1), Some(2)]
        );
    }

    #[test]
    fn other_spines_take_the_staff_on_their_left_and_do_not_declare() {
        assert_eq!(
            staves("**dynam\t**kern\t**dynam\t**kern\n*\t*\t*staff2\t*\np\t4C\tf\t4c\n*-\t*-\t*-\t*-\n"),
            [None, Some(2), Some(2), Some(1)]
        );
    }

    #[test]
    fn implied_tracks_are_numbered_as_render_output_wraps_them() {
        assert_eq!(staves("4c\t4e\n"), [Some(2), Some(1)]);
        assert_eq!(staves("*staff1\t*staff2\n4c\t4e\n"), [Some(1), Some(2)]);
    }

    #[test]
    fn parts_come_from_each_tracks_header() {
        assert_eq!(
            parts("**kern\t**dynam\t**kern\n*part2\t*part2\t*part1\n4C\tp\t4c\n*part3\t*\t*\n4D\t.\t4d\n*-\t*-\t*-\n"),
            [Some(2), Some(2), Some(1)]
        );
    }

    /// The `*staff` tokens left after removing the unread ones, as (declared
    /// number, column), checking that the staff numbers do not change.
    fn read_declarations(input: &str) -> Vec<(u32, usize)> {
        let mut document = KernDocument::parse(input).unwrap();
        remove_unread_staff_declarations(&mut document).unwrap();
        assert_eq!(staves_of(&document), staves(input), "{input}");
        assert!(super::unread_staff_declarations(&document)
            .unwrap()
            .is_empty());
        document
            .lines()
            .iter()
            .flat_map(|line| match line {
                crate::KernLine::TandemInterpretationLine(tandems) => tandems
                    .tokens()
                    .iter()
                    .enumerate()
                    .filter_map(|(column, tandem)| match tandem {
                        crate::TandemInterpretation::Staff(staff) => Some((staff.number, column)),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            })
            .collect()
    }

    #[test]
    fn verovio_reads_only_declarations_that_count_from_the_left() {
        // Standard order counts from the right with or without them.
        assert_eq!(
            read_declarations("**kern\t**kern\n*staff2\t*staff1\n4C\t4c\n*-\t*-\n"),
            []
        );
        assert_eq!(
            read_declarations("**kern\t**kern\n*staff1\t*staff2\n4C\t4c\n*-\t*-\n"),
            [(1, 0), (2, 1)]
        );
        assert_eq!(
            read_declarations("**kern\t**kern\n*staff2\t*\n4C\t4c\n*-\t*-\n"),
            [(2, 0)]
        );
        // One staff is staff 1 from either side.
        assert_eq!(read_declarations("**kern\n*staff1\n4c\n*-\n"), []);
    }

    #[test]
    fn declarations_that_number_nothing_are_unread() {
        // A second declaration, a later one, one in another subspine and one
        // on a `**dynam` spine.
        assert_eq!(
            read_declarations(
                "**kern\t**dynam\t**kern\n*staff1\t*staff2\t*staff2\n*staff3\t*\t*\n*^\t*\t*\n*staff1\t*staff1\t*\t*\n4C\t4E\tp\t4c\n*staff2\t*\t*\t*staff1\n*v\t*v\t*\t*\n4D\t.\t4d\n*-\t*-\t*-\n"
            ),
            [(1, 0), (2, 2)]
        );
        // A shared declaration places `**dynam` content between the staves
        // the `**kern` spines declare.
        assert_eq!(
            read_declarations(
                "**kern\t**kern\t**dynam\n*staff2\t*staff1\t*staff1/2\n4C\t4c\tp\n*staff1\t*\t*\n4D\t4d\t.\n*-\t*-\t*-\n"
            ),
            [(2, 0), (1, 1), (1, 2), (1, 0)]
        );
    }
}
