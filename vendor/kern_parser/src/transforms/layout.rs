//! Spine layout at a cut, read from a resolved score.

use crate::{
    score::{Score, TrackId},
    KernLine, SpineManipulator, SpineManipulatorLine, SpineType, TandemInterpretation,
    TandemInterpretationLine,
};

/// The active columns entering one line, as tracks of the exclusive
/// interpretation in force there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Spine type of every track that exclusive interpretation introduced.
    tracks: Vec<SpineType>,
    /// Whether each track is implied (see [`crate::score::Topology::is_implied`]).
    implied: Vec<bool>,
    /// Ordinal in `tracks` of each active column.
    columns: Vec<usize>,
}

impl Layout {
    /// Layout entering `line`; one past the last line is the layout after it.
    pub(crate) fn before(score: &Score, line: usize) -> Self {
        let topology = score.topology();
        let layers = if line < score.document().lines().len() {
            topology.layers_at(line)
        } else {
            topology.layers_after_last()
        };
        let header = match layers.first() {
            Some(&layer) => topology.track(topology.track_of(layer)).born,
            None => topology
                .track_ids()
                .map(|track| topology.track(track).born)
                .filter(|born| *born <= line)
                .max()
                .unwrap_or(0),
        };
        let tracks = topology
            .track_ids()
            .filter(|track| topology.track(*track).born == header)
            .collect::<Vec<TrackId>>();
        Self {
            columns: layers
                .iter()
                .map(|&layer| {
                    let track = topology.track_of(layer);
                    tracks
                        .iter()
                        .position(|candidate| *candidate == track)
                        .expect("an active layer belongs to a track of the header in force")
                })
                .collect(),
            implied: tracks
                .iter()
                .map(|&track| topology.is_implied(track))
                .collect(),
            tracks: tracks
                .iter()
                .map(|&track| topology.track(track).spine_type.clone())
                .collect(),
        }
    }

    pub(crate) fn initial_width(&self) -> usize {
        self.tracks.len()
    }

    pub(crate) fn initial_spine_types(&self) -> &[SpineType] {
        &self.tracks
    }

    /// A tandem line with one token per live track (as
    /// [`crate::score::Snapshot::tracks`] lays it out), widened to the
    /// exclusive interpretation: tracks that already ended get `*`.
    pub(crate) fn widen_tracks(&self, line: &TandemInterpretationLine) -> TandemInterpretationLine {
        let mut live = self.columns.clone();
        live.dedup();
        let mut tokens = vec![TandemInterpretation::Null; self.tracks.len()];
        for (track, token) in live.into_iter().zip(line.tokens()) {
            tokens[track] = token.clone();
        }
        TandemInterpretationLine::from_homogeneous_tokens(tokens)
    }

    /// Terminations and splits that turn the exclusive interpretation's
    /// columns into this layout.
    pub(crate) fn manipulator_lines(&self) -> Vec<KernLine> {
        let mut lines = Vec::new();
        let target = counts(&self.columns, self.tracks.len());
        let mut current = (0..self.tracks.len()).collect::<Vec<_>>();

        if target.contains(&0) {
            let tokens = current
                .iter()
                .map(|track| {
                    if target[*track] == 0 {
                        SpineManipulator::Terminate
                    } else {
                        SpineManipulator::Null
                    }
                })
                .collect();
            current.retain(|track| target[*track] > 0);
            lines.push(manipulator_line(tokens));
        }
        loop {
            let mut have = counts(&current, self.tracks.len());
            let mut next = Vec::with_capacity(current.len());
            let tokens = current
                .iter()
                .map(|&track| {
                    next.push(track);
                    if have[track] < target[track] {
                        have[track] += 1;
                        next.push(track);
                        SpineManipulator::Split
                    } else {
                        SpineManipulator::Null
                    }
                })
                .collect::<Vec<_>>();
            if tokens.iter().all(|token| *token == SpineManipulator::Null) {
                break;
            }
            current = next;
            lines.push(manipulator_line(tokens));
        }
        lines
    }
}

fn manipulator_line(tokens: Vec<SpineManipulator>) -> KernLine {
    KernLine::SpineManipulatorLine(
        SpineManipulatorLine::new(tokens).expect("a synthesized layout line is valid"),
    )
}

fn counts(columns: &[usize], tracks: usize) -> Vec<usize> {
    let mut counts = vec![0; tracks];
    for track in columns {
        counts[*track] += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        document::Framing,
        KernDocument,
        SpineManipulator::{Null, Split, Terminate},
    };

    /// Layout entering the last data line.
    fn layout(input: &str) -> Layout {
        let document = KernDocument::parse(input).unwrap();
        let score = Score::resolve(&document, Framing::Excerpt).unwrap();
        let line = document
            .lines()
            .iter()
            .rposition(|line| matches!(line, KernLine::Data(_)))
            .unwrap();
        Layout::before(&score, line)
    }

    fn manipulators(layout: &Layout) -> Vec<Vec<SpineManipulator>> {
        layout
            .manipulator_lines()
            .into_iter()
            .map(|line| match line {
                KernLine::SpineManipulatorLine(manipulators) => manipulators.tokens().to_vec(),
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn synthesizes_splits_from_original_topology() {
        let layout = layout("**kern\t**kern\n*^\t*\n*\t*^\t*\n4c\t4d\t4e\t4f\n");
        assert_eq!(
            manipulators(&layout),
            vec![vec![Split, Null], vec![Split, Null, Null]]
        );
    }

    #[test]
    fn synthesizes_terminations_from_original_topology() {
        let layout = layout("**kern\t**kern\t**kern\n*\t*-\t*\n4c\t4e\n");
        assert_eq!(manipulators(&layout), vec![vec![Null, Terminate, Null]]);
        assert_eq!(layout.initial_width(), 3);
    }

    #[test]
    fn merged_split_needs_no_manipulators() {
        let layout = layout("**kern\n*^\n4c\t4e\n*v\t*v\n4d\n");
        assert!(manipulators(&layout).is_empty());
    }

    #[test]
    fn a_wrapperless_page_starting_inside_a_split_is_one_track() {
        // The merge proves the first two columns share a staff.
        let document = KernDocument::parse("4c\t4e\t4G\n*v\t*v\t*\n2c\t2G\n").unwrap();
        let score = Score::resolve(&document, Framing::Excerpt).unwrap();
        let layout = Layout::before(&score, 0);
        assert_eq!(layout.initial_width(), 2);
        assert_eq!(manipulators(&layout), vec![vec![Split, Null]]);
    }
}
