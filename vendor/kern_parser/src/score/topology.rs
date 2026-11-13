//! Spine identity for every field of a [`KernDocument`].
//!
//! A track is an original spine, introduced by an exclusive interpretation.
//! A layer is one subspine lifetime: `*^` ends its input and starts two new
//! layers, `*v` ends its inputs and starts one, `*-` ends one. Layer ids are
//! never reused, so a layer that reappears at the same column after a merge
//! and a re-split is a different layer.

use std::{ops::Range, rc::Rc};

use thiserror::Error;

use crate::{KernDocument, KernLine, SpineManipulator, SpineType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrackId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayerId(u32);

impl TrackId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl LayerId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerOrigin {
    /// Introduced by an exclusive interpretation, or implied by the first
    /// line of a document without one.
    Exclusive,
    SplitOf(LayerId),
    MergeOf(Vec<LayerId>),
}

/// Which child of a `*^` a layer is. The left child stays in the parent's
/// column and Verovio keeps the parent's layer number for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerEnd {
    Split,
    Merged,
    Terminated,
    /// A later exclusive-interpretation line replaced every active spine.
    Replaced,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer {
    pub track: TrackId,
    pub origin: LayerOrigin,
    /// Document line index where the layer first holds a field.
    pub born: usize,
    /// Document line index of the operation that ended the layer.
    pub end: Option<(usize, LayerEnd)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    pub spine_type: SpineType,
    /// Document line index of the exclusive interpretation, or of the first
    /// line when the document has none.
    pub born: usize,
    /// Staff number as Verovio draws it (MEI `staff/@n`); see
    /// [`super::staff`]. `None` for a track left of every `**kern` track.
    pub staff: Option<u32>,
    /// The first `*partN` the track declares before its first data line.
    pub part: Option<u32>,
}

/// Layer identity of every field of every line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
    /// `entering[line]` is the active layer list a line is laid out against.
    /// For a manipulator line it is the input of the manipulation.
    entering: Vec<Rc<[LayerId]>>,
    after_last: Rc<[LayerId]>,
    layers: Vec<Layer>,
    tracks: Vec<Track>,
    /// Tracks formed by merging implied columns, with the 1-based line and
    /// column of the first merge that joined them.
    joins: Vec<(TrackId, usize, usize)>,
    /// Whether each track is implied by a document without an exclusive
    /// interpretation.
    implied: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TopologyError {
    #[error("line {line}: {context} has {found} spines, expected {expected}")]
    WidthMismatch {
        line: usize,
        context: &'static str,
        expected: usize,
        found: usize,
    },
    #[error("line {line}: merge requires at least 2 consecutive *v tokens")]
    MergeRequiresAtLeastTwo { line: usize },
    #[error("line {line} column {column}: cannot merge spines from different original spines")]
    CrossTrackMerge { line: usize, column: usize },
}

impl TopologyError {
    pub fn line(&self) -> usize {
        match self {
            Self::WidthMismatch { line, .. }
            | Self::MergeRequiresAtLeastTwo { line }
            | Self::CrossTrackMerge { line, .. } => *line,
        }
    }
}

impl Topology {
    pub fn build(document: &KernDocument) -> Result<Self, TopologyError> {
        let mut builder = Builder::default();
        // Comments and tandems may precede the exclusive interpretation that
        // opens a document; they are laid out against its spines.
        if let Some((index, spine_types)) =
            crate::initial_state::initial_spine_type_declaration(document)
                .filter(|(index, _)| *index > 0)
        {
            builder.active = spine_types
                .iter()
                .map(|spine_type| {
                    let layer = builder.new_track(index, spine_type.clone());
                    builder.layers[layer.index()].born = 0;
                    layer
                })
                .collect();
            builder.declared_ahead = Some(index);
        }
        for (index, line) in document.lines().iter().enumerate() {
            builder.line(index, document.source_line_number(index), line)?;
        }
        let mut topology = builder.finish();
        let numbers = super::staff::number(document, &topology);
        for (track, numbers) in topology.tracks.iter_mut().zip(numbers) {
            track.staff = numbers.staff;
            track.part = numbers.part;
        }
        Ok(topology)
    }

    /// Active layers a line is laid out against. Barline lines have no
    /// fields of their own but still report the active layers.
    pub fn layers_at(&self, line: usize) -> &[LayerId] {
        &self.entering[line]
    }

    pub fn layers_after_last(&self) -> &[LayerId] {
        &self.after_last
    }

    /// Active layers once `line` has applied: the next line's layout, or the
    /// layout after the last line.
    pub fn layers_after(&self, line: usize) -> &[LayerId] {
        self.entering.get(line + 1).unwrap_or(&self.after_last)
    }

    pub fn layer_at(&self, line: usize, column: usize) -> LayerId {
        self.entering[line][column]
    }

    pub fn layer(&self, id: LayerId) -> &Layer {
        &self.layers[id.index()]
    }

    pub fn track(&self, id: TrackId) -> &Track {
        &self.tracks[id.index()]
    }

    /// The side of its split a layer was born on, or `None` when a split did
    /// not create it.
    pub fn split_side(&self, layer: LayerId) -> Option<SplitSide> {
        let LayerOrigin::SplitOf(parent) = self.layer(layer).origin else {
            return None;
        };
        // A split creates its left child first, so the right child's
        // sibling has the id just before it.
        let right =
            layer.0 > 0 && self.layers[layer.index() - 1].origin == LayerOrigin::SplitOf(parent);
        Some(if right {
            SplitSide::Right
        } else {
            SplitSide::Left
        })
    }

    pub fn track_of(&self, layer: LayerId) -> TrackId {
        self.layer(layer).track
    }

    /// Staff number of the track a layer belongs to.
    pub fn staff_of(&self, layer: LayerId) -> Option<u32> {
        self.track(self.track_of(layer)).staff
    }

    pub fn spine_type(&self, layer: LayerId) -> &SpineType {
        &self.track(self.track_of(layer)).spine_type
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Where a track was joined from implied columns, as the 1-based line and
    /// column of the merge. A page without exclusive interpretations that
    /// starts inside a split shows sibling layers as separate columns; a
    /// later merge of those columns is read as proof they share a track.
    pub fn joined_at(&self, track: TrackId) -> Option<(usize, usize)> {
        self.joins
            .iter()
            .find(|(joined, _, _)| *joined == track)
            .map(|&(_, line, column)| (line, column))
    }

    /// Whether a track is implied by a document without an exclusive
    /// interpretation: a column of the first line, which a later merge may
    /// join with its neighbours (see [`Self::joined_at`]).
    pub fn is_implied(&self, track: TrackId) -> bool {
        self.implied[track.index()]
    }

    /// Every track, in the order exclusive interpretations introduce them.
    pub fn track_ids(&self) -> impl Iterator<Item = TrackId> {
        (0..self.tracks.len() as u32).map(TrackId)
    }

    /// Every layer, in creation order: a layer's split parent or merge inputs
    /// come before it.
    pub fn layer_ids(&self) -> impl Iterator<Item = LayerId> {
        (0..self.layers.len() as u32).map(LayerId)
    }

    /// Columns holding layers of `track` on `line`. Subspines of one track
    /// stay adjacent because exchange (`*x`) and add (`*+`) are unsupported.
    pub fn track_columns(&self, line: usize, track: TrackId) -> Range<usize> {
        columns_of(&self.entering[line], |layer| self.track_of(layer) == track)
    }

    /// Like [`Self::track_columns`] for the layout after the last line.
    pub fn track_columns_after_last(&self, track: TrackId) -> Range<usize> {
        columns_of(&self.after_last, |layer| self.track_of(layer) == track)
    }
}

fn columns_of(layers: &[LayerId], mut matches: impl FnMut(LayerId) -> bool) -> Range<usize> {
    let start = layers.iter().position(|&layer| matches(layer));
    match start {
        Some(start) => {
            let len = layers[start..]
                .iter()
                .take_while(|&&layer| matches(layer))
                .count();
            start..start + len
        }
        None => 0..0,
    }
}

#[derive(Default)]
struct Builder {
    entering: Vec<Rc<[LayerId]>>,
    active: Rc<[LayerId]>,
    layers: Vec<Layer>,
    tracks: Vec<Track>,
    /// Union-find parent per track. Only implied tracks are ever joined.
    parent: Vec<u32>,
    /// Tracks implied by a document without an exclusive interpretation.
    implied: Vec<bool>,
    /// Root track, 1-based line and column of each join.
    joins: Vec<(TrackId, usize, usize)>,
    /// The exclusive interpretation line whose tracks lines before it
    /// already use.
    declared_ahead: Option<usize>,
}

impl Builder {
    fn line(
        &mut self,
        index: usize,
        line_number: usize,
        line: &KernLine,
    ) -> Result<(), TopologyError> {
        match line {
            KernLine::SpineTypeLine(_) if self.declared_ahead == Some(index) => {
                self.entering.push(self.active.clone());
            }
            KernLine::SpineTypeLine(spine_types) => {
                self.replace_all(index);
                self.active = spine_types
                    .iter()
                    .map(|spine_type| self.new_track(index, spine_type.clone()))
                    .collect();
                self.entering.push(self.active.clone());
            }
            KernLine::TandemInterpretationLine(tandems) => {
                self.fields(
                    index,
                    line_number,
                    tandems.len(),
                    "tandem interpretation line",
                )?;
            }
            KernLine::LocalCommentLine(comments) => {
                self.fields(index, line_number, comments.len(), "local comment line")?;
            }
            KernLine::Data(contents) => {
                self.fields(index, line_number, contents.len(), "data line")?;
            }
            KernLine::Barline(_) => self.entering.push(self.active.clone()),
            KernLine::SpineManipulatorLine(manipulators) => {
                self.fields(
                    index,
                    line_number,
                    manipulators.len(),
                    "spine manipulator line",
                )?;
                self.manipulate(index, line_number, manipulators.tokens())?;
            }
        }
        Ok(())
    }

    /// Record a line with one field per active layer. A document without an
    /// exclusive interpretation starts with implied `**kern` tracks.
    fn fields(
        &mut self,
        index: usize,
        line_number: usize,
        width: usize,
        context: &'static str,
    ) -> Result<(), TopologyError> {
        if self.active.is_empty() && self.tracks.is_empty() {
            self.active = (0..width)
                .map(|_| self.new_track(index, SpineType::Kern))
                .collect();
            self.implied = vec![true; width];
        }
        if self.active.len() != width {
            return Err(TopologyError::WidthMismatch {
                line: line_number,
                context,
                expected: self.active.len(),
                found: width,
            });
        }
        self.entering.push(self.active.clone());
        Ok(())
    }

    fn manipulate(
        &mut self,
        index: usize,
        line_number: usize,
        manipulators: &[SpineManipulator],
    ) -> Result<(), TopologyError> {
        let input = self.active.clone();
        let mut next = Vec::with_capacity(input.len());
        let mut column = 0;
        while column < input.len() {
            match manipulators[column] {
                SpineManipulator::Null => {
                    next.push(input[column]);
                    column += 1;
                }
                SpineManipulator::Terminate => {
                    self.end(input[column], index, LayerEnd::Terminated);
                    column += 1;
                }
                SpineManipulator::Split => {
                    let parent = input[column];
                    self.end(parent, index, LayerEnd::Split);
                    let track = self.layers[parent.index()].track;
                    next.push(self.new_layer(track, LayerOrigin::SplitOf(parent), index + 1));
                    next.push(self.new_layer(track, LayerOrigin::SplitOf(parent), index + 1));
                    column += 1;
                }
                SpineManipulator::Merge => {
                    let start = column;
                    while column < input.len() && manipulators[column] == SpineManipulator::Merge {
                        column += 1;
                    }
                    let inputs = input[start..column].to_vec();
                    if inputs.len() < 2 {
                        return Err(TopologyError::MergeRequiresAtLeastTwo { line: line_number });
                    }
                    let track = self.root(self.layers[inputs[0].index()].track);
                    for (offset, layer) in inputs.iter().enumerate() {
                        let other = self.root(self.layers[layer.index()].track);
                        if other == track {
                            continue;
                        }
                        if !(self.is_implied(track) && self.is_implied(other)) {
                            return Err(TopologyError::CrossTrackMerge {
                                line: line_number,
                                column: start + offset + 1,
                            });
                        }
                        // A page that starts inside a split shows its sibling
                        // layers as separate columns; merging them proves
                        // they belong to one track.
                        self.parent[other.index()] = track.0;
                        self.joins.push((track, line_number, start + offset + 1));
                    }
                    for &layer in &inputs {
                        self.end(layer, index, LayerEnd::Merged);
                    }
                    next.push(self.new_layer(track, LayerOrigin::MergeOf(inputs), index + 1));
                }
            }
        }
        self.active = next.into();
        Ok(())
    }

    fn replace_all(&mut self, index: usize) {
        for layer in self.active.clone().iter() {
            self.end(*layer, index, LayerEnd::Replaced);
        }
    }

    fn root(&self, mut track: TrackId) -> TrackId {
        while self.parent[track.index()] != track.0 {
            track = TrackId(self.parent[track.index()]);
        }
        track
    }

    fn is_implied(&self, track: TrackId) -> bool {
        self.implied.get(track.index()).copied().unwrap_or(false)
    }

    fn new_track(&mut self, index: usize, spine_type: SpineType) -> LayerId {
        let track = TrackId(self.tracks.len() as u32);
        self.parent.push(track.0);
        self.tracks.push(Track {
            spine_type,
            born: index,
            staff: None,
            part: None,
        });
        self.new_layer(track, LayerOrigin::Exclusive, index)
    }

    fn new_layer(&mut self, track: TrackId, origin: LayerOrigin, born: usize) -> LayerId {
        let id = LayerId(self.layers.len() as u32);
        self.layers.push(Layer {
            track,
            origin,
            born,
            end: None,
        });
        id
    }

    fn end(&mut self, layer: LayerId, index: usize, end: LayerEnd) {
        self.layers[layer.index()].end = Some((index, end));
    }

    /// Number the tracks that survive joining, keeping their order.
    fn finish(self) -> Topology {
        let mut renumbered = vec![TrackId(0); self.tracks.len()];
        let mut tracks = Vec::new();
        let mut implied = Vec::new();
        for (index, track) in self.tracks.iter().enumerate() {
            let id = TrackId(index as u32);
            if self.root(id) == id {
                renumbered[index] = TrackId(tracks.len() as u32);
                tracks.push(track.clone());
                implied.push(self.is_implied(id));
            }
        }
        for index in 0..self.tracks.len() {
            renumbered[index] = renumbered[self.root(TrackId(index as u32)).index()];
        }
        let layers = self
            .layers
            .iter()
            .map(|layer| Layer {
                track: renumbered[layer.track.index()],
                ..layer.clone()
            })
            .collect();
        let mut joins: Vec<(TrackId, usize, usize)> = Vec::new();
        for &(track, line, column) in &self.joins {
            let track = renumbered[track.index()];
            if !joins.iter().any(|(joined, _, _)| *joined == track) {
                joins.push((track, line, column));
            }
        }
        Topology {
            entering: self.entering,
            after_last: self.active,
            layers,
            tracks,
            joins,
            implied,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        document: KernDocument,
        topology: Topology,
    }

    impl Fixture {
        fn new(input: &str) -> Self {
            let document = KernDocument::parse(input).unwrap();
            let topology = Topology::build(&document).unwrap();
            Self { document, topology }
        }

        /// Index of the `nth` (0-based) line matching `kind`.
        fn line(&self, kind: fn(&KernLine) -> bool, nth: usize) -> usize {
            self.document
                .lines()
                .iter()
                .enumerate()
                .filter(|(_, line)| kind(line))
                .nth(nth)
                .map(|(index, _)| index)
                .unwrap()
        }

        fn data(&self, nth: usize) -> usize {
            self.line(|line| matches!(line, KernLine::Data(_)), nth)
        }

        fn manipulator(&self, nth: usize) -> usize {
            self.line(
                |line| matches!(line, KernLine::SpineManipulatorLine(_)),
                nth,
            )
        }
    }

    #[test]
    fn split_creates_two_new_layers_of_the_same_track() {
        let f = Fixture::new("**kern\t**kern\n*^\t*\n4c\t4e\t4g\n*-\t*-\t*-\n");
        let t = &f.topology;
        let split = f.manipulator(0);
        let data = t.layers_at(f.data(0));
        assert_eq!(data.len(), 3);
        let parent = t.layer_at(split, 0);
        assert_eq!(t.layer(data[0]).origin, LayerOrigin::SplitOf(parent));
        assert_eq!(t.layer(data[1]).origin, LayerOrigin::SplitOf(parent));
        assert_ne!(data[0], parent);
        assert_eq!(t.track_of(data[0]), t.track_of(data[1]));
        assert_ne!(t.track_of(data[1]), t.track_of(data[2]));
        assert_eq!(t.layer(parent).end, Some((split, LayerEnd::Split)));
        assert_eq!(t.split_side(data[0]), Some(SplitSide::Left));
        assert_eq!(t.split_side(data[1]), Some(SplitSide::Right));
        assert_eq!(t.split_side(data[2]), None);
        assert_eq!(t.track_columns(f.data(0), t.track_of(data[0])), 0..2);
        assert_eq!(t.track_columns(f.data(0), t.track_of(data[2])), 2..3);
    }

    #[test]
    fn re_split_after_merge_never_reuses_a_layer() {
        let f = Fixture::new("**kern\n*^\n4c\t4e\n*v\t*v\n4c\n*^\n4c\t4e\n*-\t*-\n");
        let t = &f.topology;
        let first = t.layers_at(f.data(0)).to_vec();
        let second = t.layers_at(f.data(2)).to_vec();
        assert!(first.iter().all(|layer| !second.contains(layer)));
        let merged = t.layer_at(f.data(1), 0);
        assert_eq!(t.layer(merged).origin, LayerOrigin::MergeOf(first));
        // The re-split's left child follows the merged layer's id, not a
        // sibling of the same parent.
        assert_eq!(t.split_side(second[0]), Some(SplitSide::Left));
        assert_eq!(t.split_side(second[1]), Some(SplitSide::Right));
    }

    #[test]
    fn terminate_ends_the_layer_and_leaves_siblings() {
        let f = Fixture::new("**kern\n*^\n4c\t4e\n*\t*-\n4c\n*-\n");
        let t = &f.topology;
        let survivor = t.layer_at(f.data(0), 0);
        let ended = t.layer_at(f.data(0), 1);
        assert_eq!(
            t.layer(ended).end,
            Some((f.manipulator(1), LayerEnd::Terminated))
        );
        assert_eq!(t.layers_at(f.data(1)), &[survivor]);
        assert!(t.layers_after_last().is_empty());
    }

    #[test]
    fn document_without_exclusive_interpretation_implies_kern_tracks() {
        let f = Fixture::new("4c\t4e\n=\n");
        let t = &f.topology;
        assert_eq!(t.track_count(), 2);
        assert_eq!(
            t.track_ids().collect::<Vec<_>>(),
            vec![
                t.track_of(t.layer_at(f.data(0), 0)),
                t.track_of(t.layer_at(f.data(0), 1))
            ]
        );
        assert_eq!(t.spine_type(t.layer_at(f.data(0), 1)), &SpineType::Kern);
    }

    #[test]
    fn merging_implied_columns_joins_their_tracks() {
        // A page that starts inside a split of the first staff.
        let f = Fixture::new("4c\t4e\t4G\n*v\t*v\t*\n=\t=\n2c\t2G\n");
        let t = &f.topology;
        assert_eq!(t.track_count(), 2);
        let first = t.layers_at(f.data(0));
        assert_eq!(t.track_of(first[0]), t.track_of(first[1]));
        assert_ne!(t.track_of(first[1]), t.track_of(first[2]));
        assert_eq!(t.track_columns(f.data(0), t.track_of(first[0])), 0..2);
        assert_eq!(t.staff_of(first[0]), Some(2));
        assert_eq!(t.staff_of(first[2]), Some(1));
    }

    #[test]
    fn rejects_cross_track_merge() {
        let document = KernDocument::parse("**kern\t**kern\n*v\t*v\n4c\n*-\n").unwrap();
        assert_eq!(
            Topology::build(&document).unwrap_err(),
            TopologyError::CrossTrackMerge { line: 2, column: 2 }
        );
    }
}
