//! Resolved score: spine identity and interpretation state for a
//! [`KernDocument`], computed once.
//!
//! [`Score::resolve`] is the only code that interprets spine manipulators and
//! state tandems. A `Score` exists only when every state rule holds, so its
//! consumers never re-walk the document or handle state errors. The one
//! exception is [`Score::resolve_timing`], for line timing alone.

pub mod drawn;
mod key_cancellation;
mod layout;
mod staff;
pub mod state;
pub mod topology;

use std::{
    collections::{BTreeMap, VecDeque},
    fmt,
    rc::Rc,
};

use thiserror::Error;

pub use drawn::{DrawnDuration, DrawnView, NoteSize, Undrawable};
pub use key_cancellation::{CancellationIssue, NotDrawn};
pub use state::{
    Effect, LayerState, Lifecycle, Scope, StateKey, TrackState, TransitionError, Unrepresentable,
};
pub use topology::{
    Layer, LayerEnd, LayerId, LayerOrigin, SplitSide, Topology, TopologyError, Track, TrackId,
};

pub(crate) use staff::unread_staff_declarations;

use key_cancellation::{CancellationSimulation, KeyChange};

use crate::{
    document::Framing,
    meter::{ActiveMeter, MeterChange, MeterDeclaration},
    types::DiatonicStep,
    Duration, KernContent, KernDocument, KernLine, KeyCancellation, KeySignature, SpineContent,
    SpineManipulator, SpineType, TandemInterpretation, TandemInterpretationLine,
    TandemInterpretationLineKind, VisualStyle,
};

/// A document with resolved topology and interpretation state.
#[derive(Debug, Clone)]
pub struct Score<'d> {
    document: &'d KernDocument,
    topology: Topology,
    /// `entering[line][column]`: state in force for that field before the
    /// line applies. On a tandem line this is the state before its tokens.
    entering: Vec<Rc<[StateIds]>>,
    after_last: Rc<[StateIds]>,
    layer_states: Vec<LayerState>,
    track_states: Vec<TrackState>,
    events: Vec<AppliedEvent>,
    layouts: Rc<layout::LinkedLayouts>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StateIds {
    layer: u32,
    track: u32,
}

/// State in force at one field.
#[derive(Debug, Clone, Copy)]
pub struct FieldState<'s> {
    pub layer: &'s LayerState,
    pub track: &'s TrackState,
}

impl FieldState<'_> {
    /// Whether `key` holds the value every layer and track starts with.
    pub fn holds_default(&self, key: StateKey) -> bool {
        match key.scope() {
            Scope::Layer => self.layer.holds_default(key),
            Scope::Track => self.track.holds_default(key),
        }
    }
}

/// One state tandem as the resolver applied it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppliedEvent {
    /// Zero-based document line index.
    pub line_index: usize,
    /// Zero-based column.
    pub column: usize,
    pub key: StateKey,
    pub outcome: Outcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Changed,
    /// The tandem restates the value already in force and Verovio draws
    /// nothing for it.
    Redundant,
    /// The tandem restates the value already in force, but Verovio draws it
    /// again (see [`StateKey::drawn_when_restated`]).
    Restated,
    /// A sibling subspine of the same track set the same value on the same
    /// line; that sibling's token carried the change. Copies of a
    /// restatement take the restatement's outcome.
    Coalesced,
}

/// Every state rule violation in a document. Never empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateErrors(pub Vec<StateError>);

impl fmt::Display for StateErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, error) in self.0.iter().enumerate() {
            if index > 0 {
                writeln!(f)?;
            }
            write!(f, "{error}")?;
        }
        Ok(())
    }
}

impl std::error::Error for StateErrors {}

/// One state rule violation. Lines are 1-based source lines, columns 1-based.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StateError {
    #[error(transparent)]
    Topology(#[from] TopologyError),
    #[error(
        "line {line} column {column}: sibling subspines set different {} values on one line",
        .key.name()
    )]
    Conflicting {
        line: usize,
        column: usize,
        key: StateKey,
    },
    #[error("line {line} column {column}: {} closed while none is open", .key.name())]
    UnmatchedClose {
        line: usize,
        column: usize,
        key: StateKey,
    },
    #[error(
        "line {line} column {column}: {} does not match the open {:?}",
        .key.name(), .open
    )]
    SpanMismatch {
        line: usize,
        column: usize,
        key: StateKey,
        open: TandemInterpretation,
    },
    #[error(
        "line {line} column {column}: Verovio tracks *cue by layer number, so after the spine manipulation at line {manipulation} it would draw this layer {}; its own *cue/*Xcue history says {}. Write *cue or *Xcue for this layer after line {manipulation}",
        size(!*.cue), size(*.cue)
    )]
    CueRenumbered {
        line: usize,
        column: usize,
        manipulation: usize,
        /// The layer's own cue state.
        cue: bool,
    },
    #[error(
        "line {line} column {column}: grace note under *rscale:{}/{}; Verovio cannot draw scaled grace notes",
        .scale.numerator(), .scale.denominator()
    )]
    GraceUnderRhythmScale {
        line: usize,
        column: usize,
        scale: crate::RhythmScale,
    },
    #[error(
        "line {line} column {column}: duration {written} under *rscale:{}/{} has no printed note value",
        .scale.numerator(), .scale.denominator()
    )]
    UndrawableDuration {
        line: usize,
        column: usize,
        written: String,
        scale: crate::RhythmScale,
    },
    #[error(
        "line {line} column {column}: {} opened at line {opened_at} is still open where its spine ends",
        .key.name()
    )]
    OpenAtEnd {
        line: usize,
        column: usize,
        key: StateKey,
        opened_at: usize,
    },
    #[error("line {line} column {column}: `!LO:N:vis={value}` is not a duration")]
    InvalidVisualDuration {
        line: usize,
        column: usize,
        value: String,
    },
    #[error(
        "line {line} column {column}: the key signature cancels {} with a natural, but {}",
        .step.lowercase(), stray_natural_context(.step, .previous)
    )]
    StrayNatural {
        line: usize,
        column: usize,
        step: DiatonicStep,
        /// The track's key before this one, if any.
        previous: Option<KeySignature>,
    },
    #[error("line {line} column {column}: {issue}")]
    KeyCancellation {
        line: usize,
        column: usize,
        issue: CancellationIssue,
    },
}

impl StateError {
    /// The 1-based source line the error points at.
    pub fn line(&self) -> usize {
        match self {
            Self::Topology(error) => error.line(),
            Self::Conflicting { line, .. }
            | Self::UnmatchedClose { line, .. }
            | Self::SpanMismatch { line, .. }
            | Self::CueRenumbered { line, .. }
            | Self::GraceUnderRhythmScale { line, .. }
            | Self::UndrawableDuration { line, .. }
            | Self::OpenAtEnd { line, .. }
            | Self::InvalidVisualDuration { line, .. }
            | Self::StrayNatural { line, .. }
            | Self::KeyCancellation { line, .. } => *line,
        }
    }
}

fn stray_natural_context(step: &DiatonicStep, previous: &Option<KeySignature>) -> String {
    match previous {
        Some(previous) => format!(
            "the previous key signature `{}` does not alter {}",
            previous.as_kern(),
            step.lowercase()
        ),
        None => "no key signature precedes it in this spine".to_owned(),
    }
}

fn size(cue: bool) -> &'static str {
    if cue {
        "at cue size"
    } else {
        "at normal size"
    }
}

impl<'d> Score<'d> {
    pub fn resolve(document: &'d KernDocument, framing: Framing) -> Result<Self, StateErrors> {
        let topology =
            Topology::build(document).map_err(|error| StateErrors(vec![error.into()]))?;
        Resolver::new(document, framing, topology).run(|_| true)
    }

    /// Resolve an excerpt for line timing, which reads only the topology and
    /// each track's meter. Fails on topology errors and conflicting meters;
    /// other state errors (ottava pairing, cue numbering, drawability under
    /// `*rscale`) do not change timing and are left to the `state` rule. The
    /// score must not be read for anything but timing.
    pub(crate) fn resolve_timing(document: &'d KernDocument) -> Result<Self, StateErrors> {
        let topology =
            Topology::build(document).map_err(|error| StateErrors(vec![error.into()]))?;
        Resolver::new(document, Framing::Excerpt, topology).run(|error| {
            matches!(
                error,
                StateError::Topology(_)
                    | StateError::Conflicting {
                        key: StateKey::Meter,
                        ..
                    }
            )
        })
    }

    pub fn document(&self) -> &'d KernDocument {
        self.document
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// State in force at a field, before a tandem line's own tokens apply.
    pub fn state_at(&self, line_index: usize, column: usize) -> FieldState<'_> {
        self.field_state(self.entering[line_index][column])
    }

    /// State in force once `line_index` has applied. `column` indexes
    /// [`Topology::layers_after`], which differs from the line's own layout
    /// only on a spine manipulator line.
    pub fn state_after(&self, line_index: usize, column: usize) -> FieldState<'_> {
        match self.entering.get(line_index + 1) {
            Some(entering) => self.field_state(entering[column]),
            None => self.state_after_last(column),
        }
    }

    /// State in force after the last line.
    pub fn state_after_last(&self, column: usize) -> FieldState<'_> {
        self.field_state(self.after_last[column])
    }

    pub fn events(&self) -> &[AppliedEvent] {
        &self.events
    }

    /// What the page shows: drawn durations and note size.
    pub fn drawn(&self) -> DrawnView<'_, 'd> {
        DrawnView::new(self)
    }

    /// Owned state of every column entering `line_index`. One past the last
    /// line is the state after it.
    pub fn snapshot_before(&self, line_index: usize) -> Snapshot {
        if line_index == self.entering.len() {
            return self.snapshot_after_last();
        }
        self.snapshot(
            self.topology.layers_at(line_index),
            &self.entering[line_index],
        )
    }

    pub fn snapshot_after_last(&self) -> Snapshot {
        self.snapshot(self.topology.layers_after_last(), &self.after_last)
    }

    fn field_state(&self, ids: StateIds) -> FieldState<'_> {
        FieldState {
            layer: &self.layer_states[ids.layer as usize],
            track: &self.track_states[ids.track as usize],
        }
    }

    fn snapshot(&self, layers: &[LayerId], ids: &[StateIds]) -> Snapshot {
        let mut ordinals = BTreeMap::new();
        let columns = layers
            .iter()
            .zip(ids)
            .map(|(&layer, &ids)| {
                let track = self.topology.track_of(layer);
                let next = ordinals.len();
                let state = self.field_state(ids);
                SnapshotColumn {
                    track: *ordinals.entry(track).or_insert(next),
                    layer: state.layer.clone(),
                    track_state: state.track.clone(),
                }
            })
            .collect();
        Snapshot { columns }
    }
}

/// State of every column at one point, detached from its score so it can
/// bridge to another score's columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    columns: Vec<SnapshotColumn>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SnapshotColumn {
    /// Ordinal of the column's track within this snapshot.
    track: usize,
    layer: LayerState,
    track_state: TrackState,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BridgeError {
    #[error("cannot bridge between different spine layouts")]
    LayoutMismatch,
    #[error("no tandem returns {} to its previous value", .0.0.name())]
    Unrepresentable(Unrepresentable),
}

impl Snapshot {
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// The same layout with every slot at its default.
    pub fn defaulted(&self) -> Self {
        self.map(|_| (LayerState::default(), TrackState::default()))
    }

    /// The same layout with every open span closed. Toggles such as `*cue`
    /// may stay on at the end of a document and are kept.
    pub fn closed(&self) -> Self {
        self.map(|column| (column.layer.clone(), column.track_state.closed()))
    }

    /// One column per track, holding the track's state, with layer state at
    /// its default: the layout of an exclusive interpretation line before any
    /// split.
    pub fn tracks(&self) -> Self {
        let mut columns: Vec<SnapshotColumn> = Vec::new();
        for column in &self.columns {
            if columns.last().is_none_or(|last| last.track != column.track) {
                columns.push(SnapshotColumn {
                    track: column.track,
                    layer: LayerState::default(),
                    track_state: column.track_state.clone(),
                });
            }
        }
        Self { columns }
    }

    pub fn track_count(&self) -> usize {
        self.columns.last().map_or(0, |column| column.track + 1)
    }

    /// The same columns with `tracks[i]` as column `i`'s track, renumbered
    /// in column order: the layout once implied tracks another page proves
    /// to be one are joined. Columns that come to share a track must already
    /// hold the same track state.
    pub fn regroup(&self, tracks: &[usize]) -> Result<Self, BridgeError> {
        if tracks.len() != self.columns.len() {
            return Err(BridgeError::LayoutMismatch);
        }
        let mut ordinals = BTreeMap::new();
        let mut columns: Vec<SnapshotColumn> = Vec::with_capacity(self.columns.len());
        for (column, track) in self.columns.iter().zip(tracks) {
            let next = ordinals.len();
            let track = *ordinals.entry(*track).or_insert(next);
            if columns
                .iter()
                .any(|other| other.track == track && other.track_state != column.track_state)
            {
                return Err(BridgeError::LayoutMismatch);
            }
            columns.push(SnapshotColumn {
                track,
                ..column.clone()
            });
        }
        Ok(Self { columns })
    }

    /// These columns with `keys` taken from `source`. Track slots match by
    /// track ordinal, so `source` may split its tracks differently; layer
    /// slots match by column and need the same layout.
    pub fn with(&self, source: &Snapshot, keys: &[StateKey]) -> Result<Self, BridgeError> {
        self.merge_slots(source, keys, |_, _| true)
    }

    /// Like [`Self::with`], but a slot `source` leaves at its default keeps
    /// this snapshot's value.
    pub fn with_declared(&self, source: &Snapshot, keys: &[StateKey]) -> Result<Self, BridgeError> {
        self.merge_slots(source, keys, |column, key| match key.scope() {
            Scope::Layer => !column.layer.holds_default(key),
            Scope::Track => !column.track_state.holds_default(key),
        })
    }

    /// `target`, as far as tandems can move this snapshot there: a slot whose
    /// value in `target` no tandem writes (no clef declared yet, say) keeps
    /// this snapshot's value. Needs the same layout.
    pub fn toward(&self, target: &Snapshot) -> Result<Self, BridgeError> {
        if self.layout() != target.layout() {
            return Err(BridgeError::LayoutMismatch);
        }
        let mut reached = self.clone();
        for key in StateKey::ALL {
            let candidate = reached.with(target, &[key])?;
            match bridge(&reached, &candidate) {
                Ok(_) => reached = candidate,
                Err(BridgeError::Unrepresentable(_)) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(reached)
    }

    fn merge_slots(
        &self,
        source: &Snapshot,
        keys: &[StateKey],
        take: impl Fn(&SnapshotColumn, StateKey) -> bool,
    ) -> Result<Self, BridgeError> {
        if self.track_count() != source.track_count() {
            return Err(BridgeError::LayoutMismatch);
        }
        let layer_keys = keys.iter().any(|key| key.scope() == Scope::Layer);
        if layer_keys && self.layout() != source.layout() {
            return Err(BridgeError::LayoutMismatch);
        }
        let mut merged = self.clone();
        for (index, column) in merged.columns.iter_mut().enumerate() {
            let track_source = source
                .columns
                .iter()
                .find(|candidate| candidate.track == column.track)
                .expect("equal track counts share every ordinal");
            for &key in keys {
                match key.scope() {
                    Scope::Track if take(track_source, key) => {
                        column.track_state.copy_slot(&track_source.track_state, key)
                    }
                    Scope::Layer if take(&source.columns[index], key) => {
                        column.layer.copy_slot(&source.columns[index].layer, key)
                    }
                    Scope::Track | Scope::Layer => {}
                }
            }
        }
        Ok(merged)
    }

    fn layout(&self) -> Vec<usize> {
        self.columns.iter().map(|column| column.track).collect()
    }

    fn map(&self, mut state: impl FnMut(&SnapshotColumn) -> (LayerState, TrackState)) -> Self {
        Self {
            columns: self
                .columns
                .iter()
                .map(|column| {
                    let (layer, track_state) = state(column);
                    SnapshotColumn {
                        track: column.track,
                        layer,
                        track_state,
                    }
                })
                .collect(),
        }
    }
}

/// Tandem lines that move every column from `from` to `to`. Track state is
/// written in the track's first column, layer state in its own column. Each
/// line holds one kind of tandem; slots follow [`StateKey::ALL`] order.
/// Returns no lines when the snapshots are equal.
pub fn bridge(
    from: &Snapshot,
    to: &Snapshot,
) -> Result<Vec<TandemInterpretationLine>, BridgeError> {
    if from.layout() != to.layout() {
        return Err(BridgeError::LayoutMismatch);
    }

    let mut per_column: Vec<Vec<(StateKey, TandemInterpretation)>> =
        vec![Vec::new(); to.columns.len()];
    for (index, (before, after)) in from.columns.iter().zip(&to.columns).enumerate() {
        let first_of_track = index == 0 || to.columns[index - 1].track != after.track;
        if first_of_track {
            per_column[index].extend(
                after
                    .track_state
                    .path_from(&before.track_state)
                    .map_err(BridgeError::Unrepresentable)?,
            );
        }
        per_column[index].extend(
            after
                .layer
                .path_from(&before.layer)
                .map_err(BridgeError::Unrepresentable)?,
        );
    }

    let mut lines = Vec::new();
    for key in StateKey::ALL {
        let mut queues = per_column
            .iter()
            .map(|tandems| {
                tandems
                    .iter()
                    .filter(|(tandem_key, _)| *tandem_key == key)
                    .map(|(_, tandem)| tandem.clone())
                    .collect::<VecDeque<_>>()
            })
            .collect::<Vec<_>>();
        while let Some(kind) = queues
            .iter()
            .find_map(|queue| queue.front())
            .map(TandemInterpretationLineKind::from_tandem)
        {
            let tokens = queues
                .iter_mut()
                .map(|queue| {
                    let same_kind = queue.front().is_some_and(|tandem| {
                        TandemInterpretationLineKind::from_tandem(tandem) == kind
                    });
                    if same_kind {
                        queue.pop_front().expect("front checked")
                    } else {
                        TandemInterpretation::Null
                    }
                })
                .collect();
            lines.push(TandemInterpretationLine::from_homogeneous_tokens(tokens));
        }
    }
    Ok(lines)
}

/// Tandem lines at a seam: they move `keys` from their values in `from` to
/// those in `to`, laid out on `to`'s columns. A slot `to` leaves at its
/// default keeps `from`'s value, so a page that does not restate its clef
/// continues the previous page's clef.
pub fn bridge_seam(
    from: &Snapshot,
    to: &Snapshot,
    keys: &[StateKey],
) -> Result<Vec<TandemInterpretationLine>, BridgeError> {
    let carried = to.with(from, keys)?;
    bridge(&carried, &carried.with_declared(to, keys)?)
}

struct Resolver<'d> {
    document: &'d KernDocument,
    framing: Framing,
    topology: Topology,
    entering: Vec<Rc<[StateIds]>>,
    layer_states: Vec<LayerState>,
    track_states: Vec<TrackState>,
    /// Current state id per layer and per track.
    layer_current: Vec<u32>,
    track_current: Vec<u32>,
    /// Meter declaration group per track.
    meters: Vec<ActiveMeter>,
    /// The meter tandems of each track's open declaration group.
    meter_groups: Vec<Option<OpenMeterGroup>>,
    /// 1-based source line where each open span was opened.
    layer_opened: BTreeMap<(LayerId, StateKey), usize>,
    track_opened: BTreeMap<(TrackId, StateKey), usize>,
    /// Verovio's cue flag per track, indexed by layer number within the
    /// track. Verovio never clears a flag when layers renumber.
    cue_flags: Vec<Vec<bool>>,
    /// Tracks renumbered by a manipulation since the last data line, with
    /// that manipulation's line.
    renumbered: BTreeMap<TrackId, usize>,
    cancellation: CancellationSimulation,
    events: Vec<AppliedEvent>,
    errors: Vec<StateError>,
}

impl<'d> Resolver<'d> {
    fn new(document: &'d KernDocument, framing: Framing, topology: Topology) -> Self {
        Self {
            document,
            framing,
            entering: Vec::with_capacity(document.lines().len()),
            layer_states: vec![LayerState::default()],
            track_states: vec![TrackState::default()],
            layer_current: vec![0; topology.layer_count()],
            track_current: vec![0; topology.track_count()],
            meters: vec![ActiveMeter::new(); topology.track_count()],
            meter_groups: (0..topology.track_count()).map(|_| None).collect(),
            layer_opened: BTreeMap::new(),
            track_opened: BTreeMap::new(),
            cue_flags: vec![Vec::new(); topology.track_count()],
            renumbered: BTreeMap::new(),
            cancellation: CancellationSimulation::default(),
            events: Vec::new(),
            errors: Vec::new(),
            topology,
        }
    }

    /// Resolve every line. Fails when an error `fatal` accepts occurred.
    fn run(mut self, fatal: impl Fn(&StateError) -> bool) -> Result<Score<'d>, StateErrors> {
        for (index, line) in self.document.lines().iter().enumerate() {
            if !matches!(line, KernLine::TandemInterpretationLine(_)) {
                self.end_meter_groups();
            }
            let ids = self.current_ids(self.topology.layers_at(index));
            let ids = match self.entering.last() {
                Some(previous) if **previous == *ids => previous.clone(),
                _ => ids,
            };
            self.entering.push(ids);

            match line {
                KernLine::TandemInterpretationLine(tandems) => self.tandems(index, tandems),
                KernLine::SpineManipulatorLine(manipulators) => {
                    self.manipulate(index, manipulators.tokens())
                }
                KernLine::SpineTypeLine(_) => self.replace(index),
                KernLine::Data(fields) => {
                    self.check_cue_numbering(index);
                    self.check_drawable(index, fields);
                    let kern_tracks = self.kern_tracks(self.topology.layers_at(index));
                    self.cancellation.data_line(&kern_tracks, &mut self.errors);
                }
                KernLine::Barline(_) => {
                    let kern_tracks = self.kern_tracks(self.topology.layers_at(index));
                    self.cancellation.barline(&kern_tracks, &mut self.errors);
                }
                KernLine::LocalCommentLine(_) => {}
            }
        }
        self.end_meter_groups();
        self.events.sort_by_key(|event| event.line_index);
        let kern_tracks = self.kern_tracks(self.topology.layers_after_last());
        self.cancellation.finish(&kern_tracks, &mut self.errors);
        if self.framing == Framing::Complete {
            self.check_document_end();
        }
        let after_last = self.current_ids(self.topology.layers_after_last());
        let (layouts, layout_errors) = layout::link(self.document, &self.topology);
        self.errors.extend(layout_errors);

        if self.errors.iter().any(&fatal) {
            return Err(StateErrors(self.errors.into_iter().filter(fatal).collect()));
        }
        Ok(Score {
            document: self.document,
            topology: self.topology,
            entering: self.entering,
            after_last,
            layer_states: self.layer_states,
            track_states: self.track_states,
            events: self.events,
            layouts: Rc::new(layouts),
        })
    }

    fn current_ids(&self, layers: &[LayerId]) -> Rc<[StateIds]> {
        layers
            .iter()
            .map(|&layer| StateIds {
                layer: self.layer_current[layer.index()],
                track: self.track_current[self.topology.track_of(layer).index()],
            })
            .collect()
    }

    fn line_number(&self, index: usize) -> usize {
        self.document.source_line_number(index)
    }

    /// The `**kern` tracks among `layers`, each once: the staves Verovio
    /// draws.
    fn kern_tracks(&self, layers: &[LayerId]) -> Vec<TrackId> {
        let mut tracks: Vec<TrackId> = Vec::new();
        for &layer in layers {
            let track = self.topology.track_of(layer);
            if self.topology.track(track).spine_type == SpineType::Kern && !tracks.contains(&track)
            {
                tracks.push(track);
            }
        }
        tracks
    }

    fn tandems(&mut self, index: usize, line: &TandemInterpretationLine) {
        let line_number = self.line_number(index);
        let mut track_events: BTreeMap<(TrackId, StateKey), Vec<(usize, &TandemInterpretation)>> =
            BTreeMap::new();

        for (column, tandem) in line.tokens().iter().enumerate() {
            let Some(key) = StateKey::of(tandem) else {
                continue;
            };
            let layer = self.topology.layer_at(index, column);
            match key.scope() {
                Scope::Track => track_events
                    .entry((self.topology.track_of(layer), key))
                    .or_default()
                    .push((column, tandem)),
                Scope::Layer => {
                    let mut state = self.layer_state(layer).clone();
                    match state.apply(tandem) {
                        Ok(mut effect) => {
                            if key == StateKey::Cue && self.set_cue_flag(index, column, state.cue) {
                                // Clearing a stale Verovio flag is not
                                // redundant even when the layer agrees.
                                effect = Effect::Changed;
                            }
                            if effect == Effect::Changed {
                                self.layer_current[layer.index()] =
                                    push(&mut self.layer_states, state);
                                self.record_span(Owner::Layer(layer), key, line_number);
                            }
                            self.event(index, column, key, outcome(key, effect));
                        }
                        Err(error) => self.transition_error(error, line_number, column, key),
                    }
                }
            }
        }

        for ((track, key), tandems) in track_events {
            let (first_column, first) = tandems[0];
            if let Some(&(column, _)) = tandems
                .iter()
                .find(|(_, tandem)| state::visible(tandem) != state::visible(first))
            {
                // Conflicting values contradict a join of implied columns:
                // the merge that joined them crossed staves after all.
                let error = match self.topology.joined_at(track) {
                    Some((line, column)) => {
                        StateError::Topology(TopologyError::CrossTrackMerge { line, column })
                    }
                    None => StateError::Conflicting {
                        line: line_number,
                        column: column + 1,
                        key,
                    },
                };
                if !self.errors.contains(&error) {
                    self.errors.push(error);
                }
                continue;
            }
            let (chosen_column, chosen) = tandems
                .iter()
                .copied()
                .find(|(_, tandem)| **tandem == state::visible(tandem))
                .unwrap_or((first_column, first));
            if key == StateKey::Meter {
                self.declare_meter(track, index, chosen_column, chosen, &tandems);
                continue;
            }

            let mut state = self.track_states[self.track_current[track.index()] as usize].clone();
            match state.apply(chosen) {
                Ok(mut effect) => {
                    let kern = self.topology.track(track).spine_type == SpineType::Kern;
                    match chosen {
                        TandemInterpretation::VisualStyle(VisualStyle::KeyCancellation(
                            KeyCancellation { active },
                        )) => {
                            // Moving Verovio's global flag is not redundant
                            // even when the spine agrees.
                            if self.cancellation.write(
                                track,
                                line_number,
                                chosen_column + 1,
                                *active,
                            ) {
                                effect = Effect::Changed;
                            }
                        }
                        TandemInterpretation::KeySignature(key) if kern => {
                            self.cancellation.key_change(
                                KeyChange {
                                    line: line_number,
                                    column: chosen_column + 1,
                                    track,
                                    key: *key,
                                },
                                &mut self.errors,
                            );
                        }
                        _ => {}
                    }
                    if effect == Effect::Changed {
                        self.track_current[track.index()] = push(&mut self.track_states, state);
                        self.record_span(Owner::Track(track), key, line_number);
                    }
                    for &(column, _) in &tandems {
                        let outcome = if column == chosen_column || effect == Effect::Redundant {
                            outcome(key, effect)
                        } else {
                            Outcome::Coalesced
                        };
                        self.event(index, column, key, outcome);
                    }
                }
                Err(error) => self.transition_error(error, line_number, chosen_column, key),
            }
        }
    }

    fn manipulate(&mut self, index: usize, manipulators: &[SpineManipulator]) {
        let line_number = self.line_number(index);
        let layers = self.topology.layers_at(index).to_vec();

        let mut touched = Vec::new();
        for (&layer, manipulator) in layers.iter().zip(manipulators) {
            let track = self.topology.track_of(layer);
            if *manipulator != SpineManipulator::Null && !touched.contains(&track) {
                touched.push(track);
            }
        }
        let next = self.topology.layers_after(index).to_vec();
        for &layer in &next {
            if layers.contains(&layer) {
                continue;
            }
            match &self.topology.layer(layer).origin {
                LayerOrigin::SplitOf(parent) => {
                    // Verovio's flag stays with the left child's layer
                    // number; the right child starts at normal size.
                    let parent = *parent;
                    let right = self.topology.split_side(layer) == Some(SplitSide::Right);
                    let mut state = self.layer_state(parent).clone();
                    if right && state.cue {
                        state.cue = false;
                        self.layer_current[layer.index()] = push(&mut self.layer_states, state);
                    } else {
                        self.layer_current[layer.index()] = self.layer_current[parent.index()];
                        self.inherit_opened(parent, layer);
                    }
                }
                LayerOrigin::MergeOf(inputs) => {
                    // The merged layer takes the leftmost input's layer
                    // number in Verovio, and with it that input's state.
                    let leftmost = inputs[0];
                    self.layer_current[layer.index()] = self.layer_current[leftmost.index()];
                    self.inherit_opened(leftmost, layer);
                }
                LayerOrigin::Exclusive => {}
            }
        }
        for &track in &touched {
            self.renumbered.insert(track, line_number);
        }

        // Terminating every spine ends the document; only a complete document
        // must close its ottavas there.
        if self.framing == Framing::Excerpt && next.is_empty() {
            return;
        }
        for track in touched {
            if next
                .iter()
                .any(|&layer| self.topology.track_of(layer) == track)
            {
                continue;
            }
            let column = layers
                .iter()
                .rposition(|&layer| self.topology.track_of(layer) == track)
                .unwrap_or(0);
            self.check_track_end(track, line_number, column);
        }
    }

    /// A later exclusive-interpretation line ends every active track.
    fn replace(&mut self, index: usize) {
        let Some(previous) = index.checked_sub(1) else {
            return;
        };
        let line_number = self.line_number(index);
        let previous = self.topology.layers_after(previous).to_vec();
        let mut tracks = Vec::new();
        for (column, &layer) in previous.iter().enumerate() {
            let track = self.topology.track_of(layer);
            if !tracks.contains(&track) {
                tracks.push(track);
                self.check_track_end(track, line_number, column);
            }
        }
    }

    fn check_document_end(&mut self) {
        let line_number = self
            .document
            .lines()
            .len()
            .checked_sub(1)
            .map_or(0, |index| self.line_number(index));
        let layers = self.topology.layers_after_last().to_vec();
        let mut tracks = Vec::new();
        for (column, &layer) in layers.iter().enumerate() {
            let track = self.topology.track_of(layer);
            if !tracks.contains(&track) {
                tracks.push(track);
                self.check_track_end(track, line_number, column);
            }
        }
    }

    /// Set Verovio's flag for the field's layer number. Returns whether the
    /// flag changed.
    fn set_cue_flag(&mut self, index: usize, column: usize, cue: bool) -> bool {
        let layer = self.topology.layer_at(index, column);
        let track = self.topology.track_of(layer);
        let number = column - self.topology.track_columns(index, track).start;
        let flags = &mut self.cue_flags[track.index()];
        if flags.len() <= number {
            flags.resize(number + 1, false);
        }
        std::mem::replace(&mut flags[number], cue) != cue
    }

    /// Before notes are drawn after a manipulation, every layer of a
    /// renumbered track must have the cue state Verovio's per-number flag
    /// gives it.
    fn check_cue_numbering(&mut self, index: usize) {
        let line = self.line_number(index);
        for (track, manipulation) in std::mem::take(&mut self.renumbered) {
            let columns = self.topology.track_columns(index, track);
            let start = columns.start;
            for column in columns {
                let layer = self.topology.layer_at(index, column);
                let cue = self.layer_state(layer).cue;
                let flag = self.cue_flags[track.index()]
                    .get(column - start)
                    .copied()
                    .unwrap_or(false);
                if flag != cue {
                    self.errors.push(StateError::CueRenumbered {
                        line,
                        column: column + 1,
                        manipulation,
                        cue,
                    });
                }
            }
        }
    }

    /// Every duration under a non-unit `*rscale` must have a printed value,
    /// so the drawn view can project without failing.
    fn check_drawable(&mut self, index: usize, fields: &[SpineContent]) {
        let line = self.line_number(index);
        for (column, field) in fields.iter().enumerate() {
            let layer = self.topology.layer_at(index, column);
            let scale = self.track_states
                [self.track_current[self.topology.track_of(layer).index()] as usize]
                .rhythm_scale;
            if scale.is_unit() {
                continue;
            }
            let SpineContent::Kern(content) = field else {
                continue;
            };
            let durations: Vec<&Duration> = match content {
                KernContent::Grace(_) => {
                    self.errors.push(StateError::GraceUnderRhythmScale {
                        line,
                        column: column + 1,
                        scale,
                    });
                    continue;
                }
                KernContent::Chord(chord) => {
                    chord.notes.iter().map(|note| &note.duration).collect()
                }
                KernContent::Note(note) => vec![&note.duration],
                KernContent::Rest(rest) => vec![&rest.duration],
                KernContent::Null
                | KernContent::InvisibleRestSlice(_)
                | KernContent::MultiMeasureRest(_) => continue,
            };
            if let Some(written) = durations
                .into_iter()
                .find(|duration| DrawnDuration::project(duration, scale).is_err())
            {
                self.errors.push(StateError::UndrawableDuration {
                    line,
                    column: column + 1,
                    written: DrawnDuration::project(written, crate::RhythmScale::UNIT)
                        .expect("unit scale draws the written value")
                        .to_kern(),
                    scale,
                });
            }
        }
    }

    fn check_track_end(&mut self, track: TrackId, line: usize, column: usize) {
        let state = &self.track_states[self.track_current[track.index()] as usize];
        let open = StateKey::ALL
            .into_iter()
            .filter(|key| {
                key.scope() == Scope::Track
                    && key.lifecycle()
                        == (Lifecycle::Span {
                            closes_before_track_end: true,
                        })
                    && state.is_open(*key)
            })
            .collect::<Vec<_>>();
        for key in open {
            let opened_at = self.track_opened[&(track, key)];
            self.errors.push(StateError::OpenAtEnd {
                line,
                column: column + 1,
                key,
                opened_at,
            });
        }
    }

    /// A layer that inherits an open span also inherits where it opened, so
    /// later errors about it can cite the opening tandem.
    fn inherit_opened(&mut self, from: LayerId, to: LayerId) {
        let inherited = StateKey::ALL
            .into_iter()
            .filter_map(|key| Some((key, *self.layer_opened.get(&(from, key))?)))
            .collect::<Vec<_>>();
        for (key, line) in inherited {
            self.layer_opened.insert((to, key), line);
        }
    }

    fn layer_state(&self, layer: LayerId) -> &LayerState {
        &self.layer_states[self.layer_current[layer.index()] as usize]
    }

    fn record_span(&mut self, owner: Owner, key: StateKey, line: usize) {
        if !matches!(key.lifecycle(), Lifecycle::Span { .. }) {
            return;
        }
        match owner {
            Owner::Layer(layer) => {
                self.layer_opened.insert((layer, key), line);
            }
            Owner::Track(track) => {
                self.track_opened.insert((track, key), line);
            }
        }
    }

    /// Add one line's meter tandems for `track` to its open declaration
    /// group; `chosen_column` holds the tandem the group folds.
    fn declare_meter(
        &mut self,
        track: TrackId,
        index: usize,
        chosen_column: usize,
        chosen: &TandemInterpretation,
        tandems: &[(usize, &TandemInterpretation)],
    ) {
        let before = &self.track_states[self.track_current[track.index()] as usize].meter;
        let group = self.meter_groups[track.index()].get_or_insert_with(|| OpenMeterGroup {
            before: before.clone(),
            tokens: Vec::new(),
        });
        group.tokens.extend(
            tandems
                .iter()
                .map(|&(column, _)| (index, column, column == chosen_column)),
        );
        self.meters[track.index()].apply(chosen);
    }

    /// Close every track's declaration group: the completed group decides
    /// the meter in force and the outcome of each of its tandems.
    fn end_meter_groups(&mut self) {
        for track in 0..self.meters.len() {
            self.meters[track].end_group();
            let Some(group) = self.meter_groups[track].take() else {
                continue;
            };
            let declaration = self.meters[track].declaration().clone();
            let change = declaration.change_from(&group.before);
            let current = &self.track_states[self.track_current[track] as usize];
            if change != MeterChange::Silent && current.meter != declaration {
                let mut state = current.clone();
                state.meter = declaration;
                self.track_current[track] = push(&mut self.track_states, state);
            }
            for (line_index, column, declares) in group.tokens {
                let outcome = match change {
                    MeterChange::Changed if declares => Outcome::Changed,
                    MeterChange::Changed => Outcome::Coalesced,
                    MeterChange::Restated => Outcome::Restated,
                    MeterChange::Silent => Outcome::Redundant,
                };
                self.event(line_index, column, StateKey::Meter, outcome);
            }
        }
    }

    fn event(&mut self, line_index: usize, column: usize, key: StateKey, outcome: Outcome) {
        self.events.push(AppliedEvent {
            line_index,
            column,
            key,
            outcome,
        });
    }

    fn transition_error(
        &mut self,
        error: TransitionError,
        line: usize,
        column: usize,
        key: StateKey,
    ) {
        let column = column + 1;
        self.errors.push(match error {
            TransitionError::UnmatchedClose => StateError::UnmatchedClose { line, column, key },
            TransitionError::SpanMismatch { open } => StateError::SpanMismatch {
                line,
                column,
                key,
                open,
            },
            TransitionError::StrayNatural { step, previous } => StateError::StrayNatural {
                line,
                column,
                step,
                previous,
            },
        });
    }
}

/// A track's meter declaration group while its tandem lines last. Its
/// tandems wait for the group to end, since a later `*met()` can hide what an
/// earlier `*M` declared.
struct OpenMeterGroup {
    before: MeterDeclaration,
    /// Line index, column, and whether the column carried the line's value
    /// rather than a sibling copy.
    tokens: Vec<(usize, usize, bool)>,
}

#[derive(Debug, Clone, Copy)]
enum Owner {
    Layer(LayerId),
    Track(TrackId),
}

fn outcome(key: StateKey, effect: Effect) -> Outcome {
    match effect {
        Effect::Changed => Outcome::Changed,
        Effect::Redundant if key.drawn_when_restated() => Outcome::Restated,
        Effect::Redundant => Outcome::Redundant,
    }
}

fn push<T>(arena: &mut Vec<T>, value: T) -> u32 {
    arena.push(value);
    (arena.len() - 1) as u32
}

#[cfg(test)]
mod tests;
