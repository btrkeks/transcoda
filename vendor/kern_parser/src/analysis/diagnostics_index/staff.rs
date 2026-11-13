use crate::{
    score::{LayerOrigin, Score, SplitSide},
    Clef, KernLine, KeySignature, OttavaKind, SpineType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StaffIdentity {
    Known(u32),
    Unknown(usize),
}
impl StaffIdentity {
    pub fn number(self) -> Option<u32> {
        if let Self::Known(n) = self {
            Some(n)
        } else {
            None
        }
    }
}
#[derive(Debug, Clone)]
pub struct StaffColumn {
    pub staff: StaffIdentity,
    pub voice: String,
    pub part: Option<u32>,
    pub spine_type: SpineType,
    pub clef: Option<Clef>,
    pub key: Option<KeySignature>,
    /// Sounding minus written octaves for tandem ottava.
    pub ottava: Option<i8>,
    /// Performance-only conditional octave shift and pass.
    pub tremolo: bool,
}
/// Per-line, per-field view of a resolved score: staff and part numbers from
/// the topology, clef, key, ottava and tremolo from the field's track state.
#[derive(Debug, Clone)]
pub struct StaffMap {
    rows: Vec<Vec<StaffColumn>>,
}

impl StaffMap {
    /// A row holds the state once its line has applied; a spine manipulator
    /// line is laid out against, and keeps, the layers it manipulates.
    pub fn build(score: &Score<'_>) -> Self {
        let document = score.document();
        let topology = score.topology();
        let voices = voices(score);
        let mut rows = Vec::with_capacity(document.lines().len());
        for (index, line) in document.lines().iter().enumerate() {
            let layers = topology.layers_at(index);
            let manipulator = matches!(line, KernLine::SpineManipulatorLine(_));
            rows.push(
                layers
                    .iter()
                    .enumerate()
                    .map(|(column, &layer)| {
                        let track = topology.track(topology.track_of(layer));
                        let state = if manipulator {
                            score.state_at(index, column)
                        } else {
                            score.state_after(index, column)
                        }
                        .track;
                        StaffColumn {
                            staff: match track.staff {
                                Some(number) => StaffIdentity::Known(number),
                                None => StaffIdentity::Unknown(usize::MAX),
                            },
                            voice: voices[layer.index()].clone(),
                            part: track.part,
                            spine_type: track.spine_type.clone(),
                            clef: state.clef.clone(),
                            key: state.key_signature,
                            ottava: Some(state.ottava.map_or(0, octaves)),
                            tremolo: state.tremolo,
                        }
                    })
                    .collect(),
            );
        }
        Self { rows }
    }

    pub fn line(&self, index: usize) -> &[StaffColumn] {
        self.rows.get(index).map(Vec::as_slice).unwrap_or(&[])
    }
    pub fn get(&self, line: usize, field: usize) -> Option<&StaffColumn> {
        self.rows.get(line)?.get(field)
    }
}

fn octaves(kind: OttavaKind) -> i8 {
    match kind {
        OttavaKind::OctaveUp => 1,
        OttavaKind::OctaveDown => -1,
        OttavaKind::TwoOctavesUp => 2,
        OttavaKind::TwoOctavesDown => -2,
    }
}

/// Voice label of every layer, indexed by `LayerId`: `r` and the 1-based
/// ordinal of the track's `**kern` track (its own, or the nearest on its
/// left), `.1`/`.2` per split side, and a merge keeps its leftmost input's
/// label.
fn voices(score: &Score<'_>) -> Vec<String> {
    let topology = score.topology();
    let mut roots = Vec::with_capacity(topology.track_count());
    let mut kern = 0_usize;
    let mut owner = None;
    let mut born = None;
    for track in topology.track_ids() {
        let track = topology.track(track);
        if born != Some(track.born) {
            owner = None;
            born = Some(track.born);
        }
        if track.spine_type == SpineType::Kern {
            kern += 1;
            owner = Some(kern);
        }
        roots.push(owner.unwrap_or(usize::MAX));
    }
    let mut voices: Vec<String> = Vec::with_capacity(topology.layer_count());
    for layer in topology.layer_ids() {
        let voice = match &topology.layer(layer).origin {
            LayerOrigin::Exclusive => format!("r{}", roots[topology.track_of(layer).index()]),
            LayerOrigin::SplitOf(parent) => {
                let side = match topology.split_side(layer) {
                    Some(SplitSide::Right) => 2,
                    _ => 1,
                };
                format!("{}.{side}", voices[parent.index()])
            }
            LayerOrigin::MergeOf(inputs) => voices[inputs[0].index()].clone(),
        };
        voices.push(voice);
    }
    voices
}
