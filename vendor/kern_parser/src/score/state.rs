//! Registry of interpretation state.
//!
//! Every tandem interpretation that persists until it is replaced or closed
//! is a slot here. A slot's scope is the struct it lives in: [`LayerState`]
//! for state owned by one subspine, [`TrackState`] for state owned by the
//! whole original spine. This file is the only place that decides scope,
//! lifecycle, transitions and the tandems that re-establish a value.

use crate::{
    meter::{MeterDeclaration, MeterOutput},
    types::DiatonicStep,
    BracketSpan, BracketSpanKind, Clef, Cue, HalfNoteStemSide, KeyCancellation, KeyDesignation,
    KeySignature, Ottava, OttavaKind, Pedal, Repetition, RhythmScale, StaffPlacement, Stria,
    TandemInterpretation, Tremolo, Tuplet, TupletBracket, VisualStyle,
};

/// One persistent interpretation slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateKey {
    Clef,
    KeySignature,
    KeyDesignation,
    Meter,
    StaffLines,
    Ottava,
    Pedal,
    TupletNumbers,
    TupletBrackets,
    Tremolo,
    Repetition,
    Ligature,
    Coloration,
    HalfNoteStemSide,
    StaffPlacement,
    Color,
    Cue,
    RhythmScale,
    KeyCancellation,
}

/// Who owns a slot's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// One subspine. A split copies the value to both children; merging
    /// layers must agree.
    Layer,
    /// The original spine. A token in any subspine sets it for all of them.
    Track,
}

/// How a slot's value starts and ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    /// A value that holds until replaced: clef, key, meter, `*Xtuplet`,
    /// `*rscale`.
    Assignment,
    /// An on/off toggle: `*ped`, `*tremolo`. Restating either side is
    /// redundant.
    Toggle,
    /// An open/close span: closing one that is not open is an error.
    Span {
        /// Whether the span must close before the track's last layer
        /// terminates and before a complete document ends.
        closes_before_track_end: bool,
    },
}

impl StateKey {
    /// Emission order for re-establishing state: staff definition first.
    pub const ALL: [StateKey; 19] = [
        StateKey::Clef,
        StateKey::KeySignature,
        StateKey::KeyDesignation,
        StateKey::Meter,
        StateKey::StaffLines,
        StateKey::Ottava,
        StateKey::Pedal,
        StateKey::TupletNumbers,
        StateKey::TupletBrackets,
        StateKey::Tremolo,
        StateKey::Repetition,
        StateKey::Ligature,
        StateKey::Coloration,
        StateKey::HalfNoteStemSide,
        StateKey::StaffPlacement,
        StateKey::Color,
        StateKey::Cue,
        StateKey::RhythmScale,
        StateKey::KeyCancellation,
    ];

    /// The slots a page or snippet header restates: clef, key signature and
    /// meter.
    pub const STAFF_DEFINITION: [StateKey; 3] =
        [StateKey::Clef, StateKey::KeySignature, StateKey::Meter];

    pub fn of(tandem: &TandemInterpretation) -> Option<Self> {
        Some(match tandem {
            TandemInterpretation::Clef(_) => Self::Clef,
            TandemInterpretation::KeySignature(_) => Self::KeySignature,
            TandemInterpretation::KeyDesignation(_) => Self::KeyDesignation,
            TandemInterpretation::TimeSignature(_)
            | TandemInterpretation::FreeMeter
            | TandemInterpretation::MeterSymbol(_) => Self::Meter,
            TandemInterpretation::Stria(_) => Self::StaffLines,
            TandemInterpretation::Ottava(_) => Self::Ottava,
            TandemInterpretation::Pedal(_) => Self::Pedal,
            TandemInterpretation::Tuplet(_) => Self::TupletNumbers,
            TandemInterpretation::TupletBracket(_) => Self::TupletBrackets,
            TandemInterpretation::Tremolo(_) => Self::Tremolo,
            TandemInterpretation::Repetition(_) => Self::Repetition,
            TandemInterpretation::BracketSpan(BracketSpan {
                kind: BracketSpanKind::Ligature,
                ..
            }) => Self::Ligature,
            TandemInterpretation::BracketSpan(BracketSpan {
                kind: BracketSpanKind::Coloration,
                ..
            }) => Self::Coloration,
            TandemInterpretation::VisualStyle(VisualStyle::HalfNoteStemSide(_)) => {
                Self::HalfNoteStemSide
            }
            TandemInterpretation::VisualStyle(VisualStyle::StaffPlacement(_)) => {
                Self::StaffPlacement
            }
            TandemInterpretation::VisualStyle(VisualStyle::Color(_)) => Self::Color,
            TandemInterpretation::VisualStyle(VisualStyle::KeyCancellation(_)) => {
                Self::KeyCancellation
            }
            TandemInterpretation::Cue(_) => Self::Cue,
            TandemInterpretation::RhythmScale(_) => Self::RhythmScale,
            TandemInterpretation::Null
            | TandemInterpretation::TupletBreak
            | TandemInterpretation::Part(_)
            | TandemInterpretation::Staff(_)
            | TandemInterpretation::Group(_)
            | TandemInterpretation::Instrument(_)
            | TandemInterpretation::Tempo(_)
            | TandemInterpretation::SectionLabel(_)
            | TandemInterpretation::ExpansionList(_)
            | TandemInterpretation::Raw(_) => return None,
        })
    }

    pub fn scope(self) -> Scope {
        match self {
            Self::Color | Self::Cue | Self::StaffPlacement => Scope::Layer,
            _ => Scope::Track,
        }
    }

    pub fn lifecycle(self) -> Lifecycle {
        match self {
            Self::Ottava => Lifecycle::Span {
                closes_before_track_end: true,
            },
            Self::Cue
            | Self::Pedal
            | Self::Tremolo
            | Self::Repetition
            | Self::Ligature
            | Self::Coloration
            | Self::KeyCancellation => Lifecycle::Toggle,
            Self::Clef
            | Self::KeySignature
            | Self::KeyDesignation
            | Self::Meter
            | Self::StaffLines
            | Self::TupletNumbers
            | Self::TupletBrackets
            | Self::HalfNoteStemSide
            | Self::StaffPlacement
            | Self::Color
            | Self::RhythmScale => Lifecycle::Assignment,
        }
    }

    /// Whether Verovio draws a tandem that restates the value in force: the
    /// clef, key signature or time signature again, a pedal bounce or
    /// release mark, a new ligature or coloration bracket. Checked with the
    /// `verovio` CLI (6.3); restating any other slot draws nothing (Verovio
    /// reads `*stria` only before a staff's first data line).
    pub fn drawn_when_restated(self) -> bool {
        matches!(
            self,
            Self::Clef
                | Self::KeySignature
                | Self::Meter
                | Self::Pedal
                | Self::Ligature
                | Self::Coloration
        )
    }

    /// Whether the slot is part of [`Self::STAFF_DEFINITION`].
    pub fn defines_staff(self) -> bool {
        Self::STAFF_DEFINITION.contains(&self)
    }

    /// For a tandem of a toggle or span slot, whether it switches the slot on
    /// (`*ped`, `*8va`, `*cue`) rather than off (`*Xped`, `*X8va`, `*Xcue`);
    /// `None` for assignments and stateless tandems. Toggles and spans start
    /// off, so a tandem switches on exactly when it moves the default.
    pub fn switches_on(tandem: &TandemInterpretation) -> Option<bool> {
        let key = Self::of(tandem)?;
        if key.lifecycle() == Lifecycle::Assignment {
            return None;
        }
        let effect = match key.scope() {
            Scope::Layer => LayerState::default().apply(tandem),
            Scope::Track => TrackState::default().apply(tandem),
        };
        Some(effect == Ok(Effect::Changed))
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Clef => "clef",
            Self::KeySignature => "key signature",
            Self::KeyDesignation => "key designation",
            Self::Meter => "meter",
            Self::StaffLines => "staff line count",
            Self::Ottava => "ottava",
            Self::Pedal => "pedal",
            Self::TupletNumbers => "tuplet numbers",
            Self::TupletBrackets => "tuplet brackets",
            Self::Tremolo => "tremolo",
            Self::Repetition => "repetition",
            Self::Ligature => "ligature",
            Self::Coloration => "coloration",
            Self::HalfNoteStemSide => "half-note stem side",
            Self::StaffPlacement => "staff placement",
            Self::Color => "color",
            Self::Cue => "cue",
            Self::RhythmScale => "rhythm scale",
            Self::KeyCancellation => "key cancellation",
        }
    }
}

/// State owned by one subspine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayerState {
    pub color: Option<[u8; 3]>,
    pub cue: bool,
    /// `*above`/`*below`: Verovio reads it along each subspine, so two
    /// verses of a split lyrics spine can sit on opposite sides.
    pub staff_placement: Option<StaffPlacement>,
}

/// State owned by a whole original spine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackState {
    /// Stored visible; `yy` belongs to the token that wrote it.
    pub clef: Option<Clef>,
    pub key_signature: Option<KeySignature>,
    pub key_designation: Option<KeyDesignation>,
    pub meter: MeterDeclaration,
    /// `*striaN`.
    pub staff_lines: Option<u8>,
    pub ottava: Option<OttavaKind>,
    pub pedal: bool,
    pub tuplet_numbers: bool,
    pub tuplet_brackets: bool,
    pub tremolo: bool,
    pub repetition: bool,
    pub ligature: bool,
    pub coloration: bool,
    pub half_note_stem_side: Option<HalfNoteStemSide>,
    pub rhythm_scale: RhythmScale,
    /// `*kcancel` as this spine wrote it. Verovio keeps one flag for all
    /// staves, and the resolver reads it that way (see
    /// `score::key_cancellation`); the slot records where the file wrote it,
    /// so rebuilt state restates it there.
    pub key_cancellation: bool,
}

impl Default for TrackState {
    fn default() -> Self {
        Self {
            clef: None,
            key_signature: None,
            key_designation: None,
            meter: MeterDeclaration::default(),
            staff_lines: None,
            ottava: None,
            pedal: false,
            tuplet_numbers: true,
            tuplet_brackets: true,
            tremolo: false,
            repetition: false,
            ligature: false,
            coloration: false,
            half_note_stem_side: None,
            rhythm_scale: RhythmScale::UNIT,
            key_cancellation: false,
        }
    }
}

/// Result of applying one state tandem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    Changed,
    /// The tandem restates the value already in force.
    Redundant,
}

/// A transition a slot refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionError {
    /// `*X8va` with no ottava open, `*Xcue` with no cue open.
    UnmatchedClose,
    /// `*8ba` while `*8va` is open, or `*X8ba` closing `*8va`. Carries the
    /// tandem that opened the span.
    SpanMismatch { open: TandemInterpretation },
    /// A key signature's cancellation natural on a step the previous key
    /// does not alter, or in a track's first key signature.
    StrayNatural {
        step: DiatonicStep,
        previous: Option<KeySignature>,
    },
}

/// The visible spelling used to compare tokens written for the same slot on
/// one line in sibling subspines.
pub(crate) fn visible(tandem: &TandemInterpretation) -> TandemInterpretation {
    match tandem {
        TandemInterpretation::Clef(clef) => TandemInterpretation::Clef(Clef {
            invisible: false,
            ..clef.clone()
        }),
        TandemInterpretation::TimeSignature(time_signature) => {
            let mut time_signature = *time_signature;
            time_signature.invisible = false;
            TandemInterpretation::TimeSignature(time_signature)
        }
        other => other.clone(),
    }
}

impl LayerState {
    pub(crate) fn apply(
        &mut self,
        tandem: &TandemInterpretation,
    ) -> Result<Effect, TransitionError> {
        match tandem {
            TandemInterpretation::Cue(Cue { active }) => Ok(assign(&mut self.cue, *active)),
            TandemInterpretation::VisualStyle(VisualStyle::Color(color)) => {
                Ok(assign(&mut self.color, Some(*color)))
            }
            TandemInterpretation::VisualStyle(VisualStyle::StaffPlacement(placement)) => {
                Ok(assign(&mut self.staff_placement, Some(*placement)))
            }
            _ => unreachable!("not a layer-scoped tandem: {tandem:?}"),
        }
    }

    /// Tandems that move `from` to `self`, per slot, in [`StateKey::ALL`]
    /// order.
    pub(crate) fn path_from(
        &self,
        from: &Self,
    ) -> Result<Vec<(StateKey, TandemInterpretation)>, Unrepresentable> {
        let mut path = Vec::new();
        if self.color != from.color {
            let color = self.color.ok_or(Unrepresentable(StateKey::Color))?;
            path.push((
                StateKey::Color,
                TandemInterpretation::VisualStyle(VisualStyle::Color(color)),
            ));
        }
        if self.staff_placement != from.staff_placement {
            let placement = self
                .staff_placement
                .ok_or(Unrepresentable(StateKey::StaffPlacement))?;
            path.push((
                StateKey::StaffPlacement,
                TandemInterpretation::VisualStyle(VisualStyle::StaffPlacement(placement)),
            ));
        }
        if self.cue != from.cue {
            path.push((
                StateKey::Cue,
                TandemInterpretation::Cue(Cue { active: self.cue }),
            ));
        }
        Ok(path)
    }

    pub(crate) fn holds_default(&self, key: StateKey) -> bool {
        let default = Self::default();
        match key {
            StateKey::Color => self.color == default.color,
            StateKey::Cue => self.cue == default.cue,
            StateKey::StaffPlacement => self.staff_placement == default.staff_placement,
            _ => unreachable!("not a layer slot: {key:?}"),
        }
    }

    /// Set `key` to its value in `from`.
    pub(crate) fn copy_slot(&mut self, from: &Self, key: StateKey) {
        match key {
            StateKey::Color => self.color = from.color,
            StateKey::Cue => self.cue = from.cue,
            StateKey::StaffPlacement => self.staff_placement = from.staff_placement,
            _ => unreachable!("not a layer slot: {key:?}"),
        }
    }
}

impl TrackState {
    /// Apply one tandem. Meter tandems take effect per declaration group,
    /// which the resolver folds through [`crate::ActiveMeter`].
    pub(crate) fn apply(
        &mut self,
        tandem: &TandemInterpretation,
    ) -> Result<Effect, TransitionError> {
        Ok(match tandem {
            TandemInterpretation::Clef(clef) => assign(
                &mut self.clef,
                Some(Clef {
                    invisible: false,
                    ..clef.clone()
                }),
            ),
            TandemInterpretation::KeySignature(key) => {
                let previous = self.key_signature;
                if let Some(step) = key.naturals().find(|&step| {
                    previous.is_none_or(|previous| previous.accidental_for_step(step).is_none())
                }) {
                    return Err(TransitionError::StrayNatural { step, previous });
                }
                assign(&mut self.key_signature, Some(key.without_naturals()))
            }
            TandemInterpretation::KeyDesignation(key) => {
                assign(&mut self.key_designation, Some(key.clone()))
            }
            TandemInterpretation::Stria(Stria { count }) => {
                assign(&mut self.staff_lines, Some(*count))
            }
            TandemInterpretation::Ottava(Ottava { kind, active }) => {
                return apply_span(&mut self.ottava, *kind, *active)
            }
            TandemInterpretation::Pedal(Pedal { active }) => assign(&mut self.pedal, *active),
            TandemInterpretation::Tuplet(Tuplet { show_number }) => {
                assign(&mut self.tuplet_numbers, *show_number)
            }
            TandemInterpretation::TupletBracket(TupletBracket { show_bracket }) => {
                assign(&mut self.tuplet_brackets, *show_bracket)
            }
            TandemInterpretation::Tremolo(Tremolo { active }) => assign(&mut self.tremolo, *active),
            TandemInterpretation::Repetition(Repetition { active }) => {
                assign(&mut self.repetition, *active)
            }
            TandemInterpretation::BracketSpan(BracketSpan { kind, active }) => match kind {
                BracketSpanKind::Ligature => assign(&mut self.ligature, *active),
                BracketSpanKind::Coloration => assign(&mut self.coloration, *active),
            },
            TandemInterpretation::VisualStyle(VisualStyle::HalfNoteStemSide(side)) => {
                assign(&mut self.half_note_stem_side, Some(*side))
            }
            TandemInterpretation::RhythmScale(scale) => assign(&mut self.rhythm_scale, *scale),
            TandemInterpretation::VisualStyle(VisualStyle::KeyCancellation(KeyCancellation {
                active,
            })) => assign(&mut self.key_cancellation, *active),
            TandemInterpretation::TimeSignature(_)
            | TandemInterpretation::FreeMeter
            | TandemInterpretation::MeterSymbol(_) => {
                unreachable!("meter tandems apply per declaration group")
            }
            _ => unreachable!("not a track-scoped tandem: {tandem:?}"),
        })
    }

    pub(crate) fn holds_default(&self, key: StateKey) -> bool {
        let default = Self::default();
        match key {
            StateKey::Clef => self.clef == default.clef,
            StateKey::KeySignature => self.key_signature == default.key_signature,
            StateKey::KeyDesignation => self.key_designation == default.key_designation,
            StateKey::Meter => self.meter.has_same_meter(&default.meter),
            StateKey::StaffLines => self.staff_lines == default.staff_lines,
            StateKey::Ottava => self.ottava == default.ottava,
            StateKey::Pedal => self.pedal == default.pedal,
            StateKey::TupletNumbers => self.tuplet_numbers == default.tuplet_numbers,
            StateKey::TupletBrackets => self.tuplet_brackets == default.tuplet_brackets,
            StateKey::Tremolo => self.tremolo == default.tremolo,
            StateKey::Repetition => self.repetition == default.repetition,
            StateKey::Ligature => self.ligature == default.ligature,
            StateKey::Coloration => self.coloration == default.coloration,
            StateKey::HalfNoteStemSide => self.half_note_stem_side == default.half_note_stem_side,
            StateKey::RhythmScale => self.rhythm_scale == default.rhythm_scale,
            StateKey::KeyCancellation => self.key_cancellation == default.key_cancellation,
            StateKey::Color | StateKey::Cue | StateKey::StaffPlacement => {
                unreachable!("not a track slot: {key:?}")
            }
        }
    }

    /// Set `key` to its value in `from`.
    pub(crate) fn copy_slot(&mut self, from: &Self, key: StateKey) {
        match key {
            StateKey::Clef => self.clef.clone_from(&from.clef),
            StateKey::KeySignature => self.key_signature = from.key_signature,
            StateKey::KeyDesignation => self.key_designation.clone_from(&from.key_designation),
            StateKey::Meter => self.meter.clone_from(&from.meter),
            StateKey::StaffLines => self.staff_lines = from.staff_lines,
            StateKey::Ottava => self.ottava = from.ottava,
            StateKey::Pedal => self.pedal = from.pedal,
            StateKey::TupletNumbers => self.tuplet_numbers = from.tuplet_numbers,
            StateKey::TupletBrackets => self.tuplet_brackets = from.tuplet_brackets,
            StateKey::Tremolo => self.tremolo = from.tremolo,
            StateKey::Repetition => self.repetition = from.repetition,
            StateKey::Ligature => self.ligature = from.ligature,
            StateKey::Coloration => self.coloration = from.coloration,
            StateKey::HalfNoteStemSide => self.half_note_stem_side = from.half_note_stem_side,
            StateKey::RhythmScale => self.rhythm_scale = from.rhythm_scale,
            StateKey::KeyCancellation => self.key_cancellation = from.key_cancellation,
            StateKey::Color | StateKey::Cue | StateKey::StaffPlacement => {
                unreachable!("not a track slot: {key:?}")
            }
        }
    }

    pub fn is_open(&self, key: StateKey) -> bool {
        match key {
            StateKey::Ottava => self.ottava.is_some(),
            _ => false,
        }
    }

    /// Tandems that move `from` to `self`, per slot, in [`StateKey::ALL`]
    /// order. A slot changes with one tandem, except an ottava switching
    /// kind (close, then open) and a meter (time signature, then symbol).
    pub(crate) fn path_from(
        &self,
        from: &Self,
    ) -> Result<Vec<(StateKey, TandemInterpretation)>, Unrepresentable> {
        let mut path = Vec::new();
        let mut push = |key, tandem| path.push((key, tandem));

        if self.clef != from.clef {
            let clef = self.clef.clone().ok_or(Unrepresentable(StateKey::Clef))?;
            push(StateKey::Clef, TandemInterpretation::Clef(clef));
        }
        if self.key_signature != from.key_signature {
            let key = self
                .key_signature
                .ok_or(Unrepresentable(StateKey::KeySignature))?;
            push(
                StateKey::KeySignature,
                TandemInterpretation::KeySignature(key),
            );
        }
        if self.key_designation != from.key_designation {
            let key = self
                .key_designation
                .clone()
                .ok_or(Unrepresentable(StateKey::KeyDesignation))?;
            push(
                StateKey::KeyDesignation,
                TandemInterpretation::KeyDesignation(key),
            );
        }
        if !self.meter.has_same_meter(&from.meter) {
            if !self.meter.is_declared() {
                return Err(Unrepresentable(StateKey::Meter));
            }
            for tandem in self
                .meter
                .tandems(MeterOutput::AsDeclared)
                .into_iter()
                .flatten()
            {
                push(StateKey::Meter, tandem);
            }
        }
        if self.staff_lines != from.staff_lines {
            let count = self
                .staff_lines
                .ok_or(Unrepresentable(StateKey::StaffLines))?;
            push(
                StateKey::StaffLines,
                TandemInterpretation::Stria(Stria { count }),
            );
        }
        if self.ottava != from.ottava {
            if let Some(kind) = from.ottava {
                push(
                    StateKey::Ottava,
                    TandemInterpretation::Ottava(Ottava {
                        kind,
                        active: false,
                    }),
                );
            }
            if let Some(kind) = self.ottava {
                push(
                    StateKey::Ottava,
                    TandemInterpretation::Ottava(Ottava { kind, active: true }),
                );
            }
        }
        if self.pedal != from.pedal {
            push(
                StateKey::Pedal,
                TandemInterpretation::Pedal(Pedal { active: self.pedal }),
            );
        }
        if self.tuplet_numbers != from.tuplet_numbers {
            push(
                StateKey::TupletNumbers,
                TandemInterpretation::Tuplet(Tuplet {
                    show_number: self.tuplet_numbers,
                }),
            );
        }
        if self.tuplet_brackets != from.tuplet_brackets {
            push(
                StateKey::TupletBrackets,
                TandemInterpretation::TupletBracket(TupletBracket {
                    show_bracket: self.tuplet_brackets,
                }),
            );
        }
        if self.tremolo != from.tremolo {
            push(
                StateKey::Tremolo,
                TandemInterpretation::Tremolo(Tremolo {
                    active: self.tremolo,
                }),
            );
        }
        if self.repetition != from.repetition {
            push(
                StateKey::Repetition,
                TandemInterpretation::Repetition(Repetition {
                    active: self.repetition,
                }),
            );
        }
        for (key, kind, now, before) in [
            (
                StateKey::Ligature,
                BracketSpanKind::Ligature,
                self.ligature,
                from.ligature,
            ),
            (
                StateKey::Coloration,
                BracketSpanKind::Coloration,
                self.coloration,
                from.coloration,
            ),
        ] {
            if now != before {
                push(
                    key,
                    TandemInterpretation::BracketSpan(BracketSpan { kind, active: now }),
                );
            }
        }
        if self.half_note_stem_side != from.half_note_stem_side {
            let side = self
                .half_note_stem_side
                .ok_or(Unrepresentable(StateKey::HalfNoteStemSide))?;
            push(
                StateKey::HalfNoteStemSide,
                TandemInterpretation::VisualStyle(VisualStyle::HalfNoteStemSide(side)),
            );
        }
        if self.rhythm_scale != from.rhythm_scale {
            push(
                StateKey::RhythmScale,
                TandemInterpretation::RhythmScale(self.rhythm_scale),
            );
        }
        if self.key_cancellation != from.key_cancellation {
            push(
                StateKey::KeyCancellation,
                TandemInterpretation::VisualStyle(VisualStyle::KeyCancellation(KeyCancellation {
                    active: self.key_cancellation,
                })),
            );
        }
        Ok(path)
    }

    /// This state with every open span closed.
    pub(crate) fn closed(&self) -> Self {
        Self {
            ottava: None,
            ..self.clone()
        }
    }
}

/// A state difference that no tandem can express, such as returning to "no
/// clef declared".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unrepresentable(pub StateKey);

fn assign<T: PartialEq>(slot: &mut T, value: T) -> Effect {
    if *slot == value {
        Effect::Redundant
    } else {
        *slot = value;
        Effect::Changed
    }
}

fn apply_span(
    slot: &mut Option<OttavaKind>,
    kind: OttavaKind,
    active: bool,
) -> Result<Effect, TransitionError> {
    match (*slot, active) {
        (None, true) => {
            *slot = Some(kind);
            Ok(Effect::Changed)
        }
        (Some(open), true) if open == kind => Ok(Effect::Redundant),
        (Some(open), _) if open != kind => Err(TransitionError::SpanMismatch {
            open: TandemInterpretation::Ottava(Ottava {
                kind: open,
                active: true,
            }),
        }),
        (Some(_), false) => {
            *slot = None;
            Ok(Effect::Changed)
        }
        (None, false) => Err(TransitionError::UnmatchedClose),
        (Some(_), true) => unreachable!("covered above"),
    }
}
