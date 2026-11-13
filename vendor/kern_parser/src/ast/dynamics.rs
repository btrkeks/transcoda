use super::RawToken;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamContent {
    /// `.` null data token, or a token whose every part is null.
    Null,
    /// The marks of a token, one slot per kind of mark.
    ///
    /// Examples: `p`, `<`, `[[ f`, `<[`, `[ f >`.
    Marks(DynamMarks),
    /// A token with conflicting or unknown parts, such as `p f`, `< >` or
    /// `cresc.`, kept byte for byte. Normalization never edits it.
    Raw {
        text: RawToken,
        /// The hairpin characters in `text`, which Verovio still reads. Only
        /// the hairpin slots are filled, and a begin of each direction keeps
        /// its own slot.
        marks: DynamMarks,
    },
}

/// Every mark one `**dynam` token can carry.
///
/// Each kind of mark has one slot, so repeating a part (`< <`, `[ [`, `p p`)
/// has nowhere to go: the parser keeps one and the serializer writes it once,
/// in the order end, continuation, level, silence, placeholder, start, with
/// crescendo before diminuendo among ends, continuations and starts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DynamMarks {
    pub crescendo: HairpinSlots,
    pub diminuendo: HairpinSlots,
    pub level: Option<DynamicLevel>,
    /// `r` silence/rest marker.
    pub silence: bool,
    /// `???`, the marker musicxml2hum writes for direction text it could not
    /// read. It never corresponds to a printed sign.
    pub placeholder: bool,
}

impl DynamMarks {
    pub fn hairpin(&self, direction: DynamicHairpinDirection) -> &HairpinSlots {
        match direction {
            DynamicHairpinDirection::Crescendo => &self.crescendo,
            DynamicHairpinDirection::Diminuendo => &self.diminuendo,
        }
    }

    pub fn hairpin_mut(&mut self, direction: DynamicHairpinDirection) -> &mut HairpinSlots {
        match direction {
            DynamicHairpinDirection::Crescendo => &mut self.crescendo,
            DynamicHairpinDirection::Diminuendo => &mut self.diminuendo,
        }
    }

    /// No marks at all. A placeholder counts as a mark on purpose, so a token
    /// keeps its `???` unless a pass drops it explicitly.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// The hairpin marks of one direction on one token.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HairpinSlots {
    /// `[` or `[[` (crescendo), `]` or `]]` (diminuendo).
    pub end: Option<HairpinEnd>,
    /// `(` (crescendo) or `)` (diminuendo).
    pub continues: Option<HairpinContinue>,
    /// `<` or `<[` (crescendo), `>` or `>]` (diminuendo).
    pub begin: Option<HairpinBegin>,
}

impl HairpinSlots {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HairpinEnd {
    /// `[[`: the hairpin ends at this onset plus the line's duration, which
    /// reaches the barline when the token is last in its bar.
    pub to_line_end: bool,
    pub provenance: Option<DynamicProvenance>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HairpinContinue {
    pub provenance: Option<DynamicProvenance>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HairpinBegin {
    /// `<[`: the hairpin spans only the duration of the left note. An end
    /// part that follows a begin of its direction in the same token folds
    /// into this flag, so `[ <` (end, then begin) and `<[` differ by part
    /// order.
    pub single_note: bool,
    pub provenance: Option<DynamicProvenance>,
}

/// Canonical dynamic level in a `**dynam` spine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicLevel {
    /// Core level, such as `p`, `mf`, or `ff`.
    pub value: DynamicLevelValue,
    /// `s` prefix for subito dynamics, such as `sf` or `spp`.
    pub subito: bool,
    /// `z` signifier for sforzando/forte-piano behavior.
    pub sforzando: bool,
    /// `v` signifier for notated accent or stress.
    pub accent: bool,
    /// `X` or `x` suffix indicating interpreted/editorial provenance.
    pub provenance: Option<DynamicProvenance>,
}

/// Core dynamic level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicLevelValue {
    /// Repeated `p` levels: `p`, `pp`, `ppp`, etc.
    Piano { count: u8 },
    /// Repeated `f` levels: `f`, `ff`, `fff`, etc.
    Forte { count: u8 },
    /// `mp`.
    MezzoPiano,
    /// `mf`.
    MezzoForte,
    /// `fp`; `sfp` is the subito form.
    FortePiano,
    /// `rf`; `rfz` adds the sforzando signifier.
    Rinforzando,
}

/// Direction of gradual dynamic change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DynamicHairpinDirection {
    Crescendo,
    Diminuendo,
}

impl DynamicHairpinDirection {
    pub const ALL: [Self; 2] = [Self::Crescendo, Self::Diminuendo];

    pub fn opposite(self) -> Self {
        match self {
            Self::Crescendo => Self::Diminuendo,
            Self::Diminuendo => Self::Crescendo,
        }
    }
}

/// Provenance suffix for interpreted dynamic tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicProvenance {
    /// `X`: explicit interpretation not indicated in the score.
    Explicit,
    /// `x`: published/editorial interpretation indicated in the score.
    Editorial,
}
