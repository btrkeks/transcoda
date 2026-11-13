use crate::types::{
    Accidental, Articulation, AuxInterval, Bowing, CrossStaffPlacement, FermataPlacement,
    GraceKind, MarkPlacement, MordentDirection, Ornament, SlurPlacement, StemDirection, Tie,
    TiePlacement, TurnForm, TurnTiming,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct MarkerId(pub(crate) u8);

impl MarkerId {
    pub(crate) fn row(self) -> &'static Marker {
        &MARKERS[usize::from(self.0)]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EventKind {
    Note,
    Grace,
    Rest,
}

impl EventKind {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 3] = [Self::Note, Self::Grace, Self::Rest];
}

pub(crate) struct Marker {
    pub name: &'static str,
    #[cfg_attr(not(test), allow(dead_code, reason = "read by the table tests"))]
    pub spell: &'static [&'static str],
    pub lexeme: Lexeme,
    pub item: Item,
    pub slot: Option<Slot>,
    pub after: After,
    pub signifier_block: SignifierBlock,
    pub note: On,
    pub grace: On,
    pub rest: On,
}

impl Marker {
    const fn spelled(self, spell: &'static [&'static str]) -> Self {
        Self { spell, ..self }
    }

    const fn placed(self) -> Self {
        Self {
            note: self.note.placed(),
            grace: self.grace.placed(),
            ..self
        }
    }

    pub(crate) fn on(&self, event: EventKind) -> &On {
        match event {
            EventKind::Note => &self.note,
            EventKind::Grace => &self.grace,
            EventKind::Rest => &self.rest,
        }
    }

    #[cfg(test)]
    pub(crate) fn spellings(&'static self) -> &'static [&'static str] {
        match &self.lexeme {
            Lexeme::Lit(text) => std::slice::from_ref(text),
            _ => self.spell,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lexeme {
    Lit(&'static str),
    Run { byte: u8, limit: Limit },
    Letters,
    Duration,
    Bracket(u8),
    Turn { on_note: bool, base: u8 },
    Mordent(u8),
}

impl Lexeme {
    pub(crate) const fn starts_with(self, byte: u8) -> bool {
        match self {
            Self::Lit(text) => !text.is_empty() && text.as_bytes()[0] == byte,
            Self::Run { byte: run, .. } => run == byte,
            Self::Letters => matches!(byte, b'A'..=b'G' | b'a'..=b'g'),
            Self::Duration => byte.is_ascii_digit(),
            Self::Bracket(bracket) => byte == b'&' || byte == bracket,
            Self::Turn { on_note: true, .. } => byte == b's',
            Self::Turn {
                on_note: false,
                base,
            } => byte == base,
            Self::Mordent(letter) => byte.to_ascii_uppercase() == letter,
        }
    }

    const fn shape(self) -> u8 {
        match self {
            Self::Lit(_) => 0,
            Self::Run { .. } => 1,
            Self::Letters => 2,
            Self::Duration => 3,
            Self::Bracket(_) => 4,
            Self::Turn { .. } => 5,
            Self::Mordent(_) => 6,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Bucket {
    rows: [MarkerId; 8],
    len: u8,
    pub lexeme: Option<Lexeme>,
}

impl Bucket {
    pub(crate) fn rows(&self) -> &[MarkerId] {
        &self.rows[..usize::from(self.len)]
    }
}

const fn index(rows: &[Marker]) -> [Bucket; 128] {
    let mut index = [Bucket {
        rows: [MarkerId(0); 8],
        len: 0,
        lexeme: None,
    }; 128];
    let mut byte = 0;
    while byte < 128 {
        let bucket = &mut index[byte];
        let mut row = 0;
        while row < rows.len() {
            let lexeme = rows[row].lexeme;
            if lexeme.starts_with(byte as u8) {
                match bucket.lexeme {
                    Some(shared) if shared.shape() != lexeme.shape() => {
                        panic!("rows sharing a first byte must share a lexeme shape")
                    }
                    Some(_) => {}
                    None => bucket.lexeme = Some(lexeme),
                }
                bucket.rows[bucket.len as usize] = MarkerId(row as u8);
                bucket.len += 1;
            }
            row += 1;
        }
        byte += 1;
    }
    index
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Limit {
    None,
    Malformed(u8),
    FaultAt(u8),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Item {
    Head(HeadItem),
    Token(TokenItem),
    Placement(Dir),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HeadItem {
    Duration,
    Pitch,
    RestPosition,
    Rest,
    Accidental(Accidental),
    Grace(GraceKind),
    ForcedAccidental,
    InvisibleAccidental,
    PositionDisplay,
    Editorial,
    Invisible,
    CrossStaff(CrossStaffPlacement),
    Tie(Tie),
    UserSignifier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TokenItem {
    Slur { open: bool, linked: bool },
    Phrase { open: bool },
    Arpeggio,
    Fermata,
    Breath,
    Harmonic,
    Articulation(Articulation),
    Bowing(Bowing),
    Stem(StemDirection),
    Beam(BeamMark),
    Turn { form: TurnForm, timing: TurnTiming },
    Trill(AuxInterval),
    Mordent(MordentDirection),
    Ornament(Ornament),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BeamMark {
    Start,
    End,
    PartialLeft,
    PartialRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dir {
    Above,
    Below,
}

macro_rules! from_dir {
    ($($placement:ty),*) => {$(
        impl From<Dir> for $placement {
            fn from(dir: Dir) -> Self {
                match dir {
                    Dir::Above => Self::Above,
                    Dir::Below => Self::Below,
                }
            }
        }
    )*};
}

from_dir!(SlurPlacement, FermataPlacement, TiePlacement, MarkPlacement);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum After {
    Any,
    Core,
    Pitch,
    Accidental,
    Signifier,
    Rest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SignifierBlock {
    Keep,
    Block,
    Unblock,
    BlockAfterPitch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum On {
    No,
    Yes(Use),
}

impl On {
    const fn placed(self) -> Self {
        match self {
            Self::Yes(rule) => Self::Yes(Use {
                place: Place::Field,
                ..rule
            }),
            Self::No => Self::No,
        }
    }

    const fn hoist_rank(self, rank: u8) -> Self {
        match self {
            Self::Yes(rule) => Self::Yes(Use {
                hoist_rank: Some(rank),
                ..rule
            }),
            Self::No => Self::No,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Use {
    pub place: Place,
    pub dup: Dup,
    pub at: At,
    pub hoist_rank: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Place {
    Field,
    Core,
    /// The signifier attaches to nothing: kept as an atom, left out of the AST, and warned.
    Warn,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dup {
    Reject,
    /// Keep the first and warn on the repeat.
    Warn,
    LastWins,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Slot {
    Duration,
    Pitch,
    Accidental,
    AccidentalDisplay,
    Grace,
    Invisible,
    CrossStaff,
    Rest,
    RestPosition,
    Turn,
    Trill,
    Mordent,
    Fermata,
    Breath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct At(pub(crate) u8);

impl At {
    pub(crate) const SLURS: Self = Self(1 << 0);
    pub(crate) const PHRASES: Self = Self(1 << 1);
    pub(crate) const DURATION: Self = Self(1 << 2);
    pub(crate) const CORE: Self = Self(1 << 3);
    pub(crate) const SUFFIX: Self = Self(1 << 4);
    pub(crate) const ANY: Self = Self(0x1f);

    pub(crate) const fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The earliest region in `allowed` that is not before `self`: rest regions only move forward.
    pub(crate) fn earliest_allowed(self, allowed: Self) -> Option<Self> {
        let later = allowed.0 & !(self.0 - 1);
        (later != 0).then(|| Self(later & later.wrapping_neg()))
    }
}

use Dup as D;
use Place as P;

const fn both(place: Place, dup: Dup) -> On {
    uses(place, dup, At::ANY)
}

const fn uses(place: Place, dup: Dup, at: At) -> On {
    On::Yes(Use {
        place,
        dup,
        at,
        hoist_rank: None,
    })
}

const NO: On = On::No;

const BASE: Marker = Marker {
    name: "",
    spell: &[],
    lexeme: Lexeme::Lit(""),
    item: Item::Head(HeadItem::UserSignifier),
    slot: None,
    after: After::Any,
    signifier_block: SignifierBlock::Keep,
    note: NO,
    grace: NO,
    rest: NO,
};

const WARN: On = both(P::Warn, D::LastWins);

const fn barrier(name: &'static str, text: &'static str) -> Marker {
    Marker {
        name,
        lexeme: Lexeme::Lit(text),
        item: Item::Head(HeadItem::UserSignifier),
        signifier_block: SignifierBlock::Block,
        note: both(P::Reject, D::LastWins),
        grace: both(P::Reject, D::LastWins),
        ..BASE
    }
}

const fn accidental(name: &'static str, lexeme: Lexeme, sign: Accidental) -> Marker {
    Marker {
        name,
        lexeme,
        item: Item::Head(HeadItem::Accidental(sign)),
        slot: Some(Slot::Accidental),
        after: After::Pitch,
        signifier_block: SignifierBlock::Unblock,
        note: both(P::Core, D::Reject),
        grace: both(P::Core, D::Reject),
        ..BASE
    }
}

const fn grace(name: &'static str, text: &'static str, kind: GraceKind) -> Marker {
    Marker {
        name,
        lexeme: Lexeme::Lit(text),
        item: Item::Head(HeadItem::Grace(kind)),
        slot: Some(Slot::Grace),
        signifier_block: SignifierBlock::BlockAfterPitch,
        grace: both(P::Reject, D::Reject),
        ..BASE
    }
}

const fn position_display(name: &'static str, text: &'static str) -> Marker {
    Marker {
        name,
        lexeme: Lexeme::Lit(text),
        item: Item::Head(HeadItem::PositionDisplay),
        after: After::Signifier,
        signifier_block: SignifierBlock::Block,
        note: both(P::Reject, D::LastWins),
        grace: both(P::Reject, D::LastWins),
        ..BASE
    }
}

const fn accidental_display(name: &'static str, text: &'static str, item: HeadItem) -> Marker {
    Marker {
        name,
        lexeme: Lexeme::Lit(text),
        item: Item::Head(item),
        slot: Some(Slot::AccidentalDisplay),
        after: After::Accidental,
        signifier_block: SignifierBlock::Unblock,
        note: both(P::Core, D::Reject),
        grace: both(P::Core, D::Reject),
        ..BASE
    }
}

pub(crate) const fn signifier(dir: Dir) -> &'static str {
    match dir {
        Dir::Above => ">",
        Dir::Below => "<",
    }
}

const fn cross_staff(name: &'static str, dir: Dir) -> Marker {
    Marker {
        name,
        lexeme: Lexeme::Lit(signifier(dir)),
        item: Item::Head(HeadItem::CrossStaff(match dir {
            Dir::Above => CrossStaffPlacement::Above,
            Dir::Below => CrossStaffPlacement::Below,
        })),
        slot: Some(Slot::CrossStaff),
        after: After::Core,
        signifier_block: SignifierBlock::Unblock,
        note: both(P::Reject, D::Reject),
        grace: both(P::Reject, D::Reject),
        rest: uses(P::Reject, D::Reject, At::CORE),
        ..BASE
    }
}

const fn place(name: &'static str, dir: Dir) -> Marker {
    Marker {
        name,
        lexeme: Lexeme::Lit(signifier(dir)),
        item: Item::Placement(dir),
        signifier_block: SignifierBlock::Unblock,
        note: both(P::Reject, D::LastWins),
        grace: both(P::Reject, D::LastWins),
        rest: uses(P::Reject, D::LastWins, At::ANY),
        ..BASE
    }
}

const fn tie(name: &'static str, text: &'static str, tie: Tie) -> Marker {
    let on = if tie.is_end() {
        WARN
    } else if tie.is_start() {
        both(P::Field, D::LastWins).hoist_rank(2)
    } else {
        both(P::Field, D::LastWins)
    };
    Marker {
        name,
        lexeme: Lexeme::Lit(text),
        item: Item::Head(HeadItem::Tie(tie)),
        note: on,
        grace: on,
        ..BASE
    }
}

const fn token_mark(name: &'static str, lexeme: Lexeme, item: TokenItem, _fmt: u8) -> Marker {
    Marker {
        name,
        lexeme,
        item: Item::Token(item),
        note: WARN,
        grace: WARN,
        ..BASE
    }
}

const fn ornament(name: &'static str, lexeme: Lexeme, item: TokenItem, slot: Slot) -> Marker {
    Marker {
        slot: Some(slot),
        note: both(P::Field, D::Reject),
        grace: both(P::Field, D::Reject),
        ..token_mark(name, lexeme, item, 7)
    }
}

const fn turn(name: &'static str, form: TurnForm, timing: TurnTiming) -> Marker {
    ornament(
        name,
        Lexeme::Turn {
            on_note: matches!(timing, TurnTiming::OnNote),
            base: match form {
                TurnForm::Regular => b'S',
                TurnForm::Inverted => b'$',
            },
        },
        TokenItem::Turn { form, timing },
        Slot::Turn,
    )
}

pub(crate) static MARKERS: [Marker; 75] = ROWS;

pub(crate) static INDEX: [Bucket; 128] = index(&ROWS);

const GRACE_BYTES: u128 = grace_bytes(&ROWS);

pub(crate) fn is_grace_byte(byte: u8) -> bool {
    byte < 128 && GRACE_BYTES >> byte & 1 == 1
}

const fn grace_bytes(rows: &[Marker]) -> u128 {
    let mut set = 0;
    let mut row = 0;
    while row < rows.len() {
        if let (Item::Head(HeadItem::Grace(_)), Lexeme::Lit(text)) =
            (&rows[row].item, rows[row].lexeme)
        {
            let mut at = 0;
            while at < text.len() {
                set |= 1 << text.as_bytes()[at];
                at += 1;
            }
        }
        row += 1;
    }
    set
}

const ROWS: [Marker; 75] = [
    Marker {
        name: "duration",
        spell: &["4", "8.", "16..", "3%2", "0", "00", "000"],
        lexeme: Lexeme::Duration,
        item: Item::Head(HeadItem::Duration),
        slot: Some(Slot::Duration),
        signifier_block: SignifierBlock::Unblock,
        note: both(P::Reject, D::Reject),
        grace: both(P::Reject, D::Reject),
        rest: uses(P::Reject, D::Reject, At::DURATION),
        ..BASE
    },
    Marker {
        name: "rest-position",
        spell: &["e", "dd", "GG"],
        lexeme: Lexeme::Letters,
        item: Item::Head(HeadItem::RestPosition),
        slot: Some(Slot::RestPosition),
        after: After::Rest,
        signifier_block: SignifierBlock::Unblock,
        rest: uses(P::Core, D::Reject, At::CORE),
        ..BASE
    },
    Marker {
        name: "pitch",
        spell: &["c", "cc", "C", "CC"],
        lexeme: Lexeme::Letters,
        item: Item::Head(HeadItem::Pitch),
        slot: Some(Slot::Pitch),
        signifier_block: SignifierBlock::Unblock,
        note: both(P::Core, D::Reject),
        grace: both(P::Core, D::Reject),
        ..BASE
    },
    Marker {
        name: "rest",
        spell: &["r", "rr"],
        lexeme: Lexeme::Run {
            byte: b'r',
            limit: Limit::None,
        },
        item: Item::Head(HeadItem::Rest),
        slot: Some(Slot::Rest),
        signifier_block: SignifierBlock::Unblock,
        rest: uses(P::Core, D::Reject, At::CORE),
        ..BASE
    },
    accidental(
        "sharp",
        Lexeme::Run {
            byte: b'#',
            limit: Limit::Malformed(2),
        },
        Accidental::Sharp,
    )
    .spelled(&["#", "##"]),
    accidental(
        "flat",
        Lexeme::Run {
            byte: b'-',
            limit: Limit::Malformed(2),
        },
        Accidental::Flat,
    )
    .spelled(&["-", "--"]),
    accidental("natural", Lexeme::Lit("n"), Accidental::Natural),
    grace("acciaccatura", "q", GraceKind::Acciaccatura),
    grace("accented-grace", "qq", GraceKind::AccentedGrace),
    grace("groupetto", "Q", GraceKind::Groupetto),
    grace("appoggiatura", "P", GraceKind::Appoggiatura),
    grace("post-appoggiatura", "p", GraceKind::PostAppoggiatura),
    position_display("position-display-x", "X"),
    position_display("position-display-y", "y"),
    accidental_display("forced-accidental", "X", HeadItem::ForcedAccidental),
    accidental_display("invisible-accidental", "y", HeadItem::InvisibleAccidental),
    Marker {
        name: "editorial",
        lexeme: Lexeme::Lit("x"),
        item: Item::Head(HeadItem::Editorial),
        note: both(P::Core, D::LastWins),
        grace: both(P::Core, D::LastWins),
        ..BASE
    },
    Marker {
        name: "invisible",
        lexeme: Lexeme::Lit("yy"),
        item: Item::Head(HeadItem::Invisible),
        slot: Some(Slot::Invisible),
        signifier_block: SignifierBlock::Unblock,
        note: both(P::Reject, D::Reject),
        grace: both(P::Reject, D::Reject),
        rest: uses(P::Reject, D::Reject, At::SUFFIX),
        ..BASE
    },
    cross_staff("cross-staff-above", Dir::Above),
    cross_staff("cross-staff-below", Dir::Below),
    place("place-above", Dir::Above),
    place("place-below", Dir::Below),
    tie("tie-start", "[", Tie::Start),
    tie("tie-start-disjunct", "[[", Tie::DisjunctStart),
    tie("tie-start-linked", "N[", Tie::LinkedStart),
    tie("tie-continue", "_", Tie::Continue),
    tie("tie-continue-disjunct", "__", Tie::DisjunctContinue),
    tie("tie-continue-linked", "N_", Tie::LinkedContinue),
    tie("tie-end", "]", Tie::End),
    tie("tie-end-disjunct", "]]", Tie::DisjunctEnd),
    tie("tie-end-linked", "N]", Tie::LinkedEnd),
    barrier("user-i", "i"),
    barrier("user-I", "I"),
    barrier("user-j", "j"),
    barrier("user-l", "l"),
    barrier("user-N", "N"),
    barrier("user-V", "V"),
    barrier("user-Z", "Z"),
    barrier("user-at", "@"),
    barrier("user-plus", "+"),
    barrier("user-bar", "|"),
    Marker {
        name: "slur-open",
        spell: &["(", "&(", "&&("],
        lexeme: Lexeme::Bracket(b'('),
        item: Item::Token(TokenItem::Slur {
            open: true,
            linked: false,
        }),
        note: both(P::Field, D::LastWins).hoist_rank(1),
        grace: both(P::Field, D::LastWins).hoist_rank(1),
        rest: uses(P::Field, D::LastWins, At::SLURS.or(At::SUFFIX)),
        ..BASE
    },
    Marker {
        name: "slur-open-linked",
        lexeme: Lexeme::Lit("N("),
        item: Item::Token(TokenItem::Slur {
            open: true,
            linked: true,
        }),
        note: both(P::Field, D::LastWins).hoist_rank(1),
        grace: both(P::Field, D::LastWins).hoist_rank(1),
        ..BASE
    },
    Marker {
        name: "slur-close",
        spell: &[")", "&)", "&&)"],
        lexeme: Lexeme::Bracket(b')'),
        item: Item::Token(TokenItem::Slur {
            open: false,
            linked: false,
        }),
        note: WARN,
        grace: WARN,
        rest: uses(P::Reject, D::LastWins, At::SUFFIX),
        ..BASE
    },
    Marker {
        name: "slur-close-linked",
        lexeme: Lexeme::Lit("N)"),
        item: Item::Token(TokenItem::Slur {
            open: false,
            linked: true,
        }),
        note: WARN,
        grace: WARN,
        ..BASE
    },
    Marker {
        name: "phrase-open",
        spell: &["{", "&{", "&&{"],
        lexeme: Lexeme::Bracket(b'{'),
        item: Item::Token(TokenItem::Phrase { open: true }),
        note: WARN.hoist_rank(0),
        grace: WARN.hoist_rank(0),
        rest: uses(P::Reject, D::LastWins, At::PHRASES),
        ..BASE
    },
    Marker {
        name: "phrase-close",
        spell: &["}", "&}", "&&}"],
        lexeme: Lexeme::Bracket(b'}'),
        item: Item::Token(TokenItem::Phrase { open: false }),
        note: WARN,
        grace: WARN,
        rest: uses(P::Reject, D::LastWins, At::SUFFIX),
        ..BASE
    },
    token_mark(
        "arpeggio",
        Lexeme::Run {
            byte: b':',
            limit: Limit::None,
        },
        TokenItem::Arpeggio,
        9,
    )
    .spelled(&[":", "::"]),
    Marker {
        name: "fermata",
        lexeme: Lexeme::Lit(";"),
        item: Item::Token(TokenItem::Fermata),
        slot: Some(Slot::Fermata),
        note: both(P::Field, D::Warn),
        grace: both(P::Field, D::Warn),
        rest: uses(P::Field, D::Reject, At::SUFFIX),
        ..BASE
    },
    Marker {
        name: "breath",
        lexeme: Lexeme::Lit(","),
        item: Item::Token(TokenItem::Breath),
        slot: Some(Slot::Breath),
        note: both(P::Field, D::Warn),
        grace: both(P::Field, D::Warn),
        rest: uses(P::Reject, D::Reject, At::SUFFIX),
        ..BASE
    },
    token_mark("harmonic", Lexeme::Lit("o"), TokenItem::Harmonic, 12).placed(),
    token_mark(
        "staccato",
        Lexeme::Lit("'"),
        TokenItem::Articulation(Articulation::Staccato),
        8,
    )
    .placed(),
    token_mark(
        "pizzicato",
        Lexeme::Lit("\""),
        TokenItem::Articulation(Articulation::Pizzicato),
        8,
    ),
    token_mark(
        "staccatissimo",
        Lexeme::Lit("`"),
        TokenItem::Articulation(Articulation::Staccatissimo),
        8,
    )
    .placed(),
    token_mark(
        "tenuto",
        Lexeme::Lit("~"),
        TokenItem::Articulation(Articulation::Tenuto),
        8,
    )
    .placed(),
    token_mark(
        "accent",
        Lexeme::Lit("^"),
        TokenItem::Articulation(Articulation::Accent),
        8,
    )
    .placed(),
    token_mark(
        "generic-articulation",
        Lexeme::Lit("z"),
        TokenItem::Articulation(Articulation::Generic),
        8,
    )
    .placed(),
    token_mark(
        "down-bow",
        Lexeme::Lit("u"),
        TokenItem::Bowing(Bowing::DownBow),
        13,
    )
    .placed(),
    token_mark(
        "up-bow",
        Lexeme::Lit("v"),
        TokenItem::Bowing(Bowing::UpBow),
        13,
    )
    .placed(),
    token_mark(
        "stem-up",
        Lexeme::Lit("/"),
        TokenItem::Stem(StemDirection::Up),
        14,
    ),
    token_mark(
        "stem-down",
        Lexeme::Lit("\\"),
        TokenItem::Stem(StemDirection::Down),
        14,
    ),
    token_mark(
        "beam-start",
        Lexeme::Run {
            byte: b'L',
            limit: Limit::None,
        },
        TokenItem::Beam(BeamMark::Start),
        15,
    )
    .spelled(&["L", "LL"]),
    token_mark(
        "beam-end",
        Lexeme::Run {
            byte: b'J',
            limit: Limit::None,
        },
        TokenItem::Beam(BeamMark::End),
        15,
    )
    .spelled(&["J", "JJ"]),
    token_mark(
        "beam-partial-left",
        Lexeme::Run {
            byte: b'k',
            limit: Limit::None,
        },
        TokenItem::Beam(BeamMark::PartialLeft),
        15,
    )
    .spelled(&["k", "kk"]),
    token_mark(
        "beam-partial-right",
        Lexeme::Run {
            byte: b'K',
            limit: Limit::None,
        },
        TokenItem::Beam(BeamMark::PartialRight),
        15,
    )
    .spelled(&["K", "KK"]),
    turn("turn", TurnForm::Regular, TurnTiming::Delayed)
        .spelled(&["S", "SSS", "SSs", "SsS", "Sss"]),
    turn("inverted-turn", TurnForm::Inverted, TurnTiming::Delayed)
        .spelled(&["$", "$SS", "$Ss", "$sS", "$ss"]),
    turn("turn-on-note", TurnForm::Regular, TurnTiming::OnNote).spelled(&["sS", "sSSs"]),
    turn(
        "inverted-turn-on-note",
        TurnForm::Inverted,
        TurnTiming::OnNote,
    )
    .spelled(&["s$", "s$sS"]),
    ornament(
        "trill",
        Lexeme::Run {
            byte: b'T',
            limit: Limit::FaultAt(3),
        },
        TokenItem::Trill(AuxInterval::WholeTone),
        Slot::Trill,
    )
    .spelled(&["T", "TT", "TTT"]),
    ornament(
        "trill-semitone",
        Lexeme::Run {
            byte: b't',
            limit: Limit::FaultAt(3),
        },
        TokenItem::Trill(AuxInterval::Semitone),
        Slot::Trill,
    )
    .spelled(&["t", "tt", "ttt"]),
    ornament(
        "mordent",
        Lexeme::Mordent(b'M'),
        TokenItem::Mordent(MordentDirection::Upper),
        Slot::Mordent,
    )
    .spelled(&["M", "m", "MM", "mm", "Mm"]),
    ornament(
        "inverted-mordent",
        Lexeme::Mordent(b'W'),
        TokenItem::Mordent(MordentDirection::Lower),
        Slot::Mordent,
    )
    .spelled(&["W", "w", "WW", "ww", "Ww"]),
    token_mark(
        "concluding-turn",
        Lexeme::Lit("R"),
        TokenItem::Ornament(Ornament::ConcludingTurn),
        7,
    ),
    token_mark(
        "generic-ornament",
        Lexeme::Lit("O"),
        TokenItem::Ornament(Ornament::Generic),
        7,
    ),
];
