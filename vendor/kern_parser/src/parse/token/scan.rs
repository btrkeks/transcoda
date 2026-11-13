use std::ops::Range;

use super::{
    table::{
        signifier as signifier_text, After, At, BeamMark, Dir, Dup, EventKind, HeadItem, Item,
        Lexeme, Limit, Marker, MarkerId, On, Place, SignifierBlock, Slot, TokenItem, Use, INDEX,
    },
    AtomSink, Part,
};
use crate::types::{
    Accidental, Arpeggio, AuxInterval, Breath, ChordBeams, ChordModifiers, DiatonicStep, Duration,
    DurationValue, EditorialInterpretation, Elision, Fermata, ForcedAccidental, GraceKind,
    Harmonic, MordentDirection, NoteModifiers, Ornament, Phrase, Pitch, Placed, Slur,
    SlurPlacement, TrillLine, Turn, TurnForm, TurnIntervals, TurnTiming, UserSignifier,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Fault {
    pub span: Range<usize>,
    pub reason: Reason,
}

impl Fault {
    pub(super) fn at(offset: usize, reason: Reason) -> Self {
        Self {
            span: offset..offset,
            reason,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reason {
    Unknown,
    Incomplete,
    Malformed,
    NotOn(MarkerId),
    Order(MarkerId),
    Duplicate(MarkerId),
    Unattached(MarkerId),
    Shape(MarkerId),
    Scope,
    Missing(Missing),
    Chord(ChordFault),
    MeasureCount,
    Slice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChordFault {
    Spacing,
    GraceHead,
    RestHead,
    MissingGrace,
    Duration,
    Merge,
}

impl Reason {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::Unknown => "token.unknown-character",
            Self::Incomplete => "token.incomplete-mark",
            Self::Malformed => "token.malformed-mark",
            Self::NotOn(_) => "token.mark-not-allowed",
            Self::Order(_) => "token.mark-out-of-order",
            Self::Duplicate(_) => "token.duplicate-mark",
            Self::Unattached(_) => "token.unattached-signifier",
            Self::Shape(_) => "token.mark-shape",
            Self::Scope => "token.chord-mark-on-note",
            Self::Missing(Missing::Pitch) => "token.missing-pitch",
            Self::Missing(Missing::Duration) => "token.missing-duration",
            Self::Missing(Missing::Grace) => "token.missing-grace-mark",
            Self::Missing(Missing::Rest) => "token.incomplete-rest",
            Self::Missing(Missing::Beams) => "token.unbalanced-beams",
            Self::Chord(ChordFault::Spacing) => "token.chord-spacing",
            Self::Chord(ChordFault::GraceHead) => "token.grace-in-chord",
            Self::Chord(ChordFault::RestHead) => "token.rest-in-chord",
            Self::Chord(ChordFault::MissingGrace) => "token.chord-missing-grace-mark",
            Self::Chord(ChordFault::Duration) => "token.chord-duration-mismatch",
            Self::Chord(ChordFault::Merge) => "token.chord-merge-overflow",
            Self::MeasureCount => "token.measure-count",
            Self::Slice => "token.rest-slice",
        }
    }
}

/// A mark the scan keeps as an atom but leaves out of the AST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Warning {
    /// A `<`/`>` after `mark`, which takes no placement, or after an ignored duplicate.
    Unattached {
        signifier: Dir,
        mark: MarkerId,
        duplicate: bool,
    },
    /// A second `mark` on one head; the first is kept.
    Duplicate { mark: MarkerId },
}

impl Warning {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::Unattached { .. } => "token.unattached-signifier",
            Self::Duplicate { .. } => "token.duplicate-mark",
        }
    }

    pub(crate) fn message(self) -> String {
        match self {
            Self::Unattached {
                signifier,
                mark,
                duplicate: false,
            } => format!(
                "`{}` follows {}, which takes no placement; the parser ignores it",
                signifier_text(signifier),
                mark.row().name
            ),
            Self::Unattached {
                signifier, mark, ..
            } => format!(
                "`{}` follows the ignored second {}; the parser ignores it",
                signifier_text(signifier),
                mark.row().name
            ),
            Self::Duplicate { mark } => format!(
                "second {} on one head; the parser keeps the first and ignores this one",
                mark.row().name
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Missing {
    Pitch,
    Duration,
    Grace,
    Rest,
    Beams,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Lexed {
    pub range: Range<usize>,
    candidates: Candidates,
}

impl Lexed {
    pub(crate) fn candidates(&self) -> &[MarkerId] {
        &self.candidates.ids[..usize::from(self.candidates.len)]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidates {
    ids: [MarkerId; 2],
    len: u8,
}

impl Candidates {
    const NONE: Self = Self {
        ids: [MarkerId(0); 2],
        len: 0,
    };

    fn push(&mut self, id: MarkerId) {
        self.ids[usize::from(self.len)] = id;
        self.len += 1;
    }
}

pub(crate) struct Lexer<'a> {
    raw: &'a [u8],
    at: usize,
}

impl<'a> Lexer<'a> {
    pub(crate) fn new(raw: &'a str) -> Self {
        Self {
            raw: raw.as_bytes(),
            at: 0,
        }
    }

    pub(crate) fn next(&mut self) -> Result<Option<Lexed>, Fault> {
        let start = self.at;
        let Some(&byte) = self.raw.get(start) else {
            return Ok(None);
        };
        let unknown = || Fault {
            span: start..start + char_width(byte),
            reason: Reason::Unknown,
        };
        let bucket = INDEX.get(usize::from(byte)).ok_or_else(unknown)?;
        let lexeme = bucket.lexeme.ok_or_else(unknown)?;
        let mut candidates = Candidates::NONE;
        let end = match lexeme {
            Lexeme::Lit(_) => {
                let rest = &self.raw[start..];
                let mut longest = 0;
                for &id in bucket.rows() {
                    let Lexeme::Lit(text) = id.row().lexeme else {
                        continue;
                    };
                    if text.len() >= longest && rest.starts_with(text.as_bytes()) {
                        if text.len() > longest {
                            longest = text.len();
                            candidates = Candidates::NONE;
                        }
                        candidates.push(id);
                    }
                }
                if longest == 0 {
                    return Err(unknown());
                }
                start + longest
            }
            Lexeme::Run { .. } | Lexeme::Letters => self.run(start, |next| next == byte),
            Lexeme::Duration => {
                let mut end = self.run(start, |next| next.is_ascii_digit());
                if self.raw.get(end) == Some(&b'%') {
                    end = self.run(end + 1, |next| next.is_ascii_digit());
                }
                self.run(end, |next| next == b'.')
            }
            Lexeme::Bracket(_) => {
                let bracket = self.run(start, |next| next == b'&');
                if bracket - start > usize::from(u8::MAX) {
                    return Err(Fault {
                        span: start..bracket,
                        reason: Reason::Malformed,
                    });
                }
                bracket + 1
            }
            Lexeme::Turn { on_note, .. } => {
                let base = start + usize::from(on_note);
                let intervals = self.raw.get(base + 1..).unwrap_or_default();
                base + 1
                    + intervals
                        .iter()
                        .take(2)
                        .take_while(|next| matches!(next, b'S' | b's'))
                        .count()
            }
            Lexeme::Mordent(_) => self.run(start, |next| lexeme.starts_with(next)),
        };
        if candidates.len == 0 {
            let text = &self.raw[start..end.min(self.raw.len())];
            for &id in bucket.rows() {
                if lexes(id.row().lexeme, text) {
                    candidates.push(id);
                }
            }
        }
        if candidates.len == 0 {
            let at = match lexeme {
                Lexeme::Bracket(_) => end - 1,
                _ => start,
            };
            return Err(Fault::at(at, Reason::Incomplete));
        }
        self.at = end;
        Ok(Some(Lexed {
            range: start..end,
            candidates,
        }))
    }

    fn run(&self, from: usize, keep: impl Fn(u8) -> bool) -> usize {
        from + self.raw[from..]
            .iter()
            .take_while(|next| keep(**next))
            .count()
    }
}

fn lexes(lexeme: Lexeme, text: &[u8]) -> bool {
    match lexeme {
        Lexeme::Lit(spelling) => spelling.as_bytes() == text,
        Lexeme::Bracket(bracket) => text.last() == Some(&bracket),
        Lexeme::Turn { on_note, base } => text.get(usize::from(on_note)) == Some(&base),
        Lexeme::Run { .. } | Lexeme::Letters | Lexeme::Duration | Lexeme::Mordent(_) => true,
    }
}

fn char_width(first: u8) -> usize {
    (first.leading_ones() as usize).max(1)
}

#[derive(Clone, Copy)]
pub(crate) struct Admitted {
    pub id: MarkerId,
    pub row: &'static Marker,
    pub rule: &'static Use,
}

/// The rule an ignored duplicate leaves behind: a signifier after it is ignored with it.
static IGNORED_DUPLICATE: Use = Use {
    place: Place::Warn,
    dup: Dup::Warn,
    at: At::ANY,
    hoist_rank: None,
};

pub(crate) struct Rules {
    event: EventKind,
    prev: Option<Admitted>,
    signifiers_blocked: bool,
    seen: u32,
    region: At,
}

impl Rules {
    pub(crate) fn new(event: EventKind) -> Self {
        Self {
            event,
            prev: None,
            signifiers_blocked: false,
            seen: 0,
            region: At::SLURS,
        }
    }

    /// A warned atom is kept as an atom but must not reach the fold.
    pub(crate) fn admit(
        &mut self,
        lexed: &Lexed,
    ) -> Result<(Admitted, Option<Admitted>, Option<Warning>), Fault> {
        let fault = |reason| Fault {
            span: lexed.range.clone(),
            reason,
        };
        let candidates = lexed.candidates();
        let id = candidates
            .iter()
            .copied()
            .find(|id| self.holds(id.row().after))
            .ok_or_else(|| fault(Reason::Order(candidates[0])))?;
        let row = id.row();
        let On::Yes(rule) = row.on(self.event) else {
            return Err(fault(Reason::NotOn(id)));
        };
        let admitted = Admitted { id, row, rule };
        self.region = self
            .region
            .earliest_allowed(rule.at)
            .ok_or_else(|| fault(Reason::Order(id)))?;
        let slot = row.slot;
        let mut warning = None;
        if slot.is_some_and(|slot| self.has(slot)) {
            match rule.dup {
                Dup::Reject => return Err(fault(Reason::Duplicate(id))),
                Dup::Warn => warning = Some(Warning::Duplicate { mark: id }),
                Dup::LastWins => {}
            }
        }
        if let Item::Placement(dir) = row.item {
            match (self.target(), self.prev) {
                (Some(Place::Field), _) => {}
                (Some(Place::Warn), Some(prev)) => {
                    warning = Some(Warning::Unattached {
                        signifier: dir,
                        mark: prev.id,
                        duplicate: std::ptr::eq(prev.rule, &IGNORED_DUPLICATE),
                    })
                }
                _ => return Err(fault(Reason::Unattached(id))),
            }
        }
        self.signifiers_blocked = match row.signifier_block {
            SignifierBlock::Keep => self.signifiers_blocked,
            SignifierBlock::Block => true,
            SignifierBlock::Unblock => false,
            SignifierBlock::BlockAfterPitch => self.has(Slot::Pitch),
        };
        if let Some(slot) = slot {
            self.seen |= 1 << slot as u32;
        }
        let prev = match warning {
            Some(Warning::Duplicate { .. }) => Admitted {
                rule: &IGNORED_DUPLICATE,
                ..admitted
            },
            _ => admitted,
        };
        Ok((admitted, self.prev.replace(prev), warning))
    }

    fn has(&self, slot: Slot) -> bool {
        self.seen & (1 << slot as u32) != 0
    }

    fn target(&self) -> Option<Place> {
        if self.signifiers_blocked {
            return None;
        }
        self.prev.map(|prev| prev.rule.place)
    }

    fn holds(&self, after: After) -> bool {
        let prev_item = self.prev.map(|prev| &prev.row.item);
        match after {
            After::Any => true,
            After::Core => self.target() == Some(Place::Core),
            After::Pitch => self.target() == Some(Place::Core) && self.has(Slot::Pitch),
            After::Accidental => matches!(prev_item, Some(Item::Head(HeadItem::Accidental(_)))),
            After::Signifier => matches!(
                prev_item,
                Some(Item::Placement(_) | Item::Head(HeadItem::CrossStaff(_)))
            ),
            After::Rest => self.has(Slot::Rest),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Scanned {
    pub duration: Option<Duration>,
    pub core: Core,
    pub accidental: Option<Accidental>,
    pub grace: Option<(GraceKind, usize)>,
    pub head: NoteModifiers,
    pub token: ChordModifiers,
    pub first_token_mark: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Core {
    Pitch(Pitch),
    Rest { position: Option<Pitch> },
}

pub(crate) struct Fold {
    event: EventKind,
    duration: Option<Duration>,
    core: Option<Core>,
    accidental: Option<Accidental>,
    grace: Option<(GraceKind, usize)>,
    head: NoteModifiers,
    token: ChordModifiers,
    first_token_mark: Option<usize>,
    beams: BeamCounts,
    open_slur: usize,
}

impl Fold {
    pub(crate) fn new(event: EventKind) -> Self {
        Self {
            event,
            duration: None,
            core: None,
            accidental: None,
            grace: None,
            head: NoteModifiers::default(),
            token: ChordModifiers::default(),
            first_token_mark: None,
            beams: BeamCounts::default(),
            open_slur: 0,
        }
    }

    pub(crate) fn apply(
        &mut self,
        raw: &str,
        Admitted { id, row, .. }: Admitted,
        range: Range<usize>,
        prev: Option<Admitted>,
    ) -> Result<(), Fault> {
        let text = &raw[range.clone()];
        let start = range.start;
        let malformed = || Fault {
            span: range.clone(),
            reason: Reason::Malformed,
        };
        match row.lexeme {
            Lexeme::Run {
                limit: Limit::Malformed(max),
                ..
            } if text.len() > usize::from(max) => return Err(malformed()),
            Lexeme::Run {
                limit: Limit::FaultAt(max),
                ..
            } if text.len() > usize::from(max) => {
                return Err(Fault::at(start + usize::from(max), Reason::Shape(id)))
            }
            _ => {}
        }
        match &row.item {
            Item::Head(item) => match item {
                HeadItem::Duration => self.duration = Some(duration(text).ok_or_else(malformed)?),
                HeadItem::Pitch => {
                    self.core = Some(Core::Pitch(pitch(text).ok_or_else(malformed)?))
                }
                HeadItem::Rest => self.core = Some(Core::Rest { position: None }),
                HeadItem::RestPosition => {
                    self.core = Some(Core::Rest {
                        position: Some(pitch(text).ok_or_else(malformed)?),
                    })
                }
                HeadItem::Accidental(sign) => {
                    self.accidental = Some(match (sign, text.len()) {
                        (Accidental::Sharp, 2) => Accidental::DoubleSharp,
                        (Accidental::Flat, 2) => Accidental::DoubleFlat,
                        (sign, _) => sign.clone(),
                    })
                }
                HeadItem::Grace(kind) => self.grace = Some((kind.clone(), start)),
                HeadItem::ForcedAccidental => self.head.forced_accidental = Some(ForcedAccidental),
                HeadItem::InvisibleAccidental => self.head.invisible_accidental = true,
                HeadItem::PositionDisplay => {}
                HeadItem::Editorial => {
                    self.head.editorial_interpretation = Some(EditorialInterpretation)
                }
                HeadItem::Invisible => self.head.invisible = true,
                HeadItem::CrossStaff(placement) => self.head.cross_staff = Some(*placement),
                HeadItem::Tie(tie) => self.head.ties.push(*tie),
                HeadItem::UserSignifier => self.head.user_signifiers.push(UserSignifier {
                    marker: text.to_owned(),
                }),
            },
            Item::Token(item) => {
                self.first_token_mark.get_or_insert(start);
                let token = &mut self.token;
                match item {
                    TokenItem::Slur { open, linked } => {
                        let depth = elision_depth(text);
                        let slur = if *open {
                            Slur::Open {
                                elided: depth > 0,
                                depth,
                                count: 1,
                                placement: None,
                                linked: *linked,
                            }
                        } else {
                            Slur::Close {
                                elided: depth > 0,
                                depth,
                                count: 1,
                                linked: *linked,
                            }
                        };
                        let at = push_slur(&mut token.slurs, slur).ok_or_else(malformed)?;
                        if *open {
                            self.open_slur = at;
                        }
                    }
                    TokenItem::Phrase { open } => {
                        let elision = Elision {
                            depth: elision_depth(text),
                        };
                        token.phrases.push(if *open {
                            Phrase::Open { elision }
                        } else {
                            Phrase::Close { elision }
                        });
                    }
                    TokenItem::Arpeggio => {
                        let arpeggio = if text.len() == 1 {
                            Arpeggio::StaffLocal
                        } else {
                            Arpeggio::CrossStaff
                        };
                        token.arpeggio = merge_arpeggios(token.arpeggio, Some(arpeggio));
                    }
                    TokenItem::Fermata => token.fermata = Some(Fermata::default()),
                    TokenItem::Breath => token.breath = Some(Breath.into()),
                    TokenItem::Harmonic => token.harmonic = Some(Harmonic.into()),
                    TokenItem::Articulation(articulation) => {
                        token.articulations.push(articulation.clone().into())
                    }
                    TokenItem::Bowing(bowing) => token.bowing = Some(bowing.clone().into()),
                    TokenItem::Stem(stem) => token.stem_direction = Some(stem.clone()),
                    TokenItem::Beam(mark) => {
                        self.beams.add(*mark, text.len()).ok_or_else(malformed)?
                    }
                    TokenItem::Turn { form, timing } => {
                        let turn = turn(*form, *timing, text)
                            .map_err(|lone| Fault::at(start + lone, Reason::Shape(id)))?;
                        token.ornaments.push(Ornament::Turn(turn).into());
                    }
                    TokenItem::Trill(interval) => token.ornaments.push(
                        Ornament::Trill {
                            interval: *interval,
                            line: match text.len() {
                                1 => TrillLine::None,
                                2 => TrillLine::Start,
                                _ => TrillLine::Continuation,
                            },
                        }
                        .into(),
                    ),
                    TokenItem::Mordent(direction) => {
                        let mordent = mordent(*direction, text.as_bytes())
                            .ok_or_else(|| Fault::at(start, Reason::Shape(id)))?;
                        token.ornaments.push(mordent.into());
                    }
                    TokenItem::Ornament(ornament) => token.ornaments.push(ornament.clone().into()),
                }
            }
            Item::Placement(dir) => {
                if let Some(prev) = prev {
                    self.place(prev, *dir);
                }
            }
        }
        Ok(())
    }

    fn place(&mut self, prev: Admitted, dir: Dir) {
        let token = &mut self.token;
        match prev.row.item {
            Item::Token(TokenItem::Slur { open: true, .. }) => {
                set_slur_placement(&mut token.slurs, self.open_slur, dir.into())
            }
            Item::Token(TokenItem::Fermata) => {
                if let Some(fermata) = &mut token.fermata {
                    fermata.placement = Some(dir.into());
                }
            }
            Item::Token(TokenItem::Articulation(_)) => {
                place_mark(token.articulations.last_mut(), dir)
            }
            Item::Token(TokenItem::Bowing(_)) => place_mark(token.bowing.as_mut(), dir),
            Item::Token(TokenItem::Breath) => place_mark(token.breath.as_mut(), dir),
            Item::Token(TokenItem::Harmonic) => place_mark(token.harmonic.as_mut(), dir),
            Item::Token(
                TokenItem::Turn { .. }
                | TokenItem::Trill(_)
                | TokenItem::Mordent(_)
                | TokenItem::Ornament(_),
            ) => place_mark(token.ornaments.last_mut(), dir),
            Item::Head(HeadItem::Tie(_)) => {
                if let Some(tie) = self.head.ties.last_mut() {
                    *tie = tie.with_placement(dir.into());
                }
            }
            _ => unreachable!(
                "table test: every Field row has a placement arm ({})",
                prev.row.name
            ),
        }
    }

    pub(crate) fn finish(self, part: Part) -> Result<Scanned, Fault> {
        let mark = self.first_token_mark.unwrap_or(0);
        let core = match self.event {
            EventKind::Note | EventKind::Grace => {
                let core = self
                    .core
                    .ok_or_else(|| Fault::at(0, Reason::Missing(Missing::Pitch)))?;
                if self.duration.is_none() && self.grace.is_none() && part == Part::First {
                    return Err(Fault::at(mark, Reason::Missing(Missing::Duration)));
                }
                core
            }
            EventKind::Rest => match (self.core, &self.duration) {
                (Some(core), Some(_)) => core,
                _ => return Err(Fault::at(0, Reason::Missing(Missing::Rest))),
            },
        };
        let mut token = self.token;
        token.beams = self
            .beams
            .into_chord_beams()
            .ok_or_else(|| Fault::at(mark, Reason::Missing(Missing::Beams)))?;
        Ok(Scanned {
            duration: self.duration,
            core,
            accidental: self.accidental,
            grace: self.grace,
            head: self.head,
            token,
            first_token_mark: self.first_token_mark,
        })
    }
}

pub(crate) fn scan<S: AtomSink>(
    raw: &str,
    event: EventKind,
    part: Part,
    sink: &mut S,
) -> Result<Scanned, Fault> {
    let mut lexer = Lexer::new(raw);
    let mut rules = Rules::new(event);
    let mut fold = Fold::new(event);
    while let Some(lexed) = lexer.next()? {
        let (atom, prev, warning) = rules.admit(&lexed)?;
        match warning {
            None => fold.apply(raw, atom, lexed.range.clone(), prev)?,
            Some(warning) => sink.warn(lexed.range.clone(), warning),
        }
        sink.atom(lexed.range, atom.id, atom.row);
    }
    fold.finish(part)
}

fn elision_depth(text: &str) -> u8 {
    text.bytes().take_while(|byte| *byte == b'&').count() as u8
}

/// A duration spelled on its own, such as the `12.` or `16%3` of
/// `!LO:N:vis=12.`.
pub(crate) fn duration_spelling(text: &str) -> Option<Duration> {
    let number = text.trim_end_matches('.');
    if number.is_empty()
        || !number
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'%')
    {
        return None;
    }
    duration(text)
}

fn duration(text: &str) -> Option<Duration> {
    let digits = text.bytes().take_while(u8::is_ascii_digit).count();
    let (value, dots) = match text[digits..].strip_prefix('%') {
        Some(denominator) => {
            let width = denominator.bytes().take_while(u8::is_ascii_digit).count();
            let value = DurationValue::Rational {
                numerator: positive(&text[..digits])?,
                denominator: positive(&denominator[..width])?,
            };
            (value, &denominator[width..])
        }
        None => {
            let value = match &text[..digits] {
                "0" => DurationValue::Breve,
                "00" => DurationValue::Long,
                "000" => DurationValue::Maxima,
                spelling => DurationValue::Reciprocal(positive(spelling)?),
            };
            (value, &text[digits..])
        }
    };
    Some(Duration {
        value,
        augmentation_dots: u8::try_from(dots.len()).ok()?,
    })
}

pub(super) fn positive(spelling: &str) -> Option<u32> {
    spelling.parse::<u32>().ok().filter(|value| *value > 0)
}

fn pitch(text: &str) -> Option<Pitch> {
    let first = text.as_bytes()[0];
    let repetitions = i16::try_from(text.len()).ok()?;
    let step = match first.to_ascii_uppercase() {
        b'A' => DiatonicStep::A,
        b'B' => DiatonicStep::B,
        b'C' => DiatonicStep::C,
        b'D' => DiatonicStep::D,
        b'E' => DiatonicStep::E,
        b'F' => DiatonicStep::F,
        b'G' => DiatonicStep::G,
        _ => return None,
    };
    let octave = if first.is_ascii_lowercase() {
        repetitions + 3
    } else {
        4 - repetitions
    };
    let pitch = Pitch {
        step,
        octave: octave.try_into().ok()?,
    };
    pitch.is_supported_kern_pitch().then_some(pitch)
}

fn turn(form: TurnForm, timing: TurnTiming, text: &str) -> Result<Turn, usize> {
    let base = usize::from(timing == TurnTiming::OnNote);
    let intervals = match text.as_bytes()[base + 1..] {
        [] => None,
        [first, second] => {
            let (first, second) = (aux_interval(first), aux_interval(second));
            Some(match form {
                TurnForm::Regular => TurnIntervals {
                    upper: first,
                    lower: second,
                },
                TurnForm::Inverted => TurnIntervals {
                    upper: second,
                    lower: first,
                },
            })
        }
        _ => return Err(base + 1),
    };
    Ok(Turn {
        form,
        timing,
        intervals,
    })
}

fn aux_interval(letter: u8) -> AuxInterval {
    if letter.is_ascii_uppercase() {
        AuxInterval::WholeTone
    } else {
        AuxInterval::Semitone
    }
}

fn mordent(direction: MordentDirection, run: &[u8]) -> Option<Ornament> {
    let (interval, doubled) = match run {
        [single] => (Some(aux_interval(*single)), false),
        [first, second] if first == second => (Some(aux_interval(*first)), true),
        [_, _] => (None, false),
        _ => return None,
    };
    Some(Ornament::Mordent {
        direction,
        interval,
        doubled,
    })
}

pub(super) fn merge_arpeggios(
    target: Option<Arpeggio>,
    source: Option<Arpeggio>,
) -> Option<Arpeggio> {
    match (target, source) {
        (Some(Arpeggio::CrossStaff), _) | (_, Some(Arpeggio::CrossStaff)) => {
            Some(Arpeggio::CrossStaff)
        }
        (Some(Arpeggio::StaffLocal), _) | (_, Some(Arpeggio::StaffLocal)) => {
            Some(Arpeggio::StaffLocal)
        }
        (None, None) => None,
    }
}

fn place_mark<T>(mark: Option<&mut Placed<T>>, dir: Dir) {
    if let Some(mark) = mark {
        mark.placement = Some(dir.into());
    }
}

/// The index of the entry `slur` was pushed as or folded into.
pub(super) fn push_slur(slurs: &mut Vec<Slur>, slur: Slur) -> Option<usize> {
    let Some(at) = slurs
        .iter()
        .position(|existing| same_slur_kind(existing, &slur))
    else {
        slurs.push(slur);
        return Some(slurs.len() - 1);
    };
    match (&mut slurs[at], slur) {
        (
            Slur::Open { count, .. },
            Slur::Open {
                count: incoming, ..
            },
        )
        | (
            Slur::Close { count, .. },
            Slur::Close {
                count: incoming, ..
            },
        ) => {
            *count = count.checked_add(incoming)?;
        }
        _ => unreachable!("same_slur_kind only matches variants with identical kind"),
    }
    Some(at)
}

fn same_slur_kind(left: &Slur, right: &Slur) -> bool {
    match (left, right) {
        (
            Slur::Open {
                elided: left_elided,
                depth: left_depth,
                placement: left_placement,
                linked: left_linked,
                ..
            },
            Slur::Open {
                elided: right_elided,
                depth: right_depth,
                placement: right_placement,
                linked: right_linked,
                ..
            },
        ) => {
            left_elided == right_elided
                && left_depth == right_depth
                && left_placement == right_placement
                && left_linked == right_linked
        }
        (
            Slur::Close {
                elided: left_elided,
                depth: left_depth,
                linked: left_linked,
                ..
            },
            Slur::Close {
                elided: right_elided,
                depth: right_depth,
                linked: right_linked,
                ..
            },
        ) => {
            left_elided == right_elided && left_depth == right_depth && left_linked == right_linked
        }
        _ => false,
    }
}

fn set_slur_placement(slurs: &mut Vec<Slur>, at: usize, placement: SlurPlacement) {
    let Slur::Open {
        elided,
        depth,
        count,
        placement: existing,
        linked,
    } = &mut slurs[at]
    else {
        unreachable!("an open slur atom records the index of an open entry");
    };
    if *count > 1 {
        let split = Slur::Open {
            elided: *elided,
            depth: *depth,
            count: 1,
            placement: Some(placement),
            linked: *linked,
        };
        *count -= 1;
        slurs.push(split);
    } else {
        *existing = Some(placement);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct BeamCounts {
    start: u8,
    end: u8,
    partial_left: u8,
    partial_right: u8,
}

impl BeamCounts {
    fn add(&mut self, mark: BeamMark, count: usize) -> Option<()> {
        let target = match mark {
            BeamMark::Start => &mut self.start,
            BeamMark::End => &mut self.end,
            BeamMark::PartialLeft => &mut self.partial_left,
            BeamMark::PartialRight => &mut self.partial_right,
        };
        *target = target.checked_add(u8::try_from(count).ok()?)?;
        Some(())
    }

    pub(super) fn add_chord_beams(&mut self, beams: &ChordBeams) -> Option<()> {
        self.start = self.start.checked_add(beams.beam_start_count())?;
        self.end = self.end.checked_add(beams.beam_end_count())?;
        self.partial_left = self.partial_left.checked_add(beams.partial_left_count())?;
        self.partial_right = self
            .partial_right
            .checked_add(beams.partial_right_count())?;
        Some(())
    }

    pub(super) fn into_chord_beams(self) -> Option<ChordBeams> {
        let beams =
            ChordBeams::from_counts(self.start, self.end, self.partial_left, self.partial_right);
        if beams.is_some() {
            return beams;
        }
        let canceling = self.start.min(self.end);
        match (
            self.start - canceling,
            self.end - canceling,
            self.partial_left,
            self.partial_right,
        ) {
            (0, 0, 0, right) if right > 0 => ChordBeams::from_counts(0, 0, right, 0),
            (0, end, 0, right) if end > 0 && right > 0 => ChordBeams::from_counts(0, end, right, 0),
            (start, 0, left, 0) if start > 0 && left > 0 => {
                ChordBeams::from_counts(start, 0, 0, left)
            }
            _ => None,
        }
    }
}
