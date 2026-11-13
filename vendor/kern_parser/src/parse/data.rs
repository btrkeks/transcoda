use crate::{
    parse::{token, ParseOptions},
    types::*,
    ParseError,
};

impl SpineContent {
    pub fn spine_type(&self) -> SpineType {
        match self {
            Self::Kern(_) => SpineType::Kern,
            Self::Dynam(_) => SpineType::Dynam,
            Self::Fing(_) => SpineType::Fing,
            Self::Text(_) => SpineType::Text,
            Self::MxmlHarmony(_) => SpineType::MxmlChords,
            Self::Opaque { spine_type, .. } => spine_type.clone(),
        }
    }

    pub(super) fn from_token_with_options(
        spine_type: SpineType,
        raw: &str,
        options: &ParseOptions,
    ) -> Result<Self, ParseError> {
        match spine_type {
            SpineType::Kern => KernContent::from_token(raw, options).map(Self::Kern),
            SpineType::Dynam => DynamContent::from_token(raw).map(Self::Dynam),
            SpineType::Fing => FingContent::from_token(raw).map(Self::Fing),
            SpineType::Text => TextContent::from_token(raw).map(Self::Text),
            SpineType::MxmlChords => MxmlHarmonyContent::from_token(raw).map(Self::MxmlHarmony),
            SpineType::Other(_) => Ok(Self::Opaque {
                spine_type,
                raw: RawToken(raw.to_owned()),
            }),
        }
    }
}

impl KernContent {
    pub(super) fn from_token(raw: &str, options: &ParseOptions) -> Result<Self, ParseError> {
        token::content(raw, options.dialect)
    }
}

impl DynamContent {
    fn from_token(raw: &str) -> Result<Self, ParseError> {
        Ok(match parse_dynam_marks(raw) {
            Some(marks) if marks.is_empty() => Self::Null,
            Some(marks) => Self::Marks(marks),
            None => Self::Raw {
                text: RawToken(raw.to_owned()),
                marks: raw_hairpin_marks(raw),
            },
        })
    }
}

enum DynamPart {
    Null,
    Placeholder,
    Silence,
    Level(DynamicLevel),
    Begin(DynamicHairpinDirection, Option<DynamicProvenance>),
    Continue(DynamicHairpinDirection, Option<DynamicProvenance>),
    End {
        direction: DynamicHairpinDirection,
        to_line_end: bool,
        provenance: Option<DynamicProvenance>,
    },
}

/// `None` when a part is outside the grammar or conflicts with an earlier
/// part of the token.
fn parse_dynam_marks(raw: &str) -> Option<DynamMarks> {
    let mut marks = DynamMarks::default();
    for piece in raw.split(' ') {
        if piece.is_empty() {
            return None;
        }
        for part in parse_dynam_piece(piece)? {
            insert_dynam_part(&mut marks, part)?;
        }
    }
    if marks.crescendo.begin.is_some() && marks.diminuendo.begin.is_some() {
        return None;
    }
    Some(marks)
}

fn is_hairpin_char(character: char) -> bool {
    matches!(character, '<' | '>' | '(' | ')' | '[' | ']')
}

/// The parts of one space-separated piece. A glued run of hairpin characters
/// splits per character, except that a run of `[` or `]` is one end; an `X` or
/// `x` right after the run is the provenance of every part in it.
fn parse_dynam_piece(piece: &str) -> Option<Vec<DynamPart>> {
    let mut parts = Vec::new();
    let mut rest = piece;
    while !rest.is_empty() {
        let hairpins = rest.find(|c| !is_hairpin_char(c)).unwrap_or(rest.len());
        if hairpins == 0 {
            let text = rest.find(is_hairpin_char).unwrap_or(rest.len());
            parts.push(parse_dynam_text(&rest[..text])?);
            rest = &rest[text..];
            continue;
        }
        let (run, after) = rest.split_at(hairpins);
        let (after, provenance) = if let Some(after) = after.strip_prefix('X') {
            (after, Some(DynamicProvenance::Explicit))
        } else if let Some(after) = after.strip_prefix('x') {
            (after, Some(DynamicProvenance::Editorial))
        } else {
            (after, None)
        };
        push_hairpin_parts(run, provenance, &mut parts);
        rest = after;
    }
    Some(parts)
}

fn push_hairpin_parts(
    run: &str,
    provenance: Option<DynamicProvenance>,
    parts: &mut Vec<DynamPart>,
) {
    let mut characters = run.chars().peekable();
    while let Some(character) = characters.next() {
        let direction = if matches!(character, '<' | '(' | '[') {
            DynamicHairpinDirection::Crescendo
        } else {
            DynamicHairpinDirection::Diminuendo
        };
        parts.push(match character {
            '<' | '>' => DynamPart::Begin(direction, provenance),
            '(' | ')' => DynamPart::Continue(direction, provenance),
            _ => {
                let mut to_line_end = false;
                while characters.next_if_eq(&character).is_some() {
                    to_line_end = true;
                }
                DynamPart::End {
                    direction,
                    to_line_end,
                    provenance,
                }
            }
        });
    }
}

/// The hairpin marks Verovio reads from a `Raw` token: its hairpin characters,
/// skipping every other part and any part that conflicts with an earlier one.
fn raw_hairpin_marks(raw: &str) -> DynamMarks {
    let mut parts = Vec::new();
    for run in raw
        .split(|c| !is_hairpin_char(c))
        .filter(|run| !run.is_empty())
    {
        push_hairpin_parts(run, None, &mut parts);
    }
    let mut marks = DynamMarks::default();
    for part in parts {
        let _ = insert_dynam_part(&mut marks, part);
    }
    marks
}

fn parse_dynam_text(text: &str) -> Option<DynamPart> {
    match text {
        _ if text.bytes().all(|byte| byte == b'.') => Some(DynamPart::Null),
        "???" => Some(DynamPart::Placeholder),
        "r" => Some(DynamPart::Silence),
        _ => {
            let (body, provenance) = parse_dynamic_provenance(text)?;
            parse_dynamic_level(body, provenance).map(DynamPart::Level)
        }
    }
}

/// Fill an empty slot; an equal value changes nothing and a different one is
/// a conflict.
fn merge_slot<T: PartialEq>(slot: &mut Option<T>, value: T) -> Option<()> {
    match slot {
        None => *slot = Some(value),
        Some(existing) if *existing == value => {}
        Some(_) => return None,
    }
    Some(())
}

/// Repeating a part changes nothing; a part that disagrees with the slot it
/// would fill is a conflict. A begin of each direction may share a token here;
/// [`parse_dynam_marks`] rejects that.
fn insert_dynam_part(marks: &mut DynamMarks, part: DynamPart) -> Option<()> {
    match part {
        DynamPart::Null => Some(()),
        DynamPart::Placeholder => {
            marks.placeholder = true;
            Some(())
        }
        DynamPart::Silence => {
            marks.silence = true;
            Some(())
        }
        DynamPart::Level(level) => merge_slot(&mut marks.level, level),
        DynamPart::Begin(direction, provenance) => {
            let slots = marks.hairpin_mut(direction);
            let single_note = slots.begin.is_some_and(|begin| begin.single_note);
            merge_slot(
                &mut slots.begin,
                HairpinBegin {
                    single_note,
                    provenance,
                },
            )
        }
        DynamPart::Continue(direction, provenance) => merge_slot(
            &mut marks.hairpin_mut(direction).continues,
            HairpinContinue { provenance },
        ),
        DynamPart::End {
            direction,
            to_line_end,
            provenance,
        } => {
            let slots = marks.hairpin_mut(direction);
            // An end after a start of its direction closes that start on its
            // own note, whatever the end's spelling.
            if let Some(begin) = &mut slots.begin {
                if let Some(provenance) = provenance {
                    merge_slot(&mut begin.provenance, provenance)?;
                }
                begin.single_note = true;
                return Some(());
            }
            let to_line_end = to_line_end || slots.end.is_some_and(|end| end.to_line_end);
            if let Some(end) = &mut slots.end {
                end.to_line_end = to_line_end;
            }
            merge_slot(
                &mut slots.end,
                HairpinEnd {
                    to_line_end,
                    provenance,
                },
            )
        }
    }
}

fn parse_dynamic_provenance(raw: &str) -> Option<(&str, Option<DynamicProvenance>)> {
    if raw.is_empty() {
        return None;
    }

    if let Some(body) = raw.strip_suffix('X') {
        Some((body, Some(DynamicProvenance::Explicit)))
    } else if let Some(body) = raw.strip_suffix('x') {
        Some((body, Some(DynamicProvenance::Editorial)))
    } else {
        Some((raw, None))
    }
}

fn parse_dynamic_level(body: &str, provenance: Option<DynamicProvenance>) -> Option<DynamicLevel> {
    let (body, subito) = body
        .strip_prefix('s')
        .map(|body| (body, true))
        .unwrap_or((body, false));
    let (body, accent) = body
        .strip_suffix('v')
        .map(|body| (body, true))
        .unwrap_or((body, false));
    let (body, sforzando) = body
        .strip_suffix('z')
        .map(|body| (body, true))
        .unwrap_or((body, false));

    let value = match body {
        "mp" => DynamicLevelValue::MezzoPiano,
        "mf" => DynamicLevelValue::MezzoForte,
        "fp" => DynamicLevelValue::FortePiano,
        "rf" => DynamicLevelValue::Rinforzando,
        body if body.chars().all(|character| character == 'p') && !body.is_empty() => {
            DynamicLevelValue::Piano {
                count: body.len().try_into().ok()?,
            }
        }
        body if body.chars().all(|character| character == 'f') && !body.is_empty() => {
            DynamicLevelValue::Forte {
                count: body.len().try_into().ok()?,
            }
        }
        _ => return None,
    };

    Some(DynamicLevel {
        value,
        subito,
        sforzando,
        accent,
        provenance,
    })
}

impl TextContent {
    fn from_token(raw: &str) -> Result<Self, ParseError> {
        match raw {
            "." => Ok(Self::Null),
            "%" => Ok(Self::Silence),
            "|" => Ok(Self::Melisma {
                printed_dash: false,
            }),
            "||" => Ok(Self::Melisma { printed_dash: true }),
            _ => Ok(Self::Text(TextUnderlay {
                text: raw.to_owned(),
            })),
        }
    }
}

impl FingContent {
    fn from_token(raw: &str) -> Result<Self, ParseError> {
        if raw == "." {
            return Ok(Self::Null);
        }

        let slots = raw
            .split(' ')
            .map(|slot| match slot {
                "." => Ok(None),
                _ => Fingering::from_token(slot).map(Some),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if slots.iter().all(Option::is_none) {
            Err(ParseError::InvalidFingering {
                raw: raw.to_owned(),
            })
        } else {
            Ok(Self::Fingerings(slots))
        }
    }
}

impl Fingering {
    fn from_token(raw: &str) -> Result<Self, ParseError> {
        let invalid = || ParseError::InvalidFingering {
            raw: raw.to_owned(),
        };
        if let Some((from, to)) = raw.split_once('-') {
            let (Some(from), Some(to)) =
                (finger_number_from_token(from), finger_number_from_token(to))
            else {
                return Err(invalid());
            };
            return Ok(Self::substitution(from, to));
        }

        raw.split('/')
            .map(finger_number_from_token)
            .collect::<Option<Vec<_>>>()
            .and_then(|digits| Self::from_digits(&digits))
            .ok_or_else(invalid)
    }
}

fn finger_number_from_token(raw: &str) -> Option<FingerNumber> {
    let [digit] = raw.as_bytes() else {
        return None;
    };
    if !(b'0'..=b'5').contains(digit) {
        return None;
    }
    FingerNumber::try_from(digit - b'0').ok()
}

impl MxmlHarmonyContent {
    fn from_token(raw: &str) -> Result<Self, ParseError> {
        if raw == "." {
            Ok(Self::Null)
        } else {
            Ok(Self::Label(MxmlHarmonyLabel {
                text: raw.to_owned(),
            }))
        }
    }
}
