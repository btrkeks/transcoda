use crate::{types::*, ParseError};

impl Barline {
    pub(super) fn from_token(_spine: SpineIndex, raw: &str) -> Result<Self, ParseError> {
        Self::try_from(raw)
    }
}

impl TryFrom<&str> for Barline {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let double = raw.starts_with("==");
        let body = raw.trim_start_matches('=');
        let style_body = if body.starts_with(|character: char| character.is_ascii_digit()) {
            body.trim_start_matches(|character: char| character.is_ascii_alphanumeric())
        } else {
            body
        };
        let (style_body, fermata) = if let Some((before, after)) = style_body.split_once(';') {
            let rest = after.trim_start_matches(';');
            let (rest, fermata) = match after.len() - rest.len() + 1 {
                1 => match rest.as_bytes().first() {
                    Some(b'>') => (
                        &rest[1..],
                        BarlineFermata::PlacedSingle(FermataPlacement::Above),
                    ),
                    Some(b'<') => (
                        &rest[1..],
                        BarlineFermata::PlacedSingle(FermataPlacement::Below),
                    ),
                    _ => (rest, BarlineFermata::Single),
                },
                2 => (rest, BarlineFermata::Pair),
                _ => {
                    return Err(ParseError::InvalidBarline {
                        raw: raw.to_owned(),
                    })
                }
            };
            (format!("{before}{rest}"), Some(fermata))
        } else {
            (style_body.to_owned(), None)
        };
        if !(style_body == "-" || style_body.chars().all(|c| matches!(c, '|' | '!' | ':'))) {
            return Err(ParseError::InvalidBarline {
                raw: raw.to_owned(),
            });
        }
        let repeat = parse_barline_repeat(&style_body);
        let dashed = style_body == ":" && !double;
        if style_body.contains(':') && repeat == RepeatStyle::None && !dashed {
            return Err(ParseError::InvalidBarline {
                raw: raw.to_owned(),
            });
        }
        let style = if style_body == "-" {
            BarlineStyle::Invisible
        } else if dashed {
            BarlineStyle::Dashed
        } else if double {
            BarlineStyle::Double
        } else if style_body.starts_with("||") {
            BarlineStyle::Thin
        } else if repeat != RepeatStyle::None {
            BarlineStyle::Double
        } else {
            BarlineStyle::Basic
        };
        let mut appearance = SpineBarline::new(vec![style], repeat);
        appearance.fermata = fermata;
        Ok(Self::from_spine_barlines(vec![appearance]))
    }
}

fn parse_barline_repeat(body: &str) -> RepeatStyle {
    let starts = body.contains("|:") || body.contains("!:");
    let ends = body.contains(":|") || body.contains(":!");
    match (starts, ends) {
        (true, true) => RepeatStyle::StartAndEnd,
        (true, false) => RepeatStyle::Start,
        (false, true) => RepeatStyle::End,
        (false, false) => RepeatStyle::None,
    }
}
