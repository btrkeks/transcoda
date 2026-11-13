use crate::{parse::spine_state::SpineState, types::*, ParseError};

pub(super) fn parse_tandem_token(
    spine: SpineIndex,
    raw: &str,
    spine_state: &SpineState,
) -> Result<TandemInterpretation, ParseError> {
    match TandemInterpretation::try_from(raw) {
        // `*kcancel` was opaque before it was modelled and stays opaque in
        // opaque spines.
        Ok(TandemInterpretation::VisualStyle(VisualStyle::KeyCancellation(_)))
            if matches!(
                spine_state.spine_type(spine)?,
                SpineType::Fing | SpineType::Other(_)
            ) =>
        {
            Ok(TandemInterpretation::Raw(RawToken(raw.to_owned())))
        }
        Ok(
            TandemInterpretation::Cue(_)
            | TandemInterpretation::RhythmScale(_)
            | TandemInterpretation::VisualStyle(VisualStyle::KeyCancellation(_)),
        ) if spine_state.spine_type(spine)? != SpineType::Kern => {
            Err(ParseError::KernOnlyTandemInterpretation {
                raw: raw.to_owned(),
            })
        }
        Ok(tandem) => Ok(tandem),
        Err(_)
            if matches!(
                spine_state.spine_type(spine)?,
                SpineType::Fing | SpineType::Other(_)
            ) =>
        {
            Ok(TandemInterpretation::Raw(RawToken(raw.to_owned())))
        }
        Err(error) => Err(error),
    }
}

impl TryFrom<&str> for TandemInterpretation {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        if raw == "*" {
            return Ok(Self::Null);
        }

        if raw.starts_with("*clef") {
            return Clef::try_from(raw).map(Self::Clef);
        }

        if raw.starts_with("*k[") {
            return KeySignature::try_from(raw).map(Self::KeySignature);
        }

        if raw == "*MX" {
            return Ok(Self::FreeMeter);
        }

        if is_meter_token(raw) {
            return TimeSignatureTandem::try_from(raw).map(Self::TimeSignature);
        }

        if raw.starts_with("*met") {
            return MeterSymbol::try_from(raw).map(Self::MeterSymbol);
        }

        if raw.starts_with("*stria") {
            return Stria::try_from(raw).map(Self::Stria);
        }

        if raw.starts_with("*part") {
            return Part::try_from(raw).map(Self::Part);
        }

        if raw.starts_with("*staff") {
            return Staff::try_from(raw).map(Self::Staff);
        }

        if raw.starts_with("*group") {
            return Group::try_from(raw).map(Self::Group);
        }

        if raw.starts_with("*MM") {
            return Tempo::try_from(raw).map(Self::Tempo);
        }

        if raw.starts_with("*I") || raw.starts_with("*mI") || raw.starts_with("*oI") {
            return Instrument::try_from(raw).map(Self::Instrument);
        }

        if raw == "*tremolo" || raw == "*Xtremolo" {
            return Tremolo::try_from(raw).map(Self::Tremolo);
        }

        if raw == "*ped" || raw == "*Xped" {
            return Pedal::try_from(raw).map(Self::Pedal);
        }

        match raw {
            "*rep" => return Ok(Self::Repetition(Repetition { active: true })),
            "*Xrep" => return Ok(Self::Repetition(Repetition { active: false })),
            "*tupbreak" => return Ok(Self::TupletBreak),
            _ => {}
        }

        match raw {
            "*cue" => return Ok(Self::Cue(Cue { active: true })),
            "*Xcue" => return Ok(Self::Cue(Cue { active: false })),
            _ => {}
        }

        if let Some(payload) = raw.strip_prefix("*rscale:") {
            return parse_rhythm_scale(payload)
                .map(Self::RhythmScale)
                .ok_or_else(|| ParseError::InvalidRhythmScale {
                    raw: raw.to_owned(),
                });
        }

        if raw == "*tuplet" || raw == "*Xtuplet" {
            return Tuplet::try_from(raw).map(Self::Tuplet);
        }

        match raw {
            "*brackettup" => return Ok(Self::TupletBracket(TupletBracket { show_bracket: true })),
            "*Xbrackettup" => {
                return Ok(Self::TupletBracket(TupletBracket {
                    show_bracket: false,
                }))
            }
            _ => {}
        }

        if raw.starts_with("*color:") {
            return VisualStyle::try_from(raw).map(Self::VisualStyle);
        }

        if raw.starts_with("*lig")
            || raw.starts_with("*Xlig")
            || raw.starts_with("*col")
            || raw.starts_with("*Xcol")
        {
            return BracketSpan::try_from(raw).map(Self::BracketSpan);
        }

        if raw.starts_with("*>") {
            return if raw.contains('[') || raw.contains(']') {
                ExpansionList::try_from(raw).map(Self::ExpansionList)
            } else {
                SectionLabel::try_from(raw).map(Self::SectionLabel)
            };
        }

        if matches!(
            raw,
            "*2\\right" | "*2\\left" | "*above" | "*below" | "*kcancel" | "*Xkcancel"
        ) {
            return VisualStyle::try_from(raw).map(Self::VisualStyle);
        }

        if is_ottava_token(raw) {
            return Ottava::try_from(raw).map(Self::Ottava);
        }

        if is_key_designation_token(raw) {
            return KeyDesignation::try_from(raw).map(Self::KeyDesignation);
        }

        Err(ParseError::UnsupportedTandemInterpretation {
            raw: raw.to_owned(),
        })
    }
}

impl TryFrom<&str> for VisualStyle {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        if let Some(hex) = raw.strip_prefix("*color:#") {
            if hex.len() == 6 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                let rgb = [0, 2, 4].map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap());
                return Ok(Self::Color(rgb));
            }
        }
        match raw {
            "*above" => Ok(Self::StaffPlacement(StaffPlacement::Above)),
            "*below" => Ok(Self::StaffPlacement(StaffPlacement::Below)),
            "*kcancel" => Ok(Self::KeyCancellation(KeyCancellation { active: true })),
            "*Xkcancel" => Ok(Self::KeyCancellation(KeyCancellation { active: false })),
            "*2\\left" => Ok(Self::HalfNoteStemSide(HalfNoteStemSide::Left)),
            "*2\\right" => Ok(Self::HalfNoteStemSide(HalfNoteStemSide::Right)),
            _ => Err(ParseError::UnsupportedTandemInterpretation {
                raw: raw.to_owned(),
            }),
        }
    }
}

fn is_meter_token(raw: &str) -> bool {
    raw.strip_prefix("*M")
        .and_then(|body| body.chars().next())
        .is_some_and(|character| character.is_ascii_digit())
}

fn is_ottava_token(raw: &str) -> bool {
    let body = raw.strip_prefix("*X").or_else(|| raw.strip_prefix('*'));

    matches!(body, Some("8va" | "8ba" | "15ma" | "15ba"))
}

fn is_key_designation_token(raw: &str) -> bool {
    raw.starts_with('*') && raw.ends_with(':')
}

impl TryFrom<&str> for Clef {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let body = raw
            .strip_prefix("*clef")
            .ok_or_else(|| ParseError::InvalidClef {
                raw: raw.to_owned(),
            })?;
        let (body, invisible) = match body.strip_suffix("yy") {
            Some(body) => (body, true),
            None => (body, false),
        };

        let (sign_and_shift, line) = split_clef_body_line(raw, body)?;

        if let Some(rest) = sign_and_shift.strip_prefix("TAB") {
            if rest.is_empty() {
                return Ok(Self {
                    sign: ClefSign::Tab,
                    line,
                    octave_shift: 0,
                    invisible,
                });
            }

            return Err(ParseError::InvalidClef {
                raw: raw.to_owned(),
            });
        }

        if let Some(rest) = sign_and_shift.strip_prefix('X') {
            if rest.is_empty() {
                return Ok(Self {
                    sign: ClefSign::Percussion,
                    line,
                    octave_shift: 0,
                    invisible,
                });
            }

            return Err(ParseError::InvalidClef {
                raw: raw.to_owned(),
            });
        }

        if line == 0 {
            return Err(ParseError::InvalidClef {
                raw: raw.to_owned(),
            });
        }

        let (sign, shift_spelling) =
            sign_and_shift
                .split_at_checked(1)
                .ok_or_else(|| ParseError::InvalidClef {
                    raw: raw.to_owned(),
                })?;

        let sign = match sign {
            "G" => ClefSign::G,
            "F" => ClefSign::F,
            "C" => ClefSign::C,
            _ => {
                return Err(ParseError::InvalidClef {
                    raw: raw.to_owned(),
                });
            }
        };

        let octave_shift = parse_clef_octave_shift(raw, shift_spelling)?;

        Ok(Self {
            sign,
            line,
            octave_shift,
            invisible,
        })
    }
}

fn split_clef_body_line<'a>(raw: &str, body: &'a str) -> Result<(&'a str, u8), ParseError> {
    let Some((staff_line_index, staff_line_char)) = body.char_indices().next_back() else {
        return Err(ParseError::InvalidClef {
            raw: raw.to_owned(),
        });
    };

    if !staff_line_char.is_ascii_digit() {
        return Ok((body, 0));
    }

    let line = staff_line_char
        .to_digit(10)
        .and_then(|line| u8::try_from(line).ok())
        .filter(|line| *line > 0)
        .ok_or_else(|| ParseError::InvalidClef {
            raw: raw.to_owned(),
        })?;

    Ok((&body[..staff_line_index], line))
}

fn parse_clef_octave_shift(raw: &str, spelling: &str) -> Result<i8, ParseError> {
    if spelling.is_empty() {
        return Ok(0);
    }

    if spelling.chars().all(|marker| marker == 'v') {
        return Ok(-(spelling.chars().count() as i8));
    }

    if spelling.chars().all(|marker| marker == '^') {
        return Ok(spelling.chars().count() as i8);
    }

    Err(ParseError::InvalidClef {
        raw: raw.to_owned(),
    })
}

impl TryFrom<&str> for KeyDesignation {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let body = raw
            .strip_prefix('*')
            .and_then(|body| body.strip_suffix(':'))
            .ok_or_else(|| ParseError::InvalidKeyDesignation {
                raw: raw.to_owned(),
            })?;

        let mut chars = body.chars();
        let pitch = chars
            .next()
            .ok_or_else(|| ParseError::InvalidKeyDesignation {
                raw: raw.to_owned(),
            })?;

        let step = match pitch.to_ascii_uppercase() {
            'A' => DiatonicStep::A,
            'B' => DiatonicStep::B,
            'C' => DiatonicStep::C,
            'D' => DiatonicStep::D,
            'E' => DiatonicStep::E,
            'F' => DiatonicStep::F,
            'G' => DiatonicStep::G,
            _ => {
                return Err(ParseError::InvalidKeyDesignation {
                    raw: raw.to_owned(),
                });
            }
        };

        let mode = if pitch.is_ascii_uppercase() {
            KeyMode::Major
        } else if pitch.is_ascii_lowercase() {
            KeyMode::Minor
        } else {
            return Err(ParseError::InvalidKeyDesignation {
                raw: raw.to_owned(),
            });
        };

        let accidental = match chars.next() {
            Some('#') => Some(Accidental::Sharp),
            Some('-') => Some(Accidental::Flat),
            Some(_) => {
                return Err(ParseError::InvalidKeyDesignation {
                    raw: raw.to_owned(),
                });
            }
            None => None,
        };

        if chars.next().is_some() {
            return Err(ParseError::InvalidKeyDesignation {
                raw: raw.to_owned(),
            });
        }

        Ok(Self {
            step,
            accidental,
            mode,
        })
    }
}

impl TryFrom<&str> for KeySignature {
    type Error = ParseError;

    /// Accepts the key's accidentals and any cancellation naturals in any
    /// order. The accidentals must form one circle-of-fifths key and no step
    /// may appear twice.
    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        parse_key_signature(raw).ok_or_else(|| ParseError::InvalidKeySignature {
            raw: raw.to_owned(),
        })
    }
}

fn parse_key_signature(raw: &str) -> Option<KeySignature> {
    let body = raw.strip_prefix("*k[")?.strip_suffix(']')?;
    let mut chars = body.chars();
    let mut accidentals = [None; 7];
    let mut naturals = Vec::new();

    while let Some(letter) = chars.next() {
        let step = DiatonicStep::from_lowercase(letter)?;
        let accidental = chars.next()?;
        if !matches!(accidental, 'n' | '#' | '-') {
            return None;
        }
        if accidentals[letter as usize - 'a' as usize]
            .replace(accidental)
            .is_some()
        {
            return None;
        }
        if accidental == 'n' {
            naturals.push(step);
        }
    }

    let order = if accidentals.contains(&Some('#')) {
        "fcgdaeb"
    } else {
        "beadgcf"
    };
    let mut canonical = String::from("*k[");
    for letter in order.chars() {
        if let Some(accidental @ ('#' | '-')) = accidentals[letter as usize - 'a' as usize] {
            canonical.push(letter);
            canonical.push(accidental);
        }
    }
    canonical.push(']');

    let fifths = Fifths::new(canonical_key_signature_fifths(&canonical)?)
        .expect("matched key signature fifths are valid");
    KeySignature::from_fifths(fifths).with_naturals(&naturals)
}

fn canonical_key_signature_fifths(raw: &str) -> Option<i8> {
    Some(match raw {
        "*k[b-e-a-d-g-c-f-]" => -7,
        "*k[b-e-a-d-g-c-]" => -6,
        "*k[b-e-a-d-g-]" => -5,
        "*k[b-e-a-d-]" => -4,
        "*k[b-e-a-]" => -3,
        "*k[b-e-]" => -2,
        "*k[b-]" => -1,
        "*k[]" => 0,
        "*k[f#]" => 1,
        "*k[f#c#]" => 2,
        "*k[f#c#g#]" => 3,
        "*k[f#c#g#d#]" => 4,
        "*k[f#c#g#d#a#]" => 5,
        "*k[f#c#g#d#a#e#]" => 6,
        "*k[f#c#g#d#a#e#b#]" => 7,
        _ => return None,
    })
}

impl TryFrom<&str> for MeterSymbol {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        match raw {
            "*met(c)" => Ok(Self::CommonTime),
            "*met(c|)" => Ok(Self::CutTime),
            "*met(C)" => Ok(Self::Mensural(MensurationSign::TempusImperfectum)),
            "*met(C|)" => Ok(Self::Mensural(MensurationSign::TempusImperfectumDiminutum)),
            "*met()" => Ok(Self::Empty),
            _ => Err(ParseError::InvalidMeterSymbol {
                raw: raw.to_owned(),
            }),
        }
    }
}

impl TryFrom<&str> for Stria {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let body = raw
            .strip_prefix("*stria")
            .filter(|body| !body.is_empty())
            .ok_or_else(|| ParseError::InvalidStria {
                raw: raw.to_owned(),
            })?;

        let count = body.parse::<u8>().map_err(|_| ParseError::InvalidStria {
            raw: raw.to_owned(),
        })?;

        Ok(Self { count })
    }
}

impl TryFrom<&str> for Part {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let body = raw
            .strip_prefix("*part")
            .filter(|body| {
                !body.is_empty() && body.chars().all(|character| character.is_ascii_digit())
            })
            .ok_or_else(|| ParseError::InvalidPart {
                raw: raw.to_owned(),
            })?;

        let number = body
            .parse::<u32>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| ParseError::InvalidPart {
                raw: raw.to_owned(),
            })?;

        Ok(Self { number })
    }
}

impl TryFrom<&str> for Staff {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let body = raw
            .strip_prefix("*staff")
            .filter(|body| !body.is_empty())
            .ok_or_else(|| ParseError::InvalidStaff {
                raw: raw.to_owned(),
            })?;

        let mut numbers = body.split('/');
        let number = parse_staff_number(raw, numbers.next())?;
        let also = numbers
            .map(|part| parse_staff_number(raw, Some(part)))
            .collect::<Result<_, _>>()?;

        Ok(Self { number, also })
    }
}

impl TryFrom<&str> for Group {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let body = raw
            .strip_prefix("*group")
            .filter(|body| !body.is_empty())
            .ok_or_else(|| ParseError::InvalidGroup {
                raw: raw.to_owned(),
            })?;

        let mut numbers = body.split('/');
        let number = parse_group_number(raw, numbers.next())?;
        for part in numbers {
            parse_group_number(raw, Some(part))?;
        }

        Ok(Self { number })
    }
}

fn parse_group_number(raw: &str, part: Option<&str>) -> Result<u32, ParseError> {
    part.filter(|value| {
        !value.is_empty() && value.chars().all(|character| character.is_ascii_digit())
    })
    .and_then(|value| value.parse::<u32>().ok())
    .filter(|value| *value > 0)
    .ok_or_else(|| ParseError::InvalidGroup {
        raw: raw.to_owned(),
    })
}

fn parse_staff_number(raw: &str, part: Option<&str>) -> Result<u32, ParseError> {
    part.filter(|value| {
        !value.is_empty() && value.chars().all(|character| character.is_ascii_digit())
    })
    .and_then(|value| value.parse::<u32>().ok())
    .filter(|value| *value > 0)
    .ok_or_else(|| ParseError::InvalidStaff {
        raw: raw.to_owned(),
    })
}

impl TryFrom<&str> for Tempo {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let body = raw
            .strip_prefix("*MM")
            .filter(|body| !body.is_empty())
            .ok_or_else(|| ParseError::InvalidTempo {
                raw: raw.to_owned(),
            })?;

        let (integer, fraction) = body
            .split_once('.')
            .map_or((body, None), |(integer, fraction)| {
                (integer, Some(fraction))
            });

        if integer.is_empty()
            || !integer.chars().all(|character| character.is_ascii_digit())
            || fraction.is_some_and(|fraction| {
                fraction.is_empty() || !fraction.chars().all(|character| character.is_ascii_digit())
            })
        {
            return Err(ParseError::InvalidTempo {
                raw: raw.to_owned(),
            });
        }

        let scale =
            fraction
                .map_or(0, str::len)
                .try_into()
                .map_err(|_| ParseError::InvalidTempo {
                    raw: raw.to_owned(),
                })?;
        let units = integer
            .chars()
            .chain(fraction.unwrap_or("").chars())
            .try_fold(0_u32, |value, character| {
                value.checked_mul(10)?.checked_add(character.to_digit(10)?)
            })
            .filter(|value| *value > 0)
            .ok_or_else(|| ParseError::InvalidTempo {
                raw: raw.to_owned(),
            })?;

        let beats_per_minute = DecimalTempo { units, scale };

        Ok(Self { beats_per_minute })
    }
}

impl TryFrom<&str> for Instrument {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let (text, kind) = if let Some(text) = raw.strip_prefix("*mI\"") {
            (text, InstrumentKind::Name(InstrumentTextVariant::Modern))
        } else if let Some(text) = raw.strip_prefix("*oI\"") {
            (text, InstrumentKind::Name(InstrumentTextVariant::Original))
        } else if let Some(text) = raw.strip_prefix("*I\"") {
            (text, InstrumentKind::Name(InstrumentTextVariant::Regular))
        } else if let Some(text) = raw.strip_prefix("*mI'") {
            (
                text,
                InstrumentKind::Abbreviation(InstrumentTextVariant::Modern),
            )
        } else if let Some(text) = raw.strip_prefix("*oI'") {
            (
                text,
                InstrumentKind::Abbreviation(InstrumentTextVariant::Original),
            )
        } else if let Some(text) = raw.strip_prefix("*I'") {
            (
                text,
                InstrumentKind::Abbreviation(InstrumentTextVariant::Regular),
            )
        } else if let Some(text) = raw.strip_prefix("*I") {
            (text, InstrumentKind::Code)
        } else {
            return Err(ParseError::InvalidInstrument {
                raw: raw.to_owned(),
            });
        };

        if text.is_empty() && matches!(kind, InstrumentKind::Code) {
            return Err(ParseError::InvalidInstrument {
                raw: raw.to_owned(),
            });
        }

        Ok(Self {
            text: text.to_owned(),
            kind,
        })
    }
}

impl TryFrom<&str> for TimeSignatureTandem {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let (time_signature, invisible) = match raw.strip_suffix("yy") {
            Some(visible) => (visible, true),
            None => (raw, false),
        };
        Ok(Self {
            time_signature: TimeSignature::try_from(time_signature).map_err(|_| {
                ParseError::InvalidTimeSignature {
                    raw: raw.to_owned(),
                }
            })?,
            invisible,
        })
    }
}

impl TryFrom<&str> for TimeSignature {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let body = raw
            .strip_prefix("*M")
            .ok_or_else(|| ParseError::InvalidTimeSignature {
                raw: raw.to_owned(),
            })?;

        let (numerator, denominator) =
            body.split_once('/')
                .ok_or_else(|| ParseError::InvalidTimeSignature {
                    raw: raw.to_owned(),
                })?;

        let numerator = numerator
            .parse::<u32>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| ParseError::InvalidTimeSignature {
                raw: raw.to_owned(),
            })?;

        let denominator = denominator
            .parse::<u32>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| ParseError::InvalidTimeSignature {
                raw: raw.to_owned(),
            })?;

        Ok(Self {
            numerator,
            denominator,
        })
    }
}

impl TryFrom<&str> for Ottava {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let (active, body) = raw
            .strip_prefix("*X")
            .map(|body| (false, body))
            .or_else(|| raw.strip_prefix('*').map(|body| (true, body)))
            .ok_or_else(|| ParseError::InvalidOttava {
                raw: raw.to_owned(),
            })?;

        let kind = match body {
            "8va" => OttavaKind::OctaveUp,
            "8ba" => OttavaKind::OctaveDown,
            "15ma" => OttavaKind::TwoOctavesUp,
            "15ba" => OttavaKind::TwoOctavesDown,
            _ => {
                return Err(ParseError::InvalidOttava {
                    raw: raw.to_owned(),
                });
            }
        };

        Ok(Self { kind, active })
    }
}

impl TryFrom<&str> for Tremolo {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        match raw {
            "*tremolo" => Ok(Self { active: true }),
            "*Xtremolo" => Ok(Self { active: false }),
            _ => Err(ParseError::InvalidTremolo {
                raw: raw.to_owned(),
            }),
        }
    }
}

impl TryFrom<&str> for Pedal {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        match raw {
            "*ped" => Ok(Self { active: true }),
            "*Xped" => Ok(Self { active: false }),
            _ => Err(ParseError::InvalidPedal {
                raw: raw.to_owned(),
            }),
        }
    }
}

fn parse_rhythm_scale(payload: &str) -> Option<RhythmScale> {
    let (numerator, denominator) = payload.split_once('/').unwrap_or((payload, "1"));
    let digits = |text: &str| {
        (!text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit()))
            .then(|| text.parse::<u32>().ok())
            .flatten()
    };
    RhythmScale::new(digits(numerator)?, digits(denominator)?)
}

impl TryFrom<&str> for Tuplet {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        match raw {
            "*tuplet" => Ok(Self { show_number: true }),
            "*Xtuplet" => Ok(Self { show_number: false }),
            _ => Err(ParseError::InvalidTuplet {
                raw: raw.to_owned(),
            }),
        }
    }
}

impl TryFrom<&str> for BracketSpan {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        match raw {
            "*lig" => Ok(Self {
                kind: BracketSpanKind::Ligature,
                active: true,
            }),
            "*Xlig" => Ok(Self {
                kind: BracketSpanKind::Ligature,
                active: false,
            }),
            "*col" => Ok(Self {
                kind: BracketSpanKind::Coloration,
                active: true,
            }),
            "*Xcol" => Ok(Self {
                kind: BracketSpanKind::Coloration,
                active: false,
            }),
            _ => Err(ParseError::InvalidBracketSpan {
                raw: raw.to_owned(),
            }),
        }
    }
}

impl TryFrom<&str> for SectionLabel {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let label = raw
            .strip_prefix("*>")
            .filter(|label| is_valid_thru_label(label))
            .ok_or_else(|| ParseError::InvalidSectionLabel {
                raw: raw.to_owned(),
            })?;

        Ok(Self {
            label: label.to_owned(),
        })
    }
}

impl TryFrom<&str> for ExpansionList {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        let body = raw
            .strip_prefix("*>")
            .ok_or_else(|| ParseError::InvalidExpansionList {
                raw: raw.to_owned(),
            })?;
        let (version, items) =
            body.split_once('[')
                .ok_or_else(|| ParseError::InvalidExpansionList {
                    raw: raw.to_owned(),
                })?;
        let items = items
            .strip_suffix(']')
            .ok_or_else(|| ParseError::InvalidExpansionList {
                raw: raw.to_owned(),
            })?;

        if version.contains(']') {
            return Err(ParseError::InvalidExpansionList {
                raw: raw.to_owned(),
            });
        }
        let version = if version.is_empty() {
            None
        } else if is_valid_thru_label(version) {
            Some(version.to_owned())
        } else {
            return Err(ParseError::InvalidExpansionList {
                raw: raw.to_owned(),
            });
        };

        if items.contains('[') || items.contains(']') {
            return Err(ParseError::InvalidExpansionList {
                raw: raw.to_owned(),
            });
        }

        let items = if items.is_empty() {
            Vec::new()
        } else {
            items
                .split(',')
                .map(|item| {
                    let item = item.trim();
                    if item.is_empty() {
                        return Ok(ExpansionItem::Empty);
                    }
                    if is_valid_thru_label(item) {
                        Ok(ExpansionItem::Label(item.to_owned()))
                    } else {
                        Err(ParseError::InvalidExpansionList {
                            raw: raw.to_owned(),
                        })
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        Ok(Self { version, items })
    }
}

fn is_valid_thru_label(label: &str) -> bool {
    !label.trim().is_empty()
        && label.chars().all(|character| {
            character.is_ascii_alphanumeric()
                || matches!(
                    character,
                    '_' | '-' | '.' | '+' | '#' | '~' | '@' | ' ' | '>'
                )
        })
}
