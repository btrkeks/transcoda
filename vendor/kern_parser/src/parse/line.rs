use crate::{
    line::{split_spine_tokens, split_spine_tokens_preserving_empty},
    parse::{spine_state::SpineState, tandem::parse_tandem_token, ParseOptions},
    types::*,
    ParseError,
};

impl KernLine {
    /// Parse a record that produces exactly one internal line.
    ///
    /// Mixed tandem records require [`Self::parse_record`] because the internal
    /// representation keeps tandem lines homogeneous.
    pub fn parse(line_number: usize, raw: &str) -> Result<Self, ParseError> {
        match Self::parse_with_spine_types(
            line_number,
            raw,
            &SpineState::new(),
            &ParseOptions::default(),
        )? {
            ParsedRecord::Line(line) => Ok(line),
            ParsedRecord::Tandems(tokens) => {
                TandemInterpretationLine::new_on_line(line_number, tokens)
                    .map(Self::TandemInterpretationLine)
            }
        }
    }

    /// Parse one source record into one or more internal lines.
    ///
    /// Mixed tandem records expand into homogeneous lines in the order their
    /// kinds first appear. Spine positions are preserved with null `*` tokens.
    pub fn parse_record(line_number: usize, raw: &str) -> Result<Vec<Self>, ParseError> {
        Self::parse_record_with_spine_types(
            line_number,
            raw,
            &SpineState::new(),
            &ParseOptions::default(),
        )
    }

    pub(super) fn parse_record_with_spine_types(
        line_number: usize,
        raw: &str,
        spine_state: &SpineState,
        options: &ParseOptions,
    ) -> Result<Vec<Self>, ParseError> {
        Ok(
            match Self::parse_with_spine_types(line_number, raw, spine_state, options)? {
                ParsedRecord::Line(line) => vec![line],
                ParsedRecord::Tandems(tokens) => split_tandem_record(tokens),
            },
        )
    }

    fn parse_with_spine_types(
        line_number: usize,
        raw: &str,
        spine_state: &SpineState,
        options: &ParseOptions,
    ) -> Result<ParsedRecord, ParseError> {
        if raw.is_empty() {
            return Err(ParseError::EmptyKernLine { line: line_number });
        }

        let tokens = match split_spine_tokens(line_number, raw) {
            Ok(tokens) => tokens,
            Err(error @ ParseError::EmptySpineToken { .. }) => {
                return match parse_empty_data_field_line(line_number, raw, spine_state, options) {
                    Ok(line) => Ok(ParsedRecord::Line(line)),
                    Err(fallback_error) => {
                        Err(prefer_empty_data_field_error(error, fallback_error))
                    }
                };
            }
            Err(error) => return Err(error),
        };
        if tokens.iter().all(|token| token.starts_with('=')) {
            return parse_barline_line(line_number, &tokens, spine_state).map(ParsedRecord::Line);
        }

        spine_state.validate_line_width(line_number, tokens.len())?;

        let parsed = tokens
            .iter()
            .enumerate()
            .map(|(spine, token)| {
                ParsedSpineToken::parse(SpineIndex(spine), token, spine_state, options).map_err(
                    |source| ParseError::InvalidSpineToken {
                        line: line_number,
                        column: spine + 1,
                        raw: (*token).to_owned(),
                        source: Box::new(source),
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        KernLine::from_tokens(line_number, parsed)
    }

    fn from_tokens(
        line_number: usize,
        tokens: Vec<ParsedSpineToken>,
    ) -> Result<ParsedRecord, ParseError> {
        if tokens.is_empty() {
            return Err(ParseError::EmptyKernLine { line: line_number });
        }

        let resolved_kind = tokens
            .iter()
            .map(ParsedSpineToken::kind)
            .find(|kind| *kind != ParsedSpineTokenKind::NullInterpretationToken)
            .unwrap_or(ParsedSpineTokenKind::Tandem);

        for (index, token) in tokens.iter().enumerate() {
            match token.kind() {
                ParsedSpineTokenKind::NullInterpretationToken
                    if matches!(
                        resolved_kind,
                        ParsedSpineTokenKind::Tandem
                            | ParsedSpineTokenKind::SpineManipulator
                            | ParsedSpineTokenKind::NullInterpretationToken
                    ) => {}
                ParsedSpineTokenKind::NullInterpretationToken => {
                    return Err(ParseError::MixedKernLineTypes {
                        line: line_number,
                        column: index + 1,
                        expected: resolved_kind.name(),
                        found: ParsedSpineTokenKind::SpineManipulator.name(),
                    });
                }
                kind if kind == resolved_kind => {}
                kind => {
                    return Err(ParseError::MixedKernLineTypes {
                        line: line_number,
                        column: index + 1,
                        expected: resolved_kind.name(),
                        found: kind.name(),
                    });
                }
            }
        }

        let line = match resolved_kind {
            ParsedSpineTokenKind::SpineType => Ok(Self::SpineTypeLine(
                tokens
                    .into_iter()
                    .enumerate()
                    .map(|(index, token)| match token {
                        ParsedSpineToken::SpineType(spine_type) => spine_type,
                        ParsedSpineToken::NullInterpretationToken(_) => {
                            unreachable!(
                                "null interpretation token at column {} cannot appear in spine type line",
                                index + 1
                            )
                        }
                        _ => unreachable!("line kind was validated before collection"),
                    })
                    .collect(),
            )),
            ParsedSpineTokenKind::Tandem => {
                let tandems = tokens
                    .into_iter()
                    .map(|token| match token {
                        ParsedSpineToken::Tandem(tandem) => tandem,
                        ParsedSpineToken::NullInterpretationToken(_) => TandemInterpretation::Null,
                        _ => unreachable!("line kind was validated before collection"),
                    })
                    .collect();
                return Ok(ParsedRecord::Tandems(tandems));
            }
            ParsedSpineTokenKind::SpineManipulator => {
                let manipulators = tokens
                    .into_iter()
                    .map(|token| match token {
                        ParsedSpineToken::SpineManipulator(manipulator) => manipulator,
                        ParsedSpineToken::NullInterpretationToken(_) => SpineManipulator::Null,
                        _ => unreachable!("line kind was validated before collection"),
                    })
                    .collect();
                SpineManipulatorLine::new_on_line(line_number, manipulators)
                    .map(Self::SpineManipulatorLine)
            }
            ParsedSpineTokenKind::LocalComment => Ok(Self::LocalCommentLine(
                tokens
                    .into_iter()
                    .enumerate()
                    .map(|(index, token)| match token {
                        ParsedSpineToken::LocalComment(comment) => Ok(comment),
                        ParsedSpineToken::NullInterpretationToken(_) => {
                            unreachable!(
                                "null interpretation token at column {} cannot appear in local comment line",
                                index + 1
                            )
                        }
                        _ => unreachable!("line kind was validated before collection"),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            ParsedSpineTokenKind::Barline => {
                unreachable!("barline lines are parsed before generic spine token folding")
            }
            ParsedSpineTokenKind::Data => Ok(Self::Data(
                tokens
                    .into_iter()
                    .enumerate()
                    .map(|(index, token)| match token {
                        ParsedSpineToken::Data(data) => data,
                        ParsedSpineToken::NullInterpretationToken(_) => {
                            unreachable!(
                                "null interpretation token at column {} cannot appear in data line",
                                index + 1
                            )
                        }
                        _ => unreachable!("line kind was validated before collection"),
                    })
                    .collect(),
            )),
            ParsedSpineTokenKind::NullInterpretationToken => {
                unreachable!("null-only kern lines are resolved as tandem interpretation lines")
            }
        }?;
        Ok(ParsedRecord::Line(line))
    }
}

enum ParsedRecord {
    Line(KernLine),
    Tandems(Vec<TandemInterpretation>),
}

fn split_tandem_record(tokens: Vec<TandemInterpretation>) -> Vec<KernLine> {
    let width = tokens.len();
    let source_non_null = tokens
        .iter()
        .map(|token| !matches!(token, TandemInterpretation::Null))
        .collect::<Vec<_>>();
    let mut groups: Vec<(TandemInterpretationLineKind, Vec<TandemInterpretation>)> = Vec::new();
    for (column, token) in tokens.into_iter().enumerate() {
        let kind = TandemInterpretationLineKind::from_tandem(&token);
        if kind == TandemInterpretationLineKind::Null {
            continue;
        }

        if let Some((_, row)) = groups
            .iter_mut()
            .find(|(group_kind, _)| *group_kind == kind)
        {
            row[column] = token;
        } else {
            let mut row = vec![TandemInterpretation::Null; width];
            row[column] = token;
            groups.push((kind, row));
        }
    }
    if groups.is_empty() {
        groups.push((
            TandemInterpretationLineKind::Null,
            vec![TandemInterpretation::Null; width],
        ));
    }

    groups
        .into_iter()
        .map(|(_, row)| {
            let generated = row
                .iter()
                .zip(&source_non_null)
                .map(|(token, non_null)| *non_null && matches!(token, TandemInterpretation::Null))
                .collect();
            KernLine::TandemInterpretationLine(
                TandemInterpretationLine::from_homogeneous_tokens_with_generated(row, generated),
            )
        })
        .collect()
}

fn prefer_empty_data_field_error(original: ParseError, fallback: ParseError) -> ParseError {
    let Some(original_column) = parse_error_column(&original) else {
        return original;
    };
    let Some(fallback_column) = parse_error_column(&fallback) else {
        return original;
    };

    if fallback_column > original_column {
        fallback
    } else {
        original
    }
}

fn parse_error_column(error: &ParseError) -> Option<usize> {
    match error {
        ParseError::EmptySpineToken { column, .. }
        | ParseError::InvalidSpineToken { column, .. } => Some(*column),
        _ => None,
    }
}

fn parse_empty_data_field_line(
    line_number: usize,
    raw: &str,
    spine_state: &SpineState,
    options: &ParseOptions,
) -> Result<KernLine, ParseError> {
    let tokens = split_spine_tokens_preserving_empty(raw);
    spine_state.validate_line_width(line_number, tokens.len())?;

    let contents = tokens
        .iter()
        .enumerate()
        .map(|(spine, token)| {
            let spine_type = spine_state.spine_type(SpineIndex(spine))?;
            if token.is_empty() {
                if matches!(spine_type, SpineType::Kern) {
                    Ok(SpineContent::Kern(KernContent::Null))
                } else {
                    Err(ParseError::EmptySpineToken {
                        line: line_number,
                        column: spine + 1,
                    })
                }
            } else if token.starts_with('*') || token.starts_with('!') || token.starts_with('=') {
                Err(ParseError::EmptySpineToken {
                    line: line_number,
                    column: spine + 1,
                })
            } else {
                SpineContent::from_token_with_options(spine_type, token, options).map_err(
                    |source| ParseError::InvalidSpineToken {
                        line: line_number,
                        column: spine + 1,
                        raw: (*token).to_owned(),
                        source: Box::new(source),
                    },
                )
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(KernLine::Data(contents))
}

fn parse_barline_line(
    line_number: usize,
    tokens: &[&str],
    spine_state: &SpineState,
) -> Result<KernLine, ParseError> {
    if tokens.len() > 1 {
        let Some(expected) = spine_state.expected_spine_count() else {
            return Err(ParseError::RepeatedBarlineWithoutActiveSpines {
                line: line_number,
                found: tokens.len(),
            });
        };
        if tokens.len() != expected {
            return Err(ParseError::SpineCountMismatch {
                line: line_number,
                expected,
                found: tokens.len(),
            });
        }
    }

    let parsed = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| {
            Barline::from_token(SpineIndex(index), token).map_err(|source| {
                ParseError::InvalidSpineToken {
                    line: line_number,
                    column: index + 1,
                    raw: (*token).to_owned(),
                    source: Box::new(source),
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if parsed.is_empty() {
        return Err(ParseError::EmptyKernLine { line: line_number });
    }

    Ok(KernLine::Barline(Barline::from_spine_barlines(
        parsed.iter().map(SpineBarline::from).collect(),
    )))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParsedSpineTokenKind {
    NullInterpretationToken,
    SpineType,
    Tandem,
    SpineManipulator,
    LocalComment,
    Barline,
    Data,
}

impl ParsedSpineTokenKind {
    fn name(self) -> &'static str {
        match self {
            Self::SpineType => "spine type",
            Self::Tandem => "tandem interpretation",
            Self::SpineManipulator => "spine manipulator",
            Self::LocalComment => "local comment",
            Self::Barline => "barline",
            Self::Data => "data",
            Self::NullInterpretationToken => "null interpretation token",
        }
    }
}

enum ParsedSpineToken {
    NullInterpretationToken(NullInterpretationToken),
    SpineType(SpineType),
    Tandem(TandemInterpretation),
    SpineManipulator(SpineManipulator),
    LocalComment(LocalComment),
    Barline,
    Data(SpineContent),
}

impl ParsedSpineToken {
    fn parse(
        spine: SpineIndex,
        raw: &str,
        spine_state: &SpineState,
        options: &ParseOptions,
    ) -> Result<Self, ParseError> {
        match raw {
            token if token.starts_with("**") => SpineType::try_from(token).map(Self::SpineType),
            "*" => NullInterpretationToken::try_from(raw).map(Self::NullInterpretationToken),
            "*^" | "*v" | "*x" | "*+" | "*-" => {
                SpineManipulator::parse(raw).map(Self::SpineManipulator)
            }
            token if token.starts_with('*') => {
                parse_tandem_token(spine, token, spine_state).map(Self::Tandem)
            }
            token if token.starts_with('!') => {
                LocalComment::try_from(token).map(Self::LocalComment)
            }
            token if token.starts_with('=') => {
                Barline::from_token(spine, token).map(|_| Self::Barline)
            }
            _ => {
                SpineContent::from_token_with_options(spine_state.spine_type(spine)?, raw, options)
                    .map(Self::Data)
            }
        }
    }

    fn kind(&self) -> ParsedSpineTokenKind {
        match self {
            Self::NullInterpretationToken(_) => ParsedSpineTokenKind::NullInterpretationToken,
            Self::SpineType(_) => ParsedSpineTokenKind::SpineType,
            Self::Tandem(_) => ParsedSpineTokenKind::Tandem,
            Self::SpineManipulator(_) => ParsedSpineTokenKind::SpineManipulator,
            Self::LocalComment(_) => ParsedSpineTokenKind::LocalComment,
            Self::Barline => ParsedSpineTokenKind::Barline,
            Self::Data(_) => ParsedSpineTokenKind::Data,
        }
    }
}

impl TryFrom<&str> for LocalComment {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        if raw == "!" {
            return Ok(Self::Null);
        }

        let Some(body) = raw.strip_prefix("!LO:TX:") else {
            return Ok(Self::Unsupported(raw.to_owned()));
        };
        let mut placement = None;
        let mut italic = false;
        let mut text = None;

        for part in body.split(':') {
            match part {
                "a" if placement.is_none() => placement = Some(TextPlacement::Above),
                "b" if placement.is_none() => placement = Some(TextPlacement::Below),
                "i" if !italic => italic = true,
                _ if part.starts_with("t=") && text.is_none() => {
                    let value = &part[2..];
                    if value.is_empty() || !valid_generated_text(value) {
                        return Ok(Self::Unsupported(raw.to_owned()));
                    }
                    text = Some(value.to_owned());
                }
                _ => {
                    return Ok(Self::Unsupported(raw.to_owned()));
                }
            }
        }

        let Some(placement) = placement else {
            return Ok(Self::Unsupported(raw.to_owned()));
        };
        let Some(text) = text else {
            return Ok(Self::Unsupported(raw.to_owned()));
        };

        Ok(Self::LayoutText(LayoutText {
            placement,
            italic,
            text,
        }))
    }
}

fn valid_generated_text(value: &str) -> bool {
    value.chars().all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, ' ' | ',' | '.' | '-')
    })
}

struct NullInterpretationToken;

impl TryFrom<&str> for NullInterpretationToken {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        match raw {
            "*" => Ok(Self),
            _ => todo!("Throw a good error"),
        }
    }
}

impl TryFrom<&str> for SpineType {
    type Error = ParseError;

    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        match raw {
            "**kern" => Ok(Self::Kern),
            "**dynam" => Ok(Self::Dynam),
            "**fing" => Ok(Self::Fing),
            "**text" => Ok(Self::Text),
            "**mxhm" => Ok(Self::MxmlChords),
            token if token.starts_with("**") => Ok(Self::Other(raw.to_owned())),
            _ => Err(ParseError::UnknownSpineType {
                raw: raw.to_owned(),
            }),
        }
    }
}

impl SpineManipulator {
    fn parse(raw: &str) -> Result<Self, ParseError> {
        match raw {
            "*" => Ok(Self::Null),
            "*-" => Ok(Self::Terminate),
            "*^" => Ok(Self::Split),
            "*v" => Ok(Self::Merge),
            _ => Err(ParseError::UnsupportedSpineManipulator {
                raw: raw.to_owned(),
            }),
        }
    }
}
