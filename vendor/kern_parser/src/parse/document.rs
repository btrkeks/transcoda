use crate::{
    line::split_spine_tokens,
    normalization::invisible_rest_slices::{
        contains_durationless_invisible_rest_slices, rehydrate_durationless_invisible_rest_slices,
        InvisibleRestSliceError,
    },
    parse::spine_state::SpineState,
    types::*,
    ParseError,
};

use super::finalize::finalize_parsed_document;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParseDialect {
    #[default]
    HumdrumKern,
    TranscodaV2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParseOptions {
    pub dialect: ParseDialect,
}

impl KernDocument {
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        Self::parse_with_options(input, ParseOptions::default())
    }

    pub fn parse_with_options(input: &str, options: ParseOptions) -> Result<Self, ParseError> {
        let mut metadata = KernMetadata::default();
        let mut lines = Vec::new();
        let mut first_spine_type_count = None;
        let mut first_typed_line_count = None;
        let mut spine_type_lines_allowed = true;
        let mut spine_state = SpineState::new();
        let mut pending_system_break: Option<(SystemBreak, usize, String)> = None;

        for (index, raw) in input.lines().enumerate() {
            let line_number = index + 1;

            if raw.is_empty() {
                reject_pending_system_break(&mut pending_system_break)?;
                continue;
            }

            if raw.starts_with("!!") {
                if let Some(system_break) = parse_system_break(line_number, raw)? {
                    reject_pending_system_break(&mut pending_system_break)?;
                    pending_system_break = Some((system_break, line_number, raw.to_owned()));
                    continue;
                }
                reject_pending_system_break(&mut pending_system_break)?;
                if raw.starts_with("!!!") {
                    parse_reference_record(line_number, raw, &mut metadata)?;
                }
                continue;
            }

            if raw.starts_with('!')
                && is_legacy_unsupported_local_comment_line(line_number, raw, &spine_state)?
            {
                reject_pending_system_break(&mut pending_system_break)?;
                continue;
            }

            if !raw.starts_with('!')
                && raw.contains('!')
                && is_local_comment_overlay(line_number, raw, &spine_state)?
            {
                reject_pending_system_break(&mut pending_system_break)?;
                continue;
            }

            let record =
                KernLine::parse_record_with_spine_types(line_number, raw, &spine_state, &options)?;
            for mut line in record {
                attach_pending_system_break(&mut line, &mut pending_system_break)?;
                if matches!(line, KernLine::SpineTypeLine(_)) && !spine_type_lines_allowed {
                    return Err(ParseError::SpineTypeLineNotAtStart { line: line_number });
                }

                if !is_initial_comment_line(&line) {
                    spine_type_lines_allowed = false;
                }

                if let Some(spine_count) = kern_line_spine_count(&line) {
                    if first_typed_line_count.is_none() {
                        first_typed_line_count = Some(spine_count);
                    }
                    if first_spine_type_count.is_none()
                        && matches!(line, KernLine::SpineTypeLine(_))
                    {
                        first_spine_type_count = Some(spine_count);
                    }
                    match &line {
                        KernLine::SpineTypeLine(types) => spine_state.initialize(types.clone()),
                        KernLine::SpineManipulatorLine(manipulators) => {
                            if spine_state.is_empty() {
                                spine_state.initialize(vec![SpineType::Kern; spine_count]);
                            }
                            spine_state.apply_manipulators(manipulators.tokens())?;
                        }
                        KernLine::TandemInterpretationLine(_) | KernLine::LocalCommentLine(_)
                            if spine_state.is_empty() =>
                        {
                            spine_state.initialize(vec![SpineType::Kern; spine_count]);
                        }
                        KernLine::Data(_) if spine_state.is_empty() => {
                            spine_state.initialize(vec![SpineType::Kern; spine_count]);
                        }
                        _ => {}
                    }
                }
                lines.push((line, Some(line_number)));
            }
        }
        reject_pending_system_break(&mut pending_system_break)?;

        let initial_spine_count = first_spine_type_count
            .or(first_typed_line_count)
            .unwrap_or(0);

        let mut document = Self::from_parts(
            metadata,
            DocumentLayout {
                initial_spine_count: initial_spine_count as u32,
            },
            lines,
        );
        finalize_parsed_document(&mut document);
        validate_dialect(&document, options)?;

        Ok(document)
    }
}

fn attach_pending_system_break(
    line: &mut KernLine,
    pending: &mut Option<(SystemBreak, usize, String)>,
) -> Result<(), ParseError> {
    let Some((system_break, line_number, raw)) = pending.take() else {
        return Ok(());
    };

    let KernLine::Barline(barline) = line else {
        return Err(ParseError::InvalidSystemBreak {
            line: line_number,
            raw,
        });
    };
    barline.system_break_before = Some(system_break);
    Ok(())
}

fn reject_pending_system_break(
    pending: &mut Option<(SystemBreak, usize, String)>,
) -> Result<(), ParseError> {
    if let Some((_, line, raw)) = pending.take() {
        return Err(ParseError::InvalidSystemBreak { line, raw });
    }

    Ok(())
}

fn validate_dialect(document: &KernDocument, options: ParseOptions) -> Result<(), ParseError> {
    if !matches!(options.dialect, ParseDialect::TranscodaV2)
        || !contains_durationless_invisible_rest_slices(document)
    {
        return Ok(());
    }

    let mut rehydrated = document.clone();
    rehydrate_durationless_invisible_rest_slices(&mut rehydrated)
        .map_err(parse_durationless_invisible_rest_slice_error)
}

fn parse_durationless_invisible_rest_slice_error(error: InvisibleRestSliceError) -> ParseError {
    match error {
        InvisibleRestSliceError::ZeroDurationSlice { line, column } => {
            ParseError::InvalidDurationlessInvisibleRestSlice {
                line,
                column,
                message: "slice has zero duration".to_owned(),
            }
        }
        InvisibleRestSliceError::UnspellableSliceDuration {
            line,
            column,
            duration,
        } => ParseError::InvalidDurationlessInvisibleRestSlice {
            line,
            column,
            message: format!("cannot spell slice duration {duration}"),
        },
        InvisibleRestSliceError::RemainingDurationUnderflow {
            line,
            column,
            slice,
            remaining,
        } => ParseError::InvalidDurationlessInvisibleRestSlice {
            line,
            column,
            message: format!(
                "slice duration {slice} exceeds tracked remaining duration {remaining}"
            ),
        },
        InvisibleRestSliceError::DataLineDurations(source) => {
            ParseError::InvalidDurationlessInvisibleRestSlices {
                message: source.to_string(),
            }
        }
        error @ InvisibleRestSliceError::UnequalMergedRemaining { .. } => {
            ParseError::InvalidDurationlessInvisibleRestSlices {
                message: error.to_string(),
            }
        }
        InvisibleRestSliceError::TranscodaV2Lowering { message }
        | InvisibleRestSliceError::TranscodaV2Invariant { message }
        | InvisibleRestSliceError::TranscodaV2SemanticEvents { message } => {
            ParseError::InvalidDurationlessInvisibleRestSlices { message }
        }
        error @ (InvisibleRestSliceError::Dialect(_)
        | InvisibleRestSliceError::UnopenedSplitMerge { .. }
        | InvisibleRestSliceError::TimelineBreak(_)) => {
            ParseError::InvalidDurationlessInvisibleRestSlices {
                message: error.to_string(),
            }
        }
        InvisibleRestSliceError::TranscodaV2DidNotConverge { iterations } => {
            ParseError::InvalidDurationlessInvisibleRestSlices {
                message: format!(
                    "unexpected Transcoda V2 canonicalization error after {iterations} iterations"
                ),
            }
        }
    }
}

fn is_initial_comment_line(line: &KernLine) -> bool {
    matches!(line, KernLine::LocalCommentLine(_))
}

fn is_legacy_unsupported_local_comment_line(
    line_number: usize,
    raw: &str,
    spine_state: &SpineState,
) -> Result<bool, ParseError> {
    let tokens = split_spine_tokens(line_number, raw)?;
    spine_state.validate_line_width(line_number, tokens.len())?;

    Ok(tokens.iter().all(|token| token.starts_with('!'))
        && !tokens.iter().any(|token| {
            token.starts_with("!LO:TX:")
                || token.starts_with("!LO:N:")
                || token.starts_with("!ack:")
        }))
}

/// Parse the global system break namespace: `!!LO:LB` / `!!LO:LB:g=original`.
///
/// Returns `Ok(None)` for global comments outside the `!!LO:LB` namespace.
/// Once a line is in the namespace, malformed parameters are a hard error
/// instead of being silently skipped as an ordinary global comment.
fn parse_system_break(line_number: usize, raw: &str) -> Result<Option<SystemBreak>, ParseError> {
    let Some(parameters) = raw.strip_prefix("!!LO:LB") else {
        return Ok(None);
    };
    // Historical targets may append tabbed local-comment placeholders after
    // this global layout directive; they are not part of the break group.
    let parameters = parameters
        .split_once('\t')
        .map_or(parameters, |(head, _)| head);
    if !parameters.is_empty() && !parameters.starts_with(':') {
        // A longer layout code such as `!!LO:LBX` is outside this namespace.
        return Ok(None);
    }

    let group = match parameters {
        "" => None,
        _ => {
            let group = parameters
                .strip_prefix(":g=")
                .filter(|group| !group.is_empty() && !group.contains(':'));
            let Some(group) = group else {
                return Err(ParseError::InvalidSystemBreak {
                    line: line_number,
                    raw: raw.to_owned(),
                });
            };
            Some(group.to_owned())
        }
    };

    Ok(Some(SystemBreak {
        group,
        source_line: Some(line_number),
    }))
}

fn parse_reference_record(
    line_number: usize,
    raw: &str,
    metadata: &mut KernMetadata,
) -> Result<(), ParseError> {
    let Some((key, value)) = raw
        .strip_prefix("!!!")
        .and_then(|body| body.split_once(':'))
    else {
        return Ok(());
    };

    let value = value.trim_start().to_owned();
    if key == "OTA" || key.starts_with("OTA@@") {
        metadata.alternative_titles.push(value);
        return Ok(());
    }
    match key {
        "OTL" => metadata.title = Some(value),
        "COM" => metadata.composer = Some(value),
        "OPR" => metadata.parent_work = Some(value),
        "OPS" => metadata.opus = Some(value),
        "OMV" => metadata.movement = Some(value),
        "OMD" => metadata.movement_designation = Some(value),
        "ODT" => metadata.date = Some(value),
        "URL" | "SCT" => metadata.source = Some(value),
        OrnamentConvention::RECORD_KEY => {
            // A performance-changing record must never degrade silently: a
            // typo or a second declaration is an error, not `Modern`.
            let parsed = OrnamentConvention::from_record_value(value.trim_end());
            match (parsed, metadata.ornament_convention) {
                (Some(convention), None) => metadata.ornament_convention = Some(convention),
                _ => {
                    return Err(ParseError::InvalidOrnamentConvention {
                        line: line_number,
                        raw: raw.to_owned(),
                    });
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn is_local_comment_overlay(
    line_number: usize,
    raw: &str,
    spine_state: &SpineState,
) -> Result<bool, ParseError> {
    let tokens = match split_spine_tokens(line_number, raw) {
        Ok(tokens) => tokens,
        Err(ParseError::EmptySpineToken { .. }) => return Ok(false),
        Err(error) => return Err(error),
    };
    let has_comment = tokens.iter().any(|token| token.starts_with('!'));
    if !has_comment {
        return Ok(false);
    }

    spine_state.validate_line_width(line_number, tokens.len())?;

    for (index, token) in tokens.iter().enumerate() {
        if token.starts_with('!') {
            continue;
        }

        if !matches!(
            spine_state.spine_type(SpineIndex(index))?,
            SpineType::Other(_)
        ) {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Number of spine columns on a line, or `None` for column-less lines.
fn kern_line_spine_count(line: &KernLine) -> Option<usize> {
    match line {
        KernLine::SpineTypeLine(tokens) => Some(tokens.len()),
        KernLine::TandemInterpretationLine(line) => Some(line.len()),
        KernLine::SpineManipulatorLine(tokens) => Some(tokens.len()),
        KernLine::LocalCommentLine(tokens) => Some(tokens.len()),
        KernLine::Barline(_) => None,
        KernLine::Data(tokens) => Some(tokens.len()),
    }
}
