//! Bar-based document splitting for typed Humdrum `**kern` documents.
//!
//! Every cut reads its spine layout and state from the resolved score: an
//! excerpt opens with the exclusive interpretation, the staff definition per
//! track, the splits that recreate the layout at the cut and the remaining
//! state, and closes the spans still open where it ends.

use thiserror::Error;

use crate::{
    analysis::measure_spans,
    document::Framing,
    meter::is_meter_tandem,
    normalization::{restore_final_staff_topology::RestoreFinalStaffTopology, NormalizationPass},
    score::{bridge, BridgeError, Score, StateErrors, StateKey},
    transforms::layout::Layout,
    DocumentLayout, KernDocument, KernLine, KernMetadata, SpineManipulator, SpineManipulatorLine,
    TandemInterpretationLineKind,
};

use std::ops::Range;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TruncateBarsError {
    #[error("the score does not resolve: {0}")]
    State(#[from] StateErrors),
    #[error("cannot restate the state at the cut: {0}")]
    Bridge(#[from] BridgeError),
    #[error("cannot truncate zero bars")]
    ZeroBarsRequested,
    #[error("cannot truncate {requested} bars from document with {available} bars")]
    NotEnoughBars { requested: usize, available: usize },
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SliceMeasuresError {
    #[error("the score does not resolve: {0}")]
    State(#[from] StateErrors),
    #[error("cannot restate the state at the cut: {0}")]
    Bridge(#[from] BridgeError),
    #[error("measure range must not be empty")]
    EmptyRange,
    #[error("cannot slice measures {requested_start}..{requested_end} from document with {available} measures")]
    RangeOutOfBounds {
        requested_start: usize,
        requested_end: usize,
        available: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitBars {
    pub retained: KernDocument,
    pub removed: KernDocument,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceMeasureRangeOptions {
    pub prefix_policy: SlicePrefixPolicy,
}

impl Default for SliceMeasureRangeOptions {
    fn default() -> Self {
        Self {
            prefix_policy: SlicePrefixPolicy::StandaloneExcerpt,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlicePrefixPolicy {
    /// A self-contained excerpt: the state in force at the cut.
    StandaloneExcerpt,
    /// Emit page-local transcription text. Clef and key context may need to be
    /// repeated at page starts, but carried meter markings should only be shown
    /// on the first page. Source-authored opening state changes are retained
    /// from the slice body instead of being preceded by stale active state.
    PageTranscription { is_first_page: bool },
    /// A renderable continuation: retain carried meter as hidden state.
    /// Source-authored meter declarations at the cut still print.
    PageRendering { is_first_page: bool },
}

pub fn split_trailing_bars(
    document: &KernDocument,
    remove_last: usize,
) -> Result<SplitBars, TruncateBarsError> {
    split_trailing_bars_with_prefix_policy(
        document,
        remove_last,
        SlicePrefixPolicy::StandaloneExcerpt,
    )
}

pub fn split_trailing_bars_with_prefix_policy(
    document: &KernDocument,
    remove_last: usize,
    prefix_policy: SlicePrefixPolicy,
) -> Result<SplitBars, TruncateBarsError> {
    if remove_last == 0 {
        return Err(TruncateBarsError::ZeroBarsRequested);
    }

    let spans = bar_spans(document);
    if remove_last > spans.len() {
        return Err(TruncateBarsError::NotEnoughBars {
            requested: remove_last,
            available: spans.len(),
        });
    }
    let score = Score::resolve(document, Framing::Excerpt)?;

    let first_removed = spans.len() - remove_last;
    let truncation = truncation_index(document, spans[first_removed].start);
    let mut truncated = document.clone();
    truncated.truncate_lines(truncation);
    let end = score.snapshot_before(truncation);
    let position = if matches!(truncated.lines().last(), Some(KernLine::Barline(_))) {
        truncation - 1
    } else {
        truncation
    };
    for (offset, line) in bridge(&end, &end.closed())?.into_iter().enumerate() {
        truncated.insert_line(
            position + offset,
            KernLine::TandemInterpretationLine(line),
            None,
        );
    }
    append_terminal_line(&mut truncated, end.width());
    RestoreFinalStaffTopology::new(Default::default()).normalize(&mut truncated);

    let mut removed = document_line_slice(
        &score,
        spans[first_removed].start,
        document.lines().len(),
        prefix_policy,
    )?;
    RestoreFinalStaffTopology::new(Default::default()).normalize(&mut removed);

    Ok(SplitBars {
        retained: truncated,
        removed,
    })
}

pub fn musical_bar_count(document: &KernDocument) -> usize {
    bar_spans(document).len()
}

/// Return a standalone document containing a zero-based half-open logical measure range.
///
/// The returned document synthesizes the active Humdrum prefix before the first
/// retained measure and appends a valid terminal topology after the last
/// retained measure. The slice includes the leading barline for the first
/// measure and the trailing barline for the last measure when present. A
/// document whose score does not resolve is refused.
pub fn slice_measure_range(
    document: &KernDocument,
    range: Range<usize>,
) -> Result<KernDocument, SliceMeasuresError> {
    slice_measure_range_with_options(document, range, SliceMeasureRangeOptions::default())
}

pub fn slice_measure_range_with_options(
    document: &KernDocument,
    range: Range<usize>,
    options: SliceMeasureRangeOptions,
) -> Result<KernDocument, SliceMeasuresError> {
    if range.start >= range.end {
        return Err(SliceMeasuresError::EmptyRange);
    }

    let spans = measure_spans(document);
    if range.end > spans.len() {
        return Err(SliceMeasuresError::RangeOutOfBounds {
            requested_start: range.start,
            requested_end: range.end,
            available: spans.len(),
        });
    }
    let score = Score::resolve(document, Framing::Excerpt)?;

    let first_span = &spans[range.start];
    let last_span = &spans[range.end - 1];
    let mut slice_start = first_span
        .leading_barline
        .unwrap_or(first_span.content.start);
    let mut slice_end = last_span
        .trailing_barline
        .map(|line_index| line_index + 1)
        .unwrap_or(last_span.content.end);

    if range.start == 0 {
        slice_start = usize::from(matches!(
            document.lines().first(),
            Some(KernLine::SpineTypeLine(_))
        ));
    }
    if range.end == spans.len() {
        slice_end = document.lines().iter().rposition(|line| {
            matches!(line, KernLine::SpineManipulatorLine(manipulators) if is_terminal_line(manipulators.tokens()))
        }).unwrap_or(document.lines().len());
    }
    let mut sliced = document_line_slice(&score, slice_start, slice_end, options.prefix_policy)?;
    append_terminal_line(&mut sliced, score.snapshot_before(slice_end).width());
    RestoreFinalStaffTopology::new(Default::default()).normalize(&mut sliced);
    Ok(sliced)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BarSpan {
    start: usize,
}

fn bar_spans(document: &KernDocument) -> Vec<BarSpan> {
    let mut spans = Vec::new();
    let mut pending_boundary = None;
    let mut active_start = None;

    for (index, line) in document.lines().iter().enumerate() {
        match line {
            KernLine::Barline(_) => {
                pending_boundary = Some(index);
                active_start = None;
            }
            KernLine::Data(_) if active_start.is_none() => {
                let start = pending_boundary.take().unwrap_or(index);
                active_start = Some(start);
                spans.push(BarSpan { start });
            }
            KernLine::SpineManipulatorLine(manipulators)
                if is_terminal_line(manipulators.tokens()) =>
            {
                active_start = None;
            }
            _ => {}
        }
    }

    spans
}

/// Lines `slice_start..slice_end` of the score's document as an excerpt.
///
/// The prologue states the staff definition per track before any split, as a
/// page header does, then recreates the layout at the cut, then restates the
/// rest of the state in force there. A page transcription takes its staff
/// definition from its first data line and drops the clef, key and (when it
/// restates meter) meter lines of its opening. Spans still open at the end
/// are closed before the final barline.
fn document_line_slice(
    score: &Score,
    slice_start: usize,
    slice_end: usize,
    prefix_policy: SlicePrefixPolicy,
) -> Result<KernDocument, BridgeError> {
    let document = score.document();
    let (staff_from, staff_keys, synthesize_meter) = match prefix_policy {
        SlicePrefixPolicy::StandaloneExcerpt | SlicePrefixPolicy::PageRendering { .. } => {
            (slice_start, StateKey::STAFF_DEFINITION.as_slice(), true)
        }
        SlicePrefixPolicy::PageTranscription { is_first_page } => {
            let first_data =
                first_data_line(document, slice_start, slice_end).unwrap_or(slice_start);
            let synthesize_meter =
                is_first_page || declares_meter(document, slice_start, first_data);
            let keys = if synthesize_meter {
                StateKey::STAFF_DEFINITION.as_slice()
            } else {
                &StateKey::STAFF_DEFINITION[..2]
            };
            (first_data, keys, synthesize_meter)
        }
    };

    let layout = Layout::before(score, slice_start);
    let cut = score.snapshot_before(slice_start);
    let staff = cut
        .tracks()
        .defaulted()
        .with(&score.snapshot_before(staff_from).tracks(), staff_keys)?;

    let mut lines = vec![(
        KernLine::SpineTypeLine(layout.initial_spine_types().to_vec()),
        None,
    )];
    for line in bridge(&staff.defaulted(), &staff)? {
        lines.push((
            KernLine::TandemInterpretationLine(layout.widen_tracks(&line)),
            None,
        ));
    }
    lines.extend(
        layout
            .manipulator_lines()
            .into_iter()
            .map(|line| (line, None)),
    );
    let stated = cut.defaulted().with(&staff, &StateKey::STAFF_DEFINITION)?;
    let target = cut.with(&staff, &StateKey::STAFF_DEFINITION)?;
    lines.extend(
        bridge(&stated, &target)?
            .into_iter()
            .map(|line| (KernLine::TandemInterpretationLine(line), None)),
    );

    for (index, line) in document.lines()[slice_start..slice_end]
        .iter()
        .enumerate()
        .map(|(offset, line)| (slice_start + offset, line))
    {
        if index < staff_from && !keep_page_opening_line(line, synthesize_meter) {
            continue;
        }
        let line = if index == slice_start
            && document.lines()[..slice_start]
                .iter()
                .any(|line| matches!(line, KernLine::Data(_)))
        {
            match line {
                KernLine::Barline(barline) => KernLine::Barline(barline.right_face()),
                _ => line.clone(),
            }
        } else {
            line.clone()
        };
        lines.push((line, document.source(index)));
    }

    let end = score.snapshot_before(slice_end);
    let position = if matches!(lines.last(), Some((KernLine::Barline(_), _))) {
        lines.len() - 1
    } else {
        lines.len()
    };
    lines.splice(
        position..position,
        bridge(&end, &end.closed())?
            .into_iter()
            .map(|line| (KernLine::TandemInterpretationLine(line), None)),
    );
    let mut result = KernDocument::from_parts(
        KernMetadata::default(),
        DocumentLayout {
            initial_spine_count: layout.initial_width() as u32,
        },
        lines,
    );
    if matches!(
        prefix_policy,
        SlicePrefixPolicy::PageRendering {
            is_first_page: false
        }
    ) {
        let first_data = first_data_line(document, slice_start, slice_end).unwrap_or(slice_start);
        let mut authored_staffs = std::collections::BTreeSet::new();
        for index in slice_start..first_data {
            if let KernLine::TandemInterpretationLine(tandems) = &document.lines()[index] {
                for (column, token) in tandems.tokens().iter().enumerate() {
                    if is_meter_tandem(token) {
                        let layer = score.topology().layers_at(index)[column];
                        if let Some(staff) = score
                            .topology()
                            .track(score.topology().track_of(layer))
                            .staff
                        {
                            authored_staffs.insert(staff);
                        }
                    }
                }
            }
        }
        hide_opening_meter_except_staffs(&mut result, &authored_staffs)
            .expect("sliced resolved score retains meter state");
    }
    Ok(result)
}

/// Make a synthetic continuation variant without changing its active meter.
/// Only meter groups before the first data record are hidden.
pub fn hide_opening_meter(document: &mut KernDocument) -> Result<(), crate::score::StateErrors> {
    hide_opening_meter_except_staffs(document, &std::collections::BTreeSet::new())
}

/// Hide opening meters except on staffs with an authored boundary declaration.
pub fn hide_opening_meter_except_staffs(
    document: &mut KernDocument,
    preserved_staffs: &std::collections::BTreeSet<u32>,
) -> Result<(), crate::score::StateErrors> {
    use crate::{
        meter::Meter, TandemInterpretation as T, TandemInterpretationLine, TimeSignatureTandem,
    };
    let end = first_data_line(document, 0, document.lines().len()).unwrap_or(0);
    let score = Score::resolve(document, Framing::Excerpt)?;
    // A symbol can be the only declaration of the effective numeric meter.
    // Add its hidden numeric state before replacing the visible symbol.
    let mut replacements = Vec::new();
    for (index, line) in document.lines()[..end].iter().enumerate() {
        let KernLine::TandemInterpretationLine(tandems) = line else {
            continue;
        };
        if !tandems.tokens().iter().any(is_meter_tandem) {
            continue;
        }
        let hide = |column: usize| {
            let layer = score.topology().layers_at(index)[column];
            !score
                .topology()
                .track(score.topology().track_of(layer))
                .staff
                .is_some_and(|staff| preserved_staffs.contains(&staff))
        };
        let numeric = tandems
            .tokens()
            .iter()
            .enumerate()
            .map(|(column, token)| {
                if !hide(column) {
                    return match token {
                        T::TimeSignature(_) | T::FreeMeter => token.clone(),
                        _ => T::Null,
                    };
                }
                if !is_meter_tandem(token) {
                    return T::Null;
                }
                match score
                    .state_after(index, column)
                    .track
                    .meter
                    .effective_meter()
                {
                    Some(Meter::Regular(time_signature)) => T::TimeSignature(TimeSignatureTandem {
                        time_signature,
                        invisible: true,
                    }),
                    Some(Meter::Free) => T::FreeMeter,
                    None => T::Null,
                }
            })
            .collect();
        let symbols = tandems
            .tokens()
            .iter()
            .enumerate()
            .map(|(column, token)| {
                if hide(column) && is_meter_tandem(token) {
                    T::MeterSymbol(crate::MeterSymbol::Empty)
                } else if matches!(token, T::MeterSymbol(_)) {
                    token.clone()
                } else {
                    T::Null
                }
            })
            .collect();
        replacements.push((
            index,
            TandemInterpretationLine::new(numeric).expect("homogeneous meter"),
            TandemInterpretationLine::new(symbols).expect("homogeneous symbols"),
        ));
    }
    drop(score);
    for (index, numeric, symbols) in replacements.into_iter().rev() {
        let source = document.source(index);
        document.lines_mut()[index] = KernLine::TandemInterpretationLine(numeric);
        document.insert_line(
            index + 1,
            KernLine::TandemInterpretationLine(symbols),
            source,
        );
    }
    Ok(())
}

fn declares_meter(document: &KernDocument, slice_start: usize, slice_end: usize) -> bool {
    document.lines()[slice_start..slice_end].iter().any(|line| {
        matches!(
            line,
            KernLine::TandemInterpretationLine(tandems)
                if tandems.tokens().iter().any(is_meter_tandem)
        )
    })
}

fn first_data_line(document: &KernDocument, slice_start: usize, slice_end: usize) -> Option<usize> {
    document.lines()[slice_start..slice_end]
        .iter()
        .position(|line| matches!(line, KernLine::Data(_)))
        .map(|offset| slice_start + offset)
}

fn keep_page_opening_line(line: &KernLine, synthesize_meter: bool) -> bool {
    match line {
        KernLine::Barline(_) | KernLine::SpineTypeLine(_) => false,
        KernLine::TandemInterpretationLine(tandems) => match tandems.kind() {
            TandemInterpretationLineKind::Clef | TandemInterpretationLineKind::KeySignature => {
                false
            }
            TandemInterpretationLineKind::TimeSignature
            | TandemInterpretationLineKind::MeterSymbol => !synthesize_meter,
            _ => true,
        },
        KernLine::LocalCommentLine(_) | KernLine::Data(_) | KernLine::SpineManipulatorLine(_) => {
            true
        }
    }
}

fn truncation_index(document: &KernDocument, span_start: usize) -> usize {
    if matches!(document.lines().get(span_start), Some(KernLine::Barline(_))) {
        span_start + 1
    } else {
        span_start
    }
}

fn append_terminal_line(document: &mut KernDocument, width: usize) {
    if width == 0 {
        return;
    }

    let tokens = vec![SpineManipulator::Terminate; width];
    document.push_line(
        KernLine::SpineManipulatorLine(
            SpineManipulatorLine::new(tokens).expect("generated terminal line is valid"),
        ),
        None,
    );
}

fn is_terminal_line(manipulators: &[SpineManipulator]) -> bool {
    manipulators
        .iter()
        .all(|token| matches!(token, SpineManipulator::Terminate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        types::{BarlineStyle, RepeatStyle},
        KernContent, SerializePreset, SpineContent, TandemInterpretation,
    };

    fn render(document: &KernDocument) -> String {
        document
            .lines()
            .iter()
            .filter_map(render_line)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    fn render_line(line: &KernLine) -> Option<String> {
        match line {
            KernLine::SpineTypeLine(_) => None,
            KernLine::TandemInterpretationLine(tandems) => Some(
                tandems
                    .tokens()
                    .iter()
                    .map(tandem_token)
                    .collect::<Vec<_>>()
                    .join("\t"),
            ),
            KernLine::SpineManipulatorLine(manipulators) => Some(
                manipulators
                    .tokens()
                    .iter()
                    .map(manipulator_token)
                    .collect::<Vec<_>>()
                    .join("\t"),
            ),
            KernLine::LocalCommentLine(comments) => {
                Some(comments.iter().map(|_| "!").collect::<Vec<_>>().join("\t"))
            }
            KernLine::Barline(barline) => Some(barline_token(barline).to_owned()),
            KernLine::Data(contents) => Some(
                contents
                    .iter()
                    .map(|content| match content {
                        SpineContent::Kern(KernContent::Null) => ".",
                        SpineContent::Kern(
                            KernContent::Note(_)
                            | KernContent::Chord(_)
                            | KernContent::Rest(_)
                            | KernContent::InvisibleRestSlice(_)
                            | KernContent::MultiMeasureRest(_)
                            | KernContent::Grace(_),
                        ) => "data",
                        SpineContent::Dynam(_)
                        | SpineContent::Fing(_)
                        | SpineContent::Text(_)
                        | SpineContent::MxmlHarmony(_)
                        | SpineContent::Opaque { .. } => "?",
                    })
                    .collect::<Vec<_>>()
                    .join("\t"),
            ),
        }
    }

    fn tandem_token(tandem: &TandemInterpretation) -> &'static str {
        match tandem {
            TandemInterpretation::Null => "*",
            TandemInterpretation::Clef(_) => "*clef",
            TandemInterpretation::KeySignature(_) => "*k",
            TandemInterpretation::TimeSignature(_) => "*M",
            _ => "?",
        }
    }

    fn barline_token(barline: &crate::Barline) -> &'static str {
        if barline.styles().contains(&BarlineStyle::Double) {
            "=="
        } else if barline.repeat() == RepeatStyle::Start {
            "=|:"
        } else if barline.repeat() == RepeatStyle::End {
            "=:|"
        } else {
            "="
        }
    }

    fn manipulator_token(manipulator: &SpineManipulator) -> &'static str {
        match manipulator {
            SpineManipulator::Null => "*",
            SpineManipulator::Terminate => "*-",
            SpineManipulator::Split => "*^",
            SpineManipulator::Merge => "*v",
        }
    }

    #[test]
    fn split_suffix_synthesizes_active_state_header() {
        let input = KernDocument::parse(
            "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n*clefF4\n*k[b-]\n*M3/4\n=2\n4d\n==\n*-\n",
        )
        .unwrap();

        let split = split_trailing_bars(&input, 1).unwrap();

        assert_eq!(render(&split.removed), "*clef\n*k\n*M\n=\ndata\n==\n*-\n");
    }

    #[test]
    fn split_suffix_preserves_initial_staff_count_before_active_splits() {
        let input = KernDocument::parse(
            "**kern\t**kern\t**kern\t**kern\t**kern\t**kern\n\
             *clefF4\t*clefG2\t*clefF4\t*clefGv2\t*clefG2\t*clefG2\n\
             =\t=\t=\t=\t=\t=\n\
             *\t*^\t*\t*\t*\t*\n\
             4AA\t2.e\t2c#\t1r\t1r\t1r\t1r\n\
             =\t=\t=\t=\t=\t=\t=\n\
             *^\t*\t*\t*\t*\t*\t*\n\
             4AA\t2.A\t2.f#\t4e\t1r\t1r\t1r\t1r\n\
             =\t=\t=\t=\t=\t=\t=\t=\n\
             4D\t4A\t4ff#\t4a\t1r\t1r\t1r\t1r\n\
             ==\t==\t==\t==\t==\t==\t==\t==\n\
             *-\t*-\t*-\t*-\t*-\t*-\t*-\t*-\n",
        )
        .unwrap();

        let split = split_trailing_bars(&input, 1).unwrap();
        let serialized = SerializePreset::HumdrumKern.serialize_document(&split.removed);
        let serialized_lines = serialized.lines().collect::<Vec<_>>();

        assert_eq!(split.removed.staff_count(), 6);
        assert_eq!(
            serialized_lines.first(),
            Some(&"**kern\t**kern\t**kern\t**kern\t**kern\t**kern")
        );
        let split_line = serialized_lines
            .iter()
            .position(|line| *line == "*^\t*^\t*\t*\t*\t*")
            .expect("split suffix should synthesize active split topology");
        let first_data_line = serialized_lines
            .iter()
            .position(|line| line.starts_with("4D\t4A\t4ff#\t4a"))
            .expect("split suffix should retain the requested musical body");
        assert!(split_line < first_data_line);
        assert!(KernDocument::parse(&serialized).is_ok());
    }

    #[test]
    fn slice_measure_range_synthesizes_active_state_header() {
        let input = KernDocument::parse(
            "**kern\n\
             *clefG2\n\
             *k[]\n\
             *M4/4\n\
             =1\n\
             4c\n\
             *clefF4\n\
             *k[b-]\n\
             *M3/4\n\
             =2\n\
             4d\n\
             =3\n\
             4e\n\
             ==\n\
             *-\n",
        )
        .unwrap();

        let slice = slice_measure_range(&input, 1..2).unwrap();

        assert_eq!(render(&slice), "*clef\n*k\n*M\n=\ndata\n=\n*-\n");
        assert!(
            KernDocument::parse(&SerializePreset::HumdrumKern.serialize_document(&slice)).is_ok()
        );
    }

    #[test]
    fn page_transcription_slice_keeps_meter_symbol_marking_on_first_page() {
        let input = KernDocument::parse(
            "**kern\n\
             *clefG2\n\
             *k[]\n\
             *M4/4\n\
             *met(c)\n\
             =1\n\
             4c\n\
             =2\n\
             4d\n\
             *-\n",
        )
        .unwrap();

        let slice = slice_measure_range_with_options(
            &input,
            0..1,
            SliceMeasureRangeOptions {
                prefix_policy: SlicePrefixPolicy::PageTranscription {
                    is_first_page: true,
                },
            },
        )
        .unwrap();
        let serialized = SerializePreset::HumdrumKern.serialize_document(&slice);

        assert!(serialized.contains("*met(c)"));
        assert!(KernDocument::parse(&serialized).is_ok());
    }

    #[test]
    fn page_transcription_slice_keeps_numeric_meter_marking_on_first_page() {
        let input = KernDocument::parse(
            "**kern\n\
             *clefG2\n\
             *k[]\n\
             *M3/4\n\
             =1\n\
             2.c\n\
             =2\n\
             2.d\n\
             *-\n",
        )
        .unwrap();

        let slice = slice_measure_range_with_options(
            &input,
            0..1,
            SliceMeasureRangeOptions {
                prefix_policy: SlicePrefixPolicy::PageTranscription {
                    is_first_page: true,
                },
            },
        )
        .unwrap();
        let serialized = SerializePreset::HumdrumKern.serialize_document(&slice);

        assert!(serialized.contains("*M3/4"));
        assert!(KernDocument::parse(&serialized).is_ok());
    }

    #[test]
    fn page_transcription_slice_omits_carried_meter_markings_after_first_page() {
        let input = KernDocument::parse(
            "**kern\n\
             *clefG2\n\
             *k[]\n\
             *M4/4\n\
             *met(c)\n\
             =1\n\
             4c\n\
             =2\n\
             4d\n\
             =3\n\
             4e\n\
             *-\n",
        )
        .unwrap();

        let slice = slice_measure_range_with_options(
            &input,
            1..2,
            SliceMeasureRangeOptions {
                prefix_policy: SlicePrefixPolicy::PageTranscription {
                    is_first_page: false,
                },
            },
        )
        .unwrap();
        let serialized = SerializePreset::HumdrumKern.serialize_document(&slice);

        assert!(serialized.contains("*clefG2"));
        assert!(serialized.contains("*k[]"));
        assert!(!serialized.contains("*M4/4"));
        assert!(!serialized.contains("*met(c)"));
        assert!(KernDocument::parse(&serialized).is_ok());
    }

    #[test]
    fn page_transcription_slice_prefers_retained_boundary_key_signature() {
        let input = KernDocument::parse(
            "**kern\t**kern\n\
             *clefF4\t*clefG2\n\
             *k[b-e-a-d-]\t*k[b-e-a-d-]\n\
             *M4/4\t*M4/4\n\
             *met(c)\t*met(c)\n\
             =1\t=1\n\
             4C\t4c\n\
             =||\t=||\n\
             *k[f#c#g#d#]\t*k[f#c#g#d#]\n\
             4D\t4d\n\
             =\t=\n\
             *-\t*-\n",
        )
        .unwrap();

        let slice = slice_measure_range_with_options(
            &input,
            1..2,
            SliceMeasureRangeOptions {
                prefix_policy: SlicePrefixPolicy::PageTranscription {
                    is_first_page: false,
                },
            },
        )
        .unwrap();
        let serialized = SerializePreset::HumdrumKern.serialize_document(&slice);

        assert!(serialized.contains("*clefF4\t*clefG2"));
        assert!(serialized.contains("*k[f#c#g#d#]\t*k[f#c#g#d#]"));
        assert!(!serialized.contains("*k[b-e-a-d-]"));
        assert!(!serialized.contains("*M4/4"));
        assert!(!serialized.contains("*met(c)"));
        assert!(KernDocument::parse(&serialized).is_ok());
    }

    #[test]
    fn slice_measure_range_preserves_split_topology_before_slice_start() {
        let input = KernDocument::parse(
            "**kern\t**kern\n\
             *clefF4\t*clefG2\n\
             =1\t=1\n\
             *\t*^\n\
             4AA\t4c\t4e\n\
             =2\t=2\t=2\n\
             4BB\t4d\t4f\n\
             ==\t==\t==\n\
             *-\t*-\t*-\n",
        )
        .unwrap();

        let slice = slice_measure_range(&input, 1..2).unwrap();
        let serialized = SerializePreset::HumdrumKern.serialize_document(&slice);
        let serialized_lines = serialized.lines().collect::<Vec<_>>();

        assert_eq!(slice.staff_count(), 2);
        assert_eq!(serialized_lines.first(), Some(&"**kern\t**kern"));
        assert!(
            serialized_lines.contains(&"*\t*^"),
            "slice should synthesize the active split before retained data: {serialized}"
        );
        assert!(
            serialized_lines
                .iter()
                .any(|line| line.starts_with("4BB\t4d\t4f")),
            "slice should retain the requested measure body: {serialized}"
        );
        assert!(KernDocument::parse(&serialized).is_ok());
    }

    fn resolves(document: &KernDocument) -> String {
        let text = SerializePreset::HumdrumKern.serialize_document(document);
        let reparsed = KernDocument::parse(&text).unwrap();
        crate::score::Score::resolve(&reparsed, Framing::Complete)
            .unwrap_or_else(|errors| panic!("{errors}\n{text}"));
        text
    }

    #[test]
    fn clef_changed_in_one_subspine_is_stated_once_for_its_staff() {
        // The clef change in the first subspine applies to the whole staff;
        // a per-column prologue wrote F4 and G2 into the two subspines.
        let input = KernDocument::parse(
            "**kern\t**kern\n*clefG2\t*clefG2\n*^\t*\n=1\t=1\t=1\n4c\t4e\t4g\n*clefF4\t*\t*\n=2\t=2\t=2\n4C\t4E\t4g\n*v\t*v\t*\n=3\t=3\n4C\t4g\n==\t==\n*-\t*-\n",
        )
        .unwrap();

        let text = resolves(&slice_measure_range(&input, 1..2).unwrap());

        assert!(
            text.starts_with("**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*^\t*\n="),
            "{text}"
        );
    }

    #[test]
    fn slice_reopens_layer_state_per_column_after_the_splits() {
        let input = KernDocument::parse(
            "**kern\n*clefG2\n*^\n*\t*cue\n=1\t=1\n4c\t4e\n=2\t=2\n4d\t4f\n*\t*Xcue\n*v\t*v\n==\n*-\n",
        )
        .unwrap();

        let text = resolves(&slice_measure_range(&input, 1..2).unwrap());

        assert!(
            text.starts_with("**kern\n*clefG2\n*k[]\n*^\n*\t*cue\n=\t="),
            "{text}"
        );
    }

    #[test]
    fn slice_restates_the_staff_line_count() {
        let input = KernDocument::parse(
            "**kern\n*clefX\n*stria1\n*^\n=1\t=1\n4c\t4e\n=2\t=2\n4d\t4f\n*v\t*v\n==\n*-\n",
        )
        .unwrap();

        let text = resolves(&slice_measure_range(&input, 1..2).unwrap());

        assert!(
            text.starts_with("**kern\n*clefX\n*k[]\n*^\n*stria1\t*\n=\t="),
            "{text}"
        );
    }

    #[test]
    fn slice_refuses_a_document_that_does_not_resolve() {
        let input = KernDocument::parse("**kern\t**kern\n=1\t=1\n4c\t4e\n*v\t*v\n=2\n4c\n==\n*-\n")
            .unwrap();

        assert!(matches!(
            slice_measure_range(&input, 0..1),
            Err(SliceMeasuresError::State(_))
        ));
    }

    #[test]
    fn page_transcription_states_staff_definition_before_the_cut_splits() {
        let input = KernDocument::parse(
            "**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n=1\t=1\n*\t*^\n4C\t4c\t4e\n=2\t=2\t=2\n4D\t4d\t4f\n*\t*v\t*v\n==\t==\n*-\t*-\n",
        )
        .unwrap();

        let slice = slice_measure_range_with_options(
            &input,
            1..2,
            SliceMeasureRangeOptions {
                prefix_policy: SlicePrefixPolicy::PageTranscription {
                    is_first_page: false,
                },
            },
        )
        .unwrap();
        let text = resolves(&slice);

        assert!(
            text.starts_with("**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*\t*^\n4D\t4d\t4f\n"),
            "{text}"
        );
    }

    #[test]
    fn slice_measure_range_states_entering_clef_key_and_meter_before_the_split() {
        let input = KernDocument::parse(
            "**kern\t**kern\n\
             *clefF4\t*clefG2\n\
             *k[b-e-]\t*k[b-e-]\n\
             *M3/4\t*M3/4\n\
             =1\t=1\n\
             *^\t*\n\
             2.C\t2.G\t2.c\n\
             =2\t=2\t=2\n\
             2.D\t2.A\t2.d\n\
             ==\t==\t==\n\
             *v\t*v\t*\n\
             *-\t*-\n",
        )
        .unwrap();

        let slice = slice_measure_range(&input, 1..2).unwrap();

        assert_eq!(
            SerializePreset::HumdrumKern.serialize_document(&slice),
            "**kern\t**kern\n\
             *clefF4\t*clefG2\n\
             *k[b-e-]\t*k[b-e-]\n\
             *M3/4\t*M3/4\n\
             *^\t*\n\
             =\t=\t=\n\
             2.D\t2.A\t2.d\n\
             *v\t*v\t*\n\
             ==\t==\n\
             *-\t*-\n\
             !!!RDF**kern: > = above\n\
             !!!RDF**kern: < = below\n\
             !!!RDF**kern: N = linked\n\
             !!!RDF**kern: Z = editorial accidental, parentheses\n\
             !!!RDF**kern: j = editorial accidental, brackets\n"
        );
    }

    #[test]
    fn slice_measure_range_states_a_clef_written_in_one_subspine_for_its_staff() {
        let input = KernDocument::parse(
            "**kern\t**kern\n\
             *clefF4\t*clefG2\n\
             *k[]\t*k[]\n\
             =1\t=1\n\
             *^\t*\n\
             *\t*clefG2\t*\n\
             2C\t2g\t2c\n\
             =2\t=2\t=2\n\
             2D\t2a\t2d\n\
             ==\t==\t==\n\
             *v\t*v\t*\n\
             *-\t*-\n",
        )
        .unwrap();

        let slice = slice_measure_range(&input, 1..2).unwrap();

        assert_eq!(
            SerializePreset::HumdrumKern.serialize_document(&slice),
            "**kern\t**kern\n\
             *clefG2\t*clefG2\n\
             *k[]\t*k[]\n\
             *^\t*\n\
             =\t=\t=\n\
             2D\t2a\t2d\n\
             *v\t*v\t*\n\
             ==\t==\n\
             *-\t*-\n\
             !!!RDF**kern: > = above\n\
             !!!RDF**kern: < = below\n\
             !!!RDF**kern: N = linked\n\
             !!!RDF**kern: Z = editorial accidental, parentheses\n\
             !!!RDF**kern: j = editorial accidental, brackets\n"
        );
    }

    #[test]
    fn slice_measure_range_rejects_empty_range() {
        let input = KernDocument::parse("**kern\n=1\n4c\n=\n*-\n").unwrap();

        assert_eq!(
            slice_measure_range(&input, 0..0),
            Err(SliceMeasuresError::EmptyRange)
        );
    }

    #[test]
    fn slice_measure_range_rejects_out_of_bounds_range() {
        let input = KernDocument::parse("**kern\n=1\n4c\n=\n*-\n").unwrap();

        assert_eq!(
            slice_measure_range(&input, 0..2),
            Err(SliceMeasuresError::RangeOutOfBounds {
                requested_start: 0,
                requested_end: 2,
                available: 1,
            })
        );
    }
}
