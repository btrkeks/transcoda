//! Contextual data-line durations for Humdrum `**kern` documents.
//!
//! Token durations are local, but data-line durations are contextual: null
//! tokens can carry a previously-started note while another spine moves.

use thiserror::Error;

use crate::{
    measure::MusicalDuration,
    meter::is_meter_tandem,
    score::{LayerId, LayerOrigin, Score, SplitSide, StateErrors},
    KernContent, KernContentDurationError, KernDocument, KernLine, SpineContent, SpineType,
};

use super::measure_spans::measure_spans;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataLineDuration {
    /// Zero-based line index in [`KernDocument::lines`].
    pub line_index: usize,
    /// One-based source line number.
    pub source_line: usize,
    /// Duration advanced by this data line, in whole-note units.
    pub duration: MusicalDuration,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DataLineDurationError {
    #[error("the score does not resolve: {0}")]
    State(StateErrors),
    #[error("line {line}, column {column}: kern spine contains non-kern data content")]
    KernSpineContainsNonKern { line: usize, column: usize },
    #[error("line {line}, column {column}: {source}")]
    TokenDuration {
        line: usize,
        column: usize,
        source: KernContentDurationError,
    },
    #[error(
        "line {line}, column {column}: duration-bearing token starts before previous token has finished; remaining {remaining}"
    )]
    OverlappingDuration {
        line: usize,
        column: usize,
        remaining: MusicalDuration,
    },
    #[error("line {line}, column {column}: multi-measure rest has no active meter")]
    MultiMeasureRestWithoutMeter { line: usize, column: usize },
    #[error(
        "line {line}, column {column}: internal duration state underflow subtracting {duration} from {remaining}"
    )]
    DurationUnderflow {
        line: usize,
        column: usize,
        duration: MusicalDuration,
        remaining: MusicalDuration,
    },
}

impl DataLineDurationError {
    pub fn line(&self) -> usize {
        match self {
            Self::State(errors) => errors.0[0].line(),
            Self::KernSpineContainsNonKern { line, .. }
            | Self::TokenDuration { line, .. }
            | Self::OverlappingDuration { line, .. }
            | Self::MultiMeasureRestWithoutMeter { line, .. }
            | Self::DurationUnderflow { line, .. } => *line,
        }
    }

    pub fn column(&self) -> Option<usize> {
        match self {
            Self::KernSpineContainsNonKern { column, .. }
            | Self::TokenDuration { column, .. }
            | Self::OverlappingDuration { column, .. }
            | Self::MultiMeasureRestWithoutMeter { column, .. }
            | Self::DurationUnderflow { column, .. } => Some(*column),
            Self::State(_) => None,
        }
    }
}

/// Return durations for every data line in a document, timed as an excerpt.
/// Timing reads the spine topology and each track's meter; a document whose
/// topology or meters do not resolve fails with
/// [`DataLineDurationError::State`].
///
/// Durations are normalized whole-note units. Non-data lines are omitted.
/// Grace-only and null-only data lines advance by zero.
pub fn data_line_durations(
    document: &KernDocument,
) -> Result<Vec<DataLineDuration>, DataLineDurationError> {
    let score = Score::resolve_timing(document).map_err(DataLineDurationError::State)?;
    score_line_durations(&score)
}

/// Durations for every data line of a resolved score: remaining durations
/// follow layer identity, and a multi-measure rest takes its track's meter.
pub fn score_line_durations(
    score: &Score<'_>,
) -> Result<Vec<DataLineDuration>, DataLineDurationError> {
    let document = score.document();
    let mut clock = ScoreClock::new(score);
    let mut durations = Vec::new();
    for line_index in 0..document.lines().len() {
        if let Some(duration) = clock.observe(line_index)? {
            durations.push(DataLineDuration {
                line_index,
                source_line: document.source_line_number(line_index),
                duration,
            });
        }
    }
    Ok(durations)
}

/// Where a score's spines stop agreeing on time. Verovio draws a spine's
/// tokens back to back: it either closes a silence, drawing later tokens
/// earlier, or reports the rhythm as inconsistent.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TimelineBreak {
    #[error(
        "line {line}, column {column}: this token starts after a silence in its spine, \
         which Verovio cannot draw"
    )]
    AfterSilence { line: usize, column: usize },
    #[error("line {line}, column {column}: a token still sounds {remaining} past this barline")]
    PastBarline {
        line: usize,
        column: usize,
        remaining: MusicalDuration,
    },
}

/// The first token, in document order, that starts after a silence in its
/// spine in the same bar, or that still sounds at a barline. A spine is
/// silent on a line that moves time on while nothing sounds in it. Nulls
/// before a spine's first token carry a token from an earlier page, and a
/// silence that lasts to the barline delays nothing, so neither counts.
pub fn first_timeline_break(
    score: &Score<'_>,
) -> Result<Option<TimelineBreak>, DataLineDurationError> {
    let document = score.document();
    let topology = score.topology();
    let mut clock = ScoreClock::new(score);
    let mut silent = vec![false; topology.layer_count()];
    for (line_index, line) in document.lines().iter().enumerate() {
        let line_number = document.source_line_number(line_index);
        let layers = topology.layers_at(line_index);
        let carries = layers
            .iter()
            .map(|layer| clock.carries[layer.index()])
            .collect::<Vec<_>>();
        if matches!(line, KernLine::Barline(_)) {
            if let Some((column, carry)) = carries
                .iter()
                .enumerate()
                .find(|(_, carry)| !carry.remaining.is_zero())
            {
                return Ok(Some(TimelineBreak::PastBarline {
                    line: line_number,
                    column: column + 1,
                    remaining: carry.remaining,
                }));
            }
            silent.fill(false);
        }
        let advance = clock.observe(line_index)?;
        let (KernLine::Data(contents), Some(advance)) = (line, advance) else {
            continue;
        };
        for (column, (content, (layer, carry))) in
            contents.iter().zip(layers.iter().zip(&carries)).enumerate()
        {
            let SpineContent::Kern(content) = content else {
                continue;
            };
            if !matches!(content, KernContent::Null | KernContent::Grace(_)) {
                if silent[layer.index()] {
                    return Ok(Some(TimelineBreak::AfterSilence {
                        line: line_number,
                        column: column + 1,
                    }));
                }
            } else if carry.started && carry.remaining.is_zero() && !advance.is_zero() {
                silent[layer.index()] = true;
            }
        }
    }
    Ok(None)
}

/// Line-by-line timing over a resolved score.
#[derive(Clone)]
pub(crate) struct ScoreClock<'s, 'd> {
    score: &'s Score<'d>,
    initial_meter: Option<MusicalDuration>,
    /// Indexed by [`LayerId`].
    carries: Vec<Carry>,
}

impl<'s, 'd> ScoreClock<'s, 'd> {
    pub(crate) fn new(score: &'s Score<'d>) -> Self {
        let initial_meter = infer_initial_meter(score.document(), {
            let mut clock = Self::with_initial_meter(score, None);
            move |line_index| clock.observe(line_index)
        });
        Self::with_initial_meter(score, initial_meter)
    }

    fn with_initial_meter(score: &'s Score<'d>, initial_meter: Option<MusicalDuration>) -> Self {
        Self {
            score,
            initial_meter,
            carries: vec![Carry::default(); score.topology().layer_count()],
        }
    }

    /// Observe one line, in document order. Returns the duration a data line
    /// advances by.
    pub(crate) fn observe(
        &mut self,
        line_index: usize,
    ) -> Result<Option<MusicalDuration>, DataLineDurationError> {
        match &self.score.document().lines()[line_index] {
            KernLine::Data(contents) => self.advance(line_index, contents).map(Some),
            KernLine::SpineManipulatorLine(_) => {
                self.carry_into_new_layers(line_index);
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Duration a layer's current token still sounds after the last observed
    /// line. Adding that line's advance recovers a newly started token's full
    /// span, including contextual multi-measure rests.
    pub(crate) fn remaining(&self, layer: LayerId) -> MusicalDuration {
        self.carries[layer.index()].remaining
    }

    /// Time data line `line_index` as if it held `contents`, which may
    /// differ from the score's tokens but not in width.
    pub(crate) fn advance(
        &mut self,
        line_index: usize,
        contents: &[SpineContent],
    ) -> Result<MusicalDuration, DataLineDurationError> {
        let topology = self.score.topology();
        let layers = topology.layers_at(line_index);
        let mut carries = layers
            .iter()
            .map(|layer| self.carries[layer.index()])
            .collect::<Vec<_>>();
        let mut fields = layers
            .iter()
            .zip(&mut carries)
            .enumerate()
            .map(|(column, (&layer, carry))| FieldClock {
                kern: *topology.spine_type(layer) == SpineType::Kern,
                meter: self
                    .score
                    .state_at(line_index, column)
                    .track
                    .meter
                    .measure_duration()
                    .or(self.initial_meter),
                carry,
            })
            .collect::<Vec<_>>();
        let duration = advance_data_line(
            &mut fields,
            contents,
            self.score.document().source_line_number(line_index),
        )?;
        for (layer, carry) in layers.iter().zip(carries) {
            self.carries[layer.index()] = carry;
        }
        Ok(duration)
    }

    fn carry_into_new_layers(&mut self, line_index: usize) {
        let topology = self.score.topology();
        let before = topology.layers_at(line_index);
        for &layer in topology.layers_after(line_index) {
            if before.contains(&layer) {
                continue;
            }
            self.carries[layer.index()] = match &topology.layer(layer).origin {
                LayerOrigin::SplitOf(parent) => {
                    let (left, right) = self.carries[parent.index()].split();
                    match topology.split_side(layer) {
                        Some(SplitSide::Right) => right,
                        _ => left,
                    }
                }
                LayerOrigin::MergeOf(inputs) => {
                    Carry::merge(inputs.iter().map(|input| self.carries[input.index()]))
                }
                LayerOrigin::Exclusive => Carry::default(),
            };
        }
    }
}

/// One field of a data line as the clock sees it.
struct FieldClock<'a> {
    kern: bool,
    /// Measure length for multi-measure rests.
    meter: Option<MusicalDuration>,
    carry: &'a mut Carry,
}

/// Start each field's token and advance by the shortest remaining duration
/// among `**kern` fields.
fn advance_data_line(
    fields: &mut [FieldClock<'_>],
    contents: &[SpineContent],
    source_line: usize,
) -> Result<MusicalDuration, DataLineDurationError> {
    let mut saw_duration_bearing_token = false;
    let mut saw_durationless_slice = false;

    for (column_index, (field, content)) in fields.iter_mut().zip(contents).enumerate() {
        if !field.kern {
            continue;
        }

        let location = LineDurationLocation {
            line: source_line,
            column: column_index + 1,
        };
        let SpineContent::Kern(content) = content else {
            return Err(DataLineDurationError::KernSpineContainsNonKern {
                line: location.line,
                column: location.column,
            });
        };
        if matches!(content, KernContent::InvisibleRestSlice(_)) {
            saw_durationless_slice = true;
        }
        let duration = content_duration(content, field.meter, location)?;
        if duration.is_zero() {
            continue;
        }

        saw_duration_bearing_token = true;
        if !field.carry.remaining.is_zero() {
            return Err(DataLineDurationError::OverlappingDuration {
                line: location.line,
                column: location.column,
                remaining: field.carry.remaining,
            });
        }
        *field.carry = Carry {
            remaining: duration,
            secondary_starts_clear_on_split: false,
            started: true,
        };
    }

    if !saw_duration_bearing_token && !saw_durationless_slice {
        return Ok(MusicalDuration::zero());
    }

    let duration = fields
        .iter()
        .filter(|field| field.kern)
        .filter_map(|field| (!field.carry.remaining.is_zero()).then_some(field.carry.remaining))
        .min()
        .unwrap_or_else(MusicalDuration::zero);

    if duration.is_zero() {
        return Ok(duration);
    }

    for (column_index, field) in fields.iter_mut().enumerate() {
        if !field.kern || field.carry.remaining.is_zero() {
            continue;
        }
        field.carry.remaining = field.carry.remaining.checked_sub(duration).ok_or(
            DataLineDurationError::DurationUnderflow {
                line: source_line,
                column: column_index + 1,
                duration,
                remaining: field.carry.remaining,
            },
        )?;
    }

    Ok(duration)
}

/// Remaining duration of the token sounding in one field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Carry {
    remaining: MusicalDuration,
    /// An unequal merge keeps its longest carry in the merged spine. If that
    /// topology immediately reopens, the new secondary voice did not own the
    /// carry and must start clear. A fresh event resets this marker.
    secondary_starts_clear_on_split: bool,
    /// Whether the field or a layer it came from started a token on this
    /// page. Before that, a null carries a token from an earlier page.
    started: bool,
}

impl Default for Carry {
    fn default() -> Self {
        Self {
            remaining: MusicalDuration::zero(),
            secondary_starts_clear_on_split: false,
            started: false,
        }
    }
}

impl Carry {
    /// The primary (left) and secondary (right) child of a split.
    fn split(self) -> (Self, Self) {
        let mut secondary = self;
        if self.secondary_starts_clear_on_split {
            secondary.remaining = MusicalDuration::zero();
        }
        let clear = |carry: Self| Self {
            secondary_starts_clear_on_split: false,
            ..carry
        };
        (clear(self), clear(secondary))
    }

    fn merge(inputs: impl IntoIterator<Item = Self>) -> Self {
        let mut inputs = inputs.into_iter();
        let first = inputs.next().expect("a merge has inputs");
        let mut merged = first;
        for carry in inputs {
            merged.secondary_starts_clear_on_split |=
                carry.secondary_starts_clear_on_split || carry.remaining != first.remaining;
            merged.remaining = merged.remaining.max(carry.remaining);
            merged.started |= carry.started;
        }
        merged
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LineDurationLocation {
    line: usize,
    column: usize,
}

/// The measure length shared by every complete measure, for documents that
/// declare no meter. `observe` times one line, in document order; lines with
/// multi-measure rests are skipped.
fn infer_initial_meter(
    document: &KernDocument,
    mut observe: impl FnMut(usize) -> Result<Option<MusicalDuration>, DataLineDurationError>,
) -> Option<MusicalDuration> {
    if document_has_explicit_meter(document) {
        return None;
    }

    let spans = measure_spans(document);
    if spans.is_empty() {
        return None;
    }

    let mut line_to_span = vec![None; document.lines().len()];
    let mut span_valid = vec![false; spans.len()];
    let mut span_durations = vec![MusicalDuration::zero(); spans.len()];

    for (span_index, span) in spans.iter().enumerate() {
        if span.leading_barline.is_none() || span.trailing_barline.is_none() {
            continue;
        }
        span_valid[span_index] = true;
        for line_index in span.content.clone() {
            if let Some(slot) = line_to_span.get_mut(line_index) {
                *slot = Some(span_index);
            }
        }
    }

    for (line_index, line) in document.lines().iter().enumerate() {
        let span_index = line_to_span[line_index];
        if line_has_multi_measure_rest(line) {
            if let Some(span_index) = span_index {
                span_valid[span_index] = false;
            }
            continue;
        }

        match observe(line_index) {
            Ok(Some(duration)) => {
                if let Some(span_index) = span_index {
                    span_durations[span_index] += duration;
                }
            }
            Ok(None) => {}
            Err(_) => return None,
        }
    }

    let mut inferred = None;
    for (valid, duration) in span_valid.into_iter().zip(span_durations) {
        if !valid || duration.is_zero() {
            continue;
        }
        match inferred {
            Some(existing) if existing != duration => return None,
            Some(_) => {}
            None => inferred = Some(duration),
        }
    }

    inferred
}

fn document_has_explicit_meter(document: &KernDocument) -> bool {
    document.lines().iter().any(|line| {
        matches!(
            line,
            KernLine::TandemInterpretationLine(tandems)
                if tandems.tokens().iter().any(is_meter_tandem)
        )
    })
}

fn line_has_multi_measure_rest(line: &KernLine) -> bool {
    matches!(
        line,
        KernLine::Data(contents)
            if contents.iter().any(|content| {
                matches!(content, SpineContent::Kern(KernContent::MultiMeasureRest(_)))
            })
    )
}

fn content_duration(
    content: &KernContent,
    meter: Option<MusicalDuration>,
    location: LineDurationLocation,
) -> Result<MusicalDuration, DataLineDurationError> {
    match content {
        KernContent::MultiMeasureRest(rest) => {
            let meter = meter.ok_or(DataLineDurationError::MultiMeasureRestWithoutMeter {
                line: location.line,
                column: location.column,
            })?;
            Ok(meter.multiplied_by(rest.measure_count))
        }
        _ => content
            .duration()
            .map_err(|source| DataLineDurationError::TokenDuration {
                line: location.line,
                column: location.column,
                source,
            }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{document::Framing, ParseDialect, ParseOptions, SerializePreset};

    fn durations(input: &str) -> Vec<MusicalDuration> {
        let document = KernDocument::parse(input).unwrap();
        data_line_durations(&document)
            .unwrap()
            .into_iter()
            .map(|line| line.duration)
            .collect()
    }

    fn score_error(input: &str) -> DataLineDurationError {
        let document = KernDocument::parse(input).unwrap();
        let score = Score::resolve(&document, Framing::Excerpt).unwrap();
        score_line_durations(&score).unwrap_err()
    }

    fn fractions(input: &str) -> Vec<(i64, i64)> {
        durations(input)
            .into_iter()
            .map(MusicalDuration::as_fraction)
            .collect()
    }

    fn transcoda_v2_fractions(input: &str) -> Vec<(i64, i64)> {
        let document = KernDocument::parse_with_options(
            input,
            ParseOptions {
                dialect: ParseDialect::TranscodaV2,
            },
        )
        .unwrap();
        data_line_durations(&document)
            .unwrap()
            .into_iter()
            .map(|line| line.duration.as_fraction())
            .collect()
    }

    #[test]
    fn computes_sample_line_duration_from_shortest_concurrent_token() {
        assert_eq!(
            fractions("**kern\t**kern\t**kern\t**kern\t**kern\n4FF D Bn:\t4ryy\t2.aa:\t4g bn:\t4ryy\n*-\t*-\t*-\t*-\t*-\n"),
            vec![(1, 4)]
        );
    }

    #[test]
    fn carries_null_tokens_for_split_voice_duration_frontiers() {
        assert_eq!(
            fractions(
                "**kern\n\
*M4/4\n\
*^\n\
2..gg\t4b\n\
.\t4a\n\
.\t4g\n\
.\t4f\n\
8ff\t.\n\
*v\t*v\n\
=\n\
*-\n"
            ),
            vec![(1, 4), (1, 4), (1, 4), (1, 8), (1, 8)]
        );
    }

    #[test]
    fn compatible_voice_merge_preserves_the_longest_remaining_duration() {
        let document = KernDocument::parse(
            "**kern\n\
             *^\n\
             2c\t4e\n\
             *v\t*v\n\
             *-\n",
        )
        .unwrap();
        let score = Score::resolve(&document, Framing::Excerpt).unwrap();
        let topology = score.topology();
        let mut clock = ScoreClock::new(&score);
        for line_index in 0..document.lines().len() {
            clock.observe(line_index).unwrap();
        }
        let merge = document
            .lines()
            .iter()
            .position(|line| matches!(line, KernLine::SpineManipulatorLine(_)))
            .map(|split| split + 2)
            .unwrap();
        let inputs = topology.layers_at(merge);
        assert_eq!(clock.remaining(inputs[0]).as_fraction(), (1, 4));
        assert_eq!(clock.remaining(inputs[1]).as_fraction(), (0, 1));
        let merged = topology.layers_after(merge)[0];
        assert_eq!(clock.remaining(merged).as_fraction(), (1, 4));
    }

    #[test]
    fn split_after_voice_merge_carries_duration_only_in_the_primary_child() {
        assert_eq!(
            fractions(
                "**kern\n\
                 *^\n\
                 2c\t4e\n\
                 *v\t*v\n\
                 *^\n\
                 .\t4g\n\
                 *-\t*-\n"
            ),
            vec![(1, 4), (1, 4)]
        );
    }

    #[test]
    fn fresh_duration_after_unequal_merge_restores_ordinary_split_carry() {
        let document = KernDocument::parse(
            "**kern\t**kern\n\
             *^\t*\n\
             2c\t4e\t4g\n\
             *v\t*v\t*\n\
             .\t4a\n\
             2b\t4c\n\
             *^\t*\n\
             .\t4d\t4e\n\
             *-\t*-\t*-\n",
        )
        .unwrap();

        let error = data_line_durations(&document).unwrap_err();

        assert!(matches!(
            error,
            DataLineDurationError::OverlappingDuration {
                line: 8,
                column: 2,
                ..
            }
        ));
    }

    #[test]
    fn score_multi_measure_rest_takes_the_track_meter_from_any_subspine() {
        let input = "**kern\n*^\n*M3/4\t*\n.\tMM2r\n=\t=\n*v\t*v\n*-\n";
        assert_eq!(fractions(input), vec![(3, 2)]);
    }

    #[test]
    fn score_merge_of_subspines_needs_no_equal_column_meters() {
        let input = "**kern\n*M4/4\n*^\n*M3/4\t*\n2.c\t2.e\n*v\t*v\n=\n*-\n";
        assert_eq!(fractions(input), vec![(3, 4)]);
        assert!(matches!(
            score_error("**kern\nMM2r\n*-\n"),
            DataLineDurationError::MultiMeasureRestWithoutMeter { .. }
        ));
    }

    #[test]
    fn timing_needs_the_topology_and_meters_only() {
        let document = KernDocument::parse("**kern\t**kern\n*v\t*v\n4c\n*-\n").unwrap();
        assert!(matches!(
            data_line_durations(&document).unwrap_err(),
            DataLineDurationError::State(_)
        ));
        let document =
            KernDocument::parse("**kern\n*^\n*M3/4\t*M2/4\nMM2r\t.\n*v\t*v\n*-\n").unwrap();
        assert!(matches!(
            data_line_durations(&document).unwrap_err(),
            DataLineDurationError::State(_)
        ));
        // An orphan ottava closer is a state error, but times like any line.
        assert_eq!(
            fractions("**kern\n4c\n*X8va\n4d\n*-\n"),
            vec![(1, 4), (1, 4)]
        );
    }

    #[test]
    fn durationless_invisible_rest_slice_borrows_active_remaining_duration() {
        assert_eq!(
            transcoda_v2_fractions("**kern\t**kern\n2c\t4d\n.\tryy\n*-\t*-\n"),
            vec![(1, 4), (1, 4)]
        );
    }

    #[test]
    fn grace_only_line_has_zero_duration() {
        assert_eq!(
            fractions("**kern\n*M4/4\n4c\n8qdd\n4d\n*-\n"),
            vec![(1, 4), (0, 1), (1, 4)]
        );
    }

    #[test]
    fn rejects_new_duration_before_carried_token_finishes() {
        let document = KernDocument::parse("**kern\t**kern\n2c\t4e\n4d\t4f\n*-\t*-\n").unwrap();
        let error = data_line_durations(&document).unwrap_err();

        assert!(error.to_string().contains("previous token has finished"));
    }

    #[test]
    fn compact_chord_duration_counts_once() {
        assert_eq!(fractions("**kern\n4FF D Bn:\n*-\n"), vec![(1, 4)]);
    }

    #[test]
    fn multi_measure_rest_uses_active_meter() {
        assert_eq!(fractions("**kern\n*M3/4\nMM2r\n=\n*-\n"), vec![(3, 2)]);
    }

    #[test]
    fn multi_measure_rest_uses_meter_implied_by_symbol() {
        assert_eq!(fractions("**kern\n*met(C)\nMM2r\n=\n*-\n"), vec![(2, 1)]);
    }

    #[test]
    fn multi_measure_rest_uses_inferred_omitted_initial_meter() {
        assert_eq!(
            fractions(
                "**kern\n\
*clefG2\n\
MM3r\n\
=\n\
4c\n\
4d\n\
4e\n\
=\n\
*-\n"
            ),
            vec![(9, 4), (1, 4), (1, 4), (1, 4)]
        );
    }

    #[test]
    fn inferred_omitted_meter_does_not_serialize_as_visible_meter() {
        let document = KernDocument::parse(
            "**kern\n\
*clefG2\n\
MM3r\n\
=\n\
4c\n\
4d\n\
4e\n\
=\n\
*-\n",
        )
        .unwrap();

        let serialized = SerializePreset::TranscodaV2.serialize_document(&document);

        assert!(serialized.contains("MM3r"));
        assert!(!serialized.contains("*M"));
    }

    #[test]
    fn omitted_meter_inference_requires_unambiguous_complete_measures() {
        let document = KernDocument::parse(
            "**kern\n\
MM2r\n\
=\n\
4c\n\
4d\n\
=\n\
4e\n\
4f\n\
4g\n\
=\n\
*-\n",
        )
        .unwrap();
        let error = data_line_durations(&document).unwrap_err();

        assert!(matches!(
            error,
            DataLineDurationError::MultiMeasureRestWithoutMeter { .. }
        ));
    }

    #[test]
    fn multi_measure_rest_requires_active_meter() {
        let document = KernDocument::parse("**kern\nMM2r\n*-\n").unwrap();
        let error = data_line_durations(&document).unwrap_err();

        assert!(matches!(
            error,
            DataLineDurationError::MultiMeasureRestWithoutMeter { .. }
        ));
    }
}
