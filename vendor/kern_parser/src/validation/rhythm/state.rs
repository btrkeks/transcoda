use std::collections::BTreeSet;

use super::{
    boundary::{boundary_role, BoundaryRole},
    measure_error_at, measure_error_value, RhythmValidationError,
};
use crate::{
    measure::{kern_content_duration, MeasureDuration, MeasureLocation},
    meter::is_meter_tandem,
    score::{LayerId, LayerOrigin, Score},
    BarLength, KernContent, KernLine, MultiMeasureRest, SpineContent, SpineManipulator,
    SpineManipulatorLine, SpineType, TandemInterpretationLine,
};

/// Measure position of one layer. The bar length comes from its track's
/// meter in the score.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LayerMeasure {
    elapsed: MeasureDuration,
    pickup: Option<MeasureDuration>,
    pending_repeat_entry: bool,
    repeat_entry: Option<RepeatEntryMeasure>,
    has_completed_measure: bool,
    saw_measure_content: bool,
}

impl Default for LayerMeasure {
    fn default() -> Self {
        Self {
            elapsed: MeasureDuration::zero(),
            pickup: None,
            pending_repeat_entry: false,
            repeat_entry: None,
            has_completed_measure: false,
            saw_measure_content: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RepeatEntryMeasure {
    meter: MeasureDuration,
    partial: MeasureDuration,
}

/// One `**kern` layer at a line: its column, measure and bar length.
#[derive(Debug, Clone, Copy)]
struct Voice<'a> {
    column: usize,
    measure: &'a LayerMeasure,
    meter: BarLength,
}

pub(super) struct MeasureStateTracker<'s, 'd> {
    score: &'s Score<'d>,
    /// Bar length before any track declares a meter.
    initial_meter: BarLength,
    /// Indexed by [`LayerId`].
    layers: Vec<LayerMeasure>,
    saw_measure_boundary: bool,
    /// Line indexes of barlines read as dividing a bar.
    dividing_barlines: BTreeSet<usize>,
}

impl<'s, 'd> MeasureStateTracker<'s, 'd> {
    pub(super) fn new(score: &'s Score<'d>, initial_meter: BarLength) -> Self {
        Self {
            score,
            initial_meter,
            layers: vec![LayerMeasure::default(); score.topology().layer_count()],
            saw_measure_boundary: false,
            dividing_barlines: BTreeSet::new(),
        }
    }

    /// Barlines read so far as dividing a bar rather than ending it: a dashed
    /// barline or a system-break `=-` reached before the bar is full.
    pub(super) fn dividing_barlines(&self) -> &BTreeSet<usize> {
        &self.dividing_barlines
    }

    /// Apply line `line_index` of `lines`. `lines` are the score's lines, or a
    /// copy that differs only in data tokens.
    pub(super) fn apply_line(
        &mut self,
        lines: &[KernLine],
        line_index: usize,
    ) -> Result<(), RhythmValidationError> {
        let source_line = self.score.document().source_line_number(line_index);
        match &lines[line_index] {
            KernLine::TandemInterpretationLine(tandems) => {
                self.apply_tandem_interpretation_line(line_index, tandems, source_line)
            }
            KernLine::SpineManipulatorLine(manipulators) => {
                self.apply_manipulator_line(line_index, manipulators, source_line)
            }
            KernLine::Barline(barline) => {
                let no_following_data = !lines[line_index + 1..]
                    .iter()
                    .any(|line| matches!(line, KernLine::Data(_)));
                self.apply_boundary_line(
                    line_index,
                    boundary_role(barline, no_following_data),
                    source_line,
                )
            }
            KernLine::Data(contents) => self.apply_data_line(line_index, contents, source_line),
            KernLine::SpineTypeLine(_) | KernLine::LocalCommentLine(_) => Ok(()),
        }
    }

    pub(super) fn finish_document(&self, source_line: usize) -> Result<(), RhythmValidationError> {
        if !self.saw_measure_boundary {
            return Ok(());
        }
        let topology = self.score.topology();
        let voices = self.voices(topology.layers_after_last(), |column| {
            self.bar_length(self.score.state_after_last(column))
        });
        validate_measure_end(&voices, source_line)
    }

    fn bar_length(&self, state: crate::score::FieldState<'_>) -> BarLength {
        state.track.meter.bar_length().unwrap_or(self.initial_meter)
    }

    fn meter_at(&self, line_index: usize, column: usize) -> BarLength {
        self.bar_length(self.score.state_at(line_index, column))
    }

    fn is_kern(&self, layer: LayerId) -> bool {
        *self.score.topology().spine_type(layer) == SpineType::Kern
    }

    /// The `**kern` layers among `layers`, with the bar length of each column.
    fn voices(&self, layers: &[LayerId], meter: impl Fn(usize) -> BarLength) -> Vec<Voice<'_>> {
        layers
            .iter()
            .enumerate()
            .filter(|(_, &layer)| self.is_kern(layer))
            .map(|(column, &layer)| Voice {
                column,
                measure: &self.layers[layer.index()],
                meter: meter(column),
            })
            .collect()
    }

    /// A meter change applies to every layer of its track; each must stand
    /// at a measure boundary.
    fn apply_tandem_interpretation_line(
        &mut self,
        line_index: usize,
        tandems: &TandemInterpretationLine,
        source_line: usize,
    ) -> Result<(), RhythmValidationError> {
        let topology = self.score.topology();
        let mut changed = Vec::new();
        for (column, tandem) in tandems.tokens().iter().enumerate() {
            let track = topology.track_of(topology.layer_at(line_index, column));
            if !is_meter_tandem(tandem) || changed.contains(&track) {
                continue;
            }
            changed.push(track);
            let declaration = &self.score.state_after(line_index, column).track.meter;
            if let Some(conflict) = declaration.conflict() {
                return measure_error_at(
                    MeasureLocation {
                        source_line,
                        column: Some(column + 1),
                    },
                    "rhythm.conflicting-meter",
                    format!(
                        "time signature {}/{} conflicts with meter symbol implying {}/{}",
                        conflict.time_signature.numerator,
                        conflict.time_signature.denominator,
                        conflict.symbol_time_signature.numerator,
                        conflict.symbol_time_signature.denominator
                    ),
                );
            }
            // A group that only hides the meter, with no meter before it,
            // leaves the meter in force unchanged.
            let Some(meter) = declaration.bar_length() else {
                continue;
            };
            for layer_column in topology.track_columns(line_index, track) {
                let layer = topology.layer_at(line_index, layer_column);
                let before = self.meter_at(line_index, layer_column);
                let kern = self.is_kern(layer);
                let measure = &mut self.layers[layer.index()];
                if kern && !measure.elapsed.is_zero() && !is_complete_measure(measure, before) {
                    return measure_error_at(
                        MeasureLocation {
                            source_line,
                            column: Some(layer_column + 1),
                        },
                        "rhythm.meter-change-mid-measure",
                        format!(
                            "meter changes before current measure is complete; elapsed {}, expected {}",
                            measure.elapsed.display(),
                            display_bar_length(before)
                        ),
                    );
                }
                if !is_complete_measure(measure, before) || !is_complete_measure(measure, meter) {
                    measure.elapsed = MeasureDuration::zero();
                }
            }
        }

        Ok(())
    }

    /// A terminated layer must end on a measure boundary; a split copies the
    /// measure position into both children; merged layers must agree.
    fn apply_manipulator_line(
        &mut self,
        line_index: usize,
        manipulators: &SpineManipulatorLine,
        source_line: usize,
    ) -> Result<(), RhythmValidationError> {
        let topology = self.score.topology();
        let before = topology.layers_at(line_index);
        let mut merged = Vec::new();
        for &layer in topology.layers_after(line_index) {
            if before.contains(&layer) {
                continue;
            }
            self.layers[layer.index()] = match &topology.layer(layer).origin {
                LayerOrigin::SplitOf(parent) => self.layers[parent.index()].clone(),
                LayerOrigin::MergeOf(inputs) => {
                    merged.push((layer, inputs.as_slice()));
                    continue;
                }
                LayerOrigin::Exclusive => LayerMeasure::default(),
            };
        }

        for (column, (&layer, manipulator)) in before.iter().zip(manipulators.tokens()).enumerate()
        {
            let location = MeasureLocation {
                source_line,
                column: Some(column + 1),
            };
            if *manipulator == SpineManipulator::Terminate
                && self.saw_measure_boundary
                && self.is_kern(layer)
            {
                validate_spine_termination(
                    &self.layers[layer.index()],
                    self.meter_at(line_index, column),
                    location,
                )?;
            }
            if let Some(&(output, inputs)) = merged.iter().find(|(_, inputs)| inputs[0] == layer) {
                let measures = inputs
                    .iter()
                    .map(|input| &self.layers[input.index()])
                    .collect::<Vec<_>>();
                let measure = merge_measures(&measures, self.is_kern(layer), location)?;
                self.layers[output.index()] = measure;
            }
        }

        Ok(())
    }

    fn apply_boundary_line(
        &mut self,
        line_index: usize,
        boundary: BoundaryRole,
        source_line: usize,
    ) -> Result<(), RhythmValidationError> {
        self.saw_measure_boundary = true;
        let layers = self.score.topology().layers_at(line_index);
        let voices = self.voices(layers, |column| self.meter_at(line_index, column));
        if (boundary.marked_invisible_system_break || boundary.dashed)
            && voices
                .iter()
                .any(|voice| is_strictly_partial(voice.measure, voice.meter))
        {
            self.dividing_barlines.insert(line_index);
            return Ok(());
        }

        validate_free_measure_lengths(&voices, source_line)?;
        validate_pickup_consistency_at_boundary(&voices, boundary, source_line)?;
        let voices = voices
            .iter()
            .map(|voice| (voice.column, voice.meter))
            .collect::<Vec<_>>();
        for (column, meter) in voices {
            validate_measure_boundary(
                &mut self.layers[layers[column].index()],
                meter,
                MeasureLocation {
                    source_line,
                    column: Some(column + 1),
                },
                boundary,
            )?;
        }

        Ok(())
    }

    fn apply_data_line(
        &mut self,
        line_index: usize,
        contents: &[SpineContent],
        source_line: usize,
    ) -> Result<(), RhythmValidationError> {
        let layers = self.score.topology().layers_at(line_index);
        for (column, (&layer, content)) in layers.iter().zip(contents).enumerate() {
            if !self.is_kern(layer) {
                continue;
            }
            let location = MeasureLocation {
                source_line,
                column: Some(column + 1),
            };
            let SpineContent::Kern(content) = content else {
                return measure_error_at(
                    location,
                    "rhythm.non-kern-content",
                    "kern spine contains non-kern data content",
                );
            };
            let meter = self.meter_at(line_index, column);
            let measure = &mut self.layers[layer.index()];
            if let KernContent::MultiMeasureRest(rest) = content {
                validate_multi_measure_rest(measure, meter, rest, location)?;
                continue;
            }

            let duration = kern_content_duration(content).map_err(|error| {
                measure_error_value(location, "rhythm.invalid-token-duration", error.to_string())
            })?;
            if duration.is_zero() {
                continue;
            }
            measure.elapsed += duration;
            measure.saw_measure_content = true;
            if let BarLength::Fixed(meter) = meter {
                if measure.elapsed > meter {
                    return measure_error_at(
                        location,
                        "rhythm.overfull-measure",
                        format!(
                            "measure duration exceeds meter; elapsed {}, expected {}",
                            measure.elapsed.display(),
                            meter.display()
                        ),
                    );
                }
            }
        }

        Ok(())
    }
}

fn validate_pickup_consistency_at_boundary(
    voices: &[Voice<'_>],
    boundary: BoundaryRole,
    source_line: usize,
) -> Result<(), RhythmValidationError> {
    let mut pickup = None;

    for voice in voices {
        if !is_initial_pickup_candidate(voice, boundary) {
            continue;
        }

        let location = MeasureLocation {
            source_line,
            column: Some(voice.column + 1),
        };
        match pickup {
            Some(expected) if voice.measure.elapsed != expected => {
                return measure_error_at(
                    location, "rhythm.inconsistent-pickup",
                    format!(
                        "pickup duration is inconsistent across kern spines; elapsed {}, expected {}",
                        voice.measure.elapsed.display(),
                        expected.display()
                    ),
                );
            }
            Some(_) => {}
            None => pickup = Some(voice.measure.elapsed),
        }
    }

    let Some(expected) = pickup else {
        return Ok(());
    };

    for voice in voices {
        if (!voice.measure.saw_measure_content && voice.measure.elapsed.is_zero())
            || is_initial_pickup_candidate(voice, boundary)
        {
            continue;
        }

        return measure_error_at(
            MeasureLocation {
                source_line,
                column: Some(voice.column + 1),
            },
            "rhythm.inconsistent-pickup",
            format!(
                "pickup duration is inconsistent across kern spines; elapsed {}, expected {}",
                voice.measure.elapsed.display(),
                expected.display()
            ),
        );
    }

    Ok(())
}

fn is_initial_pickup_candidate(voice: &Voice<'_>, boundary: BoundaryRole) -> bool {
    !boundary.final_like
        && !boundary.repeat_end
        && !voice.measure.pending_repeat_entry
        && !voice.measure.has_completed_measure
        && voice.measure.saw_measure_content
        && is_strictly_partial(voice.measure, voice.meter)
}

fn is_strictly_partial(measure: &LayerMeasure, meter: BarLength) -> bool {
    !measure.elapsed.is_zero()
        && matches!(meter, BarLength::Fixed(meter) if measure.elapsed < meter)
}

/// Whether the layer's current measure is exactly full. A free-meter (`*MX`)
/// measure is never partial, so it is complete once it has content.
fn is_complete_measure(measure: &LayerMeasure, meter: BarLength) -> bool {
    match meter {
        BarLength::Fixed(meter) => measure.elapsed == meter,
        BarLength::Free => !measure.elapsed.is_zero(),
    }
}

fn display_bar_length(meter: BarLength) -> String {
    match meter {
        BarLength::Fixed(meter) => meter.display(),
        BarLength::Free => "free (*MX)".to_owned(),
    }
}

/// A free-meter (`*MX`) bar may have any length, but every voice in it must
/// have the same length.
fn validate_free_measure_lengths(
    voices: &[Voice<'_>],
    source_line: usize,
) -> Result<(), RhythmValidationError> {
    if !voices.iter().any(|voice| voice.meter == BarLength::Free) {
        return Ok(());
    }

    let mut expected = None;
    for voice in voices {
        if voice.measure.elapsed.is_zero() {
            continue;
        }
        match expected {
            Some(expected) if voice.measure.elapsed != expected => {
                return measure_error_at(
                    MeasureLocation {
                        source_line,
                        column: Some(voice.column + 1),
                    },
                    "rhythm.unequal-free-measure",
                    format!(
                        "voices in a free-meter (*MX) measure differ in length; elapsed {}, expected {}",
                        voice.measure.elapsed.display(),
                        expected.display()
                    ),
                );
            }
            Some(_) => {}
            None => expected = Some(voice.measure.elapsed),
        }
    }

    Ok(())
}

/// Layers merging into one must stand at the same measure position and
/// repeat state. `location` is the first merged column.
fn merge_measures(
    measures: &[&LayerMeasure],
    kern: bool,
    location: MeasureLocation,
) -> Result<LayerMeasure, RhythmValidationError> {
    let first = measures[0];
    if !kern {
        return Ok(first.clone());
    }

    for (offset, measure) in measures.iter().enumerate().skip(1) {
        if measure.elapsed != first.elapsed {
            return measure_error_at(
                location,
                "rhythm.different-duration-merge",
                unequal_merge_elapsed_message(
                    location,
                    measures.len(),
                    offset,
                    first.elapsed,
                    measure.elapsed,
                ),
            );
        }
        if measure.pending_repeat_entry != first.pending_repeat_entry
            || measure.repeat_entry != first.repeat_entry
        {
            return measure_error_at(
                location,
                "rhythm.different-repeat-state-merge",
                "cannot merge spines with different repeat measure state",
            );
        }
    }

    Ok(first.clone())
}

fn unequal_merge_elapsed_message(
    location: MeasureLocation,
    merge_count: usize,
    offset: usize,
    first_elapsed: MeasureDuration,
    found_elapsed: MeasureDuration,
) -> String {
    let Some(first_column) = location.column else {
        return format!(
            "cannot merge spines at different measure positions; first elapsed {}, found {}",
            first_elapsed.display(),
            found_elapsed.display()
        );
    };
    let found_column = first_column + offset;
    let last_column = first_column + merge_count - 1;
    let delta = merge_elapsed_delta(first_elapsed, found_elapsed);

    format!(
        "cannot merge spines at different measure positions: merge columns {first_column}-{last_column}; column {first_column} elapsed {}, column {found_column} elapsed {}; column {found_column} {delta}",
        first_elapsed.display(),
        found_elapsed.display()
    )
}

fn merge_elapsed_delta(first_elapsed: MeasureDuration, found_elapsed: MeasureDuration) -> String {
    if let Some(delta) = first_elapsed.checked_sub(found_elapsed) {
        format!("is short by {}", delta.display())
    } else if let Some(delta) = found_elapsed.checked_sub(first_elapsed) {
        format!("is ahead by {}", delta.display())
    } else {
        "differs".to_owned()
    }
}

fn validate_measure_boundary(
    spine: &mut LayerMeasure,
    meter: BarLength,
    location: MeasureLocation,
    boundary: BoundaryRole,
) -> Result<(), RhythmValidationError> {
    let BarLength::Fixed(meter) = meter else {
        // A free-meter (`*MX`) measure has no length to check; equal voice
        // lengths are checked across spines before this.
        spine.pending_repeat_entry = false;
        spine.repeat_entry = None;
        if !spine.elapsed.is_zero() {
            finish_complete_measure(spine);
        }
        finish_boundary(spine, boundary);
        return Ok(());
    };

    let was_pending_repeat_entry = spine.pending_repeat_entry;
    if was_pending_repeat_entry {
        record_repeat_entry(spine, meter);
    }

    if boundary.repeat_end {
        validate_repeat_end_measure(spine, meter, location)?;
    } else if spine.elapsed.is_zero() {
        // Empty measures at a boundary do not affect duration state.
    } else if spine.elapsed == meter {
        finish_complete_measure(spine);
    } else if was_pending_repeat_entry && spine.elapsed < meter {
        reset_measure_position(spine);
    } else if !boundary.final_like
        && !spine.has_completed_measure
        && spine.saw_measure_content
        && spine.elapsed < meter
    {
        if spine.pickup.is_none() {
            spine.pickup = Some(spine.elapsed);
        }
        reset_measure_position(spine);
    } else if boundary.final_like && spine.elapsed < meter {
        reset_measure_position(spine);
    } else {
        let mut message = format!(
            "measure duration does not match meter; elapsed {}, expected {}",
            spine.elapsed.display(),
            meter.display()
        );
        if spine.elapsed < meter {
            if boundary.invisible && !boundary.marked_invisible_system_break {
                message.push_str(
                    "; if a system break was intended, add `!!LO:LB:g=original` before the `=-`",
                );
            } else {
                message.push_str(
                    "; if the rest of this measure continues past a section boundary \
                     (for example split across a system or page break at a key change), \
                     mark the barline as a section boundary with `=||`",
                );
            }
        }
        return measure_error_at(location, "rhythm.incomplete-measure", message);
    }

    finish_boundary(spine, boundary);
    Ok(())
}

fn finish_boundary(spine: &mut LayerMeasure, boundary: BoundaryRole) {
    if boundary.final_like {
        reset_section_state(spine);
    }

    // A repeat end continuing into more music starts a new section that may
    // begin with a full measure or its own pickup, so its entry is observed
    // the same way as an explicit repeat start.
    if boundary.repeat_start || boundary.repeat_end_linear_continuation {
        spine.pending_repeat_entry = true;
        spine.repeat_entry = None;
    }
}

fn record_repeat_entry(spine: &mut LayerMeasure, meter: MeasureDuration) {
    let partial = if spine.elapsed == meter {
        MeasureDuration::zero()
    } else {
        spine.elapsed
    };
    spine.repeat_entry = Some(RepeatEntryMeasure { meter, partial });
    spine.pending_repeat_entry = false;
}

fn validate_repeat_end_measure(
    spine: &mut LayerMeasure,
    meter: MeasureDuration,
    location: MeasureLocation,
) -> Result<(), RhythmValidationError> {
    if spine.elapsed.is_zero() {
        return Ok(());
    }

    let repeat_entry = spine.repeat_entry.or_else(|| {
        spine.pickup.map(|pickup| RepeatEntryMeasure {
            meter,
            partial: pickup,
        })
    });

    if spine.elapsed < meter {
        if let Some(repeat_entry) = repeat_entry {
            if repeat_entry.meter == meter
                && !repeat_entry.partial.is_zero()
                && repeat_entry.partial + spine.elapsed == meter
            {
                finish_complete_measure(spine);
                return Ok(());
            }
        }

        reset_measure_position(spine);
        return Ok(());
    }

    let Some(repeat_entry) = repeat_entry else {
        if spine.elapsed == meter {
            finish_complete_measure(spine);
            return Ok(());
        }

        return measure_error_at(
            location,
            "rhythm.incomplete-repeat-ending",
            format!(
                "repeat ending measure does not match meter; elapsed {}, expected {}",
                spine.elapsed.display(),
                meter.display()
            ),
        );
    };

    // A section entry without a partial measure only pins where measures
    // start, so a later meter change inside the section does not constrain
    // the ending measure beyond filling the current meter.
    if repeat_entry.partial.is_zero() {
        if spine.elapsed == meter {
            finish_complete_measure(spine);
            return Ok(());
        }

        return measure_error_at(
            location,
            "rhythm.incomplete-repeat-ending",
            format!(
                "repeat ending measure does not match meter; elapsed {}, expected {}",
                spine.elapsed.display(),
                meter.display()
            ),
        );
    }

    if repeat_entry.meter != meter {
        return measure_error_at(
            location,
            "rhythm.repeat-meter-mismatch",
            format!(
                "repeat entry meter {} does not match ending meter {}",
                repeat_entry.meter.display(),
                meter.display()
            ),
        );
    }

    if repeat_entry.partial + spine.elapsed == meter {
        finish_complete_measure(spine);
        return Ok(());
    }

    measure_error_at(
        location, "rhythm.noncomplementary-repeat-ending",
        format!(
            "repeat ending measure does not complement repeat entry; entry {}, ending {}, expected {}",
            repeat_entry.partial.display(),
            spine.elapsed.display(),
            meter.display()
        ),
    )
}

fn finish_complete_measure(spine: &mut LayerMeasure) {
    spine.elapsed = MeasureDuration::zero();
    spine.has_completed_measure = true;
    spine.saw_measure_content = false;
}

fn reset_measure_position(spine: &mut LayerMeasure) {
    spine.elapsed = MeasureDuration::zero();
    spine.saw_measure_content = false;
}

fn reset_section_state(spine: &mut LayerMeasure) {
    reset_measure_position(spine);
    spine.pickup = None;
    spine.pending_repeat_entry = false;
    spine.repeat_entry = None;
    spine.has_completed_measure = false;
}

fn validate_multi_measure_rest(
    spine: &mut LayerMeasure,
    meter: BarLength,
    rest: &MultiMeasureRest,
    location: MeasureLocation,
) -> Result<(), RhythmValidationError> {
    if !spine.elapsed.is_zero() {
        return measure_error_at(
            location,
            "rhythm.multirest-mid-measure",
            format!(
                "multi-measure rest must start at a measure boundary; elapsed {}, expected 0",
                spine.elapsed.display()
            ),
        );
    }

    let BarLength::Fixed(meter) = meter else {
        return measure_error_at(
            location,
            "rhythm.multirest-free-meter",
            "multi-measure rest has no measure length under free meter (*MX)",
        );
    };
    spine.elapsed = meter;
    spine.has_completed_measure = true;
    spine.saw_measure_content = false;
    let _ = rest.measure_count;
    Ok(())
}

fn validate_measure_end(
    voices: &[Voice<'_>],
    source_line: usize,
) -> Result<(), RhythmValidationError> {
    for voice in voices {
        if voice.measure.elapsed.is_zero()
            || is_complete_measure(voice.measure, voice.meter)
            || final_measure_complements_pickup(voice.measure, voice.meter)
        {
            continue;
        }

        return measure_error_at(
            MeasureLocation {
                source_line,
                column: Some(voice.column + 1),
            },
            "rhythm.incomplete-final-measure",
            format!(
                "document ends with incomplete measure; elapsed {}, expected {}",
                voice.measure.elapsed.display(),
                display_bar_length(voice.meter)
            ),
        );
    }

    Ok(())
}

fn validate_spine_termination(
    measure: &LayerMeasure,
    meter: BarLength,
    location: MeasureLocation,
) -> Result<(), RhythmValidationError> {
    if measure.elapsed.is_zero()
        || is_complete_measure(measure, meter)
        || final_measure_complements_pickup(measure, meter)
    {
        return Ok(());
    }

    measure_error_at(
        location,
        "rhythm.incomplete-terminated-measure",
        format!(
            "spine terminates with incomplete measure; elapsed {}, expected {}",
            measure.elapsed.display(),
            display_bar_length(meter)
        ),
    )
}

fn final_measure_complements_pickup(measure: &LayerMeasure, meter: BarLength) -> bool {
    measure
        .pickup
        .is_some_and(|pickup| BarLength::Fixed(pickup + measure.elapsed) == meter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{document::Framing, Barline, KernDocument, TimeSignature};

    fn four_four() -> MeasureDuration {
        MeasureDuration::from_time_signature(&TimeSignature {
            numerator: 4,
            denominator: 4,
        })
    }

    fn initial_meter() -> BarLength {
        BarLength::Fixed(four_four())
    }

    fn apply_document(input: &str) -> Result<(), RhythmValidationError> {
        let document = KernDocument::parse(input).unwrap();
        let score = Score::resolve(&document, Framing::Excerpt).unwrap();
        let mut tracker = MeasureStateTracker::new(&score, initial_meter());
        for line_index in 0..document.lines().len() {
            tracker.apply_line(document.lines(), line_index)?;
        }
        tracker.finish_document(document.lines().len() + 1)
    }

    #[test]
    fn accepts_underfull_back_repeat() {
        apply_document("**kern\n*M4/4\n1c\n=\n4d\n4e\n=:|!\n4f\n4g\n4a\n4b\n=\n*-\n").unwrap();
    }

    #[test]
    fn rejects_ordinary_underfull_internal_measure() {
        let error =
            apply_document("**kern\n*M4/4\n1c\n=\n4d\n4e\n=\n4f\n4g\n4a\n4b\n=\n*-\n").unwrap_err();

        assert!(error
            .to_string()
            .contains("measure duration does not match meter"));
    }

    #[test]
    fn split_and_merge_preserve_equal_measure_position() {
        apply_document("**kern\n*M4/4\n4c\n*^\n4d\t4e\n*v\t*v\n2f\n=\n*-\n").unwrap();
    }

    #[test]
    fn rejects_merge_at_unequal_measure_position() {
        let error = apply_document("**kern\n*M4/4\n*^\n4c\t2e\n*v\t*v\n2f\n=\n*-\n").unwrap_err();
        assert_eq!(error.code(), "rhythm.different-duration-merge");
        assert!(error.to_string().contains(
            "line 5, column 1: cannot merge spines at different measure positions: merge columns 1-2; column 1 elapsed 1/4, column 2 elapsed 1/2; column 2 is ahead by 1/4"
        ));
    }

    #[test]
    fn meter_written_in_one_subspine_governs_its_siblings() {
        apply_document("**kern\n*M4/4\n*^\n*M3/4\t*\n2.c\t2.e\n=\t=\n*v\t*v\n*-\n").unwrap();
        let error =
            apply_document("**kern\n*M4/4\n*^\n1c\t2e\n*M3/4\t*\n*v\t*v\n*-\n").unwrap_err();
        assert_eq!(error.code(), "rhythm.meter-change-mid-measure");
        assert_eq!(error.column(), Some(2));
    }

    #[test]
    fn final_like_boundary_resets_repeat_state() {
        let mut measure = LayerMeasure {
            pickup: Some(MeasureDuration::from_time_signature(&TimeSignature {
                numerator: 1,
                denominator: 4,
            })),
            pending_repeat_entry: true,
            repeat_entry: Some(RepeatEntryMeasure {
                meter: four_four(),
                partial: MeasureDuration::zero(),
            }),
            has_completed_measure: true,
            ..LayerMeasure::default()
        };
        validate_measure_boundary(
            &mut measure,
            initial_meter(),
            MeasureLocation {
                source_line: 1,
                column: Some(1),
            },
            boundary_role(&Barline::try_from("==").unwrap(), false),
        )
        .unwrap();

        assert!(measure.pickup.is_none());
        assert!(!measure.pending_repeat_entry);
        assert!(measure.repeat_entry.is_none());
        assert!(!measure.has_completed_measure);
    }

    #[test]
    fn barline_role_marks_absence_of_following_data() {
        for (input, rejected) in [
            // Final: an underfull last measure is accepted.
            ("**kern\n*M4/4\n1c\n=\n2d\n=\n*-\n", false),
            // Followed by data: the same measure is incomplete.
            ("**kern\n*M4/4\n1c\n=\n2d\n=\n1e\n=\n*-\n", true),
        ] {
            assert_eq!(apply_document(input).is_err(), rejected, "{input}");
        }
    }
}
