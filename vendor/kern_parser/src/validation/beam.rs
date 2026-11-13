use crate::{
    score::{LayerId, LayerOrigin, Score, SplitSide, Topology},
    types::ChordBeams,
    Duration, GraceChord, KernContent, KernLine, SpineContent, SpineManipulator, SpineType,
};

use thiserror::Error;

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum BeamValidationError {
    #[error("invalid beam structure: {message}")]
    InvalidStructure {
        code: &'static str,
        line: usize,
        column: Option<usize>,
        message: String,
    },
}

impl BeamValidationError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidStructure { code, .. } => code,
        }
    }

    pub fn line(&self) -> usize {
        match self {
            Self::InvalidStructure { line, .. } => *line,
        }
    }

    pub fn column(&self) -> Option<usize> {
        match self {
            Self::InvalidStructure { column, .. } => *column,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct BeamTracks {
    regular: BeamCounter,
    grace: BeamCounter,
}

impl BeamTracks {
    fn first_open(self) -> Option<(BeamStream, u8)> {
        if self.regular.active_full_beams != 0 {
            return Some((BeamStream::Regular, self.regular.active_full_beams));
        }
        if self.grace.active_full_beams != 0 {
            return Some((BeamStream::Grace, self.grace.active_full_beams));
        }
        None
    }

    fn is_open(self) -> bool {
        self.first_open().is_some()
    }

    fn ensure_grace_closed_before_regular_content(
        self,
        location: BeamLocation,
    ) -> Result<(), BeamValidationError> {
        if self.grace.active_full_beams != 0 {
            return beam_error_at(
                location,
                "beam.grace-crosses-regular-content",
                beam_crossing_message(
                    BeamStream::Grace,
                    "regular content",
                    self.grace.active_full_beams,
                ),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct BeamCounter {
    active_full_beams: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BeamStream {
    Regular,
    Grace,
}

impl BeamStream {
    fn label(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Grace => "grace",
        }
    }
}

/// Check beam balance per layer. Beam depth comes from the drawn duration,
/// so `*rscale` changes which notes take beams.
pub fn validate(score: &Score<'_>) -> Result<(), BeamValidationError> {
    let document = score.document();
    let topology = score.topology();
    let drawn = score.drawn();
    let mut beams = vec![BeamTracks::default(); topology.layer_count()];

    for (line_index, line) in document.lines().iter().enumerate() {
        let source_line = document.source_line_number(line_index);
        let layers = topology.layers_at(line_index);
        match line {
            KernLine::SpineTypeLine(_) => {
                if let Some(previous) = line_index.checked_sub(1) {
                    ensure_all_beams_closed(
                        topology,
                        topology.layers_after(previous),
                        &beams,
                        source_line,
                        "exclusive interpretation line",
                    )?;
                }
            }
            KernLine::SpineManipulatorLine(manipulators) => {
                for (column, (&layer, manipulator)) in
                    layers.iter().zip(manipulators.tokens()).enumerate()
                {
                    if *manipulator == SpineManipulator::Terminate {
                        ensure_layer_closed(
                            topology,
                            layer,
                            beams[layer.index()],
                            BeamLocation {
                                source_line,
                                column: Some(column + 1),
                            },
                            "spine termination",
                        )?;
                    }
                }
                carry_beams(topology, line_index, &mut beams, source_line)?;
            }
            KernLine::Barline(barline) => {
                ensure_grace_beams_closed_at_barline(
                    topology,
                    layers,
                    &beams,
                    barline,
                    source_line,
                )?;
            }
            KernLine::Data(contents) => {
                for (column, (&layer, content)) in layers.iter().zip(contents).enumerate() {
                    if *topology.spine_type(layer) != SpineType::Kern {
                        continue;
                    }
                    let location = BeamLocation {
                        source_line,
                        column: Some(column + 1),
                    };
                    let SpineContent::Kern(content) = content else {
                        return beam_error_at(
                            location,
                            "beam.non-kern-content",
                            "kern spine contains non-kern data content",
                        );
                    };
                    let depth =
                        |duration: &Duration| drawn.beam_depth(line_index, column, duration);
                    validate_kern_content_beams(
                        content,
                        &mut beams[layer.index()],
                        location,
                        depth,
                    )?;
                }
            }
            KernLine::TandemInterpretationLine(_) | KernLine::LocalCommentLine(_) => {}
        }
    }

    let end_line = document
        .lines()
        .len()
        .checked_sub(1)
        .map_or(1, |line_index| document.source_line_number(line_index) + 1);
    ensure_all_beams_closed(
        topology,
        topology.layers_after_last(),
        &beams,
        end_line,
        "end of document",
    )
}

/// Give layers born on a manipulator line their beams: a split's left child
/// continues the parent's open beams and the right child starts empty; a
/// merge carries the one open beam among its inputs.
fn carry_beams(
    topology: &Topology,
    line_index: usize,
    beams: &mut [BeamTracks],
    source_line: usize,
) -> Result<(), BeamValidationError> {
    let before = topology.layers_at(line_index);
    let after = topology.layers_after(line_index);
    for &layer in after {
        if before.contains(&layer) {
            continue;
        }
        match &topology.layer(layer).origin {
            LayerOrigin::SplitOf(parent) => {
                beams[layer.index()] = match topology.split_side(layer) {
                    Some(SplitSide::Right) => BeamTracks::default(),
                    _ => beams[parent.index()],
                };
            }
            LayerOrigin::MergeOf(inputs) => {
                let mut carried = None;
                for &input in inputs {
                    let tracks = beams[input.index()];
                    if *topology.spine_type(input) != SpineType::Kern || !tracks.is_open() {
                        continue;
                    }
                    if carried.is_some() {
                        let column = before.iter().position(|&l| l == input).unwrap_or(0);
                        return beam_error_at(
                            BeamLocation {
                                source_line,
                                column: Some(column + 1),
                            },
                            "beam.multiple-open-beams-merge",
                            "multiple open beams enter one spine merge",
                        );
                    }
                    carried = Some(tracks);
                }
                beams[layer.index()] = carried.unwrap_or_default();
            }
            LayerOrigin::Exclusive => {}
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BeamLocation {
    source_line: usize,
    column: Option<usize>,
}

fn ensure_all_beams_closed(
    topology: &Topology,
    layers: &[LayerId],
    beams: &[BeamTracks],
    source_line: usize,
    context: &str,
) -> Result<(), BeamValidationError> {
    for (column, &layer) in layers.iter().enumerate() {
        ensure_layer_closed(
            topology,
            layer,
            beams[layer.index()],
            BeamLocation {
                source_line,
                column: Some(column + 1),
            },
            context,
        )?;
    }
    Ok(())
}

fn ensure_grace_beams_closed_at_barline(
    topology: &Topology,
    layers: &[LayerId],
    beams: &[BeamTracks],
    barline: &crate::Barline,
    source_line: usize,
) -> Result<(), BeamValidationError> {
    let Some((column_index, active_full_beams)) =
        layers
            .iter()
            .enumerate()
            .find_map(|(column_index, &layer)| {
                let active_full_beams = beams[layer.index()].grace.active_full_beams;
                (*topology.spine_type(layer) == SpineType::Kern
                    && !barline.is_invisible_for_spine(column_index)
                    && active_full_beams != 0)
                    .then_some((column_index, active_full_beams))
            })
    else {
        return Ok(());
    };

    beam_error_at(
        BeamLocation {
            source_line,
            column: Some(column_index + 1),
        },
        "beam.grace-crosses-barline",
        beam_crossing_message(BeamStream::Grace, "barline", active_full_beams),
    )
}

fn ensure_layer_closed(
    topology: &Topology,
    layer: LayerId,
    beams: BeamTracks,
    location: BeamLocation,
    context: &str,
) -> Result<(), BeamValidationError> {
    if let (SpineType::Kern, Some((stream, active_full_beams))) =
        (topology.spine_type(layer), beams.first_open())
    {
        return beam_error_at(
            location,
            "beam.unclosed",
            beam_crossing_message(stream, context, active_full_beams),
        );
    }

    Ok(())
}

fn validate_kern_content_beams(
    content: &KernContent,
    beams: &mut BeamTracks,
    location: BeamLocation,
    depth: impl Fn(&Duration) -> u8,
) -> Result<(), BeamValidationError> {
    match content {
        KernContent::Null => Ok(()),
        KernContent::Rest(_)
        | KernContent::InvisibleRestSlice(_)
        | KernContent::MultiMeasureRest(_) => {
            beams.ensure_grace_closed_before_regular_content(location)
        }
        KernContent::Note(note) => {
            beams.ensure_grace_closed_before_regular_content(location)?;
            validate_beam_event(
                depth(&note.duration),
                &ChordBeams::None,
                &mut beams.regular.active_full_beams,
                location,
            )
        }
        KernContent::Chord(chord) => {
            beams.ensure_grace_closed_before_regular_content(location)?;
            let Some(note) = chord.notes.first() else {
                return beam_error_at(location, "beam.empty-chord", "chord contains no notes");
            };
            validate_beam_event(
                depth(&note.duration),
                &chord.modifiers.beams,
                &mut beams.regular.active_full_beams,
                location,
            )
        }
        KernContent::Grace(grace) => validate_grace_beams(grace, beams, location, depth),
    }
}

fn validate_grace_beams(
    grace: &GraceChord,
    beams: &mut BeamTracks,
    location: BeamLocation,
    depth: impl Fn(&Duration) -> u8,
) -> Result<(), BeamValidationError> {
    let Some(note) = grace.notes.first() else {
        return beam_error_at(
            location,
            "beam.empty-grace-chord",
            "grace chord contains no notes",
        );
    };
    let Some(duration) = note.duration.as_ref() else {
        if beams.grace.active_full_beams != 0 || !grace.modifiers.beams.is_empty() {
            return beam_error_at(
                location, "beam.missing-grace-duration",
                format!(
                    "undurationed grace note participates in a beam; active_full_beams={}, beams={:?}",
                    beams.grace.active_full_beams, grace.modifiers.beams
                ),
            );
        }
        return Ok(());
    };

    validate_beam_event(
        depth(duration),
        &grace.modifiers.beams,
        &mut beams.grace.active_full_beams,
        location,
    )
}

fn beam_crossing_message(stream: BeamStream, context: &str, active_full_beams: u8) -> String {
    format!(
        "{} beam crosses {context}; active_full_beams={}",
        stream.label(),
        active_full_beams
    )
}

fn validate_beam_event(
    depth: u8,
    beams: &ChordBeams,
    active_full_beams: &mut u8,
    location: BeamLocation,
) -> Result<(), BeamValidationError> {
    let entering = *active_full_beams;
    let starts = beams.beam_start_count();
    let ends = beams.beam_end_count();
    let partial_left = beams.partial_left_count();
    let partial_right = beams.partial_right_count();

    if ends > entering {
        return beam_event_error(
            location,
            "beam.unmatched-close",
            "beam closes before it is open",
            BeamEventState {
                expected_depth: depth,
                entering,
                starts,
                ends,
                partial_left,
                partial_right,
                outgoing: None,
                left_coverage: None,
                right_coverage: None,
            },
        );
    }

    let after = entering
        .checked_sub(ends)
        .and_then(|value| value.checked_add(starts))
        .ok_or_else(|| {
            beam_error_value(
                location,
                "beam.count-overflow",
                beam_event_message(
                    "beam count overflow",
                    BeamEventState {
                        expected_depth: depth,
                        entering,
                        starts,
                        ends,
                        partial_left,
                        partial_right,
                        outgoing: None,
                        left_coverage: None,
                        right_coverage: None,
                    },
                ),
            )
        })?;

    if depth == 0 {
        if entering != 0 || after != 0 || !beams.is_empty() {
            return beam_event_error(
                location,
                "beam.unbeamable-duration",
                "unbeamable duration participates in a beam",
                BeamEventState {
                    expected_depth: depth,
                    entering,
                    starts,
                    ends,
                    partial_left,
                    partial_right,
                    outgoing: Some(after),
                    left_coverage: None,
                    right_coverage: None,
                },
            );
        }
        *active_full_beams = after;
        return Ok(());
    }

    if entering > depth {
        return beam_event_error(
            location,
            "beam.incoming-depth-exceeded",
            "incoming beam count exceeds duration depth",
            BeamEventState {
                expected_depth: depth,
                entering,
                starts,
                ends,
                partial_left,
                partial_right,
                outgoing: Some(after),
                left_coverage: None,
                right_coverage: None,
            },
        );
    }
    if after > depth {
        return beam_event_error(
            location,
            "beam.outgoing-depth-exceeded",
            "outgoing beam count exceeds duration depth",
            BeamEventState {
                expected_depth: depth,
                entering,
                starts,
                ends,
                partial_left,
                partial_right,
                outgoing: Some(after),
                left_coverage: None,
                right_coverage: None,
            },
        );
    }
    if partial_left > 0 && entering == 0 {
        return beam_event_error(
            location,
            "beam.partial-left-without-incoming",
            "partial-left beam has no incoming beam",
            BeamEventState {
                expected_depth: depth,
                entering,
                starts,
                ends,
                partial_left,
                partial_right,
                outgoing: Some(after),
                left_coverage: None,
                right_coverage: None,
            },
        );
    }
    if partial_right > 0 && after == 0 {
        return beam_event_error(
            location,
            "beam.partial-right-without-outgoing",
            "partial-right beam has no outgoing beam",
            BeamEventState {
                expected_depth: depth,
                entering,
                starts,
                ends,
                partial_left,
                partial_right,
                outgoing: Some(after),
                left_coverage: None,
                right_coverage: None,
            },
        );
    }

    let left_coverage = entering.checked_add(partial_left).ok_or_else(|| {
        beam_error_value(
            location,
            "beam.count-overflow",
            beam_event_message(
                "beam count overflow",
                BeamEventState {
                    expected_depth: depth,
                    entering,
                    starts,
                    ends,
                    partial_left,
                    partial_right,
                    outgoing: Some(after),
                    left_coverage: None,
                    right_coverage: None,
                },
            ),
        )
    })?;
    let right_coverage = after.checked_add(partial_right).ok_or_else(|| {
        beam_error_value(
            location,
            "beam.count-overflow",
            beam_event_message(
                "beam count overflow",
                BeamEventState {
                    expected_depth: depth,
                    entering,
                    starts,
                    ends,
                    partial_left,
                    partial_right,
                    outgoing: Some(after),
                    left_coverage: Some(left_coverage),
                    right_coverage: None,
                },
            ),
        )
    })?;

    if left_coverage > depth || right_coverage > depth {
        return beam_event_error(
            location,
            "beam.coverage-depth-exceeded",
            "beam coverage exceeds duration depth",
            BeamEventState {
                expected_depth: depth,
                entering,
                starts,
                ends,
                partial_left,
                partial_right,
                outgoing: Some(after),
                left_coverage: Some(left_coverage),
                right_coverage: Some(right_coverage),
            },
        );
    }

    let participates_in_beaming =
        entering != 0 || after != 0 || partial_left != 0 || partial_right != 0 || !beams.is_empty();
    if participates_in_beaming && left_coverage != depth && right_coverage != depth {
        return beam_event_error(
            location,
            "beam.coverage-depth-mismatch",
            "beam coverage does not match duration depth",
            BeamEventState {
                expected_depth: depth,
                entering,
                starts,
                ends,
                partial_left,
                partial_right,
                outgoing: Some(after),
                left_coverage: Some(left_coverage),
                right_coverage: Some(right_coverage),
            },
        );
    }

    *active_full_beams = after;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BeamEventState {
    expected_depth: u8,
    entering: u8,
    starts: u8,
    ends: u8,
    partial_left: u8,
    partial_right: u8,
    outgoing: Option<u8>,
    left_coverage: Option<u8>,
    right_coverage: Option<u8>,
}

fn beam_event_error<T>(
    location: BeamLocation,
    code: &'static str,
    reason: &str,
    state: BeamEventState,
) -> Result<T, BeamValidationError> {
    beam_error_at(location, code, beam_event_message(reason, state))
}

fn beam_event_message(reason: &str, state: BeamEventState) -> String {
    format!(
        "{reason}; expected_depth={}, entering={}, starts={}, ends={}, partial_left={}, partial_right={}, outgoing={}, left_coverage={}, right_coverage={}",
        state.expected_depth,
        state.entering,
        state.starts,
        state.ends,
        state.partial_left,
        state.partial_right,
        optional_count(state.outgoing),
        optional_count(state.left_coverage),
        optional_count(state.right_coverage),
    )
}

fn optional_count(value: Option<u8>) -> String {
    value.map_or_else(|| "n/a".to_owned(), |value| value.to_string())
}

fn beam_error_at<T>(
    location: BeamLocation,
    code: &'static str,
    message: impl Into<String>,
) -> Result<T, BeamValidationError> {
    Err(beam_error_value(location, code, message))
}

fn beam_error_value(
    location: BeamLocation,
    code: &'static str,
    message: impl Into<String>,
) -> BeamValidationError {
    BeamValidationError::InvalidStructure {
        code,
        line: location.source_line,
        column: location.column,
        message: beam_location_message(location, message),
    }
}

fn beam_location_message(location: BeamLocation, message: impl Into<String>) -> String {
    let formatted_location = match location.column {
        Some(column) => format!("line {}, column {}", location.source_line, column),
        None => format!("line {}", location.source_line),
    };

    format!("{formatted_location}: {}", message.into())
}

/// Resolve a whole document and check its beams; for tests.
#[cfg(test)]
pub(crate) fn validate_document(document: &crate::KernDocument) -> Result<(), BeamValidationError> {
    let score = Score::resolve(document, crate::document::Framing::Excerpt)
        .expect("test document resolves");
    validate(&score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernDocument;

    fn assert_valid(input: &str) {
        let document = KernDocument::parse(input).unwrap();
        validate_document(&document).unwrap();
    }

    fn assert_invalid(input: &str, expected: &str) {
        let document = KernDocument::parse(input).unwrap();
        let error = validate_document(&document).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "expected {error:?} to contain {expected:?}"
        );
    }

    #[test]
    fn validates_grace_beams_inside_open_regular_beam() {
        assert_valid(
            "**kern\t**kern\t**kern\n\
4AA-\t8b-L\t8cL\n\
.\t16qa-LL\t.\n\
.\t16qb-JJ\t.\n\
.\t8a-\t8g-\n\
4E- A- e-\t8gn\t8c\n\
.\t8g-\t8a-\n\
4AA-\t8b-\t8c\n\
.\t8ccJ\t8g-J\n\
*-\t*-\t*-\n",
        );
    }

    #[test]
    fn validates_regular_beam_interrupted_by_closed_grace_beam() {
        assert_valid("**kern\n8cL\n16qdLL\n16qeJJ\n8fJ\n*-\n");
    }

    #[test]
    fn rejects_grace_beam_crossing_regular_content() {
        assert_invalid(
            "**kern\n8cL\n16qdLL\n8e\n8fJ\n*-\n",
            "grace beam crosses regular content",
        );
    }

    #[test]
    fn rejects_grace_beam_crossing_barline() {
        assert_invalid("**kern\n16qcLL\n=\n*-\n", "grace beam crosses barline");
    }

    #[test]
    fn allows_grace_beam_crossing_invisible_barline() {
        assert_valid("**kern\n16qcLL\n=-\n16qdJJ\n4e\n*-\n");
    }

    #[test]
    fn validates_regular_beam_crossing_barline() {
        assert_valid("**kern\n8cL\n=\n8dJ\n*-\n");
        assert_valid("**kern\t**kern\n8CL\t8cL\n=\t=-\n8DJ\t8dJ\n*-\t*-\n");
    }

    #[test]
    fn validates_regular_beam_before_system_break_barline() {
        assert_valid("**kern\n8cL\n8dJ\n!!LO:LB:g=original\n=\n*-\n");
    }

    #[test]
    fn validates_rational_duration_beam_depth() {
        assert_valid("**kern\n40%3aaL\n40%3bbJ\n*-\n");
    }

    #[test]
    fn validates_rational_duration_deeper_beam_depth() {
        assert_valid("**kern\n32%2cLL\n32%2dJJ\n*-\n");
    }

    #[test]
    fn rejects_beam_mark_on_unbeamable_rational_duration() {
        assert_invalid(
            "**kern\n7%1cL\n7%1dJ\n*-\n",
            "unbeamable duration participates in a beam",
        );
    }

    #[test]
    fn rejects_insufficient_beam_coverage_for_rational_duration() {
        assert_invalid(
            "**kern\n32%2cL\n32%2dJ\n*-\n",
            "beam coverage does not match duration depth",
        );
    }

    #[test]
    fn validates_regular_beam_continuing_in_left_child_after_spine_split() {
        assert_valid("**kern\n8cL\n*^\n8e\t4c\n8f\t.\n*v\t*v\n8cJ\n*-\n");
    }

    #[test]
    fn rejects_regular_beam_continuing_in_right_child_after_spine_split() {
        assert_invalid(
            "**kern\n8cL\n*^\n4c\t8e\n.\t8f\n*v\t*v\n8cJ\n*-\n",
            "unbeamable duration participates in a beam",
        );
    }

    #[test]
    fn validates_spine_merge_carrying_one_open_beam_from_left_input() {
        assert_valid("**kern\n*^\n8cL\t4e\n8d\t.\n*v\t*v\n8fJ\n*-\n");
    }

    #[test]
    fn validates_spine_merge_carrying_one_open_beam_from_right_input() {
        assert_valid("**kern\n*^\n4c\t8eL\n.\t8f\n*v\t*v\n8gJ\n*-\n");
    }

    #[test]
    fn rejects_spine_merge_combining_two_open_beams() {
        assert_invalid(
            "**kern\n*^\n8cL\t8eL\n8d\t8f\n*v\t*v\n8gJ\n*-\n",
            "multiple open beams enter one spine merge",
        );
    }

    #[test]
    fn validates_real_world_beam_crossing_spine_merge_regression() {
        assert_valid(
            "**kern\t**kern\n\
*\t*^\n\
8E-L\t8gL\t2g\n\
8D\t16ccL\t.\n\
.\t16ddJJ\t.\n\
8C\t8ee-L\t.\n\
8D\t8dd\t.\n\
*\t*v\t*v\n\
8E-\t8g cc\n\
8EnJ\t8g b-J\n\
*-\t*-\n",
        );
    }

    #[test]
    fn beam_errors_report_source_line_numbers() {
        let error = validate_document(
            &KernDocument::parse("!! comment\n**kern\n8cL\n!!LO:LB:g=original\n=\n*-\n").unwrap(),
        )
        .unwrap_err();

        assert!(
            error.to_string().contains("line 6, column 1"),
            "expected source line 6 in {error}"
        );
    }

    #[test]
    fn rejects_regular_beam_crossing_spine_termination() {
        assert_invalid(
            "**kern\n8cL\n*-\n",
            "regular beam crosses spine termination",
        );
    }

    #[test]
    fn beams_follow_a_visual_duration_that_redraws_the_note() {
        // Ekier's Op. 15 No. 1: a triplet quarter drawn as a dotted eighth
        // beamed to a sextuplet sixteenth.
        let figure = "**kern\n!LO:N:vis=12.\n6cL\n!LO:N:vis=24\n12dJk\n*-\n";
        assert_valid(figure);
        assert_invalid(
            &figure.replace("!LO:N:vis=12.\n", ""),
            "unbeamable duration participates in a beam",
        );
        // A sixteenth drawn as an eighth takes one beam.
        assert_valid("**kern\n!LO:N:vis=8\n16cL\n8dJ\n*-\n");
    }

    #[test]
    fn beams_follow_a_visual_duration_under_its_tuplet_scaling() {
        // Verovio draws `64%3` with `vis=32%3` as a sixteenth.
        assert_valid("**kern\n!LO:N:vis=32%3\n64%3cLL\n!LO:N:vis=32%3\n64%3dJJ\n*-\n");
        assert_invalid(
            "**kern\n!LO:N:vis=32%3\n64%3cLLL\n!LO:N:vis=32%3\n64%3dJJJ\n*-\n",
            "outgoing beam count exceeds duration depth",
        );
    }

    #[test]
    fn a_visual_notehead_with_no_beams_keeps_the_written_beams() {
        // A held bass note drawn with a half-note head on a beamed stem.
        assert_valid("**kern\n!LO:N:vis=2\n8CL\n8G\n8cJ\n*-\n");
    }

    #[test]
    fn a_rhythm_scale_overrides_a_visual_duration() {
        assert_invalid(
            "**kern\n*rscale:2\n!LO:N:vis=8\n8cL\n8dJ\n*-\n",
            "unbeamable duration participates in a beam",
        );
    }

    #[test]
    fn a_visual_duration_for_one_chord_note_does_not_redraw_the_token() {
        assert_invalid(
            "**kern\n!LO:N:vis=12.:n=1\n6c 6eL\n12dJ\n*-\n",
            "unbeamable duration participates in a beam",
        );
    }

    #[test]
    fn a_visual_duration_links_to_the_next_token_of_its_layer() {
        // The parameter belongs to the null token below it, as in humlib.
        assert_invalid(
            "**kern\t**kern\n!\t!LO:N:vis=12.\n4c\t.\n.\t6dL\n.\t12eJ\n*-\t*-\n",
            "unbeamable duration participates in a beam",
        );
    }
}
