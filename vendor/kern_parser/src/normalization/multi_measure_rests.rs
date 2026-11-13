//! Multi-measure rest compaction and expansion for the OMR dialect.

use crate::{
    analysis::resolve_initial_time_signature,
    document::Framing,
    measure::MeasureDuration,
    score::Score,
    types::{
        Barline, BarlineStyle, Duration, DurationValue, DynamContent, FingContent, KernContent,
        KernDocument, KernLine, MultiMeasureRest, MxmlHarmonyContent, RepeatStyle, Rest,
        SpineContent, SpineType, TextContent,
    },
};

use super::{NormalizationPass, NormalizeOptions};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactMultiMeasureRests {
    options: NormalizeOptions,
}

impl CompactMultiMeasureRests {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for CompactMultiMeasureRests {
    fn normalize(&self, document: &mut KernDocument) {
        let _ = &self.options;
        compact_multi_measure_rests(document);
    }
}

/// Spine type and bar length of one field.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SpineMeterState {
    spine_type: SpineType,
    meter: MeasureDuration,
}

/// The fields of every line with their track's meter, or the inferred
/// initial meter before the track declares one. `None` when the document's
/// score does not resolve or its initial meter cannot be inferred; the
/// multi-measure-rest passes then leave it unchanged.
fn field_meters(document: &KernDocument) -> Option<Vec<Vec<SpineMeterState>>> {
    let score = Score::resolve(document, Framing::Excerpt).ok()?;
    let initial_meter =
        MeasureDuration::from_time_signature(&resolve_initial_time_signature(&score).ok()?);
    let topology = score.topology();
    Some(
        (0..document.lines().len())
            .map(|line_index| {
                topology
                    .layers_at(line_index)
                    .iter()
                    .enumerate()
                    .map(|(column, &layer)| SpineMeterState {
                        spine_type: topology.spine_type(layer).clone(),
                        meter: score
                            .state_at(line_index, column)
                            .track
                            .meter
                            .measure_duration()
                            .unwrap_or(initial_meter),
                    })
                    .collect()
            })
            .collect(),
    )
}

pub fn expanded_multi_measure_rests(document: &KernDocument) -> KernDocument {
    let Some(meters) = field_meters(document) else {
        return document.clone();
    };
    let mut lines = Vec::new();
    let mut saw_musical_data = false;

    for (line_index, line) in document.lines().iter().enumerate() {
        let origin = document.origin(line_index);
        let KernLine::Data(contents) = line else {
            lines.push((line.clone(), origin.clone()));
            continue;
        };

        if let Some((measure_count, expanded_contents)) =
            expand_data_line(contents, &meters[line_index])
        {
            // A duplicated barline before the first rest measure breaks
            // Verovio's condensation of the expansion back into one
            // multi-measure rest, so only synthesize the leading barline
            // when the document does not already provide one.
            if !saw_musical_data
                && !matches!(
                    lines.last().map(|(line, _)| line),
                    Some(KernLine::Barline(_))
                )
            {
                lines.push((KernLine::Barline(basic_barline()), origin.clone()));
            }
            for measure_index in 0..measure_count {
                if measure_index > 0 {
                    lines.push((KernLine::Barline(basic_barline()), origin.clone()));
                }
                lines.push((KernLine::Data(expanded_contents.clone()), origin.clone()));
            }
        } else {
            lines.push((line.clone(), origin.clone()));
        }
        if contains_musical_data(contents) {
            saw_musical_data = true;
        }
    }

    KernDocument::from_parts(document.metadata.clone(), document.layout.clone(), lines)
}

fn contains_musical_data(contents: &[SpineContent]) -> bool {
    contents.iter().any(|content| {
        matches!(
            content,
            SpineContent::Kern(
                KernContent::Rest(_)
                    | KernContent::MultiMeasureRest(_)
                    | KernContent::Note(_)
                    | KernContent::Chord(_)
                    | KernContent::Grace(_)
            )
        )
    })
}

fn compact_multi_measure_rests(document: &mut KernDocument) {
    let Some(meters) = field_meters(document) else {
        return;
    };
    let mut index = 0;
    let mut lines = Vec::with_capacity(document.lines().len());

    while index < document.lines().len() {
        let line = &document.lines()[index];
        let spines = &meters[index];

        let KernLine::Data(contents) = line else {
            lines.push((line.clone(), document.origin(index).clone()));
            index += 1;
            continue;
        };
        if index == 0
            || !has_multi_measure_rest_boundary_before_data(document.lines(), index)
            || !data_line_is_plain_full_measure_rests(contents, spines)
        {
            lines.push((line.clone(), document.origin(index).clone()));
            index += 1;
            continue;
        }

        let mut measure_count = 0usize;
        let mut cursor = index;
        let mut final_barline_index = None;

        while cursor + 1 < document.lines().len()
            && matches!(&document.lines()[cursor], KernLine::Data(candidate) if data_line_is_plain_full_measure_rests(candidate, spines))
            && is_multi_measure_rest_boundary_line(&document.lines()[cursor + 1])
        {
            measure_count += 1;
            final_barline_index = Some(cursor + 1);
            if !is_plain_barline_line(&document.lines()[cursor + 1]) {
                break;
            }
            let next_data = cursor + 2;
            if next_data >= document.lines().len()
                || !matches!(&document.lines()[next_data], KernLine::Data(candidate) if data_line_is_plain_full_measure_rests(candidate, spines))
            {
                break;
            }
            cursor = next_data;
        }

        let Some(final_barline_index) = final_barline_index else {
            lines.push((
                document.lines()[index].clone(),
                document.origin(index).clone(),
            ));
            index += 1;
            continue;
        };
        let Some(measure_count) = u32::try_from(measure_count).ok() else {
            lines.push((
                document.lines()[index].clone(),
                document.origin(index).clone(),
            ));
            index += 1;
            continue;
        };
        if measure_count < 2 {
            lines.push((
                document.lines()[index].clone(),
                document.origin(index).clone(),
            ));
            index += 1;
            continue;
        }

        lines.push((
            KernLine::Data(multi_measure_rest_contents(contents, spines, measure_count)),
            document.origin(index).clone(),
        ));
        lines.push((
            document.lines()[final_barline_index].clone(),
            document.origin(final_barline_index).clone(),
        ));
        index = final_barline_index + 1;
    }

    document.replace_lines(lines);
}

fn data_line_is_plain_full_measure_rests(
    contents: &[SpineContent],
    spines: &[SpineMeterState],
) -> bool {
    if contents.len() != spines.len() {
        return false;
    }

    let mut has_kern_rest = false;

    for (content, state) in contents.iter().zip(spines) {
        match state.spine_type {
            SpineType::Kern => {
                if !matches!(
                    content,
                    SpineContent::Kern(KernContent::Rest(rest))
                        if plain_rest_duration(rest) == Some(state.meter)
                ) {
                    return false;
                }
                has_kern_rest = true;
            }
            SpineType::Dynam
            | SpineType::Fing
            | SpineType::Text
            | SpineType::MxmlChords
            | SpineType::Other(_) => {
                if !is_null_non_kern_content(content) {
                    return false;
                }
            }
        }
    }

    has_kern_rest
}

fn is_null_non_kern_content(content: &SpineContent) -> bool {
    matches!(
        content,
        SpineContent::Dynam(DynamContent::Null)
            | SpineContent::Fing(FingContent::Null)
            | SpineContent::Text(TextContent::Null)
            | SpineContent::MxmlHarmony(MxmlHarmonyContent::Null)
    )
}

fn plain_rest_duration(rest: &Rest) -> Option<MeasureDuration> {
    (rest.cross_staff.is_none()
        && rest.placement.is_none()
        && !rest.invisible
        && rest.breath.is_none()
        && rest.fermata.is_none()
        && rest.slurs.is_empty())
    .then(|| MeasureDuration::from_duration(&rest.duration))
}

fn is_plain_barline_line(line: &KernLine) -> bool {
    matches!(line, KernLine::Barline(barline) if is_plain_barline(barline))
}

fn is_plain_barline(barline: &Barline) -> bool {
    !barline.has_fermata()
        && barline.repeat() == RepeatStyle::None
        && barline
            .styles()
            .iter()
            .all(|style| matches!(style, BarlineStyle::Basic))
}

fn is_multi_measure_rest_boundary_line(line: &KernLine) -> bool {
    matches!(line, KernLine::Barline(barline) if is_multi_measure_rest_boundary(barline))
}

fn is_multi_measure_rest_boundary(barline: &Barline) -> bool {
    barline.repeat() == RepeatStyle::None
        && barline.styles().iter().all(|style| {
            matches!(
                style,
                BarlineStyle::Basic
                    | BarlineStyle::Double
                    | BarlineStyle::Thin
                    | BarlineStyle::Dashed
            )
        })
}

fn has_multi_measure_rest_boundary_before_data(lines: &[KernLine], data_index: usize) -> bool {
    lines[..data_index]
        .iter()
        .rev()
        .find(|line| !matches!(line, KernLine::TandemInterpretationLine(_)))
        .is_some_and(is_multi_measure_rest_boundary_line)
}

fn multi_measure_rest_contents(
    contents: &[SpineContent],
    spines: &[SpineMeterState],
    measure_count: u32,
) -> Vec<SpineContent> {
    contents
        .iter()
        .zip(spines)
        .map(|(_, state)| match state.spine_type {
            SpineType::Kern => {
                SpineContent::Kern(KernContent::MultiMeasureRest(MultiMeasureRest {
                    measure_count,
                }))
            }
            SpineType::Dynam
            | SpineType::Fing
            | SpineType::Text
            | SpineType::MxmlChords
            | SpineType::Other(_) => SpineContent::Kern(KernContent::Null),
        })
        .collect()
}

fn expand_data_line(
    contents: &[SpineContent],
    spines: &[SpineMeterState],
) -> Option<(u32, Vec<SpineContent>)> {
    if contents.len() != spines.len() {
        return None;
    }

    let mut measure_count = None;
    let mut expanded = Vec::with_capacity(contents.len());

    for (content, state) in contents.iter().zip(spines) {
        match (&state.spine_type, content) {
            (SpineType::Kern, SpineContent::Kern(KernContent::MultiMeasureRest(rest))) => {
                if measure_count
                    .replace(rest.measure_count)
                    .is_some_and(|count| count != rest.measure_count)
                {
                    return None;
                }
                expanded.push(SpineContent::Kern(KernContent::Rest(Rest {
                    duration: full_measure_rest_duration(state.meter)?,
                    cross_staff: None,
                    placement: None,
                    invisible: false,
                    breath: None,
                    fermata: None,
                    slurs: Vec::new(),
                    phrases: Vec::new(),
                })));
            }
            (SpineType::Kern, SpineContent::Kern(KernContent::Null)) => {
                expanded.push(content.clone());
            }
            _ => return None,
        }
    }

    measure_count.map(|count| (count, expanded))
}

fn basic_barline() -> Barline {
    Barline::new(vec![BarlineStyle::Basic], RepeatStyle::None)
}

/// Rest duration spanning a full measure of `meter`.
///
/// Unlike [`MeasureDuration::into_duration`], prefers dotted reciprocal
/// notation (`2.` over `3%4`) because expansion output must read like
/// ordinary engraved rests.
fn full_measure_rest_duration(meter: MeasureDuration) -> Option<Duration> {
    match meter.as_fraction() {
        (8, 1) => {
            return Some(Duration {
                value: DurationValue::Maxima,
                augmentation_dots: 0,
            });
        }
        (4, 1) => {
            return Some(Duration {
                value: DurationValue::Long,
                augmentation_dots: 0,
            });
        }
        (2, 1) => {
            return Some(Duration {
                value: DurationValue::Breve,
                augmentation_dots: 0,
            });
        }
        _ => {}
    }

    for augmentation_dots in 0..=4 {
        let dotted_numerator = (1_i64 << (augmentation_dots + 1)) - 1;
        let dot_multiplier = 1_i64 << augmentation_dots;
        let (numerator, denominator) = meter.as_fraction();
        if numerator <= 0 || denominator <= 0 {
            return None;
        }
        let reciprocal_numerator = denominator * dotted_numerator;
        let reciprocal_denominator = dot_multiplier * numerator;
        if reciprocal_numerator % reciprocal_denominator != 0 {
            continue;
        }
        let Ok(reciprocal) = u32::try_from(reciprocal_numerator / reciprocal_denominator) else {
            continue;
        };
        if !(1..=4096).contains(&reciprocal) {
            continue;
        }
        let candidate = Duration {
            value: DurationValue::Reciprocal(reciprocal),
            augmentation_dots,
        };
        if MeasureDuration::from_duration(&candidate) == meter {
            return Some(candidate);
        }
    }

    rational_measure_duration(meter)
}

fn rational_measure_duration(meter: MeasureDuration) -> Option<Duration> {
    let (numerator, denominator) = meter.as_fraction();
    if numerator <= 0 || denominator <= 0 {
        return None;
    }

    Some(Duration {
        value: DurationValue::Rational {
            numerator: u32::try_from(denominator).ok()?,
            denominator: u32::try_from(numerator).ok()?,
        },
        augmentation_dots: 0,
    })
}

#[cfg(test)]
fn full_measure_rest_duration_bruteforce(meter: MeasureDuration) -> Option<Duration> {
    for value in [
        DurationValue::Breve,
        DurationValue::Long,
        DurationValue::Maxima,
    ] {
        let candidate = Duration {
            value,
            augmentation_dots: 0,
        };
        if MeasureDuration::from_duration(&candidate) == meter {
            return Some(candidate);
        }
    }

    for augmentation_dots in 0..=4 {
        for reciprocal in 1..=4096 {
            let candidate = Duration {
                value: DurationValue::Reciprocal(reciprocal),
                augmentation_dots,
            };
            if MeasureDuration::from_duration(&candidate) == meter {
                return Some(candidate);
            }
        }
    }
    rational_measure_duration(meter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LineOrigin;

    fn parse(input: &str) -> KernDocument {
        KernDocument::parse(input).expect("test input parses")
    }

    fn compacted_lines(input: &str) -> Vec<KernLine> {
        let mut document = parse(input);
        CompactMultiMeasureRests::new(Default::default()).normalize(&mut document);
        document.lines().to_vec()
    }

    #[test]
    fn compacts_rest_run_after_spine_merge() {
        let input = "**kern\n\
                     *M4/4\n\
                     *^\n\
                     *^\t*\n\
                     4c\t4e\t4g\n\
                     *v\t*v\t*\n\
                     =\t=\n\
                     1r\t1r\n\
                     =\t=\n\
                     1r\t1r\n\
                     =\t=\n\
                     *-\t*-\n";
        let expected = "**kern\n\
                        *M4/4\n\
                        *^\n\
                        *^\t*\n\
                        4c\t4e\t4g\n\
                        *v\t*v\t*\n\
                        =\t=\n\
                        MM2r\tMM2r\n\
                        =\t=\n\
                        *-\t*-\n";

        assert_eq!(compacted_lines(input), parse(expected).lines());
    }

    #[test]
    fn compacts_rest_run_after_spine_split() {
        let input = "**kern\n\
                     *M4/4\n\
                     1c\n\
                     *^\n\
                     =\t=\n\
                     1r\t1r\n\
                     =\t=\n\
                     1r\t1r\n\
                     =\t=\n\
                     *-\t*-\n";
        let expected = "**kern\n\
                        *M4/4\n\
                        1c\n\
                        *^\n\
                        =\t=\n\
                        MM2r\tMM2r\n\
                        =\t=\n\
                        *-\t*-\n";

        assert_eq!(compacted_lines(input), parse(expected).lines());
    }

    #[test]
    fn compacts_against_meter_active_at_merge() {
        let input = "**kern\n\
                     *M4/4\n\
                     *^\n\
                     4c\t4e\n\
                     *M3/4\t*M3/4\n\
                     *v\t*v\n\
                     =\n\
                     2.r\n\
                     =\n\
                     2.r\n\
                     =\n\
                     *-\n";
        let expected = "**kern\n\
                        *M4/4\n\
                        *^\n\
                        4c\t4e\n\
                        *M3/4\t*M3/4\n\
                        *v\t*v\n\
                        =\n\
                        MM2r\n\
                        =\n\
                        *-\n";

        assert_eq!(compacted_lines(input), parse(expected).lines());
    }

    #[test]
    fn does_not_compact_rests_shorter_than_meter_after_merge() {
        let input = "**kern\n\
                     *M4/4\n\
                     *^\n\
                     4c\t4e\n\
                     *M3/4\t*M3/4\n\
                     *v\t*v\n\
                     *clefG2\n\
                     =\n\
                     1r\n\
                     =\n\
                     1r\n\
                     =\n\
                     *-\n";

        assert_eq!(compacted_lines(input), parse(input).lines());
    }

    #[test]
    fn expands_with_meter_active_at_merge() {
        let document = parse(
            "**kern\n\
             *M4/4\n\
             *^\n\
             4c\t4e\n\
             *M3/4\t*M3/4\n\
             *v\t*v\n\
             =\n\
             MM2r\n\
             =\n\
             *-\n",
        );
        let expected = parse(
            "**kern\n\
             *M4/4\n\
             *^\n\
             4c\t4e\n\
             *M3/4\t*M3/4\n\
             *v\t*v\n\
             =\n\
             2.r\n\
             =\n\
             2.r\n\
             =\n\
             *-\n",
        );

        assert_eq!(
            expanded_multi_measure_rests(&document).lines(),
            expected.lines()
        );
    }

    #[test]
    fn compaction_and_expansion_round_trip_after_merge() {
        let original = parse(
            "**kern\n\
             *M4/4\n\
             *^\n\
             4c\t4e\n\
             *M3/4\t*M3/4\n\
             *v\t*v\n\
             =\n\
             2.r\n\
             =\n\
             2.r\n\
             =\n\
             *-\n",
        );

        let mut compacted = original.clone();
        CompactMultiMeasureRests::new(Default::default()).normalize(&mut compacted);
        assert_ne!(compacted.lines(), original.lines());

        assert_eq!(
            expanded_multi_measure_rests(&compacted).lines(),
            original.lines()
        );
    }

    #[test]
    fn compacted_multi_measure_rest_keeps_first_rest_source() {
        let mut document = parse(
            "**kern\n\
             *M4/4\n\
             =\n\
             1r\n\
             =\n\
             1r\n\
             =\n\
             *-\n",
        );

        CompactMultiMeasureRests::new(Default::default()).normalize(&mut document);

        let mm_rest_index = document
            .lines()
            .iter()
            .position(|line| matches!(line, KernLine::Data(_)))
            .expect("compacted data line exists");
        assert_eq!(document.source(mm_rest_index), Some(4));
    }

    #[test]
    fn compaction_keeps_a_generated_line_generated() {
        let mut document = parse("**kern\n*M4/4\n=\n1r\n=\n1r\n=\n4c\n*-\n");
        let grace = KernLine::parse(1, "8qd").unwrap();
        let note_index = document.lines().len() - 2;
        document.insert_line(note_index, grace.clone(), LineOrigin::Generated);

        CompactMultiMeasureRests::new(Default::default()).normalize(&mut document);

        let grace_index = document
            .lines()
            .iter()
            .position(|line| *line == grace)
            .expect("compaction keeps the grace");
        assert_eq!(grace_index, 7);
        assert_eq!(document.origin(grace_index), &LineOrigin::Generated);
    }

    #[test]
    fn full_measure_rest_duration_uses_zero_system_for_long_values() {
        assert_eq!(
            full_measure_rest_duration(MeasureDuration::new(8, 1)),
            Some(Duration {
                value: DurationValue::Maxima,
                augmentation_dots: 0,
            })
        );
        assert_eq!(
            full_measure_rest_duration(MeasureDuration::new(4, 1)),
            Some(Duration {
                value: DurationValue::Long,
                augmentation_dots: 0,
            })
        );
        assert_eq!(
            full_measure_rest_duration(MeasureDuration::new(2, 1)),
            Some(Duration {
                value: DurationValue::Breve,
                augmentation_dots: 0,
            })
        );
    }

    #[test]
    fn expanded_multi_measure_rest_in_breve_meter_uses_breve_rests() {
        let compacted = parse(
            "**kern\t**kern\n\
             *M4/2\t*M4/2\n\
             =\t=\n\
             MM2r\tMM2r\n\
             =\t=\n\
             *-\t*-\n",
        );

        let expanded = expanded_multi_measure_rests(&compacted);
        let expected = parse(
            "**kern\t**kern\n\
             *M4/2\t*M4/2\n\
             =\t=\n\
             0r\t0r\n\
             =\t=\n\
             0r\t0r\n\
             =\t=\n\
             *-\t*-\n",
        );

        assert_eq!(expanded.lines(), expected.lines());
    }

    #[test]
    fn full_measure_rest_duration_matches_bruteforce_oracle() {
        for numerator in 1..=64 {
            for denominator in 1..=64 {
                let meter = MeasureDuration::new(numerator, denominator);

                assert_eq!(
                    full_measure_rest_duration(meter),
                    full_measure_rest_duration_bruteforce(meter),
                    "{numerator}/{denominator}",
                );
            }
        }
    }
}
