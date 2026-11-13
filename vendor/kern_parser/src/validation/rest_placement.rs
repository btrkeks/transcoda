//! Warnings for visible rests Verovio cannot position among split layers.

use std::collections::BTreeMap;

use crate::{
    analysis::line_durations::ScoreClock,
    measure::MusicalDuration,
    score::{Score, TrackId},
    validation::{
        duration_state::DurationStateValidationError, SourceLocation, ValidationSeverity,
        ValidationWarning,
    },
    KernContent, KernLine, SpineContent, SpineType,
};

#[derive(Clone, Copy)]
struct TimeSpan {
    start: MusicalDuration,
    end: MusicalDuration,
}

struct UnplacedRest {
    track: TrackId,
    layer: usize,
    span: TimeSpan,
    location: SourceLocation,
}

pub(crate) fn check(
    score: &Score<'_>,
) -> Result<Vec<ValidationWarning>, DurationStateValidationError> {
    let document = score.document();
    let topology = score.topology();
    let mut clock = ScoreClock::new(score);
    let mut onset = MusicalDuration::zero();
    let mut staffs = BTreeMap::<TrackId, BTreeMap<usize, Vec<TimeSpan>>>::new();
    let mut rests = Vec::new();

    for (line_index, line) in document.lines().iter().enumerate() {
        let advance = clock.observe(line_index)?;
        let KernLine::Data(contents) = line else {
            continue;
        };
        let advance = advance.expect("data lines have a duration");
        let line_number = document.source_line_number(line_index);

        for (field_index, (content, &layer)) in contents
            .iter()
            .zip(topology.layers_at(line_index))
            .enumerate()
        {
            if *topology.spine_type(layer) != SpineType::Kern {
                continue;
            }
            // Number siblings within their staff, rather than by physical
            // column or split occurrence. A merged/reopened layer reuses its
            // number, as it does in engraving.
            let track = topology.track_of(layer);
            let layer_number = field_index - topology.track_columns(line_index, track).start;
            let SpineContent::Kern(content) = content else {
                continue;
            };
            // Verovio counts notes, chords, visible rests and multi-measure
            // rests, but not spaces (invisible rests) or durationless grace notes.
            match content {
                KernContent::Note(_) | KernContent::Chord(_) | KernContent::MultiMeasureRest(_) => {
                }
                KernContent::Rest(rest) if !rest.invisible => {}
                _ => continue,
            }
            let span = TimeSpan {
                start: onset,
                end: onset + advance + clock.remaining(layer),
            };
            // Cross-staff RDF markers do not change the encoded staff here.
            // This deliberately approximates Verovio's destination staff.
            let spans = staffs
                .entry(track)
                .or_default()
                .entry(layer_number)
                .or_default();
            // Topology changes can assign another event to a layer before an
            // earlier span ends. Store the union so end times stay ordered.
            if let Some(previous) = spans
                .last_mut()
                .filter(|previous| previous.end >= span.start)
            {
                previous.end = previous.end.max(span.end);
            } else {
                spans.push(span);
            }

            if matches!(content, KernContent::Rest(rest) if rest.placement.is_none()) {
                rests.push(UnplacedRest {
                    track,
                    layer: layer_number,
                    span,
                    location: SourceLocation {
                        line: line_number,
                        column: Some(field_index + 1),
                    },
                });
            }
        }
        onset += advance;
    }

    Ok(rests
        .into_iter()
        .filter_map(|rest| {
            let other_voices = staffs[&rest.track]
                .iter()
                .filter(|(layer, _)| **layer != rest.layer)
                .filter(|(_, spans)| {
                    let index = spans.partition_point(|span| span.end <= rest.span.start);
                    spans
                        .get(index)
                        .is_some_and(|span| span.start < rest.span.end)
                })
                .count();
            if other_voices < 2 {
                return None;
            }
            let line = rest.location.line;
            let field = rest.location.column.expect("rests have a field");
            Some(ValidationWarning {
                span: None,
                severity: ValidationSeverity::Warning,
                code: "rest-placement.missing-placement",
                location: rest.location,
                message: format!(
                    "rest at line {line}, field {field} shares the staff with {other_voices} other sounding voices; Verovio places rests automatically only next to one other voice and will draw this one on the middle line. Add a placement such as 4rB (above) or 4rFF (below)."
                ),
                loose_end: None,
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KernDocument, ParseDialect, ParseOptions};

    fn resolved_check(document: &KernDocument) -> Vec<ValidationWarning> {
        check(&Score::resolve(document, crate::document::Framing::Excerpt).unwrap()).unwrap()
    }

    fn warnings(input: &str) -> Vec<ValidationWarning> {
        resolved_check(&KernDocument::parse(input).unwrap())
    }

    fn three_layers(data: &str) -> String {
        format!("**kern\n*^\n*^\t*\n{data}*-\t*-\t*-\n")
    }

    #[test]
    fn warns_on_top_and_bottom_rests_around_a_held_half_note() {
        let warnings = warnings(&three_layers("4r\t2C\t4r\n4e\t.\t4G\n"));
        assert_eq!(warnings.len(), 2);
        assert_eq!(warnings[0].code, "rest-placement.missing-placement");
        assert_eq!(
            warnings[0].location,
            SourceLocation {
                line: 4,
                column: Some(1)
            }
        );
        assert_eq!(
            warnings[1].location,
            SourceLocation {
                line: 4,
                column: Some(3)
            }
        );
        assert_eq!(warnings[0].message, "rest at line 4, field 1 shares the staff with 2 other sounding voices; Verovio places rests automatically only next to one other voice and will draw this one on the middle line. Add a placement such as 4rB (above) or 4rFF (below).");
    }

    #[test]
    fn accepts_explicit_rest_placements() {
        assert!(warnings(&three_layers("4rB\t2C\t4rFF\n4e\t.\t4G\n")).is_empty());
    }

    #[test]
    fn accepts_two_layers() {
        assert!(warnings("**kern\n*^\n4r\t2C\n4e\t.\n*-\t*-\n").is_empty());
    }

    #[test]
    fn accepts_two_layers_that_merge_and_reopen_during_a_rest() {
        assert!(warnings("**kern\n*^\n2r\t4C\n*v\t*v\n*^\n.\t4D\n*-\t*-\n").is_empty());
    }

    #[test]
    fn ignores_a_silent_third_layer() {
        assert!(warnings(&three_layers("4r\t4C\t4ryy\n")).is_empty());
    }

    #[test]
    fn ignores_invisible_rests_themselves() {
        assert!(warnings(&three_layers("4ryy\t4C\t4E\n")).is_empty());
    }

    #[test]
    fn excludes_elements_ending_at_the_rest_start_or_starting_at_its_end() {
        assert!(warnings(&three_layers("4e\t4C\t4G\n4r\t4D\t4ryy\n4f\t4E\t4A\n")).is_empty());
    }

    #[test]
    fn ignores_grace_notes_in_another_layer() {
        assert!(warnings(&three_layers("4r\t4C\t8qG\n")).is_empty());
    }

    #[test]
    fn grace_notes_do_not_hide_a_carried_note_in_the_same_layer() {
        let warnings = warnings(&three_layers("4e\t2C\t2G\n4r\t.\t8qA\n"));
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].location.line, 5);
    }

    #[test]
    fn warns_when_a_third_voice_enters_during_a_rest() {
        let warnings = warnings(&three_layers("2r\t4C\t4ryy\n.\t4D\t4G\n"));
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings[0].location,
            SourceLocation {
                line: 4,
                column: Some(1)
            }
        );
    }

    #[test]
    fn counts_distinct_layers_over_the_whole_span_even_if_they_take_turns() {
        let warnings = warnings(&three_layers("2r\t4C\t4ryy\n.\t4ryy\t4G\n"));
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn counts_each_layer_once_even_if_it_has_a_chord_and_multiple_events() {
        let warnings = warnings(&three_layers("2r\t4C 4E\t4G\n.\t4D 4F\t4A\n"));
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("2 other sounding voices"));
    }

    #[test]
    fn supports_free_meter() {
        let warnings = warnings("**kern\n*MX\n*^\n*^\t*\n4r\t2C\t4r\n4e\t.\t4G\n*-\t*-\t*-\n");
        assert_eq!(warnings.len(), 2);
    }

    #[test]
    fn ignores_independent_staff_roots() {
        assert!(warnings("**kern\t**kern\t**kern\n4r\t4C\t4G\n*-\t*-\t*-\n").is_empty());
        assert!(warnings("**kern\t**kern\n*^\t*\n4r\t4C\t4G\n*-\t*-\t*-\n").is_empty());
    }

    #[test]
    fn counts_cross_staff_notes_on_their_encoded_staff() {
        // The RDF moves C to the upper staff when rendered, but the lint
        // intentionally counts it with the lower staff's split siblings.
        let warnings = warnings("!!!RDF**kern: > = above\n**kern\t**kern\n*^\t*\n*^\t*\t*\n4r\t4C>\t4G\t4c\n*-\t*-\t*-\t*-\n");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].location.line, 5);
    }

    #[test]
    fn preserves_layer_identity_when_an_unrelated_staff_changes_columns() {
        let warnings = warnings("**kern\t**kern\n*\t*^\n*\t*^\t*\n4ryy\t2r\t4C\t4G\n*^\t*\t*\t*\n4ryy\t4ryy\t.\t4D\t4A\n*-\t*-\t*-\t*-\t*-\n");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("2 other sounding voices"));
    }

    #[test]
    fn supports_durationless_invisible_slices() {
        let document = KernDocument::parse_with_options(
            &three_layers("2r\t4C\tryy\n.\t4D\tryy\n"),
            ParseOptions {
                dialect: ParseDialect::TranscodaV2,
            },
        )
        .unwrap();
        assert!(resolved_check(&document).is_empty());
    }

    #[test]
    fn counts_carried_multi_measure_rests() {
        let warnings =
            warnings("**kern\n*M4/4\n*^\n*^\t*\n4ryy\tMM2r\t4ryy\n4r\t.\t4G\n*-\t*-\t*-\n");
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings[0].location,
            SourceLocation {
                line: 6,
                column: Some(1)
            }
        );
    }
}
