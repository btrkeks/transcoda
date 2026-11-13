use super::*;
use crate::{document::Framing, Clef, ClefSign, KernDocument, KernLine, RhythmScale};

fn parse(input: &str) -> KernDocument {
    KernDocument::parse(input).unwrap()
}

fn errors(input: &str) -> Vec<StateError> {
    let document = parse(input);
    Score::resolve(&document, Framing::Complete).unwrap_err().0
}

/// Index of the `nth` (0-based) data line.
fn data(document: &KernDocument, nth: usize) -> usize {
    document
        .lines()
        .iter()
        .enumerate()
        .filter(|(_, line)| matches!(line, KernLine::Data(_)))
        .nth(nth)
        .map(|(index, _)| index)
        .unwrap()
}

fn clef(sign: ClefSign, line: u8) -> Option<Clef> {
    Some(Clef {
        sign,
        line,
        octave_shift: 0,
        invisible: false,
    })
}

#[test]
fn cue_is_layer_state() {
    let document = parse("**kern\n*^\n*\t*cue\n4c\t4e\n*\t*Xcue\n*v\t*v\n4c\n*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let line = data(&document, 0);
    assert!(!score.state_at(line, 0).layer.cue);
    assert!(score.state_at(line, 1).layer.cue);
}

#[test]
fn split_keeps_cue_on_the_left_child_only() {
    let document = parse("**kern\n*cue\n4c\n*^\n4c\t4e\n*v\t*v\n4c\n*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let split = data(&document, 1);
    assert!(score.state_at(split, 0).layer.cue);
    assert!(!score.state_at(split, 1).layer.cue);
    // A merge takes the leftmost input's state, as Verovio's layer number does.
    assert!(score.state_at(data(&document, 2), 0).layer.cue);
}

#[test]
fn renumbering_a_cue_layer_is_an_error_until_rewritten() {
    // Splitting the first voice moves the cue voice from layer 2 to layer 3;
    // Verovio's layer-2 flag would now apply to the new right child.
    let input =
        "**kern\n*^\n*\t*cue\n4c\t4e\n*^\t*\n4c\t4d\t4e\n*\t*\t*Xcue\n*v\t*v\t*\n*v\t*v\n*-\n";
    assert_eq!(
        errors(input),
        vec![
            StateError::CueRenumbered {
                line: 6,
                column: 2,
                manipulation: 5,
                cue: false,
            },
            StateError::CueRenumbered {
                line: 6,
                column: 3,
                manipulation: 5,
                cue: true,
            },
        ]
    );
    let fixed = parse("**kern\n*^\n*\t*cue\n4c\t4e\n*^\t*\n*\t*Xcue\t*cue\n4c\t4d\t4e\n*\t*\t*Xcue\n*v\t*v\t*\n*v\t*v\n*-\n");
    let score = Score::resolve(&fixed, Framing::Complete).unwrap();
    let rewrite = score
        .events()
        .iter()
        .filter(|event| event.key == StateKey::Cue)
        .map(|event| event.outcome)
        .collect::<Vec<_>>();
    // `*Xcue` clears a stale flag and `*cue` sets the moved voice's flag;
    // neither is redundant.
    assert_eq!(
        rewrite,
        vec![
            Outcome::Changed,
            Outcome::Changed,
            Outcome::Changed,
            Outcome::Changed
        ]
    );
}

#[test]
fn manipulating_another_track_leaves_cue_open() {
    let document = parse(
        "**kern\t**kern\n*cue\t*\n4c\t4e\n*\t*^\n4c\t4e\t4g\n*\t*v\t*v\n*Xcue\t*\n4c\t4e\n*-\t*-\n",
    );
    assert!(Score::resolve(&document, Framing::Complete).is_ok());
}

#[test]
fn cue_may_stay_open_to_the_end() {
    let document = parse("**kern\n*cue\n4c\n*-\n");
    assert!(Score::resolve(&document, Framing::Complete).is_ok());
}

#[test]
fn toggles_and_spans_switch_on_or_off() {
    let side = |raw: &str| StateKey::switches_on(&TandemInterpretation::try_from(raw).unwrap());
    for raw in [
        "*8va", "*8ba", "*ped", "*tremolo", "*rep", "*lig", "*col", "*cue",
    ] {
        assert_eq!(side(raw), Some(true), "{raw}");
    }
    for raw in [
        "*X8va",
        "*X8ba",
        "*Xped",
        "*Xtremolo",
        "*Xrep",
        "*Xlig",
        "*Xcol",
        "*Xcue",
    ] {
        assert_eq!(side(raw), Some(false), "{raw}");
    }
    for raw in [
        "*clefG2",
        "*M3/4",
        "*Xtuplet",
        "*Xbrackettup",
        "*rscale:2",
        "*above",
        "*staff1",
        "*MM120",
    ] {
        assert_eq!(side(raw), None, "{raw}");
    }
}

#[test]
fn restating_cue_is_redundant() {
    let document = parse("**kern\n*Xcue\n4c\n*cue\n4c\n*cue\n*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let outcomes = score
        .events()
        .iter()
        .filter(|event| event.key == StateKey::Cue)
        .map(|event| event.outcome)
        .collect::<Vec<_>>();
    assert_eq!(
        outcomes,
        vec![Outcome::Redundant, Outcome::Changed, Outcome::Redundant]
    );
}

/// Meter outcomes in event order.
fn meter_outcomes(input: &str) -> Vec<Outcome> {
    let document = parse(input);
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    score
        .events()
        .iter()
        .filter(|event| event.key == StateKey::Meter)
        .map(|event| event.outcome)
        .collect()
}

#[test]
fn a_meter_group_decides_its_outcome_once_it_ends() {
    // A hidden restatement prints nothing and keeps the bar length.
    assert_eq!(
        meter_outcomes("**kern\n*M3/4\n2.c\n*M3/4\n*met()\n2.d\n*-\n"),
        [Outcome::Changed, Outcome::Redundant, Outcome::Redundant]
    );
    // Verovio prints a visible restatement again.
    assert_eq!(
        meter_outcomes("**kern\n*M3/4\n2.c\n*M3/4\n2.d\n*-\n"),
        [Outcome::Changed, Outcome::Restated]
    );
    // A hidden group that ends free meter changes the bar length.
    assert_eq!(
        meter_outcomes("**kern\n*M2/4\n2c\n*MX\n1d\n*M2/4\n*met()\n2e\n*-\n"),
        [
            Outcome::Changed,
            Outcome::Changed,
            Outcome::Changed,
            Outcome::Changed
        ]
    );
}

#[test]
fn a_hidden_restatement_keeps_the_printed_meter_in_force() {
    let document = parse("**kern\n*met(c)\n1c\n*M4/4\n*met()\n1d\n*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();

    assert_eq!(
        score.state_at(data(&document, 1), 0).track.meter.display(),
        crate::MeterDisplay::Symbol(crate::MeterSymbol::CommonTime)
    );
}

#[test]
fn rhythm_scale_is_track_state_and_may_stay_set() {
    let document = parse("**kern\n*^\n*rscale:2\t*\n4c\t4e\n*v\t*v\n4c\n*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let line = data(&document, 0);
    for column in 0..2 {
        let scale = score.state_at(line, column).track.rhythm_scale;
        assert_eq!((scale.numerator(), scale.denominator()), (2, 1));
    }
    assert_eq!(
        score.state_at(data(&document, 1), 0).track.rhythm_scale,
        RhythmScale::new(2, 1).unwrap()
    );
}

#[test]
fn staff_lines_are_track_state_and_a_restatement_draws_nothing() {
    let document = parse("**kern\n*stria1\n*^\n4c\t4e\n*\t*stria1\n4c\t4e\n*v\t*v\n*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    for column in 0..2 {
        assert_eq!(
            score.state_at(data(&document, 0), column).track.staff_lines,
            Some(1)
        );
    }
    let outcomes = score
        .events()
        .iter()
        .filter(|event| event.key == StateKey::StaffLines)
        .map(|event| event.outcome)
        .collect::<Vec<_>>();
    assert_eq!(outcomes, vec![Outcome::Changed, Outcome::Redundant]);
}

#[test]
fn clef_written_in_one_subspine_applies_to_the_whole_track() {
    let document = parse("**kern\n*clefG2\n*^\n4c\t4e\n*clefF4\t*\n4C\t4E\n*v\t*v\n4C\n*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let line = data(&document, 1);
    assert_eq!(score.state_at(line, 1).track.clef, clef(ClefSign::F, 4));
    assert_eq!(
        score.state_at(data(&document, 2), 0).track.clef,
        clef(ClefSign::F, 4)
    );
}

#[test]
fn state_after_a_line_includes_its_own_tokens() {
    let document = parse("**kern\n*clefG2\n4c\n*clefF4\n*^\n4C\t4E\n*-\t*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let tandem = data(&document, 0) + 1;
    assert_eq!(score.state_at(tandem, 0).track.clef, clef(ClefSign::G, 2));
    assert_eq!(
        score.state_after(tandem, 0).track.clef,
        clef(ClefSign::F, 4)
    );
    // After a manipulator line, columns follow the new layout.
    assert_eq!(
        score.state_after(tandem + 1, 1).track.clef,
        clef(ClefSign::F, 4)
    );
}

#[test]
fn sibling_clefs_coalesce_when_equal_and_conflict_when_different() {
    let document = parse("**kern\n*^\n*clefF4\t*clefF4yy\n4C\t4E\n*v\t*v\n*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let clef_events = score
        .events()
        .iter()
        .filter(|event| event.key == StateKey::Clef && event.line_index > 1)
        .map(|event| (event.column, event.outcome))
        .collect::<Vec<_>>();
    assert_eq!(
        clef_events,
        vec![(0, Outcome::Changed), (1, Outcome::Coalesced)]
    );

    // Restating the clef in both subspines changes no state; Verovio draws
    // the visible copy again.
    let document = parse("**kern\n*clefF4\n*^\n*clefF4\t*clefF4yy\n4C\t4E\n*v\t*v\n*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let clef_events = score
        .events()
        .iter()
        .filter(|event| event.key == StateKey::Clef && event.line_index > 2)
        .map(|event| (event.column, event.outcome))
        .collect::<Vec<_>>();
    assert_eq!(
        clef_events,
        vec![(0, Outcome::Restated), (1, Outcome::Restated)]
    );
    // An ottava restated in both subspines draws nothing.
    let document = parse("**kern\n*8va\n*^\n*8va\t*8va\n4c\t4e\n*v\t*v\n*X8va\n*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let ottava_events = score
        .events()
        .iter()
        .filter(|event| event.key == StateKey::Ottava && event.line_index + 1 == data(&document, 0))
        .map(|event| event.outcome)
        .collect::<Vec<_>>();
    assert_eq!(ottava_events, vec![Outcome::Redundant, Outcome::Redundant]);

    assert!(matches!(
        errors("**kern\n*^\n*clefF4\t*clefG2\n4C\t4e\n*v\t*v\n*-\n").as_slice(),
        [StateError::Conflicting {
            key: StateKey::Clef,
            column: 2,
            ..
        }]
    ));
}

#[test]
fn ottava_is_track_state_and_must_close_before_the_track_ends() {
    let document = parse("**kern\n*^\n*8va\t*\n4c\t4e\n*\t*X8va\n*v\t*v\n*-\n");
    assert!(Score::resolve(&document, Framing::Complete).is_ok());

    assert!(matches!(
        errors("**kern\n*8va\n4c\n*-\n").as_slice(),
        [StateError::OpenAtEnd {
            key: StateKey::Ottava,
            opened_at: 2,
            ..
        }]
    ));
    assert!(matches!(
        errors("**kern\n*8va\n4c\n*8ba\n*X8ba\n*-\n").as_slice(),
        [
            StateError::SpanMismatch {
                key: StateKey::Ottava,
                ..
            },
            ..
        ]
    ));
}

/// Cases carried over from the `paired-tandem` rule, which this resolver
/// replaces.
#[test]
fn ottava_spans_follow_the_paired_tandem_cases() {
    for input in [
        "**kern\n*8va\n4c\n*X8va\n*-\n",
        "**kern\n*8va\n4c\n*8va\n4d\n*X8va\n*-\n",
        "**kern\t**kern\n*8va\t*\n4c\t4e\n*X8va\t*\n*-\t*-\n",
        "**kern\n*8va\n*^\n4c\t4e\n*v\t*v\n*X8va\n*-\n",
        "**kern\n*^\n*\t*8va\n4c\t4e\n*v\t*v\n4g\n*X8va\n*-\n",
        "**kern\n*^\n*8va\t*8va\n4c\t4e\n*v\t*v\n4g\n*X8va\n*-\n",
        "**kern\n*8va\n*^\n4c\t4e\n*\t*X8va\n4d\t4f\n*v\t*v\n*-\n",
        "**kern\n*8va\n*^\n4c\t4e\n*-\t*\n*X8va\n*-\n",
    ] {
        let document = parse(input);
        assert!(
            Score::resolve(&document, Framing::Complete).is_ok(),
            "{input}"
        );
    }

    let open_at_end = |line, column, opened_at| StateError::OpenAtEnd {
        line,
        column,
        key: StateKey::Ottava,
        opened_at,
    };
    let unmatched = |line, column| StateError::UnmatchedClose {
        line,
        column,
        key: StateKey::Ottava,
    };
    assert_eq!(errors("**kern\n*X8va\n4c\n*-\n"), [unmatched(2, 1)]);
    assert_eq!(
        errors("**kern\t**kern\n*\t*8va\n4c\t4e\n*-\t*-\n"),
        [open_at_end(4, 2, 2)]
    );
    assert_eq!(
        errors("**kern\t**kern\n*8va\t*\n4c\t4e\n*\t*X8va\n*-\t*-\n"),
        [unmatched(4, 2), open_at_end(5, 1, 2)]
    );
    assert_eq!(
        errors("**kern\n*8va\n*^\n4c\t4e\n*-\t*-\n"),
        [open_at_end(5, 2, 2)]
    );
    assert_eq!(
        errors("**kern\n*^\n*8va\t*\n4c\t4e\n*v\t*v\n*-\n"),
        [open_at_end(6, 1, 3)]
    );
    // A complete document must close its ottavas; an excerpt may end open.
    assert_eq!(errors("**kern\n*8va\n4c\n"), [open_at_end(3, 1, 2)]);
    let excerpt = parse("**kern\n*8va\n4c\n");
    assert!(Score::resolve(&excerpt, Framing::Excerpt).is_ok());
    // Terminating every spine ends an excerpt; ending one track early does not.
    let excerpt = parse("**kern\n*8va\n4c\n*-\n");
    assert!(Score::resolve(&excerpt, Framing::Excerpt).is_ok());
    let excerpt = parse("**kern\t**kern\n*\t*8va\n4c\t4e\n*\t*-\n4d\n*-\n");
    assert_eq!(
        Score::resolve(&excerpt, Framing::Excerpt).unwrap_err().0,
        [open_at_end(4, 2, 2)]
    );
    // One span at a time: another kind, opened or closed, is a mismatch.
    assert!(matches!(
        errors("**kern\n*8va\n*X15ma\n*-\n").as_slice(),
        [StateError::SpanMismatch { line: 3, .. }, ..]
    ));
}

#[test]
fn bridge_reopens_track_and_layer_state() {
    let document = parse(
        "**kern\t**kern\n*clefG2\t*clefF4\n*M3/4\t*M3/4\n*\t*^\n*\t*rscale:2\t*\n*\t*\t*cue\n4c\t4C\t4E\n*\t*\t*Xcue\n*\t*v\t*v\n*-\t*-\n",
    );
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let at = score.snapshot_before(data(&document, 0));
    let lines = bridge(&at.defaulted(), &at)
        .unwrap()
        .into_iter()
        .map(|line| {
            line.tokens()
                .iter()
                .map(|tandem| {
                    crate::serialize::omr::serialize_tandem(
                        tandem,
                        &crate::serialize::omr::OmrSerializeOptions {
                            tandem_policy: crate::serialize::TandemPolicy::HumdrumKern,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
                .join("\t")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lines,
        vec![
            "*clefG2\t*clefF4\t*",
            "*k[]\t*k[]\t*",
            "*M3/4\t*M3/4\t*",
            "*\t*\t*cue",
            "*\t*rscale:2\t*",
        ]
    );
    assert!(bridge(&at, &at).unwrap().is_empty());
    // Cue may stay on at the end of a document; closing leaves it.
    assert!(bridge(&at, &at.closed()).unwrap().is_empty());
}

fn tandem_lines(lines: Vec<TandemInterpretationLine>) -> Vec<String> {
    lines
        .into_iter()
        .map(|line| {
            line.tokens()
                .iter()
                .map(|tandem| {
                    crate::serialize::omr::serialize_tandem(
                        tandem,
                        &crate::serialize::omr::OmrSerializeOptions {
                            tandem_policy: crate::serialize::TandemPolicy::HumdrumKern,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
                .join("\t")
        })
        .collect()
}

#[test]
fn closing_a_snapshot_closes_its_ottavas() {
    let document =
        parse("**kern\t**kern\n*^\t*\n*\t*8va\t*\n4c\t4e\t4G\n*v\t*v\t*\n*X8va\t*\n*-\t*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let at = score.snapshot_before(data(&document, 0));
    assert_eq!(
        tandem_lines(bridge(&at, &at.closed()).unwrap()),
        ["*X8va\t*\t*"]
    );
    // One past the last line is the state after it.
    assert_eq!(
        score.snapshot_before(document.lines().len()),
        score.snapshot_after_last()
    );
}

#[test]
fn track_view_holds_one_column_per_track() {
    let document = parse(
        "**kern\t**kern\n*clefF4\t*clefG2\n*^\t*\n*\t*cue\t*\n4c\t4e\t4g\n*v\t*v\t*\n*-\t*-\n",
    );
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let at = score.snapshot_before(data(&document, 0));
    assert_eq!(at.width(), 3);
    let tracks = at.tracks();
    assert_eq!((tracks.width(), tracks.track_count()), (2, 2));
    // Layer state stays behind in the split columns; parsing supplies the
    // default key signature.
    assert_eq!(
        tandem_lines(bridge(&tracks.defaulted(), &tracks).unwrap()),
        ["*clefF4\t*clefG2", "*k[]\t*k[]"]
    );
}

#[test]
fn with_takes_track_slots_by_track_and_layer_slots_by_column() {
    let document = parse("**kern\t**kern\n*clefF4\t*clefG2\n*k[f#]\t*k[f#]\n*^\t*\n*\t*cue\t*\n4c\t4e\t4g\n*v\t*v\t*\n*-\t*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let split = score.snapshot_before(data(&document, 0));
    let header = score.snapshot_before(1);
    let clefs = header.with(&split, &[StateKey::Clef]).unwrap();
    assert_eq!(
        tandem_lines(bridge(&header, &clefs).unwrap()),
        ["*clefF4\t*clefG2"]
    );
    // Layer slots need the same columns.
    assert_eq!(
        header.with(&split, &[StateKey::Cue]),
        Err(BridgeError::LayoutMismatch)
    );
    assert_eq!(
        tandem_lines(
            bridge(
                &split.defaulted(),
                &split.defaulted().with(&split, &[StateKey::Cue]).unwrap()
            )
            .unwrap()
        ),
        ["*\t*cue\t*"]
    );
}

#[test]
fn seam_bridge_writes_changed_staff_definition_and_keeps_unstated_slots() {
    let left =
        parse("**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n4C\t4c\n*-\t*-\n");
    let right = parse("**kern\t**kern\n*clefG2\t*\n*k[b-]\t*k[b-]\n4c\t4d\n*-\t*-\n");
    let left_score = Score::resolve(&left, Framing::Excerpt).unwrap();
    let right_score = Score::resolve(&right, Framing::Excerpt).unwrap();
    let from = left_score.snapshot_before(left.lines().len() - 1);
    let to = right_score.snapshot_before(data(&right, 0));
    // The right page restates one clef and the key; its unstated clef and
    // meter continue the left page's.
    assert_eq!(
        tandem_lines(bridge_seam(&from, &to, &StateKey::STAFF_DEFINITION).unwrap()),
        ["*clefG2\t*", "*k[b-]\t*k[b-]"]
    );
}

#[test]
fn toward_reaches_every_slot_a_tandem_can_write() {
    // The end of the first pass: F clef, pedal down, ottava open.
    let document =
        parse("**kern\n*clefG2\n*M2/4\n=1\n4c\n*clefF4\n*ped\n*8va\n4C\n*X8va\n*Xped\n*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let end_of_pass = score.snapshot_before(data(&document, 1) + 1);
    // Back to the start, where nothing is declared yet: the clef and meter
    // stay, the pedal and ottava close.
    let start = score.snapshot_before(0);
    assert_eq!(
        tandem_lines(bridge(&end_of_pass, &end_of_pass.toward(&start).unwrap()).unwrap()),
        ["*X8va", "*Xped"]
    );
    // Back to the first measure: G clef again.
    let first_measure = score.snapshot_before(data(&document, 0));
    assert_eq!(
        tandem_lines(bridge(&end_of_pass, &end_of_pass.toward(&first_measure).unwrap()).unwrap()),
        ["*clefG2", "*X8va", "*Xped"]
    );
}

#[test]
fn a_join_of_implied_columns_is_undone_by_conflicting_values() {
    // Columns one and two later merge, so they read as one staff that the
    // page starts inside of.
    let document = parse("*clefG2\t*clefG2\t*clefF4\n4c\t4e\t4G\n*v\t*v\t*\n2c\t2G\n");
    assert!(Score::resolve(&document, Framing::Excerpt).is_ok());
    // Their clefs disagree, so the merge crosses staves.
    assert_eq!(
        errors("*clefF4\t*clefG2\t*clefG2\n4c\t4e\t4G\n*v\t*v\t*\n2c\t2G\n"),
        [StateError::Topology(TopologyError::CrossTrackMerge {
            line: 3,
            column: 2
        })]
    );
}

#[test]
fn staff_placement_is_layer_state() {
    // Two verses of a split lyrics spine, one above the staff and one below.
    let document =
        parse("**kern\t**text\n*\t*^\n*\t*above\t*below\n4c\tla\tli\n*\t*v\t*v\n*-\t*-\n");
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let line = data(&document, 0);
    assert_eq!(
        score.state_at(line, 1).layer.staff_placement,
        Some(crate::StaffPlacement::Above)
    );
    assert_eq!(
        score.state_at(line, 2).layer.staff_placement,
        Some(crate::StaffPlacement::Below)
    );
}
