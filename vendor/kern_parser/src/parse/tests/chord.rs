use crate::types::{
    Arpeggio, Articulation, AuxInterval, Bowing, Breath, ChordBeams, ChordModifiers, DiatonicStep,
    DurationValue, Elision, Fermata, FermataPlacement, ForcedAccidental, Harmonic, KernContent,
    MordentDirection, Ornament, Phrase, Placed, Slur, StemDirection, Tie, TrillLine, Turn,
    TurnForm, TurnIntervals, TurnTiming,
};
use crate::*;

#[test]
fn parses_basic_chord_in_source_order_and_exposes_pitch_order() {
    let chord = Chord::try_from("4g 4c 4e").unwrap();

    assert_eq!(chord.notes.len(), 3);
    assert_eq!(
        chord
            .notes
            .iter()
            .map(|note| note.pitch.step)
            .collect::<Vec<_>>(),
        vec![DiatonicStep::G, DiatonicStep::C, DiatonicStep::E]
    );
    assert_eq!(
        chord
            .notes_by_pitch()
            .into_iter()
            .map(|note| note.pitch.step)
            .collect::<Vec<_>>(),
        vec![DiatonicStep::C, DiatonicStep::E, DiatonicStep::G]
    );
    assert_eq!(chord.lowest_note().unwrap().pitch.step, DiatonicStep::C);
    assert_eq!(chord.highest_note().unwrap().pitch.step, DiatonicStep::G);
    assert!(chord.notes.iter().all(|note| {
        note.duration
            == Duration {
                value: DurationValue::Reciprocal(4),
                augmentation_dots: 0,
            }
    }));
    assert_eq!(chord.modifiers, ChordModifiers::default());
}

#[test]
fn parses_compact_chord_and_inherits_first_duration() {
    let chord = Chord::try_from("4G C E").unwrap();

    assert_eq!(chord.notes.len(), 3);
    assert_eq!(
        chord
            .notes
            .iter()
            .map(|note| note.pitch.step)
            .collect::<Vec<_>>(),
        vec![DiatonicStep::G, DiatonicStep::C, DiatonicStep::E]
    );
    assert!(chord.notes.iter().all(|note| {
        note.duration
            == Duration {
                value: DurationValue::Reciprocal(4),
                augmentation_dots: 0,
            }
    }));
}

#[test]
fn parses_compact_chord_with_note_modifiers() {
    let chord = Chord::try_from("[4f#X a-] ccn_").unwrap();

    assert!(chord.notes.iter().all(|note| {
        note.duration
            == Duration {
                value: DurationValue::Reciprocal(4),
                augmentation_dots: 0,
            }
    }));
    assert_eq!(
        chord.notes[0].accidental,
        Some(crate::types::Accidental::Sharp)
    );
    assert_eq!(
        chord.notes[0].modifiers.forced_accidental,
        Some(ForcedAccidental)
    );
    assert_eq!(chord.notes[0].modifiers.ties, vec![Tie::Start]);
    assert_eq!(
        chord.notes[1].accidental,
        Some(crate::types::Accidental::Flat)
    );
    assert_eq!(chord.notes[1].modifiers.ties, vec![Tie::End]);
    assert_eq!(
        chord.notes[2].accidental,
        Some(crate::types::Accidental::Natural)
    );
    assert_eq!(chord.notes[2].modifiers.ties, vec![Tie::Continue]);
}

#[test]
fn parses_compact_chord_with_canonical_cross_staff_markers() {
    let chord = Chord::try_from("4c e> g#<L").unwrap();

    assert_eq!(
        chord
            .notes
            .iter()
            .map(|note| note.modifiers.cross_staff)
            .collect::<Vec<_>>(),
        vec![
            None,
            Some(crate::types::CrossStaffPlacement::Above),
            Some(crate::types::CrossStaffPlacement::Below),
        ]
    );
    assert_eq!(chord.modifiers.beams, ChordBeams::Start { level: 1 });

    let tied = Chord::try_from("12A[</J").unwrap();
    assert_eq!(
        tied.notes[0].modifiers.ties,
        vec![Tie::PlacedOpen {
            phase: crate::types::TieOpenPhase::Start,
            resolution: crate::types::TieResolution::Ordinary,
            placement: crate::types::TiePlacement::Below,
        }]
    );
    assert_eq!(tied.notes[0].modifiers.cross_staff, None);
    assert_eq!(tied.modifiers.stem_direction, Some(StemDirection::Up));
    assert_eq!(tied.modifiers.beams, ChordBeams::End { level: 1 });

    let beamed = Chord::try_from("16C#JJ<)").unwrap();
    assert_eq!(beamed.notes[0].modifiers.cross_staff, None);
    assert_eq!(beamed.modifiers.beams, ChordBeams::End { level: 2 });

    let articulated = Chord::try_from("16e^LL>").unwrap();
    assert_eq!(articulated.notes[0].modifiers.cross_staff, None);
    assert_eq!(articulated.modifiers.beams, ChordBeams::Start { level: 2 });
}

#[test]
fn parses_compact_chord_with_mixed_explicit_matching_duration() {
    let chord = Chord::try_from("4c e 4g").unwrap();

    assert!(chord.notes.iter().all(|note| {
        note.duration
            == Duration {
                value: DurationValue::Reciprocal(4),
                augmentation_dots: 0,
            }
    }));
}

#[test]
fn parses_single_note_chord() {
    let chord = Chord::try_from("4c").unwrap();

    assert_eq!(chord.notes, vec![Note::try_from("4c").unwrap()]);
    assert_eq!(chord.modifiers, ChordModifiers::default());
}

#[test]
fn parses_chord_note_accidentals_rational_durations_and_ties() {
    let chord = Chord::try_from("[3%5..f# 3%5..a-] 3%5..ccn_").unwrap();

    assert_eq!(
        chord
            .notes
            .iter()
            .map(|note| note.pitch.step)
            .collect::<Vec<_>>(),
        vec![DiatonicStep::F, DiatonicStep::A, DiatonicStep::C]
    );
    assert!(chord.notes.iter().all(|note| {
        note.duration
            == Duration {
                value: DurationValue::Rational {
                    numerator: 3,
                    denominator: 5,
                },
                augmentation_dots: 2,
            }
    }));
    assert_eq!(
        chord.notes[0].accidental,
        Some(crate::types::Accidental::Sharp)
    );
    assert_eq!(chord.notes[0].modifiers.ties, vec![Tie::Start]);
    assert_eq!(
        chord.notes[1].accidental,
        Some(crate::types::Accidental::Flat)
    );
    assert_eq!(chord.notes[1].modifiers.ties, vec![Tie::End]);
    assert_eq!(
        chord.notes[2].accidental,
        Some(crate::types::Accidental::Natural)
    );
    assert_eq!(chord.notes[2].modifiers.ties, vec![Tie::Continue]);
}

#[test]
fn parses_chord_modifiers_from_note_subtokens() {
    let chord = Chord::try_from("(&{4c:'zv/LLKK;,oMT 4e&) 4g}").unwrap();

    assert_eq!(chord.modifiers.arpeggio, Some(Arpeggio::StaffLocal));
    assert_eq!(
        chord.modifiers.articulations,
        vec![
            Placed::from(Articulation::Staccato),
            Placed::from(Articulation::Generic)
        ]
    );
    assert_eq!(
        chord.modifiers.beams,
        ChordBeams::StartAndPartialRight {
            start_level: 2,
            partial_right_level: 2
        }
    );
    assert_eq!(chord.modifiers.stem_direction, Some(StemDirection::Up));
    assert_eq!(chord.modifiers.breath, Some(Breath.into()));
    assert_eq!(chord.modifiers.harmonic, Some(Harmonic.into()));
    assert_eq!(
        chord.modifiers.ornaments,
        vec![
            Placed::from(Ornament::Mordent {
                direction: MordentDirection::Upper,
                interval: Some(AuxInterval::WholeTone),
                doubled: false,
            }),
            Placed::from(Ornament::Trill {
                interval: AuxInterval::WholeTone,
                line: TrillLine::None,
            }),
        ]
    );
    assert_eq!(chord.modifiers.fermata, Some(Fermata::default()));
    assert_eq!(
        chord.modifiers.slurs,
        vec![
            Slur::Open {
                elided: false,
                depth: 0,
                count: 1,
                placement: None,
                linked: false,
            },
            Slur::Close {
                elided: true,
                depth: 1,
                count: 1,
                linked: false,
            },
        ]
    );
    assert_eq!(
        chord.modifiers.phrases,
        vec![
            Phrase::Open {
                elision: Elision { depth: 1 },
            },
            Phrase::Close {
                elision: Elision { depth: 0 },
            },
        ]
    );
    assert_eq!(chord.modifiers.bowing, Some(Bowing::UpBow.into()));
}

#[test]
fn parses_mordent_spellings_as_single_structured_ornaments() {
    let cases = [
        (
            "4cM",
            mordent(MordentDirection::Upper, Some(AuxInterval::WholeTone), false),
        ),
        (
            "4cm",
            mordent(MordentDirection::Upper, Some(AuxInterval::Semitone), false),
        ),
        (
            "4cW",
            mordent(MordentDirection::Lower, Some(AuxInterval::WholeTone), false),
        ),
        (
            "4cw",
            mordent(MordentDirection::Lower, Some(AuxInterval::Semitone), false),
        ),
        (
            "4cMM",
            mordent(MordentDirection::Upper, Some(AuxInterval::WholeTone), true),
        ),
        (
            "4cmm",
            mordent(MordentDirection::Upper, Some(AuxInterval::Semitone), true),
        ),
        (
            "4cWW",
            mordent(MordentDirection::Lower, Some(AuxInterval::WholeTone), true),
        ),
        (
            "4cww",
            mordent(MordentDirection::Lower, Some(AuxInterval::Semitone), true),
        ),
        ("4cMm", mordent(MordentDirection::Upper, None, false)),
        ("4cmM", mordent(MordentDirection::Upper, None, false)),
        ("4cWw", mordent(MordentDirection::Lower, None, false)),
        ("4cwW", mordent(MordentDirection::Lower, None, false)),
    ];

    for (raw, expected) in cases {
        let chord = Chord::try_from(raw).unwrap();
        assert_eq!(
            chord.modifiers.ornaments,
            vec![Placed::from(expected)],
            "{raw} parsed with unexpected ornaments",
        );
    }
}

#[test]
fn rejects_invalid_mordent_spellings() {
    for raw in ["4cMMM", "4cmmm", "4cWWW", "4cwww", "4cMW", "4cWM", "4cMmW"] {
        assert!(
            matches!(
                Chord::try_from(raw).unwrap_err(),
                ParseError::InvalidNote { .. }
            ),
            "{raw} should be an invalid note"
        );
    }
}

#[test]
fn parses_and_merges_arpeggio_scope_without_inventing_or_losing_cross_staff_intent() {
    let cases = [
        ("4c:", Arpeggio::StaffLocal),
        ("4c::", Arpeggio::CrossStaff),
        ("4c:::", Arpeggio::CrossStaff),
        ("4c: 4e:", Arpeggio::StaffLocal),
        ("4c: 4e::", Arpeggio::CrossStaff),
        ("4c:: 4e:", Arpeggio::CrossStaff),
    ];

    for (raw, expected) in cases {
        let chord = Chord::try_from(raw).unwrap();
        assert_eq!(chord.modifiers.arpeggio, Some(expected), "{raw}");
    }
}

#[test]
fn parses_vhv_turn_spellings_as_single_structured_ornaments() {
    let cases = [
        ("4cS", turn(TurnForm::Regular, TurnTiming::Delayed, None)),
        ("4c$", turn(TurnForm::Inverted, TurnTiming::Delayed, None)),
        (
            "4cSSS",
            turn(
                TurnForm::Regular,
                TurnTiming::Delayed,
                Some((AuxInterval::WholeTone, AuxInterval::WholeTone)),
            ),
        ),
        (
            "4cSss",
            turn(
                TurnForm::Regular,
                TurnTiming::Delayed,
                Some((AuxInterval::Semitone, AuxInterval::Semitone)),
            ),
        ),
        (
            "4cSSs",
            turn(
                TurnForm::Regular,
                TurnTiming::Delayed,
                Some((AuxInterval::WholeTone, AuxInterval::Semitone)),
            ),
        ),
        (
            "4cSsS",
            turn(
                TurnForm::Regular,
                TurnTiming::Delayed,
                Some((AuxInterval::Semitone, AuxInterval::WholeTone)),
            ),
        ),
        (
            "4csSSS",
            turn(
                TurnForm::Regular,
                TurnTiming::OnNote,
                Some((AuxInterval::WholeTone, AuxInterval::WholeTone)),
            ),
        ),
        (
            "4c$Ss",
            turn(
                TurnForm::Inverted,
                TurnTiming::Delayed,
                Some((AuxInterval::Semitone, AuxInterval::WholeTone)),
            ),
        ),
        (
            "4cs$sS",
            turn(
                TurnForm::Inverted,
                TurnTiming::OnNote,
                Some((AuxInterval::WholeTone, AuxInterval::Semitone)),
            ),
        ),
    ];

    for (raw, expected) in cases {
        let chord = Chord::try_from(raw).unwrap();
        assert_eq!(
            chord.modifiers.ornaments,
            vec![Placed::from(expected)],
            "{raw} parsed with unexpected ornaments",
        );
    }
}

#[test]
fn parses_trill_spellings_as_single_structured_ornaments() {
    let cases = [
        (
            "4cT",
            Ornament::Trill {
                interval: AuxInterval::WholeTone,
                line: TrillLine::None,
            },
        ),
        (
            "4ct",
            Ornament::Trill {
                interval: AuxInterval::Semitone,
                line: TrillLine::None,
            },
        ),
        (
            "4cTT",
            Ornament::Trill {
                interval: AuxInterval::WholeTone,
                line: TrillLine::Start,
            },
        ),
        (
            "4ctt",
            Ornament::Trill {
                interval: AuxInterval::Semitone,
                line: TrillLine::Start,
            },
        ),
        (
            "4cTTT",
            Ornament::Trill {
                interval: AuxInterval::WholeTone,
                line: TrillLine::Continuation,
            },
        ),
        (
            "4cttt",
            Ornament::Trill {
                interval: AuxInterval::Semitone,
                line: TrillLine::Continuation,
            },
        ),
    ];

    for (raw, expected) in cases {
        let chord = Chord::try_from(raw).unwrap();
        assert_eq!(
            chord.modifiers.ornaments,
            vec![Placed::from(expected)],
            "{raw} parsed with unexpected ornaments",
        );
    }
}

#[test]
fn rejects_conflicting_trill_spellings_on_one_token() {
    for raw in ["4cTt", "4ctT", "4cTTt", "4cTTTT", "4ctttt"] {
        assert!(
            matches!(
                Chord::try_from(raw).unwrap_err(),
                ParseError::InvalidNote { .. }
            ),
            "{raw} should be an invalid note"
        );
    }
}

#[test]
fn rejects_invalid_vhv_turn_spellings() {
    for raw in ["4cs", "4csT", "4cSS", "4c$s", "4cS$", "4cSSSS"] {
        assert!(
            matches!(
                Chord::try_from(raw).unwrap_err(),
                ParseError::InvalidNote { .. }
            ),
            "{raw} should be an invalid note"
        );
    }
}

fn turn(
    form: TurnForm,
    timing: TurnTiming,
    intervals: Option<(AuxInterval, AuxInterval)>,
) -> Ornament {
    Ornament::Turn(Turn {
        form,
        timing,
        intervals: intervals.map(|(upper, lower)| TurnIntervals { upper, lower }),
    })
}

fn mordent(direction: MordentDirection, interval: Option<AuxInterval>, doubled: bool) -> Ornament {
    Ornament::Mordent {
        direction,
        interval,
        doubled,
    }
}

#[test]
fn folds_duplicate_slur_markers_in_chords_into_multiplicity() {
    let chord = Chord::try_from("4c( 4e( 4g(").unwrap();

    assert_eq!(
        chord.modifiers.slurs,
        vec![Slur::Open {
            elided: false,
            depth: 0,
            count: 3,
            placement: None,
            linked: false,
        }]
    );

    let chord = Chord::try_from("4c&( 4e&( 4g&&(").unwrap();

    assert_eq!(
        chord.modifiers.slurs,
        vec![
            Slur::Open {
                elided: true,
                depth: 1,
                count: 2,
                placement: None,
                linked: false,
            },
            Slur::Open {
                elided: true,
                depth: 2,
                count: 1,
                placement: None,
                linked: false,
            },
        ]
    );
}

#[test]
fn parses_rdf_slur_placement_after_the_open_marker() {
    let chord = Chord::try_from("((>4c").unwrap();

    assert_eq!(
        chord.modifiers.slurs,
        vec![
            Slur::Open {
                elided: false,
                depth: 0,
                count: 1,
                placement: None,
                linked: false,
            },
            Slur::Open {
                elided: false,
                depth: 0,
                count: 1,
                placement: Some(crate::types::SlurPlacement::Above),
                linked: false,
            },
        ]
    );

    let chord = Chord::try_from("&(<4c").unwrap();
    assert_eq!(
        chord.modifiers.slurs,
        vec![Slur::Open {
            elided: true,
            depth: 1,
            count: 1,
            placement: Some(crate::types::SlurPlacement::Below),
            linked: false,
        }]
    );
}

#[test]
fn kern_line_parses_chord_data_tokens() {
    let line = KernLine::parse(3, ".\t4c 4e 4g\t4r").unwrap();

    assert_eq!(
        line,
        KernLine::Data(vec![
            SpineContent::Kern(KernContent::Null),
            SpineContent::Kern(KernContent::Chord(Chord::try_from("4c 4e 4g").unwrap())),
            SpineContent::Kern(KernContent::Rest(Rest::try_from("4r").unwrap())),
        ])
    );
}

#[test]
fn kern_line_parses_compact_chord_data_tokens() {
    let line = KernLine::parse(3, ".\t4c e g\t4r").unwrap();

    assert_eq!(
        line,
        KernLine::Data(vec![
            SpineContent::Kern(KernContent::Null),
            SpineContent::Kern(KernContent::Chord(Chord::try_from("4c e g").unwrap())),
            SpineContent::Kern(KernContent::Rest(Rest::try_from("4r").unwrap())),
        ])
    );
}

#[test]
fn normalizes_chord_beam_markers() {
    let chord = Chord::try_from("4cLJ").unwrap();
    assert_eq!(chord.modifiers.beams, ChordBeams::None);
    assert_eq!(chord.modifiers.beams.beam_start_count(), 0);
    assert_eq!(chord.modifiers.beams.beam_end_count(), 0);
    assert_eq!(chord.modifiers.beams.partial_left_count(), 0);
    assert_eq!(chord.modifiers.beams.partial_right_count(), 0);
    assert!(chord.modifiers.beams.is_empty());

    let chord = Chord::try_from("4cLLJ").unwrap();
    assert_eq!(chord.modifiers.beams, ChordBeams::Start { level: 1 });
    assert_eq!(chord.modifiers.beams.beam_start_count(), 1);

    let chord = Chord::try_from("4cLJJ").unwrap();
    assert_eq!(chord.modifiers.beams, ChordBeams::End { level: 1 });
    assert_eq!(chord.modifiers.beams.beam_end_count(), 1);

    let chord = Chord::try_from("4cJk").unwrap();
    assert_eq!(
        chord.modifiers.beams,
        ChordBeams::EndAndPartialLeft {
            end_level: 1,
            partial_left_level: 1
        }
    );

    let chord = Chord::try_from("4cJK").unwrap();
    assert_eq!(
        chord.modifiers.beams,
        ChordBeams::EndAndPartialLeft {
            end_level: 1,
            partial_left_level: 1
        }
    );

    let chord = Chord::try_from("4cJKK").unwrap();
    assert_eq!(
        chord.modifiers.beams,
        ChordBeams::EndAndPartialLeft {
            end_level: 1,
            partial_left_level: 2
        }
    );

    let chord = Chord::try_from("4cLJK").unwrap();
    assert_eq!(chord.modifiers.beams, ChordBeams::PartialLeft { level: 1 });

    let chord = Chord::try_from("4cLJJKK").unwrap();
    assert_eq!(
        chord.modifiers.beams,
        ChordBeams::EndAndPartialLeft {
            end_level: 1,
            partial_left_level: 2
        }
    );

    let chord = Chord::try_from("4cLK").unwrap();
    assert_eq!(
        chord.modifiers.beams,
        ChordBeams::StartAndPartialRight {
            start_level: 1,
            partial_right_level: 1
        }
    );

    let chord = Chord::try_from("4cLk").unwrap();
    assert_eq!(
        chord.modifiers.beams,
        ChordBeams::StartAndPartialRight {
            start_level: 1,
            partial_right_level: 1
        }
    );

    let chord = Chord::try_from("4cLLJk").unwrap();
    assert_eq!(
        chord.modifiers.beams,
        ChordBeams::StartAndPartialRight {
            start_level: 1,
            partial_right_level: 1
        }
    );

    let chord = Chord::try_from("4ck").unwrap();
    assert_eq!(chord.modifiers.beams, ChordBeams::PartialLeft { level: 1 });
    assert_eq!(chord.modifiers.beams.partial_left_count(), 1);

    let chord = Chord::try_from("4cK").unwrap();
    assert_eq!(chord.modifiers.beams, ChordBeams::PartialLeft { level: 1 });
    assert_eq!(chord.modifiers.beams.partial_left_count(), 1);

    let chord = Chord::try_from("4cKK").unwrap();
    assert_eq!(chord.modifiers.beams, ChordBeams::PartialLeft { level: 2 });
    assert_eq!(chord.modifiers.beams.partial_left_count(), 2);
}

#[test]
fn rejects_invalid_chord_beam_marker_combinations() {
    for raw in ["4cLkK", "4cJkK"] {
        assert!(matches!(
            Chord::try_from(raw).unwrap_err(),
            ParseError::InvalidNote { .. }
        ));
    }
}

#[test]
fn chord_parser_rejects_invalid_chord_shapes() {
    for raw in ["4c  4e", "4c ", " 4c 4e", "4c\t4e", "4c 8e", "4c 8e 4g"] {
        assert_eq!(
            Chord::try_from(raw).unwrap_err(),
            ParseError::InvalidChord {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn chord_parser_rejects_compact_chord_without_initial_duration() {
    assert!(matches!(
        Chord::try_from("c e g").unwrap_err(),
        ParseError::InvalidNote { raw, .. } if raw == "c"
    ));
}

#[test]
fn chord_parser_rejects_invalid_note_subtokens() {
    for raw in ["4r 4e", "4c 4r 4g", "4c? 4e", "4c 4eiL>"] {
        assert!(matches!(
            Chord::try_from(raw).unwrap_err(),
            ParseError::InvalidNote { .. }
        ));
    }
}

#[test]
fn parses_rdf_fermata_placement_after_the_fermata_marker() {
    let fermata = |token: &str| Chord::try_from(token).unwrap().modifiers.fermata;

    assert_eq!(fermata("4c;"), Some(Fermata { placement: None }));
    assert_eq!(
        fermata("4c;<"),
        Some(Fermata {
            placement: Some(FermataPlacement::Below)
        })
    );
    assert_eq!(
        fermata("4c 4e;>"),
        Some(Fermata {
            placement: Some(FermataPlacement::Above)
        })
    );
}

#[test]
fn position_signifier_before_the_fermata_stays_cross_staff() {
    let chord = Chord::try_from("4c<;").unwrap();

    assert_eq!(chord.modifiers.fermata, Some(Fermata { placement: None }));
    assert_eq!(
        chord.notes[0].modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Below)
    );
}
