use crate::types::{DiatonicStep, DurationValue, KernContent, Tie};
use crate::*;

#[test]
fn parses_basic_note_tokens() {
    assert_eq!(
        Note::try_from("4c").unwrap(),
        Note {
            duration: Duration {
                value: DurationValue::Reciprocal(4),
                augmentation_dots: 0,
            },
            pitch: Pitch {
                step: DiatonicStep::C,
                octave: 4,
            },
            accidental: None,
            modifiers: Default::default(),
        }
    );
}

#[test]
fn parses_long_note_durations() {
    let cases = [
        ("0c", DurationValue::Breve, 0),
        ("00.c", DurationValue::Long, 1),
        ("000..c", DurationValue::Maxima, 2),
    ];

    for (raw, value, augmentation_dots) in cases {
        assert_eq!(
            Note::try_from(raw).unwrap().duration,
            Duration {
                value,
                augmentation_dots,
            }
        );
    }
}

#[test]
fn parses_note_accidentals_rational_durations_and_ties() {
    assert_eq!(
        Note::try_from("[3%5..ff#]").unwrap(),
        Note {
            duration: Duration {
                value: DurationValue::Rational {
                    numerator: 3,
                    denominator: 5,
                },
                augmentation_dots: 2,
            },
            pitch: Pitch {
                step: DiatonicStep::F,
                octave: 5,
            },
            accidental: Some(crate::types::Accidental::Sharp),
            modifiers: crate::types::NoteModifiers {
                ties: vec![Tie::Start, Tie::End],
                forced_accidental: None,
                invisible_accidental: false,
                editorial_interpretation: None,
                invisible: false,
                cross_staff: None,
                user_signifiers: Vec::new(),
            },
        }
    );
}

#[test]
fn note_parser_rejects_pitches_outside_supported_range() {
    for raw in ["4CCCCCC", "4BBBBBB", "4cccccc", "4bbbbbb"] {
        assert!(Note::try_from(raw).is_ok(), "{raw} should be valid");
    }

    for raw in ["4CCCCCCC", "4GGGGGGG", "4ccccccc", "4ggggggg"] {
        assert!(matches!(
            Note::try_from(raw).unwrap_err(),
            ParseError::InvalidNote { .. }
        ));
    }
}

#[test]
fn parses_note_user_signifiers() {
    assert_eq!(
        Note::try_from("4f#i").unwrap().modifiers,
        crate::types::NoteModifiers {
            ties: Vec::new(),
            forced_accidental: None,
            invisible_accidental: false,
            editorial_interpretation: None,
            invisible: false,
            cross_staff: None,
            user_signifiers: vec![crate::types::UserSignifier {
                marker: "i".to_owned(),
            }],
        }
    );
    assert_eq!(
        Note::try_from("4f#N").unwrap().modifiers.user_signifiers,
        vec![crate::types::UserSignifier {
            marker: "N".to_owned(),
        }]
    );
}

#[test]
fn parses_disjunct_and_linked_ties() {
    assert_eq!(
        Note::try_from("4c[[__]]").unwrap().modifiers.ties,
        vec![Tie::DisjunctStart, Tie::DisjunctContinue, Tie::DisjunctEnd,]
    );

    let start = Note::try_from("N[8f<").unwrap();
    assert_eq!(start.modifiers.ties, vec![Tie::LinkedStart]);
    assert_eq!(
        start.modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Below)
    );

    assert_eq!(
        Note::try_from("2.fN]").unwrap().modifiers.ties,
        vec![Tie::LinkedEnd]
    );
    assert_eq!(
        Note::try_from("4fN_").unwrap().modifiers.ties,
        vec![Tie::LinkedContinue]
    );
}

#[test]
fn parses_tie_placement_after_start_and_continuation_markers() {
    let above = Note::try_from("4c[>").unwrap();
    assert_eq!(
        above.modifiers.ties,
        vec![Tie::PlacedOpen {
            phase: crate::types::TieOpenPhase::Start,
            resolution: crate::types::TieResolution::Ordinary,
            placement: crate::types::TiePlacement::Above,
        }]
    );

    let below = Note::try_from("4c_<").unwrap();
    assert_eq!(
        below.modifiers.ties,
        vec![Tie::PlacedOpen {
            phase: crate::types::TieOpenPhase::Continue,
            resolution: crate::types::TieResolution::Ordinary,
            placement: crate::types::TiePlacement::Below,
        }]
    );

    let linked = Note::try_from("4cN[>").unwrap();
    assert_eq!(
        linked.modifiers.ties[0].placement(),
        Some(crate::types::TiePlacement::Above)
    );
    assert_eq!(
        linked.modifiers.ties[0].resolution(),
        crate::types::TieResolution::Linked
    );
}

#[test]
fn parses_cross_staff_placement_markers() {
    assert_eq!(
        Note::try_from("4c>").unwrap().modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Above)
    );
    assert_eq!(
        Note::try_from("4c<").unwrap().modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Below)
    );

    assert_eq!(
        Note::try_from("4f#>").unwrap().modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Above)
    );
    assert_eq!(
        Note::try_from("4fn<").unwrap().modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Below)
    );
    assert_eq!(
        Note::try_from("4c#X<").unwrap().modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Below)
    );

    let forced_cross_staff = Note::try_from("4f#X<").unwrap();
    assert_eq!(
        forced_cross_staff.modifiers.forced_accidental,
        Some(crate::types::ForcedAccidental)
    );
    assert_eq!(
        forced_cross_staff.modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Below)
    );

    let hidden_cross_staff = Note::try_from("4f#y<").unwrap();
    assert!(hidden_cross_staff.modifiers.invisible_accidental);
    assert_eq!(
        hidden_cross_staff.modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Below)
    );

    let dead_forced = Note::try_from("4f#<X").unwrap();
    assert_eq!(dead_forced.modifiers.forced_accidental, None);
    assert_eq!(
        dead_forced.modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Below)
    );

    let dead_hidden = Note::try_from("4f#<y").unwrap();
    assert!(!dead_hidden.modifiers.invisible_accidental);
    assert_eq!(
        dead_hidden.modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Below)
    );

    let tie_below = Note::try_from("12A[<").unwrap();
    assert_eq!(
        tie_below.modifiers.ties,
        vec![Tie::PlacedOpen {
            phase: crate::types::TieOpenPhase::Start,
            resolution: crate::types::TieResolution::Ordinary,
            placement: crate::types::TiePlacement::Below,
        }]
    );
    assert_eq!(tie_below.modifiers.cross_staff, None);

    let tie_end_below = Note::try_from("12A]<").unwrap();
    assert_eq!(tie_end_below.modifiers.ties, vec![Tie::End]);
    assert_eq!(tie_end_below.modifiers.cross_staff, None);

    let modifiers = Note::try_from("4f#>i").unwrap().modifiers;
    assert_eq!(
        modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Above)
    );
    assert_eq!(
        modifiers.user_signifiers,
        vec![crate::types::UserSignifier {
            marker: "i".to_owned(),
        }]
    );
}

#[test]
fn cross_staff_placement_marker_must_be_unique() {
    for raw in ["4c>>", "4c<<", "4c><", "4c<>"] {
        assert!(matches!(
            Note::try_from(raw).unwrap_err(),
            ParseError::InvalidNote { .. }
        ));
    }
}

#[test]
fn cross_staff_placement_marker_must_follow_note_core() {
    for raw in [">4c", "4>c", "4c#i>", "4c#q>", "4c>#"] {
        let error = Note::try_from(raw).unwrap_err();
        assert!(matches!(
            error,
            ParseError::InvalidNote {
                raw: ref error_raw,
                location: InvalidNoteLocation {
                    position: Some(_),
                    character: Some('<') | Some('>') | Some('#') | Some('X') | Some('y'),
                },
            } if error_raw == raw
        ));
    }
}

#[test]
fn parses_invisible_note_marker() {
    assert!(Note::try_from("4byy").unwrap().modifiers.invisible);
    let tied_flat = Note::try_from("[3dd-yy").unwrap();
    assert!(tied_flat.modifiers.invisible);
    assert!(!tied_flat.modifiers.invisible_accidental);
    assert_eq!(tied_flat.accidental, Some(crate::types::Accidental::Flat));
    assert_eq!(tied_flat.modifiers.ties, vec![Tie::Start]);
    let sharp = Note::try_from("4c#yy").unwrap();
    assert!(sharp.modifiers.invisible);
    assert!(!sharp.modifiers.invisible_accidental);
    assert!(
        Chord::try_from("12C#<Lyy").unwrap().notes[0]
            .modifiers
            .invisible
    );
    assert!(
        Chord::try_from("[32C:yy").unwrap().notes[0]
            .modifiers
            .invisible
    );
}

#[test]
fn invisible_note_marker_must_be_exactly_double_y() {
    for raw in ["4by", "4byyy", "4byyyy"] {
        assert!(matches!(
            Note::try_from(raw).unwrap_err(),
            ParseError::InvalidNote { .. }
        ));
    }
}

#[test]
fn parses_note_editorial_interpretation() {
    assert_eq!(
        Note::try_from("4cx").unwrap().modifiers,
        crate::types::NoteModifiers {
            ties: Vec::new(),
            forced_accidental: None,
            invisible_accidental: false,
            editorial_interpretation: Some(crate::types::EditorialInterpretation),
            invisible: false,
            cross_staff: None,
            user_signifiers: Vec::new(),
        }
    );
}

#[test]
fn parses_invisible_accidental_marker() {
    assert!(
        Note::try_from("4c#y")
            .unwrap()
            .modifiers
            .invisible_accidental
    );
    assert_eq!(
        Note::try_from("2B##y]").unwrap().accidental,
        Some(crate::types::Accidental::DoubleSharp)
    );
    assert!(
        Note::try_from("2B##y]")
            .unwrap()
            .modifiers
            .invisible_accidental
    );
    assert!(
        Note::try_from("4g--y_")
            .unwrap()
            .modifiers
            .invisible_accidental
    );
    assert!(
        Note::try_from("4dny")
            .unwrap()
            .modifiers
            .invisible_accidental
    );
    assert!(
        GraceNote::try_from("8e#yq")
            .unwrap()
            .modifiers
            .invisible_accidental
    );

    let chord = Chord::try_from("32c#y 32f] 32cc").unwrap();
    assert!(chord.notes.iter().any(|note| {
        note.pitch.step == DiatonicStep::C
            && note.pitch.octave == 4
            && note.accidental == Some(crate::types::Accidental::Sharp)
            && note.modifiers.invisible_accidental
    }));
}

#[test]
fn parses_forced_accidental_marker() {
    assert_eq!(
        Note::try_from("4c#X").unwrap().modifiers.forced_accidental,
        Some(crate::types::ForcedAccidental)
    );
    assert_eq!(
        Note::try_from("2B##X]")
            .unwrap()
            .modifiers
            .forced_accidental,
        Some(crate::types::ForcedAccidental)
    );
    assert_eq!(
        Note::try_from("4g--X_")
            .unwrap()
            .modifiers
            .forced_accidental,
        Some(crate::types::ForcedAccidental)
    );
}

#[test]
fn forced_accidental_marker_must_immediately_follow_accidental() {
    for raw in ["4cX", "4c#]X", "4c#XX"] {
        let error = Note::try_from(raw).unwrap_err();
        assert!(matches!(
            error,
            ParseError::InvalidNote {
                raw: ref error_raw,
                location: InvalidNoteLocation {
                    position: Some(_),
                    character: Some('X'),
                },
            } if error_raw == raw
        ));
    }
}

#[test]
fn invisible_accidental_marker_must_immediately_follow_accidental() {
    for raw in ["4cy", "4c#]y", "4c#y]y", "4c#Xy", "4c#yX"] {
        let error = Note::try_from(raw).unwrap_err();
        assert!(matches!(
            error,
            ParseError::InvalidNote {
                raw: ref error_raw,
                location: InvalidNoteLocation {
                    position: Some(_),
                    character: Some('y') | Some('X'),
                },
            } if error_raw == raw
        ));
    }
}

#[test]
fn note_parser_rejects_unmodeled_chord_modifiers() {
    assert!(matches!(
        Note::try_from("4c;").unwrap_err(),
        ParseError::InvalidNote {
            raw,
            location: InvalidNoteLocation {
                position: Some(3),
                character: Some(';'),
            },
        } if raw == "4c;"
    ));

    assert!(matches!(
        Note::try_from("4cLJ").unwrap_err(),
        ParseError::InvalidNote {
            raw,
            location: InvalidNoteLocation {
                position: Some(3),
                character: Some('L'),
            },
        } if raw == "4cLJ"
    ));
}

#[test]
fn kern_line_parses_note_data_tokens() {
    let line = KernLine::parse(3, ".\t4c\t4r").unwrap();

    assert_eq!(
        line,
        KernLine::Data(vec![
            SpineContent::Kern(KernContent::Null),
            SpineContent::Kern(KernContent::Chord(Chord::try_from("4c").unwrap())),
            SpineContent::Kern(KernContent::Rest(Rest::try_from("4r").unwrap())),
        ])
    );
}

#[test]
fn kern_line_parses_tuplet_note_data_tokens() {
    let line = KernLine::parse(3, "24FFLL\t24fLL").unwrap();

    assert_eq!(
        line,
        KernLine::Data(vec![
            SpineContent::Kern(KernContent::Chord(Chord::try_from("24FFLL").unwrap())),
            SpineContent::Kern(KernContent::Chord(Chord::try_from("24fLL").unwrap())),
        ])
    );
}
