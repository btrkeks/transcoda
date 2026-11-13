use crate::types::{ChordBeams, DiatonicStep, DurationValue, GraceKind, KernContent, Slur};
use crate::*;

#[test]
fn parses_single_grace_note_tokens() {
    let grace = GraceChord::try_from("16qqcc#LL").unwrap();
    assert_eq!(grace.notes.len(), 1);
    assert_eq!(grace.notes[0].kind, GraceKind::AccentedGrace);
    assert_eq!(
        grace.notes[0].duration,
        Some(Duration {
            value: DurationValue::Reciprocal(16),
            augmentation_dots: 0,
        })
    );
    assert_eq!(grace.notes[0].pitch.step, DiatonicStep::C);
    assert_eq!(grace.notes[0].pitch.octave, 5);
    assert_eq!(
        grace.notes[0].accidental,
        Some(crate::types::Accidental::Sharp)
    );
    assert_eq!(grace.modifiers.beams, ChordBeams::Start { level: 2 });

    let grace = GraceChord::try_from("8e#q(").unwrap();
    assert_eq!(grace.notes[0].kind, GraceKind::Acciaccatura);
    assert_eq!(grace.notes[0].pitch.step, DiatonicStep::E);
    assert_eq!(
        grace.notes[0].accidental,
        Some(crate::types::Accidental::Sharp)
    );
    assert_eq!(
        grace.modifiers.slurs,
        vec![Slur::Open {
            elided: false,
            depth: 0,
            count: 1,
            placement: None,
            linked: false,
        }]
    );

    assert_eq!(
        GraceNote::try_from("4Qe").unwrap().kind,
        GraceKind::Groupetto
    );
    assert_eq!(
        GraceNote::try_from("4Pe").unwrap().kind,
        GraceKind::Appoggiatura
    );
    assert_eq!(
        GraceNote::try_from("4ep").unwrap().kind,
        GraceKind::PostAppoggiatura
    );

    let grace = GraceNote::try_from("ccq").unwrap();
    assert_eq!(grace.kind, GraceKind::Acciaccatura);
    assert_eq!(grace.duration, None);
    assert_eq!(grace.pitch.step, DiatonicStep::C);
    assert_eq!(grace.pitch.octave, 5);
}

#[test]
fn parses_grace_chord_in_source_order_and_exposes_pitch_order() {
    let grace = GraceChord::try_from("8ccc#q 8eeq").unwrap();

    assert_eq!(grace.notes.len(), 2);
    assert_eq!(
        grace
            .notes
            .iter()
            .map(|note| note.pitch.step)
            .collect::<Vec<_>>(),
        vec![DiatonicStep::C, DiatonicStep::E]
    );
    assert_eq!(
        grace
            .notes_by_pitch()
            .into_iter()
            .map(|note| note.pitch.step)
            .collect::<Vec<_>>(),
        vec![DiatonicStep::E, DiatonicStep::C]
    );
    assert_eq!(grace.lowest_note().unwrap().pitch.step, DiatonicStep::E);
    assert_eq!(grace.highest_note().unwrap().pitch.step, DiatonicStep::C);
    assert!(grace.notes.iter().all(|note| {
        note.kind == GraceKind::Acciaccatura
            && note.duration
                == Some(Duration {
                    value: DurationValue::Reciprocal(8),
                    augmentation_dots: 0,
                })
    }));
    assert_eq!(
        grace.notes[0].accidental,
        Some(crate::types::Accidental::Sharp)
    );
}

#[test]
fn grace_chord_notes_inherit_duration_and_kind_from_first_note() {
    let grace = GraceChord::try_from("16qqCC# C#LL").unwrap();

    assert_eq!(grace, GraceChord::try_from("16qqCC# 16qqC#LL").unwrap());
    assert_eq!(grace.notes.len(), 2);
    assert!(grace.notes.iter().all(|note| {
        note.kind == GraceKind::AccentedGrace
            && note.duration
                == Some(Duration {
                    value: DurationValue::Reciprocal(16),
                    augmentation_dots: 0,
                })
    }));
    assert_eq!(grace.modifiers.beams, ChordBeams::Start { level: 2 });

    let mixed = GraceChord::try_from("8qcc Qee").unwrap();
    assert_eq!(mixed.notes[0].kind, GraceKind::Acciaccatura);
    assert_eq!(mixed.notes[1].kind, GraceKind::Groupetto);
    assert_eq!(mixed.notes[0].duration, mixed.notes[1].duration);
}

#[test]
fn grace_notes_parse_cross_staff_only_after_pitch_and_accidental() {
    assert_eq!(
        GraceNote::try_from("8qe#>").unwrap().modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Above)
    );
    assert_eq!(
        GraceChord::try_from("8qcc dd#<")
            .unwrap()
            .notes
            .iter()
            .map(|note| note.modifiers.cross_staff)
            .collect::<Vec<_>>(),
        vec![None, Some(crate::types::CrossStaffPlacement::Below)]
    );
    assert_eq!(
        GraceNote::try_from("8qe#X>").unwrap().modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Above)
    );
    let hidden = GraceNote::try_from("8qe#y>").unwrap();
    assert!(hidden.modifiers.invisible_accidental);
    assert_eq!(
        hidden.modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Above)
    );

    let dead_forced = GraceNote::try_from("8qe#>X").unwrap();
    assert_eq!(dead_forced.modifiers.forced_accidental, None);
    assert_eq!(
        dead_forced.modifiers.cross_staff,
        Some(crate::types::CrossStaffPlacement::Above)
    );

    let tied = GraceNote::try_from("8qe[>").unwrap();
    assert_eq!(
        tied.modifiers.ties,
        vec![crate::types::Tie::PlacedOpen {
            phase: crate::types::TieOpenPhase::Start,
            resolution: crate::types::TieResolution::Ordinary,
            placement: crate::types::TiePlacement::Above,
        }]
    );
    assert_eq!(tied.modifiers.cross_staff, None);

    for raw in ["8e#q>", "8e#Xq>"] {
        assert!(matches!(
            GraceNote::try_from(raw).unwrap_err(),
            ParseError::InvalidNote { raw: error_raw, .. } if error_raw == raw
        ));
    }
}

#[test]
fn kern_line_parses_grace_data_tokens() {
    let line = KernLine::parse(3, ".\t16qqcc#LL\t8eeq 8ccc#q\t4r").unwrap();

    assert_eq!(
        line,
        KernLine::Data(vec![
            SpineContent::Kern(KernContent::Null),
            SpineContent::Kern(KernContent::Grace(
                GraceChord::try_from("16qqcc#LL").unwrap()
            )),
            SpineContent::Kern(KernContent::Grace(
                GraceChord::try_from("8eeq 8ccc#q").unwrap()
            )),
            SpineContent::Kern(KernContent::Rest(Rest::try_from("4r").unwrap())),
        ])
    );
}

#[test]
fn grace_parser_rejects_invalid_shapes() {
    {
        let raw = "8qqqcc";
        assert!(matches!(
            GraceChord::try_from(raw).unwrap_err(),
            ParseError::InvalidNote { raw: error_raw, .. } if error_raw == raw
        ));
    }

    assert_eq!(
        GraceChord::try_from("4c 8qqd").unwrap_err(),
        ParseError::InvalidChord {
            raw: "4c 8qqd".to_owned(),
        }
    );
    assert_eq!(
        Chord::try_from("4c 8qqd").unwrap_err(),
        ParseError::InvalidChord {
            raw: "4c 8qqd".to_owned(),
        }
    );
    assert_eq!(
        GraceChord::try_from("8qcc 16qdd").unwrap_err(),
        ParseError::InvalidChord {
            raw: "8qcc 16qdd".to_owned(),
        }
    );
}
