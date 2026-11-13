use std::ops::Range;

use super::{
    content, refusal,
    scan::{scan, ChordFault, Fault, Reason, Scanned, Warning},
    table::{is_grace_byte, At, Dir, Dup, EventKind, On, Place, MARKERS},
    warnings, AtomSink, Marker, MarkerId, Part,
};
use crate::{
    parse::ParseDialect,
    types::{
        Articulation, AuxInterval, Bowing, Breath, Chord, Fermata, FermataPlacement, GraceChord,
        Harmonic, MarkPlacement, Ornament, Placed, Rest, Slur, SlurPlacement, TrillLine,
    },
    InvalidNoteLocation, ParseError,
};

#[derive(Default)]
struct Recorder(Vec<(Range<usize>, MarkerId)>);

impl AtomSink for Recorder {
    fn atom(&mut self, range: Range<usize>, id: MarkerId, _: &'static Marker) {
        self.0.push((range, id));
    }
}

fn ids() -> impl Iterator<Item = MarkerId> {
    (0..MARKERS.len()).map(|index| MarkerId(index as u8))
}

fn allowed(id: MarkerId, event: EventKind) -> bool {
    matches!(id.row().on(event), On::Yes(_))
}

fn context(id: MarkerId) -> (EventKind, &'static str) {
    use super::table::After;
    let event = EventKind::ALL
        .into_iter()
        .find(|event| allowed(id, *event))
        .expect("every row is allowed on some event");
    let prefix = match id.row().after {
        After::Any => "(",
        After::Core | After::Pitch => "4c",
        After::Accidental => "4c#",
        After::Signifier => "4c<",
        After::Rest => "4r",
    };
    (event, prefix)
}

#[test]
fn every_spelling_lexes_to_its_own_row_as_one_atom() {
    for id in ids() {
        let (event, prefix) = context(id);
        for spelling in id.row().spellings() {
            let raw = format!("{prefix}{spelling}");
            let mut atoms = Recorder::default();
            let _ = scan(&raw, event, Part::First, &mut atoms);
            let atom = (prefix.len()..raw.len(), id);
            assert!(
                atoms.0.last() == Some(&atom),
                "{raw:?} on {event:?}: expected one {} atom, got {:?}",
                id.row().name,
                atoms
                    .0
                    .iter()
                    .map(|(range, id)| (range.clone(), id.row().name))
                    .collect::<Vec<_>>()
            );
        }
    }
}

fn scanned(raw: &str, event: EventKind) -> Result<Scanned, Reason> {
    scan(raw, event, Part::First, &mut ()).map_err(|fault| fault.reason)
}

#[test]
fn every_field_row_places_its_own_mark() {
    for id in ids() {
        for (event, base) in [(EventKind::Note, "4c"), (EventKind::Grace, "8qc")] {
            let On::Yes(rule) = id.row().on(event) else {
                continue;
            };
            if rule.place != Place::Field {
                continue;
            }
            for spelling in id.row().spellings() {
                let bare = scanned(&format!("{base}{spelling}"), event);
                for signifier in ["<", ">"] {
                    let placed = scanned(&format!("{base}{spelling}{signifier}"), event);
                    assert!(
                        placed.is_ok() && placed != bare,
                        "{base}{spelling}{signifier} on {event:?} does not place {}",
                        id.row().name
                    );
                }
            }
        }
    }
}

#[test]
fn note_and_grace_rows_stand_anywhere() {
    for id in ids() {
        for event in [EventKind::Note, EventKind::Grace] {
            if let On::Yes(rule) = id.row().on(event) {
                assert_eq!(rule.at, At::ANY, "{} on {event:?}", id.row().name);
            }
        }
    }
}

/// A repeat of a slotted mark is refused or warned, never replaced silently.
#[test]
fn a_row_has_a_slot_exactly_when_a_repeat_is_refused_or_warned() {
    for id in ids() {
        for event in EventKind::ALL {
            if let On::Yes(rule) = id.row().on(event) {
                assert_eq!(
                    id.row().slot.is_some(),
                    rule.dup != Dup::LastWins,
                    "{} on {event:?}",
                    id.row().name
                );
            }
        }
    }
}

fn located(raw: &str, position: usize, character: char) -> ParseError {
    ParseError::InvalidNote {
        raw: raw.to_owned(),
        location: InvalidNoteLocation {
            position: Some(position),
            character: Some(character),
        },
    }
}

#[test]
fn faults_keep_todays_positions() {
    for (raw, position, character) in [
        ("4###", 2, '#'),
        ("4cS'Ss", 5, 'S'),
        ("8cq(>", 5, '>'),
        ("4ci(>", 5, '>'),
        ("4cTTTT", 6, 'T'),
        ("4cyyy", 5, 'y'),
        ("4c&x", 4, 'x'),
        ("4c→(", 3, '→'),
    ] {
        let result = if raw.contains('q') {
            GraceChord::try_from(raw).map(|_| ())
        } else {
            Chord::try_from(raw).map(|_| ())
        };
        assert_eq!(result, Err(located(raw, position, character)), "{raw}");
    }
    assert_eq!(
        Chord::try_from("4c###"),
        Err(ParseError::InvalidNote {
            raw: "4c###".into(),
            location: InvalidNoteLocation {
                position: None,
                character: None,
            },
        })
    );
}

#[test]
fn suppression_follows_the_grace_mark_position() {
    assert!(GraceChord::try_from("8qc(>").is_ok());
    assert!(GraceChord::try_from("4iq(>c").is_ok());
    assert_eq!(GraceChord::try_from("8cq(>"), Err(located("8cq(>", 5, '>')));
}

#[test]
fn placement_after_a_merged_slur_count_splits_it() {
    let chord = Chord::try_from("((>4c").expect("parses");
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
                placement: Some(SlurPlacement::Above),
                linked: false,
            },
        ]
    );
}

fn placed<T>(mark: T, placement: MarkPlacement) -> Placed<T> {
    Placed {
        mark,
        placement: Some(placement),
    }
}

#[test]
fn a_signifier_after_an_articulation_ornament_or_bowing_places_that_mark() {
    let token = |raw| Chord::try_from(raw).expect("parses").modifiers;
    assert_eq!(
        token("4c'<").articulations,
        vec![placed(Articulation::Staccato, MarkPlacement::Below)]
    );
    assert_eq!(
        token("4cz>").articulations,
        vec![placed(Articulation::Generic, MarkPlacement::Above)]
    );
    assert_eq!(
        token("4cT>").ornaments,
        vec![placed(
            Ornament::Trill {
                interval: AuxInterval::WholeTone,
                line: TrillLine::None,
            },
            MarkPlacement::Above
        )]
    );
    assert_eq!(
        token("4cu<").bowing,
        Some(placed(Bowing::DownBow, MarkPlacement::Below))
    );
    assert_eq!(
        GraceChord::try_from("8qc^^>")
            .expect("parses")
            .modifiers
            .articulations,
        vec![
            Placed::from(Articulation::Accent),
            placed(Articulation::Accent, MarkPlacement::Above),
        ]
    );
}

#[test]
fn a_signifier_after_a_breath_or_harmonic_places_that_mark() {
    let token = |raw| Chord::try_from(raw).expect("parses").modifiers;
    assert_eq!(
        token("4c,<").breath,
        Some(placed(Breath, MarkPlacement::Below))
    );
    assert_eq!(
        token("4co>").harmonic,
        Some(placed(Harmonic, MarkPlacement::Above))
    );
    let grace = GraceChord::try_from("8qc,>o<").expect("parses").modifiers;
    assert_eq!(grace.breath, Some(placed(Breath, MarkPlacement::Above)));
    assert_eq!(grace.harmonic, Some(placed(Harmonic, MarkPlacement::Below)));
    assert_eq!(Rest::try_from("4r,").expect("parses").breath, Some(Breath));
    assert_eq!(
        Rest::try_from("4r,<"),
        Err(ParseError::InvalidRest { raw: "4r,<".into() })
    );
}

#[test]
fn chord_merge_concatenates_placed_marks_and_keeps_the_first_bowing() {
    let modifiers = Chord::try_from("4c'<u> 4e~>v<").expect("parses").modifiers;
    assert_eq!(
        modifiers.articulations,
        vec![
            placed(Articulation::Staccato, MarkPlacement::Below),
            placed(Articulation::Tenuto, MarkPlacement::Above),
        ]
    );
    assert_eq!(
        modifiers.bowing,
        Some(placed(Bowing::DownBow, MarkPlacement::Above))
    );
}

#[test]
fn chord_merge_keeps_one_copy_of_an_ornament_repeated_on_another_head() {
    let ornaments = |raw| Chord::try_from(raw).expect("parses").modifiers.ornaments;
    let trill = |interval| Ornament::Trill {
        interval,
        line: TrillLine::None,
    };
    assert_eq!(
        ornaments("4cT 4eT 4gT"),
        vec![Placed::from(trill(AuxInterval::WholeTone))]
    );
    assert_eq!(
        ornaments("4cT 4et"),
        vec![
            Placed::from(trill(AuxInterval::WholeTone)),
            Placed::from(trill(AuxInterval::Semitone)),
        ]
    );
    assert_eq!(
        ornaments("4cT< 4eT"),
        vec![
            placed(trill(AuxInterval::WholeTone), MarkPlacement::Below),
            Placed::from(trill(AuxInterval::WholeTone)),
        ]
    );
    assert_eq!(
        ornaments("4cRR 4eR"),
        vec![
            Placed::from(Ornament::ConcludingTurn),
            Placed::from(Ornament::ConcludingTurn),
        ]
    );
}

#[test]
fn a_slur_signifier_places_the_entry_its_open_folded_into() {
    let open = |count, placement| Slur::Open {
        elided: false,
        depth: 0,
        count,
        placement,
        linked: false,
    };
    assert_eq!(
        Chord::try_from("4c()(>").expect("parses").modifiers.slurs,
        vec![
            open(1, None),
            Slur::Close {
                elided: false,
                depth: 0,
                count: 1,
                linked: false,
            },
            open(1, Some(SlurPlacement::Above)),
        ]
    );
}

#[test]
fn chord_merge_overflow_is_an_invalid_chord() {
    let raw = format!("{}4c {}4e", "(".repeat(200), "(".repeat(200));
    assert_eq!(
        Chord::try_from(raw.as_str()),
        Err(ParseError::InvalidChord { raw: raw.clone() })
    );
    assert_eq!(
        Chord::try_from("4cLK 4ek"),
        Err(ParseError::InvalidChord {
            raw: "4cLK 4ek".into()
        })
    );
}

#[test]
fn content_routes_whole_tokens_before_scanning() {
    use crate::{
        parse::ParseDialect::{HumdrumKern, TranscodaV2},
        types::{Fermata, InvisibleRestSlice, KernContent},
    };
    let invalid_rest = |raw: &str| Err(ParseError::InvalidRest { raw: raw.into() });
    assert_eq!(content(".", HumdrumKern), Ok(KernContent::Null));
    assert_eq!(
        content("ryy;", TranscodaV2),
        Ok(KernContent::InvisibleRestSlice(InvisibleRestSlice {
            fermata: Some(Fermata::default())
        }))
    );
    assert_eq!(content("ryy;", HumdrumKern), invalid_rest("ryy;"));
    assert_eq!(content("ryy(", TranscodaV2), invalid_rest("ryy("));
    assert_eq!(content("MM2r;", HumdrumKern), invalid_rest("MM2r;"));
    assert_eq!(content("MM1r", HumdrumKern), invalid_rest("MM1r"));
    assert_eq!(
        content("MMq", HumdrumKern),
        Err(ParseError::InvalidNote {
            raw: "MMq".into(),
            location: InvalidNoteLocation {
                position: Some(1),
                character: Some('M'),
            },
        })
    );
}

#[test]
fn grace_bytes_are_the_grace_lexeme_bytes() {
    let grace: Vec<u8> = (0..=u8::MAX).filter(|byte| is_grace_byte(*byte)).collect();
    assert_eq!(grace, b"PQpq");
}

#[test]
fn rest_regions_only_move_forward() {
    let suffix_or_core = At::CORE.or(At::SUFFIX);
    assert_eq!(At::SLURS.earliest_allowed(suffix_or_core), Some(At::CORE));
    assert_eq!(
        At::SUFFIX.earliest_allowed(suffix_or_core),
        Some(At::SUFFIX)
    );
    assert_eq!(At::SUFFIX.earliest_allowed(At::SLURS), None);
    assert_eq!(At::DURATION.earliest_allowed(At::ANY), Some(At::DURATION));
}

#[test]
fn at_most_two_rows_share_a_lexeme() {
    for id in ids() {
        let sharing = ids()
            .filter(|other| other.row().lexeme == id.row().lexeme)
            .count();
        assert!(
            sharing <= 2,
            "{} shares its lexeme with {} rows",
            id.row().name,
            sharing - 1
        );
    }
}

#[test]
fn lit_rows_take_their_spelling_from_the_lexeme() {
    for id in ids() {
        let row = id.row();
        if matches!(row.lexeme, super::table::Lexeme::Lit(_)) {
            assert!(
                row.spell.is_empty(),
                "{} sets spell on a Lit lexeme",
                row.name
            );
        }
    }
}

fn slur(count: u8, placement: Option<SlurPlacement>) -> Slur {
    Slur::Open {
        elided: false,
        depth: 0,
        count,
        placement,
        linked: false,
    }
}

#[test]
fn rest_slurs_fold_like_note_slurs() {
    let slurs = |raw| Rest::try_from(raw).expect("parses").slurs;
    assert_eq!(slurs("12r(>"), vec![slur(1, Some(SlurPlacement::Above))]);
    assert_eq!(slurs("12r(<"), vec![slur(1, Some(SlurPlacement::Below))]);
    assert_eq!(
        slurs("(>(>4r"),
        vec![
            slur(1, Some(SlurPlacement::Above)),
            slur(1, Some(SlurPlacement::Above))
        ]
    );
    assert_eq!(slurs("((4r("), vec![slur(3, None)]);
    assert_eq!(
        slurs("4r)("),
        vec![
            Slur::Close {
                elided: false,
                depth: 0,
                count: 1,
                linked: false,
            },
            slur(1, None),
        ]
    );
}

fn marker(name: &str) -> MarkerId {
    ids()
        .find(|id| id.row().name == name)
        .unwrap_or_else(|| panic!("no row {name}"))
}

#[test]
fn a_repeated_fermata_or_breath_keeps_the_first_and_warns() {
    let token = |raw| Chord::try_from(raw).expect("parses").modifiers;
    assert_eq!(
        token("4c;>;").fermata,
        Some(Fermata {
            placement: Some(FermataPlacement::Above)
        })
    );
    assert_eq!(
        warnings("4c;>;", ParseDialect::HumdrumKern),
        [(
            4..5,
            Warning::Duplicate {
                mark: marker("fermata")
            }
        )]
    );
    assert_eq!(
        token("4c,<,").breath,
        Some(placed(Breath, MarkPlacement::Below))
    );
    assert_eq!(
        warnings("8qc,<,", ParseDialect::HumdrumKern),
        [(
            5..6,
            Warning::Duplicate {
                mark: marker("breath")
            }
        )]
    );
}

#[test]
fn a_signifier_after_an_ignored_duplicate_is_ignored_with_it() {
    assert_eq!(
        Chord::try_from("4c;<;>").expect("parses").modifiers.fermata,
        Some(Fermata {
            placement: Some(FermataPlacement::Below)
        })
    );
    assert_eq!(
        warnings("4c;<;>", ParseDialect::HumdrumKern),
        [
            (
                4..5,
                Warning::Duplicate {
                    mark: marker("fermata")
                }
            ),
            (
                5..6,
                Warning::Unattached {
                    signifier: Dir::Above,
                    mark: marker("fermata"),
                    duplicate: true,
                }
            ),
        ]
    );
    assert_eq!(Chord::try_from("4c;;>>"), Err(located("4c;;>>", 6, '>')));
}

#[test]
fn warning_spans_count_bytes_in_the_whole_token() {
    assert_eq!(
        warnings("4c 4e)> 4g:<", ParseDialect::HumdrumKern),
        [
            (
                6..7,
                Warning::Unattached {
                    signifier: Dir::Above,
                    mark: marker("slur-close"),
                    duplicate: false,
                }
            ),
            (
                11..12,
                Warning::Unattached {
                    signifier: Dir::Below,
                    mark: marker("arpeggio"),
                    duplicate: false,
                }
            ),
        ]
    );
    assert_eq!(warnings("4r;", ParseDialect::HumdrumKern), []);
}

#[test]
fn warnings_name_the_signifier_and_the_mark() {
    let messages = |raw| -> Vec<String> {
        warnings(raw, ParseDialect::HumdrumKern)
            .into_iter()
            .map(|(_, warning)| format!("{}: {}", warning.code(), warning.message()))
            .collect()
    };
    assert_eq!(
        messages("4c)<"),
        ["token.unattached-signifier: `<` follows slur-close, which takes no placement; the parser ignores it"]
    );
    assert_eq!(
        messages("4c;;>"),
        [
            "token.duplicate-mark: second fermata on one head; the parser keeps the first and ignores this one",
            "token.unattached-signifier: `>` follows the ignored second fermata; the parser ignores it",
        ]
    );
}

#[test]
fn a_chord_holding_a_rest_is_refused_at_the_rest() {
    let refused = refusal("4c 16r# c##L", ParseDialect::HumdrumKern).expect("refused");
    assert_eq!(
        refused.error,
        ParseError::InvalidChord {
            raw: "4c 16r# c##L".into()
        }
    );
    assert_eq!(
        refused.fault,
        Fault {
            span: 3..7,
            reason: Reason::Chord(ChordFault::RestHead)
        }
    );
    assert_eq!(refused.fault.reason.code(), "token.rest-in-chord");
    assert_eq!(
        content("16r# c##L", ParseDialect::HumdrumKern),
        Err(ParseError::InvalidChord {
            raw: "16r# c##L".into()
        })
    );
}

#[test]
fn only_mm_digits_r_is_a_multi_measure_rest() {
    use crate::types::{KernContent, MordentDirection, MultiMeasureRest};
    assert_eq!(
        content("MM10r", ParseDialect::HumdrumKern),
        Ok(KernContent::MultiMeasureRest(MultiMeasureRest {
            measure_count: 10
        }))
    );
    let Ok(KernContent::Chord(chord)) = content("MM4c", ParseDialect::HumdrumKern) else {
        panic!("MM4c is a note")
    };
    assert_eq!(
        chord.modifiers.ornaments,
        vec![Placed::from(Ornament::Mordent {
            direction: MordentDirection::Upper,
            interval: Some(AuxInterval::WholeTone),
            doubled: true,
        })]
    );
    assert_eq!(
        content("MM2", ParseDialect::HumdrumKern),
        Err(located("MM2", 1, 'M'))
    );
    assert_eq!(
        refusal("MM2", ParseDialect::HumdrumKern).map(|refused| refused.fault),
        Some(Fault {
            span: 0..0,
            reason: Reason::Missing(super::scan::Missing::Pitch)
        })
    );
    assert_eq!(
        refusal("MM0r", ParseDialect::HumdrumKern).map(|refused| refused.fault.reason),
        Some(Reason::MeasureCount)
    );
}

#[test]
fn a_refusal_keeps_the_scanner_reason_and_span_in_the_token() {
    let fault = |raw| refusal(raw, ParseDialect::HumdrumKern).map(|refused| refused.fault);
    assert_eq!(
        fault("4c 4eTt"),
        Some(Fault {
            span: 6..7,
            reason: Reason::Duplicate(marker("trill-semitone"))
        })
    );
    assert_eq!(
        fault("4r)>"),
        Some(Fault {
            span: 3..4,
            reason: Reason::Unattached(marker("place-above"))
        })
    );
    assert_eq!(
        fault("4c 8e"),
        Some(Fault {
            span: 3..5,
            reason: Reason::Chord(ChordFault::Duration)
        })
    );
}
