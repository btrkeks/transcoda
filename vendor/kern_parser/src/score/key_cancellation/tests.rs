use crate::{
    document::Framing,
    score::{CancellationIssue, NotDrawn, Outcome, Score, StateError, StateKey},
    types::DiatonicStep,
    KernDocument, KeySignature,
};

fn errors(input: &str) -> Vec<StateError> {
    let document = KernDocument::parse(input).unwrap();
    match Score::resolve(&document, Framing::Complete) {
        Ok(_) => Vec::new(),
        Err(errors) => errors.0,
    }
}

fn key(raw: &str) -> KeySignature {
    KeySignature::try_from(raw).unwrap()
}

fn issue(line: usize, column: usize, issue: CancellationIssue) -> StateError {
    StateError::KeyCancellation {
        line,
        column,
        issue,
    }
}

#[test]
fn naturals_must_cancel_an_accidental_of_the_previous_key() {
    assert_eq!(
        errors("**kern\n*k[f#c#]\n4c\n=\n*k[cnf#]\n4c\n*-\n"),
        vec![]
    );
    assert_eq!(
        errors("**kern\n*k[f#]\n4c\n=\n*k[bn]\n4c\n*-\n"),
        vec![StateError::StrayNatural {
            line: 5,
            column: 1,
            step: DiatonicStep::B,
            previous: Some(key("*k[f#]")),
        }]
    );
    assert_eq!(
        errors("**kern\n*k[fn]\n4c\n*-\n"),
        vec![StateError::StrayNatural {
            line: 2,
            column: 1,
            step: DiatonicStep::F,
            previous: None,
        }]
    );
}

#[test]
fn track_state_holds_the_key_without_its_naturals() {
    let document =
        KernDocument::parse("**kern\n*k[f#c#]\n4c\n=\n*k[cnf#]\n4c\n=\n*k[f#]\n4c\n*-\n").unwrap();
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let last_data = document.lines().len() - 2;
    assert_eq!(
        score.state_at(last_data, 0).track.key_signature,
        Some(key("*k[f#]"))
    );
    let restatement = score
        .events()
        .iter()
        .rfind(|event| event.key == StateKey::KeySignature)
        .unwrap();
    assert_eq!(restatement.outcome, Outcome::Restated);
}

#[test]
fn verovio_drops_naturals_in_the_middle_of_a_measure() {
    assert_eq!(
        errors("**kern\n*k[f#]\n4c\n*k[fn]\n4c\n*-\n"),
        vec![issue(4, 1, CancellationIssue::NaturalsMidMeasure)]
    );
}

#[test]
fn kcancel_draws_naturals_from_the_next_barline() {
    assert_eq!(
        errors("**kern\n*kcancel\n*k[f#c#]\n4c\n=\n*k[f#]\n4c\n*-\n"),
        vec![]
    );
    // Verovio builds the key change before it reads this measure's
    // `*kcancel`.
    assert_eq!(
        errors("**kern\n*k[f#c#]\n4c\n=\n*kcancel\n*k[f#]\n4c\n*-\n"),
        vec![issue(
            6,
            1,
            CancellationIssue::NotDrawn {
                reason: NotDrawn::FlagOff,
                spelled: key("*k[cnf#]"),
            }
        )]
    );
    // A change to C draws naturals without `*kcancel`.
    assert_eq!(errors("**kern\n*k[f#c#]\n4c\n=\n*k[]\n4c\n*-\n"), vec![]);
    // A change that drops nothing draws none either way.
    assert_eq!(
        errors("**kern\n*k[f#]\n4c\n=\n*kcancel\n*k[f#c#]\n4c\n*-\n"),
        vec![]
    );
}

#[test]
fn one_kcancel_applies_to_every_staff() {
    assert_eq!(
        errors("**kern\t**kern\n*kcancel\t*\n*k[f#c#]\t*k[f#c#]\n4c\t4c\n=\t=\n*k[f#]\t*k[f#]\n4c\t4c\n*-\t*-\n"),
        vec![]
    );
}

#[test]
fn xkcancel_takes_effect_from_the_next_barline() {
    assert_eq!(
        errors("**kern\n*kcancel\n*k[f#c#]\n4c\n=\n*Xkcancel\n*k[f#]\n4c\n*-\n"),
        vec![issue(7, 1, CancellationIssue::Unrequested)]
    );
    assert_eq!(
        errors("**kern\n*kcancel\n*k[f#c#]\n4c\n*Xkcancel\n=\n*k[f#]\n4c\n*-\n"),
        vec![]
    );
}

#[test]
fn automatic_naturals_need_every_staff_to_change_to_one_key() {
    assert_eq!(
        errors("**kern\t**kern\n*kcancel\t*kcancel\n*k[f#c#]\t*k[f#c#]\n4c\t4c\n=\t=\n*k[f#]\t*\n4c\t4c\n*-\t*-\n"),
        vec![issue(
            6,
            1,
            CancellationIssue::NotDrawn {
                reason: NotDrawn::PerStaff,
                spelled: key("*k[cnf#]"),
            }
        )]
    );
    // Spelled naturals are drawn on one staff.
    assert_eq!(
        errors("**kern\t**kern\n*kcancel\t*kcancel\n*k[f#c#]\t*k[f#c#]\n4c\t4c\n=\t=\n*k[cnf#]\t*\n4c\t4c\n*-\t*-\n"),
        vec![]
    );
}

#[test]
fn no_automatic_naturals_after_a_key_that_spells_its_own() {
    assert_eq!(
        errors("**kern\n*kcancel\n*k[f#c#g#]\n4c\n=\n*k[gnf#c#]\n4c\n=\n*k[f#]\n4c\n*-\n"),
        vec![issue(
            9,
            1,
            CancellationIssue::NotDrawn {
                reason: NotDrawn::AfterNaturals,
                spelled: key("*k[cnf#]"),
            }
        )]
    );
}

#[test]
fn no_automatic_naturals_in_the_middle_of_a_measure() {
    assert_eq!(
        errors("**kern\n*kcancel\n*k[f#c#]\n4c\n*k[f#]\n4c\n*-\n"),
        vec![issue(
            5,
            1,
            CancellationIssue::NotDrawn {
                reason: NotDrawn::MidMeasure,
                spelled: key("*k[cnf#]"),
            }
        )]
    );
}

#[test]
fn spines_must_agree_on_the_flag_at_a_barline() {
    assert_eq!(
        errors("**kern\t**kern\n*k[f#]\t*k[f#]\n4c\t4c\n*kcancel\t*Xkcancel\n4c\t4c\n=\t=\n4c\t4c\n*-\t*-\n"),
        vec![issue(4, 2, CancellationIssue::ConflictingFlags)]
    );
}

#[test]
fn restating_kcancel_is_redundant_unless_it_moves_verovios_flag() {
    let document = KernDocument::parse(
        "**kern\t**kern\n*kcancel\t*kcancel\n4c\t4c\n=\t=\n*\t*Xkcancel\n4c\t4c\n=\t=\n*kcancel\t*\n4c\t4c\n=\t=\n*kcancel\t*\n4c\t4c\n*-\t*-\n",
    )
    .unwrap();
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let outcomes = score
        .events()
        .iter()
        .filter(|event| event.key == StateKey::KeyCancellation)
        .map(|event| (document.source_line_number(event.line_index), event.outcome))
        .collect::<Vec<_>>();
    assert_eq!(
        outcomes,
        vec![
            (2, Outcome::Changed),
            (2, Outcome::Changed),
            (5, Outcome::Changed),
            (8, Outcome::Changed),
            (11, Outcome::Redundant),
        ]
    );
}

#[test]
fn kcancel_is_restated_where_state_is_rebuilt() {
    let document = KernDocument::parse("**kern\n*kcancel\n*k[f#]\n4c\n*-\n").unwrap();
    let score = Score::resolve(&document, Framing::Complete).unwrap();
    let last_data = document.lines().len() - 2;
    let lines = crate::score::bridge(
        &score.snapshot_before(last_data).defaulted(),
        &score.snapshot_before(last_data),
    )
    .unwrap();
    let spelled = lines
        .iter()
        .map(|line| {
            line.tokens()
                .iter()
                .map(|tandem| {
                    crate::serialize::omr::spell_tandem(
                        tandem,
                        crate::serialize::omr::VisibilityMarkerStyle::Preserve,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
                .join("\t")
        })
        .collect::<Vec<_>>();
    assert_eq!(spelled, vec!["*clefG2", "*k[f#]", "*kcancel"]);
}
