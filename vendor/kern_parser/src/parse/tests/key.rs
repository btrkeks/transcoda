use crate::types::{Accidental, DiatonicStep};
use crate::*;

fn tandem_line(tokens: Vec<TandemInterpretation>) -> KernLine {
    KernLine::TandemInterpretationLine(TandemInterpretationLine::new(tokens).unwrap())
}

#[test]
fn parses_key_signature_fifths() {
    let cases = [
        ("*k[b-e-a-d-g-c-f-]", -7),
        ("*k[b-e-a-d-g-c-]", -6),
        ("*k[b-e-a-d-g-]", -5),
        ("*k[b-e-a-d-]", -4),
        ("*k[b-e-a-]", -3),
        ("*k[b-e-]", -2),
        ("*k[b-]", -1),
        ("*k[]", 0),
        ("*k[f#]", 1),
        ("*k[f#c#]", 2),
        ("*k[f#c#g#]", 3),
        ("*k[f#c#g#d#]", 4),
        ("*k[f#c#g#d#a#]", 5),
        ("*k[f#c#g#d#a#e#]", 6),
        ("*k[f#c#g#d#a#e#b#]", 7),
    ];

    for (raw, fifths) in cases {
        let signature = KeySignature::try_from(raw).unwrap();

        assert_eq!(signature.fifths().get(), fifths);
        assert_eq!(signature.as_kern(), raw);
    }
}

#[test]
fn key_signature_parser_accepts_out_of_order_accidentals() {
    let cases = [
        ("*k[b-a-e-d-]", -4, "*k[b-e-a-d-]"),
        ("*k[c#f#]", 2, "*k[f#c#]"),
        ("*k[b#e#a#d#g#c#f#]", 7, "*k[f#c#g#d#a#e#b#]"),
        ("*k[f-c-g-d-a-e-b-]", -7, "*k[b-e-a-d-g-c-f-]"),
        ("*k[f#c#gn]", 2, "*k[gnf#c#]"),
        ("*k[c#gnf#]", 2, "*k[gnf#c#]"),
        ("*k[b-gna-e-]", -3, "*k[gnb-e-a-]"),
        ("*k[cnfn]", 0, "*k[cnfn]"),
    ];

    for (raw, fifths, canonical) in cases {
        let signature = KeySignature::try_from(raw).unwrap();

        assert_eq!(signature.fifths().get(), fifths, "{raw}");
        assert_eq!(signature.as_kern(), canonical, "{raw}");
        assert_eq!(
            KernLine::parse(3, raw).unwrap(),
            tandem_line(vec![TandemInterpretation::KeySignature(signature)]),
            "{raw}"
        );
    }
}

#[test]
fn key_signature_parser_keeps_cancellation_naturals_in_written_order() {
    let cases = [
        (
            "*k[bnenf#]",
            1,
            "*k[bnenf#]",
            vec![DiatonicStep::B, DiatonicStep::E],
        ),
        (
            "*k[f#bnen]",
            1,
            "*k[bnenf#]",
            vec![DiatonicStep::B, DiatonicStep::E],
        ),
        (
            "*k[enbnf#]",
            1,
            "*k[enbnf#]",
            vec![DiatonicStep::E, DiatonicStep::B],
        ),
        (
            "*k[bnenandngnf#c#]",
            2,
            "*k[bnenandngnf#c#]",
            vec![
                DiatonicStep::B,
                DiatonicStep::E,
                DiatonicStep::A,
                DiatonicStep::D,
                DiatonicStep::G,
            ],
        ),
        ("*k[fn]", 0, "*k[fn]", vec![DiatonicStep::F]),
    ];

    for (raw, fifths, canonical, naturals) in cases {
        let signature = KeySignature::try_from(raw).unwrap();

        assert_eq!(signature.fifths().get(), fifths, "{raw}");
        assert_eq!(signature.as_kern(), canonical, "{raw}");
        assert_eq!(signature.naturals().collect::<Vec<_>>(), naturals, "{raw}");
        assert_eq!(
            signature.without_naturals().as_kern(),
            KeySignature::from_fifths(signature.fifths()).as_kern(),
            "{raw}"
        );
    }
}

#[test]
fn key_signatures_with_different_naturals_differ() {
    let plain = KeySignature::try_from("*k[f#]").unwrap();
    let cancelling = KeySignature::try_from("*k[bnf#]").unwrap();

    assert_ne!(plain, cancelling);
    assert_eq!(cancelling.without_naturals(), plain);
    assert_eq!(
        cancelling.accidental_for_step(DiatonicStep::B),
        None,
        "a cancellation natural is not part of the key"
    );
}

#[test]
fn dropped_accidentals_follow_the_previous_key_order() {
    let key = |raw| KeySignature::try_from(raw).unwrap();

    assert_eq!(
        key("*k[f#]").dropped_from(key("*k[f#c#g#]")),
        vec![DiatonicStep::C, DiatonicStep::G]
    );
    assert_eq!(
        key("*k[b-]").dropped_from(key("*k[f#c#]")),
        vec![DiatonicStep::F, DiatonicStep::C]
    );
    assert_eq!(
        key("*k[]").dropped_from(key("*k[b-e-]")),
        vec![DiatonicStep::B, DiatonicStep::E]
    );
    assert!(key("*k[f#c#g#]").dropped_from(key("*k[f#]")).is_empty());
}

#[test]
fn key_signature_parser_rejects_malformed_or_unsupported_signatures() {
    for raw in [
        "*k[c#]",
        "*k[f#d#]",
        "*k[b-e-a-d-g-c-f-b-]",
        "*k[f#c#g#d#a#e#b#x#]",
        "*k[f#b-]",
        "*k[c#gn]",
        "*k[f#f#]",
        "*k[fnfn]",
        "*k[fnf#]",
        "*k[f#fn]",
        "*k[fn-]",
        "*k[fn-cn-gn-dn-an-b-e-a-]",
        "*k[cancel]",
        "*k[f##]",
        "*k[b--]",
        "*k[f]",
        "*k[F#]",
        "*k[f# c#]",
        "*k[f#",
        "*k[f#]]",
        "*k",
    ] {
        assert_eq!(
            KeySignature::try_from(raw).unwrap_err(),
            ParseError::InvalidKeySignature {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn parses_key_designations() {
    let cases = [
        ("*C:", DiatonicStep::C, None, KeyMode::Major),
        ("*c:", DiatonicStep::C, None, KeyMode::Minor),
        (
            "*B-:",
            DiatonicStep::B,
            Some(Accidental::Flat),
            KeyMode::Major,
        ),
        (
            "*f#:",
            DiatonicStep::F,
            Some(Accidental::Sharp),
            KeyMode::Minor,
        ),
        ("*G:", DiatonicStep::G, None, KeyMode::Major),
        ("*D:", DiatonicStep::D, None, KeyMode::Major),
        ("*A:", DiatonicStep::A, None, KeyMode::Major),
        ("*e:", DiatonicStep::E, None, KeyMode::Minor),
    ];

    for (raw, step, accidental, mode) in cases {
        assert_eq!(
            KeyDesignation::try_from(raw).unwrap(),
            KeyDesignation {
                step,
                accidental,
                mode,
            }
        );
    }
}

#[test]
fn key_designation_parser_rejects_invalid_tokens() {
    for raw in ["*C##:", "*cn:", "*H:", "*C", "*C:m"] {
        assert_eq!(
            KeyDesignation::try_from(raw).unwrap_err(),
            ParseError::InvalidKeyDesignation {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn kern_line_parses_key_designation_tandem() {
    let line = KernLine::parse(3, "*\t*B-:\t*e:").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::KeyDesignation(KeyDesignation {
                step: DiatonicStep::B,
                accidental: Some(Accidental::Flat),
                mode: KeyMode::Major,
            }),
            TandemInterpretation::KeyDesignation(KeyDesignation {
                step: DiatonicStep::E,
                accidental: None,
                mode: KeyMode::Minor,
            }),
        ])
    );
}

#[test]
fn kern_line_parses_key_signature_tandem() {
    let line = KernLine::parse(3, "*\t*k[f#c#g#]").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::KeySignature(KeySignature::from_fifths(Fifths::new(3).unwrap())),
        ])
    );
}

#[test]
fn rejects_out_of_range_fifths() {
    assert_eq!(Fifths::new(-8), None);
    assert_eq!(Fifths::new(8), None);
}

#[test]
fn key_cancellation_parses_in_kern_spines_only() {
    let document = KernDocument::parse("**kern\n*kcancel\n4c\n*Xkcancel\n4c\n*-\n").unwrap();
    let tokens = document
        .lines()
        .iter()
        .filter_map(|line| match line {
            KernLine::TandemInterpretationLine(tandems) => Some(tandems.tokens()[0].clone()),
            _ => None,
        })
        .filter(|tandem| matches!(tandem, TandemInterpretation::VisualStyle(_)))
        .collect::<Vec<_>>();
    assert_eq!(
        tokens,
        vec![
            TandemInterpretation::VisualStyle(VisualStyle::KeyCancellation(KeyCancellation {
                active: true
            })),
            TandemInterpretation::VisualStyle(VisualStyle::KeyCancellation(KeyCancellation {
                active: false
            })),
        ]
    );

    let error = KernDocument::parse("**kern\t**dynam\n*\t*kcancel\n4c\t.\n*-\t*-\n").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("\"*kcancel\" is only valid in **kern spines"),
        "{error}"
    );
    let fing = KernDocument::parse("**kern\t**fing\n*\t*kcancel\n4c\t1\n*-\t*-\n").unwrap();
    assert!(fing.lines().iter().any(|line| matches!(
        line,
        KernLine::TandemInterpretationLine(tandems)
            if tandems.tokens()[1] == TandemInterpretation::Raw(RawToken("*kcancel".to_owned()))
    )));
}
