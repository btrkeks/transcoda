use crate::*;

fn tandem_line(tokens: Vec<TandemInterpretation>) -> KernLine {
    KernLine::TandemInterpretationLine(TandemInterpretationLine::new(tokens).unwrap())
}

fn instrument(text: &str, kind: InstrumentKind) -> Instrument {
    Instrument {
        text: text.to_owned(),
        kind,
    }
}

#[test]
fn parses_instrument_codes() {
    for (raw, text) in [
        ("*Ivioln", "violn"),
        ("*Icello", "cello"),
        ("*ICstr", "Cstr"),
        ("*IGcont", "Gcont"),
        ("*I#2", "#2"),
    ] {
        assert_eq!(
            Instrument::try_from(raw).unwrap(),
            instrument(text, InstrumentKind::Code)
        );
    }
}

#[test]
fn parses_instrument_names() {
    let cases = [
        ("*I\"Piano", "Piano"),
        ("*I\"Grand Piano", "Grand Piano"),
        ("*I\"Tenor", "Tenor"),
        ("*I\"Bass", "Bass"),
        ("*I\"Alto", "Alto"),
        ("*I\"Soprano", "Soprano"),
        ("*I\"Voice", "Voice"),
        ("*I\"Trombone", "Trombone"),
        ("*I\"Flute", "Flute"),
        ("*I\"Violin", "Violin"),
        ("*I\"Violoncello", "Violoncello"),
        ("*I\"B♭ Trumpet", "B♭ Trumpet"),
        ("*I\"Viola", "Viola"),
        ("*I\"Drumset", "Drumset"),
        ("*I\"Tuba", "Tuba"),
        ("*I\"Alto Saxophone", "Alto Saxophone"),
        ("*I\"Grand Piano, Soprano", "Grand Piano, Soprano"),
        ("*I\"B♭ Clarinet", "B♭ Clarinet"),
        ("*I\"Tenor Saxophone", "Tenor Saxophone"),
        ("*I\"Horn in F", "Horn in F"),
        ("*I\"Tenor 1", "Tenor 1"),
        ("*I\"Tenor 2", "Tenor 2"),
        ("*I\"Euphonium", "Euphonium"),
        ("*I\"Baritone Saxophone", "Baritone Saxophone"),
    ];

    for (raw, code) in cases {
        assert_eq!(
            Instrument::try_from(raw).unwrap(),
            instrument(code, InstrumentKind::Name(InstrumentTextVariant::Regular))
        );
    }
}

#[test]
fn parses_modern_and_original_instrument_names() {
    for (raw, text, variant) in [
        ("*mI\"Violin", "Violin", InstrumentTextVariant::Modern),
        ("*oI\"Violino", "Violino", InstrumentTextVariant::Original),
    ] {
        assert_eq!(
            Instrument::try_from(raw).unwrap(),
            instrument(text, InstrumentKind::Name(variant))
        );
    }
}

#[test]
fn parses_instrument_abbreviations() {
    for (raw, text, variant) in [
        ("*I'Vln.", "Vln.", InstrumentTextVariant::Regular),
        ("*mI'Vln.", "Vln.", InstrumentTextVariant::Modern),
        ("*oI'Vl.", "Vl.", InstrumentTextVariant::Original),
    ] {
        assert_eq!(
            Instrument::try_from(raw).unwrap(),
            instrument(text, InstrumentKind::Abbreviation(variant))
        );
    }
}

#[test]
fn kern_line_parses_instrument_tandem() {
    let line = KernLine::parse(3, "*\t*I\"B♭ Clarinet").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::Instrument(instrument(
                "B♭ Clarinet",
                InstrumentKind::Name(InstrumentTextVariant::Regular)
            )),
        ])
    );
}

#[test]
fn kern_line_parses_modern_instrument_name() {
    let line = KernLine::parse(3, "*\t*mI\"Violin").unwrap();

    assert_eq!(
        line,
        tandem_line(vec![
            TandemInterpretation::Null,
            TandemInterpretation::Instrument(instrument(
                "Violin",
                InstrumentKind::Name(InstrumentTextVariant::Modern)
            )),
        ])
    );
}

#[test]
fn parses_empty_display_labels() {
    for (raw, kind) in [
        ("*I\"", InstrumentKind::Name(InstrumentTextVariant::Regular)),
        ("*mI\"", InstrumentKind::Name(InstrumentTextVariant::Modern)),
        (
            "*oI\"",
            InstrumentKind::Name(InstrumentTextVariant::Original),
        ),
        (
            "*I'",
            InstrumentKind::Abbreviation(InstrumentTextVariant::Regular),
        ),
        (
            "*mI'",
            InstrumentKind::Abbreviation(InstrumentTextVariant::Modern),
        ),
        (
            "*oI'",
            InstrumentKind::Abbreviation(InstrumentTextVariant::Original),
        ),
    ] {
        assert_eq!(Instrument::try_from(raw).unwrap(), instrument("", kind));
    }
}

#[test]
fn instrument_parser_rejects_empty_codes() {
    {
        let raw = "*I";
        assert_eq!(
            Instrument::try_from(raw).unwrap_err(),
            ParseError::InvalidInstrument {
                raw: raw.to_owned(),
            }
        );
    }
}

#[test]
fn instrument_parser_rejects_unsupported_modori_code_forms() {
    for raw in ["*mIvioln", "*oIvioln"] {
        assert_eq!(
            Instrument::try_from(raw).unwrap_err(),
            ParseError::InvalidInstrument {
                raw: raw.to_owned(),
            }
        );
    }
}
