use crate::*;

fn dynam(token: &str) -> DynamContent {
    let document = KernDocument::parse(&format!("**kern\t**dynam\n4c\t{token}\n*-\t*-\n")).unwrap();
    let contents = document
        .lines()
        .iter()
        .find_map(|line| match line {
            KernLine::Data(contents) => Some(contents),
            _ => None,
        })
        .expect("a data line");
    let SpineContent::Dynam(content) = &contents[1] else {
        panic!("expected a dynam field");
    };
    content.clone()
}

fn canonical(token: &str) -> String {
    let document = KernDocument::parse(&format!("**kern\t**dynam\n4c\t{token}\n*-\t*-\n")).unwrap();
    let text = crate::serialize::render::to_string(&document).unwrap();
    let line = text
        .lines()
        .find(|line| line.starts_with("4c"))
        .expect("the data line");
    line.split_once('\t').unwrap().1.to_owned()
}

#[test]
fn repeated_parts_are_written_once() {
    for (token, expected) in [
        ("< <", "<"),
        ("<<", "<"),
        ("[ [", "["),
        ("] ] ]", "]"),
        ("p p", "p"),
        ("..", "."),
        (". .", "."),
        ("( (", "("),
    ] {
        assert_eq!(canonical(token), expected, "{token}");
    }
}

#[test]
fn a_run_of_ends_keeps_the_line_end() {
    for (token, expected) in [("[[[", "[["), ("[ [[", "[["), ("[[ [", "[["), ("]]", "]]")] {
        assert_eq!(canonical(token), expected, "{token}");
    }
}

#[test]
fn an_end_after_a_start_closes_it_on_its_note() {
    for (token, expected) in [
        ("< [", "<["),
        ("< [[", "<["),
        ("<[", "<["),
        ("< < [[", "<["),
        ("> ]", ">]"),
    ] {
        assert_eq!(canonical(token), expected, "{token}");
    }
}

#[test]
fn glued_parts_split_and_order_as_end_continuation_level_start() {
    for (token, expected) in [
        (".[", "["),
        ("p<", "p <"),
        ("[ f >", "[ f >"),
        ("f [ >", "[ f >"),
        ("[<", "[ <"),
        ("< (", "( <"),
        ("] [", "[ ]"),
    ] {
        assert_eq!(canonical(token), expected, "{token}");
    }
}

#[test]
fn conflicting_parts_keep_the_token_raw() {
    for token in ["p f", "< >", "<>", "p cresc.", "p  <", "pX p"] {
        let DynamContent::Raw { text, .. } = dynam(token) else {
            panic!("expected {token} to stay raw");
        };
        assert_eq!(text, RawToken(token.to_owned()));
        assert_eq!(canonical(token), token);
    }
}

#[test]
fn a_raw_token_keeps_its_hairpins_and_nothing_else() {
    let begin = Some(HairpinBegin {
        single_note: false,
        provenance: None,
    });
    let mut marks = DynamMarks::default();
    marks.crescendo.end = Some(HairpinEnd {
        to_line_end: false,
        provenance: None,
    });
    marks.crescendo.begin = begin;
    marks.diminuendo.begin = begin;
    assert_eq!(
        dynam("[ [ f mf > <"),
        DynamContent::Raw {
            text: RawToken("[ [ f mf > <".to_owned()),
            marks,
        }
    );
}

#[test]
fn provenance_stays_on_its_part() {
    let DynamContent::Marks(marks) = dynam("[[X") else {
        panic!("expected marks");
    };
    assert_eq!(
        marks.crescendo.end,
        Some(HairpinEnd {
            to_line_end: true,
            provenance: Some(DynamicProvenance::Explicit),
        })
    );
    for token in ["[[X", "<x", "pX", "]x f <X", "<[X"] {
        assert_eq!(canonical(token), token, "{token}");
    }
}

#[test]
fn accented_levels_round_trip() {
    for token in ["fp", "rf", "sfp", "rfz", "sfz", "mp", "pp", "sffz"] {
        assert_eq!(canonical(token), token, "{token}");
    }
    let DynamContent::Marks(marks) = dynam("sfp") else {
        panic!("expected marks");
    };
    assert_eq!(
        marks.level,
        Some(DynamicLevel {
            value: DynamicLevelValue::FortePiano,
            subito: true,
            sforzando: false,
            accent: false,
            provenance: None,
        })
    );
}

#[test]
fn placeholders_and_silence_are_marks() {
    assert_eq!(canonical("??? ???"), "???");
    assert_eq!(canonical("r"), "r");
    assert_eq!(canonical("??? p"), "p ???");
}
