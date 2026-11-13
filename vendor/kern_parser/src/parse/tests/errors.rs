use crate::*;

#[test]
fn kern_document_rejects_data_line_with_too_few_spines() {
    assert_eq!(
        KernDocument::parse("**kern\t**kern\n4c\n").unwrap_err(),
        ParseError::SpineCountMismatch {
            line: 2,
            expected: 2,
            found: 1,
        }
    );
}

#[test]
fn kern_document_rejects_data_line_with_too_many_spines() {
    assert_eq!(
        KernDocument::parse("**kern\n4c\t4e\n").unwrap_err(),
        ParseError::SpineCountMismatch {
            line: 2,
            expected: 1,
            found: 2,
        }
    );
}

#[test]
fn kern_document_rejects_underwide_line_after_split() {
    assert_eq!(
        KernDocument::parse("**kern\n*^\n4c\n").unwrap_err(),
        ParseError::SpineCountMismatch {
            line: 3,
            expected: 2,
            found: 1,
        }
    );
}

#[test]
fn kern_document_rejects_local_comment_line_with_wrong_spine_count() {
    assert_eq!(
        KernDocument::parse("**kern\t**kern\n! one comment\n").unwrap_err(),
        ParseError::SpineCountMismatch {
            line: 2,
            expected: 2,
            found: 1,
        }
    );
}

#[test]
fn kern_document_rejects_repeated_spine_type_line_with_wrong_spine_count() {
    assert_eq!(
        KernDocument::parse("**kern\t**kern\n**kern\n").unwrap_err(),
        ParseError::SpineCountMismatch {
            line: 2,
            expected: 2,
            found: 1,
        }
    );
}

#[test]
fn kern_document_allows_spine_type_line_after_initial_comments() {
    let document =
        KernDocument::parse("!!!OTL: Prelude\n!! global comment\n!LO:TX:a:t=Intro\n**kern\n4c\n")
            .unwrap();

    assert!(document
        .lines()
        .iter()
        .any(|line| matches!(line, KernLine::SpineTypeLine(_))));
}

#[test]
fn kern_document_allows_documents_without_spine_type_line() {
    KernDocument::parse("*clefG2\n*k[]\n4c\n").unwrap();
}

#[test]
fn kern_document_rejects_repeated_spine_type_line_at_start() {
    assert_eq!(
        KernDocument::parse("**kern\n**kern\n").unwrap_err(),
        ParseError::SpineTypeLineNotAtStart { line: 2 }
    );
}

#[test]
fn kern_document_rejects_spine_type_line_after_tandem_line() {
    assert_eq!(
        KernDocument::parse("*clefG2\n**kern\n").unwrap_err(),
        ParseError::SpineTypeLineNotAtStart { line: 2 }
    );
}

#[test]
fn kern_document_rejects_spine_type_line_after_data_line() {
    assert_eq!(
        KernDocument::parse("4c\n**kern\n").unwrap_err(),
        ParseError::SpineTypeLineNotAtStart { line: 2 }
    );
}

#[test]
fn kern_document_rejects_unsupported_typed_tokens() {
    let error = KernDocument::parse("*unknown\n").unwrap_err();

    assert_eq!(
        error,
        ParseError::InvalidSpineToken {
            line: 1,
            column: 1,
            raw: "*unknown".to_owned(),
            source: Box::new(ParseError::UnsupportedTandemInterpretation {
                raw: "*unknown".to_owned(),
            }),
        }
    );
}

#[test]
fn kern_document_rejects_noncanonical_key_signatures() {
    let error = KernDocument::parse("*k[bef#]\n").unwrap_err();

    assert_eq!(
        error,
        ParseError::InvalidSpineToken {
            line: 1,
            column: 1,
            raw: "*k[bef#]".to_owned(),
            source: Box::new(ParseError::InvalidKeySignature {
                raw: "*k[bef#]".to_owned(),
            }),
        }
    );
}

#[test]
fn kern_document_preserves_existing_line_errors() {
    assert_eq!(
        KernDocument::parse("**kern\t\n").unwrap_err(),
        ParseError::EmptySpineToken { line: 1, column: 2 }
    );

    assert_eq!(
        KernDocument::parse("!ok\t4c\n").unwrap_err(),
        ParseError::MixedKernLineTypes {
            line: 1,
            column: 2,
            expected: "local comment",
            found: "data",
        }
    );
}

#[test]
fn kern_document_rejects_empty_fields_on_non_data_lines() {
    for input in [
        "**kern\t**kern\n*clefG2\t\n",
        "**kern\t**kern\n*-\t\n",
        "**kern\t**kern\n=\t\n",
        "**kern\t**kern\n!LO:TX:a:t=Allegro\t\n",
    ] {
        assert_eq!(
            KernDocument::parse(input).unwrap_err(),
            ParseError::EmptySpineToken { line: 2, column: 2 },
            "{input}"
        );
    }
}

#[test]
fn kern_document_rejects_empty_fields_on_permissive_non_data_lines() {
    for input in [
        "**text\t**kern\n*clefG2\t\n",
        "**dynam\t**kern\n*-\t\n",
        "**foo\t**kern\n=\t\n",
        "**text\t**kern\n!LO:TX:a:t=Allegro\t\n",
    ] {
        assert_eq!(
            KernDocument::parse(input).unwrap_err(),
            ParseError::EmptySpineToken { line: 2, column: 2 },
            "{input}"
        );
    }
}

#[test]
fn kern_document_reports_later_errors_after_valid_empty_kern_data_field() {
    assert_eq!(
        KernDocument::parse("**kern\t**dynam\n\t\n").unwrap_err(),
        ParseError::EmptySpineToken { line: 2, column: 2 }
    );

    assert!(matches!(
        KernDocument::parse("**kern\t**kern\n\tmf\n").unwrap_err(),
        ParseError::InvalidSpineToken {
            line: 2,
            column: 2,
            raw,
            source,
        } if raw == "mf" && matches!(*source, ParseError::InvalidNote { .. })
    ));
}

#[test]
fn kern_line_rejects_empty_lines() {
    let error = KernLine::parse(7, "").unwrap_err();

    assert_eq!(error, ParseError::EmptyKernLine { line: 7 });

    let document = KernDocument::parse("\n\n").unwrap();
    assert!(document.lines().is_empty());
}

#[test]
fn kern_line_rejects_null_interpretation_token_in_data_line() {
    let error = KernLine::parse(3, ".\t*").unwrap_err();

    assert_eq!(
        error,
        ParseError::MixedKernLineTypes {
            line: 3,
            column: 2,
            expected: "data",
            found: "spine manipulator",
        }
    );
}

#[test]
fn kern_line_rejects_null_interpretation_token_in_barline() {
    let error = KernLine::parse(3, "=1\t*").unwrap_err();

    assert_eq!(
        error,
        ParseError::MixedKernLineTypes {
            line: 3,
            column: 2,
            expected: "barline",
            found: "spine manipulator",
        }
    );
}

#[test]
fn kern_line_rejects_null_interpretation_token_in_spine_type_line() {
    let error = KernLine::parse(3, "**kern\t*").unwrap_err();

    assert_eq!(
        error,
        ParseError::MixedKernLineTypes {
            line: 3,
            column: 2,
            expected: "spine type",
            found: "spine manipulator",
        }
    );
}
