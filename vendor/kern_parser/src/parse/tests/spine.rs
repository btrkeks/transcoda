use crate::*;

#[test]
fn kern_line_parses_spine_manipulators() {
    let line = KernLine::parse(3, "*^\t*").unwrap();

    assert_eq!(
        line,
        KernLine::SpineManipulatorLine(
            SpineManipulatorLine::new(vec![SpineManipulator::Split, SpineManipulator::Null])
                .unwrap()
        )
    );
}

#[test]
fn kern_line_parses_split_split_null_manipulators() {
    let line = KernLine::parse(3, "*^\t*^\t*").unwrap();

    assert_eq!(
        line,
        KernLine::SpineManipulatorLine(
            SpineManipulatorLine::new(vec![
                SpineManipulator::Split,
                SpineManipulator::Split,
                SpineManipulator::Null,
            ])
            .unwrap()
        )
    );
}

#[test]
fn kern_line_parses_merge_merge_then_null_manipulators() {
    let line = KernLine::parse(3, "*v\t*v\t*\t*\t*").unwrap();

    assert_eq!(
        line,
        KernLine::SpineManipulatorLine(
            SpineManipulatorLine::new(vec![
                SpineManipulator::Merge,
                SpineManipulator::Merge,
                SpineManipulator::Null,
                SpineManipulator::Null,
                SpineManipulator::Null,
            ])
            .unwrap()
        )
    );
}

#[test]
fn kern_line_parses_null_then_merge_merge_manipulators() {
    let line = KernLine::parse(3, "*\t*v\t*v\t*\t*").unwrap();

    assert_eq!(
        line,
        KernLine::SpineManipulatorLine(
            SpineManipulatorLine::new(vec![
                SpineManipulator::Null,
                SpineManipulator::Merge,
                SpineManipulator::Merge,
                SpineManipulator::Null,
                SpineManipulator::Null,
            ])
            .unwrap()
        )
    );
}

#[test]
fn kern_line_parses_split_and_merge_manipulators_on_same_line() {
    let line = KernLine::parse(3, "*^\t*v\t*v").unwrap();

    assert_eq!(
        line,
        KernLine::SpineManipulatorLine(
            SpineManipulatorLine::new(vec![
                SpineManipulator::Split,
                SpineManipulator::Merge,
                SpineManipulator::Merge,
            ])
            .unwrap()
        )
    );
}

#[test]
fn kern_document_tracks_same_line_split_and_merge() {
    let document = KernDocument::parse("**kern\t**kern\t**kern\n*^\t*v\t*v\n4c\t4e\t4g\n").unwrap();

    assert_eq!(document.lines().len(), 5);
}

#[test]
fn kern_line_rejects_lone_merge_manipulator() {
    let error = KernLine::parse(3, "*v\t*\t*v").unwrap_err();

    assert_eq!(
        error,
        ParseError::InvalidSpineManipulatorLine {
            line: 3,
            column: 1,
            message: "merge requires at least 2 consecutive *v tokens".to_owned(),
        }
    );
}

#[test]
fn kern_line_rejects_mixed_spine_manipulator_types() {
    let error = KernLine::parse(3, "*^\t*-\t*").unwrap_err();

    assert_eq!(
        error,
        ParseError::MixedSpineManipulatorTypes {
            line: 3,
            column: 2,
            expected: "split",
            found: "terminate",
        }
    );
}

#[test]
fn kern_line_rejects_merge_mixed_with_other_manipulator_types() {
    let error = KernLine::parse(3, "*v\t*v\t*-").unwrap_err();

    assert_eq!(
        error,
        ParseError::MixedSpineManipulatorTypes {
            line: 3,
            column: 3,
            expected: "merge",
            found: "terminate",
        }
    );
}

#[test]
fn kern_line_rejects_unsupported_spine_manipulators() {
    let error = KernLine::parse(3, "*x").unwrap_err();

    assert_eq!(
        error,
        ParseError::InvalidSpineToken {
            line: 3,
            column: 1,
            raw: "*x".to_owned(),
            source: Box::new(ParseError::UnsupportedSpineManipulator {
                raw: "*x".to_owned(),
            }),
        }
    );
}

#[test]
fn parses_unknown_exclusive_interpretation_as_opaque_spine_type() {
    assert_eq!(
        SpineType::try_from("**foo").unwrap(),
        SpineType::Other("**foo".to_owned())
    );
}
