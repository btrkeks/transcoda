use kern_parser::{ParseError, StaffPlacement, TandemInterpretation, VisualStyle};

#[test]
fn parses_staff_placement_as_typed_visual_state() {
    for (raw, placement) in [
        ("*above", StaffPlacement::Above),
        ("*below", StaffPlacement::Below),
    ] {
        assert_eq!(
            TandemInterpretation::try_from(raw).unwrap(),
            TandemInterpretation::VisualStyle(VisualStyle::StaffPlacement(placement))
        );
    }

    for raw in ["*above:2", "*below:2", "*Xabove", "*Xbelow", "*abovefoo"] {
        assert_eq!(
            TandemInterpretation::try_from(raw).unwrap_err(),
            ParseError::UnsupportedTandemInterpretation {
                raw: raw.to_owned(),
            }
        );
    }
}
