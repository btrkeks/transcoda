use crate::{
    types::{BarlineStyle, RepeatStyle},
    Barline,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BoundaryRole {
    pub(super) final_like: bool,
    pub(super) invisible: bool,
    pub(super) marked_invisible_system_break: bool,
    /// A dashed barline (`=:`), which may divide a bar instead of ending it.
    pub(super) dashed: bool,
    pub(super) repeat_start: bool,
    pub(super) repeat_end: bool,
    pub(super) repeat_end_linear_continuation: bool,
}

pub(super) fn boundary_role(barline: &Barline, no_following_data: bool) -> BoundaryRole {
    let invisible = barline.styles().contains(&BarlineStyle::Invisible);
    let repeat_start = matches!(
        barline.repeat(),
        RepeatStyle::Start | RepeatStyle::StartAndEnd
    );
    let repeat_end = matches!(
        barline.repeat(),
        RepeatStyle::End | RepeatStyle::StartAndEnd
    );
    BoundaryRole {
        final_like: !repeat_end
            && (no_following_data || is_piece_boundary_barline(barline) || repeat_start),
        invisible,
        marked_invisible_system_break: invisible && barline.system_break_before.is_some(),
        dashed: barline.styles().contains(&BarlineStyle::Dashed),
        repeat_start,
        repeat_end,
        repeat_end_linear_continuation: barline.repeat() == RepeatStyle::End && !no_following_data,
    }
}

fn is_piece_boundary_barline(barline: &Barline) -> bool {
    barline.repeat() == RepeatStyle::None
        && barline
            .styles()
            .iter()
            .all(|style| matches!(style, BarlineStyle::Double | BarlineStyle::Thin))
        && barline
            .styles()
            .iter()
            .any(|style| matches!(style, BarlineStyle::Double | BarlineStyle::Thin))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SystemBreak;

    fn role(raw: &str, no_following_data: bool) -> BoundaryRole {
        boundary_role(&Barline::try_from(raw).unwrap(), no_following_data)
    }

    #[test]
    fn classifies_repeat_roles() {
        let start = role("=|:", false);
        assert!(start.repeat_start);
        assert!(!start.repeat_end);
        assert!(start.final_like);

        let end = role("=:|!", false);
        assert!(!end.repeat_start);
        assert!(end.repeat_end);
        assert!(!end.final_like);
        assert!(end.repeat_end_linear_continuation);

        let both = role("=:|!|:", false);
        assert!(both.repeat_start);
        assert!(both.repeat_end);
        assert!(!both.final_like);
        assert!(!both.repeat_end_linear_continuation);
    }

    #[test]
    fn classifies_final_like_piece_boundaries() {
        assert!(role("=", true).final_like);
        assert!(role("=||", false).final_like);
        assert!(role("==", false).final_like);
        assert!(!role("=", false).final_like);
    }

    #[test]
    fn classifies_dashed_barlines() {
        assert!(role("=:", false).dashed);
        assert!(!role("=", false).dashed);
        assert!(!role("=:|!", false).dashed);
    }

    #[test]
    fn classifies_only_marked_invisible_barlines_as_system_breaks() {
        let unmarked_invisible = role("=-", false);
        assert!(unmarked_invisible.invisible);
        assert!(!unmarked_invisible.marked_invisible_system_break);

        let mut marked_invisible = Barline::try_from("=-").unwrap();
        marked_invisible.system_break_before = Some(SystemBreak {
            group: Some("original".to_owned()),
            source_line: Some(1),
        });
        let marked_invisible = boundary_role(&marked_invisible, false);
        assert!(marked_invisible.invisible);
        assert!(marked_invisible.marked_invisible_system_break);

        let mut marked_visible = Barline::try_from("=").unwrap();
        marked_visible.system_break_before = Some(SystemBreak {
            group: None,
            source_line: Some(1),
        });
        let marked_visible = boundary_role(&marked_visible, false);
        assert!(!marked_visible.invisible);
        assert!(!marked_visible.marked_invisible_system_break);
    }
}
