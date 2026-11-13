//! Normalization for note-adjacent control tandem interpretations.
//!
//! The pass keeps span/control interpretations close to the data they affect
//! without imposing a global order on tandem lines. Opening controls move
//! toward following data; closing controls move toward preceding data. The
//! movement is deliberately narrow and only crosses staff-definition lines.

use crate::{
    normalization::NormalizationPass, score::StateKey, KernDocument, KernLine,
    TandemInterpretation, TandemInterpretationLine,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Move selected control tandem lines next to the data they affect.
pub struct MoveControlTandemsTowardData {
    options: NormalizeOptions,
}

impl MoveControlTandemsTowardData {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for MoveControlTandemsTowardData {
    fn normalize(&self, document: &mut KernDocument) {
        let mut start = document.leading_tandem_header_range().end;
        while start < document.lines().len() {
            if !matches!(
                document.lines()[start],
                KernLine::TandemInterpretationLine(_)
            ) {
                start += 1;
                continue;
            }

            let end = document.lines()[start..]
                .iter()
                .position(|line| !matches!(line, KernLine::TandemInterpretationLine(_)))
                .map_or(document.lines().len(), |offset| start + offset);
            reorder_window(document, start, end);
            start = end;
        }
    }
}

fn reorder_window(document: &mut KernDocument, start: usize, end: usize) {
    if end.saturating_sub(start) < 2 {
        return;
    }

    let mut changed = true;
    while changed {
        changed = false;
        for index in start..(end - 1) {
            if should_swap(&document.lines()[index], &document.lines()[index + 1]) {
                document.swap_lines(index, index + 1);
                changed = true;
            }
        }
    }
}

fn should_swap(left: &KernLine, right: &KernLine) -> bool {
    matches!(
        (classify_line(left), classify_line(right)),
        (
            TandemMobility::OpeningControl,
            TandemMobility::StaffDefinition
        ) | (
            TandemMobility::StaffDefinition,
            TandemMobility::ClosingControl
        )
    )
}

fn classify_line(line: &KernLine) -> TandemMobility {
    let KernLine::TandemInterpretationLine(tandems) = line else {
        return TandemMobility::Fixed;
    };

    classify_control_line(tandems).unwrap_or_else(|| classify_non_control_line(tandems))
}

/// A line of toggle or span tandems that all switch the same way.
fn classify_control_line(tandems: &TandemInterpretationLine) -> Option<TandemMobility> {
    let mut active = None;
    for tandem in tandems.tokens() {
        let Some(tandem_active) = StateKey::switches_on(tandem) else {
            if matches!(tandem, TandemInterpretation::Null) {
                continue;
            }
            return None;
        };

        match active {
            Some(active) if active != tandem_active => return None,
            Some(_) => {}
            None => active = Some(tandem_active),
        }
    }

    active.map(|active| {
        if active {
            TandemMobility::OpeningControl
        } else {
            TandemMobility::ClosingControl
        }
    })
}

fn classify_non_control_line(tandems: &TandemInterpretationLine) -> TandemMobility {
    let defines_staff = tandems
        .tokens()
        .iter()
        .find_map(StateKey::of)
        .is_some_and(StateKey::defines_staff);
    if defines_staff {
        TandemMobility::StaffDefinition
    } else {
        TandemMobility::Fixed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TandemMobility {
    OpeningControl,
    ClosingControl,
    StaffDefinition,
    Fixed,
}

#[cfg(test)]
mod tests {
    use crate::{
        normalization::{
            control_tandem_adjacency::MoveControlTandemsTowardData, NormalizationPass,
        },
        serialize::omr::{serialize_tandem, OmrSerializeOptions},
        KernDocument, KernLine,
    };

    fn moved_lines(input: &str) -> Vec<String> {
        let mut document = KernDocument::parse(input).unwrap();
        MoveControlTandemsTowardData::new(Default::default()).normalize(&mut document);
        document
            .lines()
            .iter()
            .enumerate()
            .filter(|(index, _)| document.source_line_number(*index) != 0)
            .map(|(_, line)| line_string(line))
            .collect()
    }

    fn line_string(line: &KernLine) -> String {
        match line {
            KernLine::SpineTypeLine(_) => "spines".to_owned(),
            KernLine::TandemInterpretationLine(tandems) => tandems
                .tokens()
                .iter()
                .map(|tandem| {
                    serialize_tandem(tandem, &OmrSerializeOptions::default())
                        .unwrap_or_else(|| "*".to_owned())
                })
                .collect::<Vec<_>>()
                .join("\t"),
            KernLine::Data(_) => "data".to_owned(),
            KernLine::Barline(_) => "barline".to_owned(),
            KernLine::SpineManipulatorLine(_) => "spine-manipulator".to_owned(),
            KernLine::LocalCommentLine(_) => "local-comment".to_owned(),
        }
    }

    #[test]
    fn moves_opening_controls_toward_following_data() {
        let input = "**kern\n4c\n*8va\n*clefC1\n4d\n*-\n";

        assert_eq!(
            moved_lines(input),
            [
                "spines",
                "data",
                "*clefC1",
                "*8va",
                "data",
                "spine-manipulator"
            ]
        );
    }

    #[test]
    fn moves_closing_controls_toward_preceding_data() {
        let input = "**kern\n4c\n*clefC1\n*X8va\n4d\n*-\n";

        assert_eq!(
            moved_lines(input),
            [
                "spines",
                "data",
                "*X8va",
                "*clefC1",
                "data",
                "spine-manipulator"
            ]
        );
    }

    #[test]
    fn preserves_same_direction_control_order() {
        let input = "**kern\n4c\n*8va\n*ped\n*clefC1\n4d\n*-\n";

        assert_eq!(
            moved_lines(input),
            [
                "spines",
                "data",
                "*clefC1",
                "*8va",
                "*ped",
                "data",
                "spine-manipulator"
            ]
        );
    }

    #[test]
    fn moves_every_toggle_like_the_registry_switches_it() {
        let input = "**kern\n4c\n*rep\n*k[f#]\n4d\n*k[]\n*Xrep\n4e\n*-\n";

        assert_eq!(
            moved_lines(input),
            [
                "spines",
                "data",
                "*k[f#]",
                "*rep",
                "data",
                "*Xrep",
                "*k[]",
                "data",
                "spine-manipulator"
            ]
        );
    }

    #[test]
    fn does_not_reorder_staff_definitions_with_each_other() {
        let input = "**kern\n4c\n*clefF4\n*M3/4\n4d\n*-\n";

        assert_eq!(
            moved_lines(input),
            [
                "spines",
                "data",
                "*clefF4",
                "*M3/4",
                "data",
                "spine-manipulator"
            ]
        );
    }

    #[test]
    fn does_not_cross_non_tandem_lines() {
        let input = "**kern\n4c\n*8va\n=1\n*clefC1\n4d\n*-\n";

        assert_eq!(
            moved_lines(input),
            [
                "spines",
                "data",
                "*8va",
                "barline",
                "*clefC1",
                "data",
                "spine-manipulator"
            ]
        );
    }

    #[test]
    fn does_not_move_mixed_active_inactive_control_lines() {
        let input = "**kern\t**kern\n4c\t4e\n*8va\t*X8va\n*clefC1\t*clefC1\n4d\t4f\n*-\t*-\n";

        assert_eq!(
            moved_lines(input),
            [
                "spines",
                "data",
                "*8va\t*X8va",
                "*clefC1\t*clefC1",
                "data",
                "spine-manipulator"
            ]
        );
    }

    #[test]
    fn skips_leading_tandem_header() {
        let input = "**kern\n*8va\n*clefC1\n4c\n*-\n";

        assert_eq!(
            moved_lines(input),
            ["spines", "*8va", "*clefC1", "data", "spine-manipulator"]
        );
    }
}
