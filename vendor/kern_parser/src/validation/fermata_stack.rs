//! Warnings for simultaneous fermatas Verovio draws on the same side of one staff.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ast::{Fermata, FermataPlacement},
    score::Score,
    validation::{SourceLocation, ValidationSeverity, ValidationWarning},
    CrossStaffPlacement, KernContent, KernLine, SpineContent, SpineType,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Side {
    Above,
    Below,
}

impl Side {
    fn word(self) -> &'static str {
        match self {
            Self::Above => "above",
            Self::Below => "below",
        }
    }
}

pub(crate) fn check(score: &Score<'_>) -> Vec<ValidationWarning> {
    let document = score.document();
    let topology = score.topology();
    let staffs: BTreeSet<u32> = topology
        .track_ids()
        .map(|track| topology.track(track))
        .filter(|track| track.spine_type == SpineType::Kern)
        .filter_map(|track| track.staff)
        .collect();
    let mut warnings = Vec::new();
    for (line_index, line) in document.lines().iter().enumerate() {
        let KernLine::Data(contents) = line else {
            continue;
        };
        let line_number = document.source_line_number(line_index);
        let mut first_at = BTreeMap::<(u32, Side), usize>::new();
        for (column, content) in contents.iter().enumerate() {
            let SpineContent::Kern(content) = content else {
                continue;
            };
            let Some((fermata, shift)) = drawn_fermata(content) else {
                continue;
            };
            let layer = topology.layer_at(line_index, column);
            let track = topology.track_of(layer);
            let Some(staff) = topology
                .staff_of(layer)
                .and_then(|staff| destination(&staffs, staff, shift))
            else {
                continue;
            };
            let side = side(
                fermata,
                column - topology.track_columns(line_index, track).start,
            );
            let field = column + 1;
            let Some(&first) = first_at.get(&(staff, side)) else {
                first_at.insert((staff, side), field);
                continue;
            };
            let side = side.word();
            warnings.push(ValidationWarning {
                span: None,
                severity: ValidationSeverity::Warning,
                code: "fermata-stack.same-side",
                location: SourceLocation {
                    line: line_number,
                    column: Some(field),
                },
                message: format!(
                    "fermata at line {line_number}, field {field} is drawn {side} staff {staff} at the same beat as the fermata in field {first}, so Verovio stacks the two. Delete the duplicate, or move one with `;<` (below) or `;>` (above)."
                ),
                loose_end: None,
            });
        }
    }
    warnings
}

/// The fermata Verovio draws at this token's onset, and the staff change
/// shared by all of its noteheads.
///
/// Verovio suppresses a fermata on any token containing `yy`, and attaches a
/// grace note's fermata to the grace note rather than the beat.
fn drawn_fermata(content: &KernContent) -> Option<(Fermata, Option<CrossStaffPlacement>)> {
    match content {
        KernContent::Chord(chord) => {
            let fermata = chord.modifiers.fermata?;
            if chord.notes.iter().any(|note| note.modifiers.invisible) {
                return None;
            }
            let first = chord.notes.first()?.modifiers.cross_staff;
            let shared = chord
                .notes
                .iter()
                .all(|note| note.modifiers.cross_staff == first);
            Some((fermata, first.filter(|_| shared)))
        }
        KernContent::Rest(rest) if !rest.invisible => {
            rest.fermata.map(|fermata| (fermata, rest.cross_staff))
        }
        _ => None,
    }
}

/// Verovio's side for a fermata without a signifier: layer 1 above, layer 2
/// below, and further layers fall back to the fermata default above.
fn side(fermata: Fermata, layer: usize) -> Side {
    match fermata.placement {
        Some(FermataPlacement::Above) => Side::Above,
        Some(FermataPlacement::Below) => Side::Below,
        None if layer == 1 => Side::Below,
        None => Side::Above,
    }
}

/// The staff the fermata is drawn on, or `None` when a staff change points
/// past the outermost staff (the cross-staff rule reports that). Staff 1 is
/// the top staff.
fn destination(
    staffs: &BTreeSet<u32>,
    staff: u32,
    shift: Option<CrossStaffPlacement>,
) -> Option<u32> {
    let drawn = match shift {
        None => staff,
        Some(CrossStaffPlacement::Below) => staff + 1,
        Some(CrossStaffPlacement::Above) => staff.checked_sub(1)?,
    };
    staffs.contains(&drawn).then_some(drawn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{document::Framing, KernDocument};

    const GRAND_STAFF: &str = "!!!RDF**kern: < = below\n!!!RDF**kern: > = above\n**kern\t**kern\n";

    fn check_text(input: &str) -> Vec<ValidationWarning> {
        let document = KernDocument::parse(input).unwrap();
        check(&Score::resolve(&document, Framing::Complete).unwrap())
    }

    fn warnings(data: &str) -> Vec<ValidationWarning> {
        check_text(&format!("{GRAND_STAFF}{data}*-\t*-\n"))
    }

    #[test]
    fn warns_when_an_upper_chord_crosses_onto_a_lower_staff_fermata() {
        let warnings = warnings("4EE-\\;\t4B-< 4G< 4E-<;\n");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].code, "fermata-stack.same-side");
        assert_eq!(
            warnings[0].location,
            SourceLocation {
                line: 4,
                column: Some(2)
            }
        );
        assert_eq!(
            warnings[0].message,
            "fermata at line 4, field 2 is drawn above staff 2 at the same beat as the fermata in field 1, so Verovio stacks the two. Delete the duplicate, or move one with `;<` (below) or `;>` (above)."
        );
    }

    #[test]
    fn accepts_a_moved_fermata() {
        assert!(warnings("4EE-\\;<\t4B-< 4G< 4E-<;\n").is_empty());
    }

    #[test]
    fn accepts_one_fermata_per_staff() {
        assert!(warnings("4EE-;\t4e-;\n").is_empty());
    }

    #[test]
    fn accepts_a_partly_crossed_chord_on_its_own_staff() {
        assert!(warnings("4EE-;\t4B-< 4e-;\n").is_empty());
    }

    #[test]
    fn warns_when_a_lower_note_crosses_onto_an_upper_staff_fermata() {
        let warnings = warnings("4c>;\t4e;\n");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].location.column, Some(2));
        assert!(warnings[0].message.contains("above staff 1"));
    }

    #[test]
    fn ignores_crossings_past_the_outermost_staff() {
        assert!(warnings("4C<;\t4e;\n").is_empty());
    }

    #[test]
    fn places_unsignified_fermatas_by_layer() {
        let input = "**kern\n*^\n*^\t*\n4e;\t4c;\t4G;\n*-\t*-\t*-\n";
        let warnings = check_text(input);
        // Layer 1 is above and layer 2 below; layer 3 falls back to above.
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].location.column, Some(3));
        assert!(warnings[0].message.contains("above staff 1"));
        assert!(warnings[0].message.contains("field 1"));
    }

    #[test]
    fn accepts_two_layers_with_default_placement() {
        let input = "**kern\n*^\n4e;\t4c;\n*v\t*v\n*-\n";
        assert!(check_text(input).is_empty());
    }

    #[test]
    fn warns_on_layers_forced_to_the_same_side() {
        let input = "**kern\n*^\n4e;\t4c;>\n*v\t*v\n*-\n";
        let warnings = check_text(input);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].location.column, Some(2));
    }

    #[test]
    fn counts_visible_rests() {
        assert_eq!(warnings("4r;\t4c<;\n").len(), 1);
    }

    #[test]
    fn moves_a_crossed_rest_fermata_with_the_rest() {
        assert_eq!(warnings("4C;\t4r<;\n").len(), 1);
        assert!(warnings("4C;\t4r;\n").is_empty());
    }

    #[test]
    fn ignores_fermatas_verovio_hides() {
        assert!(warnings("4ryy;\t4c<;\n").is_empty());
        assert!(warnings("4C;\t4c<yy;\n").is_empty());
    }

    #[test]
    fn ignores_grace_note_fermatas() {
        assert!(warnings("4C;\t8qc<;\n.\t4c\n").is_empty());
    }

    #[test]
    fn ignores_fermatas_on_different_beats() {
        assert!(warnings("4C;\t8c<\n.\t8c<;\n").is_empty());
    }

    #[test]
    fn numbers_staves_like_verovio_when_declarations_ascend() {
        // `*staff1 *staff2` makes the left spine the top staff, so `>` on the
        // right spine moves onto it.
        let warnings = check_text("**kern\t**kern\n*staff1\t*staff2\n4e;\t4c>;\n*-\t*-\n");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("above staff 1"));
        assert!(warnings[0].message.contains("field 1"));
    }

    #[test]
    fn ignores_non_kern_spines_when_numbering_staffs() {
        let input = format!(
            "{}**kern\t**dynam\t**kern\n4EE-;\tp\t4B-< 4G<;\n*-\t*-\t*-\n",
            &GRAND_STAFF[..GRAND_STAFF.len() - "**kern\t**kern\n".len()]
        );
        let warnings = check_text(&input);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].location.column, Some(3));
    }
}
