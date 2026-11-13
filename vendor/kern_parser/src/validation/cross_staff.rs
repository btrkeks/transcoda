use std::collections::BTreeSet;

use crate::{score::Score, CrossStaffPlacement, KernPitchedNote, SpineType};

use super::{SourceLocation, ValidationSeverity, ValidationWarning};

struct StaffBounds {
    upper: u32,
    lower: u32,
}

impl StaffBounds {
    /// `None` when any `**kern` track has no staff number.
    fn of(score: &Score<'_>) -> Option<Self> {
        let topology = score.topology();
        let kern = topology
            .track_ids()
            .map(|track| topology.track(track))
            .filter(|track| track.spine_type == SpineType::Kern)
            .map(|track| track.staff)
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            upper: kern.iter().copied().min()?,
            lower: kern.iter().copied().max()?,
        })
    }
}

pub(crate) fn check(score: &Score<'_>) -> Vec<ValidationWarning> {
    let document = score.document();
    let topology = score.topology();
    if !document
        .located_noteheads()
        .any(|notehead| placement(notehead.note).is_some())
    {
        return Vec::new();
    }
    let Some(bounds) = StaffBounds::of(score) else {
        return Vec::new();
    };
    let mut warnings = Vec::new();
    let mut warned = BTreeSet::new();
    for notehead in document.located_noteheads() {
        let location = notehead.location;
        let layer = topology.layer_at(location.line_index, location.spine_index.0);
        let Some(staff) = topology.staff_of(layer) else {
            continue;
        };
        let (code, marker, boundary, direction) = match placement(notehead.note) {
            Some(CrossStaffPlacement::Above) if staff == bounds.upper => {
                ("cross-staff.no-staff-above", '>', "uppermost", "above")
            }
            Some(CrossStaffPlacement::Below) if staff == bounds.lower => {
                ("cross-staff.no-staff-below", '<', "lowest", "below")
            }
            _ => continue,
        };
        if warned.insert((location.line_index, location.spine_index.0, code)) {
            warnings.push(ValidationWarning {
                span: None,
                severity: ValidationSeverity::Warning,
                loose_end: None,
                code,
                location: SourceLocation {
                    line: location.line_number,
                    column: Some(location.spine_index.0 + 1),
                },
                message: format!(
                    "pitch-attached `{marker}` moves a note {direction} the {boundary} staff {staff}, but no destination staff exists {direction} it"
                ),
            });
        }
    }
    warnings
}

fn placement(note: KernPitchedNote<'_>) -> Option<CrossStaffPlacement> {
    match note {
        KernPitchedNote::Regular(note) => note.modifiers.cross_staff,
        KernPitchedNote::Grace(note) => note.modifiers.cross_staff,
    }
}
