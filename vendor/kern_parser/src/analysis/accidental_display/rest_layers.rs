//! `fillContentsOfLayer` returns early for full-measure rest layers. Staff
//! controls in those layers, including *tremolo and *Xtremolo, are never applied.
use crate::{KernContent, KernDocument, KernLine, MusicalDuration, SpineContent};

pub(super) fn skips_staff_state(
    score: &KernDocument,
    line: usize,
    field: usize,
    meter: Option<MusicalDuration>,
) -> bool {
    let Some(meter) = meter else { return false };
    let lines = score.lines();
    let Some(start) = (0..line)
        .rev()
        .find(|&i| matches!(lines[i], KernLine::Barline(_)))
    else {
        return false;
    };
    let end = (line + 1..lines.len())
        .find(|&i| matches!(lines[i], KernLine::Barline(_)))
        .unwrap_or(lines.len());
    let mut totals = Vec::new();
    let mut count = 0;
    let mut elapsed = MusicalDuration::zero();
    for entry in &lines[start + 1..end] {
        match entry {
            KernLine::SpineManipulatorLine(manipulators)
                if !manipulators
                    .tokens()
                    .iter()
                    .all(|t| matches!(t, crate::SpineManipulator::Terminate)) =>
            {
                return false;
            }
            KernLine::Data(contents) => {
                totals.resize(contents.len(), MusicalDuration::zero());
                let mut slice = None;
                for (fi, content) in contents.iter().enumerate() {
                    let SpineContent::Kern(content) = content else {
                        continue;
                    };
                    let Ok(duration) = content.duration() else {
                        return false;
                    };
                    totals[fi] += duration;
                    if !duration.is_zero() {
                        slice =
                            Some(slice.map_or(duration, |old: MusicalDuration| old.min(duration)));
                    }
                    if fi == field && !matches!(content, KernContent::Null) {
                        count += 1;
                        if count > 1
                            || !matches!(content, KernContent::Rest(_))
                            || !elapsed.is_zero()
                        {
                            return false;
                        }
                    }
                }
                elapsed += slice.unwrap_or(MusicalDuration::zero());
            }
            _ => {}
        }
    }
    // For a stable-width rest layer, its onset is the measure start and the
    // longest voice gives the measure duration. Null data do not add duration.
    totals.into_iter().max() == Some(meter)
}
