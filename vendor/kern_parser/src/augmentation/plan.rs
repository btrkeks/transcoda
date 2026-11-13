use std::collections::BTreeMap;

use crate::ast::DurationValue;
use crate::{
    document::Framing,
    score::{LayerId, Score, TrackId},
    KernContent, KernDocument, KernLine, SpineContent, SpineType, TandemInterpretation,
};

use super::{profile::RenderFeatureProfile, AugmentationError};

#[derive(Debug, Clone)]
pub(super) struct AugmentationPlan {
    pub(super) articulation_sites: Vec<MusicSite>,
    pub(super) fermata_sites: Vec<MusicSite>,
    pub(super) arpeggio_sites: Vec<MusicSite>,
    pub(super) mordent_sites: Vec<MusicSite>,
    pub(super) music_sites: Vec<MusicSite>,
    pub(super) tuplet_runs: Vec<TupletRun>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MusicSite {
    pub(super) line_index: usize,
    pub(super) spine_index: usize,
    pub(super) staff_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TupletRun {
    pub(super) spine_index: usize,
    pub(super) staff_index: usize,
    pub(super) first_line_index: usize,
    pub(super) last_line_index: usize,
}

impl TupletRun {
    pub(super) fn first_site(self) -> MusicSite {
        MusicSite {
            line_index: self.first_line_index,
            spine_index: self.spine_index,
            staff_index: self.staff_index,
        }
    }

    pub(super) fn last_site(self) -> MusicSite {
        MusicSite {
            line_index: self.last_line_index,
            spine_index: self.spine_index,
            staff_index: self.staff_index,
        }
    }
}

pub(super) fn plan_candidates(
    document: &KernDocument,
    _profile: &RenderFeatureProfile,
) -> Result<AugmentationPlan, AugmentationError> {
    let score = Score::resolve(document, Framing::Excerpt).map_err(AugmentationError::State)?;
    let topology = score.topology();
    let staff_count = document.staff_count().max(1);
    // Tuplet runs keep the staff of their track: its `**kern` position on
    // the exclusive interpretation line that introduced it.
    let tuplet_staff = |track: TrackId| {
        let born = topology.track(track).born;
        topology
            .track_ids()
            .take_while(|&other| other != track)
            .filter(|&other| {
                topology.track(other).born == born
                    && topology.track(other).spine_type == SpineType::Kern
            })
            .count()
            .min(staff_count - 1)
    };
    let mut articulation_sites = Vec::new();
    let mut fermata_sites = Vec::new();
    let mut arpeggio_sites = Vec::new();
    let mut mordent_sites = Vec::new();
    let mut music_sites = Vec::new();
    let mut tuplet_runs = Vec::new();
    let mut open_runs: BTreeMap<LayerId, TupletRun> = BTreeMap::new();

    for (line_index, line) in document.lines().iter().enumerate() {
        match line {
            KernLine::SpineTypeLine(_)
            | KernLine::SpineManipulatorLine(_)
            | KernLine::Barline(_) => {
                tuplet_runs.extend(std::mem::take(&mut open_runs).into_values());
            }
            KernLine::TandemInterpretationLine(tandems) => {
                // Tuplet number visibility is track state: a change in one
                // subspine ends the runs of the whole track.
                for (column, tandem) in tandems.tokens().iter().enumerate() {
                    if !matches!(tandem, TandemInterpretation::Tuplet(_)) {
                        continue;
                    }
                    let track = topology.track_of(topology.layer_at(line_index, column));
                    for layer in topology.layers_at(line_index) {
                        if topology.track_of(*layer) == track {
                            tuplet_runs.extend(open_runs.remove(layer));
                        }
                    }
                }
            }
            KernLine::LocalCommentLine(_) | KernLine::Data(_) => {}
        }

        let KernLine::Data(contents) = line else {
            continue;
        };

        let mut kern_ordinal = 0;
        for (spine_index, content) in contents.iter().enumerate() {
            let SpineContent::Kern(kern) = content else {
                continue;
            };
            let staff_index = kern_ordinal.min(staff_count - 1);
            kern_ordinal += 1;

            if is_pitched_content(kern) {
                let site = MusicSite {
                    line_index,
                    spine_index,
                    staff_index,
                };
                music_sites.push(site);
                if is_articulatable_content(kern) && !has_articulation(kern) {
                    articulation_sites.push(site);
                }
                if !has_ornament(kern) {
                    mordent_sites.push(site);
                }
            }

            if is_fermata_candidate(kern) && !has_fermata(kern) {
                fermata_sites.push(MusicSite {
                    line_index,
                    spine_index,
                    staff_index,
                });
            }

            if is_arpeggio_candidate(kern) && !score.state_at(line_index, spine_index).track.tremolo
            {
                arpeggio_sites.push(MusicSite {
                    line_index,
                    spine_index,
                    staff_index,
                });
            }
        }

        for (spine_index, (content, &layer)) in contents
            .iter()
            .zip(topology.layers_at(line_index))
            .enumerate()
        {
            let show_number = score.state_at(line_index, spine_index).track.tuplet_numbers;
            match content {
                SpineContent::Kern(KernContent::Null) => {}
                SpineContent::Kern(kern) if show_number && is_tupletted_content(kern) => {
                    let track = topology.track_of(layer);
                    open_runs
                        .entry(layer)
                        .and_modify(|run| run.last_line_index = line_index)
                        .or_insert_with(|| TupletRun {
                            spine_index,
                            staff_index: if topology.track(track).spine_type == SpineType::Kern {
                                tuplet_staff(track)
                            } else {
                                0
                            },
                            first_line_index: line_index,
                            last_line_index: line_index,
                        });
                }
                _ => tuplet_runs.extend(open_runs.remove(&layer)),
            }
        }
    }

    tuplet_runs.extend(open_runs.into_values());
    tuplet_runs.sort_by_key(|run| (run.first_line_index, run.spine_index, run.last_line_index));

    Ok(AugmentationPlan {
        articulation_sites,
        fermata_sites,
        arpeggio_sites,
        mordent_sites,
        music_sites,
        tuplet_runs,
    })
}

fn is_pitched_content(content: &KernContent) -> bool {
    matches!(
        content,
        KernContent::Note(_) | KernContent::Chord(_) | KernContent::Grace(_)
    )
}

fn is_articulatable_content(content: &KernContent) -> bool {
    is_pitched_content(content)
}

// Verovio can erase a rest fermata during multi-rest compaction and does not
// retain its source address on all full-measure rests. Pitched sites stay labelled.
fn is_fermata_candidate(content: &KernContent) -> bool {
    matches!(
        content,
        KernContent::Note(_) | KernContent::Chord(_) | KernContent::Grace(_)
    )
}

fn is_arpeggio_candidate(content: &KernContent) -> bool {
    matches!(content, KernContent::Chord(chord) if chord.notes.len() > 1 && chord.modifiers.arpeggio.is_none())
}

fn has_articulation(content: &KernContent) -> bool {
    match content {
        KernContent::Chord(chord) => !chord.modifiers.articulations.is_empty(),
        KernContent::Grace(grace) => !grace.modifiers.articulations.is_empty(),
        KernContent::Note(_)
        | KernContent::Rest(_)
        | KernContent::InvisibleRestSlice(_)
        | KernContent::Null
        | KernContent::MultiMeasureRest(_) => false,
    }
}

fn has_fermata(content: &KernContent) -> bool {
    match content {
        KernContent::Chord(chord) => chord.modifiers.fermata.is_some(),
        KernContent::Grace(grace) => grace.modifiers.fermata.is_some(),
        KernContent::Rest(rest) => rest.fermata.is_some(),
        KernContent::Note(_)
        | KernContent::InvisibleRestSlice(_)
        | KernContent::Null
        | KernContent::MultiMeasureRest(_) => false,
    }
}

fn has_ornament(content: &KernContent) -> bool {
    match content {
        KernContent::Chord(chord) => !chord.modifiers.ornaments.is_empty(),
        KernContent::Grace(grace) => !grace.modifiers.ornaments.is_empty(),
        KernContent::Note(_)
        | KernContent::Rest(_)
        | KernContent::InvisibleRestSlice(_)
        | KernContent::Null
        | KernContent::MultiMeasureRest(_) => false,
    }
}

fn is_tupletted_content(content: &KernContent) -> bool {
    match content {
        KernContent::Note(note) => is_tupletted_duration_value(&note.duration.value),
        KernContent::Chord(chord) => chord
            .notes
            .first()
            .is_some_and(|note| is_tupletted_duration_value(&note.duration.value)),
        KernContent::Rest(rest) => is_tupletted_duration_value(&rest.duration.value),
        KernContent::Grace(_)
        | KernContent::InvisibleRestSlice(_)
        | KernContent::Null
        | KernContent::MultiMeasureRest(_) => false,
    }
}

fn is_tupletted_duration_value(value: &DurationValue) -> bool {
    match value {
        DurationValue::Reciprocal(value) => !value.is_power_of_two(),
        DurationValue::Rational { .. } => true,
        DurationValue::Breve | DurationValue::Long | DurationValue::Maxima => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuplet_runs(input: &str) -> Vec<TupletRun> {
        let document = KernDocument::parse(input).unwrap();
        plan_candidates(&document, &RenderFeatureProfile::from_document(&document))
            .unwrap()
            .tuplet_runs
    }

    #[test]
    fn hidden_tuplet_numbers_in_one_subspine_hide_the_whole_track() {
        let shown = "**kern\t**kern\n*^\t*\n6c\t6e\t4G\n6d\t6f\t.\n6e\t6g\t.\n*v\t*v\t*\n*-\t*-\n";
        let runs = tuplet_runs(shown);
        assert_eq!(runs.len(), 2);
        assert!(runs.iter().all(|run| run.staff_index == 0));
        let hidden = shown.replace("*^\t*\n", "*^\t*\n*Xtuplet\t*\t*\n");
        assert!(tuplet_runs(&hidden).is_empty());
    }
}
