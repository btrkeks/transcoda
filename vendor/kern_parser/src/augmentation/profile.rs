use crate::ast::{AuxInterval, Ornament, Placed};
use crate::{
    score::StateKey, DynamContent, KernContent, KernDocument, KernLine, LocalComment, SpineContent,
    TandemInterpretation,
};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct RenderFeatureProfile {
    pub(super) articulation_count: usize,
    pub(super) fermata_count: usize,
    pub(super) arpeggio_count: usize,
    pub(super) mordent_count: usize,
    pub(super) has_dynamic_levels: bool,
    pub(super) has_hairpins: bool,
    pub(super) has_pedal: bool,
    pub(super) has_ottava: bool,
    pub(super) has_tempo: bool,
    pub(super) has_staff_names: bool,
    pub(super) has_layout_text: bool,
    pub(super) has_title: bool,
    pub(super) has_composer: bool,
    pub(super) has_movement_designation: bool,
}

impl RenderFeatureProfile {
    pub(super) fn from_document(document: &KernDocument) -> Self {
        let mut profile = Self {
            has_title: document.metadata.title.is_some(),
            has_composer: document.metadata.composer.is_some(),
            has_movement_designation: document.metadata.movement_designation.is_some(),
            ..Self::default()
        };

        for line in document.lines() {
            profile.record_line(line);
        }

        profile
    }

    fn record_line(&mut self, line: &KernLine) {
        match line {
            KernLine::TandemInterpretationLine(tandems) => {
                for tandem in tandems.tokens() {
                    self.record_tandem(tandem);
                }
            }
            KernLine::LocalCommentLine(comments) => {
                if comments
                    .iter()
                    .any(|comment| matches!(comment, LocalComment::LayoutText(_)))
                {
                    self.has_layout_text = true;
                }
            }
            KernLine::Data(contents) => {
                for content in contents {
                    self.record_spine_content(content);
                }
            }
            KernLine::SpineTypeLine(_)
            | KernLine::SpineManipulatorLine(_)
            | KernLine::Barline(_) => {}
        }
    }

    fn record_tandem(&mut self, tandem: &TandemInterpretation) {
        match (StateKey::of(tandem), tandem) {
            (Some(StateKey::Pedal), _) => self.has_pedal = true,
            (Some(StateKey::Ottava), _) => self.has_ottava = true,
            (_, TandemInterpretation::Tempo(_)) => self.has_tempo = true,
            (_, TandemInterpretation::Instrument(_)) => self.has_staff_names = true,
            _ => {}
        }
    }

    fn record_spine_content(&mut self, content: &SpineContent) {
        match content {
            SpineContent::Kern(kern) => self.record_kern_content(kern),
            SpineContent::Dynam(DynamContent::Marks(marks)) => {
                self.has_dynamic_levels |= marks.level.is_some();
                self.has_hairpins |= !marks.crescendo.is_empty() || !marks.diminuendo.is_empty();
            }
            SpineContent::Dynam(DynamContent::Null | DynamContent::Raw { .. })
            | SpineContent::Fing(_)
            | SpineContent::Text(_)
            | SpineContent::MxmlHarmony(_)
            | SpineContent::Opaque { .. } => {}
        }
    }

    fn record_kern_content(&mut self, content: &KernContent) {
        match content {
            KernContent::Note(_) | KernContent::Null | KernContent::MultiMeasureRest(_) => {}
            KernContent::InvisibleRestSlice(slice) => {
                if slice.fermata.is_some() {
                    self.fermata_count += 1;
                }
            }
            KernContent::Rest(rest) => {
                if rest.fermata.is_some() {
                    self.fermata_count += 1;
                }
            }
            KernContent::Chord(chord) => {
                self.articulation_count += chord.modifiers.articulations.len();
                if chord.modifiers.fermata.is_some() {
                    self.fermata_count += 1;
                }
                if chord.modifiers.arpeggio.is_some() {
                    self.arpeggio_count += 1;
                }
                self.mordent_count += count_mordents(&chord.modifiers.ornaments);
            }
            KernContent::Grace(grace) => {
                self.articulation_count += grace.modifiers.articulations.len();
                if grace.modifiers.fermata.is_some() {
                    self.fermata_count += 1;
                }
                if grace.modifiers.arpeggio.is_some() {
                    self.arpeggio_count += 1;
                }
                self.mordent_count += count_mordents(&grace.modifiers.ornaments);
            }
        }
    }
}

fn count_mordents(ornaments: &[Placed<Ornament>]) -> usize {
    ornaments
        .iter()
        .filter(|ornament| {
            matches!(
                ornament.mark,
                Ornament::Mordent {
                    direction: crate::ast::MordentDirection::Upper,
                    interval: Some(AuxInterval::WholeTone),
                    doubled: false,
                }
            )
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(input: &str) -> RenderFeatureProfile {
        RenderFeatureProfile::from_document(&KernDocument::parse(input).unwrap())
    }

    #[test]
    fn records_pedal_ottava_tempo_and_instrument_tandems() {
        let found = profile("**kern\n*Ipiano\n*MM96\n*ped\n*8va\n4c\n*X8va\n*Xped\n*-\n");
        assert!(found.has_pedal && found.has_ottava && found.has_tempo && found.has_staff_names);

        let none = profile("**kern\n*clefG2\n*tremolo\n*rep\n4c\n*Xrep\n*Xtremolo\n*-\n");
        assert!(!(none.has_pedal || none.has_ottava || none.has_tempo || none.has_staff_names));
    }
}
