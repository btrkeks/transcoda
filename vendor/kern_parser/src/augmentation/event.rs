use crate::{Articulation, DynamicHairpinDirection, DynamicLevel, LayoutText, OttavaKind, Tempo};

use super::plan::{MusicSite, TupletRun};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum AugmentationEvent {
    Articulation {
        site: MusicSite,
        articulation: Articulation,
    },
    Fermata {
        site: MusicSite,
    },
    Arpeggio {
        site: MusicSite,
    },
    Mordent {
        site: MusicSite,
    },
    DynamicLevel {
        site: MusicSite,
        level: DynamicLevel,
    },
    Hairpin {
        start: MusicSite,
        end: MusicSite,
        direction: DynamicHairpinDirection,
    },
    Pedal {
        start: MusicSite,
        end: MusicSite,
    },
    Ottava {
        start: MusicSite,
        end: MusicSite,
        kind: OttavaKind,
    },
    TupletVisibility {
        run: TupletRun,
    },
    ReferenceMetadata {
        title: Option<String>,
        composer: Option<String>,
        movement_designation: Option<String>,
    },
    StaffNames {
        names: Vec<String>,
    },
    MetronomeTempo {
        tempo: Tempo,
    },
    TextDirection {
        site: MusicSite,
        text: LayoutText,
    },
}
