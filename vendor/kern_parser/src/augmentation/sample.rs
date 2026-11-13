use crate::{
    Articulation, DecimalTempo, DynamicHairpinDirection, DynamicLevel, DynamicLevelValue,
    LayoutText, OttavaKind, Tempo, TextPlacement,
};

use super::config::{
    RenderAugmentationConfig, RenderAugmentationSeed, SpanAugmentationConfig,
    TupletVisibilityAugmentationConfig,
};
use super::event::AugmentationEvent;
use super::plan::{AugmentationPlan, MusicSite};
use super::profile::RenderFeatureProfile;

pub(super) fn sample_events(
    plan: &AugmentationPlan,
    config: &RenderAugmentationConfig,
    profile: &RenderFeatureProfile,
    rng: &mut SmallRng,
) -> Vec<AugmentationEvent> {
    let mut events = Vec::new();

    if config.articulations.enabled {
        let mut count = 0;
        let max_count =
            remaining_budget(config.articulations.max_count, profile.articulation_count);
        for site in &plan.articulation_sites {
            if count >= max_count {
                break;
            }
            if rng.chance(config.articulations.probability_per_candidate) {
                events.push(AugmentationEvent::Articulation {
                    site: *site,
                    articulation: random_articulation(rng),
                });
                count += 1;
            }
        }
    }

    if config.fermatas.enabled && rng.chance(config.fermata_page_probability) {
        let mut count = 0;
        let max_count = remaining_budget(config.fermatas.max_count, profile.fermata_count);
        for site in rotated_sites(&plan.fermata_sites, rng) {
            if count >= max_count {
                break;
            }
            if rng.chance(config.fermatas.probability_per_candidate) {
                events.push(AugmentationEvent::Fermata { site: *site });
                count += 1;
            }
        }
    }

    if config.arpeggios.enabled
        && !plan.arpeggio_sites.is_empty()
        && remaining_budget(config.arpeggios.max_count, profile.arpeggio_count) > 0
        && rng.chance(config.arpeggio_page_probability)
    {
        let mut count = 0;
        let max_count = remaining_budget(config.arpeggios.max_count, profile.arpeggio_count);
        for site in rotated_sites(&plan.arpeggio_sites, rng) {
            if count >= max_count {
                break;
            }
            if rng.chance(config.arpeggios.probability_per_candidate) {
                events.push(AugmentationEvent::Arpeggio { site: *site });
                count += 1;
            }
        }
    }

    if config.dynamics.enabled && !profile.has_dynamic_levels {
        let mut count = 0;
        for site in &plan.music_sites {
            if count >= config.dynamics.max_count {
                break;
            }
            if rng.chance(config.dynamics.probability_per_candidate) {
                events.push(AugmentationEvent::DynamicLevel {
                    site: *site,
                    level: random_dynamic_level(rng),
                });
                count += 1;
            }
        }
    }

    if config.hairpins.enabled && !profile.has_hairpins {
        sample_span_events(
            plan,
            &config.hairpins,
            rng,
            |start, end, rng| AugmentationEvent::Hairpin {
                start,
                end,
                direction: if rng.next_usize(2) == 0 {
                    DynamicHairpinDirection::Crescendo
                } else {
                    DynamicHairpinDirection::Diminuendo
                },
            },
            &mut events,
        );
    }

    if config.pedal.enabled && !profile.has_pedal {
        sample_span_events(
            plan,
            &config.pedal,
            rng,
            |start, end, _| AugmentationEvent::Pedal { start, end },
            &mut events,
        );
    }

    if config.ottavas.enabled && !profile.has_ottava {
        sample_ottava_events(plan, &config.ottavas, rng, &mut events);
    }

    if config.tuplet_visibility.enabled {
        sample_tuplet_visibility_events(plan, &config.tuplet_visibility, rng, &mut events);
    }

    if config.reference_metadata.enabled
        && !(profile.has_title && profile.has_composer && profile.has_movement_designation)
    {
        sample_reference_metadata_events(config, rng, &mut events);
    }

    if config.staff_names.enabled
        && config.staff_names.max_count > 0
        && !profile.has_staff_names
        && rng.chance(config.staff_names.probability_per_candidate)
    {
        events.push(AugmentationEvent::StaffNames {
            names: random_staff_names(plan, rng),
        });
    }

    if config.metronome_tempos.enabled
        && config.metronome_tempos.max_count > 0
        && !profile.has_tempo
        && rng.chance(config.metronome_tempos.probability_per_candidate)
    {
        events.push(AugmentationEvent::MetronomeTempo {
            tempo: random_tempo(rng),
        });
    }

    if config.text_directions.enabled && !profile.has_layout_text {
        let mut count = 0;
        for site in &plan.music_sites {
            if count >= config.text_directions.max_count {
                break;
            }
            if rng.chance(config.text_directions.probability_per_candidate) {
                events.push(AugmentationEvent::TextDirection {
                    site: *site,
                    text: random_layout_text(rng),
                });
                count += 1;
            }
        }
    }

    if config.mordents.enabled {
        let mut count = 0;
        let max_count = remaining_budget(config.mordents.max_count, profile.mordent_count);
        for site in &plan.mordent_sites {
            if count >= max_count {
                break;
            }
            if rng.chance(config.mordents.probability_per_candidate) {
                events.push(AugmentationEvent::Mordent { site: *site });
                count += 1;
            }
        }
    }

    events
}

fn rotated_sites<'a>(
    sites: &'a [MusicSite],
    rng: &mut SmallRng,
) -> impl Iterator<Item = &'a MusicSite> {
    let offset = if sites.is_empty() {
        0
    } else {
        rng.next_usize(sites.len())
    };
    sites[offset..].iter().chain(&sites[..offset])
}

fn remaining_budget(max_count: usize, existing_count: usize) -> usize {
    max_count.saturating_sub(existing_count)
}

fn sample_reference_metadata_events(
    config: &RenderAugmentationConfig,
    rng: &mut SmallRng,
    events: &mut Vec<AugmentationEvent>,
) {
    let mut remaining = config.reference_metadata.max_count;
    let mut title = None;
    let mut composer = None;
    let mut movement_designation = None;

    if remaining > 0 && rng.chance(config.reference_metadata.probability_per_candidate) {
        title = Some(random_title(rng).to_owned());
        remaining -= 1;
    }
    if remaining > 0 && rng.chance(config.reference_metadata.probability_per_candidate) {
        composer = Some(random_composer(rng).to_owned());
        remaining -= 1;
    }
    if remaining > 0 && rng.chance(config.reference_metadata.probability_per_candidate) {
        movement_designation = Some(random_movement_designation(rng).to_owned());
    }

    if title.is_some() || composer.is_some() || movement_designation.is_some() {
        events.push(AugmentationEvent::ReferenceMetadata {
            title,
            composer,
            movement_designation,
        });
    }
}

fn sample_span_events<F>(
    plan: &AugmentationPlan,
    config: &SpanAugmentationConfig,
    rng: &mut SmallRng,
    make_event: F,
    events: &mut Vec<AugmentationEvent>,
) where
    F: Fn(MusicSite, MusicSite, &mut SmallRng) -> AugmentationEvent,
{
    let mut count = 0;
    for start in &plan.music_sites {
        if count >= config.max_count {
            break;
        }
        if !rng.chance(config.probability_per_candidate) {
            continue;
        }
        let choices = plan
            .music_sites
            .iter()
            .copied()
            .filter(|end| {
                end.staff_index == start.staff_index
                    && end.line_index >= start.line_index + config.min_line_span
                    && end.line_index <= start.line_index + config.max_line_span
            })
            .collect::<Vec<_>>();
        if choices.is_empty() {
            continue;
        }
        let end = choices[rng.next_usize(choices.len())];
        events.push(make_event(*start, end, rng));
        count += 1;
    }
}

fn sample_ottava_events(
    plan: &AugmentationPlan,
    config: &SpanAugmentationConfig,
    rng: &mut SmallRng,
    events: &mut Vec<AugmentationEvent>,
) {
    let mut count = 0;
    let mut spans = Vec::<(usize, usize, usize)>::new();
    for start in &plan.music_sites {
        if count >= config.max_count {
            break;
        }
        if !rng.chance(config.probability_per_candidate) {
            continue;
        }
        let choices = plan
            .music_sites
            .iter()
            .copied()
            .filter(|end| {
                end.staff_index == start.staff_index
                    && end.line_index >= start.line_index + config.min_line_span
                    && end.line_index <= start.line_index + config.max_line_span
                    && !spans.iter().any(|(staff, span_start, span_end)| {
                        *staff == start.staff_index
                            && ranges_overlap(
                                start.line_index,
                                end.line_index,
                                *span_start,
                                *span_end,
                            )
                    })
            })
            .collect::<Vec<_>>();
        if choices.is_empty() {
            continue;
        }
        let end = choices[rng.next_usize(choices.len())];
        let kind = if rng.next_usize(2) == 0 {
            OttavaKind::OctaveUp
        } else {
            OttavaKind::OctaveDown
        };
        events.push(AugmentationEvent::Ottava {
            start: *start,
            end,
            kind,
        });
        spans.push((start.staff_index, start.line_index, end.line_index));
        count += 1;
    }
}

fn sample_tuplet_visibility_events(
    plan: &AugmentationPlan,
    config: &TupletVisibilityAugmentationConfig,
    rng: &mut SmallRng,
    events: &mut Vec<AugmentationEvent>,
) {
    let mut count = 0;
    for run in &plan.tuplet_runs {
        if count >= config.max_count {
            break;
        }
        if !rng.chance(config.probability_per_candidate) {
            continue;
        }
        events.push(AugmentationEvent::TupletVisibility { run: *run });
        count += 1;
    }
}

fn ranges_overlap(
    left_start: usize,
    left_end: usize,
    right_start: usize,
    right_end: usize,
) -> bool {
    left_start <= right_end && right_start <= left_end
}

fn random_articulation(rng: &mut SmallRng) -> Articulation {
    match rng.next_usize(3) {
        0 => Articulation::Accent,
        1 => Articulation::Tenuto,
        _ => Articulation::Staccato,
    }
}

fn random_dynamic_level(rng: &mut SmallRng) -> DynamicLevel {
    let value = match rng.next_usize(5) {
        0 => DynamicLevelValue::Piano { count: 1 },
        1 => DynamicLevelValue::Piano { count: 2 },
        2 => DynamicLevelValue::MezzoPiano,
        3 => DynamicLevelValue::MezzoForte,
        _ => DynamicLevelValue::Forte { count: 1 },
    };
    DynamicLevel {
        value,
        subito: false,
        sforzando: false,
        accent: false,
        provenance: None,
    }
}

fn random_title(rng: &mut SmallRng) -> &'static str {
    const TITLES: &[&str] = &[
        "Prelude",
        "Nocturne",
        "Fugue",
        "Intermezzo",
        "Sonatina",
        "Serenade",
        "Caprice",
        "Arietta",
    ];
    TITLES[rng.next_usize(TITLES.len())]
}

fn random_composer(rng: &mut SmallRng) -> &'static str {
    const COMPOSERS: &[&str] = &[
        "Adler, Clara",
        "Bennett, Elias",
        "Carter, Miriam",
        "Fischer, Anton",
        "Moreau, Elise",
        "Novak, Simon",
        "Rossi, Lucia",
        "Weber, Julian",
    ];
    COMPOSERS[rng.next_usize(COMPOSERS.len())]
}

fn random_movement_designation(rng: &mut SmallRng) -> &'static str {
    const MOVEMENTS: &[&str] = &[
        "Allegro",
        "Andante",
        "Adagio",
        "Allegretto",
        "Largo",
        "Moderato",
        "Vivace",
        "Andante con moto",
    ];
    MOVEMENTS[rng.next_usize(MOVEMENTS.len())]
}

fn random_staff_names(plan: &AugmentationPlan, rng: &mut SmallRng) -> Vec<String> {
    const NAMES: &[&str] = &[
        "Piano", "Violin", "Cello", "Flute", "Clarinet", "Horn", "Viola", "Oboe",
    ];
    let staff_count = plan
        .music_sites
        .iter()
        .map(|site| site.staff_index)
        .max()
        .map_or(1, |index| index + 1);
    (0..staff_count)
        .map(|_| NAMES[rng.next_usize(NAMES.len())].to_owned())
        .collect()
}

fn random_tempo(rng: &mut SmallRng) -> Tempo {
    const TEMPOS: &[u32] = &[56, 66, 72, 84, 96, 108, 120, 132, 144];
    Tempo {
        beats_per_minute: DecimalTempo {
            units: TEMPOS[rng.next_usize(TEMPOS.len())],
            scale: 0,
        },
    }
}

fn random_layout_text(rng: &mut SmallRng) -> LayoutText {
    LayoutText {
        placement: TextPlacement::Above,
        italic: false,
        text: random_movement_designation(rng).to_owned(),
    }
}

#[derive(Debug, Clone)]
pub(super) struct SmallRng {
    state: u64,
}

impl SmallRng {
    pub(super) fn new(seed: RenderAugmentationSeed) -> Self {
        let mut state = seed.0;
        if state == 0 {
            state = RenderAugmentationSeed::default().0;
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }

    fn chance(&mut self, probability: f64) -> bool {
        if probability <= 0.0 {
            return false;
        }
        if probability >= 1.0 {
            return true;
        }
        let value = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        value < probability
    }
}
