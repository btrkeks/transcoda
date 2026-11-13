//! Render-only augmentation for typed Humdrum `**kern` documents.
//!
//! Supervised fermatas and arpeggios must also enter targets. The report stores
//! the recipe so target generation replays the same augmented document.

mod apply;
mod config;
mod event;
mod plan;
mod profile;
mod sample;

pub use config::{
    AugmentationError, NoteMarkAugmentationConfig, PointAugmentationConfig,
    RenderAugmentationConfig, RenderAugmentationSeed, SpanAugmentationConfig,
    TupletVisibilityAugmentationConfig,
};

use crate::KernDocument;
use serde::{Deserialize, Serialize};

use apply::apply_events;
use config::validate_config;
use event::AugmentationEvent;
use plan::plan_candidates;
use profile::RenderFeatureProfile;
use sample::{sample_events, SmallRng};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct RenderAugmentationReport {
    /// Absent only in legacy render receipts.
    #[serde(default)]
    pub config: Option<RenderAugmentationConfig>,
    pub seed: u64,
    #[serde(default)]
    pub injected_fermata_count: usize,
    #[serde(default)]
    pub injected_arpeggio_count: usize,
    pub tuplet_visibility_count: usize,
    pub tuplet_visibility_spans: Vec<TupletVisibilitySpanReport>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TupletVisibilitySpanReport {
    pub staff_index: usize,
    pub spine_index: usize,
    pub start_source_line: usize,
    pub end_source_line: usize,
}

pub fn augment_for_render(
    document: &KernDocument,
    config: &RenderAugmentationConfig,
    seed: RenderAugmentationSeed,
) -> Result<KernDocument, AugmentationError> {
    let (document, _) = augment_for_render_with_report(document, config, seed)?;
    Ok(document)
}

pub fn augment_for_render_with_report(
    document: &KernDocument,
    config: &RenderAugmentationConfig,
    seed: RenderAugmentationSeed,
) -> Result<(KernDocument, RenderAugmentationReport), AugmentationError> {
    validate_config(config)?;
    let profile = RenderFeatureProfile::from_document(document);
    let plan = plan_candidates(document, &profile)?;
    let mut rng = SmallRng::new(seed);
    let events = sample_events(&plan, config, &profile, &mut rng);
    let report = render_augmentation_report(document, config, seed, &events);
    let augmented = apply_events(document, &events)?;
    Ok((augmented, report))
}

fn render_augmentation_report(
    document: &KernDocument,
    config: &RenderAugmentationConfig,
    seed: RenderAugmentationSeed,
    events: &[AugmentationEvent],
) -> RenderAugmentationReport {
    let tuplet_visibility_spans = events
        .iter()
        .filter_map(|event| {
            let AugmentationEvent::TupletVisibility { run } = event else {
                return None;
            };
            Some(TupletVisibilitySpanReport {
                staff_index: run.staff_index,
                spine_index: run.spine_index,
                start_source_line: document.source_line_number(run.first_line_index),
                end_source_line: document.source_line_number(run.last_line_index),
            })
        })
        .collect::<Vec<_>>();

    RenderAugmentationReport {
        config: Some(config.clone()),
        seed: seed.0,
        injected_fermata_count: events
            .iter()
            .filter(|event| matches!(event, AugmentationEvent::Fermata { .. }))
            .count(),
        injected_arpeggio_count: events
            .iter()
            .filter(|event| matches!(event, AugmentationEvent::Arpeggio { .. }))
            .count(),
        tuplet_visibility_count: tuplet_visibility_spans.len(),
        tuplet_visibility_spans,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_enabled_config() -> RenderAugmentationConfig {
        RenderAugmentationConfig {
            fermata_page_probability: 1.0,
            arpeggio_page_probability: 1.0,
            articulations: NoteMarkAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 16,
            },
            fermatas: NoteMarkAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 16,
            },
            arpeggios: NoteMarkAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 16,
            },
            mordents: NoteMarkAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 16,
            },
            dynamics: PointAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 8,
            },
            hairpins: SpanAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                min_line_span: 1,
                max_line_span: 4,
                max_count: 4,
            },
            pedal: SpanAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                min_line_span: 1,
                max_line_span: 4,
                max_count: 2,
            },
            ottavas: SpanAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                min_line_span: 2,
                max_line_span: 4,
                max_count: 2,
            },
            tuplet_visibility: TupletVisibilityAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 2,
            },
            reference_metadata: PointAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 3,
            },
            staff_names: PointAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 1,
            },
            metronome_tempos: PointAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 1,
            },
            text_directions: PointAugmentationConfig {
                enabled: true,
                probability_per_candidate: 1.0,
                max_count: 2,
            },
        }
    }

    fn only_symbol_config() -> RenderAugmentationConfig {
        RenderAugmentationConfig {
            fermata_page_probability: 1.0,
            arpeggio_page_probability: 1.0,
            articulations: NoteMarkAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                max_count: 0,
            },
            fermatas: NoteMarkAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                max_count: 0,
            },
            arpeggios: NoteMarkAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                max_count: 0,
            },
            mordents: NoteMarkAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                max_count: 0,
            },
            dynamics: PointAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                max_count: 0,
            },
            hairpins: SpanAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                min_line_span: 1,
                max_line_span: 1,
                max_count: 0,
            },
            pedal: SpanAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                min_line_span: 1,
                max_line_span: 1,
                max_count: 0,
            },
            ottavas: SpanAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                min_line_span: 1,
                max_line_span: 1,
                max_count: 0,
            },
            tuplet_visibility: TupletVisibilityAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                max_count: 0,
            },
            reference_metadata: PointAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                max_count: 0,
            },
            staff_names: PointAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                max_count: 0,
            },
            metronome_tempos: PointAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                max_count: 0,
            },
            text_directions: PointAugmentationConfig {
                enabled: false,
                probability_per_candidate: 0.0,
                max_count: 0,
            },
        }
    }

    fn count_matches(haystack: &str, needle: &str) -> usize {
        haystack.matches(needle).count()
    }

    fn tuplet_visibility_config(max_count: usize) -> RenderAugmentationConfig {
        let mut config = only_symbol_config();
        config.tuplet_visibility = TupletVisibilityAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count,
        };
        config
    }

    #[test]
    fn same_seed_produces_same_document() {
        let document = KernDocument::parse("**kern\n4c\n4d\n4e\n*-\n").unwrap();
        let config = all_enabled_config();

        let left = augment_for_render(&document, &config, RenderAugmentationSeed(1234)).unwrap();
        let right = augment_for_render(&document, &config, RenderAugmentationSeed(1234)).unwrap();

        assert_eq!(left, right);
    }

    #[test]
    fn different_seed_can_produce_different_document() {
        let document = KernDocument::parse("**kern\n4c\n4d\n4e\n4f\n4g\n*-\n").unwrap();
        let config = all_enabled_config();

        let left = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let right = augment_for_render(&document, &config, RenderAugmentationSeed(2)).unwrap();

        assert_ne!(left, right);
    }

    #[test]
    fn adds_render_only_visual_markings() {
        let document = KernDocument::parse("**kern\n4c\n4d\n4e\n*-\n").unwrap();
        let original_omr = document.to_compact_omr_string();
        let config = all_enabled_config();

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(99)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert!(rendered.contains("**dynam"));
        assert!(rendered.contains("!!!COM:"));
        assert!(rendered.contains("!!!OTL:"));
        assert!(rendered.contains("!!!OMD:"));
        assert!(rendered.contains("*I\""));
        assert!(rendered.contains("*MM"));
        assert!(rendered.contains("!LO:TX:a:t="));
        assert!(rendered.contains("*ped"));
        assert!(rendered.contains("*Xped"));
        assert!(rendered.contains(';'));
        assert!(rendered.contains('*') && (rendered.contains("8va") || rendered.contains("8ba")));
        assert!(
            rendered.contains('^') || rendered.contains('~') || rendered.contains('\''),
            "{rendered}"
        );
        KernDocument::parse(&rendered).unwrap();
        assert_eq!(document.to_compact_omr_string(), original_omr);
    }

    #[test]
    fn handles_multiple_staves() {
        let document = KernDocument::parse("**kern\t**kern\n4c\t4e\n4d\t4f\n*-\t*-\n").unwrap();
        let config = all_enabled_config();

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(42)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert!(rendered.contains("**kern\t**kern\t**dynam\t**dynam\n"));
        KernDocument::parse(&rendered).unwrap();
    }

    #[test]
    fn adds_fermatas_to_notes_and_preserves_rests() {
        let document = KernDocument::parse("**kern\n4c\n4r\n*-\n").unwrap();
        let original_omr = document.to_compact_omr_string();
        let mut config = only_symbol_config();
        config.fermatas = NoteMarkAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 8,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert!(rendered.contains("4c;"), "{rendered}");
        assert!(rendered.contains("\n4r\n"), "{rendered}");
        KernDocument::parse(&rendered).unwrap();
        assert_eq!(document.to_compact_omr_string(), original_omr);
    }

    #[test]
    fn existing_articulation_uses_budget_and_site_is_not_augmented_again() {
        let document = KernDocument::parse("**kern\n4c'\n4d\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.articulations = NoteMarkAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 1,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(augmented, document);

        config.articulations.max_count = 8;
        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert!(rendered.contains("\n4c'\n"), "{rendered}");
        assert!(!rendered.contains("4c''"), "{rendered}");
        assert!(!rendered.contains("4c'~"), "{rendered}");
        assert!(!rendered.contains("4c'^"), "{rendered}");
    }

    #[test]
    fn existing_fermata_uses_budget_and_site_is_not_augmented_again() {
        let document = KernDocument::parse("**kern\n4c;\n4d\n4r;\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.fermatas = NoteMarkAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 1,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(augmented, document);

        config.fermatas.max_count = 8;
        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert!(rendered.contains("\n4c;\n"), "{rendered}");
        assert!(!rendered.contains("4c;;"), "{rendered}");
        assert!(rendered.contains("\n4r;\n"), "{rendered}");
        assert!(rendered.contains("\n4d;\n"), "{rendered}");
    }

    #[test]
    fn adds_arpeggios_only_to_multi_note_chords() {
        let document = KernDocument::parse("**kern\n4c\n4c 4e\n4r\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.arpeggios = NoteMarkAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 8,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert!(rendered.contains("4c 4e:"), "{rendered}");
        assert!(!rendered.contains("4c:"), "{rendered}");
        assert!(!rendered.contains("4r:"), "{rendered}");
        KernDocument::parse(&rendered).unwrap();
    }

    #[test]
    fn existing_arpeggio_count_reduces_synthetic_arpeggios() {
        let document = KernDocument::parse("**kern\n4c 4e:\n4d 4f\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.arpeggios = NoteMarkAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 1,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(augmented, document);
    }

    #[test]
    fn adds_mordents_to_pitched_content_only() {
        let document = KernDocument::parse("**kern\n4c\n4d 4f\n4r\n*-\n").unwrap();
        let original_omr = document.to_compact_omr_string();
        let mut config = only_symbol_config();
        config.mordents = NoteMarkAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 8,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert!(rendered.contains("\n4cM\n"), "{rendered}");
        assert!(rendered.contains("\n4d 4fM\n"), "{rendered}");
        assert!(rendered.contains("\n4r\n"), "{rendered}");
        assert!(!rendered.contains("4rM"), "{rendered}");
        KernDocument::parse(&rendered).unwrap();
        assert_eq!(document.to_compact_omr_string(), original_omr);
    }

    #[test]
    fn existing_mordent_uses_budget_and_existing_ornament_sites_are_skipped() {
        let document = KernDocument::parse("**kern\n4cM\n4dT\n4e\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.mordents = NoteMarkAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 1,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(augmented, document);

        config.mordents.max_count = 8;
        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert!(rendered.contains("\n4cM\n"), "{rendered}");
        assert!(!rendered.contains("4cMM"), "{rendered}");
        assert!(rendered.contains("\n4dT\n"), "{rendered}");
        assert!(!rendered.contains("4dTM"), "{rendered}");
        assert!(rendered.contains("\n4eM\n"), "{rendered}");
    }

    #[test]
    fn adds_paired_ottava_tandems() {
        let document = KernDocument::parse("**kern\n4c\n4d\n4e\n4f\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.ottavas = SpanAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            min_line_span: 2,
            max_line_span: 3,
            max_count: 1,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert!(
            (rendered.contains("*8va") && rendered.contains("*X8va"))
                || (rendered.contains("*8ba") && rendered.contains("*X8ba")),
            "{rendered}"
        );
        KernDocument::parse(&rendered).unwrap();
    }

    #[test]
    fn existing_pedal_suppresses_synthetic_pedal_spans() {
        let document = KernDocument::parse("**kern\n*ped\n4c\n4d\n*Xped\n4e\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.pedal = SpanAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            min_line_span: 1,
            max_line_span: 4,
            max_count: 2,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert_eq!(count_matches(&rendered, "*ped"), 1, "{rendered}");
        assert_eq!(count_matches(&rendered, "*Xped"), 1, "{rendered}");
        assert_eq!(augmented, document);
    }

    #[test]
    fn existing_ottava_suppresses_synthetic_ottava_spans() {
        let document = KernDocument::parse("**kern\n*8va\n4c\n4d\n*X8va\n4e\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.ottavas = SpanAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            min_line_span: 1,
            max_line_span: 4,
            max_count: 2,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert_eq!(count_matches(&rendered, "*8va"), 1, "{rendered}");
        assert_eq!(count_matches(&rendered, "*X8va"), 1, "{rendered}");
        assert_eq!(augmented, document);
    }

    #[test]
    fn existing_dynamic_level_suppresses_synthetic_dynamic_levels() {
        let document = KernDocument::parse("**kern\t**dynam\n4c\tp\n4d\t.\n*-\t*-\n").unwrap();
        let mut config = only_symbol_config();
        config.dynamics = PointAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 4,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(augmented, document);
    }

    #[test]
    fn existing_hairpin_suppresses_synthetic_hairpins() {
        let document =
            KernDocument::parse("**kern\t**dynam\n4c\t<\n4d\t(\n4e\t[\n*-\t*-\n").unwrap();
        let mut config = only_symbol_config();
        config.hairpins = SpanAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            min_line_span: 1,
            max_line_span: 4,
            max_count: 2,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(augmented, document);
    }

    #[test]
    fn existing_layout_text_suppresses_synthetic_text_directions() {
        let document = KernDocument::parse("**kern\n!LO:TX:a:t=dolce\n4c\n4d\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.text_directions = PointAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 2,
        };

        let augmented = augment_for_render(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert_eq!(count_matches(&rendered, "!LO:TX:a:t="), 1, "{rendered}");
        assert_eq!(augmented, document);
    }

    #[test]
    fn adds_tuplet_visibility_once_for_a_complete_tupletted_run() {
        let document = KernDocument::parse("**kern\n12d\n12e\n12f\n4g\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.tuplet_visibility = TupletVisibilityAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 2,
        };

        let (augmented, report) =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(1)).unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert_eq!(report.tuplet_visibility_count, 1, "{report:?}");
        assert_eq!(
            report.tuplet_visibility_spans,
            vec![TupletVisibilitySpanReport {
                staff_index: 0,
                spine_index: 0,
                start_source_line: 2,
                end_source_line: 4,
            }]
        );
        assert_eq!(count_matches(&rendered, "*Xtuplet"), 1, "{rendered}");
        assert_eq!(count_matches(&rendered, "*tuplet"), 1, "{rendered}");
        assert!(rendered.contains("*Xtuplet\n12d"), "{rendered}");
        assert!(rendered.contains("12f\n*tuplet"), "{rendered}");
        KernDocument::parse(&rendered).unwrap();

        let plain = KernDocument::parse("**kern\n4d\n4e\n4f\n*-\n").unwrap();
        let plain_augmented =
            augment_for_render(&plain, &config, RenderAugmentationSeed(1)).unwrap();
        let plain_rendered = plain_augmented.to_render_string().unwrap();
        assert!(!plain_rendered.contains("*Xtuplet"), "{plain_rendered}");
        assert!(!plain_rendered.contains("*tuplet"), "{plain_rendered}");
    }

    #[test]
    fn reports_tuplet_visibility_spans() {
        let document = KernDocument::parse("**kern\n12d\n12e\n12f\n4g\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.tuplet_visibility = TupletVisibilityAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 1,
        };

        let (_, report) =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(report.seed, 1);
        assert_eq!(report.tuplet_visibility_count, 1);
        assert_eq!(report.tuplet_visibility_spans.len(), 1);
        let span = &report.tuplet_visibility_spans[0];
        assert_eq!(span.staff_index, 0);
        assert_eq!(span.spine_index, 0);
        assert_eq!(span.start_source_line, 2);
        assert_eq!(span.end_source_line, 4);

        let plain = KernDocument::parse("**kern\n4d\n4e\n4f\n*-\n").unwrap();
        let (_, plain_report) =
            augment_for_render_with_report(&plain, &config, RenderAugmentationSeed(1)).unwrap();
        assert_eq!(plain_report.tuplet_visibility_count, 0);
        assert!(plain_report.tuplet_visibility_spans.is_empty());
    }

    #[test]
    fn skips_tuplet_visibility_when_explicit_tuplet_tandem_exists() {
        let document =
            KernDocument::parse("**kern\n*Xtuplet\n12d\n12e\n12f\n*tuplet\n4g\n*-\n").unwrap();
        let mut config = only_symbol_config();
        config.tuplet_visibility = TupletVisibilityAugmentationConfig {
            enabled: true,
            probability_per_candidate: 1.0,
            max_count: 1,
        };

        let (augmented, report) =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(augmented, document);
        assert_eq!(report.tuplet_visibility_count, 0);
        assert!(report.tuplet_visibility_spans.is_empty());
    }

    #[test]
    fn tuple_visibility_runs_respect_content_and_document_boundaries() {
        let document = KernDocument::parse(
            "**kern\n12c\n! keep the run open\n*clefG2\n.\n12r\n4e\n12f\n12g\n=1\n12a\n12b\n*-\n",
        )
        .unwrap();
        let config = tuplet_visibility_config(8);

        let (_, report) =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(
            report.tuplet_visibility_spans,
            vec![
                TupletVisibilitySpanReport {
                    staff_index: 0,
                    spine_index: 0,
                    start_source_line: 2,
                    end_source_line: 6,
                },
                TupletVisibilitySpanReport {
                    staff_index: 0,
                    spine_index: 0,
                    start_source_line: 8,
                    end_source_line: 9,
                },
                TupletVisibilitySpanReport {
                    staff_index: 0,
                    spine_index: 0,
                    start_source_line: 11,
                    end_source_line: 12,
                },
            ]
        );
    }

    #[test]
    fn tuple_visibility_tracks_explicit_state_per_spine() {
        let document = KernDocument::parse(
            "**kern\t**kern\n*Xtuplet\t*\n12c\t12e\n12d\t.\n*tuplet\t*\n12e\t12f\n*-\t*-\n",
        )
        .unwrap();
        let config = tuplet_visibility_config(8);

        let (_, report) =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(
            report.tuplet_visibility_spans,
            vec![
                TupletVisibilitySpanReport {
                    staff_index: 1,
                    spine_index: 1,
                    start_source_line: 3,
                    end_source_line: 6,
                },
                TupletVisibilitySpanReport {
                    staff_index: 0,
                    spine_index: 0,
                    start_source_line: 6,
                    end_source_line: 6,
                },
            ]
        );
    }

    #[test]
    fn tuple_visibility_runs_stop_at_spine_manipulators_and_keep_staff_identity() {
        let document =
            KernDocument::parse("**kern\n12c\n*^\n12d\t12e\n4f\t4g\n*v\t*v\n12a\n*-\n").unwrap();
        let config = tuplet_visibility_config(8);

        let (_, report) =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(
            report.tuplet_visibility_spans,
            vec![
                TupletVisibilitySpanReport {
                    staff_index: 0,
                    spine_index: 0,
                    start_source_line: 2,
                    end_source_line: 2,
                },
                TupletVisibilitySpanReport {
                    staff_index: 0,
                    spine_index: 0,
                    start_source_line: 4,
                    end_source_line: 4,
                },
                TupletVisibilitySpanReport {
                    staff_index: 0,
                    spine_index: 1,
                    start_source_line: 4,
                    end_source_line: 4,
                },
                TupletVisibilitySpanReport {
                    staff_index: 0,
                    spine_index: 0,
                    start_source_line: 7,
                    end_source_line: 7,
                },
            ]
        );
    }

    #[test]
    fn tuple_visibility_conservatively_merges_authored_hidden_state() {
        let document = KernDocument::parse("**kern\n*^\n*Xtuplet\t*\n*v\t*v\n12c\n*-\n").unwrap();
        let config = tuplet_visibility_config(8);

        let (_, report) =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(report.tuplet_visibility_count, 0);
        assert!(report.tuplet_visibility_spans.is_empty());
    }

    #[test]
    fn tuple_visibility_handles_wrapperless_leading_spine_manipulator() {
        let parsed = KernDocument::parse("*^\n12c\t12d\n*-\t*-\n").unwrap();
        let source_lines = parsed
            .lines()
            .iter()
            .enumerate()
            .filter_map(|(index, line)| parsed.source(index).map(|source| (line.clone(), source)))
            .map(|(line, source)| (line, Some(source)))
            .collect();
        let document = KernDocument::from_parts(parsed.metadata, parsed.layout, source_lines);
        let config = tuplet_visibility_config(8);

        let (_, report) =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(1)).unwrap();

        assert_eq!(
            report.tuplet_visibility_spans,
            vec![
                TupletVisibilitySpanReport {
                    staff_index: 0,
                    spine_index: 0,
                    start_source_line: 2,
                    end_source_line: 2,
                },
                TupletVisibilitySpanReport {
                    staff_index: 0,
                    spine_index: 1,
                    start_source_line: 2,
                    end_source_line: 2,
                },
            ]
        );
    }

    #[test]
    fn tuple_visibility_sampling_honors_probability_budget_and_seed() {
        let document = KernDocument::parse("**kern\n12c\n4d\n12e\n=1\n12f\n*-\n").unwrap();
        let mut config = tuplet_visibility_config(2);

        let first =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(9)).unwrap();
        let second =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(9)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.1.tuplet_visibility_count, 2);

        config.tuplet_visibility.probability_per_candidate = 0.0;
        let (_, zero_probability) =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(9)).unwrap();
        assert_eq!(zero_probability.tuplet_visibility_count, 0);

        config.tuplet_visibility.probability_per_candidate = 1.0;
        config.tuplet_visibility.enabled = false;
        let (_, disabled) =
            augment_for_render_with_report(&document, &config, RenderAugmentationSeed(9)).unwrap();
        assert_eq!(disabled.tuplet_visibility_count, 0);
    }

    #[test]
    fn default_config_does_not_add_ottavas() {
        let document = KernDocument::parse("**kern\n4c\n4d\n4e\n4f\n4g\n4a\n*-\n").unwrap();

        let augmented = augment_for_render(
            &document,
            &RenderAugmentationConfig::default(),
            RenderAugmentationSeed(1),
        )
        .unwrap();
        let rendered = augmented.to_render_string().unwrap();

        assert!(!rendered.contains("*8va"), "{rendered}");
        assert!(!rendered.contains("*8ba"), "{rendered}");
        assert!(!rendered.contains("*X8va"), "{rendered}");
        assert!(!rendered.contains("*X8ba"), "{rendered}");
    }

    #[test]
    fn rejects_invalid_probability() {
        let mut config = RenderAugmentationConfig::default();
        config.dynamics.probability_per_candidate = 2.0;
        let document = KernDocument::parse("**kern\n4c\n*-\n").unwrap();

        assert!(matches!(
            augment_for_render(&document, &config, RenderAugmentationSeed(1)),
            Err(AugmentationError::InvalidConfig(_))
        ));
    }
}
