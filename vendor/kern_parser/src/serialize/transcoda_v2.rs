//! Explicit staged serialization for the Transcoda V2 training dialect.

use std::collections::BTreeMap;

use crate::{
    analysis::{
        accidental_display::{canonicalize_accidental_display, DisplayChangeKind},
        measure_renders_empty, measure_spans, semantic_events, SemanticEventError,
    },
    document::{Document, Framing, InputOptions},
    normalization::{
        emitted_content_projection::try_project_transcoda_v2_content,
        invisible_rest_compaction::{compact_consecutive_invisible_rests_omitting, Omitted},
        invisible_rest_measures::RemoveInvisibleRestMeasures,
        invisible_rest_slices::{
            rehydrate_durationless_invisible_rest_slices_if_present, InvisibleRestSliceError,
        },
        omr::{try_canonicalize_transcoda_v2_document, OmrNormalizationError},
        NormalizationPass, NormalizeOptions,
    },
    score::Score,
    types::{SpineContent, SpineType},
    KernDocument, KernLine, ParseDialect,
};

use super::{
    omr::{BarlineEmissionPolicy, NullDataTokenStyle},
    ContentKind, ContentPolicy, DialectError, SerializeOptions,
};

/// A checked rich document projected onto the Transcoda V2 content domain.
///
/// Its fields are private so callers cannot bypass lowering:
///
/// ```compile_fail
/// use kern_parser::{serialize::transcoda_v2::ProjectedTranscodaV2, KernDocument};
/// let document = KernDocument::parse("**kern\n4c\n*-").unwrap();
/// let _ = ProjectedTranscodaV2 { document, policy: Default::default() };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectedTranscodaV2 {
    document: KernDocument,
    policy: ContentPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalTranscodaV2 {
    // Only checked lowering and canonicalization can construct this model.
    document: KernDocument,
    policy: ContentPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodaV2Syntax {
    PerSpineBarlinesDotNulls,
    PerSpineBarlinesEmptyNulls,
    CompactBarlinesDotNulls,
    CompactBarlinesEmptyNulls,
}

impl TranscodaV2Syntax {
    fn policies(self) -> (BarlineEmissionPolicy, NullDataTokenStyle) {
        match self {
            Self::PerSpineBarlinesDotNulls => (
                BarlineEmissionPolicy::PerActiveSpine,
                NullDataTokenStyle::Dot,
            ),
            Self::PerSpineBarlinesEmptyNulls => (
                BarlineEmissionPolicy::PerActiveSpine,
                NullDataTokenStyle::Empty,
            ),
            Self::CompactBarlinesDotNulls => {
                (BarlineEmissionPolicy::Compact, NullDataTokenStyle::Dot)
            }
            Self::CompactBarlinesEmptyNulls => {
                (BarlineEmissionPolicy::Compact, NullDataTokenStyle::Empty)
            }
        }
    }
}

pub fn try_lower_transcoda_v2(
    document: &KernDocument,
    policy: ContentPolicy,
) -> Result<ProjectedTranscodaV2, InvisibleRestSliceError> {
    let score = Score::resolve(document, Framing::Excerpt).map_err(DialectError::from)?;
    policy.tandem_policy().check(&score)?;
    let topology = score.topology();
    if let Some((line, column)) = topology
        .track_ids()
        .filter_map(|track| topology.joined_at(track))
        .min()
    {
        return Err(InvisibleRestSliceError::UnopenedSplitMerge { line, column });
    }
    let mut document = document.clone();
    rehydrate_durationless_invisible_rest_slices_if_present(&mut document)?;
    try_project_transcoda_v2_content(&mut document, policy).map_err(|error| {
        InvisibleRestSliceError::TranscodaV2Lowering {
            message: error.to_string(),
        }
    })?;
    crate::normalization::omr::RewriteNonSerializableTandems::new(
        NormalizeOptions::default(),
        policy.tandem_policy(),
    )
    .normalize(&mut document);
    if policy.drops(ContentKind::InvisibleRestMeasure) {
        RemoveInvisibleRestMeasures::new(NormalizeOptions::default()).normalize(&mut document);
    }
    validate_projected_invariant(&document, policy)?;
    Ok(ProjectedTranscodaV2 { document, policy })
}

pub fn canonicalize_transcoda_v2(
    mut projected: ProjectedTranscodaV2,
) -> Result<CanonicalTranscodaV2, InvisibleRestSliceError> {
    let policy = projected.policy;
    let before = document_events(&projected.document)?;
    try_canonicalize_transcoda_v2_document(&mut projected.document, policy).map_err(|error| {
        match error {
            OmrNormalizationError::DidNotConverge { iterations } => {
                InvisibleRestSliceError::TranscodaV2DidNotConverge { iterations }
            }
        }
    })?;
    rehydrate_durationless_invisible_rest_slices_if_present(&mut projected.document)?;
    validate_projected_invariant(&projected.document, policy)?;

    // Preserve the historical post-normalization boundary while making it an
    // explicit canonicalization responsibility.
    let score =
        Score::resolve(&projected.document, Framing::Excerpt).map_err(DialectError::from)?;
    if let Some(timeline_break) = crate::analysis::first_timeline_break(&score)? {
        return Err(timeline_break.into());
    }
    if policy.rewrites(ContentKind::InvisibleRestRun) {
        compact_consecutive_invisible_rests_omitting(
            &mut projected.document,
            Omitted {
                tandems: policy.tandem_policy(),
                local_comments: true,
            },
        );
    }
    validate_projected_invariant(&projected.document, policy)?;
    let after = document_events(&projected.document)?;
    if before != after {
        return Err(InvisibleRestSliceError::TranscodaV2SemanticEvents {
            message: first_event_difference(&before, &after),
        });
    }

    Ok(CanonicalTranscodaV2 {
        document: projected.document,
        policy,
    })
}

pub fn print_transcoda_v2(
    canonical: &CanonicalTranscodaV2,
    syntax: TranscodaV2Syntax,
    options: SerializeOptions,
) -> String {
    let (barline_policy, null_style) = syntax.policies();
    let options = crate::serialize::omr::transcoda_v2_options(
        canonical.policy,
        options,
        barline_policy,
        null_style,
    );
    let score = Score::resolve(&canonical.document, Framing::Excerpt)
        .expect("canonicalization checked that the document resolves");
    crate::serialize::omr::OmrSerializer::new(options).serialize_prepared_document(&score)
}

pub(crate) fn try_serialize_transcoda_v2(
    document: &KernDocument,
    syntax: TranscodaV2Syntax,
    policy: ContentPolicy,
    options: SerializeOptions,
) -> Result<String, InvisibleRestSliceError> {
    let mut canonical = canonicalize_transcoda_v2(try_lower_transcoda_v2(document, policy)?)?;
    // Canonicalization can leave a measure of only invisible rests that
    // lowering did not see, such as one whose merge line moved before its
    // barline. Lowering again removes it, so a label serializes to itself.
    for _ in 0..RELOWER_LIMIT {
        if !policy.drops(ContentKind::InvisibleRestMeasure)
            || !has_invisible_rest_measure(&canonical.document)
        {
            break;
        }
        canonical =
            canonicalize_transcoda_v2(try_lower_transcoda_v2(&canonical.document, policy)?)?;
    }
    let label = print_transcoda_v2(&canonical, syntax, options);
    Ok(if policy.rewrites(ContentKind::RedundantNatural) {
        strip_redundant_naturals(&label).unwrap_or(label)
    } else {
        label
    })
}

const RELOWER_LIMIT: usize = 4;

fn has_invisible_rest_measure(document: &KernDocument) -> bool {
    measure_spans(document)
        .iter()
        .any(|span| measure_renders_empty(document, span))
}

/// Drop every `n` whose natural Verovio draws anyway; `None` leaves the
/// label as printed. The accidental display analysis decides, on the label
/// itself, so a stripped label strips nothing more.
fn strip_redundant_naturals(label: &str) -> Option<String> {
    let options = InputOptions {
        dialect: ParseDialect::TranscodaV2,
        framing: Framing::Excerpt,
    };
    let document = Document::parse(label, options).ok()?;
    let canonical = canonicalize_accidental_display(&document).ok()?;
    let strips: BTreeMap<_, _> = canonical
        .changes
        .iter()
        .filter(|change| change.kind == DisplayChangeKind::StripN)
        .map(|change| {
            let at = change.source_address;
            ((at.line, at.field, at.subtoken), change)
        })
        .collect();
    if strips.is_empty() {
        return None;
    }
    let mut patched = 0;
    let lines: Vec<String> = label
        .split('\n')
        .enumerate()
        .map(|(index, line)| {
            line.split('\t')
                .enumerate()
                .map(|(field, token)| {
                    token
                        .split(' ')
                        .enumerate()
                        .map(|(subtoken, note)| {
                            match strips.get(&(index + 1, field + 1, subtoken + 1)) {
                                Some(change) if change.before == note => {
                                    patched += 1;
                                    change.after.as_str()
                                }
                                _ => note,
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect::<Vec<_>>()
                .join("\t")
        })
        .collect();
    (patched == strips.len()).then(|| lines.join("\n"))
}

/// Fail when any line still holds content `policy` would project away.
fn validate_projected_invariant(
    document: &KernDocument,
    policy: ContentPolicy,
) -> Result<(), InvisibleRestSliceError> {
    let keeps_non_kern = !policy.drops(ContentKind::NonKernSpine);
    for (line_index, line) in document.lines().iter().enumerate() {
        let source_line = document.source_line_number(line_index);
        let invalid = match line {
            KernLine::SpineTypeLine(spines) => spines
                .iter()
                .any(|spine| !keeps_non_kern && !matches!(spine, SpineType::Kern)),
            KernLine::TandemInterpretationLine(tandems) => tandems
                .tokens()
                .iter()
                .any(|tandem| !policy.tandem_policy().writes(tandem)),
            KernLine::LocalCommentLine(comments) => comments
                .iter()
                .any(|comment| !policy.keeps_local_comment(comment)),
            KernLine::Data(contents) => contents.iter().any(|content| match content {
                SpineContent::Kern(content) => !policy.keeps_content(content),
                _ => !keeps_non_kern,
            }),
            KernLine::SpineManipulatorLine(_) | KernLine::Barline(_) => false,
        };
        if invalid {
            return Err(InvisibleRestSliceError::TranscodaV2Invariant {
                message: format!("line {source_line} retains non-V2 content: {line:?}"),
            });
        }
    }
    Ok(())
}

fn document_events(
    document: &KernDocument,
) -> Result<Vec<crate::analysis::SemanticEvent>, InvisibleRestSliceError> {
    let score = Score::resolve(document, Framing::Excerpt).map_err(|error| {
        InvisibleRestSliceError::TranscodaV2SemanticEvents {
            message: error.to_string(),
        }
    })?;
    semantic_events(&score).map_err(|SemanticEventError::Duration(source)| {
        InvisibleRestSliceError::DataLineDurations(source)
    })
}

fn first_event_difference(
    before: &[crate::analysis::SemanticEvent],
    after: &[crate::analysis::SemanticEvent],
) -> String {
    let index = before
        .iter()
        .zip(after)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| before.len().min(after.len()));
    format!(
        "event {index} differs: before={:?}, after={:?}",
        before.get(index),
        after.get(index)
    )
}

#[cfg(test)]
mod tests {
    use crate::{serialize::SerializePreset, KernDocument};

    fn label(source: &str) -> String {
        SerializePreset::TranscodaV2Training
            .try_serialize_document(&KernDocument::parse(source).unwrap())
            .unwrap()
    }

    fn relabel(label: &str) -> String {
        let options = crate::document::InputOptions {
            dialect: crate::ParseDialect::TranscodaV2,
            framing: crate::document::Framing::Excerpt,
        };
        let document = crate::document::Document::parse(label, options).unwrap();
        SerializePreset::TranscodaV2Training
            .try_serialize_document(document.score())
            .unwrap()
    }

    fn serialize_label(label: &str) -> Result<String, String> {
        let options = crate::ParseOptions {
            dialect: crate::ParseDialect::TranscodaV2,
        };
        SerializePreset::TranscodaV2Training
            .try_serialize_document(&KernDocument::parse_with_options(label, options).unwrap())
            .map_err(|error| error.to_string())
    }

    #[test]
    fn a_page_starting_inside_a_split_is_rejected_as_the_renderer_rejects_it() {
        let page = "*clefF4\t*clefG2\t*clefG2\n*M2/4\t*M2/4\t*M2/4\n4C\t4c\t4e\n\
                    *\t*v\t*v\n4D\t4d\n=\n";
        assert_eq!(
            serialize_label(page),
            Err(
                "line 4 column 3: this merge closes a split the page never opens; \
                 a label renders the split's columns as separate spines, which cannot merge"
                    .to_owned()
            )
        );
        assert_eq!(
            crate::serialize::render::to_string(&KernDocument::parse(page).unwrap())
                .unwrap_err()
                .to_string(),
            "the score does not resolve: line 4 column 3: \
             cannot merge spines from different original spines"
        );
    }

    #[test]
    fn a_note_carried_from_the_previous_page_is_kept() {
        let page = "*clefF4\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n\t2c\n2C\t2e\n=\n1D\t1f\n=\n";
        assert_eq!(serialize_label(page), Ok(page.to_owned()));
    }

    #[test]
    fn a_spine_silent_until_the_barline_is_kept() {
        let page = "*clefF4\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n*\t*^\n1C\t2c\t4e\n\t\t4f\n\
                    \t2d\t\n*\t*v\t*v\n=\n";
        assert_eq!(serialize_label(page), Ok(page.to_owned()));
    }

    #[test]
    fn a_token_sounding_past_its_barline_is_rejected() {
        let page = "*clefF4\t*clefG2\n*M2/4\t*M2/4\n2C\t4c\n\t2d\n=\n2E\t\n=\n";
        assert_eq!(
            serialize_label(page),
            Err("line 5, column 2: a token still sounds 1/4 past this barline".to_owned())
        );
    }

    #[test]
    fn spellings_verovio_draws_alike_get_one_label() {
        let page = |body: &str| format!("**kern\n*clefG2\n*k[b-e-a-]\n*M4/4\n{body}\n=\n*-\n");
        for (left, right) in [
            (
                "16cLL\n16dJJ\n16eLL\n16fJJ\n4g\n2a",
                "16cL\n16dJ\n16eL\n16fJ\n4g\n2a",
            ),
            ("8qcL\n8qdJ\n4e\n4f\n2g", "8qqcL\n8qqdJ\n4e\n4f\n2g"),
            ("8en\n8e\n4f\n2g", "8e\n8e\n4f\n2g"),
            ("4c#\n4cn\n2d", "4c#\n4c\n2d"),
        ] {
            assert_eq!(
                label(&page(left)),
                label(&page(right)),
                "{left:?} vs {right:?}"
            );
        }
    }

    #[test]
    fn a_courtesy_natural_keeps_its_sign() {
        let page = "**kern\n*clefG2\n*k[]\n*M4/4\n4cn\n4d\n2e\n=\n*-\n";
        assert!(label(page).contains("4cn\n"));
    }

    #[test]
    fn invisible_rests_merge_in_one_pass() {
        let page = "**kern\t**kern\t**kern\n*M2/8\t*M2/8\t*M2/8\n8ryy\t8c\t16g\n.\t.\t16g\n\
                    8ryy\t8d\t16ryy\n.\t.\t16ryy\n=\t=\t=\n*-\t*-\t*-\n";
        let once = label(page);
        assert_eq!(relabel(&once), once);
    }

    #[test]
    fn a_chord_of_invisible_notes_keeps_its_time_in_the_label() {
        let page = "**kern\t**kern\n*clefF4\t*clefG2\n*M3/4\t*M3/4\n2.C\t4c\n\
                    .\t4b-yy bb-yy\n.\t4r\n=\t=\n*-\t*-\n";
        let once = label(page);
        assert_eq!(
            once,
            "*clefF4\t*clefG2\n*k[]\t*k[]\n*M3/4\t*M3/4\n2.C\t4c\n\t4ryy\n\t4r\n=\n"
        );
        assert_eq!(relabel(&once), once);
    }

    #[test]
    fn a_measure_left_invisible_by_a_moved_merge_is_removed() {
        let page = "**kern\t**kern\n*M3/4\t*M3/4\n*^\t*^\n2.c\t2.e\t2.g\t2.b\n*v\t*v\t*\t*\n\
                    =\t=\t=\n*\t*v\t*v\n2.ryy\t2.ryy\n=\t=\n*-\t*-\n";
        let once = label(page);
        assert!(!once.contains("ryy"), "{once}");
        assert_eq!(relabel(&once), once);
    }
}
