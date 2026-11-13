//! Spine-local compaction of consecutive invisible rests.
//!
//! Consecutive duration-bearing invisible rests in one `**kern` spine can be
//! represented as a single longer invisible rest followed by null continuations.
//! This keeps the source Humdrum shape simpler while preserving data-line timing
//! for other active spines.

use crate::{
    analysis::{data_line_durations, DataLineDuration},
    serialize::TandemPolicy,
    KernContent, KernDocument, KernLine, LocalComment, MusicalDuration, SpineContent,
};

use super::{
    invisible_rest_tokens::{
        is_kern_null, is_non_kern_data_null, plain_invisible_rest_duration,
        plain_invisible_rest_mut,
    },
    NormalizationPass,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactConsecutiveInvisibleRests {
    omitted: Omitted,
}

/// What the output leaves out, so it cannot interrupt an invisible-rest run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Omitted {
    /// Tandems the output's policy does not write.
    pub(crate) tandems: TandemPolicy,
    /// Whether the output drops local comments, as compacted OMR and Transcoda V2 output do.
    pub(crate) local_comments: bool,
}

impl Omitted {
    /// Transcoda V2 output.
    pub(crate) const TRANSCODA_V2: Self = Self {
        tandems: TandemPolicy::TranscodaV2,
        local_comments: true,
    };
}

impl Default for CompactConsecutiveInvisibleRests {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactConsecutiveInvisibleRests {
    /// Compaction for Transcoda V2 output.
    pub fn new() -> Self {
        Self::omitting(Omitted::TRANSCODA_V2)
    }

    pub(crate) fn omitting(omitted: Omitted) -> Self {
        Self { omitted }
    }
}

impl NormalizationPass for CompactConsecutiveInvisibleRests {
    fn normalize(&self, document: &mut KernDocument) {
        compact_consecutive_invisible_rests_omitting(document, self.omitted);
    }
}

/// Compaction for Transcoda V2 output.
pub fn compact_consecutive_invisible_rests(document: &mut KernDocument) {
    compact_consecutive_invisible_rests_omitting(document, Omitted::TRANSCODA_V2);
}

/// Merge consecutive plain invisible rests of one `**kern` spine. Tandem and
/// comment lines the output leaves out (`omitted`) do not interrupt a run;
/// anything it prints does.
pub(crate) fn compact_consecutive_invisible_rests_omitting(
    document: &mut KernDocument,
    omitted: Omitted,
) {
    // One round keeps only the compactions that are safe together, so a run
    // dropped for colliding with another can merge in the next round.
    for _ in 0..COMPACTION_ROUND_LIMIT {
        if !compact_round(document, omitted) {
            return;
        }
    }
}

const COMPACTION_ROUND_LIMIT: usize = 16;

/// One round of compaction; whether it changed the document.
fn compact_round(document: &mut KernDocument, omitted: Omitted) -> bool {
    let Ok(original_durations) = data_line_durations(document) else {
        return false;
    };

    let original_line_durations = line_durations_by_index(document, &original_durations);
    let compactions = retain_safe_compactions(
        document,
        &original_line_durations,
        find_compactions(document, &original_line_durations, omitted),
    );
    if compactions.is_empty() {
        return false;
    }

    let mut candidate = document.clone();
    apply_compactions(&mut candidate, &compactions);

    if data_line_durations(&candidate).is_err() || candidate == *document {
        return false;
    }
    *document = candidate;
    true
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InvisibleRestRun {
    column: usize,
    head_line: usize,
    tail_lines: Vec<usize>,
    total_duration: MusicalDuration,
    remaining: MusicalDuration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InvisibleRestCompaction {
    column: usize,
    head_line: usize,
    tail_lines: Vec<usize>,
    duration: crate::Duration,
}

fn find_compactions(
    document: &KernDocument,
    line_durations: &[MusicalDuration],
    omitted: Omitted,
) -> Vec<InvisibleRestCompaction> {
    let mut compactions = Vec::new();
    let mut active_runs = Vec::<Option<InvisibleRestRun>>::new();

    for (line_index, line) in document.lines().iter().enumerate() {
        let contents = match line {
            KernLine::Data(contents) => contents,
            // Local comments consume no time, and compacted OMR output drops
            // every kind of local comment. A dropped
            // comment line must not interrupt an invisible-rest run: the
            // emitted text has no trace of it, so the next parse/serialize
            // round trip would compact further and the output would drift.
            KernLine::LocalCommentLine(comments)
                if omitted.local_comments && !contains_emitted_local_comment(comments) =>
            {
                continue;
            }
            // Where the output prints local comments, only an all-null
            // comment line carries nothing to keep in place.
            KernLine::LocalCommentLine(comments)
                if comments
                    .iter()
                    .all(|comment| matches!(comment, LocalComment::Null)) =>
            {
                continue;
            }
            // Likewise, tandem lines made only of tokens that compacted OMR
            // output omits (pedal marks, section labels, ...) leave no trace
            // in the emitted text and consume no time.
            KernLine::TandemInterpretationLine(tandems)
                if tandems.tokens().iter().all(|tandem| {
                    matches!(tandem, crate::TandemInterpretation::Null)
                        || super::tandem_omitted_by(omitted.tandems, tandem)
                }) =>
            {
                continue;
            }
            _ => {
                finish_all_runs(&mut active_runs, &mut compactions);
                continue;
            }
        };

        if contents.len() != active_runs.len() {
            finish_all_runs(&mut active_runs, &mut compactions);
            active_runs.resize_with(contents.len(), || None);
        }

        for (column, content) in contents.iter().enumerate() {
            if let Some(duration) = plain_invisible_rest_duration(content) {
                observe_invisible_rest(
                    &mut active_runs[column],
                    column,
                    line_index,
                    duration,
                    &mut compactions,
                );
            } else if is_kern_null(content) {
                // A null cell on a zero-duration line (kept alive only by
                // grace notes or non-kern spine content) does not advance
                // time and must not end a completed run; compacted OMR
                // output may drop such lines entirely.
                if !line_durations[line_index].is_zero() {
                    observe_kern_null(&mut active_runs[column], &mut compactions);
                }
            } else {
                finish_run(&mut active_runs[column], &mut compactions);
            }
        }

        advance_runs(
            &mut active_runs,
            line_durations[line_index],
            &mut compactions,
        );
    }

    finish_all_runs(&mut active_runs, &mut compactions);
    compactions
}

fn line_durations_by_index(
    document: &KernDocument,
    data_line_durations: &[DataLineDuration],
) -> Vec<MusicalDuration> {
    let mut durations = vec![MusicalDuration::zero(); document.lines().len()];
    for data_line_duration in data_line_durations {
        durations[data_line_duration.line_index] = data_line_duration.duration;
    }
    durations
}

fn retain_safe_compactions(
    document: &KernDocument,
    original_line_durations: &[MusicalDuration],
    mut compactions: Vec<InvisibleRestCompaction>,
) -> Vec<InvisibleRestCompaction> {
    loop {
        let mut candidate = document.clone();
        apply_compactions(&mut candidate, &compactions);
        let Ok(candidate_durations) = data_line_durations(&candidate) else {
            return Vec::new();
        };
        let candidate_line_durations = line_durations_by_index(&candidate, &candidate_durations);

        let before = compactions.len();
        compactions.retain(|compaction| {
            compaction_tail_lines_are_safe(
                &candidate,
                original_line_durations,
                &candidate_line_durations,
                compaction,
            )
        });

        if compactions.len() == before {
            return compactions;
        }
    }
}

fn compaction_tail_lines_are_safe(
    candidate: &KernDocument,
    original_line_durations: &[MusicalDuration],
    candidate_line_durations: &[MusicalDuration],
    compaction: &InvisibleRestCompaction,
) -> bool {
    compaction.tail_lines.iter().all(|line_index| {
        is_null_data_line(candidate.lines().get(*line_index))
            || original_line_durations[*line_index] == candidate_line_durations[*line_index]
    })
}

/// Whether the compacted OMR serializer emits this local comment line.
///
/// Local comments are dropped from compacted output.
fn contains_emitted_local_comment(_comments: &[LocalComment]) -> bool {
    false
}

fn is_null_data_line(line: Option<&KernLine>) -> bool {
    let Some(KernLine::Data(contents)) = line else {
        return false;
    };
    !contents.is_empty()
        && contents
            .iter()
            .all(|content| is_kern_null(content) || is_non_kern_data_null(content))
}

fn observe_invisible_rest(
    run: &mut Option<InvisibleRestRun>,
    column: usize,
    line_index: usize,
    duration: MusicalDuration,
    compactions: &mut Vec<InvisibleRestCompaction>,
) {
    match run {
        Some(active_run) if active_run.remaining.is_zero() => {
            active_run.tail_lines.push(line_index);
            active_run.total_duration += duration;
            active_run.remaining = duration;
        }
        Some(_) => {
            finish_run(run, compactions);
            *run = Some(new_run(column, line_index, duration));
        }
        None => {
            *run = Some(new_run(column, line_index, duration));
        }
    }
}

fn observe_kern_null(
    run: &mut Option<InvisibleRestRun>,
    compactions: &mut Vec<InvisibleRestCompaction>,
) {
    if run
        .as_ref()
        .is_some_and(|active_run| active_run.remaining.is_zero())
    {
        finish_run(run, compactions);
    }
}

fn advance_runs(
    active_runs: &mut [Option<InvisibleRestRun>],
    line_duration: MusicalDuration,
    compactions: &mut Vec<InvisibleRestCompaction>,
) {
    if line_duration.is_zero() {
        return;
    }

    for run in active_runs {
        let Some(active_run) = run else {
            continue;
        };
        if active_run.remaining.is_zero() {
            continue;
        }
        let Some(remaining) = active_run.remaining.checked_sub(line_duration) else {
            finish_run(run, compactions);
            continue;
        };
        active_run.remaining = remaining;
    }
}

fn new_run(column: usize, line_index: usize, duration: MusicalDuration) -> InvisibleRestRun {
    InvisibleRestRun {
        column,
        head_line: line_index,
        tail_lines: Vec::new(),
        total_duration: duration,
        remaining: duration,
    }
}

fn finish_all_runs(
    active_runs: &mut [Option<InvisibleRestRun>],
    compactions: &mut Vec<InvisibleRestCompaction>,
) {
    for run in active_runs {
        finish_run(run, compactions);
    }
}

fn finish_run(run: &mut Option<InvisibleRestRun>, compactions: &mut Vec<InvisibleRestCompaction>) {
    let Some(run) = run.take() else {
        return;
    };
    if run.tail_lines.is_empty() {
        return;
    }
    let Some(duration) = run.total_duration.into_duration_prefer_dots() else {
        return;
    };
    compactions.push(InvisibleRestCompaction {
        column: run.column,
        head_line: run.head_line,
        tail_lines: run.tail_lines,
        duration,
    });
}

fn apply_compactions(document: &mut KernDocument, compactions: &[InvisibleRestCompaction]) {
    for compaction in compactions {
        let Some(KernLine::Data(contents)) = document.lines_mut().get_mut(compaction.head_line)
        else {
            continue;
        };
        let Some(content) = contents.get_mut(compaction.column) else {
            continue;
        };
        if let Some(rest) = plain_invisible_rest_mut(content) {
            rest.duration = compaction.duration.clone();
        }

        for line_index in &compaction.tail_lines {
            let Some(KernLine::Data(contents)) = document.lines_mut().get_mut(*line_index) else {
                continue;
            };
            let Some(content) = contents.get_mut(compaction.column) else {
                continue;
            };
            if plain_invisible_rest_mut(content).is_some() {
                *content = SpineContent::Kern(KernContent::Null);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    fn compacted(input: &str, omitted: Omitted) -> String {
        let mut document = KernDocument::parse(input).unwrap();
        CompactConsecutiveInvisibleRests::omitting(omitted).normalize(&mut document);
        document.to_humdrum_kern_string()
    }

    const OMR_DATASET: Omitted = Omitted {
        tandems: crate::serialize::TandemPolicy::Omr,
        local_comments: false,
    };

    #[test]
    fn a_pedal_interrupts_a_run_only_where_the_output_drops_it() {
        let input = "**kern\n*M4/4\n=1\n4c\n8ryy\n*ped\n8ryy\n4ryy\n4d\n=2\n*-\n";
        assert!(compacted(input, Omitted::TRANSCODA_V2).contains("2ryy\n*ped\n.\n.\n4d"));
        let kept = compacted(input, OMR_DATASET);
        assert!(kept.contains("8ryy\n*ped\n4.ryy\n.\n4d"), "{kept}");
    }

    #[test]
    fn a_printed_local_comment_interrupts_a_run() {
        let input = "**kern\n*M4/4\n=1\n4c\n8ryy\n!LO:TX:a:t=dolce\n8ryy\n4ryy\n4d\n=2\n*-\n";
        assert!(
            compacted(input, Omitted::TRANSCODA_V2).contains("2ryy\n!LO:TX:a:t=dolce\n.\n.\n4d")
        );
        let kept = compacted(input, OMR_DATASET);
        assert!(
            kept.contains("8ryy\n!LO:TX:a:t=dolce\n4.ryy\n.\n4d"),
            "{kept}"
        );
    }

    use super::*;
    use crate::{
        normalization::{null_lines::RemoveNullLines, NormalizationPass},
        parse::{ParseDialect, ParseOptions},
        types::DurationValue,
        KernDocument, SerializePreset,
    };

    fn compact(input: &str) -> KernDocument {
        let mut document = KernDocument::parse(input).unwrap();
        CompactConsecutiveInvisibleRests::new().normalize(&mut document);
        document
    }

    fn invisible_rest_durations(document: &KernDocument) -> Vec<(u32, u8)> {
        document
            .lines()
            .iter()
            .flat_map(|line| match line {
                KernLine::Data(contents) => contents.iter().collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .filter_map(|content| match content {
                SpineContent::Kern(KernContent::Rest(rest)) if rest.invisible => {
                    match rest.duration.value {
                        DurationValue::Reciprocal(value) => {
                            Some((value, rest.duration.augmentation_dots))
                        }
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn compacts_consecutive_rests_in_one_spine() {
        let input = [
            "**kern\t**kern",
            "*M4/4\t*M4/4",
            "8c\t8ryy",
            "8d\t8ryy",
            "8e\t8ryy",
            "8f\t8ryy",
            "*-\t*-",
        ]
        .join("\n");

        let output = compact(&input).to_humdrum_kern_string();

        assert!(output.contains("8c\t2ryy\n8d\t.\n8e\t.\n8f\t."));
    }

    #[test]
    fn preserves_continuation_lines_until_null_line_cleanup() {
        let input = [
            "**kern\t**kern",
            "*M3/4\t*M3/4",
            "4.GG 4.G\t8ryy",
            ".\t4ryy",
            "*-\t*-",
        ]
        .join("\n");

        let mut document = compact(&input);
        assert!(document
            .to_humdrum_kern_string()
            .contains("4.GG 4.G\t4.ryy\n.\t."));

        RemoveNullLines::new(Default::default()).normalize(&mut document);
        assert!(document
            .to_humdrum_kern_string()
            .contains("4.GG 4.G\t4.ryy\n*-\t*-"));
    }

    #[test]
    fn concat_005730_shape_serializes_to_single_compacted_invisible_rest_after_null_cleanup() {
        let input = [
            "**kern\t**kern\t**kern",
            "*M3/4\t*M3/4\t*M3/4",
            "*\t*^\t*",
            "4.D\t8bL\t4.f#\t16ff#LL",
            ".\t.\t.\t16b",
            ".\t8b-\t.\t16dd",
            ".\t.\t.\t16cc#",
            ".\t8anJ\t.\t16ccn",
            ".\t.\t.\t16aaJJ",
            "4.GG 4.G\t4.g\t8ryy\t4.b 4.gg",
            ".\t.\t4ryy\t.",
            "*\t*v\t*v\t*",
            "=\t=\t=",
            "*-\t*-\t*-",
        ]
        .join("\n");

        let mut document = compact(&input);
        RemoveNullLines::new(Default::default()).normalize(&mut document);
        let target = SerializePreset::TranscodaV2.serialize_document(&document);

        assert!(target.contains("4.GG G\t4.g\t4.ryy\t4.b gg\n*\t*v\t*v\t*"));
        assert!(!target.contains("\tryy"));

        let reparsed = KernDocument::parse_with_options(
            &target,
            ParseOptions {
                dialect: ParseDialect::TranscodaV2,
            },
        )
        .unwrap();
        assert!(data_line_durations(&reparsed).is_ok());
    }

    #[test]
    fn does_not_cross_barlines_or_spine_operations() {
        let input = [
            "**kern\t**kern",
            "*M4/4\t*M4/4",
            "8c\t8ryy",
            "=\t=",
            "8d\t8ryy",
            "*^\t*",
            "8e\t.\t8ryy",
            "*-\t*-\t*-",
        ]
        .join("\n");

        let output = compact(&input).to_humdrum_kern_string();

        assert!(output.contains("8c\t8ryy\n=\t=\n8d\t8ryy"));
        assert!(output.contains("*^\t*\n8e\t.\t8ryy"));
    }

    #[test]
    fn crosses_dropped_comments_but_not_emitted_tandems() {
        // Emitted tandems (clefs, meters, ...) interrupt runs: they survive
        // serialization, so splitting on them is stable across round trips.
        // Layout-text comments are dropped from compacted output and consume
        // no time, so runs must span them; splitting on them in memory made
        // the next parse/serialize round trip compact further.
        let input = [
            "**kern\t**kern",
            "*M4/4\t*M4/4",
            "8c\t8ryy",
            "*clefG2\t*clefG2",
            "8d\t8ryy",
            "!LO:TX:a:t=note\t!",
            "8e\t8ryy",
            "8f\t8ryy",
            "*-\t*-",
        ]
        .join("\n");

        let document = compact(&input);

        assert_eq!(invisible_rest_durations(&document), vec![(8, 0), (4, 1)]);
    }

    #[test]
    fn compacts_multiple_spines_independently() {
        let input = [
            "**kern\t**kern\t**kern",
            "*M4/4\t*M4/4\t*M4/4",
            "8c\t8ryy\t4ryy",
            "8d\t8ryy\t.",
            "8e\t8ryy\t4ryy",
            "8f\t8ryy\t.",
            "*-\t*-\t*-",
        ]
        .join("\n");

        let output = compact(&input).to_humdrum_kern_string();

        assert!(output.contains("8c\t2ryy\t2ryy\n8d\t.\t.\n8e\t.\t.\n8f\t.\t."));
    }

    #[test]
    fn does_not_cross_inactive_kern_nulls() {
        let input = [
            "**kern\t**kern",
            "*M4/4\t*M4/4",
            "16ryy\t16c",
            ".\t16d",
            "16ryy\t16e",
            ".\t16f",
            "*-\t*-",
        ]
        .join("\n");

        let output = compact(&input).to_humdrum_kern_string();

        assert!(output.contains("16ryy\t16c\n.\t16d\n16ryy\t16e\n.\t16f"));
        assert!(!output.contains("8ryy"));
    }

    #[test]
    fn does_not_shift_retained_tail_line_annotations() {
        let input = [
            "**kern\t**kern\t**dynam",
            "*M4/4\t*M4/4\t*",
            "4c\t8ryy\t.",
            ".\t8ryy\tp",
            "*-\t*-\t*-",
        ]
        .join("\n");

        let output = compact(&input).to_humdrum_kern_string();

        assert!(output.contains("4c\t8ryy\t.\n.\t8ryy\tp"));
        assert!(!output.contains("4c\t4ryy\t.\n.\t.\tp"));
    }

    #[test]
    fn does_not_compact_non_plain_invisible_rests() {
        let input = [
            "**kern\t**kern",
            "*M4/4\t*M4/4",
            "8c\t8ryy;",
            "8d\t8ryy",
            "8e\t8ryy",
            "*-\t*-",
        ]
        .join("\n");

        let output = compact(&input).to_humdrum_kern_string();

        assert!(output.contains("8c\t8ryy;\n8d\t4ryy\n8e\t."));
    }
}
