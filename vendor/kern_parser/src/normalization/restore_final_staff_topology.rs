//! Restore final `**kern` split descendants to the original staff topology.
//!
//! This pass only merges live spines that can be proven to be descendants of
//! the same initial `**kern` staff. When an authored topology change makes that
//! restoration ambiguous, the pass leaves the final topology unchanged.

use std::collections::BTreeSet;

use crate::{
    normalization::NormalizationPass, KernDocument, KernLine, SpineManipulator,
    SpineManipulatorLine, SpineType, TandemInterpretationLine,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Merge open split descendants before the final barline or terminator.
pub struct RestoreFinalStaffTopology {
    options: NormalizeOptions,
}

impl RestoreFinalStaffTopology {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for RestoreFinalStaffTopology {
    fn normalize(&self, document: &mut KernDocument) {
        let insertion_index = insertion_index(document);
        if document.lines()[insertion_index..].iter().any(
            |line| matches!(line, KernLine::Barline(barline) if !barline.is_topology_invariant()),
        ) {
            return;
        }
        let live = live_before_line(document, insertion_index);
        let steps = restoration_steps(&live, &initial_kern_roots(document));
        if steps.is_empty() {
            return;
        }

        let mut current_live = live;
        let mut generated = Vec::new();
        let mut suffix = document.lines()[insertion_index..]
            .iter()
            .cloned()
            .enumerate()
            .map(|(offset, line)| (line, document.source(insertion_index + offset)))
            .collect::<Vec<_>>();

        for step in steps {
            let line = render_step(&current_live, &step);
            let next_live = apply_step(&current_live, &step);
            suffix = suffix
                .iter()
                .filter_map(|(line, source)| {
                    project_suffix_line(line, &current_live, &next_live, &step)
                        .map(|line| (line, *source))
                })
                .collect();
            generated.push((
                KernLine::SpineManipulatorLine(
                    SpineManipulatorLine::new(line)
                        .expect("generated final staff restoration merge line is valid"),
                ),
                None,
            ));
            current_live = next_live;
        }

        generated.extend(suffix);
        document.splice_lines(insertion_index..document.lines().len(), generated);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LiveSpine {
    roots: BTreeSet<usize>,
    spine_type: SpineType,
    kern_roots: BTreeSet<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MergeStep {
    start: usize,
    len: usize,
}

fn insertion_index(document: &KernDocument) -> usize {
    if let Some(index) = final_barline_index(document) {
        return index;
    }

    document
        .lines()
        .iter()
        .rposition(is_terminal_line)
        .unwrap_or(document.lines().len())
}

fn final_barline_index(document: &KernDocument) -> Option<usize> {
    let last_data = document
        .lines()
        .iter()
        .rposition(|line| matches!(line, KernLine::Data(_)));

    document
        .lines()
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, line)| {
            (matches!(line, KernLine::Barline(_)) && last_data.is_none_or(|data| index > data))
                .then_some(index)
        })
}

fn is_terminal_line(line: &KernLine) -> bool {
    matches!(
        line,
        KernLine::SpineManipulatorLine(manipulators)
            if manipulators
                .tokens()
                .iter()
                .all(|token| matches!(token, SpineManipulator::Terminate))
    )
}

fn live_before_line(document: &KernDocument, target_line: usize) -> Vec<LiveSpine> {
    let mut live = initial_live_spines(document);

    for line in document.lines().iter().take(target_line) {
        let KernLine::SpineManipulatorLine(manipulators) = line else {
            continue;
        };
        live = apply_manipulators(&live, manipulators.tokens());
    }

    live
}

fn initial_live_spines(document: &KernDocument) -> Vec<LiveSpine> {
    let spine_types = match document.lines().first() {
        Some(KernLine::SpineTypeLine(spine_types)) => spine_types.clone(),
        _ => vec![SpineType::Kern; document.layout.initial_spine_count as usize],
    };

    spine_types
        .into_iter()
        .enumerate()
        .map(|(root, spine_type)| LiveSpine {
            roots: BTreeSet::from([root]),
            kern_roots: if matches!(spine_type, SpineType::Kern) {
                BTreeSet::from([root])
            } else {
                BTreeSet::new()
            },
            spine_type,
        })
        .collect()
}

fn initial_kern_roots(document: &KernDocument) -> BTreeSet<usize> {
    initial_live_spines(document)
        .into_iter()
        .flat_map(|spine| spine.kern_roots)
        .collect()
}

fn apply_manipulators(live: &[LiveSpine], manipulators: &[SpineManipulator]) -> Vec<LiveSpine> {
    if manipulators.len() != live.len() {
        panic!(
            "internal error: final staff topology pass saw manipulator width {} for active width {}",
            manipulators.len(),
            live.len()
        );
    }

    let mut next = Vec::new();
    let mut index = 0;
    while index < manipulators.len() {
        match manipulators[index] {
            SpineManipulator::Null => {
                next.push(live[index].clone());
                index += 1;
            }
            SpineManipulator::Terminate => {
                index += 1;
            }
            SpineManipulator::Split => {
                next.push(live[index].clone());
                next.push(live[index].clone());
                index += 1;
            }
            SpineManipulator::Merge => {
                let start = index;
                while index < manipulators.len()
                    && matches!(manipulators[index], SpineManipulator::Merge)
                {
                    index += 1;
                }
                let group = &live[start..index];
                next.push(merge_group(group));
            }
        }
    }

    next
}

fn merge_group(group: &[LiveSpine]) -> LiveSpine {
    if group.len() < 2 {
        panic!("internal error: final staff topology pass saw a merge group shorter than 2");
    }

    let first = &group[0];
    if group
        .iter()
        .any(|spine| spine.spine_type != first.spine_type)
    {
        panic!(
            "internal error: cannot restore final staff topology after merging different spine types"
        );
    }

    LiveSpine {
        roots: group
            .iter()
            .flat_map(|spine| spine.roots.iter().copied())
            .collect(),
        spine_type: first.spine_type.clone(),
        kern_roots: group
            .iter()
            .flat_map(|spine| spine.kern_roots.iter().copied())
            .collect(),
    }
}

fn restoration_steps(live: &[LiveSpine], initial_kern_roots: &BTreeSet<usize>) -> Vec<MergeStep> {
    let mut groups = Vec::new();
    let mut index = 0;

    while index < live.len() {
        let start = index;
        let root = single_kern_root(&live[index], initial_kern_roots);
        while index < live.len()
            && single_kern_root(&live[index], initial_kern_roots) == root
            && live[index].spine_type == live[start].spine_type
        {
            index += 1;
        }

        let len = index - start;
        if root.is_some() && len > 1 {
            groups.push(MergeStep { start, len });
        }
    }

    if (!groups.is_empty() && has_ambiguous_split_descendants(live, initial_kern_roots))
        || has_noncontiguous_kern_descendants(live, initial_kern_roots)
    {
        return Vec::new();
    }

    groups.sort_by_key(|step| std::cmp::Reverse(step.start));
    groups
}

fn single_kern_root(spine: &LiveSpine, initial_kern_roots: &BTreeSet<usize>) -> Option<usize> {
    if !matches!(spine.spine_type, SpineType::Kern) || spine.kern_roots.len() != 1 {
        return None;
    }
    let root = *spine.kern_roots.iter().next()?;
    initial_kern_roots.contains(&root).then_some(root)
}

fn has_ambiguous_split_descendants(
    live: &[LiveSpine],
    initial_kern_roots: &BTreeSet<usize>,
) -> bool {
    initial_kern_roots.iter().any(|root| {
        let split_descendants = live
            .iter()
            .filter(|spine| spine.kern_roots.len() == 1 && spine.kern_roots.contains(root))
            .count();
        let merged_descendants = live
            .iter()
            .filter(|spine| spine.kern_roots.len() > 1 && spine.kern_roots.contains(root))
            .count();

        split_descendants > 0 && merged_descendants > 0
    })
}

fn has_noncontiguous_kern_descendants(
    live: &[LiveSpine],
    initial_kern_roots: &BTreeSet<usize>,
) -> bool {
    initial_kern_roots.iter().any(|root| {
        let positions = live
            .iter()
            .enumerate()
            .filter_map(|(index, spine)| {
                (spine.kern_roots.len() == 1 && spine.kern_roots.contains(root)).then_some(index)
            })
            .collect::<Vec<_>>();
        positions.len() > 1
            && positions
                .windows(2)
                .any(|pair| pair[1] != pair[0].saturating_add(1))
    })
}

fn render_step(live: &[LiveSpine], step: &MergeStep) -> Vec<SpineManipulator> {
    (0..live.len())
        .map(|index| {
            if (step.start..step.start + step.len).contains(&index) {
                SpineManipulator::Merge
            } else {
                SpineManipulator::Null
            }
        })
        .collect()
}

fn apply_step(live: &[LiveSpine], step: &MergeStep) -> Vec<LiveSpine> {
    let mut next = Vec::with_capacity(live.len() - step.len + 1);
    next.extend_from_slice(&live[..step.start]);
    next.push(live[step.start].clone());
    next.extend_from_slice(&live[step.start + step.len..]);
    next
}

fn project_suffix_line(
    line: &KernLine,
    live: &[LiveSpine],
    next_live: &[LiveSpine],
    step: &MergeStep,
) -> Option<KernLine> {
    match line {
        KernLine::SpineTypeLine(_) => {
            panic!("internal error: cannot project spine type line after final staff restoration")
        }
        KernLine::TandemInterpretationLine(tandems) => Some(KernLine::TandemInterpretationLine(
            TandemInterpretationLine::new(project_tokens(tandems.tokens(), step))
                .expect("projected tandem suffix line remains homogeneous"),
        )),
        KernLine::LocalCommentLine(comments) => {
            Some(KernLine::LocalCommentLine(project_tokens(comments, step)))
        }
        KernLine::Barline(_) => Some(line.clone()),
        KernLine::Data(_) => {
            panic!("internal error: cannot project data line after final staff restoration")
        }
        KernLine::SpineManipulatorLine(manipulators) => {
            project_suffix_manipulators(manipulators.tokens(), live, next_live, step)
        }
    }
}

fn project_tokens<T: Clone + PartialEq>(tokens: &[T], step: &MergeStep) -> Vec<T> {
    if tokens.len() < step.start + step.len {
        panic!("internal error: cannot project suffix line with mismatched width");
    }

    let first = &tokens[step.start];
    if tokens[step.start..step.start + step.len]
        .iter()
        .any(|token| token != first)
    {
        panic!("internal error: cannot project conflicting final suffix tokens");
    }

    let mut projected = Vec::with_capacity(tokens.len() - step.len + 1);
    projected.extend_from_slice(&tokens[..step.start]);
    projected.push(first.clone());
    projected.extend_from_slice(&tokens[step.start + step.len..]);
    projected
}

fn project_suffix_manipulators(
    tokens: &[SpineManipulator],
    live: &[LiveSpine],
    next_live: &[LiveSpine],
    step: &MergeStep,
) -> Option<KernLine> {
    if tokens == render_step(live, step).as_slice() {
        return None;
    }

    if tokens
        .iter()
        .all(|token| matches!(token, SpineManipulator::Terminate))
    {
        let projected_width = if tokens.len() == live.len() {
            next_live.len()
        } else if tokens.len() == next_live.len() {
            tokens.len()
        } else {
            panic!("internal error: cannot project terminal suffix line with mismatched width");
        };

        let tokens = vec![SpineManipulator::Terminate; projected_width];
        return Some(KernLine::SpineManipulatorLine(
            SpineManipulatorLine::new(tokens).expect("projected terminal suffix line is valid"),
        ));
    }

    if tokens.len() == live.len()
        && tokens[step.start..step.start + step.len]
            .iter()
            .all(|token| matches!(token, SpineManipulator::Null))
    {
        let tokens = project_tokens(tokens, step);
        return Some(KernLine::SpineManipulatorLine(
            SpineManipulatorLine::new(tokens)
                .expect("projected non-terminal suffix line remains valid"),
        ));
    }

    panic!(
        "internal error: cannot project non-terminal spine manipulator after final staff restoration"
    );
}

#[cfg(test)]
mod tests {
    use super::{restoration_steps, LiveSpine, RestoreFinalStaffTopology};
    use crate::{
        normalization::NormalizationPass, types::BarlineStyle, types::RepeatStyle, KernContent,
        KernDocument, KernLine, SpineContent, SpineManipulator, TandemInterpretation,
    };

    fn pass_string(input: &str) -> String {
        let mut document = KernDocument::parse(input).unwrap();
        RestoreFinalStaffTopology::new(Default::default()).normalize(&mut document);
        render_lines(&document)
    }

    fn render_lines(document: &KernDocument) -> String {
        document
            .lines()
            .iter()
            .enumerate()
            .filter(|(index, _)| document.source_line_number(*index) != 0)
            .map(|(_, line)| line)
            .filter_map(render_line)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    fn render_line(line: &KernLine) -> Option<String> {
        match line {
            KernLine::SpineTypeLine(_) => None,
            KernLine::TandemInterpretationLine(tandems) => Some(
                tandems
                    .tokens()
                    .iter()
                    .map(tandem_token)
                    .collect::<Vec<_>>()
                    .join("\t"),
            ),
            KernLine::SpineManipulatorLine(manipulators) => Some(
                manipulators
                    .tokens()
                    .iter()
                    .map(manipulator_token)
                    .collect::<Vec<_>>()
                    .join("\t"),
            ),
            KernLine::LocalCommentLine(comments) => {
                Some(comments.iter().map(|_| "!").collect::<Vec<_>>().join("\t"))
            }
            KernLine::Barline(barline) => Some(barline_token(barline).to_owned()),
            KernLine::Data(contents) => Some(
                contents
                    .iter()
                    .map(|content| match content {
                        SpineContent::Kern(KernContent::Null) => ".",
                        SpineContent::Kern(KernContent::Note(_))
                        | SpineContent::Kern(KernContent::Chord(_))
                        | SpineContent::Kern(KernContent::Rest(_))
                        | SpineContent::Kern(KernContent::InvisibleRestSlice(_))
                        | SpineContent::Kern(KernContent::MultiMeasureRest(_)) => "data",
                        SpineContent::Kern(KernContent::Grace(_))
                        | SpineContent::Dynam(_)
                        | SpineContent::Fing(_)
                        | SpineContent::Text(_)
                        | SpineContent::MxmlHarmony(_)
                        | SpineContent::Opaque { .. } => "?",
                    })
                    .collect::<Vec<_>>()
                    .join("\t"),
            ),
        }
    }

    fn tandem_token(tandem: &TandemInterpretation) -> &'static str {
        match tandem {
            TandemInterpretation::Null => "*",
            TandemInterpretation::Clef(_) => "*clef",
            TandemInterpretation::KeySignature(_) => "*k",
            _ => "?",
        }
    }

    fn barline_token(barline: &crate::Barline) -> &'static str {
        if barline.styles().contains(&BarlineStyle::Double) {
            "=="
        } else if barline.repeat() == RepeatStyle::Start {
            "=|:"
        } else if barline.repeat() == RepeatStyle::End {
            "=:|"
        } else {
            "="
        }
    }

    fn manipulator_token(manipulator: &SpineManipulator) -> &'static str {
        match manipulator {
            SpineManipulator::Null => "*",
            SpineManipulator::Terminate => "*-",
            SpineManipulator::Split => "*^",
            SpineManipulator::Merge => "*v",
        }
    }

    #[test]
    fn restores_single_open_split_before_final_barline() {
        let input = "**kern\n*^\n4c\t4e\n=\t=\n*-\t*-\n";

        assert_eq!(pass_string(input), "*^\ndata\tdata\n*v\t*v\n=\n*-\n");
    }

    #[test]
    fn removes_redundant_final_merge_after_final_barline() {
        let input = "**kern\t**kern\n*\t*^\n4c\t4e\t4g\n=\t=\t=\n*\t*v\t*v\n*-\t*-\n";

        assert_eq!(
            pass_string(input),
            "*\t*^\ndata\tdata\tdata\n*\t*v\t*v\n=\n*-\t*-\n"
        );
    }

    #[test]
    fn restores_before_final_terminal_when_no_final_barline_exists() {
        let input = "**kern\n*^\n4c\t4e\n*-\t*-\n";

        assert_eq!(pass_string(input), "*^\ndata\tdata\n*v\t*v\n*-\n");
    }

    #[test]
    fn appends_restoration_when_no_final_barline_or_terminal_exists() {
        let input = "**kern\n*^\n4c\t4e\n";

        assert_eq!(pass_string(input), "*^\ndata\tdata\n*v\t*v\n");
    }

    #[test]
    fn restores_nested_split_descendants_as_one_merge_group() {
        let input = "**kern\n*^\n*\t*^\n4c\t4e\t4g\n=\t=\t=\n*-\t*-\t*-\n";

        assert_eq!(
            pass_string(input),
            "*^\n*\t*^\ndata\tdata\tdata\n*v\t*v\t*v\n=\n*-\n"
        );
    }

    #[test]
    fn restores_adjacent_staff_groups_on_separate_lines() {
        let input = [
            "**kern\t**kern",
            "*^\t*^",
            "4c\t4e\t4g\t4b",
            "=\t=\t=\t=",
            "*-\t*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            pass_string(&input),
            "*^\t*^\ndata\tdata\tdata\tdata\n*\t*\t*v\t*v\n*v\t*v\t*\n=\n*-\t*-\n"
        );
    }

    #[test]
    fn projects_authored_final_merge_across_unrelated_restoration_step() {
        let input = [
            "**kern\t**kern",
            "*^\t*^",
            "4c\t4e\t4g\t4b",
            "=\t=\t=\t=",
            "*v\t*v\t*\t*",
            "*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            pass_string(&input),
            "*^\t*^\ndata\tdata\tdata\tdata\n*\t*\t*v\t*v\n*v\t*v\t*\n=\n*-\t*-\n"
        );
    }

    #[test]
    fn restores_remaining_staff_split_when_other_staff_has_authored_final_merge() {
        let input = [
            "**kern\t**kern",
            "*^\t*",
            "4c\t4e\t4g",
            "*\t*\t*^",
            "4d\t4f\t4a\t4b",
            "=\t=\t=\t=",
            "*v\t*v\t*\t*",
            "*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            pass_string(&input),
            "*^\t*\ndata\tdata\tdata\n*\t*\t*^\ndata\tdata\tdata\tdata\n*\t*\t*v\t*v\n*v\t*v\t*\n=\n*-\t*-\n"
        );
    }

    #[test]
    fn preserves_non_kern_spines_while_restoring_kern_descendants() {
        let input = [
            "**kern\t**dynam",
            "*^\t*",
            "4c\t4e\tmf",
            "=\t=\t=",
            "*-\t*-\t*-",
        ]
        .join("\n");

        assert_eq!(
            pass_string(&input),
            "*^\t*\ndata\tdata\t?\n*v\t*v\t*\n=\n*-\t*-\n"
        );
    }

    #[test]
    fn already_restored_topology_is_idempotent() {
        let input = "**kern\n*^\n4c\t4e\n*v\t*v\n=\n*-\n";
        let mut document = KernDocument::parse(input).unwrap();
        RestoreFinalStaffTopology::new(Default::default()).normalize(&mut document);
        let once = document.clone();
        RestoreFinalStaffTopology::new(Default::default()).normalize(&mut document);

        assert_eq!(document, once);
    }

    #[test]
    fn skips_restoration_on_cross_staff_merge() {
        let live = vec![
            LiveSpine {
                roots: std::collections::BTreeSet::from([0]),
                spine_type: crate::SpineType::Kern,
                kern_roots: std::collections::BTreeSet::from([0]),
            },
            LiveSpine {
                roots: std::collections::BTreeSet::from([0]),
                spine_type: crate::SpineType::Kern,
                kern_roots: std::collections::BTreeSet::from([0]),
            },
            LiveSpine {
                roots: std::collections::BTreeSet::from([0, 1]),
                spine_type: crate::SpineType::Kern,
                kern_roots: std::collections::BTreeSet::from([0, 1]),
            },
        ];

        assert!(restoration_steps(&live, &std::collections::BTreeSet::from([0, 1])).is_empty());
    }

    #[test]
    fn skips_restoration_on_non_adjacent_descendants() {
        let live = vec![
            LiveSpine {
                roots: std::collections::BTreeSet::from([0]),
                spine_type: crate::SpineType::Kern,
                kern_roots: std::collections::BTreeSet::from([0]),
            },
            LiveSpine {
                roots: std::collections::BTreeSet::from([1]),
                spine_type: crate::SpineType::Kern,
                kern_roots: std::collections::BTreeSet::from([1]),
            },
            LiveSpine {
                roots: std::collections::BTreeSet::from([0]),
                spine_type: crate::SpineType::Kern,
                kern_roots: std::collections::BTreeSet::from([0]),
            },
        ];

        assert!(restoration_steps(&live, &std::collections::BTreeSet::from([0, 1])).is_empty());
    }

    #[test]
    fn preserves_open_topology_for_mixed_final_barline_tokens() {
        let input = "**kern\n*^\n4c\t4e\n=\t==\n*-\t*-\n";
        let mut document = KernDocument::parse(input).unwrap();
        RestoreFinalStaffTopology::new(Default::default()).normalize(&mut document);

        assert!(document.lines().iter().any(
            |line| matches!(line, KernLine::Barline(barline) if !barline.is_topology_invariant())
        ));
        assert!(!document.lines().iter().enumerate().any(|(index, line)| {
            document.source_line_number(index) == 0
                && matches!(line, KernLine::SpineManipulatorLine(_))
        }));
    }
}
