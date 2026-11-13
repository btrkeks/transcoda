//! Normalization for adjacent spine manipulator lines.
//!
//! This pass is intentionally narrow: it only rewrites adjacent two-line
//! manipulator windows when the combined topology change can be represented by
//! one valid [`crate::SpineManipulatorLine`].
//!
//! Example:
//! ```text
//! Before:
//! *\t*^
//! *^\t*\t*
//!
//! After:
//! *^\t*^
//! ```
//!
//! The two independent splits occur on different original spines, so they can
//! be represented as one wider manipulator line.
//!
//! This is different from
//! [`crate::normalization::canceling_split_merge_regions`]: there is no
//! matching merge here, no region to project, and no intervening musical
//! content to inspect. The pass only compresses adjacent topology-only lines
//! when their combined effect has the same final spine order.

use crate::{
    normalization::{
        spine_topology::{
            all_ops_renderable, apply_ops, combine_ops_for_initial_live, contains_terminate,
            has_adjacent_independent_merge_groups, has_mixed_split_and_merge, initial_live,
            line_to_ops, live_range, render_batch, SpineId, SpineOp,
        },
        NormalizationPass,
    },
    KernDocument, KernLine, SpineManipulator, SpineManipulatorLine,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Merge adjacent spine manipulator lines when one legal line is equivalent.
pub struct MergeAdjacentSpineManipulators {
    options: NormalizeOptions,
}

impl MergeAdjacentSpineManipulators {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for MergeAdjacentSpineManipulators {
    fn normalize(&self, document: &mut KernDocument) {
        let mut index = 0;
        while index < document.lines().len() {
            let Some(rewrite) = split_mixed_manipulator_line(&document.lines()[index]) else {
                index += 1;
                continue;
            };

            let inserted_len = rewrite.len();
            let source = document.source(index);
            document.splice_lines(
                index..index + 1,
                rewrite
                    .into_iter()
                    .map(|line| (KernLine::SpineManipulatorLine(line), source))
                    .collect(),
            );
            index += inserted_len;
        }

        let mut index = 0;
        while index + 1 < document.lines().len() {
            let Some(rewrite) =
                rewrite_pair(&document.lines()[index], &document.lines()[index + 1])
            else {
                index += 1;
                continue;
            };

            let source = document.source(index);
            document.splice_lines(
                index..index + 2,
                rewrite
                    .into_iter()
                    .map(|line| (KernLine::SpineManipulatorLine(line), source))
                    .collect(),
            );
            index = index.saturating_sub(1);
        }
    }
}

fn split_mixed_manipulator_line(line: &KernLine) -> Option<Vec<SpineManipulatorLine>> {
    let KernLine::SpineManipulatorLine(line) = line else {
        return None;
    };
    if contains_terminate(line) || !line_has_split_and_merge(line) {
        return None;
    }

    let initial_live = initial_live(line.len());
    let ops = line_to_ops(line, &initial_live)?;
    let original_final_live = apply_ops(&initial_live, &ops)?;

    let merge_ops = ops
        .iter()
        .filter(|op| matches!(op, SpineOp::Merge { .. }))
        .cloned()
        .collect::<Vec<_>>();
    let split_ops = ops
        .iter()
        .filter(|op| matches!(op, SpineOp::Split { .. }))
        .cloned()
        .collect::<Vec<_>>();

    let mut rewrite = Vec::new();
    let mut live = initial_live;
    if !merge_ops.is_empty() {
        let rendered = render_batch(&live, &merge_ops)?;
        rewrite.push(SpineManipulatorLine::new(rendered).ok()?);
        live = apply_ops(&live, &merge_ops)?;
    }
    if !split_ops.is_empty() {
        let rendered = render_batch(&live, &split_ops)?;
        rewrite.push(SpineManipulatorLine::new(rendered).ok()?);
        live = apply_ops(&live, &split_ops)?;
    }

    (live == original_final_live).then_some(rewrite)
}

fn line_has_split_and_merge(line: &SpineManipulatorLine) -> bool {
    line.tokens()
        .iter()
        .any(|token| matches!(token, SpineManipulator::Split))
        && line
            .tokens()
            .iter()
            .any(|token| matches!(token, SpineManipulator::Merge))
}

fn rewrite_pair(left: &KernLine, right: &KernLine) -> Option<Vec<SpineManipulatorLine>> {
    let (KernLine::SpineManipulatorLine(left), KernLine::SpineManipulatorLine(right)) =
        (left, right)
    else {
        return None;
    };
    if contains_terminate(left) || contains_terminate(right) {
        return None;
    }

    let initial_live = initial_live(left.len());
    let left_ops = line_to_ops(left, &initial_live)?;
    let intermediate_live = apply_ops(&initial_live, &left_ops)?;
    let right_ops = line_to_ops(right, &intermediate_live)?;
    let final_live = apply_ops(&intermediate_live, &right_ops)?;

    if let Some(rewrite) =
        cancel_immediate_split_merges(&initial_live, &left_ops, &right_ops, &final_live)
    {
        return rewrite_differs(left, right, &rewrite).then_some(rewrite);
    }

    if let Some(rewrite) = canonicalize_independent_split_merge_order(
        &initial_live,
        &left_ops,
        &right_ops,
        &final_live,
    ) {
        return rewrite_differs(left, right, &rewrite).then_some(rewrite);
    }

    if let Some(rewrite) =
        canonicalize_independent_merge_order(&initial_live, &left_ops, &right_ops, &final_live)
    {
        return rewrite_differs(left, right, &rewrite).then_some(rewrite);
    }

    collapse_pair(&initial_live, &left_ops, &right_ops, &final_live)
        .map(|line| vec![line])
        .filter(|rewrite| rewrite_differs(left, right, rewrite))
}

fn collapse_pair(
    initial_live: &[SpineId],
    left_ops: &[SpineOp],
    right_ops: &[SpineOp],
    final_live: &[SpineId],
) -> Option<SpineManipulatorLine> {
    let combined_ops = combine_ops_for_initial_live(initial_live, left_ops, right_ops)?;
    if has_mixed_split_and_merge(&combined_ops) {
        return None;
    }
    if has_adjacent_independent_merge_groups(initial_live, &combined_ops) {
        return None;
    }

    let rendered = render_batch(initial_live, &combined_ops)?;
    let collapsed = SpineManipulatorLine::new(rendered).ok()?;
    let combined_final_live = apply_ops(initial_live, &combined_ops)?;

    (combined_final_live == final_live).then_some(collapsed)
}

fn cancel_immediate_split_merges(
    initial_live: &[SpineId],
    left_ops: &[SpineOp],
    right_ops: &[SpineOp],
    _final_live: &[SpineId],
) -> Option<Vec<SpineManipulatorLine>> {
    let (remaining_left, remaining_right) = remove_canceling_split_merges(left_ops, right_ops)?;
    let simplified_final_live = apply_two_step(initial_live, &remaining_left, &remaining_right)?;
    render_rewrite(
        initial_live,
        &remaining_left,
        &remaining_right,
        &simplified_final_live,
    )
}

fn remove_canceling_split_merges(
    left_ops: &[SpineOp],
    right_ops: &[SpineOp],
) -> Option<(Vec<SpineOp>, Vec<SpineOp>)> {
    let mut canceled_left = vec![false; left_ops.len()];
    let mut canceled_right = vec![false; right_ops.len()];

    for (right_index, right_op) in right_ops.iter().enumerate() {
        let SpineOp::Merge { inputs, .. } = right_op else {
            continue;
        };
        let Some(left_index) = left_ops
            .iter()
            .enumerate()
            .find_map(|(left_index, left_op)| match left_op {
                SpineOp::Split { outputs, .. }
                    if !canceled_left[left_index] && inputs.as_slice() == outputs.as_slice() =>
                {
                    Some(left_index)
                }
                SpineOp::Split { .. } | SpineOp::Merge { .. } => None,
            })
        else {
            continue;
        };

        canceled_left[left_index] = true;
        canceled_right[right_index] = true;
    }

    if !canceled_left.iter().any(|canceled| *canceled) {
        return None;
    }

    let remaining_left = left_ops
        .iter()
        .enumerate()
        .filter(|(index, _)| !canceled_left[*index])
        .map(|(_, op)| op.clone())
        .collect::<Vec<_>>();
    let remaining_right = right_ops
        .iter()
        .enumerate()
        .filter(|(index, _)| !canceled_right[*index])
        .map(|(_, op)| op.clone())
        .collect::<Vec<_>>();

    Some((remaining_left, remaining_right))
}

fn canonicalize_independent_split_merge_order(
    initial_live: &[SpineId],
    left_ops: &[SpineOp],
    right_ops: &[SpineOp],
    final_live: &[SpineId],
) -> Option<Vec<SpineManipulatorLine>> {
    if !left_ops
        .iter()
        .any(|op| matches!(op, SpineOp::Split { .. }))
        || !right_ops
            .iter()
            .any(|op| matches!(op, SpineOp::Merge { .. }))
    {
        return None;
    }
    if !ops_are_independent_split_then_merge(initial_live, left_ops, right_ops) {
        return None;
    }

    let reordered_left = right_ops
        .iter()
        .filter(|op| matches!(op, SpineOp::Merge { .. }))
        .cloned()
        .collect::<Vec<_>>();
    let reordered_right = left_ops
        .iter()
        .filter(|op| matches!(op, SpineOp::Split { .. }))
        .cloned()
        .collect::<Vec<_>>();

    render_two_line_rewrite(initial_live, &reordered_left, &reordered_right, final_live)
}

fn canonicalize_independent_merge_order(
    initial_live: &[SpineId],
    left_ops: &[SpineOp],
    right_ops: &[SpineOp],
    final_live: &[SpineId],
) -> Option<Vec<SpineManipulatorLine>> {
    let mut merge_ops = combine_ops_for_initial_live(initial_live, left_ops, right_ops)?;
    if merge_ops.len() < 2
        || merge_ops
            .iter()
            .any(|op| !matches!(op, SpineOp::Merge { .. }))
    {
        return None;
    }

    for op in &merge_ops {
        let SpineOp::Merge { inputs, .. } = op else {
            return None;
        };
        live_range(initial_live, inputs)?;
    }

    merge_ops.sort_by_key(|op| match op {
        SpineOp::Merge { inputs, .. } => live_range(initial_live, inputs),
        SpineOp::Split { .. } => None,
    });

    let mut remaining = merge_ops;
    let mut live = initial_live.to_vec();
    let mut rewrite = Vec::new();
    while !remaining.is_empty() {
        let mut batch = Vec::new();
        let mut index = 0;
        while index < remaining.len() {
            let mut candidate = batch.clone();
            candidate.push(remaining[index].clone());
            if !has_adjacent_independent_merge_groups(&live, &candidate)
                && all_ops_renderable(&live, &candidate)
            {
                batch = candidate;
                remaining.remove(index);
            } else {
                index += 1;
            }
        }

        if batch.is_empty() {
            return None;
        }

        let rendered = render_batch(&live, &batch)?;
        rewrite.push(SpineManipulatorLine::new(rendered).ok()?);
        live = apply_ops(&live, &batch)?;
    }

    (live == final_live).then_some(rewrite)
}

fn apply_two_step(
    initial_live: &[SpineId],
    left_ops: &[SpineOp],
    right_ops: &[SpineOp],
) -> Option<Vec<SpineId>> {
    let intermediate_live = apply_ops(initial_live, left_ops)?;
    apply_ops(&intermediate_live, right_ops)
}

fn ops_are_independent_split_then_merge(
    initial_live: &[SpineId],
    left_ops: &[SpineOp],
    right_ops: &[SpineOp],
) -> bool {
    if left_ops
        .iter()
        .any(|op| !matches!(op, SpineOp::Split { .. }))
        || right_ops
            .iter()
            .any(|op| !matches!(op, SpineOp::Merge { .. }))
    {
        return false;
    }

    let split_inputs = left_ops
        .iter()
        .filter_map(|op| match op {
            SpineOp::Split { input, .. } => Some(input),
            SpineOp::Merge { .. } => None,
        })
        .collect::<Vec<_>>();

    right_ops.iter().all(|op| match op {
        SpineOp::Merge { inputs, .. } => {
            live_range(initial_live, inputs).is_some()
                && !inputs.iter().any(|input| split_inputs.contains(&input))
        }
        SpineOp::Split { .. } => false,
    })
}

fn render_rewrite(
    initial_live: &[SpineId],
    left_ops: &[SpineOp],
    right_ops: &[SpineOp],
    final_live: &[SpineId],
) -> Option<Vec<SpineManipulatorLine>> {
    if let Some(one_line) = render_one_line_rewrite(initial_live, left_ops, right_ops, final_live) {
        return Some(vec![one_line]);
    }

    render_two_line_rewrite(initial_live, left_ops, right_ops, final_live)
}

fn render_one_line_rewrite(
    initial_live: &[SpineId],
    left_ops: &[SpineOp],
    right_ops: &[SpineOp],
    final_live: &[SpineId],
) -> Option<SpineManipulatorLine> {
    let combined_ops = combine_ops_for_initial_live(initial_live, left_ops, right_ops)?;
    if has_mixed_split_and_merge(&combined_ops) {
        return None;
    }
    if has_adjacent_independent_merge_groups(initial_live, &combined_ops) {
        return None;
    }

    let rendered = render_batch(initial_live, &combined_ops)?;
    let line = SpineManipulatorLine::new(rendered).ok()?;
    let combined_final_live = apply_ops(initial_live, &combined_ops)?;

    (combined_final_live == final_live).then_some(line)
}

fn render_two_line_rewrite(
    initial_live: &[SpineId],
    left_ops: &[SpineOp],
    right_ops: &[SpineOp],
    final_live: &[SpineId],
) -> Option<Vec<SpineManipulatorLine>> {
    if left_ops.is_empty() && right_ops.is_empty() {
        return (initial_live == final_live).then_some(Vec::new());
    }

    let mut rewrite = Vec::new();
    let mut live = initial_live.to_vec();
    if !left_ops.is_empty() {
        if has_adjacent_independent_merge_groups(&live, left_ops) {
            return None;
        }
        let rendered = render_batch(&live, left_ops)?;
        rewrite.push(SpineManipulatorLine::new(rendered).ok()?);
        live = apply_ops(&live, left_ops)?;
    }

    if !right_ops.is_empty() {
        if has_adjacent_independent_merge_groups(&live, right_ops) {
            return None;
        }
        let rendered = render_batch(&live, right_ops)?;
        rewrite.push(SpineManipulatorLine::new(rendered).ok()?);
        live = apply_ops(&live, right_ops)?;
    }

    (live == final_live).then_some(rewrite)
}

fn rewrite_differs(
    left: &SpineManipulatorLine,
    right: &SpineManipulatorLine,
    rewrite: &[SpineManipulatorLine],
) -> bool {
    match rewrite {
        [rewritten_left, rewritten_right] => rewritten_left != left || rewritten_right != right,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::MergeAdjacentSpineManipulators;
    use crate::{normalization::NormalizationPass, KernDocument, KernLine, SpineManipulator};

    fn pass_output(input: &str) -> Vec<String> {
        let mut document = KernDocument::parse(input).unwrap();
        MergeAdjacentSpineManipulators::new(Default::default()).normalize(&mut document);
        manipulator_lines(&document)
    }

    fn manipulator_lines(document: &KernDocument) -> Vec<String> {
        document
            .lines()
            .iter()
            .filter_map(|line| match line {
                KernLine::SpineManipulatorLine(manipulators) => Some(
                    manipulators
                        .tokens()
                        .iter()
                        .map(manipulator_token)
                        .collect::<Vec<_>>()
                        .join("\t"),
                ),
                KernLine::SpineTypeLine(_)
                | KernLine::TandemInterpretationLine(_)
                | KernLine::LocalCommentLine(_)
                | KernLine::Barline(_)
                | KernLine::Data(_) => None,
            })
            .collect()
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
    fn chained_merge_flattens() {
        let input = "**kern\t**kern\t**kern\t**kern\n*v\t*v\t*\t*\n*v\t*v\t*\n";

        assert_eq!(pass_output(input), vec!["*v\t*v\t*v\t*"]);
    }

    #[test]
    fn independent_splits_collapse() {
        let input = "**kern\t**kern\n*\t*^\n*^\t*\t*\n";

        assert_eq!(pass_output(input), vec!["*^\t*^"]);
    }

    #[test]
    fn split_chain_does_not_collapse() {
        let input = "**kern\t**kern\n*\t*^\n*\t*\t*^\n";

        assert_eq!(pass_output(input), vec!["*\t*^", "*\t*\t*^"]);
    }

    #[test]
    fn adjacent_independent_merges_preserve_boundaries() {
        let input = "**kern\t**kern\t**kern\t**kern\t**kern\n*v\t*v\t*\t*\t*\n*\t*v\t*v\t*\n";

        assert_eq!(pass_output(input), vec!["*v\t*v\t*\t*\t*", "*\t*v\t*v\t*"]);
    }

    #[test]
    fn adjacent_independent_merges_order_left_to_right() {
        let input = "**kern\t**kern\t**kern\t**kern\n*\t*\t*v\t*v\n*v\t*v\t*\n";

        assert_eq!(pass_output(input), vec!["*v\t*v\t*\t*", "*\t*v\t*v"]);
    }

    #[test]
    fn non_adjacent_independent_merges_collapse() {
        let input =
            "**kern\t**kern\t**kern\t**kern\t**kern\t**kern\n*v\t*v\t*\t*\t*\t*\n*\t*\t*\t*v\t*v\n";

        assert_eq!(pass_output(input), vec!["*v\t*v\t*\t*\t*v\t*v"]);
    }

    #[test]
    fn non_adjacent_independent_merges_order_left_to_right_before_collapsing() {
        let input =
            "**kern\t**kern\t**kern\t**kern\t**kern\t**kern\n*\t*\t*\t*\t*v\t*v\n*v\t*v\t*\t*\t*\n";

        assert_eq!(pass_output(input), vec!["*v\t*v\t*\t*\t*v\t*v"]);
    }

    #[test]
    fn split_then_merge_of_split_children_cancels() {
        let input = "**kern\t**kern\n*\t*^\n*\t*v\t*v\n";

        assert_eq!(pass_output(input), Vec::<String>::new());
    }

    #[test]
    fn partial_split_merge_cancellation_preserves_unrelated_split() {
        let input = "**kern\t**kern\n*^\t*^\n*v\t*v\t*\t*\n";

        assert_eq!(pass_output(input), vec!["*\t*^"]);
    }

    #[test]
    fn partial_split_merge_cancellation_preserves_unrelated_right_split() {
        let input = "**kern\t**kern\n*\t*^\n*^\t*v\t*v\n";

        assert_eq!(pass_output(input), vec!["*^\t*"]);
    }

    #[test]
    fn split_then_independent_merge_reorders_to_merge_then_split() {
        let input = "**kern\t**kern\t**kern\n*\t*\t*^\n*v\t*v\t*\t*\n";

        assert_eq!(pass_output(input), vec!["*v\t*v\t*", "*\t*^"]);
    }

    #[test]
    fn single_line_split_then_merge_splits_to_merge_then_split_lines() {
        let input = "**kern\t**kern\t**kern\n*^\t*v\t*v\n";

        assert_eq!(pass_output(input), vec!["*\t*v\t*v", "*^\t*"]);
    }

    #[test]
    fn single_line_merge_then_split_splits_to_merge_then_split_lines() {
        let input = "**kern\t**kern\t**kern\n*v\t*v\t*^\n";

        assert_eq!(pass_output(input), vec!["*v\t*v\t*", "*\t*^"]);
    }

    #[test]
    fn single_operation_lines_are_not_split() {
        let split_input = "**kern\t**kern\n*^\t*\n";
        let merge_input = "**kern\t**kern\t**kern\n*v\t*v\t*\n";

        assert_eq!(pass_output(split_input), vec!["*^\t*"]);
        assert_eq!(pass_output(merge_input), vec!["*v\t*v\t*"]);
    }

    #[test]
    fn dependent_split_then_merge_does_not_reorder() {
        let input = "**kern\t**kern\n*\t*^\n*v\t*v\t*\n";

        assert_eq!(pass_output(input), vec!["*\t*^", "*v\t*v\t*"]);
    }

    #[test]
    fn adjacent_independent_merge_groups_stay_separate_when_reordering() {
        let input = "**kern\t**kern\t**kern\t**kern\n*\t*^\t*\t*\n*v\t*v\t*\t*v\t*v\n";

        assert_eq!(pass_output(input), vec!["*\t*^\t*\t*", "*v\t*v\t*\t*v\t*v"]);
    }

    #[test]
    fn common_merge_then_split_window_stays_stable() {
        let input = "**kern\t**kern\t**kern\n*\t*v\t*v\n*\t*^\n";

        assert_eq!(pass_output(input), vec!["*\t*v\t*v", "*\t*^"]);
    }

    #[test]
    fn top100_split_then_independent_merge_reorders() {
        let input = "**kern\t**kern\t**kern\n*\t*\t*^\n*v\t*v\t*\t*\n";

        assert_eq!(pass_output(input), vec!["*v\t*v\t*", "*\t*^"]);
    }

    #[test]
    fn termination_blocks_stay_unchanged() {
        let input = "**kern\t**kern\n*-\t*\n*^\n";

        assert_eq!(pass_output(input), vec!["*-\t*", "*^"]);
    }
}
