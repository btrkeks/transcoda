//! Symbolic helpers for reasoning about Humdrum spine topology.
//!
//! Normalization passes often need to parse a spine manipulator line, simulate
//! the resulting topology, and render a canonical operation batch.  This module
//! keeps that symbolic machinery in one place so passes can stay focused on
//! their local rewrite policy.

use crate::{SpineManipulator, SpineManipulatorLine};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpineId {
    Original(usize),
    Split { parent: Box<SpineId>, child: usize },
    Merge { children: Vec<SpineId> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpineOp {
    Split {
        input: SpineId,
        outputs: [SpineId; 2],
    },
    Merge {
        inputs: Vec<SpineId>,
        output: SpineId,
    },
}

pub(crate) fn initial_live(width: usize) -> Vec<SpineId> {
    (0..width).map(SpineId::Original).collect()
}

pub(crate) fn contains_terminate(line: &SpineManipulatorLine) -> bool {
    line.tokens()
        .iter()
        .any(|token| matches!(token, SpineManipulator::Terminate))
}

pub(crate) fn line_to_ops(line: &SpineManipulatorLine, live: &[SpineId]) -> Option<Vec<SpineOp>> {
    let tokens = line.tokens();
    if tokens.len() != live.len() {
        return None;
    }

    let mut ops = Vec::new();
    let mut index = 0;
    while index < tokens.len() {
        match tokens[index] {
            SpineManipulator::Null => index += 1,
            SpineManipulator::Terminate => return None,
            SpineManipulator::Split => {
                let input = live[index].clone();
                let outputs = [
                    SpineId::Split {
                        parent: Box::new(input.clone()),
                        child: 0,
                    },
                    SpineId::Split {
                        parent: Box::new(input.clone()),
                        child: 1,
                    },
                ];
                ops.push(SpineOp::Split { input, outputs });
                index += 1;
            }
            SpineManipulator::Merge => {
                let start = index;
                while index < tokens.len() && matches!(tokens[index], SpineManipulator::Merge) {
                    index += 1;
                }
                let inputs = live[start..index].to_vec();
                ops.push(SpineOp::Merge {
                    output: SpineId::Merge {
                        children: inputs.clone(),
                    },
                    inputs,
                });
            }
        }
    }

    Some(ops)
}

pub(crate) fn apply_ops(live: &[SpineId], ops: &[SpineOp]) -> Option<Vec<SpineId>> {
    let mut next = Vec::new();
    let mut index = 0;
    while index < live.len() {
        if let Some(op) = op_starting_at(live, ops, index) {
            match op {
                SpineOp::Split { outputs, .. } => {
                    next.extend(outputs.iter().cloned());
                    index += 1;
                }
                SpineOp::Merge { inputs, output } => {
                    if !live[index..].starts_with(inputs) {
                        return None;
                    }
                    next.push(output.clone());
                    index += inputs.len();
                }
            }
        } else {
            next.push(live[index].clone());
            index += 1;
        }
    }

    Some(next)
}

pub(crate) fn render_batch(live: &[SpineId], batch: &[SpineOp]) -> Option<Vec<SpineManipulator>> {
    if batch.is_empty() {
        return None;
    }

    let mut tokens = Vec::new();
    let mut index = 0;
    while index < live.len() {
        if let Some(op) = op_starting_at(live, batch, index) {
            match op {
                SpineOp::Split { .. } => {
                    tokens.push(SpineManipulator::Split);
                    index += 1;
                }
                SpineOp::Merge { inputs, .. } => {
                    tokens.extend((0..inputs.len()).map(|_| SpineManipulator::Merge));
                    index += inputs.len();
                }
            }
        } else {
            tokens.push(SpineManipulator::Null);
            index += 1;
        }
    }

    Some(tokens)
}

pub(crate) fn combine_ops_for_initial_live(
    initial_live: &[SpineId],
    left_ops: &[SpineOp],
    right_ops: &[SpineOp],
) -> Option<Vec<SpineOp>> {
    let mut consumed_left = vec![false; left_ops.len()];
    let mut additions = Vec::new();

    for right_op in right_ops {
        if op_inputs_live_in(right_op, initial_live) {
            additions.push(right_op.clone());
            continue;
        }

        let (flattened, consumed) = flatten_merge_chain(initial_live, left_ops, right_op)?;
        for index in consumed {
            consumed_left[index] = true;
        }
        additions.push(flattened);
    }

    let mut combined = left_ops
        .iter()
        .enumerate()
        .filter(|(index, _)| !consumed_left[*index])
        .map(|(_, op)| op.clone())
        .collect::<Vec<_>>();
    combined.extend(additions);

    all_ops_renderable(initial_live, &combined).then_some(combined)
}

pub(crate) fn all_ops_renderable(live: &[SpineId], ops: &[SpineOp]) -> bool {
    let mut rendered_count = 0;
    let mut index = 0;
    while index < live.len() {
        let Some(op) = op_starting_at(live, ops, index) else {
            index += 1;
            continue;
        };

        rendered_count += 1;
        index += op.input_len();
    }

    rendered_count == ops.len()
}

pub(crate) fn has_adjacent_independent_merge_groups(live: &[SpineId], ops: &[SpineOp]) -> bool {
    let mut ranges = ops
        .iter()
        .filter_map(|op| match op {
            SpineOp::Merge { inputs, .. } => live_range(live, inputs),
            SpineOp::Split { .. } => None,
        })
        .collect::<Vec<_>>();
    ranges.sort_unstable();

    ranges.windows(2).any(|pair| pair[0].1 >= pair[1].0)
}

pub(crate) fn has_mixed_split_and_merge(ops: &[SpineOp]) -> bool {
    ops.iter().any(|op| matches!(op, SpineOp::Split { .. }))
        && ops.iter().any(|op| matches!(op, SpineOp::Merge { .. }))
}

pub(crate) fn live_range(live: &[SpineId], inputs: &[SpineId]) -> Option<(usize, usize)> {
    if inputs.is_empty() || inputs.len() > live.len() {
        return None;
    }

    live.windows(inputs.len())
        .position(|window| window == inputs)
        .map(|start| (start, start + inputs.len()))
}

pub(crate) fn op_inputs_live_in(op: &SpineOp, live: &[SpineId]) -> bool {
    match op {
        SpineOp::Split { input, .. } => live.contains(input),
        SpineOp::Merge { inputs, .. } => live_range(live, inputs).is_some(),
    }
}

fn flatten_merge_chain(
    initial_live: &[SpineId],
    left_ops: &[SpineOp],
    right_op: &SpineOp,
) -> Option<(SpineOp, Vec<usize>)> {
    let SpineOp::Merge { inputs, output } = right_op else {
        return None;
    };

    let mut expanded = Vec::new();
    let mut consumed = Vec::new();
    for input in inputs {
        if initial_live.contains(input) {
            expanded.push(input.clone());
            continue;
        }

        let (index, producer_inputs) =
            left_ops
                .iter()
                .enumerate()
                .find_map(|(index, op)| match op {
                    SpineOp::Merge {
                        inputs,
                        output: producer_output,
                    } if producer_output == input => Some((index, inputs)),
                    SpineOp::Split { .. } | SpineOp::Merge { .. } => None,
                })?;

        consumed.push(index);
        expanded.extend(producer_inputs.iter().cloned());
    }

    consumed.sort_unstable();
    consumed.dedup();
    if consumed.is_empty() || live_range(initial_live, &expanded).is_none() {
        return None;
    }

    Some((
        SpineOp::Merge {
            inputs: expanded,
            output: output.clone(),
        },
        consumed,
    ))
}

fn op_starting_at<'a>(live: &[SpineId], ops: &'a [SpineOp], index: usize) -> Option<&'a SpineOp> {
    let matching = ops
        .iter()
        .filter(|op| match op {
            SpineOp::Split { input, .. } => live.get(index) == Some(input),
            SpineOp::Merge { inputs, .. } => live[index..].starts_with(inputs),
        })
        .collect::<Vec<_>>();

    if matching.len() == 1 {
        Some(matching[0])
    } else {
        None
    }
}

impl SpineOp {
    pub(crate) fn input_len(&self) -> usize {
        match self {
            SpineOp::Split { .. } => 1,
            SpineOp::Merge { inputs, .. } => inputs.len(),
        }
    }
}
