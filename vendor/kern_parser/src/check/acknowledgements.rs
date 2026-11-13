use super::{CheckDiagnostic, CheckPhase, CheckSeverity};
use crate::{
    score::{LayerOrigin, Topology},
    KernDocument, KernLine, LocalComment, SpineContent, TandemInterpretation,
};
use std::collections::BTreeMap;

/// Mark diagnostics that a `!ack:` comment acknowledges.
pub(super) fn acknowledge_diagnostics(score: &KernDocument, diagnostics: &mut [CheckDiagnostic]) {
    apply_acknowledgements(&acknowledged_tokens(score), diagnostics);
}

/// Demote each validation finding on an acknowledged token to info and
/// record the comment. Rule-blind: any validation finding on the token.
pub(super) fn apply_acknowledgements(
    tokens: &BTreeMap<(usize, usize), String>,
    diagnostics: &mut [CheckDiagnostic],
) {
    for diagnostic in diagnostics {
        if diagnostic.phase != CheckPhase::Validation {
            continue;
        }
        if let Some(ack) = diagnostic
            .line
            .zip(diagnostic.column)
            .and_then(|address| tokens.get(&address))
        {
            diagnostic.acknowledged = Some(ack.clone());
            diagnostic.severity = CheckSeverity::Info;
        }
    }
}

/// The token each `!ack:` comment acknowledges, keyed by source line and
/// zero-based field, with the comment text after `!ack:`. A comment applies
/// to the next note or clef of its layer, through a split into both
/// children, and through a merge when every input carries the same text.
/// A document whose topology does not build acknowledges nothing; its
/// `state` errors are reported instead.
pub(crate) fn acknowledged_tokens(score: &KernDocument) -> BTreeMap<(usize, usize), String> {
    let Ok(topology) = Topology::build(score) else {
        return BTreeMap::new();
    };
    let mut pending: Vec<Option<String>> = vec![None; topology.layer_count()];
    let mut tokens = BTreeMap::new();
    for (li, record) in score.lines().iter().enumerate() {
        let layers = topology.layers_at(li);
        match record {
            KernLine::LocalCommentLine(comments) => {
                for (&layer, comment) in layers.iter().zip(comments) {
                    if let LocalComment::Unsupported(raw) = comment {
                        if let Some(text) = raw.strip_prefix("!ack:") {
                            pending[layer.index()] = Some(text.to_owned());
                        }
                    }
                }
            }
            KernLine::SpineManipulatorLine(_) => {
                for &layer in topology.layers_after(li) {
                    if layers.contains(&layer) {
                        continue;
                    }
                    pending[layer.index()] = match &topology.layer(layer).origin {
                        LayerOrigin::SplitOf(parent) => pending[parent.index()].clone(),
                        LayerOrigin::MergeOf(inputs) => {
                            pending[inputs[0].index()].clone().filter(|ack| {
                                inputs
                                    .iter()
                                    .all(|input| pending[input.index()].as_ref() == Some(ack))
                            })
                        }
                        LayerOrigin::Exclusive => None,
                    };
                }
            }
            KernLine::Data(contents) => {
                for (field, (&layer, content)) in layers.iter().zip(contents).enumerate() {
                    if let (Some(ack), SpineContent::Kern(content)) =
                        (&pending[layer.index()], content)
                    {
                        if content.noteheads().next().is_some() {
                            tokens.insert((score.source(li).unwrap_or(li + 1), field), ack.clone());
                        }
                    }
                }
                for layer in layers {
                    pending[layer.index()] = None;
                }
            }
            KernLine::Barline(_) => {
                for layer in layers {
                    pending[layer.index()] = None;
                }
            }
            KernLine::SpineTypeLine(_) => {}
            KernLine::TandemInterpretationLine(tandems) => {
                for (field, (&layer, tandem)) in layers.iter().zip(tandems.tokens()).enumerate() {
                    if matches!(tandem, TandemInterpretation::Clef(_)) {
                        if let Some(ack) = pending[layer.index()].take() {
                            tokens.insert((score.source(li).unwrap_or(li + 1), field), ack);
                        }
                    }
                }
            }
        }
    }
    tokens
}
