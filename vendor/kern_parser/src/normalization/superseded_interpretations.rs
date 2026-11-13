//! Normalization for tandem interpretations that are cancelled before display.
//!
//! The pass is deliberately local: it only reasons within consecutive tandem
//! interpretation windows, so data, barlines, and spine-path operations all
//! commit the current visible state. Within a window it compares the
//! subspines of one track together, since clef, key, meter, ottava and pedal
//! are track state.
//!
//! Example:
//! ```text
//! Before:
//! *k[f#]
//! *k[f#c#]
//! 4c
//!
//! After:
//! *
//! *k[f#c#]
//! 4c
//! ```
//!
//! The first key signature is superseded before any rendered event can observe
//! it, so it is rewritten to a null tandem.
//!
//! A window's meter tandems form one declaration group, so the pass writes the
//! group as the output spells it ([`MeterOutput`]), on the group's last lines.

use std::ops::Range;

use crate::{
    meter::is_meter_tandem, normalization::NormalizationPass, score::Topology, ActiveMeter,
    KernDocument, KernLine, MeterOutput, OttavaKind, TandemInterpretation,
    TandemInterpretationLine,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Remove serializable tandem interpretations that cannot affect rendering.
pub struct RemoveSupersededInterpretations {
    options: NormalizeOptions,
    meter_output: MeterOutput,
}

impl RemoveSupersededInterpretations {
    pub fn new(options: NormalizeOptions, meter_output: MeterOutput) -> Self {
        Self {
            options,
            meter_output,
        }
    }
}

impl NormalizationPass for RemoveSupersededInterpretations {
    /// A document whose spine topology does not build is left unchanged.
    fn normalize(&self, document: &mut KernDocument) {
        let Ok(topology) = Topology::build(document) else {
            return;
        };
        let mut windows = Vec::new();
        let mut start = 0;
        while start < document.lines().len() {
            if !matches!(
                document.lines()[start],
                KernLine::TandemInterpretationLine(_)
            ) {
                start += 1;
                continue;
            }

            let end = document.lines()[start..]
                .iter()
                .position(|line| !matches!(line, KernLine::TandemInterpretationLine(_)))
                .map_or(document.lines().len(), |offset| start + offset);
            windows.push((
                start,
                superseded_in_window(document, &topology, start, end, self.meter_output),
            ));
            start = end;
        }
        for (start, edits) in windows {
            apply_tandem_edits(document, start, edits);
        }
    }
}

/// Replacement tandems per field of a window, by line offset. A removed
/// tandem is replaced by a null tandem.
type Edits = Vec<Vec<Option<TandemInterpretation>>>;

/// Edits for the window `start..end`, which holds no spine manipulator and so
/// keeps one layout throughout.
fn superseded_in_window(
    document: &KernDocument,
    topology: &Topology,
    start: usize,
    end: usize,
    meter_output: MeterOutput,
) -> Edits {
    let mut edits = document.lines()[start..end]
        .iter()
        .map(|line| match line {
            KernLine::TandemInterpretationLine(tandems) => vec![None; tandems.len()],
            _ => Vec::new(),
        })
        .collect::<Vec<_>>();

    for columns in track_columns(topology, start) {
        let window = Window {
            document,
            lines: start..end,
            columns,
        };
        mark_superseded_key_signatures(&window, &mut edits);
        respell_meter_group(&window, meter_output, &mut edits);
        mark_superseded_ottavas(&window, &mut edits);
        mark_superseded_pedals(&window, &mut edits);
    }

    edits
}

/// The columns of each track on `line`, left to right.
fn track_columns(topology: &Topology, line: usize) -> Vec<Range<usize>> {
    let mut tracks = topology
        .layers_at(line)
        .iter()
        .map(|&layer| topology.track_of(layer))
        .collect::<Vec<_>>();
    tracks.dedup();
    tracks
        .into_iter()
        .map(|track| topology.track_columns(line, track))
        .collect()
}

/// The fields of one track in a window of consecutive tandem lines.
struct Window<'a> {
    document: &'a KernDocument,
    lines: Range<usize>,
    columns: Range<usize>,
}

/// The positions on one line of the tandems a predicate accepts; sibling
/// subspines may write copies of one value.
type Group = Vec<(usize, usize)>;

impl Window<'_> {
    /// One group per line that holds a tandem `predicate` accepts.
    fn groups(&self, predicate: impl Fn(&TandemInterpretation) -> bool) -> Vec<Group> {
        self.lines
            .clone()
            .filter_map(|line_index| {
                let group = self
                    .columns
                    .clone()
                    .filter(|&column| {
                        tandem_at(self.document, line_index, column).is_some_and(&predicate)
                    })
                    .map(|column| (line_index, column))
                    .collect::<Group>();
                (!group.is_empty()).then_some(group)
            })
            .collect()
    }

    /// The tandem a group writes, when its copies agree.
    fn written(&self, group: &Group) -> Option<&TandemInterpretation> {
        let mut tandems = group
            .iter()
            .map(|&(line_index, column)| tandem_at(self.document, line_index, column));
        let first = tandems.next()??;
        tandems.all(|tandem| tandem == Some(first)).then_some(first)
    }
}

fn mark_superseded_key_signatures(window: &Window<'_>, edits: &mut Edits) {
    let groups = window.groups(|tandem| matches!(tandem, TandemInterpretation::KeySignature(_)));
    // The last key's naturals cancel the key before it; without that key
    // they would cancel nothing.
    let last_has_naturals = groups.last().is_some_and(|group| {
        group.iter().any(|&(line_index, column)| {
            matches!(
                tandem_at(window.document, line_index, column),
                Some(TandemInterpretation::KeySignature(key)) if key.has_naturals()
            )
        })
    });
    if last_has_naturals {
        return;
    }

    mark_all_but_last(&groups, window.lines.start, edits);
}

/// Write the window's meter declaration group as `output` spells it: one
/// tandem per spelled slot on the group's last lines, and nothing on its
/// earlier lines. A group whose sibling copies disagree is left as written.
fn respell_meter_group(window: &Window<'_>, output: MeterOutput, edits: &mut Edits) {
    let groups = window.groups(is_meter_tandem);
    let mut meter = ActiveMeter::new();
    for group in &groups {
        let Some(tandem) = window.written(group) else {
            return;
        };
        meter.apply(tandem);
    }
    let spelled = meter
        .declaration()
        .tandems(output)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let Some(cleared) = groups.len().checked_sub(spelled.len()) else {
        return;
    };
    let (earlier, later) = groups.split_at(cleared);
    mark_all(earlier, window.lines.start, edits);
    for (group, tandem) in later.iter().zip(spelled) {
        if window.written(group) != Some(&tandem) {
            write(group, &tandem, window.lines.start, edits);
        }
    }
}

fn mark_superseded_ottavas(window: &Window<'_>, edits: &mut Edits) {
    let mut pending_active: Option<(&Group, OttavaKind)> = None;

    for group in &window.groups(|tandem| matches!(tandem, TandemInterpretation::Ottava(_))) {
        let Some(TandemInterpretation::Ottava(ottava)) = window.written(group) else {
            pending_active = None;
            continue;
        };

        if ottava.active {
            if let Some((pending, _)) = pending_active {
                mark(pending, window.lines.start, edits);
            }
            pending_active = Some((group, ottava.kind));
            continue;
        }

        if let Some((pending, kind)) = pending_active {
            if kind == ottava.kind {
                mark(pending, window.lines.start, edits);
                mark(group, window.lines.start, edits);
                pending_active = None;
            }
        }
    }
}

fn mark_superseded_pedals(window: &Window<'_>, edits: &mut Edits) {
    let mut pending_active: Option<&Group> = None;

    for group in &window.groups(|tandem| matches!(tandem, TandemInterpretation::Pedal(_))) {
        let Some(TandemInterpretation::Pedal(pedal)) = window.written(group) else {
            pending_active = None;
            continue;
        };

        if pedal.active {
            if let Some(pending) = pending_active {
                mark(pending, window.lines.start, edits);
            }
            pending_active = Some(group);
            continue;
        }

        if let Some(pending) = pending_active {
            mark(pending, window.lines.start, edits);
            mark(group, window.lines.start, edits);
            pending_active = None;
        }
    }
}

fn tandem_at(
    document: &KernDocument,
    line_index: usize,
    column: usize,
) -> Option<&TandemInterpretation> {
    let KernLine::TandemInterpretationLine(tandems) = document.lines().get(line_index)? else {
        return None;
    };
    tandems.tokens().get(column)
}

fn mark(group: &Group, start: usize, edits: &mut Edits) {
    write(group, &TandemInterpretation::Null, start, edits);
}

fn mark_all(groups: &[Group], start: usize, edits: &mut Edits) {
    for group in groups {
        mark(group, start, edits);
    }
}

fn mark_all_but_last(groups: &[Group], start: usize, edits: &mut Edits) {
    if let Some((_, earlier)) = groups.split_last() {
        mark_all(earlier, start, edits);
    }
}

fn write(group: &Group, tandem: &TandemInterpretation, start: usize, edits: &mut Edits) {
    for &(line_index, column) in group {
        if let Some(edit) = edits
            .get_mut(line_index - start)
            .and_then(|line_edits| line_edits.get_mut(column))
        {
            *edit = Some(tandem.clone());
        }
    }
}

fn apply_tandem_edits(document: &mut KernDocument, start: usize, edits: Edits) {
    for (offset, line_edits) in edits.into_iter().enumerate() {
        if line_edits.iter().all(Option::is_none) {
            continue;
        }

        let KernLine::TandemInterpretationLine(tandems) = &mut document.lines_mut()[start + offset]
        else {
            continue;
        };
        let mut generated = tandems.generated_tokens().to_vec();
        for (index, (tandem, edit)) in tandems.tokens_mut().iter_mut().zip(line_edits).enumerate() {
            let Some(edit) = edit else {
                continue;
            };
            if edit == TandemInterpretation::Null {
                if let Some(generated) = generated.get_mut(index) {
                    *generated = false;
                }
            }
            *tandem = edit;
        }
        *tandems = TandemInterpretationLine::from_homogeneous_tokens_with_generated(
            tandems.tokens().to_vec(),
            generated,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::omr::{spell_tandem, VisibilityMarkerStyle};

    /// The tandem lines left after the pass.
    fn tandem_lines(input: &str) -> Vec<String> {
        spelled_tandem_lines(input, MeterOutput::SingleSlot)
    }

    fn spelled_tandem_lines(input: &str, meter_output: MeterOutput) -> Vec<String> {
        let mut document = KernDocument::parse(input).unwrap();
        RemoveSupersededInterpretations::new(NormalizeOptions::default(), meter_output)
            .normalize(&mut document);
        document
            .lines()
            .iter()
            .filter_map(|line| match line {
                KernLine::TandemInterpretationLine(tandems) => Some(
                    tandems
                        .tokens()
                        .iter()
                        .map(|tandem| {
                            spell_tandem(tandem, VisibilityMarkerStyle::Preserve)
                                .unwrap_or_else(|| "*".to_owned())
                        })
                        .collect::<Vec<_>>()
                        .join("\t"),
                ),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_sibling_subspine_supersedes_key_ottava_and_pedal() {
        let lines = tandem_lines(
            "**kern\n*clefG2\n*^\n4c\t4e\n*k[f#]\t*\n*\t*k[b-]\n*\t*8va\n*X8va\t*\n*ped\t*\n*\t*Xped\n4d\t4f\n*v\t*v\n*-\n",
        );

        assert_eq!(
            lines,
            [
                "*clefG2",
                "*k[]",
                "*\t*",
                "*\t*k[b-]",
                "*\t*",
                "*\t*",
                "*\t*",
                "*\t*"
            ]
        );
    }

    #[test]
    fn separate_tracks_do_not_supersede_each_other() {
        let lines = tandem_lines(
            "**kern\t**kern\n4c\t4e\n*k[f#]\t*\n*\t*k[b-]\n*8va\t*\n*\t*X8va\n4d\t4f\n*-\t*-\n",
        );

        assert_eq!(
            lines,
            [
                "*clefG2\t*clefG2",
                "*k[]\t*k[]",
                "*k[f#]\t*",
                "*\t*k[b-]",
                "*8va\t*",
                "*\t*X8va"
            ]
        );
    }

    #[test]
    fn a_meter_group_is_written_as_the_output_spells_it() {
        let cases = [
            // An empty symbol hides the time signature; it does not replace it.
            ("*M4/4\n*met()", ["*", "*M4/4yy"], ["*", "*"]),
            ("*M4/4yy\n*met(c)", ["*", "*M4/4yy"], ["*", "*"]),
            ("*M4/4\n*met(c)", ["*", "*met(c)"], ["*", "*met(c)"]),
            ("*MX\n*met(c)", ["*MX", "*met(c)"], ["*", "*met(c)"]),
            ("*M3/4\n*M4/4", ["*", "*M4/4"], ["*", "*M4/4"]),
        ];
        for (group, single_slot, printed) in cases {
            let input = format!("**kern\n*M3/4\n2.c\n{group}\n1d\n*-\n");
            let body = |output| spelled_tandem_lines(&input, output)[3..].to_vec();

            assert_eq!(body(MeterOutput::SingleSlot), single_slot, "{group}");
            assert_eq!(body(MeterOutput::Printed), printed, "{group}");
        }
    }
}
