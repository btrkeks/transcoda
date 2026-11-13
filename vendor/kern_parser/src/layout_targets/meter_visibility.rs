//! Project repeated meter declarations against the actual selected engraving.
use super::{has_class, parse_measure_id, LayoutTargetError};
use crate::{
    document::Framing,
    score::{Outcome, Score, StateKey},
    KernDocument, KernLine, MeterSymbol, TandemInterpretation,
};
use std::collections::BTreeMap;

/// Suppress meter groups whose staff has no corresponding drawn meter glyph.
/// Keep numeric repetitions and symbols that the selected engraving draws.
/// The rich document retains hidden meter state until target serialization.
pub fn project_meter_visibility(
    document: &mut KernDocument,
    svg: &str,
    rendered_sources: &[Option<usize>],
) -> Result<usize, LayoutTargetError> {
    let score = Score::resolve(document, Framing::Excerpt).map_err(|error| {
        LayoutTargetError::MalformedSvg {
            message: format!("cannot resolve meter projection: {error}"),
        }
    })?;
    if !score.events().iter().any(|event| {
        event.key == StateKey::Meter
            && matches!(event.outcome, Outcome::Changed | Outcome::Restated)
    }) {
        return Ok(0);
    }
    let xml = roxmltree::Document::parse(svg).map_err(|error| LayoutTargetError::MalformedSvg {
        message: error.to_string(),
    })?;
    let mut visible = BTreeMap::new();
    let mut starts = Vec::new();
    for measure in xml.descendants().filter(|node| has_class(*node, "measure")) {
        let rendered_line = parse_measure_id(measure.attribute("id").unwrap_or_default())?;
        let source = rendered_sources
            .get(rendered_line - 1)
            .ok_or(LayoutTargetError::RenderedLineOutOfRange { rendered_line })?
            .or_else(|| (starts.is_empty() && rendered_line == 1).then_some(0))
            .ok_or(LayoutTargetError::MissingSourceForRenderedLine { rendered_line })?;
        starts.push(source);
        for (staff_index, staff) in measure
            .children()
            .filter(|node| has_class(*node, "staff"))
            .enumerate()
        {
            let drawn = staff
                .descendants()
                .filter(|&node| {
                    (has_class(node, "meterSig") || has_class(node, "mensur"))
                        && !node.ancestors().any(|ancestor| {
                            ancestor.attribute("visibility") == Some("hidden")
                                || ancestor.attribute("display") == Some("none")
                                || ancestor.attribute("style").is_some_and(|style| {
                                    style.replace(' ', "").contains("display:none")
                                })
                        })
                        && node
                            .descendants()
                            .any(|child| child.is_element() && child.tag_name().name() == "use")
                })
                .count();
            visible.insert((source, staff_index as u32 + 1), drawn);
        }
    }
    starts.sort_unstable();
    // A glyph belongs to a declaration group, not to each of its *M/*met
    // fields. Count new declarations first so a later change cannot lend its
    // glyph to an earlier invisible restatement in the same measure.
    let mut groups = BTreeMap::<_, Vec<_>>::new();
    for event in score
        .events()
        .iter()
        .filter(|event| event.key == StateKey::Meter)
    {
        let layer = score.topology().layers_at(event.line_index)[event.column];
        let Some(staff) = score
            .topology()
            .track(score.topology().track_of(layer))
            .staff
        else {
            continue;
        };
        let mut start = event.line_index;
        while start > 0
            && matches!(
                document.lines()[start - 1],
                KernLine::TandemInterpretationLine(_)
            )
        {
            start -= 1;
        }
        groups.entry((start, staff)).or_default().push(event);
    }
    let mut restatements = Vec::new();
    let mut suppressed_groups = Vec::new();
    for ((_start, staff), events) in groups {
        let line = events
            .iter()
            .map(|event| {
                document
                    .source(event.line_index)
                    .unwrap_or(event.line_index + 1)
            })
            .min()
            .unwrap();
        let measure = starts
            .iter()
            .rev()
            .find(|&&start| start <= line)
            .copied()
            .ok_or(LayoutTargetError::MissingSourceForRenderedLine {
                rendered_line: line,
            })?;
        let slot = (measure, staff);
        if events
            .iter()
            .all(|event| event.outcome == Outcome::Restated)
        {
            restatements.push((slot, events));
        } else if events.iter().any(|event| event.outcome == Outcome::Changed) {
            let hidden = events.iter().any(|event| {
                let KernLine::TandemInterpretationLine(tandems) = &document.lines()[event.line_index] else { unreachable!() };
                matches!(&tandems.tokens()[event.column], TandemInterpretation::TimeSignature(signature) if signature.invisible)
                    || matches!(&tandems.tokens()[event.column], TandemInterpretation::MeterSymbol(MeterSymbol::Empty) | TandemInterpretation::FreeMeter)
            });
            if !hidden {
                let remaining = visible.entry(slot).or_default();
                if *remaining > 0 {
                    *remaining -= 1;
                } else {
                    suppressed_groups.push(events);
                }
            }
        }
    }
    // Verovio can draw a numeric restatement while suppressing a lone
    // repeated symbol in that same measure. Give that explicit declaration
    // its glyph before considering symbol-only restatements.
    restatements.sort_by_key(|(_, events)| !events.iter().any(|event| {
        let KernLine::TandemInterpretationLine(tandems) = &document.lines()[event.line_index] else { unreachable!() };
        matches!(&tandems.tokens()[event.column], TandemInterpretation::TimeSignature(signature) if !signature.invisible)
    }));
    let mut suppress = Vec::new();
    for (slot, events) in restatements {
        let remaining = visible.entry(slot).or_default();
        if *remaining > 0 {
            *remaining -= 1;
        } else {
            suppressed_groups.push(events);
        }
    }
    // A symbol-only declaration can change duration even when Verovio draws
    // nothing. Keep that numeric state hidden instead of erasing the event.
    let mut carriers = BTreeMap::new();
    for events in suppressed_groups {
        let first = events.iter().map(|event| event.line_index).min().unwrap();
        let last = events.iter().map(|event| event.line_index).max().unwrap();
        for event in &events {
            suppress.push((event.line_index, event.column));
            let KernLine::TandemInterpretationLine(tandems) = &document.lines()[event.line_index]
            else {
                unreachable!()
            };
            if !matches!(
                tandems.tokens()[event.column],
                TandemInterpretation::MeterSymbol(_)
            ) {
                continue;
            }
            let has_numeric = events.iter().any(|other| {
                if other.column != event.column {
                    return false;
                }
                let KernLine::TandemInterpretationLine(line) = &document.lines()[other.line_index]
                else {
                    unreachable!()
                };
                matches!(
                    line.tokens()[other.column],
                    TandemInterpretation::TimeSignature(_) | TandemInterpretation::FreeMeter
                )
            });
            if has_numeric {
                continue;
            }
            if let Some(time_signature) = score
                .state_after(last, event.column)
                .track
                .meter
                .effective_time_signature()
            {
                let tokens = carriers
                    .entry(first)
                    .or_insert_with(|| vec![TandemInterpretation::Null; tandems.len()]);
                tokens[event.column] =
                    TandemInterpretation::TimeSignature(crate::TimeSignatureTandem {
                        time_signature,
                        invisible: true,
                    });
            }
        }
    }
    drop(score);
    for &(line, column) in &suppress {
        let KernLine::TandemInterpretationLine(tandems) = &mut document.lines_mut()[line] else {
            unreachable!()
        };
        match &mut tandems.tokens_mut()[column] {
            TandemInterpretation::TimeSignature(signature) => signature.invisible = true,
            token @ TandemInterpretation::MeterSymbol(_) => {
                *token = TandemInterpretation::MeterSymbol(MeterSymbol::Empty)
            }
            _ => {}
        }
    }
    for (line, tokens) in carriers.into_iter().rev() {
        let source = document.source(line);
        document.insert_line(
            line,
            KernLine::TandemInterpretationLine(
                crate::TandemInterpretationLine::new(tokens)
                    .expect("homogeneous hidden meter state"),
            ),
            source,
        );
    }
    Ok(suppress.len())
}
