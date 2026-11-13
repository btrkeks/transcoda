//! Clef/key restatements are label events only when the selected SVG draws them.
use super::{has_class, parse_measure_id, LayoutTargetError};
use crate::{
    document::Framing,
    score::{Outcome, Score, StateKey},
    KernDocument, KernLine, TandemInterpretation,
};
use std::collections::BTreeMap;

pub fn project_staff_visibility(
    document: &mut KernDocument,
    svg: &str,
    rendered_sources: &[Option<usize>],
) -> Result<usize, LayoutTargetError> {
    let score = Score::resolve(document, Framing::Excerpt).map_err(|error| {
        LayoutTargetError::MalformedSvg {
            message: error.to_string(),
        }
    })?;
    let last_data = document
        .lines()
        .iter()
        .rposition(|line| matches!(line, KernLine::Data(_)))
        .unwrap_or(usize::MAX);
    if !score.events().iter().any(|event| {
        matches!(event.key, StateKey::Clef | StateKey::KeySignature)
            && (event.outcome == Outcome::Restated || event.line_index > last_data)
            && document.lines()[..event.line_index]
                .iter()
                .any(|line| matches!(line, KernLine::Data(_)))
    }) {
        return Ok(0);
    }
    let xml = roxmltree::Document::parse(svg).map_err(|error| LayoutTargetError::MalformedSvg {
        message: error.to_string(),
    })?;
    let first_data = document
        .lines()
        .iter()
        .position(|line| matches!(line, KernLine::Data(_)))
        .unwrap_or(usize::MAX);
    let mut measures = Vec::new();
    let mut visible = BTreeMap::new();
    for system in xml.descendants().filter(|node| has_class(*node, "system")) {
        for (measure_index, measure) in system
            .descendants()
            .filter(|node| has_class(*node, "measure"))
            .enumerate()
        {
            let rendered_line = parse_measure_id(measure.attribute("id").unwrap_or_default())?;
            let source = rendered_sources
                .get(rendered_line - 1)
                .ok_or(LayoutTargetError::RenderedLineOutOfRange { rendered_line })?
                .or_else(|| (measures.is_empty() && rendered_line == 1).then_some(0))
                .ok_or(LayoutTargetError::MissingSourceForRenderedLine { rendered_line })?;
            measures.push(source);
            for (staff_index, staff) in measure
                .children()
                .filter(|node| has_class(*node, "staff"))
                .enumerate()
            {
                for (key, class) in [(StateKey::Clef, "clef"), (StateKey::KeySignature, "keySig")] {
                    let count = staff
                        .descendants()
                        .filter(|node| has_class(*node, class) && drawn(*node))
                        .count();
                    // A system's automatic header is not an authored restatement.
                    let automatic = measure_index == 0;
                    visible.insert(
                        (source, staff_index as u32 + 1, key),
                        count.saturating_sub(usize::from(automatic)),
                    );
                }
            }
        }
    }
    measures.sort_unstable();
    let mut groups = BTreeMap::<_, Vec<_>>::new();
    for event in score
        .events()
        .iter()
        .filter(|event| matches!(event.key, StateKey::Clef | StateKey::KeySignature))
    {
        if event.line_index < first_data {
            continue;
        }
        let layer = score.topology().layers_at(event.line_index)[event.column];
        let Some(staff) = score
            .topology()
            .track(score.topology().track_of(layer))
            .staff
        else {
            continue;
        };
        groups
            .entry((event.line_index, staff, event.key))
            .or_default()
            .push(event);
    }
    // Changes are retained, including cautionary copies at the preceding system
    // end. They must not lend a glyph to a suppressed restatement in the measure.
    let mut restatements = Vec::new();
    for ((line_index, staff, key), events) in groups {
        let source = document.source(line_index).unwrap_or(line_index + 1);
        let measure = measures
            .iter()
            .rev()
            .find(|&&start| start <= source)
            .copied()
            .ok_or(LayoutTargetError::MissingSourceForRenderedLine {
                rendered_line: source,
            })?;
        let slot = (measure, staff, key);
        if line_index <= last_data && events.iter().any(|event| event.outcome == Outcome::Changed) {
            let remaining = visible.entry(slot).or_default();
            *remaining = remaining.saturating_sub(1);
        } else {
            restatements.push((slot, events));
        }
    }
    let mut suppress = Vec::new();
    for (slot, events) in restatements {
        let remaining = visible.entry(slot).or_default();
        if *remaining > 0 {
            *remaining -= 1;
        } else {
            suppress.extend(events.iter().map(|event| (event.line_index, event.column)));
        }
    }
    drop(score);
    for &(line, column) in &suppress {
        let KernLine::TandemInterpretationLine(tandems) = &mut document.lines_mut()[line] else {
            unreachable!()
        };
        tandems.tokens_mut()[column] = TandemInterpretation::Null;
    }
    Ok(suppress.len())
}

fn drawn(node: roxmltree::Node<'_, '_>) -> bool {
    !node.ancestors().any(|ancestor| {
        ancestor.attribute("visibility") == Some("hidden")
            || ancestor.attribute("display") == Some("none")
            || ancestor
                .attribute("style")
                .is_some_and(|style| style.replace(' ', "").contains("display:none"))
    }) && node
        .descendants()
        .any(|child| child.is_element() && child.tag_name().name() == "use")
}
