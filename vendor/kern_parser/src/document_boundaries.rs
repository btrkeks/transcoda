use crate::{
    score::{bridge_seam, BridgeError, Score, Snapshot, StateKey},
    KernDocument, KernLine, SpineManipulator, TandemInterpretationLineKind,
};

pub(crate) fn strip_optional_spine_type(lines: &[KernLine]) -> &[KernLine] {
    if matches!(lines.first(), Some(KernLine::SpineTypeLine(_))) {
        &lines[1..]
    } else {
        lines
    }
}

pub(crate) fn strip_final_terminal(mut lines: Vec<KernLine>) -> Vec<KernLine> {
    while lines.last().is_some_and(is_terminal_line) {
        lines.pop();
    }
    lines
}

pub(crate) fn leading_state_header_end(lines: &[KernLine]) -> usize {
    lines
        .iter()
        .position(|line| !is_state_header_line(line))
        .unwrap_or(lines.len())
}

/// Index where a document's body begins: after its optional exclusive
/// interpretation and the clef, key and meter lines heading it.
pub(crate) fn body_start(document: &KernDocument) -> usize {
    let lines = document.lines();
    let offset = lines.len() - strip_optional_spine_type(lines).len();
    offset + leading_state_header_end(&lines[offset..])
}

/// Index of the first of a document's final `*-` lines, or its length.
pub(crate) fn body_end(document: &KernDocument) -> usize {
    let lines = document.lines();
    lines.len()
        - lines
            .iter()
            .rev()
            .take_while(|line| is_terminal_line(line))
            .count()
}

/// State where the body of a document begins.
pub(crate) fn opening_state(score: &Score) -> Snapshot {
    state_entering_body(score, body_start(score.document()))
}

/// State entering line `body_start` of the score's document. A meter
/// declaration group lasts while tandem lines do, so a header meter followed
/// by another tandem, such as `*Xtuplet`, takes effect only where that run
/// ends; the meter is read there.
pub(crate) fn state_entering_body(score: &Score, body_start: usize) -> Snapshot {
    let lines = score.document().lines();
    let meter_settled = body_start
        + lines[body_start..]
            .iter()
            .take_while(|line| matches!(line, KernLine::TandemInterpretationLine(_)))
            .count();
    let opening = score.snapshot_before(body_start);
    opening
        .with(&score.snapshot_before(meter_settled), &[StateKey::Meter])
        .expect("tandem lines keep the spine layout")
}

/// State where the body of a document ends, before its final `*-` lines.
pub(crate) fn closing_state(score: &Score) -> Snapshot {
    score.snapshot_before(body_end(score.document()))
}

/// Lines restating the clef, key and meter the next document declares in its
/// header where they differ from the state the previous one ends in, laid
/// out on the next document's columns. A slot the next document leaves
/// unstated continues.
pub(crate) fn seam_state_lines(
    previous_end: &Snapshot,
    next_start: &Snapshot,
) -> Result<Vec<KernLine>, BridgeError> {
    Ok(
        bridge_seam(previous_end, next_start, &StateKey::STAFF_DEFINITION)?
            .into_iter()
            .map(KernLine::TandemInterpretationLine)
            .collect(),
    )
}

pub(crate) fn has_musical_data(document: &KernDocument) -> bool {
    document
        .lines()
        .iter()
        .any(|line| matches!(line, KernLine::Data(_)))
}

pub(crate) fn is_state_header_line(line: &KernLine) -> bool {
    matches!(
        line,
        KernLine::TandemInterpretationLine(tandems)
            if matches!(
                tandems.kind(),
                TandemInterpretationLineKind::Clef
                    | TandemInterpretationLineKind::KeySignature
                    | TandemInterpretationLineKind::TimeSignature
                    | TandemInterpretationLineKind::MeterSymbol
            )
    )
}

pub(crate) fn is_terminal_line(line: &KernLine) -> bool {
    matches!(
        line,
        KernLine::SpineManipulatorLine(manipulators)
            if manipulators
                .tokens()
                .iter()
                .all(|token| matches!(token, SpineManipulator::Terminate))
    )
}
