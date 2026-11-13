use std::collections::BTreeMap;

use crate::ast::{Arpeggio, AuxInterval, Fermata, MordentDirection, Ornament, Placed};
use crate::{
    Articulation, Chord, DynamContent, DynamMarks, GraceChord, HairpinBegin, HairpinEnd,
    Instrument, InstrumentKind, InstrumentTextVariant, KernContent, KernDocument, KernLine,
    LayoutText, LocalComment, Ottava, Pedal, SpineContent, SpineManipulator, SpineManipulatorLine,
    SpineType, TandemInterpretation, TandemInterpretationLine, TandemInterpretationLineKind,
    Tuplet,
};

use super::config::AugmentationError;
use super::event::AugmentationEvent;
use super::plan::MusicSite;

pub(super) fn apply_events(
    document: &KernDocument,
    events: &[AugmentationEvent],
) -> Result<KernDocument, AugmentationError> {
    let mut augmented = document.clone();

    for event in events {
        match event {
            AugmentationEvent::Articulation { site, articulation } => {
                apply_articulation(&mut augmented, *site, articulation.clone())?;
            }
            AugmentationEvent::Fermata { site } => {
                apply_fermata(&mut augmented, *site)?;
            }
            AugmentationEvent::Arpeggio { site } => {
                apply_arpeggio(&mut augmented, *site)?;
            }
            AugmentationEvent::Mordent { site } => {
                apply_mordent(&mut augmented, *site)?;
            }
            AugmentationEvent::DynamicLevel { .. }
            | AugmentationEvent::Hairpin { .. }
            | AugmentationEvent::Pedal { .. }
            | AugmentationEvent::Ottava { .. }
            | AugmentationEvent::TupletVisibility { .. }
            | AugmentationEvent::ReferenceMetadata { .. }
            | AugmentationEvent::StaffNames { .. }
            | AugmentationEvent::MetronomeTempo { .. }
            | AugmentationEvent::TextDirection { .. } => {}
        }
    }

    let needs_dynamic_spines = events.iter().any(|event| {
        matches!(
            event,
            AugmentationEvent::DynamicLevel { .. } | AugmentationEvent::Hairpin { .. }
        )
    });
    let dynamic_spine_count = if needs_dynamic_spines {
        document.staff_count().max(1)
    } else {
        0
    };

    let inserted_header = if dynamic_spine_count > 0 {
        append_dynamic_spines(&mut augmented, dynamic_spine_count)?
    } else {
        false
    };

    if dynamic_spine_count > 0 {
        apply_dynamic_events(&mut augmented, events, dynamic_spine_count, inserted_header)?;
    }

    apply_tandem_span_events(&mut augmented, events, dynamic_spine_count, inserted_header)?;
    apply_reference_metadata(&mut augmented, events);
    apply_header_tandems(&mut augmented, events)?;
    Ok(augmented)
}

fn apply_reference_metadata(document: &mut KernDocument, events: &[AugmentationEvent]) {
    for event in events {
        let AugmentationEvent::ReferenceMetadata {
            title,
            composer,
            movement_designation,
        } = event
        else {
            continue;
        };
        if document.metadata.title.is_none() {
            document.metadata.title = title.clone();
        }
        if document.metadata.composer.is_none() {
            document.metadata.composer = composer.clone();
        }
        if document.metadata.movement_designation.is_none() {
            document.metadata.movement_designation = movement_designation.clone();
        }
    }
}

fn apply_header_tandems(
    document: &mut KernDocument,
    events: &[AugmentationEvent],
) -> Result<(), AugmentationError> {
    for event in events {
        match event {
            AugmentationEvent::StaffNames { names }
                if !leading_header_has_kind(document, TandemInterpretationLineKind::Instrument) =>
            {
                insert_header_tandem_line(document, header_insert_start(document), |spine| {
                    names.get(spine.kern_ordinal).map(|name| {
                        TandemInterpretation::Instrument(Instrument {
                            text: sanitize_generated_text(name)
                                .expect("curated staff name is valid")
                                .to_owned(),
                            kind: InstrumentKind::Name(InstrumentTextVariant::Regular),
                        })
                    })
                })?;
            }
            AugmentationEvent::MetronomeTempo { tempo }
                if !leading_header_has_kind(document, TandemInterpretationLineKind::Tempo) =>
            {
                insert_header_tandem_line(document, tempo_insert_index(document), |spine| {
                    matches!(spine.spine_type, SpineType::Kern)
                        .then(|| TandemInterpretation::Tempo(tempo.clone()))
                })?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn apply_articulation(
    document: &mut KernDocument,
    site: MusicSite,
    articulation: Articulation,
) -> Result<(), AugmentationError> {
    let KernLine::Data(contents) = document.lines_mut().get_mut(site.line_index).ok_or(
        AugmentationError::ExpectedDataLine {
            line_index: site.line_index,
        },
    )?
    else {
        return Err(AugmentationError::ExpectedDataLine {
            line_index: site.line_index,
        });
    };
    let Some(SpineContent::Kern(content)) = contents.get_mut(site.spine_index) else {
        return Ok(());
    };

    match content {
        KernContent::Note(note) => {
            let chord = Chord {
                notes: vec![note.clone()],
                modifiers: Default::default(),
            };
            *content = KernContent::Chord(chord);
            if let KernContent::Chord(chord) = content {
                push_articulation(&mut chord.modifiers.articulations, articulation);
            }
        }
        KernContent::Chord(chord) => {
            push_articulation(&mut chord.modifiers.articulations, articulation);
        }
        KernContent::Grace(GraceChord { modifiers, .. }) => {
            push_articulation(&mut modifiers.articulations, articulation);
        }
        KernContent::Null
        | KernContent::Rest(_)
        | KernContent::InvisibleRestSlice(_)
        | KernContent::MultiMeasureRest(_) => {}
    }
    Ok(())
}

fn push_articulation(articulations: &mut Vec<Placed<Articulation>>, articulation: Articulation) {
    if !articulations
        .iter()
        .any(|placed| placed.mark == articulation)
    {
        articulations.push(articulation.into());
    }
}

fn apply_fermata(document: &mut KernDocument, site: MusicSite) -> Result<(), AugmentationError> {
    let KernLine::Data(contents) = document.lines_mut().get_mut(site.line_index).ok_or(
        AugmentationError::ExpectedDataLine {
            line_index: site.line_index,
        },
    )?
    else {
        return Err(AugmentationError::ExpectedDataLine {
            line_index: site.line_index,
        });
    };
    let Some(SpineContent::Kern(content)) = contents.get_mut(site.spine_index) else {
        return Ok(());
    };

    match content {
        KernContent::Note(note) => {
            let chord = Chord {
                notes: vec![note.clone()],
                modifiers: Default::default(),
            };
            *content = KernContent::Chord(chord);
            if let KernContent::Chord(chord) = content {
                chord.modifiers.fermata.get_or_insert(Fermata::default());
            }
        }
        KernContent::Chord(chord) => {
            chord.modifiers.fermata.get_or_insert(Fermata::default());
        }
        KernContent::Grace(GraceChord { modifiers, .. }) => {
            modifiers.fermata.get_or_insert(Fermata::default());
        }
        KernContent::Rest(rest) => {
            rest.fermata.get_or_insert(Fermata::default());
        }
        KernContent::Null
        | KernContent::InvisibleRestSlice(_)
        | KernContent::MultiMeasureRest(_) => {}
    }
    Ok(())
}

fn apply_arpeggio(document: &mut KernDocument, site: MusicSite) -> Result<(), AugmentationError> {
    let KernLine::Data(contents) = document.lines_mut().get_mut(site.line_index).ok_or(
        AugmentationError::ExpectedDataLine {
            line_index: site.line_index,
        },
    )?
    else {
        return Err(AugmentationError::ExpectedDataLine {
            line_index: site.line_index,
        });
    };
    let Some(SpineContent::Kern(KernContent::Chord(chord))) = contents.get_mut(site.spine_index)
    else {
        return Ok(());
    };
    if chord.notes.len() > 1 {
        chord.modifiers.arpeggio.get_or_insert(Arpeggio::StaffLocal);
    }
    Ok(())
}

fn apply_mordent(document: &mut KernDocument, site: MusicSite) -> Result<(), AugmentationError> {
    let KernLine::Data(contents) = document.lines_mut().get_mut(site.line_index).ok_or(
        AugmentationError::ExpectedDataLine {
            line_index: site.line_index,
        },
    )?
    else {
        return Err(AugmentationError::ExpectedDataLine {
            line_index: site.line_index,
        });
    };
    let Some(SpineContent::Kern(content)) = contents.get_mut(site.spine_index) else {
        return Ok(());
    };

    match content {
        KernContent::Note(note) => {
            let chord = Chord {
                notes: vec![note.clone()],
                modifiers: Default::default(),
            };
            *content = KernContent::Chord(chord);
            if let KernContent::Chord(chord) = content {
                push_mordent(&mut chord.modifiers.ornaments);
            }
        }
        KernContent::Chord(chord) => {
            push_mordent(&mut chord.modifiers.ornaments);
        }
        KernContent::Grace(GraceChord { modifiers, .. }) => {
            push_mordent(&mut modifiers.ornaments);
        }
        KernContent::Null
        | KernContent::Rest(_)
        | KernContent::InvisibleRestSlice(_)
        | KernContent::MultiMeasureRest(_) => {}
    }
    Ok(())
}

fn push_mordent(ornaments: &mut Vec<Placed<Ornament>>) {
    if ornaments.is_empty() {
        ornaments.push(
            Ornament::Mordent {
                direction: MordentDirection::Upper,
                interval: Some(AuxInterval::WholeTone),
                doubled: false,
            }
            .into(),
        );
    }
}

fn append_dynamic_spines(
    document: &mut KernDocument,
    dynamic_spine_count: usize,
) -> Result<bool, AugmentationError> {
    let mut inserted_header = false;
    if matches!(document.lines().first(), Some(KernLine::SpineTypeLine(_))) {
        if let Some(KernLine::SpineTypeLine(spine_types)) = document.lines_mut().first_mut() {
            spine_types.extend(std::iter::repeat_n(SpineType::Dynam, dynamic_spine_count));
        }
    } else {
        inserted_header = true;
        let initial_width = document.layout.initial_spine_count as usize;
        let mut spine_types = vec![SpineType::Kern; initial_width];
        spine_types.extend(std::iter::repeat_n(SpineType::Dynam, dynamic_spine_count));
        document.insert_line(0, KernLine::SpineTypeLine(spine_types), None);
    }
    document.layout.initial_spine_count += dynamic_spine_count as u32;

    let start_index = usize::from(matches!(
        document.lines().first(),
        Some(KernLine::SpineTypeLine(_))
    ));
    for line in document.lines_mut().iter_mut().skip(start_index) {
        append_dynamic_nulls(line, dynamic_spine_count)?;
    }

    Ok(inserted_header)
}

fn append_dynamic_nulls(
    line: &mut KernLine,
    dynamic_spine_count: usize,
) -> Result<(), AugmentationError> {
    match line {
        KernLine::SpineTypeLine(_) => {}
        KernLine::TandemInterpretationLine(tandems) => {
            let mut tokens = tandems.tokens().to_vec();
            tokens.extend(std::iter::repeat_n(
                TandemInterpretation::Null,
                dynamic_spine_count,
            ));
            *tandems = TandemInterpretationLine::new(tokens)?;
        }
        KernLine::SpineManipulatorLine(manipulators) => {
            let mut tokens = manipulators.tokens().to_vec();
            let pad = if tokens
                .iter()
                .all(|token| matches!(token, SpineManipulator::Terminate))
            {
                SpineManipulator::Terminate
            } else {
                SpineManipulator::Null
            };
            tokens.extend(std::iter::repeat_n(pad, dynamic_spine_count));
            *manipulators = SpineManipulatorLine::new(tokens)?;
        }
        KernLine::LocalCommentLine(comments) => {
            comments.extend(std::iter::repeat_n(LocalComment::Null, dynamic_spine_count));
        }
        KernLine::Barline(_) => {}
        KernLine::Data(contents) => {
            contents.extend(std::iter::repeat_n(
                SpineContent::Dynam(DynamContent::Null),
                dynamic_spine_count,
            ));
        }
    }
    Ok(())
}

fn apply_dynamic_events(
    document: &mut KernDocument,
    events: &[AugmentationEvent],
    dynamic_spine_count: usize,
    inserted_header: bool,
) -> Result<(), AugmentationError> {
    for event in events {
        match event {
            AugmentationEvent::DynamicLevel { site, level } => {
                set_dynamic_content(
                    document,
                    *site,
                    dynamic_spine_count,
                    inserted_header,
                    DynamContent::Marks(DynamMarks {
                        level: Some(level.clone()),
                        ..DynamMarks::default()
                    }),
                )?;
            }
            AugmentationEvent::Hairpin {
                start,
                end,
                direction,
            } => {
                let mut begin = DynamMarks::default();
                begin.hairpin_mut(*direction).begin = Some(HairpinBegin {
                    single_note: false,
                    provenance: None,
                });
                let mut close = DynamMarks::default();
                close.hairpin_mut(*direction).end = Some(HairpinEnd {
                    to_line_end: false,
                    provenance: None,
                });
                set_dynamic_content(
                    document,
                    *start,
                    dynamic_spine_count,
                    inserted_header,
                    DynamContent::Marks(begin),
                )?;
                set_dynamic_content(
                    document,
                    *end,
                    dynamic_spine_count,
                    inserted_header,
                    DynamContent::Marks(close),
                )?;
            }
            AugmentationEvent::Articulation { .. }
            | AugmentationEvent::Fermata { .. }
            | AugmentationEvent::Arpeggio { .. }
            | AugmentationEvent::Mordent { .. }
            | AugmentationEvent::Pedal { .. }
            | AugmentationEvent::Ottava { .. }
            | AugmentationEvent::TupletVisibility { .. }
            | AugmentationEvent::ReferenceMetadata { .. }
            | AugmentationEvent::StaffNames { .. }
            | AugmentationEvent::MetronomeTempo { .. }
            | AugmentationEvent::TextDirection { .. } => {}
        }
    }
    Ok(())
}

fn set_dynamic_content(
    document: &mut KernDocument,
    site: MusicSite,
    dynamic_spine_count: usize,
    inserted_header: bool,
    content: DynamContent,
) -> Result<(), AugmentationError> {
    let line_index = line_index_after_inserted_header(site.line_index, inserted_header);
    let KernLine::Data(contents) =
        document
            .lines_mut()
            .get_mut(line_index)
            .ok_or(AugmentationError::ExpectedDataLine {
                line_index: site.line_index,
            })?
    else {
        return Err(AugmentationError::ExpectedDataLine {
            line_index: site.line_index,
        });
    };
    let slot = contents
        .len()
        .checked_sub(dynamic_spine_count)
        .and_then(|start| start.checked_add(site.staff_index))
        .filter(|slot| *slot < contents.len())
        .ok_or(AugmentationError::MissingDynamicSlot {
            line_index: site.line_index,
            staff_index: site.staff_index,
        })?;
    contents[slot] = SpineContent::Dynam(content);
    Ok(())
}

fn apply_tandem_span_events(
    document: &mut KernDocument,
    events: &[AugmentationEvent],
    dynamic_spine_count: usize,
    inserted_header: bool,
) -> Result<(), AugmentationError> {
    let mut insertions: BTreeMap<usize, Vec<KernLine>> = BTreeMap::new();
    for event in events {
        let Some((start, end, start_token, end_token)) = tandem_span_event_tokens(event) else {
            continue;
        };
        insertions
            .entry(start.line_index)
            .or_default()
            .push(tandem_line(
                document,
                start,
                start_token,
                dynamic_spine_count,
                inserted_header,
            )?);
        insertions
            .entry(end.line_index.saturating_add(1))
            .or_default()
            .push(tandem_line(
                document,
                end,
                end_token,
                dynamic_spine_count,
                inserted_header,
            )?);
    }
    for event in events {
        let AugmentationEvent::TextDirection { site, text } = event else {
            continue;
        };
        insertions
            .entry(site.line_index)
            .or_default()
            .push(local_comment_line(
                document,
                *site,
                text.clone(),
                dynamic_spine_count,
                inserted_header,
            )?);
    }

    let header_offset = usize::from(inserted_header);
    let mut inserted = 0;
    for (original_index, mut lines) in insertions {
        let index = original_index + header_offset + inserted;
        inserted += lines.len();
        for (offset, line) in lines.drain(..).enumerate() {
            document.insert_line(index + offset, line, None);
        }
    }
    Ok(())
}

fn tandem_span_event_tokens(
    event: &AugmentationEvent,
) -> Option<(
    MusicSite,
    MusicSite,
    TandemInterpretation,
    TandemInterpretation,
)> {
    match event {
        AugmentationEvent::Pedal { start, end } => Some((
            *start,
            *end,
            TandemInterpretation::Pedal(Pedal { active: true }),
            TandemInterpretation::Pedal(Pedal { active: false }),
        )),
        AugmentationEvent::Ottava { start, end, kind } => Some((
            *start,
            *end,
            TandemInterpretation::Ottava(Ottava {
                kind: *kind,
                active: true,
            }),
            TandemInterpretation::Ottava(Ottava {
                kind: *kind,
                active: false,
            }),
        )),
        AugmentationEvent::TupletVisibility { run } => Some((
            run.first_site(),
            run.last_site(),
            TandemInterpretation::Tuplet(Tuplet { show_number: false }),
            TandemInterpretation::Tuplet(Tuplet { show_number: true }),
        )),
        AugmentationEvent::Articulation { .. }
        | AugmentationEvent::Fermata { .. }
        | AugmentationEvent::Arpeggio { .. }
        | AugmentationEvent::Mordent { .. }
        | AugmentationEvent::DynamicLevel { .. }
        | AugmentationEvent::Hairpin { .. }
        | AugmentationEvent::ReferenceMetadata { .. }
        | AugmentationEvent::StaffNames { .. }
        | AugmentationEvent::MetronomeTempo { .. }
        | AugmentationEvent::TextDirection { .. } => None,
    }
}

fn tandem_line(
    document: &KernDocument,
    site: MusicSite,
    token: TandemInterpretation,
    dynamic_spine_count: usize,
    inserted_header: bool,
) -> Result<KernLine, AugmentationError> {
    let line_index = line_index_after_inserted_header(site.line_index, inserted_header);
    let width = match document.lines().get(line_index) {
        Some(KernLine::Data(contents)) => contents.len(),
        _ => {
            return Err(AugmentationError::ExpectedDataLine {
                line_index: site.line_index,
            })
        }
    };
    let mut tokens = vec![TandemInterpretation::Null; width];
    if site.spine_index < width.saturating_sub(dynamic_spine_count) {
        tokens[site.spine_index] = token;
    }
    Ok(KernLine::TandemInterpretationLine(
        TandemInterpretationLine::new(tokens)?,
    ))
}

fn local_comment_line(
    document: &KernDocument,
    site: MusicSite,
    text: LayoutText,
    dynamic_spine_count: usize,
    inserted_header: bool,
) -> Result<KernLine, AugmentationError> {
    let line_index = line_index_after_inserted_header(site.line_index, inserted_header);
    let width = match document.lines().get(line_index) {
        Some(KernLine::Data(contents)) => contents.len(),
        _ => {
            return Err(AugmentationError::ExpectedDataLine {
                line_index: site.line_index,
            })
        }
    };
    let mut tokens = vec![LocalComment::Null; width];
    if site.spine_index < width.saturating_sub(dynamic_spine_count) {
        tokens[site.spine_index] = LocalComment::LayoutText(text);
    }
    Ok(KernLine::LocalCommentLine(tokens))
}

#[derive(Debug, Clone)]
struct HeaderSpine {
    spine_type: SpineType,
    kern_ordinal: usize,
}

fn header_spines(document: &KernDocument) -> Vec<HeaderSpine> {
    let spine_types = match document.lines().first() {
        Some(KernLine::SpineTypeLine(spine_types)) => spine_types.clone(),
        _ => vec![SpineType::Kern; document.layout.initial_spine_count as usize],
    };
    let mut kern_ordinal = 0;
    spine_types
        .into_iter()
        .map(|spine_type| {
            let ordinal = kern_ordinal;
            if matches!(&spine_type, SpineType::Kern) {
                kern_ordinal += 1;
            }
            HeaderSpine {
                spine_type,
                kern_ordinal: ordinal,
            }
        })
        .collect()
}

fn insert_header_tandem_line<F>(
    document: &mut KernDocument,
    index: usize,
    mut token_for_spine: F,
) -> Result<(), AugmentationError>
where
    F: FnMut(HeaderSpine) -> Option<TandemInterpretation>,
{
    let tokens = header_spines(document)
        .into_iter()
        .map(|spine| token_for_spine(spine).unwrap_or(TandemInterpretation::Null))
        .collect::<Vec<_>>();
    document.insert_line(
        index.min(document.lines().len()),
        KernLine::TandemInterpretationLine(TandemInterpretationLine::new(tokens)?),
        None,
    );
    Ok(())
}

fn leading_header_has_kind(document: &KernDocument, kind: TandemInterpretationLineKind) -> bool {
    document
        .leading_tandem_header_lines()
        .iter()
        .any(|line| matches!(line, KernLine::TandemInterpretationLine(tandems) if tandems.kind() == kind))
}

fn header_insert_start(document: &KernDocument) -> usize {
    usize::from(matches!(
        document.lines().first(),
        Some(KernLine::SpineTypeLine(_))
    ))
}

fn tempo_insert_index(document: &KernDocument) -> usize {
    let header = document.leading_tandem_header_range();
    let meter_index = document.lines()[header.clone()]
        .iter()
        .rposition(|line| matches!(line, KernLine::TandemInterpretationLine(tandems) if tandems.kind() == TandemInterpretationLineKind::TimeSignature));
    meter_index.map_or(header.end, |offset| header.start + offset + 1)
}

fn sanitize_generated_text(value: &str) -> Option<&str> {
    (!value.is_empty()
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, ' ' | ',' | '.' | '-')
        }))
    .then_some(value)
}

fn line_index_after_inserted_header(original_index: usize, inserted_header: bool) -> usize {
    if inserted_header {
        original_index + 1
    } else {
        original_index
    }
}
