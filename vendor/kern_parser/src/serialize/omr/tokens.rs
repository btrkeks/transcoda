use super::options::{
    MarkPlacementStyle, RestPlacementStyle, SlurPlacementStyle, TiePlacementStyle,
};
use crate::serialize::omr::{
    ArticulationStyle, BeamStyle, BowingStyle, ChordDurationStyle, ChordNoteOrder, CrossStaffStyle,
    InvisibleNoteStyle, InvisibleRestModifierStyle, InvisibleRestStyle, NoteModifierStyle,
    NullDataTokenStyle, OmrSerializeOptions, OrnamentStyle, StemDirectionStyle, TrillWavyLineStyle,
    VisibilityMarkerStyle,
};
use crate::types::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ArpeggioStyle {
    Preserve,
    #[default]
    CollapseCrossStaff,
}

pub(crate) fn serialize_kern_content(
    content: &KernContent,
    options: &OmrSerializeOptions,
) -> String {
    serialize_kern_content_with_arpeggio_style(content, options, ArpeggioStyle::default())
}

pub(crate) fn serialize_kern_content_preserving_arpeggio_scope(
    content: &KernContent,
    options: &OmrSerializeOptions,
) -> String {
    serialize_kern_content_with_arpeggio_style(content, options, ArpeggioStyle::Preserve)
}

fn serialize_kern_content_with_arpeggio_style(
    content: &KernContent,
    options: &OmrSerializeOptions,
    arpeggio_style: ArpeggioStyle,
) -> String {
    match content {
        KernContent::Null => match options.null_data_token_style {
            NullDataTokenStyle::Dot => ".".to_owned(),
            NullDataTokenStyle::Empty => String::new(),
        },
        KernContent::Rest(rest) => serialize_rest(rest, options),
        KernContent::InvisibleRestSlice(slice) => match options.invisible_rest_style {
            InvisibleRestStyle::DurationlessSlices => {
                serialize_invisible_rest_slice(slice, options)
            }
            InvisibleRestStyle::DurationBearing => {
                panic!("durationless invisible rest slice reached duration-bearing serializer")
            }
        },
        KernContent::MultiMeasureRest(rest) => format!("MM{}r", rest.measure_count),
        KernContent::Note(note) if omits_invisible(&note.modifiers, options) => String::new(),
        KernContent::Note(note) => serialize_heads(
            &[Head {
                lead: serialize_duration(&note.duration),
                pitch: note.pitch,
                accidental: note.accidental.as_ref(),
                modifiers: &note.modifiers,
            }],
            None,
            options,
            arpeggio_style,
        ),
        KernContent::Chord(chord) => serialize_chord(chord, options, arpeggio_style),
        KernContent::Grace(grace_chord) => {
            serialize_grace_chord(grace_chord, options, arpeggio_style)
        }
    }
}

fn serialize_invisible_rest_slice(
    slice: &InvisibleRestSlice,
    options: &OmrSerializeOptions,
) -> String {
    let mut output = "ryy".to_owned();
    if let Some(fermata) = slice.fermata.filter(|_| {
        matches!(
            options.invisible_rest_modifier_style,
            InvisibleRestModifierStyle::Preserve
        )
    }) {
        output.push_str(&serialize_fermata(fermata, options.mark_placement_style));
    }
    output
}

fn serialize_rest(rest: &Rest, options: &OmrSerializeOptions) -> String {
    let every_marker = options.mark_placement_style.writes_every_marker();
    let (opens, closes): (Vec<_>, Vec<_>) = rest
        .phrases
        .iter()
        .filter(|_| every_marker)
        .partition(|phrase| matches!(phrase, Phrase::Open { .. }));
    // A rest takes phrase opens only before its duration.
    let mut output = String::new();
    write_phrases(&mut output, opens);
    output.push_str(&serialize_duration(&rest.duration));
    output.push('r');
    if let Some(placement) = &rest.placement {
        if matches!(options.rest_placement_style, RestPlacementStyle::Preserve) {
            output.push_str(&serialize_pitch(placement.position));
        }
    }
    output.push_str(serialize_cross_staff_placement(
        rest.cross_staff,
        options.cross_staff_style,
    ));
    if rest.invisible
        && matches!(
            options.visibility_marker_style,
            VisibilityMarkerStyle::Preserve
        )
    {
        output.push_str("yy");
    }
    if let Some(fermata) = rest.fermata.filter(|_| {
        !rest.invisible
            || matches!(
                options.invisible_rest_modifier_style,
                InvisibleRestModifierStyle::Preserve
            )
    }) {
        output.push_str(&serialize_fermata(fermata, options.mark_placement_style));
    }
    if rest.breath.is_some() && every_marker {
        output.push(',');
    }
    output.push_str(&serialize_slurs(&rest.slurs, options.slur_placement_style));
    write_phrases(&mut output, closes);
    output
}

fn write_phrases<'a>(output: &mut String, phrases: impl IntoIterator<Item = &'a Phrase>) {
    for phrase in phrases {
        let (bracket, elision) = match phrase {
            Phrase::Open { elision } => ('{', elision),
            Phrase::Close { elision } => ('}', elision),
        };
        output.extend(std::iter::repeat_n('&', elision.depth.into()));
        output.push(bracket);
    }
}

/// One notehead subtoken: the text before its pitch (duration, grace kind)
/// and the marks it owns.
struct Head<'a> {
    lead: String,
    pitch: Pitch,
    accidental: Option<&'a Accidental>,
    modifiers: &'a NoteModifiers,
}

fn serialize_chord(
    chord: &Chord,
    options: &OmrSerializeOptions,
    arpeggio_style: ArpeggioStyle,
) -> String {
    let ordered_notes = match options.chord_note_order {
        ChordNoteOrder::Source => chord.notes.iter().collect::<Vec<_>>(),
        ChordNoteOrder::Pitch => chord.notes_by_pitch(),
    };
    let heads = ordered_notes
        .into_iter()
        .filter(|note| !omits_invisible(&note.modifiers, options))
        .enumerate()
        .map(|(index, note)| Head {
            lead: if index == 0
                || matches!(options.chord_duration_style, ChordDurationStyle::Repeated)
            {
                serialize_duration(&note.duration)
            } else {
                String::new()
            },
            pitch: note.pitch,
            accidental: note.accidental.as_ref(),
            modifiers: &note.modifiers,
        })
        .collect::<Vec<_>>();
    serialize_heads(&heads, Some(&chord.modifiers), options, arpeggio_style)
}

fn serialize_grace_chord(
    chord: &GraceChord,
    options: &OmrSerializeOptions,
    arpeggio_style: ArpeggioStyle,
) -> String {
    let ordered_notes = match options.chord_note_order {
        ChordNoteOrder::Source => chord.notes.iter().collect::<Vec<_>>(),
        ChordNoteOrder::Pitch => chord.notes_by_pitch(),
    };
    let first_kind = ordered_notes.first().map(|note| note.kind.clone());
    let heads = ordered_notes
        .into_iter()
        .filter(|note| !omits_invisible(&note.modifiers, options))
        .enumerate()
        .map(|(index, note)| {
            let lead = if index == 0
                || matches!(options.chord_duration_style, ChordDurationStyle::Repeated)
            {
                let duration = match &note.duration {
                    Some(duration) => serialize_duration(duration),
                    None if options.mark_placement_style.writes_every_marker() => String::new(),
                    // The Transcoda targets spell every grace note with a duration.
                    None => "8".to_owned(),
                };
                duration + serialize_grace_kind(&note.kind)
            } else if first_kind.as_ref() != Some(&note.kind) {
                serialize_grace_kind(&note.kind).to_owned()
            } else {
                String::new()
            };
            Head {
                lead,
                pitch: note.pitch,
                accidental: note.accidental.as_ref(),
                modifiers: &note.modifiers,
            }
        })
        .collect::<Vec<_>>();
    serialize_heads(&heads, Some(&chord.modifiers), options, arpeggio_style)
}

/// Writes the heads as space-separated subtokens. The chord's own marks go
/// after the last head.
fn serialize_heads(
    heads: &[Head],
    chord: Option<&ChordModifiers>,
    options: &OmrSerializeOptions,
    arpeggio_style: ArpeggioStyle,
) -> String {
    let mut output = String::new();
    for (index, head) in heads.iter().enumerate() {
        if index > 0 {
            output.push(' ');
        }
        output.push_str(&head.lead);
        write_head(&mut output, head, options);
        if index + 1 == heads.len() {
            if let Some(modifiers) = chord {
                write_chord_modifiers(&mut output, modifiers, options, arpeggio_style);
            }
        }
        write_user_signifiers(&mut output, head.modifiers, options, Spot::EndOfSubtoken);
    }
    output
}

#[derive(PartialEq)]
enum Spot {
    AfterAccidental,
    EndOfSubtoken,
}

/// The standard targets write every user signifier last in its subtoken. A
/// signifier stops a later `<`/`>` from attaching, and an `N` before a tie or
/// slur would link it (`N[`). The Transcoda targets write only the editorial
/// accidentals `Z` and `j`, right after the accidental.
fn write_user_signifiers(
    output: &mut String,
    modifiers: &NoteModifiers,
    options: &OmrSerializeOptions,
    spot: Spot,
) {
    let every_marker = options.mark_placement_style.writes_every_marker();
    let own_spot = if every_marker {
        Spot::EndOfSubtoken
    } else {
        Spot::AfterAccidental
    };
    if spot != own_spot {
        return;
    }
    for sign in &modifiers.user_signifiers {
        if every_marker || sign.is_editorial_accidental() {
            output.push_str(&sign.marker);
        }
    }
}

fn write_head(output: &mut String, head: &Head, options: &OmrSerializeOptions) {
    let preserve_modifiers = matches!(options.note_modifier_style, NoteModifierStyle::Preserve);
    let preserve_visibility = matches!(
        options.visibility_marker_style,
        VisibilityMarkerStyle::Preserve
    );
    let modifiers = head.modifiers;
    output.push_str(&serialize_pitch(head.pitch));
    if let Some(accidental) = head.accidental {
        output.push_str(serialize_accidental(accidental));
    }
    if preserve_modifiers {
        if modifiers.forced_accidental.is_some() {
            output.push('X');
        }
        if modifiers.invisible_accidental && preserve_visibility {
            output.push('y');
        }
    }
    output.push_str(serialize_cross_staff_placement(
        modifiers.cross_staff,
        options.cross_staff_style,
    ));
    // After the cross-staff signifier, so that render lowering still finds
    // `<`/`>` right after `X`/`y`.
    if modifiers.editorial_interpretation.is_some()
        && options.mark_placement_style.writes_every_marker()
    {
        output.push('x');
    }
    write_user_signifiers(output, modifiers, options, Spot::AfterAccidental);
    if preserve_modifiers {
        output.push_str(&serialize_ties(
            &modifiers.ties,
            options.tie_placement_style,
        ));
        if modifiers.invisible
            && preserve_visibility
            && matches!(options.invisible_note_style, InvisibleNoteStyle::Preserve)
        {
            output.push_str("yy");
        }
    }
}

fn omits_invisible(modifiers: &NoteModifiers, options: &OmrSerializeOptions) -> bool {
    modifiers.invisible && matches!(options.invisible_note_style, InvisibleNoteStyle::Omit)
}

fn serialize_cross_staff_placement(
    placement: Option<CrossStaffPlacement>,
    style: CrossStaffStyle,
) -> &'static str {
    if matches!(style, CrossStaffStyle::Omit) {
        return "";
    }

    match placement {
        Some(CrossStaffPlacement::Above) => ">",
        Some(CrossStaffPlacement::Below) => "<",
        None => "",
    }
}

fn serialize_grace_kind(kind: &GraceKind) -> &'static str {
    match kind {
        GraceKind::Acciaccatura => "q",
        GraceKind::AccentedGrace => "qq",
        GraceKind::Groupetto => "Q",
        GraceKind::Appoggiatura => "P",
        GraceKind::PostAppoggiatura => "p",
    }
}

fn write_chord_modifiers(
    output: &mut String,
    modifiers: &ChordModifiers,
    options: &OmrSerializeOptions,
    arpeggio_style: ArpeggioStyle,
) {
    let mark_placement_style = options.mark_placement_style;
    if matches!(options.articulation_style, ArticulationStyle::Preserve) {
        for articulation in &modifiers.articulations {
            output.push(match articulation.mark {
                Articulation::Staccato => '\'',
                Articulation::Pizzicato => '"',
                Articulation::Staccatissimo => '`',
                Articulation::Tenuto => '~',
                Articulation::Accent => '^',
                Articulation::Generic => 'z',
            });
            output.push_str(mark_signifier(articulation, mark_placement_style));
        }
    }
    output.push_str(&serialize_ornaments(
        &modifiers.ornaments,
        options.ornament_style,
        options.trill_wavy_line_style,
        mark_placement_style,
    ));
    if let Some(fermata) = modifiers.fermata {
        output.push_str(&serialize_fermata(fermata, mark_placement_style));
    }
    if mark_placement_style.writes_every_marker() {
        if let Some(breath) = &modifiers.breath {
            output.push(',');
            output.push_str(mark_signifier(breath, mark_placement_style));
        }
        if let Some(harmonic) = &modifiers.harmonic {
            output.push('o');
            output.push_str(mark_signifier(harmonic, mark_placement_style));
        }
    }
    if matches!(options.bowing_style, BowingStyle::Preserve) {
        if let Some(bowing) = &modifiers.bowing {
            output.push_str(match bowing.mark {
                Bowing::UpBow => "v",
                Bowing::DownBow => "u",
            });
            output.push_str(mark_signifier(bowing, mark_placement_style));
        }
    }
    if matches!(options.stem_direction_style, StemDirectionStyle::Preserve) {
        if let Some(stem_direction) = &modifiers.stem_direction {
            output.push_str(match stem_direction {
                StemDirection::Up => "/",
                StemDirection::Down => "\\",
            });
        }
    }
    if let Some(arpeggio) = modifiers.arpeggio {
        output.push_str(match (arpeggio, arpeggio_style) {
            (Arpeggio::CrossStaff, ArpeggioStyle::Preserve) => "::",
            (Arpeggio::StaffLocal | Arpeggio::CrossStaff, _) => ":",
        });
    }
    if matches!(options.note_modifier_style, NoteModifierStyle::Preserve) {
        output.push_str(&serialize_slurs(
            &modifiers.slurs,
            options.slur_placement_style,
        ));
    }
    if mark_placement_style.writes_every_marker() {
        write_phrases(output, &modifiers.phrases);
    }
    if matches!(options.beam_style, BeamStyle::Preserve) {
        modifiers.beams.write_kern_beam_spelling(output);
    }
}

fn mark_signifier<T>(placed: &Placed<T>, placement_style: MarkPlacementStyle) -> &'static str {
    match (placement_style, placed.placement) {
        (MarkPlacementStyle::Preserve, Some(MarkPlacement::Above)) => ">",
        (MarkPlacementStyle::Preserve, Some(MarkPlacement::Below)) => "<",
        _ => "",
    }
}

fn serialize_duration(duration: &Duration) -> String {
    let mut output = match duration.value {
        DurationValue::Reciprocal(value) => value.to_string(),
        DurationValue::Rational {
            numerator,
            denominator,
        } => format!("{numerator}%{denominator}"),
        DurationValue::Breve => "0".to_owned(),
        DurationValue::Long => "00".to_owned(),
        DurationValue::Maxima => "000".to_owned(),
    };
    output.push_str(&".".repeat(duration.augmentation_dots.into()));
    output
}

fn serialize_pitch(pitch: Pitch) -> String {
    let character = match pitch.step {
        DiatonicStep::A => 'A',
        DiatonicStep::B => 'B',
        DiatonicStep::C => 'C',
        DiatonicStep::D => 'D',
        DiatonicStep::E => 'E',
        DiatonicStep::F => 'F',
        DiatonicStep::G => 'G',
    };

    if pitch.octave >= 4 {
        character
            .to_ascii_lowercase()
            .to_string()
            .repeat((pitch.octave - 3).max(1) as usize)
    } else {
        character
            .to_string()
            .repeat((4 - pitch.octave).max(1) as usize)
    }
}

fn serialize_accidental(accidental: &Accidental) -> &'static str {
    match accidental {
        Accidental::DoubleFlat => "--",
        Accidental::Flat => "-",
        Accidental::Natural => "n",
        Accidental::Sharp => "#",
        Accidental::DoubleSharp => "##",
    }
}

fn serialize_ties(ties: &[Tie], placement_style: TiePlacementStyle) -> String {
    ties.iter()
        .map(|tie| {
            let marker = match tie {
                Tie::Start => "[",
                Tie::Continue => "_",
                Tie::End => "]",
                Tie::DisjunctStart => "[[",
                Tie::DisjunctContinue => "__",
                Tie::DisjunctEnd => "]]",
                Tie::LinkedStart => "N[",
                Tie::LinkedContinue => "N_",
                Tie::LinkedEnd => "N]",
                Tie::PlacedOpen {
                    phase, resolution, ..
                } => match (phase, resolution) {
                    (TieOpenPhase::Start, TieResolution::Ordinary) => "[",
                    (TieOpenPhase::Continue, TieResolution::Ordinary) => "_",
                    (TieOpenPhase::Start, TieResolution::Disjunct) => "[[",
                    (TieOpenPhase::Continue, TieResolution::Disjunct) => "__",
                    (TieOpenPhase::Start, TieResolution::Linked) => "N[",
                    (TieOpenPhase::Continue, TieResolution::Linked) => "N_",
                },
            };
            let placement = match (placement_style, tie.placement()) {
                (TiePlacementStyle::Preserve, Some(TiePlacement::Above)) => ">",
                (TiePlacementStyle::Preserve, Some(TiePlacement::Below)) => "<",
                _ => "",
            };
            format!("{marker}{placement}")
        })
        .collect()
}

fn serialize_ornaments(
    ornaments: &[Placed<Ornament>],
    ornament_style: OrnamentStyle,
    trill_wavy_line_style: TrillWavyLineStyle,
    placement_style: MarkPlacementStyle,
) -> String {
    if matches!(ornament_style, OrnamentStyle::OmitAll) {
        return String::new();
    }

    let mut ornaments = ornaments
        .iter()
        .filter(|ornament| {
            !matches!(
                (ornament_style, &ornament.mark),
                (OrnamentStyle::OmitConcludingTurn, Ornament::ConcludingTurn)
            )
        })
        .collect::<Vec<_>>();
    if !placement_style.writes_every_marker() {
        ornaments.sort_by_key(|ornament| ornament_rank(&ornament.mark));
    }
    ornaments
        .into_iter()
        .map(|ornament| {
            render_ornament(&ornament.mark, trill_wavy_line_style)
                + mark_signifier(ornament, placement_style)
        })
        .collect()
}

fn ornament_rank(ornament: &Ornament) -> u8 {
    match ornament {
        Ornament::Trill {
            interval: AuxInterval::WholeTone,
            ..
        } => 0,
        Ornament::Trill {
            interval: AuxInterval::Semitone,
            ..
        } => 1,
        Ornament::Mordent {
            direction: MordentDirection::Upper,
            interval: Some(AuxInterval::WholeTone),
            ..
        } => 2,
        Ornament::Mordent {
            direction: MordentDirection::Upper,
            interval: Some(AuxInterval::Semitone),
            ..
        } => 3,
        Ornament::Mordent {
            direction: MordentDirection::Upper,
            interval: None,
            ..
        } => 4,
        Ornament::Mordent {
            direction: MordentDirection::Lower,
            interval: Some(AuxInterval::WholeTone),
            ..
        } => 5,
        Ornament::Mordent {
            direction: MordentDirection::Lower,
            interval: Some(AuxInterval::Semitone),
            ..
        } => 6,
        Ornament::Mordent {
            direction: MordentDirection::Lower,
            interval: None,
            ..
        } => 7,
        Ornament::Turn(Turn {
            form: TurnForm::Regular,
            ..
        }) => 8,
        Ornament::Turn(Turn {
            form: TurnForm::Inverted,
            ..
        }) => 9,
        Ornament::ConcludingTurn => 10,
        Ornament::Generic => 11,
    }
}

fn render_ornament(ornament: &Ornament, trill_wavy_line_style: TrillWavyLineStyle) -> String {
    match ornament {
        Ornament::Trill {
            interval: AuxInterval::WholeTone,
            line,
        } => render_trill('T', *line, trill_wavy_line_style),
        Ornament::Trill {
            interval: AuxInterval::Semitone,
            line,
        } => render_trill('t', *line, trill_wavy_line_style),
        Ornament::Mordent {
            direction,
            interval,
            doubled,
        } => render_mordent(*direction, *interval, *doubled),
        Ornament::Turn(turn) => render_turn(turn),
        Ornament::ConcludingTurn => "R".to_owned(),
        Ornament::Generic => "O".to_owned(),
    }
}

fn render_trill(
    marker: char,
    line: TrillLine,
    trill_wavy_line_style: TrillWavyLineStyle,
) -> String {
    let marker_count = if matches!(trill_wavy_line_style, TrillWavyLineStyle::Omit) {
        1
    } else {
        match line {
            TrillLine::None => 1,
            TrillLine::Start => 2,
            TrillLine::Continuation => 3,
        }
    };
    marker.to_string().repeat(marker_count)
}

fn render_mordent(
    direction: MordentDirection,
    interval: Option<AuxInterval>,
    doubled: bool,
) -> String {
    let single = match (direction, interval) {
        (MordentDirection::Upper, Some(AuxInterval::WholeTone)) => "M",
        (MordentDirection::Upper, Some(AuxInterval::Semitone)) => "m",
        (MordentDirection::Upper, None) => "Mm",
        (MordentDirection::Lower, Some(AuxInterval::WholeTone)) => "W",
        (MordentDirection::Lower, Some(AuxInterval::Semitone)) => "w",
        (MordentDirection::Lower, None) => "Ww",
    };
    if doubled {
        single.repeat(2)
    } else {
        single.to_owned()
    }
}

fn render_turn(turn: &Turn) -> String {
    let mut output = String::new();
    if matches!(turn.timing, TurnTiming::OnNote) {
        output.push('s');
    }
    output.push(match turn.form {
        TurnForm::Regular => 'S',
        TurnForm::Inverted => '$',
    });
    if let Some(intervals) = turn.intervals {
        match turn.form {
            TurnForm::Regular => {
                output.push(render_aux_interval(intervals.upper));
                output.push(render_aux_interval(intervals.lower));
            }
            TurnForm::Inverted => {
                output.push(render_aux_interval(intervals.lower));
                output.push(render_aux_interval(intervals.upper));
            }
        }
    }
    output
}

fn render_aux_interval(interval: AuxInterval) -> char {
    match interval {
        AuxInterval::WholeTone => 'S',
        AuxInterval::Semitone => 's',
    }
}

fn serialize_slurs(slurs: &[Slur], placement_style: SlurPlacementStyle) -> String {
    let mut output = String::new();
    for slur in slurs {
        let (marker, elided, depth, count, linked, placement) = match slur {
            Slur::Open {
                elided,
                depth,
                count,
                placement,
                linked,
            } => ('(', elided, depth, count, linked, *placement),
            Slur::Close {
                elided,
                depth,
                count,
                linked,
            } => (')', elided, depth, count, linked, None),
        };
        for _ in 0..*count {
            if *elided {
                output.push_str(&"&".repeat((*depth).into()));
            }
            if *linked {
                output.push('N');
            }
            output.push(marker);
            if matches!(placement_style, SlurPlacementStyle::Preserve) {
                output.push_str(slur_placement_signifier(placement));
            }
        }
    }
    output
}

fn serialize_fermata(fermata: Fermata, placement_style: MarkPlacementStyle) -> String {
    let mut output = ";".to_owned();
    if matches!(placement_style, MarkPlacementStyle::Preserve) {
        output.push_str(match fermata.placement {
            Some(FermataPlacement::Above) => ">",
            Some(FermataPlacement::Below) => "<",
            None => "",
        });
    }
    output
}

fn slur_placement_signifier(placement: Option<SlurPlacement>) -> &'static str {
    match placement {
        Some(SlurPlacement::Above) => ">",
        Some(SlurPlacement::Below) => "<",
        None => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omr_serialization_preserves_elided_slur_depth() {
        let shallow = KernContent::Chord("4c&(".try_into().unwrap());
        let deep = KernContent::Chord("4d&&)".try_into().unwrap());

        assert_eq!(
            serialize_kern_content(&shallow, &OmrSerializeOptions::default()),
            "4c&("
        );
        assert_eq!(
            serialize_kern_content(&deep, &OmrSerializeOptions::default()),
            "4d&&)"
        );
    }

    #[test]
    fn standard_kern_token_serialization_preserves_slur_placement() {
        let rest = Rest::try_from("(>4r").unwrap();
        assert_eq!(
            crate::serialize::token::kern_token_string(&KernContent::Rest(rest)),
            "4r(>"
        );
    }

    #[test]
    fn rest_placement_is_emitted_only_when_preserved() {
        let rest = KernContent::Rest("16rGG<yy;".try_into().unwrap());

        assert_eq!(
            serialize_kern_content(&rest, &OmrSerializeOptions::default()),
            "16r<yy;"
        );
        let options = OmrSerializeOptions {
            rest_placement_style: RestPlacementStyle::Preserve,
            ..OmrSerializeOptions::default()
        };
        assert_eq!(serialize_kern_content(&rest, &options), "16rGG<yy;");
        assert_eq!(
            crate::serialize::token::kern_token_string(&rest),
            "16rGG<yy;"
        );
    }

    #[test]
    fn omr_ornament_rendering_omits_trill_wavy_line_spelling() {
        let whole_tone = KernContent::Chord("4cTT".try_into().unwrap());
        let semitone = KernContent::Chord("4ctt".try_into().unwrap());
        let whole_tone_continuation = KernContent::Chord("4dTTT".try_into().unwrap());
        let semitone_continuation = KernContent::Chord("4dttt".try_into().unwrap());

        assert_eq!(
            serialize_kern_content(&whole_tone, &OmrSerializeOptions::default()),
            "4cT"
        );
        assert_eq!(
            serialize_kern_content(&semitone, &OmrSerializeOptions::default()),
            "4ct"
        );
        assert_eq!(
            serialize_kern_content(&whole_tone_continuation, &OmrSerializeOptions::default()),
            "4dT"
        );
        assert_eq!(
            serialize_kern_content(&semitone_continuation, &OmrSerializeOptions::default()),
            "4dt"
        );
    }

    #[test]
    fn omr_rest_rendering_preserves_cross_staff_markers() {
        let rest = KernContent::Rest("4.r>yy".try_into().unwrap());

        assert_eq!(
            serialize_kern_content(&rest, &OmrSerializeOptions::default()),
            "4.r>yy"
        );

        let options = OmrSerializeOptions {
            cross_staff_style: CrossStaffStyle::Omit,
            ..OmrSerializeOptions::default()
        };
        assert_eq!(serialize_kern_content(&rest, &options), "4.ryy");
    }
}
