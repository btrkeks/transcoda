//! Exact analysis for ordinary `**kern` tie chains over a resolved score.

use std::collections::BTreeMap;

use serde::Serialize;
use thiserror::Error;

use super::{LooseEnd, LooseEndSide, Spanner};

use crate::{
    analysis::{score_line_durations, DataLineDuration, DataLineDurationError},
    ast::note::pitch_to_sounding_semitone,
    measure::MusicalDuration,
    score::{LayerId, LayerOrigin, Score, SplitSide, Topology},
    types::{Accidental, DiatonicStep, NoteModifiers, Pitch, TiePhase, TieResolution},
    KernContent, KernLine, KernPitchedNote, SpineContent, SpineType,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TieChain {
    pub id: String,
    pub staff: u32,
    pub pitch: String,
    pub resolution: TieChainResolution,
    pub status: TieChainStatus,
    pub members: Vec<TieChainMember>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TieChainResolution {
    Ordinary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TieChainStatus {
    Complete,
    Partial,
    Ambiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TieChainMember {
    pub phase: TieChainPhase,
    /// One-based source line.
    pub source_line: usize,
    /// Zero-based Humdrum field index.
    pub field_index: usize,
    /// Zero-based source-subtoken index; `None` for single-note tokens.
    pub source_chord_note_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TieChainPhase {
    Start,
    Continue,
    End,
}

impl From<TiePhase> for TieChainPhase {
    fn from(phase: TiePhase) -> Self {
        match phase {
            TiePhase::Start => Self::Start,
            TiePhase::Continue => Self::Continue,
            TiePhase::End => Self::End,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TieChainFinding {
    pub code: &'static str,
    pub source_line: usize,
    pub field_index: usize,
    pub message: String,
    pub loose_end: Option<LooseEnd>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TieChainAnalysis {
    pub chains: Vec<TieChain>,
    pub findings: Vec<TieChainFinding>,
}

#[derive(Debug, Error)]
pub enum TieChainAnalysisError {
    #[error(transparent)]
    Timing(#[from] DataLineDurationError),
}

#[derive(Debug, Clone)]
struct TieEvent {
    member: TieChainMember,
    onset: MusicalDuration,
    end: MusicalDuration,
    staff: u32,
    lineage: LayerId,
    sounding_pitch: i16,
    pitch_label: String,
    starts: bool,
    continues: bool,
    ends: bool,
}

#[derive(Debug, Clone)]
struct OpenChain {
    chain: TieChain,
    end: MusicalDuration,
    /// Onset of the chain's latest member: a chain that a note at this onset
    /// opened or continued never competes with that note's own close.
    last_onset: MusicalDuration,
    lineage: LayerId,
    sounding_pitch: i16,
    reported: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ClaimKey {
    staff: u32,
    sounding_pitch: i16,
    onset: MusicalDuration,
}

/// Resolve ordinary ties using Verovio's staff/pitch/exact-time rule.
///
/// Disjunct and explicitly linked tie markers are intentionally ignored.
pub fn analyze_tie_chains(score: &Score<'_>) -> Result<TieChainAnalysis, TieChainAnalysisError> {
    analyze_timed_tie_chains(score, score_line_durations(score)?)
}

/// [`analyze_tie_chains`] with line durations the caller already computed.
pub(crate) fn analyze_timed_tie_chains(
    score: &Score<'_>,
    durations: Vec<DataLineDuration>,
) -> Result<TieChainAnalysis, TieChainAnalysisError> {
    let document = score.document();
    let topology = score.topology();
    let lineages = lineages(topology);
    let mut duration_by_line = BTreeMap::new();
    for duration in durations {
        duration_by_line.insert(duration.line_index, duration.duration);
    }

    let mut onset = MusicalDuration::zero();
    let mut events = Vec::new();
    for (line_index, line) in document.lines().iter().enumerate() {
        let KernLine::Data(contents) = line else {
            continue;
        };
        let source_line = document.source_line_number(line_index);
        for (field_index, (&layer, content)) in topology
            .layers_at(line_index)
            .iter()
            .zip(contents)
            .enumerate()
        {
            let (SpineType::Kern, Some(staff), SpineContent::Kern(content)) = (
                topology.spine_type(layer),
                topology.staff_of(layer),
                content,
            ) else {
                continue;
            };
            let field = TieField {
                source_line,
                field_index,
                staff,
                lineage: lineages[layer.index()],
                onset,
            };
            match content {
                KernContent::Note(note) => {
                    push_note_event(&mut events, KernPitchedNote::Regular(note), &field, None)
                }
                KernContent::Chord(chord) => {
                    for (index, note) in chord.notes.iter().enumerate() {
                        push_note_event(
                            &mut events,
                            KernPitchedNote::Regular(note),
                            &field,
                            (chord.notes.len() > 1).then_some(index),
                        );
                    }
                }
                KernContent::Grace(grace) => {
                    for (index, note) in grace.notes.iter().enumerate() {
                        push_note_event(
                            &mut events,
                            KernPitchedNote::Grace(note),
                            &field,
                            (grace.notes.len() > 1).then_some(index),
                        );
                    }
                }
                KernContent::Null
                | KernContent::Rest(_)
                | KernContent::InvisibleRestSlice(_)
                | KernContent::MultiMeasureRest(_) => {}
            }
        }
        onset += duration_by_line
            .get(&line_index)
            .copied()
            .unwrap_or_else(MusicalDuration::zero);
    }

    Ok(resolve_events(events))
}

/// The voice each layer continues, indexed by [`LayerId`]: a split's left
/// child continues its parent, a merge its leftmost input. A tie close
/// prefers an open from its own voice.
fn lineages(topology: &Topology) -> Vec<LayerId> {
    let mut lineages = Vec::with_capacity(topology.layer_count());
    for layer in topology.layer_ids() {
        let lineage = match &topology.layer(layer).origin {
            LayerOrigin::SplitOf(parent) if topology.split_side(layer) == Some(SplitSide::Left) => {
                lineages[parent.index()]
            }
            LayerOrigin::MergeOf(inputs) => lineages[inputs[0].index()],
            _ => layer,
        };
        lineages.push(lineage);
    }
    lineages
}

/// Where a data field sits, shared by every note it holds.
struct TieField {
    source_line: usize,
    field_index: usize,
    staff: u32,
    lineage: LayerId,
    onset: MusicalDuration,
}

fn push_note_event(
    events: &mut Vec<TieEvent>,
    note: KernPitchedNote<'_>,
    field: &TieField,
    source_chord_note_index: Option<usize>,
) {
    // Grace notes take no metric time, so a tie leaving one ends at its own onset.
    let (modifiers, duration) = match note {
        KernPitchedNote::Regular(note) => (
            &note.modifiers,
            MusicalDuration::from_duration(&note.duration),
        ),
        KernPitchedNote::Grace(note) => (&note.modifiers, MusicalDuration::zero()),
    };
    // A compound `][` ends the tie that arrives and starts the next one, as
    // Verovio draws it: the note closes one chain and opens another.
    let starts = has_ordinary_phase(modifiers, TiePhase::Start);
    let continues = has_ordinary_phase(modifiers, TiePhase::Continue);
    let ends = has_ordinary_phase(modifiers, TiePhase::End);
    if !starts && !continues && !ends {
        return;
    }
    let sounding_pitch = pitch_to_sounding_semitone(note.pitch(), note.accidental());
    events.push(TieEvent {
        member: TieChainMember {
            phase: if continues {
                TieChainPhase::Continue
            } else if ends {
                TieChainPhase::End
            } else {
                TieChainPhase::Start
            },
            source_line: field.source_line,
            field_index: field.field_index,
            source_chord_note_index,
        },
        onset: field.onset,
        end: field.onset + duration,
        staff: field.staff,
        lineage: field.lineage,
        sounding_pitch,
        pitch_label: format_note_pitch(note.pitch(), note.accidental()),
        starts,
        continues,
        ends,
    });
}

fn has_ordinary_phase(modifiers: &NoteModifiers, phase: TiePhase) -> bool {
    modifiers
        .ties
        .iter()
        .any(|tie| tie.resolution() == TieResolution::Ordinary && tie.phase() == phase)
}

fn resolve_events(mut events: Vec<TieEvent>) -> TieChainAnalysis {
    events.sort_by_key(|event| {
        (
            event.onset,
            event.member.source_line,
            event.member.field_index,
            event.member.source_chord_note_index,
        )
    });
    let mut open = Vec::<OpenChain>::new();
    let mut chains = Vec::new();
    let mut findings = Vec::new();
    let mut index = 0;

    while index < events.len() {
        let onset = events[index].onset;
        let group_end = events[index..]
            .iter()
            .position(|event| event.onset != onset)
            .map_or(events.len(), |offset| index + offset);
        let mut claims = BTreeMap::<ClaimKey, TieChainMember>::new();

        for event in &events[index..group_end] {
            if event.continues || event.ends {
                let key = claim_key(event);
                let selected = select_open_chain(&open, event);
                if let Some(selected) = selected {
                    let mut chain = open.remove(selected);
                    chain.chain.members.push(event.member);
                    claims.entry(key).or_insert(event.member);
                    if event.continues {
                        chain.end = event.end;
                        chain.last_onset = event.onset;
                        chain.lineage = event.lineage;
                        open.push(chain);
                    } else {
                        chain.chain.status = if chain.reported {
                            TieChainStatus::Partial
                        } else {
                            TieChainStatus::Complete
                        };
                        chains.push(chain.chain);
                    }
                } else {
                    findings.push(unmatched_member_finding(event));
                    let chain = chain_from_event(event, event.member.phase, true);
                    if event.continues {
                        open.push(chain);
                    } else {
                        chains.push(chain.chain);
                    }
                }
            }
            if event.starts {
                open.push(chain_from_event(event, TieChainPhase::Start, false));
            }
        }

        let mut claimed_duplicates = Vec::new();
        for (open_index, chain) in open.iter().enumerate() {
            let key = ClaimKey {
                staff: chain.chain.staff,
                sounding_pitch: chain.sounding_pitch,
                onset: chain.end,
            };
            // A chain a note at this onset opened or continued (a grace `][`
            // or `_`, which ends where it starts) is the claim's own tie.
            if chain.end == onset && chain.last_onset != onset && claims.contains_key(&key) {
                claimed_duplicates.push(open_index);
            }
        }
        for open_index in claimed_duplicates.into_iter().rev() {
            let mut chain = open.remove(open_index);
            let key = ClaimKey {
                staff: chain.chain.staff,
                sounding_pitch: chain.sounding_pitch,
                onset: chain.end,
            };
            let member = claims[&key];
            let open_member = chain.chain.members[0];
            findings.push(TieChainFinding {
                code: "tie.competing-open",
                loose_end: None,
                source_line: open_member.source_line,
                field_index: open_member.field_index,
                message: format!(
                    "competing tie open on {} at line {}, field {}; the tie {} at line {}, field {} can pair with only one open",
                    chain.chain.pitch,
                    open_member.source_line,
                    open_member.field_index + 1,
                    phase_label(member.phase),
                    member.source_line,
                    member.field_index + 1,
                ),
            });
            chain.chain.status = TieChainStatus::Ambiguous;
            chains.push(chain.chain);
        }

        index = group_end;
    }

    for mut chain in open {
        if !chain.reported {
            let member = chain.chain.members[0];
            findings.push(TieChainFinding {
                code: "tie.unclosed-open",
                loose_end: Some(LooseEnd {
                    spanner: Spanner::Tie,
                    side: LooseEndSide::Open,
                    staff: chain.chain.staff,
                    pitch: Some(chain.chain.pitch.clone()),
                }),
                source_line: member.source_line,
                field_index: member.field_index,
                message: format!(
                    "tie open on {} at line {}, field {} has no matching continuation or end",
                    chain.chain.pitch,
                    member.source_line,
                    member.field_index + 1,
                ),
            });
        }
        chain.chain.status = TieChainStatus::Partial;
        chains.push(chain.chain);
    }

    chains.sort_by(|left, right| chain_sort_key(left).cmp(&chain_sort_key(right)));
    findings.sort_by_key(|finding| (finding.source_line, finding.field_index));
    TieChainAnalysis { chains, findings }
}

fn select_open_chain(open: &[OpenChain], event: &TieEvent) -> Option<usize> {
    let candidates = open
        .iter()
        .enumerate()
        .filter(|(_, chain)| {
            chain.chain.staff == event.staff
                && chain.sounding_pitch == event.sounding_pitch
                && chain.end == event.onset
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    candidates
        .iter()
        .copied()
        .find(|index| open[*index].lineage == event.lineage)
        .or_else(|| candidates.first().copied())
}

fn chain_from_event(event: &TieEvent, phase: TieChainPhase, reported: bool) -> OpenChain {
    let mut member = event.member;
    member.phase = phase;
    OpenChain {
        chain: TieChain {
            id: chain_id(member),
            staff: event.staff,
            pitch: event.pitch_label.clone(),
            resolution: TieChainResolution::Ordinary,
            status: TieChainStatus::Partial,
            members: vec![member],
        },
        end: event.end,
        last_onset: event.onset,
        lineage: event.lineage,
        sounding_pitch: event.sounding_pitch,
        reported,
    }
}

fn unmatched_member_finding(event: &TieEvent) -> TieChainFinding {
    TieChainFinding {
        code: "tie.unmatched-continuation-or-end",
        loose_end: Some(LooseEnd {
            spanner: Spanner::Tie,
            side: LooseEndSide::Close,
            staff: event.staff,
            pitch: Some(event.pitch_label.clone()),
        }),
        source_line: event.member.source_line,
        field_index: event.member.field_index,
        message: format!(
            "tie {} on {} at line {}, field {} has no matching open ending at its onset",
            phase_label(event.member.phase),
            event.pitch_label,
            event.member.source_line,
            event.member.field_index + 1,
        ),
    }
}

fn claim_key(event: &TieEvent) -> ClaimKey {
    ClaimKey {
        staff: event.staff,
        sounding_pitch: event.sounding_pitch,
        onset: event.onset,
    }
}

fn chain_id(member: TieChainMember) -> String {
    let mut id = format!("tie-L{}F{}", member.source_line, member.field_index + 1);
    if let Some(source_chord_note_index) = member.source_chord_note_index {
        id.push_str(&format!("S{}", source_chord_note_index + 1));
    }
    id
}

fn chain_sort_key(chain: &TieChain) -> (usize, usize, Option<usize>, &str) {
    let member = chain.members[0];
    (
        member.source_line,
        member.field_index,
        member.source_chord_note_index,
        chain.id.as_str(),
    )
}

fn phase_label(phase: TieChainPhase) -> &'static str {
    match phase {
        TieChainPhase::Start => "open",
        TieChainPhase::Continue => "continuation",
        TieChainPhase::End => "end",
    }
}

fn format_note_pitch(pitch: Pitch, accidental: Option<&Accidental>) -> String {
    let character = match pitch.step {
        DiatonicStep::A => 'A',
        DiatonicStep::B => 'B',
        DiatonicStep::C => 'C',
        DiatonicStep::D => 'D',
        DiatonicStep::E => 'E',
        DiatonicStep::F => 'F',
        DiatonicStep::G => 'G',
    };
    let mut output = if pitch.octave >= 4 {
        character
            .to_ascii_lowercase()
            .to_string()
            .repeat((pitch.octave - 3).max(1) as usize)
    } else {
        character
            .to_string()
            .repeat((4 - pitch.octave).max(1) as usize)
    };
    output.push_str(match accidental {
        None | Some(Accidental::Natural) => "",
        Some(Accidental::DoubleFlat) => "--",
        Some(Accidental::Flat) => "-",
        Some(Accidental::Sharp) => "#",
        Some(Accidental::DoubleSharp) => "##",
    });
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{document::Framing, KernDocument};

    fn analyze(input: &str) -> TieChainAnalysis {
        let document = KernDocument::parse(input).unwrap();
        analyze_tie_chains(&Score::resolve(&document, Framing::Excerpt).unwrap()).unwrap()
    }

    #[test]
    fn resolves_balanced_chain_with_exact_members() {
        let analysis = analyze("**kern\n4c[\n4c_\n4c]\n*-\n");

        assert!(analysis.findings.is_empty());
        assert_eq!(analysis.chains.len(), 1);
        assert_eq!(analysis.chains[0].status, TieChainStatus::Complete);
        assert_eq!(
            analysis.chains[0]
                .members
                .iter()
                .map(|member| (member.phase, member.source_line))
                .collect::<Vec<_>>(),
            vec![
                (TieChainPhase::Start, 2),
                (TieChainPhase::Continue, 3),
                (TieChainPhase::End, 4),
            ]
        );
    }

    #[test]
    fn duplicate_same_staff_voice_open_matches_current_lineage_first() {
        let analysis = analyze("**kern\n*staff1\n*^\n4f#[\t4f#[\n*v\t*v\n4f#_\n4f#]\n*-\n");

        assert_eq!(analysis.findings.len(), 1);
        assert_eq!(analysis.findings[0].source_line, 4);
        assert_eq!(analysis.findings[0].field_index, 1);
        assert!(analysis.findings[0].message.contains("line 6, field 1"));
        assert_eq!(analysis.chains.len(), 2);
        assert_eq!(analysis.chains[0].status, TieChainStatus::Complete);
        assert_eq!(analysis.chains[1].status, TieChainStatus::Ambiguous);
    }

    #[test]
    fn exact_time_and_staff_are_required() {
        let timing = analyze("**kern\n2c[\n4c\n4c]\n*-\n");
        assert_eq!(timing.findings.len(), 2);

        let staves = analyze("**kern\t**kern\n*staff1\t*staff2\n4c[\t4r\n.\t4c]\n*-\t*-\n");
        assert_eq!(staves.findings.len(), 2);
    }

    #[test]
    fn preserves_source_chord_note_indexes() {
        let analysis = analyze("**kern\n4c 4e[\n4c 4e]\n*-\n");

        assert!(analysis.findings.is_empty());
        assert_eq!(
            analysis.chains[0].members[0].source_chord_note_index,
            Some(1)
        );
        assert_eq!(analysis.chains[0].id, "tie-L2F1S2");
    }

    #[test]
    fn tie_members_address_source_subtokens_when_chords_are_not_pitch_sorted() {
        let analysis =
            analyze("**kern\n*staff1\n*^\n*\t*^\n4cc[\t4f[\t4a[\n*v\t*v\t*v\n4cc] 4f] 4a]\n*-\n");

        assert!(analysis.findings.is_empty());
        assert_eq!(analysis.chains.len(), 3);
        let endpoints = analysis
            .chains
            .iter()
            .map(|chain| {
                (
                    chain.pitch.as_str(),
                    chain.members[0].source_chord_note_index,
                    chain.members[1].source_chord_note_index,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            endpoints,
            vec![
                ("cc", None, Some(0)),
                ("f", None, Some(1)),
                ("a", None, Some(2))
            ]
        );
    }

    #[test]
    fn ignores_nonordinary_tie_resolutions() {
        let analysis = analyze("**kern\n4c[[\n4c]]\n*-\n");

        assert!(analysis.chains.is_empty());
        assert!(analysis.findings.is_empty());
    }

    #[test]
    fn timing_failure_does_not_fabricate_tie_findings() {
        let document = KernDocument::parse("**kern\t**kern\n2c[\t4e\n4d\t4f\n*-\t*-\n").unwrap();

        assert!(matches!(
            analyze_tie_chains(&Score::resolve(&document, Framing::Excerpt).unwrap()),
            Err(TieChainAnalysisError::Timing(_))
        ));
    }

    #[test]
    fn explicit_accidental_controls_sounding_pitch_without_key_inference() {
        let natural = analyze("**kern\n*k[f#]\n4f[\n4f]\n*-\n");
        assert!(natural.findings.is_empty());

        let mismatch = analyze("**kern\n*k[f#]\n4f#[\n4f]\n*-\n");
        assert_eq!(mismatch.findings.len(), 2);
    }

    #[test]
    fn resolves_supported_kern_pitches_outside_midi_range() {
        let low = analyze("**kern\n4CCCCCC[\n4CCCCCC]\n*-\n");
        let high = analyze("**kern\n4aaaaaa[\n4aaaaaa]\n*-\n");

        assert!(low.findings.is_empty());
        assert_eq!(low.chains.len(), 1);
        assert_eq!(low.chains[0].status, TieChainStatus::Complete);
        assert!(high.findings.is_empty());
        assert_eq!(high.chains.len(), 1);
        assert_eq!(high.chains[0].status, TieChainStatus::Complete);
    }

    #[test]
    fn grace_note_tie_resolves_into_principal_at_same_onset() {
        let analysis = analyze(
            "**kern\t**kern\n*staff2\t*staff1\n[8qBBB-/\t[8qeeee/\n6BBB-] 6BB-\t6eee 6eeee]\n*-\t*-\n",
        );

        assert!(analysis.findings.is_empty(), "{:?}", analysis.findings);
        assert_eq!(analysis.chains.len(), 2);
        assert!(analysis
            .chains
            .iter()
            .all(|chain| chain.status == TieChainStatus::Complete));
        let bass = analysis
            .chains
            .iter()
            .find(|chain| chain.pitch == "BBB-")
            .unwrap();
        assert_eq!(
            bass.members
                .iter()
                .map(|member| (
                    member.phase,
                    member.source_line,
                    member.source_chord_note_index
                ))
                .collect::<Vec<_>>(),
            vec![
                (TieChainPhase::Start, 3, None),
                (TieChainPhase::End, 4, Some(0)),
            ]
        );
    }

    #[test]
    fn unmatched_grace_note_tie_open_is_reported() {
        let analysis = analyze("**kern\n[8qc\n4d\n*-\n");

        assert_eq!(analysis.findings.len(), 1);
        assert_eq!(analysis.findings[0].code, "tie.unclosed-open");
        assert_eq!(analysis.findings[0].source_line, 2);
    }

    #[test]
    fn multiple_continuations_claim_multiple_eligible_opens() {
        let analysis = analyze("**kern\n*staff1\n*^\n4c[\t4c[\n4c_\t4c_\n4c]\t4c]\n*-\t*-\n");

        assert!(analysis.findings.is_empty());
        assert_eq!(analysis.chains.len(), 2);
        assert!(analysis
            .chains
            .iter()
            .all(|chain| chain.status == TieChainStatus::Complete));
    }

    #[test]
    fn a_compound_close_and_open_ends_one_chain_and_starts_the_next() {
        let analysis = analyze("**kern\n2c[\n2c][\n2c]\n*-\n");

        assert!(analysis.findings.is_empty());
        let members = analysis
            .chains
            .iter()
            .map(|chain| {
                (
                    chain.status,
                    chain
                        .members
                        .iter()
                        .map(|member| (member.phase, member.source_line))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            members,
            vec![
                (
                    TieChainStatus::Complete,
                    vec![(TieChainPhase::Start, 2), (TieChainPhase::End, 3)]
                ),
                (
                    TieChainStatus::Complete,
                    vec![(TieChainPhase::Start, 3), (TieChainPhase::End, 4)]
                ),
            ]
        );
    }

    #[test]
    fn a_compound_close_and_open_at_the_end_leaves_its_own_tie_open() {
        // The page continues the second tie overleaf: its loose end is the
        // compound note, where Verovio hangs the terminal tie.
        let analysis = analyze("**kern\n2c[\n2c][\n*-\n");

        assert_eq!(analysis.findings.len(), 1);
        assert_eq!(analysis.findings[0].code, "tie.unclosed-open");
        assert_eq!(
            (
                analysis.findings[0].source_line,
                analysis.findings[0].field_index
            ),
            (3, 0)
        );
        assert_eq!(analysis.chains[0].status, TieChainStatus::Complete);
        assert_eq!(analysis.chains[1].id, "tie-L3F1");
        assert_eq!(analysis.chains[1].status, TieChainStatus::Partial);
    }

    #[test]
    fn a_compound_close_and_open_in_a_new_subspine_closes_its_parent_voice_tie() {
        // Mazurka Op. 56 No. 2, page 10: `2d[` in a chord, then `2.d][` in the
        // right subspine of a split, open at the page's end.
        let analysis =
            analyze("**kern\n*M3/4\n=1\n2.dd 2.d[\n=2\n*^\n2.a\t2.d][\n=3\t=3\n*-\t*-\n");

        assert_eq!(
            analysis
                .findings
                .iter()
                .map(|finding| (finding.code, finding.source_line, finding.field_index))
                .collect::<Vec<_>>(),
            vec![("tie.unclosed-open", 7, 1)]
        );
        assert_eq!(analysis.chains.len(), 2);
        assert_eq!(analysis.chains[0].status, TieChainStatus::Complete);
        assert_eq!(
            analysis.chains[0]
                .members
                .iter()
                .map(|member| (member.phase, member.source_line, member.field_index))
                .collect::<Vec<_>>(),
            vec![(TieChainPhase::Start, 4, 0), (TieChainPhase::End, 7, 1)]
        );
        assert_eq!(analysis.chains[1].id, "tie-L7F2");
    }

    #[test]
    fn a_grace_note_that_closes_and_reopens_never_competes_with_its_own_tie() {
        // The grace note ends where it starts: the tie it opens ends at the
        // onset its own close claims, and is still a loose end of its own.
        let open_at_end = analyze("**kern\n1c[\n8qc][\n*-\n");
        assert_eq!(
            open_at_end
                .findings
                .iter()
                .map(|finding| (finding.code, finding.source_line))
                .collect::<Vec<_>>(),
            vec![("tie.unclosed-open", 3)]
        );
        assert_eq!(open_at_end.chains[0].status, TieChainStatus::Complete);
        assert_eq!(open_at_end.chains[1].status, TieChainStatus::Partial);

        let closed = analyze("**kern\n2c[\n8qc][\n2c]\n*-\n");
        assert!(closed.findings.is_empty(), "{:?}", closed.findings);
        assert!(closed
            .chains
            .iter()
            .all(|chain| chain.status == TieChainStatus::Complete));

        let continued = analyze("**kern\n2c[\n8qc_\n2c]\n*-\n");
        assert!(continued.findings.is_empty(), "{:?}", continued.findings);
        assert_eq!(continued.chains.len(), 1);
    }

    #[test]
    fn an_earlier_open_the_grace_reopen_cannot_claim_is_still_reported() {
        // Two opens arrive at the grace note's onset; it closes only one.
        let analysis = analyze("**kern\t**kern\n*staff1\t*staff1\n1c[\t1c[\n8qc][\t.\n*-\t*-\n");

        assert!(
            analysis
                .findings
                .iter()
                .any(|finding| finding.source_line == 3 && finding.field_index == 1),
            "{:?}",
            analysis.findings
        );
        assert!(analysis
            .findings
            .iter()
            .all(
                |finding| (finding.source_line, finding.field_index) != (4, 0)
                    || finding.code == "tie.unclosed-open"
            ));
    }
}
