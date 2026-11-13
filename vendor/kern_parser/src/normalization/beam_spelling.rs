//! Beam marks respelled from the beams Verovio draws.
//!
//! Verovio's Humdrum import (`HumdrumInput::analyzeLayerBeams`, 6.2.1) reads
//! only the running count of `L` minus `J` in each layer of each measure. A
//! beam group is a run where the count is positive, and a note where the
//! count dips below both neighbours keeps only that many beams to the next
//! note (`@breaksec`). `k` and `K` are ignored, and so is how high the count
//! climbs: `16cLL 16dJJ`, `16cL 16dJ` and `16cLK 16dJk` draw the same beam.
//! Labels write one spelling per drawing: between two notes as many full
//! beams as both durations take (fewer at a secondary break), and `k`/`K` for
//! the rest of a note's depth. Verovio draws no slash on beamed grace notes,
//! so a beamed `q` is written `qq`.
//!
//! Before the import, humlib (`HumdrumFileContent::analyzeKernBeams`) pairs
//! every `J` with the latest open `L` of its spine, page-wide: a pair that
//! reaches a barline becomes a `beamSpan`, an unpaired mark a hanging beam,
//! and a `J` with no open `L` in its layer takes one from another layer. So
//! a spine is respelled only in measures where each of its layers opens and
//! closes all its marks, grace and regular counted together: the running
//! count never goes negative and ends at zero. Respelling such a measure
//! keeps every pair inside it. Other measures keep their marks, and so does
//! a measure layer whose conventional spelling would not draw the same
//! groups and breaks, such as a quarter note inside a beam or a beam that
//! must change depth on a rest.
//!
//! Verovio also reads raw marks in two places the analysis does not model:
//! whether a tuplet bracket hides depends on `L` and `J` at the tuplet's
//! ends (`shouldHideBeamBracket`), and `*tremolo` turns beamed groups into
//! tremolos from their `L`. A measure layer with tuplet notes keeps its
//! marks, and a document with `*tremolo` is left alone.

use std::collections::BTreeMap;

use crate::{
    document::Framing,
    score::{LayerOrigin, Score, SplitSide},
    types::ChordBeams,
    types::DurationValue,
    Duration, GraceKind, KernContent, KernDocument, KernLine, SpineContent, SpineType,
    TandemInterpretation,
};

use super::NormalizationPass;

const EIGHTH_DEPTH: u8 = 1;
/// Invisible rests draw no beams, so they bound no link. Their written value
/// is also not stable: compaction merges `16ryy 16ryy` into `8ryy` after this
/// pass, and a label must respell the same either way.
const INVISIBLE_DEPTH: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CanonicalizeBeamSpelling;

impl NormalizationPass for CanonicalizeBeamSpelling {
    fn normalize(&self, document: &mut KernDocument) {
        canonicalize_beam_spelling(document);
    }
}

pub fn canonicalize_beam_spelling(document: &mut KernDocument) {
    let rewrites = {
        let Ok(score) = Score::resolve(document, Framing::Excerpt) else {
            return;
        };
        plan(&score)
    };
    for rewrite in rewrites {
        let KernLine::Data(contents) = &mut document.lines_mut()[rewrite.line] else {
            continue;
        };
        match &mut contents[rewrite.column] {
            SpineContent::Kern(KernContent::Chord(chord)) => chord.modifiers.beams = rewrite.beams,
            SpineContent::Kern(KernContent::Grace(grace)) => {
                grace.modifiers.beams = rewrite.beams;
                if rewrite.beamed_grace {
                    for note in &mut grace.notes {
                        if note.kind == GraceKind::Acciaccatura {
                            note.kind = GraceKind::AccentedGrace;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stream {
    Regular,
    Grace,
}

/// One data token of a Verovio layer in one measure.
#[derive(Debug, Clone)]
struct Event {
    line: usize,
    column: usize,
    stream: Stream,
    /// Chords and grace chords carry beam marks; rests cannot.
    markable: bool,
    depth: u8,
    beams: ChordBeams,
    acciaccatura: bool,
    tuplet: bool,
}

impl Event {
    fn delta(&self) -> i32 {
        i32::from(self.beams.beam_start_count()) - i32::from(self.beams.beam_end_count())
    }
}

#[derive(Debug)]
struct Rewrite {
    line: usize,
    column: usize,
    beams: ChordBeams,
    beamed_grace: bool,
}

fn plan(score: &Score<'_>) -> Vec<Rewrite> {
    let document = score.document();
    let tremolo = document.lines().iter().any(|line| {
        matches!(line, KernLine::TandemInterpretationLine(tandems)
            if tandems.tokens().iter().any(|tandem| matches!(tandem, TandemInterpretation::Tremolo(_))))
    });
    if tremolo {
        return Vec::new();
    }
    let topology = score.topology();
    let drawn = score.drawn();
    // Verovio numbers layers by subspine: a split's left child and a merge
    // continue their first parent's layer, a split's right child starts one.
    let mut chain: Vec<Option<usize>> = vec![None; topology.layer_count()];
    let mut chains = 0;
    let mut measure: BTreeMap<usize, Segment> = BTreeMap::new();
    let close_measure = |measure: &mut BTreeMap<usize, Segment>, rewrites: &mut Vec<Rewrite>| {
        let unbalanced: Vec<usize> = measure
            .values()
            .filter(|segment| !segment.balanced())
            .map(|segment| segment.spine)
            .collect();
        for segment in std::mem::take(measure).into_values() {
            if !unbalanced.contains(&segment.spine) {
                flush(segment, rewrites);
            }
        }
    };
    let mut rewrites = Vec::new();

    for (line_index, line) in document.lines().iter().enumerate() {
        let layers = topology.layers_at(line_index);
        for &layer in layers {
            if chain[layer.index()].is_some() {
                continue;
            }
            let inherited = match &topology.layer(layer).origin {
                LayerOrigin::SplitOf(parent)
                    if topology.split_side(layer) != Some(SplitSide::Right) =>
                {
                    chain[parent.index()]
                }
                LayerOrigin::MergeOf(inputs) => {
                    inputs.first().and_then(|input| chain[input.index()])
                }
                _ => None,
            };
            chain[layer.index()] = Some(inherited.unwrap_or_else(|| {
                chains += 1;
                chains - 1
            }));
        }
        match line {
            KernLine::Barline(_) => close_measure(&mut measure, &mut rewrites),
            KernLine::Data(contents) => {
                for (column, (&layer, content)) in layers.iter().zip(contents).enumerate() {
                    if *topology.spine_type(layer) != SpineType::Kern {
                        continue;
                    }
                    let SpineContent::Kern(content) = content else {
                        continue;
                    };
                    let depth = |duration| drawn.beam_depth(line_index, column, duration);
                    let event = |stream, markable, depth, beams: &ChordBeams, acciaccatura| Event {
                        line: line_index,
                        column,
                        stream,
                        markable,
                        depth,
                        beams: beams.clone(),
                        acciaccatura,
                        tuplet: content_is_tuplet(content),
                    };
                    let event = match content {
                        KernContent::Null => continue,
                        KernContent::Chord(chord) => {
                            let depth = chord.notes.first().map_or(0, |note| depth(&note.duration));
                            event(Stream::Regular, true, depth, &chord.modifiers.beams, false)
                        }
                        KernContent::Grace(grace) => {
                            // A grace note without a duration is written `8`
                            // (`GraceDuration`), so it takes an eighth's beam.
                            let depth = grace
                                .notes
                                .first()
                                .and_then(|note| note.duration.as_ref())
                                .map_or(EIGHTH_DEPTH, depth);
                            let acciaccatura = grace
                                .notes
                                .iter()
                                .any(|note| note.kind == GraceKind::Acciaccatura);
                            event(
                                Stream::Grace,
                                true,
                                depth,
                                &grace.modifiers.beams,
                                acciaccatura,
                            )
                        }
                        KernContent::Rest(rest) => event(
                            Stream::Regular,
                            false,
                            if rest.invisible {
                                INVISIBLE_DEPTH
                            } else {
                                depth(&rest.duration)
                            },
                            &ChordBeams::None,
                            false,
                        ),
                        KernContent::Note(note) => event(
                            Stream::Regular,
                            false,
                            depth(&note.duration),
                            &ChordBeams::None,
                            false,
                        ),
                        KernContent::InvisibleRestSlice(_) => event(
                            Stream::Regular,
                            false,
                            INVISIBLE_DEPTH,
                            &ChordBeams::None,
                            false,
                        ),
                        KernContent::MultiMeasureRest(_) => {
                            event(Stream::Regular, false, 0, &ChordBeams::None, false)
                        }
                    };
                    let id = chain[layer.index()].expect("every layer at a line has a chain");
                    let segment = measure.entry(id).or_default();
                    segment.spine = topology.track_of(layer).index();
                    segment.events.push(event);
                }
            }
            _ => {}
        }
    }
    close_measure(&mut measure, &mut rewrites);
    rewrites
}

/// A Verovio layer's events in one measure.
#[derive(Debug, Default)]
struct Segment {
    events: Vec<Event>,
    spine: usize,
}

impl Segment {
    /// Whether every mark pairs inside the segment, both under humlib's
    /// matching (grace and regular together) and within each stream: each
    /// running count of `L` minus `J` never goes negative and ends at zero.
    /// A regular `L` closed by a grace `J` balances only the first way.
    fn balanced(&self) -> bool {
        let balanced = |stream: Option<Stream>| {
            let mut state = 0;
            for event in &self.events {
                if stream.is_none_or(|stream| event.stream == stream) {
                    state += event.delta();
                    if state < 0 {
                        return false;
                    }
                }
            }
            state == 0
        };
        balanced(None) && balanced(Some(Stream::Regular)) && balanced(Some(Stream::Grace))
    }
}

fn content_is_tuplet(content: &KernContent) -> bool {
    let tuplet = |duration: &Duration| match duration.value {
        DurationValue::Reciprocal(value) => value != 0 && !value.is_power_of_two(),
        DurationValue::Rational { .. } => true,
        _ => false,
    };
    match content {
        KernContent::Chord(chord) => chord.notes.iter().any(|note| tuplet(&note.duration)),
        KernContent::Grace(grace) => grace
            .notes
            .iter()
            .any(|note| note.duration.as_ref().is_some_and(tuplet)),
        KernContent::Rest(rest) => tuplet(&rest.duration),
        KernContent::Note(note) => tuplet(&note.duration),
        _ => false,
    }
}

fn flush(segment: Segment, rewrites: &mut Vec<Rewrite>) {
    let keep = segment.events.iter().any(|event| event.tuplet);
    let events = segment.events;
    let drawn = analyze(&events, Event::delta);
    let respelled = (!keep).then(|| respell(&events, &drawn)).flatten();
    for (index, event) in events.iter().enumerate() {
        let beams = respelled
            .as_ref()
            .map_or_else(|| event.beams.clone(), |beams| beams[index].clone());
        let beamed_grace =
            event.stream == Stream::Grace && drawn.grace.group[index] != 0 && event.acciaccatura;
        if beams != event.beams || beamed_grace {
            rewrites.push(Rewrite {
                line: event.line,
                column: event.column,
                beams,
                beamed_grace,
            });
        }
    }
}

/// What Verovio draws for one stream: the beam group of every event (0 for
/// none) and the secondary breaks.
#[derive(Debug, PartialEq, Eq)]
struct StreamBeams {
    state: Vec<i32>,
    group: Vec<u32>,
    breaks: BTreeMap<usize, i32>,
}

#[derive(Debug, PartialEq, Eq)]
struct Drawn {
    regular: StreamBeams,
    grace: StreamBeams,
}

impl Drawn {
    fn stream(&self, stream: Stream) -> &StreamBeams {
        match stream {
            Stream::Regular => &self.regular,
            Stream::Grace => &self.grace,
        }
    }
}

/// `analyzeLayerBeams` and `storeBreaksec` for one measure layer.
fn analyze(events: &[Event], delta: impl Fn(&Event) -> i32) -> Drawn {
    let mut regular = vec![0; events.len()];
    let mut grace = vec![0; events.len()];
    for (index, event) in events.iter().enumerate() {
        let (previous_regular, previous_grace) = index
            .checked_sub(1)
            .map_or((0, 0), |previous| (regular[previous], grace[previous]));
        let (own_regular, own_grace) = match event.stream {
            Stream::Regular => (delta(event), 0),
            Stream::Grace => (0, delta(event)),
        };
        regular[index] = previous_regular + own_regular;
        grace[index] = previous_grace + own_grace;
    }
    let min = regular.iter().copied().min().unwrap_or(0);
    if min < 0 {
        regular.iter_mut().for_each(|state| *state -= min);
    }
    if regular.last().is_some_and(|&last| last > 0) {
        for state in regular.iter_mut().rev() {
            if *state == 0 {
                break;
            }
            *state = 0;
        }
    }
    if grace.iter().any(|&state| state < 0) || grace.last().is_some_and(|&last| last != 0) {
        grace.fill(0);
    }
    Drawn {
        regular: stream_beams(events, regular, Stream::Regular),
        grace: stream_beams(events, grace, Stream::Grace),
    }
}

fn stream_beams(events: &[Event], state: Vec<i32>, stream: Stream) -> StreamBeams {
    let mut group = vec![0; state.len()];
    let mut counter = 1;
    for index in 0..state.len() {
        if state[index] != 0 {
            group[index] = counter;
        } else if index > 0 && state[index - 1] != 0 {
            group[index] = counter;
            counter += 1;
        }
    }
    let mut breaks = BTreeMap::new();
    for members in members(events, &group, stream).values() {
        for window in members.windows(3) {
            let [before, at, after] = [window[0], window[1], window[2]];
            if state[at] < state[before] && state[at] < state[after] {
                breaks.insert(at, state[at]);
            }
        }
    }
    StreamBeams {
        state,
        group,
        breaks,
    }
}

/// The events of `stream` in each group, in order.
fn members(events: &[Event], group: &[u32], stream: Stream) -> BTreeMap<u32, Vec<usize>> {
    let mut members: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (index, event) in events.iter().enumerate() {
        if group[index] != 0 && event.stream == stream {
            members.entry(group[index]).or_default().push(index);
        }
    }
    members
}

fn base_link(events: &[Event], left: usize, right: usize) -> i32 {
    i32::from(events[left].depth.min(events[right].depth))
}

/// The conventional spelling with the same groups and breaks, or `None` when
/// no conventional spelling draws them. A break's value matters even at the
/// depth both notes take: it decides which side a partial beam goes.
fn respell(events: &[Event], drawn: &Drawn) -> Option<Vec<ChordBeams>> {
    let mut target = [vec![0; events.len()], vec![0; events.len()]];
    for (slot, stream) in [Stream::Regular, Stream::Grace].into_iter().enumerate() {
        let beams = drawn.stream(stream);
        let mut links = BTreeMap::new();
        for members in members(events, &beams.group, stream).values() {
            if members.len() < 2 {
                return None;
            }
            let mut previous = 0;
            for pair in members.windows(2) {
                let link = if !events[pair[0]].markable {
                    previous
                } else if let Some(&value) = beams.breaks.get(&pair[0]) {
                    value
                } else {
                    base_link(events, pair[0], pair[1])
                };
                if link <= 0 {
                    return None;
                }
                links.insert(pair[0], link);
                previous = link;
            }
        }
        // Each event leaves its own stream at its link (0 outside a group or
        // at a group's last note); the other stream's events carry it.
        let mut state = 0;
        for (index, event) in events.iter().enumerate() {
            if event.stream == stream {
                state = links.get(&index).copied().unwrap_or(0);
            }
            target[slot][index] = state;
        }
    }

    let mut respelled = Vec::with_capacity(events.len());
    for (index, event) in events.iter().enumerate() {
        let slot = match event.stream {
            Stream::Regular => 0,
            Stream::Grace => 1,
        };
        let right = target[slot][index];
        let left = index
            .checked_sub(1)
            .map_or(0, |previous| target[slot][previous]);
        if !event.markable {
            if right != left {
                return None;
            }
            respelled.push(ChordBeams::None);
            continue;
        }
        let (mut partial_left, mut partial_right) = (0, 0);
        let full = left.max(right);
        let depth = i32::from(event.depth);
        if full > 0 && full < depth {
            if right > left {
                partial_right = depth - full;
            } else {
                partial_left = depth - full;
            }
        }
        let count = |value: i32| u8::try_from(value.max(0)).ok();
        respelled.push(ChordBeams::from_counts(
            count(right - left)?,
            count(left - right)?,
            count(partial_left)?,
            count(partial_right)?,
        )?);
    }

    let candidate: Vec<_> = events
        .iter()
        .zip(&respelled)
        .map(|(event, beams)| Event {
            beams: beams.clone(),
            ..event.clone()
        })
        .collect();
    let redrawn = analyze(&candidate, Event::delta);
    let same = [Stream::Regular, Stream::Grace].into_iter().all(|stream| {
        drawn.stream(stream).group == redrawn.stream(stream).group
            && drawn.stream(stream).breaks == redrawn.stream(stream).breaks
    });
    same.then_some(respelled)
}

#[cfg(test)]
mod tests {
    use crate::{normalization::NormalizationPass, KernDocument};

    use super::CanonicalizeBeamSpelling;

    fn canonical(body: &str) -> String {
        let source = format!("**kern\n*M4/4\n{body}\n=\n*-\n");
        let mut document = KernDocument::parse(&source).unwrap();
        CanonicalizeBeamSpelling.normalize(&mut document);
        let text = crate::serialize::SerializePreset::HumdrumKern
            .try_serialize_document(&document)
            .unwrap();
        text.lines()
            .filter(|line| !line.starts_with(['*', '=', '!']))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn full_beams_follow_the_durations() {
        let conventional = "16cLL 16dJJ 16eLL 16fJJ 4g 2a";
        assert_eq!(canonical("16cL\n16dJ\n16eL\n16fJ\n4g\n2a"), conventional);
        assert_eq!(
            canonical("16cLK\n16dJk\n16eLL\n16fJJ\n4g\n2a"),
            conventional
        );
        assert_eq!(
            canonical("16cLL\n16dJJ\n16eLL\n16fJJ\n4g\n2a"),
            conventional
        );
    }

    #[test]
    fn partial_beams_fill_the_depth() {
        assert_eq!(canonical("8.cL\n16dJ\n4e\n2f"), "8.cL 16dJk 4e 2f");
        assert_eq!(canonical("16cL\n8.dJ\n4e\n2f"), "16cLK 8.dJ 4e 2f");
        assert_eq!(canonical("16.cLL\n32dJJ\n4e\n2f"), "16.cLL 32dJJk 4e 2f");
    }

    #[test]
    fn secondary_breaks_survive() {
        let broken = "16cLL 16dJ 16eL 16fJJ 4g 2a";
        assert_eq!(canonical("16cLL\n16dJ\n16eL\n16fJJ\n4g\n2a"), broken);
        assert_eq!(canonical("16cLLL\n16dJJ\n16eLL\n16fJJJ\n4g\n2a"), broken);
    }

    #[test]
    fn breaks_the_conventional_spelling_cannot_draw_keep_their_marks() {
        assert_eq!(
            canonical("8cLL\n8dJ\n8eL\n8fJJ\n2g"),
            "8cLL 8dJ 8eL 8fJJ 2g"
        );
        // Rests carry no marks, so the conventional links would dip at `16d`.
        assert_eq!(
            canonical("32cLLK\n32r\n16d\n32ek\n32r\n16fJJ\n4g\n2a"),
            "32cLLK 32r 16d 32ek 32r 16fJJ 4g 2a"
        );
    }

    #[test]
    fn a_beam_closed_from_an_earlier_page_keeps_its_marks() {
        assert_eq!(canonical("8cJ\n8dL\n8eJ\n4f\n2g"), "8cJ 8dL 8eJ 4f 2g");
    }

    #[test]
    fn a_regular_beam_closed_by_a_grace_note_keeps_its_marks() {
        assert_eq!(
            canonical("16cLL\n16qqdJJ\n16e\n8.f\n2g"),
            "16cLL 16qqdJJ 16e 8.f 2g"
        );
    }

    #[test]
    fn invisible_rests_in_a_beam_do_not_bound_its_links() {
        let merged = canonical("16dLL\n8ryy\n16BJJ\n4c\n2d");
        let split = canonical("16dLL\n16ryy\n16ryy\n16BJJ\n4c\n2d");
        assert_eq!(merged, "16dLL 8ryy 16BJJ 4c 2d");
        assert_eq!(split, "16dLL 16ryy 16ryy 16BJJ 4c 2d");
    }

    #[test]
    fn unbalanced_measures_keep_their_marks() {
        // humlib pairs the extra `J` with a mark of another layer or measure.
        assert_eq!(canonical("16.cLL\n32dJJJ\n4e\n2f"), "16.cLL 32dJJJ 4e 2f");
        // A beam never closed on the page is a hanging beam.
        assert_eq!(canonical("4c\n4d\n4e\n8fL\n8g"), "4c 4d 4e 8fL 8g");
    }

    #[test]
    fn beamed_grace_notes_lose_the_slash() {
        assert_eq!(canonical("8qcL\n8qdJ\n4e\n4f\n2g"), "8qqcL 8qqdJ 4e 4f 2g");
        assert_eq!(canonical("8qc\n4e\n4f\n2g"), "8qc 4e 4f 2g");
    }

    #[test]
    fn beams_across_a_barline_keep_their_marks() {
        let source = "**kern\n*M2/8\n8cL\n8d\n=\n8eJ\n16fLK\n16gJk\n=\n*-\n";
        let mut document = KernDocument::parse(source).unwrap();
        CanonicalizeBeamSpelling.normalize(&mut document);
        let text = crate::serialize::SerializePreset::HumdrumKern
            .try_serialize_document(&document)
            .unwrap();
        assert!(text.contains("8cL\n8d\n=\n8eJ\n16fLK\n16gJk"), "{text}");
    }

    #[test]
    fn a_grace_note_without_duration_beams_as_an_eighth() {
        assert_eq!(
            canonical("16qqcLL\nqqd\n16qqeJJ\n4f\n4g\n2a"),
            "16qqcLK qqd 16qqeJk 4f 4g 2a"
        );
    }

    #[test]
    fn tuplets_and_tremolos_keep_their_marks() {
        assert_eq!(canonical("12cL\n12d\n12eJ\n4f\n2g"), "12cL 12d 12eJ 4f 2g");
        assert_eq!(
            canonical("24cL\n24d\n24eJ\n8f\n4g\n2a"),
            "24cL 24d 24eJ 8f 4g 2a"
        );
        let source = "**kern\n*M4/4\n*tremolo\n16cL\n16dJ\n4e\n4f\n4g\n=\n*-\n";
        let mut document = KernDocument::parse(source).unwrap();
        CanonicalizeBeamSpelling.normalize(&mut document);
        assert!(crate::serialize::SerializePreset::HumdrumKern
            .try_serialize_document(&document)
            .unwrap()
            .contains("16cL\n16dJ"));
    }

    #[test]
    fn grace_notes_in_a_measure_with_a_beam_across_the_barline_keep_their_slash() {
        let source = "**kern\n*M2/4\n8qcL\n8qd\n8eL\n8f\n4g\n=\n8qcJ\n8aJ\n8b\n4c\n=\n*-\n";
        let mut document = KernDocument::parse(source).unwrap();
        CanonicalizeBeamSpelling.normalize(&mut document);
        let text = crate::serialize::SerializePreset::HumdrumKern
            .try_serialize_document(&document)
            .unwrap();
        assert!(!text.contains("qq"), "{text}");
    }

    #[test]
    fn unrepresentable_groups_keep_their_marks() {
        assert_eq!(canonical("8cL\n4d\n8eJ\n2f"), "8cL 4d 8eJ 2f");
    }
}
