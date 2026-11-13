//! Document-level slur pairing.
//!
//! Pairing mirrors humlib's `analyzeKernSlurs`, because downstream editors treat
//! the slurs Verovio draws as the truth:
//!
//! - every `**kern` spine is a track with one stack of pending opens per layer
//!   position, the rank of a field among the track's fields in its row;
//! - a close takes the newest pending open of its key at its own position, else
//!   the newest at the lowest position that has one, else it stays unpaired;
//! - closes never pair with opens in another `**kern` spine;
//! - the key is the marker's elision state and elision depth;
//! - `count` is expanded as that many opens or closes;
//! - on one token all closes pair before any open is pushed, so `4d()` ends one
//!   slur and starts the next, and a lone `(4d)` pairs with nothing;
//! - `*^`, `*v`, and `*-` only change a field's layer position within its
//!   track. Pending opens never move;
//! - linked markers (`N(`, `N)`) skip the stacks. Opens and closes are each
//!   listed track by track, then by row and layer position, and the i-th open
//!   pairs with the i-th close whatever their staves or order. An ordinary open
//!   listed before a linked open on the same field counts as linked too,
//!   because humlib's `isLinkedSlurBegin` searches forward from the marker
//!   instead of checking the byte before it.
//!
//! Spans follow layer positions, not voices. After a split shifts the layers,
//! Verovio pairs a close with the open at the lowest position even when the
//! other voice opened it, and so does this resolver. Do not read spans as
//! musical phrasing.

use crate::{
    score::Score,
    types::{KernContent, KernLine, Slur, SpineContent, SpineType},
};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SlurSpanAnalysis {
    /// One entry per paired marker, so `((4c` closed by `4e))` yields two.
    pub spans: Vec<SlurSpan>,
    /// Markers Verovio draws nothing for, sorted by location.
    pub unpaired: Vec<UnpairedSlur>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlurSpan {
    pub key: SlurKey,
    pub open: SlurLocation,
    pub close: SlurLocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnpairedSlur {
    pub key: SlurKey,
    pub side: SlurSide,
    pub location: SlurLocation,
    pub staff: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SlurSide {
    Close,
    Open,
}

/// One-based `line` and `column`, the `L` and `F` of Verovio's `slur-L18F3-L24F4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SlurLocation {
    pub line: usize,
    pub column: usize,
    pub line_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SlurKey {
    pub elided: bool,
    pub depth: u8,
    pub linked: bool,
}

/// Pair the slurs of a resolved score. Unpaired markers carry their track's
/// staff number.
pub fn resolve_slur_spans(score: &Score<'_>) -> SlurSpanAnalysis {
    let document = score.document();
    let topology = score.topology();
    let mut analysis = SlurSpanAnalysis::default();
    let mut tracks = Vec::<Track>::new();
    tracks.resize_with(topology.track_count(), Track::default);

    for (line_index, line) in document.lines().iter().enumerate() {
        let KernLine::Data(contents) = line else {
            continue;
        };
        for (column, (&layer, content)) in topology
            .layers_at(line_index)
            .iter()
            .zip(contents)
            .enumerate()
        {
            let track = topology.track_of(layer);
            let (SpineType::Kern, SpineContent::Kern(content)) =
                (topology.spine_type(layer), content)
            else {
                continue;
            };
            observe_kern_content(
                &mut tracks[track.index()],
                &mut analysis,
                column - topology.track_columns(line_index, track).start,
                content,
                SlurLocation {
                    line: document.source_line_number(line_index),
                    column: column + 1,
                    line_index,
                },
                topology.track(track).staff,
            );
        }
    }

    pair_linked_markers(&tracks, &mut analysis);
    for track in &mut tracks {
        for pending in track.layers.drain(..).flatten() {
            analysis.unpaired.push(UnpairedSlur {
                key: pending.key,
                side: SlurSide::Open,
                location: pending.location,
                staff: pending.staff,
            });
        }
    }
    analysis
        .unpaired
        .sort_by_key(|unpaired| (unpaired.location, unpaired.side, unpaired.key));

    analysis
}

#[derive(Debug, Default)]
struct Track {
    /// `layers[position]` holds the opens pushed from that position, oldest first.
    layers: Vec<Vec<Marker>>,
    linked_opens: Vec<Marker>,
    linked_closes: Vec<Marker>,
}

#[derive(Debug, Clone, Copy)]
struct Marker {
    key: SlurKey,
    location: SlurLocation,
    staff: Option<u32>,
}

impl Track {
    fn open(&mut self, position: usize, pending: Marker) {
        if self.layers.len() <= position {
            self.layers.resize_with(position + 1, Vec::new);
        }
        self.layers[position].push(pending);
    }

    fn close(&mut self, position: usize, key: SlurKey) -> Option<SlurLocation> {
        let own = self
            .layers
            .get_mut(position)
            .and_then(|layer| take_newest(layer, key));
        own.or_else(|| {
            self.layers
                .iter_mut()
                .find_map(|layer| take_newest(layer, key))
        })
    }
}

fn take_newest(layer: &mut Vec<Marker>, key: SlurKey) -> Option<SlurLocation> {
    let index = layer.iter().rposition(|pending| pending.key == key)?;
    Some(layer.remove(index).location)
}

fn pair_linked_markers(tracks: &[Track], analysis: &mut SlurSpanAnalysis) {
    let mut opens = tracks.iter().flat_map(|track| &track.linked_opens);
    let mut closes = tracks.iter().flat_map(|track| &track.linked_closes);
    loop {
        let (side, marker) = match (opens.next(), closes.next()) {
            (Some(open), Some(close)) => {
                analysis.spans.push(SlurSpan {
                    key: open.key,
                    open: open.location,
                    close: close.location,
                });
                continue;
            }
            (Some(open), None) => (SlurSide::Open, open),
            (None, Some(close)) => (SlurSide::Close, close),
            (None, None) => return,
        };
        analysis.unpaired.push(UnpairedSlur {
            key: marker.key,
            side,
            location: marker.location,
            staff: marker.staff,
        });
    }
}

fn observe_kern_content(
    track: &mut Track,
    analysis: &mut SlurSpanAnalysis,
    position: usize,
    content: &KernContent,
    location: SlurLocation,
    staff: Option<u32>,
) {
    let slurs = slurs_for_content(content);
    for slur in slurs {
        let Slur::Close {
            elided,
            depth,
            count,
            linked,
        } = *slur
        else {
            continue;
        };
        let key = SlurKey {
            elided,
            depth,
            linked,
        };
        for _ in 0..count {
            if linked {
                track.linked_closes.push(Marker {
                    key,
                    location,
                    staff,
                });
                continue;
            }
            match track.close(position, key) {
                Some(open) => analysis.spans.push(SlurSpan {
                    key,
                    open,
                    close: location,
                }),
                None => analysis.unpaired.push(UnpairedSlur {
                    key,
                    side: SlurSide::Close,
                    location,
                    staff,
                }),
            }
        }
    }
    let last_linked_open = slurs
        .iter()
        .rposition(|slur| matches!(slur, Slur::Open { linked: true, .. }));
    for (index, slur) in slurs.iter().enumerate() {
        let Slur::Open {
            elided,
            depth,
            count,
            ..
        } = *slur
        else {
            continue;
        };
        let key = SlurKey {
            elided,
            depth,
            linked: last_linked_open.is_some_and(|last| index <= last),
        };
        let marker = Marker {
            key,
            location,
            staff,
        };
        for _ in 0..count {
            if key.linked {
                track.linked_opens.push(marker);
            } else {
                track.open(position, marker);
            }
        }
    }
}

pub(crate) fn slurs_for_content(content: &KernContent) -> &[Slur] {
    match content {
        KernContent::Chord(chord) => &chord.modifiers.slurs,
        KernContent::Grace(grace) => &grace.modifiers.slurs,
        KernContent::Rest(rest) => &rest.slurs,
        KernContent::Null
        | KernContent::InvisibleRestSlice(_)
        | KernContent::MultiMeasureRest(_)
        | KernContent::Note(_) => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{document::Framing, KernDocument};

    fn resolve(input: &str) -> SlurSpanAnalysis {
        let document = KernDocument::parse(input).unwrap();
        resolve_slur_spans(&Score::resolve(&document, Framing::Excerpt).unwrap())
    }

    fn span_lines(analysis: &SlurSpanAnalysis) -> Vec<(usize, usize)> {
        analysis
            .spans
            .iter()
            .map(|span| (span.open.line, span.close.line))
            .collect()
    }

    fn unpaired(analysis: &SlurSpanAnalysis) -> Vec<(SlurSide, usize, usize)> {
        analysis
            .unpaired
            .iter()
            .map(|unpaired| {
                (
                    unpaired.side,
                    unpaired.location.line,
                    unpaired.location.column,
                )
            })
            .collect()
    }

    #[test]
    fn resolves_nested_slurs_newest_first() {
        let analysis = resolve("**kern\n(4c\n(4d\n4e)\n4f)\n*-\n");

        assert_eq!(span_lines(&analysis), vec![(3, 4), (2, 5)]);
        assert_eq!(unpaired(&analysis), vec![]);
    }

    #[test]
    fn expands_counted_markers() {
        let analysis = resolve("**kern\n((4c\n4d))\n*-\n");

        assert_eq!(span_lines(&analysis), vec![(2, 3), (2, 3)]);
    }

    #[test]
    fn keeps_elided_and_ordinary_slurs_apart() {
        let analysis = resolve("**kern\n&(4c\n(4d\n4e&)\n4f)\n*-\n");

        assert_eq!(span_lines(&analysis), vec![(2, 4), (3, 5)]);
        assert!(analysis.spans[0].key.elided);
        assert!(!analysis.spans[1].key.elided);
    }

    #[test]
    fn close_after_a_merge_reaches_an_open_from_the_right_layer() {
        let analysis = resolve("**kern\n*^\n4c\t(4e\n*v\t*v\n4d)\n*-\n");

        assert_eq!(span_lines(&analysis), vec![(3, 5)]);
        assert_eq!(analysis.spans[0].open.column, 2);
    }

    #[test]
    fn reports_unmatched_close_and_unclosed_open() {
        let analysis = resolve("**kern\n4c)\n(4d\n*-\n");

        assert_eq!(analysis.spans, vec![]);
        assert_eq!(
            unpaired(&analysis),
            vec![(SlurSide::Close, 2, 1), (SlurSide::Open, 3, 1)]
        );
    }

    #[test]
    fn close_in_another_kern_spine_stays_unpaired() {
        let analysis = resolve("**kern\t**kern\n(4C\t4c\n4D\t4d)\n*-\t*-\n");

        assert_eq!(analysis.spans, vec![]);
        assert_eq!(
            unpaired(&analysis),
            vec![(SlurSide::Open, 2, 1), (SlurSide::Close, 3, 2)]
        );
    }

    #[test]
    fn open_and_close_on_one_token_do_not_pair_with_each_other() {
        let analysis = resolve("**kern\n(4c)\n*-\n");

        assert_eq!(analysis.spans, vec![]);
        assert_eq!(
            unpaired(&analysis),
            vec![(SlurSide::Close, 2, 1), (SlurSide::Open, 2, 1)]
        );
    }

    #[test]
    fn unpaired_markers_carry_the_verovio_staff() {
        let analysis = resolve("**kern\t**kern\n(4C\t4c)\n*-\t*-\n");

        let staves = analysis
            .unpaired
            .iter()
            .map(|unpaired| (unpaired.location.column, unpaired.staff))
            .collect::<Vec<_>>();
        assert_eq!(staves, vec![(1, Some(2)), (2, Some(1))]);
        assert_eq!(resolve("(4c\t4e)\n").unpaired[0].staff, Some(2));
    }
}
