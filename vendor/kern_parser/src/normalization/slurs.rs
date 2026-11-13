//! Slur normalization.
//!
//! Normalizes slur markers that open and close on the same data token.
//! These immediate same-event spans do not describe a useful OMR-visible slur,
//! so matching open/close counts are cancelled in place.
//!
//! This is intentionally token-local. It does not resolve document-level slur
//! pairing, deduplicate equivalent spans across later notes, or validate slur
//! balance across the piece.
//!
//! Example:
//! ```text
//! Before: (4c) ((4e)
//! After:  4c   (4e
//! ```
//!
//! A same-token open/close pair cancels. If more opens than closes appear on
//! the token, the unmatched opening slur is preserved.

use crate::{
    normalization::NormalizationPass, types::Slur, KernContent, KernDocument, KernLine,
    SpineContent,
};

use super::NormalizeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Remove matching slur opens and closes that occur on the same chord token.
///
/// If a token has more opens than closes, or more closes than opens, only the
/// matched count is removed and the unmatched remainder is preserved.
pub struct RemoveImmediateSlurs {
    options: NormalizeOptions,
}

impl RemoveImmediateSlurs {
    pub fn new(options: NormalizeOptions) -> Self {
        Self { options }
    }
}

impl NormalizationPass for RemoveImmediateSlurs {
    fn normalize(&self, document: &mut KernDocument) {
        for line in document.lines_mut() {
            let KernLine::Data(contents) = line else {
                continue;
            };

            for content in contents {
                let SpineContent::Kern(content) = content else {
                    continue;
                };

                match content {
                    KernContent::Chord(chord) => remove_immediate_slurs(&mut chord.modifiers.slurs),
                    KernContent::Grace(grace) => remove_immediate_slurs(&mut grace.modifiers.slurs),
                    KernContent::Rest(rest) => remove_immediate_slurs(&mut rest.slurs),
                    KernContent::Null
                    | KernContent::InvisibleRestSlice(_)
                    | KernContent::MultiMeasureRest(_)
                    | KernContent::Note(_) => {}
                }
            }
        }
    }
}

fn remove_immediate_slurs(slurs: &mut Vec<Slur>) {
    let mut cancellations = Vec::new();
    for slur in slurs.iter() {
        let key = slur_key(slur);
        if cancellations
            .iter()
            .any(|cancellation: &Cancellation| cancellation.key == key)
        {
            continue;
        }

        if key.linked {
            continue;
        }
        let opens = slur_count_for(slurs, key, SlurDirection::Open);
        let closes = slur_count_for(slurs, key, SlurDirection::Close);
        let count = opens.min(closes);
        if count > 0 {
            cancellations.push(Cancellation {
                key,
                remaining_opens: count,
                remaining_closes: count,
            });
        }
    }

    if cancellations.is_empty() {
        return;
    }

    let mut normalized = Vec::new();
    for slur in slurs.iter() {
        let key = slur_key(slur);
        let direction = slur_direction(slur);
        let count = slur_count(slur);
        let Some(cancellation) = cancellations
            .iter_mut()
            .find(|cancellation| cancellation.key == key)
        else {
            normalized.push(slur.clone());
            continue;
        };

        let remaining_cancellations = match direction {
            SlurDirection::Open => &mut cancellation.remaining_opens,
            SlurDirection::Close => &mut cancellation.remaining_closes,
        };
        let removed = count.min(*remaining_cancellations);
        *remaining_cancellations -= removed;
        let remaining = count - removed;
        if remaining > 0 {
            normalized.push(slur_with_count(slur, remaining));
        }
    }

    *slurs = normalized;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlurKey {
    elided: bool,
    depth: u8,
    linked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlurDirection {
    Open,
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cancellation {
    key: SlurKey,
    remaining_opens: u8,
    remaining_closes: u8,
}

fn slur_key(slur: &Slur) -> SlurKey {
    match slur {
        Slur::Open {
            elided,
            depth,
            linked,
            ..
        }
        | Slur::Close {
            elided,
            depth,
            linked,
            ..
        } => SlurKey {
            elided: *elided,
            depth: *depth,
            linked: *linked,
        },
    }
}

fn slur_direction(slur: &Slur) -> SlurDirection {
    match slur {
        Slur::Open { .. } => SlurDirection::Open,
        Slur::Close { .. } => SlurDirection::Close,
    }
}

fn slur_count(slur: &Slur) -> u8 {
    match slur {
        Slur::Open { count, .. } | Slur::Close { count, .. } => *count,
    }
}

fn slur_count_for(slurs: &[Slur], key: SlurKey, direction: SlurDirection) -> u8 {
    slurs
        .iter()
        .filter(|slur| slur_key(slur) == key && slur_direction(slur) == direction)
        .map(slur_count)
        .sum()
}

fn slur_with_count(slur: &Slur, count: u8) -> Slur {
    match slur {
        Slur::Open {
            elided,
            depth,
            placement,
            linked,
            ..
        } => Slur::Open {
            elided: *elided,
            depth: *depth,
            count,
            placement: *placement,
            linked: *linked,
        },
        Slur::Close {
            elided,
            depth,
            linked,
            ..
        } => Slur::Close {
            elided: *elided,
            depth: *depth,
            count,
            linked: *linked,
        },
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        normalization::{slurs::RemoveImmediateSlurs, NormalizationPass},
        types::Slur,
        KernContent, KernDocument, KernLine, SpineContent,
    };

    fn normalize_slurs(document: &mut KernDocument) {
        RemoveImmediateSlurs::new(Default::default()).normalize(document);
    }

    #[test]
    fn removes_immediate_same_event_slurs() {
        let mut document = KernDocument::parse("**kern\n(4c)\n").unwrap();

        normalize_slurs(&mut document);

        let KernLine::Data(contents) = document
            .lines()
            .iter()
            .find(|line| matches!(line, KernLine::Data(_)))
            .expect("expected data line")
        else {
            panic!("expected data line");
        };
        let SpineContent::Kern(KernContent::Chord(chord)) = &contents[0] else {
            panic!("expected chord");
        };
        assert!(chord.modifiers.slurs.is_empty());
    }

    #[test]
    fn preserves_unmatched_immediate_slur_counts() {
        let mut document = KernDocument::parse("**kern\n((4c)\n").unwrap();

        normalize_slurs(&mut document);

        let KernLine::Data(contents) = document
            .lines()
            .iter()
            .find(|line| matches!(line, KernLine::Data(_)))
            .expect("expected data line")
        else {
            panic!("expected data line");
        };
        let SpineContent::Kern(KernContent::Chord(chord)) = &contents[0] else {
            panic!("expected chord");
        };
        assert_eq!(
            chord.modifiers.slurs,
            vec![Slur::Open {
                elided: false,
                depth: 0,
                count: 1,
                placement: None,
                linked: false,
            }]
        );
    }
}
