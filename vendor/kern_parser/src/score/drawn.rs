//! What the page shows: drawn durations and note size.
//!
//! Logical time stays in [`crate::MusicalDuration`], which has arithmetic and
//! no notation methods. A [`DrawnDuration`] is the opposite: it answers beam
//! depth and note value, and cannot be summed or converted back to time. The
//! only way to obtain one is [`super::Score::drawn`], because it depends on
//! the `*rscale` in force.

use num_rational::Rational64;

use super::Score;
use crate::{types::DurationValue, Duration, MusicalDuration, RhythmScale};

/// A printed note value: base value plus augmentation dots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawnDuration {
    value: DurationValue,
    dots: u8,
}

/// A scaled duration with no printable note value, such as `8` under
/// `*rscale:5` (five eighths).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Undrawable;

impl DrawnDuration {
    /// humlib `analyzeRScale`: scale the undotted written value, spell the
    /// result as a recip with up to three dots of its own, then append the
    /// written dots.
    pub(crate) fn project(written: &Duration, scale: RhythmScale) -> Result<Self, Undrawable> {
        if scale.is_unit() {
            return Ok(Self {
                value: written.value.clone(),
                dots: written.augmentation_dots,
            });
        }
        let undotted = MusicalDuration::from_duration(&Duration {
            value: written.value.clone(),
            augmentation_dots: 0,
        });
        let scaled = Rational64::new(undotted.numerator(), undotted.denominator())
            * Rational64::new(i64::from(scale.numerator()), i64::from(scale.denominator()));
        let (value, dots) = spell(scaled).ok_or(Undrawable)?;
        Ok(Self {
            value,
            dots: dots + written.augmentation_dots,
        })
    }

    /// Verovio's drawing of a note written `written` with `!LO:N:vis=visual`:
    /// the undotted `vis` value times the tuplet scaling of the written value,
    /// plus the `vis` dots. `None` when that is no plain note value, where
    /// Verovio sets no printed duration either.
    fn visual(written: &Duration, visual: &Duration) -> Option<Self> {
        let scaled = undotted_whole_notes(visual) * tuplet_scaling(written);
        if !is_power_of_two(scaled) {
            return None;
        }
        let (value, _) = spell(scaled)?;
        Some(Self {
            value,
            dots: visual.augmentation_dots,
        })
    }

    pub fn value(&self) -> &DurationValue {
        &self.value
    }

    pub fn dots(&self) -> u8 {
        self.dots
    }

    /// Beams (or flags) the printed value takes. Dots do not affect it.
    pub fn beam_depth(&self) -> u8 {
        let Some(log2) = base_reciprocal_floor_log2(&self.value) else {
            return 0;
        };
        if log2 < 3 {
            return 0;
        }
        (log2 - 2).try_into().unwrap_or(u8::MAX)
    }

    /// `**kern` spelling of the printed value, such as `4`, `8.` or `0`.
    pub fn to_kern(&self) -> String {
        let base = match self.value {
            DurationValue::Reciprocal(value) => value.to_string(),
            DurationValue::Rational {
                numerator,
                denominator,
            } => format!("{numerator}%{denominator}"),
            DurationValue::Breve => "0".to_owned(),
            DurationValue::Long => "00".to_owned(),
            DurationValue::Maxima => "000".to_owned(),
        };
        format!("{base}{}", ".".repeat(self.dots.into()))
    }
}

/// humlib `Convert::durationToRecip` for a whole-note duration, without the
/// `%` fallback: that spelling has no printed note value.
fn spell(duration: Rational64) -> Option<(DurationValue, u8)> {
    let (numerator, denominator) = (*duration.numer(), *duration.denom());
    if numerator == 1 {
        return Some((DurationValue::Reciprocal(denominator.try_into().ok()?), 0));
    }
    if denominator == 1 {
        let breve = match numerator {
            2 => Some((DurationValue::Breve, 0)),
            3 => Some((DurationValue::Breve, 1)),
            4 => Some((DurationValue::Long, 0)),
            6 => Some((DurationValue::Long, 1)),
            8 => Some((DurationValue::Maxima, 0)),
            12 => Some((DurationValue::Maxima, 1)),
            _ => None,
        };
        if breve.is_some() {
            return breve;
        }
    }
    for (dots, factor) in [
        (1u8, Rational64::new(2, 3)),
        (2, Rational64::new(4, 7)),
        (3, Rational64::new(8, 15)),
    ] {
        let undotted = duration * factor;
        if *undotted.numer() == 1 {
            return Some((
                DurationValue::Reciprocal((*undotted.denom()).try_into().ok()?),
                dots,
            ));
        }
    }
    None
}

fn undotted_whole_notes(duration: &Duration) -> Rational64 {
    let undotted = MusicalDuration::from_duration(&Duration {
        value: duration.value.clone(),
        augmentation_dots: 0,
    });
    Rational64::new(undotted.numerator(), undotted.denominator())
}

/// The factor Verovio multiplies a tuplet note's value by to draw it
/// (`HumdrumInput::prepareBeamAndTupletGroups`). It depends only on the
/// note's own undotted value: a value that is no power of two is scaled to
/// the next higher power of two below a quarter note, or to a quarter note
/// from there up, halved again when the value is three times a power of two.
fn tuplet_scaling(written: &Duration) -> Rational64 {
    let quarters = undotted_whole_notes(written) * 4;
    if is_power_of_two(quarters) {
        return Rational64::from_integer(1);
    }
    let mut target = Rational64::from_integer(1);
    while target / 2 >= quarters {
        target /= 2;
    }
    if *quarters.numer() == 3 && is_power_of_two(Rational64::from_integer(*quarters.denom())) {
        target /= 2;
    }
    let ratio = quarters / target;
    let (top, mut bottom) = (*ratio.denom(), *ratio.numer());
    if quarters == Rational64::new(4, 3) {
        bottom = 2;
    }
    Rational64::new(top, bottom)
}

fn is_power_of_two(value: Rational64) -> bool {
    let (numerator, denominator) = (*value.numer(), *value.denom());
    numerator > 0
        && ((numerator == 1 && (denominator & (denominator - 1)) == 0)
            || (denominator == 1 && (numerator & (numerator - 1)) == 0))
}

fn base_reciprocal_floor_log2(value: &DurationValue) -> Option<u32> {
    match *value {
        DurationValue::Reciprocal(value) if value > 0 => {
            Some(u32::BITS - 1 - value.leading_zeros())
        }
        DurationValue::Rational {
            numerator,
            denominator,
        } if numerator > 0 && denominator > 0 => {
            let numerator = u128::from(numerator);
            let denominator = u128::from(denominator);
            if numerator < denominator {
                return None;
            }
            let mut log2 = 0;
            while log2 < u32::BITS - 1 && (denominator << (log2 + 1)) <= numerator {
                log2 += 1;
            }
            Some(log2)
        }
        _ => None,
    }
}

/// Printed size of a note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteSize {
    Normal,
    Cue,
}

/// The drawn view of a score.
#[derive(Debug, Clone, Copy)]
pub struct DrawnView<'s, 'd> {
    score: &'s Score<'d>,
}

impl<'s, 'd> DrawnView<'s, 'd> {
    pub(super) fn new(score: &'s Score<'d>) -> Self {
        Self { score }
    }

    /// Printed value of a duration written in the field at `line_index`,
    /// `column`. The resolver has proved every written duration drawable.
    ///
    /// Like Verovio, a `*rscale` other than 1 scales the written value and
    /// overrides `!LO:N:vis`; otherwise `vis` replaces the written value.
    /// Inside a tuplet `vis` is in tuplet units, as the written value is.
    pub fn duration(&self, line_index: usize, column: usize, written: &Duration) -> DrawnDuration {
        let scale = self.score.state_at(line_index, column).track.rhythm_scale;
        if scale.is_unit() {
            if let Some(drawn) = self
                .score
                .layouts
                .visual_duration(line_index, column)
                .and_then(|visual| DrawnDuration::visual(written, visual))
            {
                return drawn;
            }
        }
        DrawnDuration::project(written, scale)
            .expect("Score::resolve rejects durations with no printed value")
    }

    /// Beams on a note. A `vis` that draws a value with no beams of its own,
    /// such as a half-note head on a beamed eighth's stem, leaves the beams
    /// to the written value.
    pub fn beam_depth(&self, line_index: usize, column: usize, written: &Duration) -> u8 {
        match self.duration(line_index, column, written).beam_depth() {
            0 => {
                let scale = self.score.state_at(line_index, column).track.rhythm_scale;
                DrawnDuration::project(written, scale)
                    .expect("Score::resolve rejects durations with no printed value")
                    .beam_depth()
            }
            depth => depth,
        }
    }

    pub fn size(&self, line_index: usize, column: usize) -> NoteSize {
        if self.score.state_at(line_index, column).layer.cue {
            NoteSize::Cue
        } else {
            NoteSize::Normal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn written(kern: &str) -> Duration {
        let token = crate::parse_kern_token(&format!("{kern}c"), Default::default()).unwrap();
        let crate::KernContent::Chord(chord) = token else {
            panic!("expected a note")
        };
        chord.notes[0].duration.clone()
    }

    fn drawn_unit(duration: &Duration) -> DrawnDuration {
        DrawnDuration::project(duration, RhythmScale::UNIT).unwrap()
    }

    fn reciprocal_duration(value: u32) -> Duration {
        Duration {
            value: DurationValue::Reciprocal(value),
            augmentation_dots: 0,
        }
    }

    fn rational_duration(numerator: u32, denominator: u32) -> Duration {
        Duration {
            value: DurationValue::Rational {
                numerator,
                denominator,
            },
            augmentation_dots: 0,
        }
    }

    #[test]
    fn expected_beam_depth_uses_reciprocal_duration_buckets() {
        for (value, expected) in [
            (1, 0),
            (4, 0),
            (7, 0),
            (8, 1),
            (12, 1),
            (15, 1),
            (16, 2),
            (24, 2),
            (31, 2),
            (32, 3),
            (48, 3),
            (63, 3),
            (64, 4),
            (96, 4),
            (127, 4),
            (128, 5),
            (192, 5),
            (255, 5),
            (256, 6),
        ] {
            assert_eq!(
                drawn_unit(&reciprocal_duration(value)).beam_depth(),
                expected,
                "duration {value}"
            );
        }
    }

    #[test]
    fn expected_beam_depth_uses_rational_duration_buckets() {
        for (numerator, denominator, expected) in [
            (7, 1, 0),
            (8, 1, 1),
            (40, 3, 1),
            (31, 2, 1),
            (32, 2, 2),
            (127, 4, 2),
            (128, 4, 3),
            (3, 5, 0),
        ] {
            assert_eq!(
                drawn_unit(&rational_duration(numerator, denominator)).beam_depth(),
                expected,
                "duration {numerator}%{denominator}"
            );
        }
    }

    #[test]
    fn expected_beam_depth_ignores_dots_and_unbeamable_values() {
        assert_eq!(
            drawn_unit(&Duration {
                value: DurationValue::Reciprocal(8),
                augmentation_dots: 3,
            })
            .beam_depth(),
            1
        );
        assert_eq!(
            drawn_unit(&Duration {
                value: DurationValue::Rational {
                    numerator: 40,
                    denominator: 3,
                },
                augmentation_dots: 3,
            })
            .beam_depth(),
            1
        );
        assert_eq!(
            drawn_unit(&Duration {
                value: DurationValue::Breve,
                augmentation_dots: 0,
            })
            .beam_depth(),
            0
        );
    }

    fn drawn(kern: &str, numerator: u32, denominator: u32) -> Option<String> {
        DrawnDuration::project(
            &written(kern),
            RhythmScale::new(numerator, denominator).unwrap(),
        )
        .ok()
        .map(|duration| duration.to_kern())
    }

    #[test]
    fn projection_matches_humlib_rscale() {
        assert_eq!(drawn("8", 2, 1).as_deref(), Some("4"));
        assert_eq!(drawn("8.", 2, 1).as_deref(), Some("4."));
        assert_eq!(drawn("20", 2, 1).as_deref(), Some("10"));
        assert_eq!(drawn("4", 1, 2).as_deref(), Some("8"));
        assert_eq!(drawn("4", 3, 2).as_deref(), Some("4."));
        assert_eq!(drawn("4.", 3, 2).as_deref(), Some("4.."));
        assert_eq!(drawn("4", 2, 3).as_deref(), Some("6"));
        assert_eq!(drawn("1", 2, 1).as_deref(), Some("0"));
        assert_eq!(drawn("2", 3, 1).as_deref(), Some("1."));
        // humlib spells five eighths as a triple-dotted triplet half.
        assert_eq!(drawn("8", 5, 1).as_deref(), Some("3..."));
        assert_eq!(drawn("4", 5, 1), None);
    }

    #[test]
    fn beam_depth_follows_the_printed_value() {
        let depth = |kern: &str, n, d| {
            DrawnDuration::project(&written(kern), RhythmScale::new(n, d).unwrap())
                .unwrap()
                .beam_depth()
        };
        assert_eq!(depth("16", 1, 1), 2);
        assert_eq!(depth("16", 2, 1), 1);
        assert_eq!(depth("8", 2, 1), 0);
        assert_eq!(depth("20", 2, 1), 1);
    }

    #[test]
    fn tuplet_scaling_matches_verovio() {
        let scaling = |kern: &str| tuplet_scaling(&written(kern));
        assert_eq!(scaling("8"), Rational64::from_integer(1));
        assert_eq!(scaling("6"), Rational64::new(3, 2));
        assert_eq!(scaling("12."), Rational64::new(3, 2));
        assert_eq!(scaling("3"), Rational64::new(3, 2));
        assert_eq!(scaling("64%3"), Rational64::new(2, 3));
        assert_eq!(scaling("32%3"), Rational64::new(2, 3));
        assert_eq!(scaling("104%3"), Rational64::new(13, 12));
    }

    #[test]
    fn visual_duration_is_drawn_under_the_written_tuplet_scaling() {
        let drawn = |kern: &str, vis: &str| {
            DrawnDuration::visual(&written(kern), &written(vis)).map(|drawn| drawn.to_kern())
        };
        assert_eq!(drawn("6", "12.").as_deref(), Some("8."));
        assert_eq!(drawn("16", "8").as_deref(), Some("8"));
        assert_eq!(drawn("8", "2").as_deref(), Some("2"));
        assert_eq!(drawn("12", "3.").as_deref(), Some("2."));
        assert_eq!(drawn("64%3", "32%3").as_deref(), Some("16"));
        assert_eq!(drawn("104%3", "52%3").as_deref(), Some("16"));
        assert_eq!(drawn("8", "12"), None);
    }

    #[test]
    fn an_unreadable_visual_duration_is_a_state_error() {
        let document = crate::KernDocument::parse("**kern\n!LO:N:vis=x\n4c\n*-\n").unwrap();
        let errors = Score::resolve(&document, crate::document::Framing::Excerpt).unwrap_err();
        assert!(matches!(
            errors.0.as_slice(),
            [super::super::StateError::InvalidVisualDuration { line: 2, column: 1, value }]
                if value == "x"
        ));
    }
}
