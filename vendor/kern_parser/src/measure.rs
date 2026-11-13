use std::fmt;

use num_rational::Rational64;

use crate::{types::DurationValue, Duration, KernContent, TimeSignature};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MeasureLocation {
    pub(crate) source_line: usize,
    pub(crate) column: Option<usize>,
}

impl fmt::Display for MeasureLocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.column {
            Some(column) => write!(formatter, "line {}, column {}", self.source_line, column),
            None => write!(formatter, "line {}", self.source_line),
        }
    }
}

/// Rational musical duration value.
///
/// The stored value is normalized and usually represents whole-note units:
/// `1/4` is a quarter note, `3/8` is a dotted quarter, and `2/1` is a breve.
/// Methods that expose fractions return plain integer pairs so callers do not
/// depend on the crate's internal rational-number implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MusicalDuration(Rational64);

pub(crate) type MeasureDuration = MusicalDuration;

impl MusicalDuration {
    pub(crate) fn new(numerator: i64, denominator: i64) -> Self {
        assert_ne!(denominator, 0, "duration denominator must be nonzero");
        Self(Rational64::new(numerator, denominator))
    }

    pub(crate) fn zero() -> Self {
        Self::new(0, 1)
    }

    pub(crate) fn from_time_signature(time_signature: &TimeSignature) -> Self {
        Self::new(
            i64::from(time_signature.numerator),
            i64::from(time_signature.denominator),
        )
    }

    pub(crate) fn from_duration(duration: &Duration) -> Self {
        let base = match duration.value {
            DurationValue::Reciprocal(value) => {
                assert_ne!(value, 0, "duration reciprocal denominator must be nonzero");
                Rational64::new(1, i64::from(value))
            }
            DurationValue::Rational {
                numerator,
                denominator,
            } => {
                assert_ne!(numerator, 0, "duration numerator must be nonzero");
                Rational64::new(i64::from(denominator), i64::from(numerator))
            }
            DurationValue::Breve => Rational64::new(2, 1),
            DurationValue::Long => Rational64::new(4, 1),
            DurationValue::Maxima => Rational64::new(8, 1),
        };
        let mut total = base;
        let mut dot_value = base;
        for _ in 0..duration.augmentation_dots {
            dot_value /= 2;
            total += dot_value;
        }
        Self(total)
    }

    /// Numerator of the normalized duration fraction.
    pub fn numerator(self) -> i64 {
        *self.0.numer()
    }

    /// Denominator of the normalized duration fraction.
    pub fn denominator(self) -> i64 {
        *self.0.denom()
    }

    /// Normalized duration fraction as `(numerator, denominator)`.
    pub fn as_fraction(self) -> (i64, i64) {
        (self.numerator(), self.denominator())
    }

    /// Normalized quarter-note count as `(numerator, denominator)`.
    pub fn as_quarter_notes_fraction(self) -> (i64, i64) {
        (self * 4).as_fraction()
    }

    pub fn is_zero(self) -> bool {
        self == Self::zero()
    }

    /// Human-readable normalized fraction.
    ///
    /// Whole numbers are displayed without `/1`.
    pub fn display(self) -> String {
        if self.denominator() == 1 {
            self.numerator().to_string()
        } else {
            format!("{}/{}", self.numerator(), self.denominator())
        }
    }

    pub(crate) fn multiplied_by(self, multiplier: u32) -> Self {
        Self(self.0 * i64::from(multiplier))
    }

    pub(crate) fn checked_sub(self, rhs: Self) -> Option<Self> {
        (self >= rhs).then_some(Self(self.0 - rhs.0))
    }

    pub(crate) fn missing_from(self, meter: Self) -> Option<Self> {
        (self > Self::zero() && self < meter).then_some(Self(meter.0 - self.0))
    }

    pub(crate) fn into_duration(self) -> Option<Duration> {
        let numerator = self.numerator();
        let denominator = self.denominator();
        if numerator <= 0 || denominator <= 0 {
            return None;
        }

        let value = if denominator == 1 && numerator == 8 {
            DurationValue::Maxima
        } else if denominator == 1 && numerator == 4 {
            DurationValue::Long
        } else if denominator == 1 && numerator == 2 {
            DurationValue::Breve
        } else if numerator == 1 {
            DurationValue::Reciprocal(u32::try_from(denominator).ok()?)
        } else {
            DurationValue::Rational {
                numerator: u32::try_from(denominator).ok()?,
                denominator: u32::try_from(numerator).ok()?,
            }
        };
        Some(Duration {
            value,
            augmentation_dots: 0,
        })
    }

    pub(crate) fn into_duration_prefer_dots(self) -> Option<Duration> {
        let numerator = self.numerator();
        let denominator = self.denominator();
        if numerator <= 0 || denominator <= 0 {
            return None;
        }

        for augmentation_dots in 0..=3 {
            let dot_factor = Rational64::new(
                (1_i64 << (augmentation_dots + 1)) - 1,
                1_i64 << augmentation_dots,
            );
            let base = self.0 / dot_factor;
            if let Some(value) = duration_value_for_base(base) {
                return Some(Duration {
                    value,
                    augmentation_dots: augmentation_dots as u8,
                });
            }
        }

        self.into_duration()
    }

    pub(crate) fn to_time_signature_prefer_quarter(self) -> Option<TimeSignature> {
        if self.0 <= Rational64::new(0, 1) {
            return None;
        }

        let quarter_numerator = self.0 * 4;
        if *quarter_numerator.denom() == 1 {
            return Some(TimeSignature {
                numerator: u32::try_from(*quarter_numerator.numer()).ok()?,
                denominator: 4,
            });
        }

        Some(TimeSignature {
            numerator: u32::try_from(*self.0.numer()).ok()?,
            denominator: u32::try_from(*self.0.denom()).ok()?,
        })
    }
}

fn duration_value_for_base(base: Rational64) -> Option<DurationValue> {
    let numerator = *base.numer();
    let denominator = *base.denom();
    if numerator <= 0 || denominator <= 0 {
        return None;
    }

    if denominator == 1 && numerator == 8 {
        Some(DurationValue::Maxima)
    } else if denominator == 1 && numerator == 4 {
        Some(DurationValue::Long)
    } else if denominator == 1 && numerator == 2 {
        Some(DurationValue::Breve)
    } else if numerator == 1 && is_power_of_two_i64(denominator) {
        Some(DurationValue::Reciprocal(u32::try_from(denominator).ok()?))
    } else {
        None
    }
}

fn is_power_of_two_i64(value: i64) -> bool {
    value > 0 && (value & (value - 1)) == 0
}

impl fmt::Display for MusicalDuration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.display())
    }
}

impl std::ops::Add for MusicalDuration {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for MusicalDuration {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::ops::Mul<i64> for MusicalDuration {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernContentDurationError {
    ChordContainsNoNotes,
    ChordContainsMixedDurations,
}

impl fmt::Display for KernContentDurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChordContainsNoNotes => formatter.write_str("chord contains no notes"),
            Self::ChordContainsMixedDurations => {
                formatter.write_str("chord contains mixed durations")
            }
        }
    }
}

impl std::error::Error for KernContentDurationError {}

pub(crate) fn kern_content_duration(
    content: &KernContent,
) -> Result<MeasureDuration, KernContentDurationError> {
    match content {
        KernContent::Null
        | KernContent::InvisibleRestSlice(_)
        | KernContent::Grace(_)
        | KernContent::MultiMeasureRest(_) => Ok(MeasureDuration::zero()),
        KernContent::Rest(rest) => Ok(MeasureDuration::from_duration(&rest.duration)),
        KernContent::Note(note) => Ok(MeasureDuration::from_duration(&note.duration)),
        KernContent::Chord(chord) => {
            let Some(first) = chord.notes.first() else {
                return Err(KernContentDurationError::ChordContainsNoNotes);
            };
            for note in &chord.notes[1..] {
                if note.duration != first.duration {
                    return Err(KernContentDurationError::ChordContainsMixedDurations);
                }
            }
            Ok(MeasureDuration::from_duration(&first.duration))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{types::ChordModifiers, Chord, Note};

    fn duration(value: DurationValue, augmentation_dots: u8) -> Duration {
        Duration {
            value,
            augmentation_dots,
        }
    }

    #[test]
    fn duration_conversions_expose_normalized_whole_and_quarter_note_fractions() {
        let quarter = duration(DurationValue::Reciprocal(4), 0);
        assert_eq!(quarter.as_rational_whole_notes().as_fraction(), (1, 4));
        assert_eq!(quarter.as_quarter_notes().as_fraction(), (1, 1));
        assert_eq!(
            quarter
                .as_rational_whole_notes()
                .as_quarter_notes_fraction(),
            (1, 1)
        );

        assert_eq!(
            duration(DurationValue::Reciprocal(4), 1)
                .as_rational_whole_notes()
                .as_fraction(),
            (3, 8)
        );
        assert_eq!(
            duration(DurationValue::Reciprocal(8), 2)
                .as_rational_whole_notes()
                .as_fraction(),
            (7, 32)
        );
        assert_eq!(
            duration(
                DurationValue::Rational {
                    numerator: 3,
                    denominator: 5,
                },
                0,
            )
            .as_rational_whole_notes()
            .as_fraction(),
            (5, 3)
        );
        assert_eq!(
            duration(
                DurationValue::Rational {
                    numerator: 1,
                    denominator: 2,
                },
                0,
            )
            .as_rational_whole_notes()
            .as_fraction(),
            (2, 1)
        );
        assert_eq!(
            duration(
                DurationValue::Rational {
                    numerator: 2,
                    denominator: 1,
                },
                0,
            )
            .as_rational_whole_notes()
            .as_fraction(),
            (1, 2)
        );
    }

    #[test]
    fn duration_conversions_support_large_kern_values() {
        assert_eq!(
            duration(DurationValue::Breve, 0)
                .as_rational_whole_notes()
                .as_fraction(),
            (2, 1)
        );
        assert_eq!(
            duration(DurationValue::Long, 0)
                .as_rational_whole_notes()
                .as_fraction(),
            (4, 1)
        );
        assert_eq!(
            duration(DurationValue::Maxima, 0)
                .as_rational_whole_notes()
                .as_fraction(),
            (8, 1)
        );
    }

    #[test]
    fn whole_note_fractions_convert_to_vhv_duration_spellings() {
        assert_eq!(
            MeasureDuration::new(8, 1).into_duration(),
            Some(duration(DurationValue::Maxima, 0))
        );
        assert_eq!(
            MeasureDuration::new(4, 1).into_duration(),
            Some(duration(DurationValue::Long, 0))
        );
        assert_eq!(
            MeasureDuration::new(2, 1).into_duration(),
            Some(duration(DurationValue::Breve, 0))
        );
        assert_eq!(
            MeasureDuration::new(1, 2).into_duration(),
            Some(duration(DurationValue::Reciprocal(2), 0))
        );
        assert_eq!(
            MeasureDuration::new(3, 4).into_duration(),
            Some(duration(
                DurationValue::Rational {
                    numerator: 4,
                    denominator: 3,
                },
                0
            ))
        );
    }

    #[test]
    fn whole_note_fractions_can_prefer_dotted_duration_spellings() {
        assert_eq!(
            MeasureDuration::new(3, 8).into_duration_prefer_dots(),
            Some(duration(DurationValue::Reciprocal(4), 1))
        );
        assert_eq!(
            MeasureDuration::new(7, 16).into_duration_prefer_dots(),
            Some(duration(DurationValue::Reciprocal(4), 2))
        );
        assert_eq!(
            MeasureDuration::new(5, 16).into_duration_prefer_dots(),
            Some(duration(
                DurationValue::Rational {
                    numerator: 16,
                    denominator: 5,
                },
                0
            ))
        );
    }

    #[test]
    fn kern_content_duration_rejects_unusable_chords() {
        let empty = KernContent::Chord(Chord {
            notes: Vec::new(),
            modifiers: ChordModifiers::default(),
        });
        assert_eq!(
            empty.duration(),
            Err(KernContentDurationError::ChordContainsNoNotes)
        );

        let mixed = KernContent::Chord(Chord {
            notes: vec![Note::try_from("4c").unwrap(), Note::try_from("8e").unwrap()],
            modifiers: ChordModifiers::default(),
        });
        assert_eq!(
            mixed.duration(),
            Err(KernContentDurationError::ChordContainsMixedDurations)
        );
    }
}
