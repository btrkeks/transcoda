//! Exact fractional duration arithmetic using rational numbers.

use num_rational::Ratio;
use serde::Serialize;
use std::fmt;
use std::ops::{Add, AddAssign, Sub};

/// A rhythmic duration represented as a fraction of a whole note.
///
/// Uses exact rational arithmetic to avoid floating-point errors.
/// For example:
/// - Quarter note = 1/4
/// - Dotted quarter = 3/8
/// - Triplet eighth = 1/12
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct RhythmDuration(Ratio<u32>);

impl RhythmDuration {
    /// Zero duration.
    pub const ZERO: Self = RhythmDuration(Ratio::new_raw(0, 1));

    /// Create a new duration from a ratio.
    pub fn new(numerator: u32, denominator: u32) -> Self {
        RhythmDuration(Ratio::new(numerator, denominator))
    }

    /// Create a duration from reciprocal notation (as used in **kern).
    ///
    /// # Arguments
    /// * `reciprocal` - The reciprocal value (4 = quarter note, 8 = eighth note, etc.)
    /// * `dots` - Number of augmentation dots
    ///
    /// # Examples
    /// ```
    /// use rhythm_checker::duration::RhythmDuration;
    ///
    /// // Quarter note: 1/4
    /// assert_eq!(RhythmDuration::from_reciprocal(4, 0), RhythmDuration::new(1, 4));
    ///
    /// // Dotted quarter: 1/4 + 1/8 = 3/8
    /// assert_eq!(RhythmDuration::from_reciprocal(4, 1), RhythmDuration::new(3, 8));
    ///
    /// // Double-dotted quarter: 1/4 + 1/8 + 1/16 = 7/16
    /// assert_eq!(RhythmDuration::from_reciprocal(4, 2), RhythmDuration::new(7, 16));
    /// ```
    pub fn from_reciprocal(reciprocal: u32, dots: u32) -> Self {
        if reciprocal == 0 {
            return Self::ZERO;
        }

        // Handle breve (0 = double whole note) - but we use 0 for zero duration above
        // In kern, breve is typically written as "0" but we'll handle that specially

        // Base duration: 1/reciprocal of a whole note
        let base = Ratio::new(1u32, reciprocal);

        // Apply dots: each dot adds half of the previous value
        // total = base * (2 - 1/2^dots) = base * (2^(dots+1) - 1) / 2^dots
        if dots == 0 {
            RhythmDuration(base)
        } else {
            let multiplier_num = (1u32 << (dots + 1)) - 1; // 2^(dots+1) - 1
            let multiplier_den = 1u32 << dots; // 2^dots
            RhythmDuration(base * Ratio::new(multiplier_num, multiplier_den))
        }
    }

    /// Create a duration from rational notation (e.g., `3%5` = 3/5 of a whole note).
    ///
    /// # Arguments
    /// * `numerator` - The numerator of the fraction
    /// * `denominator` - The denominator of the fraction
    /// * `dots` - Number of augmentation dots
    ///
    /// # Examples
    /// ```
    /// use rhythm_checker::duration::RhythmDuration;
    ///
    /// // 3/5 of a whole note
    /// assert_eq!(RhythmDuration::from_rational(3, 5, 0), RhythmDuration::new(3, 5));
    ///
    /// // Dotted 3/5: 3/5 * 3/2 = 9/10
    /// assert_eq!(RhythmDuration::from_rational(3, 5, 1), RhythmDuration::new(9, 10));
    /// ```
    pub fn from_rational(numerator: u32, denominator: u32, dots: u32) -> Self {
        if numerator == 0 || denominator == 0 {
            return Self::ZERO;
        }

        let base = Ratio::new(numerator, denominator);

        if dots == 0 {
            RhythmDuration(base)
        } else {
            let multiplier_num = (1u32 << (dots + 1)) - 1;
            let multiplier_den = 1u32 << dots;
            RhythmDuration(base * Ratio::new(multiplier_num, multiplier_den))
        }
    }

    /// Create a duration for a breve (double whole note).
    pub fn breve(dots: u32) -> Self {
        let base = Ratio::new(2u32, 1u32);
        if dots == 0 {
            RhythmDuration(base)
        } else {
            let multiplier_num = (1u32 << (dots + 1)) - 1;
            let multiplier_den = 1u32 << dots;
            RhythmDuration(base * Ratio::new(multiplier_num, multiplier_den))
        }
    }

    /// Create a duration for a whole note (semibreve).
    pub fn whole(dots: u32) -> Self {
        Self::from_reciprocal(1, dots)
    }

    /// Returns true if this duration is zero.
    pub fn is_zero(&self) -> bool {
        *self.0.numer() == 0
    }

    /// Get the numerator of the reduced fraction.
    pub fn numerator(&self) -> u32 {
        *self.0.numer()
    }

    /// Get the denominator of the reduced fraction.
    pub fn denominator(&self) -> u32 {
        *self.0.denom()
    }

    /// Get the underlying ratio.
    pub fn ratio(&self) -> Ratio<u32> {
        self.0
    }
}

impl Add for RhythmDuration {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        RhythmDuration(self.0 + rhs.0)
    }
}

impl AddAssign for RhythmDuration {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for RhythmDuration {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        RhythmDuration(self.0 - rhs.0)
    }
}

impl fmt::Display for RhythmDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self.0.denom() == 1 {
            write!(f, "{}", self.0.numer())
        } else {
            write!(f, "{}/{}", self.0.numer(), self.0.denom())
        }
    }
}

impl Serialize for RhythmDuration {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_durations() {
        // Whole note
        assert_eq!(
            RhythmDuration::from_reciprocal(1, 0),
            RhythmDuration::new(1, 1)
        );
        // Half note
        assert_eq!(
            RhythmDuration::from_reciprocal(2, 0),
            RhythmDuration::new(1, 2)
        );
        // Quarter note
        assert_eq!(
            RhythmDuration::from_reciprocal(4, 0),
            RhythmDuration::new(1, 4)
        );
        // Eighth note
        assert_eq!(
            RhythmDuration::from_reciprocal(8, 0),
            RhythmDuration::new(1, 8)
        );
        // Sixteenth note
        assert_eq!(
            RhythmDuration::from_reciprocal(16, 0),
            RhythmDuration::new(1, 16)
        );
    }

    #[test]
    fn test_dotted_durations() {
        // Dotted quarter: 1/4 + 1/8 = 3/8
        assert_eq!(
            RhythmDuration::from_reciprocal(4, 1),
            RhythmDuration::new(3, 8)
        );
        // Dotted half: 1/2 + 1/4 = 3/4
        assert_eq!(
            RhythmDuration::from_reciprocal(2, 1),
            RhythmDuration::new(3, 4)
        );
        // Double-dotted quarter: 1/4 + 1/8 + 1/16 = 7/16
        assert_eq!(
            RhythmDuration::from_reciprocal(4, 2),
            RhythmDuration::new(7, 16)
        );
        // Dotted eighth: 1/8 + 1/16 = 3/16
        assert_eq!(
            RhythmDuration::from_reciprocal(8, 1),
            RhythmDuration::new(3, 16)
        );
    }

    #[test]
    fn test_triplet_durations() {
        // Triplet half (3 = 3 in the time of 2 half notes = 1/3)
        assert_eq!(
            RhythmDuration::from_reciprocal(3, 0),
            RhythmDuration::new(1, 3)
        );
        // Triplet quarter (6 = 3 in time of 2 quarters = 1/6)
        assert_eq!(
            RhythmDuration::from_reciprocal(6, 0),
            RhythmDuration::new(1, 6)
        );
        // Triplet eighth (12 = 3 in time of 2 eighths = 1/12)
        assert_eq!(
            RhythmDuration::from_reciprocal(12, 0),
            RhythmDuration::new(1, 12)
        );
        // Triplet sixteenth (24 = 3 in time of 2 sixteenths = 1/24)
        assert_eq!(
            RhythmDuration::from_reciprocal(24, 0),
            RhythmDuration::new(1, 24)
        );
    }

    #[test]
    fn test_triplet_sums() {
        // 3 triplet eighths = 1 quarter
        let triplet_eighth = RhythmDuration::from_reciprocal(12, 0);
        let quarter = RhythmDuration::from_reciprocal(4, 0);
        assert_eq!(triplet_eighth + triplet_eighth + triplet_eighth, quarter);

        // 3 triplet quarters = 1 half
        let triplet_quarter = RhythmDuration::from_reciprocal(6, 0);
        let half = RhythmDuration::from_reciprocal(2, 0);
        assert_eq!(triplet_quarter + triplet_quarter + triplet_quarter, half);
    }

    #[test]
    fn test_addition() {
        let quarter = RhythmDuration::from_reciprocal(4, 0);
        let eighth = RhythmDuration::from_reciprocal(8, 0);
        // 1/4 + 1/8 = 3/8
        assert_eq!(quarter + eighth, RhythmDuration::new(3, 8));

        // Four quarters = whole
        let whole = RhythmDuration::from_reciprocal(1, 0);
        assert_eq!(quarter + quarter + quarter + quarter, whole);
    }

    #[test]
    fn test_subtraction() {
        let half = RhythmDuration::from_reciprocal(2, 0);
        let quarter = RhythmDuration::from_reciprocal(4, 0);
        assert_eq!(half - quarter, quarter);
    }

    #[test]
    fn test_breve() {
        let breve = RhythmDuration::breve(0);
        assert_eq!(breve, RhythmDuration::new(2, 1));

        let dotted_breve = RhythmDuration::breve(1);
        assert_eq!(dotted_breve, RhythmDuration::new(3, 1));
    }

    #[test]
    fn test_display() {
        assert_eq!(RhythmDuration::from_reciprocal(4, 0).to_string(), "1/4");
        assert_eq!(RhythmDuration::from_reciprocal(1, 0).to_string(), "1");
        assert_eq!(RhythmDuration::new(3, 8).to_string(), "3/8");
    }

    #[test]
    fn test_zero() {
        assert!(RhythmDuration::ZERO.is_zero());
        assert!(!RhythmDuration::from_reciprocal(4, 0).is_zero());
    }

    #[test]
    fn test_rational_basic() {
        // 3/5 of a whole note
        assert_eq!(
            RhythmDuration::from_rational(3, 5, 0),
            RhythmDuration::new(3, 5)
        );
        // 1/4 via rational should equal quarter note
        assert_eq!(
            RhythmDuration::from_rational(1, 4, 0),
            RhythmDuration::from_reciprocal(4, 0)
        );
        // 1/2 via rational should equal half note
        assert_eq!(
            RhythmDuration::from_rational(1, 2, 0),
            RhythmDuration::from_reciprocal(2, 0)
        );
    }

    #[test]
    fn test_rational_dotted() {
        // Dotted 3/5: 3/5 * 3/2 = 9/10
        assert_eq!(
            RhythmDuration::from_rational(3, 5, 1),
            RhythmDuration::new(9, 10)
        );
        // Double-dotted 3/5: 3/5 * 7/4 = 21/20
        assert_eq!(
            RhythmDuration::from_rational(3, 5, 2),
            RhythmDuration::new(21, 20)
        );
    }

    #[test]
    fn test_rational_zero_handling() {
        // Zero numerator
        assert_eq!(RhythmDuration::from_rational(0, 5, 0), RhythmDuration::ZERO);
        // Zero denominator
        assert_eq!(RhythmDuration::from_rational(3, 0, 0), RhythmDuration::ZERO);
        // Both zero
        assert_eq!(RhythmDuration::from_rational(0, 0, 0), RhythmDuration::ZERO);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn addition_is_commutative(
            a_recip in 1u32..64,
            b_recip in 1u32..64
        ) {
            let a = RhythmDuration::from_reciprocal(a_recip, 0);
            let b = RhythmDuration::from_reciprocal(b_recip, 0);
            prop_assert_eq!(a + b, b + a);
        }

        #[test]
        fn addition_is_associative(
            a_recip in 1u32..32,
            b_recip in 1u32..32,
            c_recip in 1u32..32
        ) {
            let a = RhythmDuration::from_reciprocal(a_recip, 0);
            let b = RhythmDuration::from_reciprocal(b_recip, 0);
            let c = RhythmDuration::from_reciprocal(c_recip, 0);
            prop_assert_eq!((a + b) + c, a + (b + c));
        }

        #[test]
        fn zero_is_identity(recip in 1u32..64, dots in 0u32..3) {
            let d = RhythmDuration::from_reciprocal(recip, dots);
            prop_assert_eq!(d + RhythmDuration::ZERO, d);
            prop_assert_eq!(RhythmDuration::ZERO + d, d);
        }
    }
}
