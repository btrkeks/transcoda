//! Time signature parsing and representation.

use crate::duration::RhythmDuration;
use crate::error::ParseError;
use serde::Serialize;
use std::fmt;

/// A musical time signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct TimeSignature {
    /// Number of beats per measure (numerator).
    pub beats: u32,
    /// Note value that gets one beat (denominator).
    pub beat_unit: u32,
}

impl TimeSignature {
    /// Create a new time signature.
    pub fn new(beats: u32, beat_unit: u32) -> Self {
        TimeSignature { beats, beat_unit }
    }

    /// Parse a time signature from a **kern tandem interpretation.
    ///
    /// # Examples
    /// ```
    /// use rhythm_checker::time_signature::TimeSignature;
    ///
    /// let ts = TimeSignature::parse("*M4/4").unwrap();
    /// assert_eq!(ts, TimeSignature::new(4, 4));
    ///
    /// let ts = TimeSignature::parse("*M6/8").unwrap();
    /// assert_eq!(ts, TimeSignature::new(6, 8));
    /// ```
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        // Must start with *M
        let rest = s
            .strip_prefix("*M")
            .ok_or_else(|| ParseError::InvalidTimeSignature(format!("missing *M prefix: {}", s)))?;

        // Handle special cases like *MX (unmeasured) or *M? (unknown)
        if rest == "X" || rest == "?" {
            return Err(ParseError::InvalidTimeSignature(format!(
                "special time signature not supported: {}",
                s
            )));
        }

        // Split on /
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() != 2 {
            return Err(ParseError::InvalidTimeSignature(format!(
                "expected beats/unit format: {}",
                s
            )));
        }

        let beats: u32 = parts[0].parse().map_err(|_| {
            ParseError::InvalidTimeSignature(format!("invalid beats: {}", parts[0]))
        })?;

        let beat_unit: u32 = parts[1].parse().map_err(|_| {
            ParseError::InvalidTimeSignature(format!("invalid beat unit: {}", parts[1]))
        })?;

        if beats == 0 {
            return Err(ParseError::InvalidTimeSignature(
                "beats cannot be zero".to_string(),
            ));
        }

        if beat_unit == 0 || !beat_unit.is_power_of_two() {
            return Err(ParseError::InvalidTimeSignature(format!(
                "beat unit must be a power of 2: {}",
                beat_unit
            )));
        }

        Ok(TimeSignature { beats, beat_unit })
    }

    /// Returns true if this string is a time signature interpretation.
    pub fn is_time_signature(s: &str) -> bool {
        s.starts_with("*M") && s.len() > 2 && s.chars().nth(2).map_or(false, |c| c.is_ascii_digit())
    }

    /// Calculate the expected duration of a complete measure.
    ///
    /// # Examples
    /// ```
    /// use rhythm_checker::time_signature::TimeSignature;
    /// use rhythm_checker::duration::RhythmDuration;
    ///
    /// // 4/4 = 4 quarter notes = 1 whole note
    /// let ts = TimeSignature::new(4, 4);
    /// assert_eq!(ts.measure_duration(), RhythmDuration::new(1, 1));
    ///
    /// // 6/8 = 6 eighth notes = 3/4 of a whole note
    /// let ts = TimeSignature::new(6, 8);
    /// assert_eq!(ts.measure_duration(), RhythmDuration::new(3, 4));
    ///
    /// // 3/4 = 3 quarter notes = 3/4 of a whole note
    /// let ts = TimeSignature::new(3, 4);
    /// assert_eq!(ts.measure_duration(), RhythmDuration::new(3, 4));
    /// ```
    pub fn measure_duration(&self) -> RhythmDuration {
        // Each beat is 1/beat_unit of a whole note
        // Total duration = beats / beat_unit
        RhythmDuration::new(self.beats, self.beat_unit)
    }
}

impl Default for TimeSignature {
    /// Default to 4/4 time.
    fn default() -> Self {
        TimeSignature::new(4, 4)
    }
}

impl fmt::Display for TimeSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.beats, self.beat_unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_common_time_signatures() {
        assert_eq!(
            TimeSignature::parse("*M4/4").unwrap(),
            TimeSignature::new(4, 4)
        );
        assert_eq!(
            TimeSignature::parse("*M3/4").unwrap(),
            TimeSignature::new(3, 4)
        );
        assert_eq!(
            TimeSignature::parse("*M6/8").unwrap(),
            TimeSignature::new(6, 8)
        );
        assert_eq!(
            TimeSignature::parse("*M2/4").unwrap(),
            TimeSignature::new(2, 4)
        );
        assert_eq!(
            TimeSignature::parse("*M2/2").unwrap(),
            TimeSignature::new(2, 2)
        );
        assert_eq!(
            TimeSignature::parse("*M12/8").unwrap(),
            TimeSignature::new(12, 8)
        );
    }

    #[test]
    fn test_parse_invalid() {
        assert!(TimeSignature::parse("M4/4").is_err()); // missing *
        assert!(TimeSignature::parse("*M").is_err()); // no fraction
        assert!(TimeSignature::parse("*M4").is_err()); // no denominator
        assert!(TimeSignature::parse("*MX").is_err()); // unmeasured
        assert!(TimeSignature::parse("*M?").is_err()); // unknown
        assert!(TimeSignature::parse("*M0/4").is_err()); // zero beats
        assert!(TimeSignature::parse("*M4/3").is_err()); // non-power-of-2
    }

    #[test]
    fn test_measure_duration() {
        // 4/4 = 1 whole note
        assert_eq!(
            TimeSignature::new(4, 4).measure_duration(),
            RhythmDuration::new(1, 1)
        );

        // 3/4 = 3/4 of a whole note
        assert_eq!(
            TimeSignature::new(3, 4).measure_duration(),
            RhythmDuration::new(3, 4)
        );

        // 6/8 = 6/8 = 3/4 of a whole note
        assert_eq!(
            TimeSignature::new(6, 8).measure_duration(),
            RhythmDuration::new(3, 4)
        );

        // 2/2 = 2/2 = 1 whole note
        assert_eq!(
            TimeSignature::new(2, 2).measure_duration(),
            RhythmDuration::new(1, 1)
        );

        // 5/4 = 5/4 of a whole note
        assert_eq!(
            TimeSignature::new(5, 4).measure_duration(),
            RhythmDuration::new(5, 4)
        );
    }

    #[test]
    fn test_is_time_signature() {
        assert!(TimeSignature::is_time_signature("*M4/4"));
        assert!(TimeSignature::is_time_signature("*M6/8"));
        assert!(!TimeSignature::is_time_signature("*MX")); // starts with M but no digit
        assert!(!TimeSignature::is_time_signature("*clefG2"));
        assert!(!TimeSignature::is_time_signature("*^"));
    }

    #[test]
    fn test_display() {
        assert_eq!(TimeSignature::new(4, 4).to_string(), "4/4");
        assert_eq!(TimeSignature::new(6, 8).to_string(), "6/8");
    }
}
