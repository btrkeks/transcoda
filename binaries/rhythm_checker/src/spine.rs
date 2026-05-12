//! Spine state tracking for **kern files.
//!
//! Handles spine splits (*^), merges (*v), and exchange (*x) operations.

use crate::duration::RhythmDuration;
use crate::error::ParseError;
use crate::time_signature::TimeSignature;

/// State of a single spine during validation.
#[derive(Debug, Clone)]
pub struct SpineState {
    /// Current time signature for this spine (for next measure).
    pub time_signature: Option<TimeSignature>,
    /// Expected duration for the current measure (set at measure start).
    pub current_measure_expected: Option<RhythmDuration>,
    /// Time signature in effect for current measure (for error reporting).
    pub current_measure_time_sig: Option<TimeSignature>,
    /// Accumulated duration in the current measure.
    pub measure_duration: RhythmDuration,
    /// Whether the current measure has consumed any data tokens.
    ///
    /// This excludes tie carryover applied at barline reset so tandem
    /// interpretations immediately after a barline can still retarget the
    /// current measure.
    pub has_data_in_measure: bool,
    /// Current measure number (1-indexed).
    pub measure_number: u32,
    /// Whether this spine contains **kern data.
    pub is_kern: bool,
    /// Excess duration carried over from ties across barlines.
    pub tie_carryover: RhythmDuration,
}

impl SpineState {
    /// Create a new spine state.
    pub fn new(is_kern: bool) -> Self {
        SpineState {
            time_signature: None,
            current_measure_expected: None,
            current_measure_time_sig: None,
            measure_duration: RhythmDuration::ZERO,
            has_data_in_measure: false,
            measure_number: 1,
            is_kern,
            tie_carryover: RhythmDuration::ZERO,
        }
    }

    /// Create a kern spine.
    pub fn kern() -> Self {
        Self::new(true)
    }

    /// Create a non-kern spine.
    pub fn non_kern() -> Self {
        Self::new(false)
    }

    /// Add duration to the current measure.
    pub fn add_duration(&mut self, duration: RhythmDuration) {
        self.measure_duration += duration;
        self.has_data_in_measure = true;
    }

    /// Start a new measure, capturing current time signature as expected.
    pub fn start_measure(&mut self) {
        self.current_measure_expected = self.time_signature.map(|ts| ts.measure_duration());
        self.current_measure_time_sig = self.time_signature;
        self.has_data_in_measure = false;
    }

    /// Reset for a new measure (called at barline after validation).
    pub fn reset_measure(&mut self) {
        self.measure_duration = self.tie_carryover;
        self.tie_carryover = RhythmDuration::ZERO;
        self.measure_number += 1;
        // Capture current time signature for the new measure
        self.start_measure();
    }

    /// Get the expected measure duration based on time signature at measure start.
    pub fn expected_duration(&self) -> Option<RhythmDuration> {
        self.current_measure_expected
    }

    /// Get the time signature for error reporting.
    pub fn effective_time_signature(&self) -> Option<TimeSignature> {
        self.current_measure_time_sig
    }
}

/// Tracks the state of all spines in a **kern file.
#[derive(Debug)]
pub struct SpineTracker {
    /// State of each active spine.
    spines: Vec<SpineState>,
}

impl SpineTracker {
    /// Create a new tracker with no spines.
    pub fn new() -> Self {
        SpineTracker { spines: Vec::new() }
    }

    /// Initialize spines from exclusive interpretation line.
    ///
    /// # Example
    /// ```
    /// use rhythm_checker::spine::SpineTracker;
    ///
    /// let mut tracker = SpineTracker::new();
    /// tracker.initialize(&["**kern", "**dynam", "**kern"]);
    /// assert_eq!(tracker.spine_count(), 3);
    /// ```
    pub fn initialize(&mut self, tokens: &[&str]) {
        self.spines.clear();
        for token in tokens {
            let is_kern = *token == "**kern";
            self.spines.push(SpineState::new(is_kern));
        }
    }

    /// Get the number of active spines.
    pub fn spine_count(&self) -> usize {
        self.spines.len()
    }

    /// Get a reference to a spine by index.
    pub fn get(&self, index: usize) -> Option<&SpineState> {
        self.spines.get(index)
    }

    /// Get a mutable reference to a spine by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut SpineState> {
        self.spines.get_mut(index)
    }

    /// Get all kern spine indices.
    pub fn kern_indices(&self) -> Vec<usize> {
        self.spines
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_kern)
            .map(|(i, _)| i)
            .collect()
    }

    /// Iterator over all spines.
    pub fn iter(&self) -> impl Iterator<Item = &SpineState> {
        self.spines.iter()
    }

    /// Mutable iterator over all spines.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut SpineState> {
        self.spines.iter_mut()
    }

    /// Process tandem interpretation tokens to handle spine operations.
    ///
    /// Handles:
    /// - `*^` - spine split (one becomes two)
    /// - `*v` - spine merge (two or more become one)
    /// - `*x` - spine exchange (swap positions)
    /// - `*` - null interpretation (no change)
    pub fn process_tandem(&mut self, tokens: &[&str]) -> Result<(), ParseError> {
        // First, validate that token count matches current spine count
        // (except for split/merge operations which change it)

        let mut new_spines = Vec::new();
        let mut old_idx = 0;
        let mut token_idx = 0;

        while token_idx < tokens.len() && old_idx < self.spines.len() {
            let token = tokens[token_idx];

            if token == "*^" {
                // Split: duplicate this spine
                let spine = &self.spines[old_idx];
                new_spines.push(spine.clone());
                new_spines.push(spine.clone());
                old_idx += 1;
                token_idx += 1;
            } else if token == "*v" {
                // Merge: consume consecutive *v tokens, merge those spines
                let merge_start = old_idx;
                let mut merge_count = 0;

                while token_idx < tokens.len() && tokens[token_idx] == "*v" {
                    merge_count += 1;
                    token_idx += 1;
                }

                if merge_count < 2 {
                    return Err(ParseError::InvalidSpineOperation(
                        "merge requires at least 2 consecutive *v tokens".to_string(),
                    ));
                }

                if merge_start + merge_count > self.spines.len() {
                    return Err(ParseError::InvalidSpineOperation(format!(
                        "not enough spines to merge: have {}, need {}",
                        self.spines.len() - merge_start,
                        merge_count
                    )));
                }

                // Merge: take properties from first spine
                // Sum durations from all merged spines
                let mut merged = self.spines[merge_start].clone();
                for i in (merge_start + 1)..(merge_start + merge_count) {
                    // Note: we could sum durations, but typically merged spines
                    // should have the same content. Take max to be safe.
                    if self.spines[i].measure_duration > merged.measure_duration {
                        merged.measure_duration = self.spines[i].measure_duration;
                    }
                }
                new_spines.push(merged);
                old_idx += merge_count;
            } else if token == "*x" {
                // Exchange: swap with next spine
                if token_idx + 1 >= tokens.len() || tokens[token_idx + 1] != "*x" {
                    return Err(ParseError::InvalidSpineOperation(
                        "exchange requires paired *x tokens".to_string(),
                    ));
                }
                if old_idx + 1 >= self.spines.len() {
                    return Err(ParseError::InvalidSpineOperation(
                        "not enough spines to exchange".to_string(),
                    ));
                }

                // Swap the two spines
                new_spines.push(self.spines[old_idx + 1].clone());
                new_spines.push(self.spines[old_idx].clone());
                old_idx += 2;
                token_idx += 2;
            } else {
                // Regular tandem interpretation or null (*) - keep spine as-is
                new_spines.push(self.spines[old_idx].clone());
                old_idx += 1;
                token_idx += 1;
            }
        }

        // Handle case where tokens may add more spines than we have
        // (shouldn't happen with valid files, but be defensive)

        self.spines = new_spines;
        Ok(())
    }

    /// Update time signatures from tandem interpretation tokens.
    pub fn update_time_signatures(&mut self, tokens: &[&str]) {
        for (i, token) in tokens.iter().enumerate() {
            if let Some(spine) = self.spines.get_mut(i) {
                if TimeSignature::is_time_signature(token) {
                    if let Ok(ts) = TimeSignature::parse(token) {
                        spine.time_signature = Some(ts);
                        // If the measure has not consumed any data tokens yet,
                        // this tandem interpretation applies to the current measure.
                        if !spine.has_data_in_measure {
                            spine.current_measure_expected = Some(ts.measure_duration());
                            spine.current_measure_time_sig = Some(ts);
                        }
                    }
                }
            }
        }
    }

    /// Reset all spine measures (at barline).
    pub fn reset_all_measures(&mut self) {
        for spine in &mut self.spines {
            spine.reset_measure();
        }
    }

    /// Start all measures (capture current time signatures).
    pub fn start_all_measures(&mut self) {
        for spine in &mut self.spines {
            spine.start_measure();
        }
    }
}

impl Default for SpineTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize() {
        let mut tracker = SpineTracker::new();
        tracker.initialize(&["**kern", "**dynam", "**kern"]);

        assert_eq!(tracker.spine_count(), 3);
        assert!(tracker.get(0).unwrap().is_kern);
        assert!(!tracker.get(1).unwrap().is_kern);
        assert!(tracker.get(2).unwrap().is_kern);
    }

    #[test]
    fn test_kern_indices() {
        let mut tracker = SpineTracker::new();
        tracker.initialize(&["**kern", "**dynam", "**kern", "**text"]);

        assert_eq!(tracker.kern_indices(), vec![0, 2]);
    }

    #[test]
    fn test_spine_split() {
        let mut tracker = SpineTracker::new();
        tracker.initialize(&["**kern", "**dynam"]);

        tracker.process_tandem(&["*^", "*"]).unwrap();

        assert_eq!(tracker.spine_count(), 3);
        assert!(tracker.get(0).unwrap().is_kern);
        assert!(tracker.get(1).unwrap().is_kern);
        assert!(!tracker.get(2).unwrap().is_kern);
    }

    #[test]
    fn test_spine_merge() {
        let mut tracker = SpineTracker::new();
        tracker.initialize(&["**kern", "**kern", "**dynam"]);

        tracker.process_tandem(&["*v", "*v", "*"]).unwrap();

        assert_eq!(tracker.spine_count(), 2);
        assert!(tracker.get(0).unwrap().is_kern);
        assert!(!tracker.get(1).unwrap().is_kern);
    }

    #[test]
    fn test_spine_exchange() {
        let mut tracker = SpineTracker::new();
        tracker.initialize(&["**kern", "**dynam"]);

        tracker.process_tandem(&["*x", "*x"]).unwrap();

        assert_eq!(tracker.spine_count(), 2);
        assert!(!tracker.get(0).unwrap().is_kern);
        assert!(tracker.get(1).unwrap().is_kern);
    }

    #[test]
    fn test_time_signature_update() {
        let mut tracker = SpineTracker::new();
        tracker.initialize(&["**kern", "**kern"]);

        tracker.update_time_signatures(&["*M4/4", "*M3/4"]);

        assert_eq!(
            tracker.get(0).unwrap().time_signature,
            Some(TimeSignature::new(4, 4))
        );
        assert_eq!(
            tracker.get(1).unwrap().time_signature,
            Some(TimeSignature::new(3, 4))
        );
    }

    #[test]
    fn test_duration_accumulation() {
        let mut tracker = SpineTracker::new();
        tracker.initialize(&["**kern"]);

        let spine = tracker.get_mut(0).unwrap();
        spine.add_duration(RhythmDuration::from_reciprocal(4, 0));
        spine.add_duration(RhythmDuration::from_reciprocal(4, 0));

        assert_eq!(
            tracker.get(0).unwrap().measure_duration,
            RhythmDuration::new(1, 2)
        );
    }

    #[test]
    fn test_measure_reset() {
        let mut tracker = SpineTracker::new();
        tracker.initialize(&["**kern"]);

        {
            let spine = tracker.get_mut(0).unwrap();
            spine.add_duration(RhythmDuration::from_reciprocal(4, 0));
        }

        tracker.reset_all_measures();

        assert_eq!(
            tracker.get(0).unwrap().measure_duration,
            RhythmDuration::ZERO
        );
        assert_eq!(tracker.get(0).unwrap().measure_number, 2);
    }
}
