//! Repeat boundary tracking for anacrusis pairing validation.
//!
//! Handles the common case where an initial pickup measure combines with an
//! incomplete pre-repeat measure to form a complete measure when the piece repeats.

use crate::duration::RhythmDuration;

/// Error when repeat pairing validation fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepeatPairingError {
    /// Spine index where the error occurred.
    pub spine: usize,
    /// The initial pickup duration.
    pub pickup: RhythmDuration,
    /// The pre-repeat measure duration.
    pub pre_repeat: RhythmDuration,
    /// The sum of pickup + pre_repeat.
    pub sum: RhythmDuration,
    /// The expected complete measure duration.
    pub expected: RhythmDuration,
}

impl RepeatPairingError {
    /// Returns true if the sum is less than expected.
    pub fn is_short(&self) -> bool {
        self.sum < self.expected
    }

    /// Returns the difference between expected and sum.
    pub fn difference(&self) -> RhythmDuration {
        if self.expected >= self.sum {
            self.expected - self.sum
        } else {
            self.sum - self.expected
        }
    }
}

/// Tracks repeat boundaries to validate anacrusis pairing.
///
/// When a piece has an initial anacrusis (pickup), the last measure before
/// a repeat should be incomplete in a complementary way. For example, in 6/8:
///
/// ```text
/// 8ee          (1/8 pickup)
/// ...
/// 4a 8g 4f     (5/8 pre-repeat)
/// =:|!|:
/// ```
///
/// The pickup (1/8) + pre-repeat (5/8) = 6/8 (complete measure).
///
/// Also handles section anacrusis: in AABB form, after a segue marker (`=:|!|:`),
/// the B section may have its own pickup measure that pairs with a later repeat.
#[derive(Debug, Clone)]
pub struct RepeatTracker {
    /// Initial anacrusis durations (one per kern spine).
    initial_pickup: Vec<RhythmDuration>,
    /// Expected measure durations at the start of the repeat section.
    expected_duration: Vec<RhythmDuration>,
    /// Whether we have recorded an initial pickup.
    has_initial_pickup: bool,
    /// Whether we're expecting a section pickup after a start repeat.
    awaiting_section_pickup: bool,
    /// Section anacrusis durations (one per kern spine).
    section_pickup: Vec<RhythmDuration>,
    /// Whether we have recorded a section pickup.
    has_section_pickup: bool,
}

impl RepeatTracker {
    /// Create a new repeat tracker.
    pub fn new() -> Self {
        RepeatTracker {
            initial_pickup: Vec::new(),
            expected_duration: Vec::new(),
            has_initial_pickup: false,
            awaiting_section_pickup: false,
            section_pickup: Vec::new(),
            has_section_pickup: false,
        }
    }

    /// Record the initial anacrusis durations from the first measure.
    ///
    /// Call this when the first barline shows incomplete measures (pickup).
    pub fn record_initial_pickup(&mut self, durations: &[RhythmDuration]) {
        self.initial_pickup = durations.to_vec();
        self.has_initial_pickup = true;
    }

    /// Record the expected measure duration when entering a repeat section.
    ///
    /// Call this when encountering a start repeat marker.
    pub fn on_start_repeat(&mut self, expected: &[RhythmDuration]) {
        self.expected_duration = expected.to_vec();
    }

    /// Check if we have an initial pickup that can be paired.
    pub fn has_initial_pickup(&self) -> bool {
        self.has_initial_pickup
    }

    /// Get the initial pickup duration for a specific spine.
    pub fn get_initial_pickup(&self, spine_idx: usize) -> Option<RhythmDuration> {
        self.initial_pickup.get(spine_idx).copied()
    }

    /// Mark that we're awaiting a section pickup after a start repeat.
    ///
    /// Call this after encountering a start repeat marker when anacrusis is allowed.
    pub fn mark_awaiting_section_pickup(&mut self) {
        self.awaiting_section_pickup = true;
    }

    /// Check if we're expecting a section pickup.
    pub fn is_awaiting_section_pickup(&self) -> bool {
        self.awaiting_section_pickup
    }

    /// Clear the awaiting section pickup flag (e.g., when a full measure follows start repeat).
    pub fn clear_awaiting_section_pickup(&mut self) {
        self.awaiting_section_pickup = false;
    }

    /// Record section anacrusis durations.
    ///
    /// Call this when the first barline after a start repeat shows incomplete measures.
    pub fn record_section_pickup(&mut self, durations: &[RhythmDuration]) {
        self.section_pickup = durations.to_vec();
        self.has_section_pickup = true;
        self.awaiting_section_pickup = false;
    }

    /// Check if we have a section pickup that can be paired.
    pub fn has_section_pickup(&self) -> bool {
        self.has_section_pickup
    }

    /// Get the section pickup duration for a specific spine.
    pub fn get_section_pickup(&self, spine_idx: usize) -> Option<RhythmDuration> {
        self.section_pickup.get(spine_idx).copied()
    }

    /// Clear section pickup state after an end repeat.
    pub fn clear_section_pickup(&mut self) {
        self.section_pickup.clear();
        self.has_section_pickup = false;
        self.awaiting_section_pickup = false;
    }

    /// Validate that pre-repeat durations pair correctly with initial pickup.
    ///
    /// Returns errors for any spines where pickup + pre_repeat != expected.
    pub fn validate_end_repeat(&self, pre_repeat: &[RhythmDuration]) -> Vec<RepeatPairingError> {
        let mut errors = Vec::new();

        // If no initial pickup was recorded, we can't validate pairing
        if !self.has_initial_pickup {
            return errors;
        }

        for (spine_idx, &pre) in pre_repeat.iter().enumerate() {
            let pickup = self
                .initial_pickup
                .get(spine_idx)
                .copied()
                .unwrap_or(RhythmDuration::ZERO);
            let expected = self
                .expected_duration
                .get(spine_idx)
                .copied()
                .unwrap_or(RhythmDuration::ZERO);

            // Skip if expected is zero (no time signature)
            if expected.is_zero() {
                continue;
            }

            let sum = pickup + pre;

            if sum != expected {
                errors.push(RepeatPairingError {
                    spine: spine_idx,
                    pickup,
                    pre_repeat: pre,
                    sum,
                    expected,
                });
            }
        }

        errors
    }
}

impl Default for RepeatTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_pairing() {
        // 6/8 time: 1/8 pickup + 5/8 pre-repeat = 6/8
        let mut tracker = RepeatTracker::new();

        let pickup = vec![RhythmDuration::new(1, 8)];
        let expected = vec![RhythmDuration::new(3, 4)]; // 6/8 = 3/4

        tracker.record_initial_pickup(&pickup);
        tracker.on_start_repeat(&expected);

        let pre_repeat = vec![RhythmDuration::new(5, 8)];
        let errors = tracker.validate_end_repeat(&pre_repeat);

        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn test_mismatched_pairing() {
        // 6/8 time: 1/8 pickup + 4/8 pre-repeat = 5/8 (should be 6/8)
        let mut tracker = RepeatTracker::new();

        let pickup = vec![RhythmDuration::new(1, 8)];
        let expected = vec![RhythmDuration::new(3, 4)];

        tracker.record_initial_pickup(&pickup);
        tracker.on_start_repeat(&expected);

        let pre_repeat = vec![RhythmDuration::new(4, 8)];
        let errors = tracker.validate_end_repeat(&pre_repeat);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].spine, 0);
        assert_eq!(errors[0].pickup, RhythmDuration::new(1, 8));
        assert_eq!(errors[0].pre_repeat, RhythmDuration::new(4, 8));
        assert_eq!(errors[0].sum, RhythmDuration::new(5, 8));
        assert_eq!(errors[0].expected, RhythmDuration::new(3, 4));
        assert!(errors[0].is_short());
    }

    #[test]
    fn test_multiple_spines() {
        // Two kern spines, each validated independently
        let mut tracker = RepeatTracker::new();

        let pickup = vec![RhythmDuration::new(1, 8), RhythmDuration::new(1, 4)];
        let expected = vec![RhythmDuration::new(3, 4), RhythmDuration::new(3, 4)];

        tracker.record_initial_pickup(&pickup);
        tracker.on_start_repeat(&expected);

        // First spine: 1/8 + 5/8 = 6/8 (correct)
        // Second spine: 1/4 + 1/2 = 3/4 (correct)
        let pre_repeat = vec![RhythmDuration::new(5, 8), RhythmDuration::new(1, 2)];
        let errors = tracker.validate_end_repeat(&pre_repeat);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_no_initial_pickup() {
        // If no pickup was recorded, no errors should be generated
        let tracker = RepeatTracker::new();

        let pre_repeat = vec![RhythmDuration::new(5, 8)];
        let errors = tracker.validate_end_repeat(&pre_repeat);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_has_initial_pickup() {
        let mut tracker = RepeatTracker::new();
        assert!(!tracker.has_initial_pickup());

        tracker.record_initial_pickup(&[RhythmDuration::new(1, 8)]);
        assert!(tracker.has_initial_pickup());
    }

    #[test]
    fn test_get_initial_pickup() {
        let mut tracker = RepeatTracker::new();
        tracker.record_initial_pickup(&[RhythmDuration::new(1, 8), RhythmDuration::new(1, 4)]);

        assert_eq!(
            tracker.get_initial_pickup(0),
            Some(RhythmDuration::new(1, 8))
        );
        assert_eq!(
            tracker.get_initial_pickup(1),
            Some(RhythmDuration::new(1, 4))
        );
        assert_eq!(tracker.get_initial_pickup(2), None);
    }

    #[test]
    fn test_section_pickup_tracking() {
        let mut tracker = RepeatTracker::new();

        // Initially no section pickup
        assert!(!tracker.is_awaiting_section_pickup());
        assert!(!tracker.has_section_pickup());

        // After start repeat, mark awaiting
        tracker.mark_awaiting_section_pickup();
        assert!(tracker.is_awaiting_section_pickup());
        assert!(!tracker.has_section_pickup());

        // Record section pickup
        tracker.record_section_pickup(&[RhythmDuration::new(1, 8)]);
        assert!(!tracker.is_awaiting_section_pickup()); // Cleared after recording
        assert!(tracker.has_section_pickup());
        assert_eq!(
            tracker.get_section_pickup(0),
            Some(RhythmDuration::new(1, 8))
        );
        assert_eq!(tracker.get_section_pickup(1), None);

        // Clear section pickup
        tracker.clear_section_pickup();
        assert!(!tracker.has_section_pickup());
        assert_eq!(tracker.get_section_pickup(0), None);
    }

    #[test]
    fn test_clear_awaiting_without_recording() {
        let mut tracker = RepeatTracker::new();

        tracker.mark_awaiting_section_pickup();
        assert!(tracker.is_awaiting_section_pickup());

        // Clear without recording (full measure after start repeat)
        tracker.clear_awaiting_section_pickup();
        assert!(!tracker.is_awaiting_section_pickup());
        assert!(!tracker.has_section_pickup());
    }
}
