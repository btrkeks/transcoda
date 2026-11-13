use super::CrossStaffPlacement;
use super::{Breath, Duration, Fermata, Phrase, Pitch, Slur};

/// Rest token.
///
/// Examples: `4r`, `2.r`, `2.rr`, `4ryy`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rest {
    /// `4r` -> 4, `2.r` -> 2., `12r` -> 12
    pub duration: Duration,
    /// `4r>`, `4.r<yy`. RDF above/below markers place the rest on an adjacent staff.
    pub cross_staff: Option<CrossStaffPlacement>,
    /// `4re`, `4rg`, `16rF`, `8rGG`, `4rdd`, etc. Pitch letters after `r` encode visual staff placement.
    pub placement: Option<RestPlacement>,
    /// `2ryy`, `4.ryy`, `8reyy`, `4rGGyy`
    pub invisible: bool,
    /// `0r,`, `8re,`. A rest's breath mark takes no RDF placement.
    pub breath: Option<Breath>,
    /// `2r;`, `8re;`, `2.r;`
    pub fermata: Option<Fermata>,
    /// `4r(`, `4r)`, `&(4r`, `4r&)`, etc.
    pub slurs: Vec<Slur>,
    /// `{8.r`, `4r}`, `&{4r&}`.
    ///
    /// NOTE: This should not be serialized
    pub phrases: Vec<Phrase>,
}

impl Rest {
    /// An invisible rest of `duration` with no other marks, as `2ryy` parses.
    pub fn invisible(duration: Duration) -> Self {
        Self {
            duration,
            cross_staff: None,
            placement: None,
            invisible: true,
            breath: None,
            fermata: None,
            slurs: Vec::new(),
            phrases: Vec::new(),
        }
    }
}

/// Compact OMR-dialect multi-measure rest token.
///
/// `MM55r` means a printed 55-measure rest, not a duration value of 55.
/// Counts start at two; single full-measure rests use ordinary rest durations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiMeasureRest {
    pub measure_count: u32,
}

/// Transcoda v2 OMR-dialect durationless invisible rest slice.
///
/// `ryy` means "invisibly silent for this data-line time slice". It is not a
/// canonical duration-bearing `**kern` rest and must be rehydrated before
/// renderable Humdrum export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvisibleRestSlice {
    /// `ryy;`
    pub fermata: Option<Fermata>,
}

/// Visual staff placement for a rest.
///
/// The spelling reuses `**kern` pitch letters, but this is not a sounding pitch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestPlacement {
    pub position: Pitch,
}
