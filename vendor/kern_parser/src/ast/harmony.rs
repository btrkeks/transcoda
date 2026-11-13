#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MxmlHarmonyContent {
    /// `.` null data token.
    Null,
    /// A harmony/chord label as written in the spine.
    ///
    /// Examples: `G major`, `A minor`, `D dominant`.
    Label(MxmlHarmonyLabel),
}

/// Opaque MusicXML-derived harmony/chord label.
///
/// Examples: `G major`, `A minor`, `D dominant`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MxmlHarmonyLabel {
    pub text: String,
}
