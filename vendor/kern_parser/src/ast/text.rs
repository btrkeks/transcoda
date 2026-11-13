#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextContent {
    /// `.` null data token.
    Null,
    /// `%` silence/no vocal sound in Humdrum `**text`.
    Silence,
    /// `|` or `||` melisma continuation.
    Melisma {
        /// `||` indicates an explicitly printed dash during melisma.
        printed_dash: bool,
    },
    /// Lyric/text underlay as written in the spine.
    ///
    /// Examples: `In`, `te,`, `Do-`, `-ne,`, `-’ry`, `Through`.
    Text(TextUnderlay),
}

/// Opaque lyric/text underlay token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextUnderlay {
    pub text: String,
}
