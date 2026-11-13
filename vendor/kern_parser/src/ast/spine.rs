/// Zero-based spine/column index within a tab-separated Humdrum line.
///
/// In `4c\t4e`, `4c` is `SpineIndex(0)` and `4e` is `SpineIndex(1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpineIndex(pub usize);

/// Original text for a parsed token when the typed representation is partial.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RawToken(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpineType {
    /// **kern
    Kern,
    /// **dynam
    ///
    /// NOTE: This should not be serialized
    Dynam,
    /// **fing
    Fing,
    /// **text
    ///
    /// NOTE: This should not be serialized
    Text,
    /// **mxhm
    ///
    /// NOTE: This should not be serialized
    MxmlChords,
    /// Unknown or intentionally opaque exclusive interpretation.
    ///
    /// NOTE: This should be skipped during serialization
    Other(String),
}

/// Humdrum spine-path operation or null interpretation.
///
/// Examples: `*` keeps a spine unchanged, `*^` splits it, `*v\t*v` joins
/// adjacent spines, and `*-` terminates it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpineManipulator {
    /// `*`
    Null,
    /// `*-`
    Terminate,
    /// `*^`
    Split,
    /// `*v`
    Merge,
}
