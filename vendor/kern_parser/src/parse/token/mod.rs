mod assemble;
mod scan;
pub(crate) mod table;
#[cfg(test)]
mod tests;

use std::ops::Range;

pub(crate) use assemble::{content, refusal, warnings};
pub(crate) use scan::{duration_spelling, Warning};
pub(crate) use table::{Marker, MarkerId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Part {
    First,
    Later,
}

impl Part {
    pub(crate) fn new(later: bool) -> Self {
        if later {
            Self::Later
        } else {
            Self::First
        }
    }
}

pub(crate) trait AtomSink {
    fn atom(&mut self, range: Range<usize>, id: MarkerId, row: &'static Marker);

    fn warn(&mut self, _span: Range<usize>, _warning: Warning) {}

    /// Later ranges are relative to the subtoken that starts at byte `start` of the token.
    fn subtoken(&mut self, _start: usize) {}
}

impl AtomSink for () {
    #[inline(always)]
    fn atom(&mut self, _: Range<usize>, _: MarkerId, _: &'static Marker) {}
}

pub(crate) fn char_span(text: &str, bytes: Range<usize>) -> Range<usize> {
    text[..bytes.start].chars().count()..text[..bytes.end].chars().count()
}
