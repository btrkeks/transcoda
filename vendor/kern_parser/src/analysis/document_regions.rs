//! Coarse document-region analysis for normalization passes.

use crate::KernDocument;

/// Return the half-open range containing the leading tandem header block.
///
/// The header starts after an optional exclusive interpretation line and
/// includes the initial contiguous run of tandem interpretation lines.
pub(crate) fn leading_header_bounds(document: &KernDocument) -> (usize, usize) {
    let range = document.leading_tandem_header_range();
    (range.start, range.end)
}

#[cfg(test)]
mod tests {
    use super::leading_header_bounds;
    use crate::KernDocument;

    #[test]
    fn finds_header_after_explicit_spine_type_line() {
        let document = KernDocument::parse("**kern\n*clefG2\n*k[]\n4c\n").unwrap();

        assert_eq!(leading_header_bounds(&document), (1, 3));
    }

    #[test]
    fn finds_wrapperless_header_at_start() {
        let document = KernDocument::parse("*clefG2\n*k[]\n4c\n").unwrap();

        assert_eq!(leading_header_bounds(&document), (0, 2));
    }

    #[test]
    fn header_stops_at_barline_data_or_spine_manipulator() {
        let document = KernDocument::parse("**kern\n*clefG2\n=\n*k[]\n4c\n").unwrap();
        assert_eq!(leading_header_bounds(&document), (1, 2));

        for input in [
            "**kern\n*clefG2\n4c\n*k[]\n",
            "**kern\n*clefG2\n*^\n*k[]\t*k[]\n4c\t4e\n",
        ] {
            let document = KernDocument::parse(input).unwrap();
            assert_eq!(leading_header_bounds(&document), (1, 3));
        }
    }
}
