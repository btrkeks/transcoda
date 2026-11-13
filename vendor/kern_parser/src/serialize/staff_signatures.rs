//! The clefs and key signatures each Verovio staff takes on in render output.
//!
//! Verovio opens every system with the clef and key signature in force on
//! each staff. A render-side audit that finds a system-start clef drawn
//! hidden or not at all has to tell a mistake from an author who hid the clef
//! with `yy` in every layer, and to point at the token that set it. Both need
//! the spine topology, which stays here instead of being re-derived from the
//! SVG or the text by the caller.

use serde::Serialize;

use crate::{score::Topology, KernDocument, KernLine, SpineType, TandemInterpretation};

use super::render::{RenderAddress, RenderOutput};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RenderSignatureKind {
    Clef,
    Key,
}

/// A clef or key signature a staff takes on at one line of render output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RenderStaffSignature {
    /// MEI `staff/@n`, as Verovio numbers staves.
    pub staff: u32,
    pub kind: RenderSignatureKind,
    /// The rendered token Verovio takes the staff's state from: the visible
    /// one in the last layer that writes one, else the last that writes one.
    pub rendered: RenderAddress,
    /// The written token `rendered` comes from, when it has one.
    pub source: Option<RenderAddress>,
    /// The token as written, without `yy`.
    pub value: String,
    /// Every layer that writes the token hides it with `yy`.
    pub hidden_in_every_layer: bool,
}

/// Every clef and key signature of the `**kern` staves in `output`, in line
/// order. Output whose topology does not build has none.
pub fn staff_signatures(output: &RenderOutput) -> Vec<RenderStaffSignature> {
    let Ok(document) = KernDocument::parse(&output.canonical_text()) else {
        return Vec::new();
    };
    let Ok(topology) = Topology::build(&document) else {
        return Vec::new();
    };
    let source_of = |rendered: RenderAddress| {
        output
            .token_provenance()
            .iter()
            .find(|token| token.rendered == rendered)
            .and_then(|token| token.source)
    };

    let mut signatures = Vec::new();
    for (line_index, line) in document.lines().iter().enumerate() {
        let KernLine::TandemInterpretationLine(tandems) = line else {
            continue;
        };
        let tokens = tandems.tokens();
        let line_number = document.source(line_index).unwrap_or(line_index + 1);
        let mut column = 0;
        while column < tokens.len() {
            let track = topology.track_of(topology.layer_at(line_index, column));
            let columns = topology.track_columns(line_index, track);
            column = columns.end;
            let track = topology.track(track);
            let (SpineType::Kern, Some(staff)) = (&track.spine_type, track.staff) else {
                continue;
            };
            let address = |column: usize| RenderAddress::field(line_number, column + 1);

            let clefs = columns
                .clone()
                .filter_map(|column| match &tokens[column] {
                    TandemInterpretation::Clef(clef) => Some((column, clef)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if let Some(&(drawn, clef)) = clefs
                .iter()
                .rev()
                .find(|(_, clef)| !clef.invisible)
                .or(clefs.last())
            {
                let rendered = address(drawn);
                signatures.push(RenderStaffSignature {
                    staff,
                    kind: RenderSignatureKind::Clef,
                    rendered,
                    source: source_of(rendered),
                    value: crate::Clef {
                        invisible: false,
                        ..clef.clone()
                    }
                    .as_kern(),
                    hidden_in_every_layer: clefs.iter().all(|(_, clef)| clef.invisible),
                });
            }

            if let Some((written, key)) =
                columns
                    .clone()
                    .rev()
                    .find_map(|column| match &tokens[column] {
                        TandemInterpretation::KeySignature(key) => Some((column, *key)),
                        _ => None,
                    })
            {
                let rendered = address(written);
                signatures.push(RenderStaffSignature {
                    staff,
                    kind: RenderSignatureKind::Key,
                    rendered,
                    source: source_of(rendered),
                    value: key.as_kern(),
                    hidden_in_every_layer: false,
                });
            }
        }
    }
    signatures
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::render::{to_render_output, RenderSerializeOptions};

    fn signatures(text: &str) -> Vec<RenderStaffSignature> {
        let document = KernDocument::parse(text).unwrap();
        staff_signatures(&to_render_output(&document, RenderSerializeOptions::default()).unwrap())
    }

    fn clefs(text: &str) -> Vec<(u32, String, bool)> {
        signatures(text)
            .into_iter()
            .filter(|signature| signature.kind == RenderSignatureKind::Clef)
            .map(|signature| {
                (
                    signature.staff,
                    signature.value,
                    signature.hidden_in_every_layer,
                )
            })
            .collect()
    }

    #[test]
    fn numbers_staves_as_verovio_does() {
        assert_eq!(
            clefs("**kern\t**kern\n*clefF4\t*clefG2\n4C\t4c\n*-\t*-\n"),
            [
                (2, "*clefF4".to_owned(), false),
                (1, "*clefG2".to_owned(), false)
            ]
        );
    }

    #[test]
    fn a_split_spine_clef_change_is_read_from_the_last_subspine() {
        let text = "**kern\t**kern\n*clefG2\t*clefG2\n*^\t*\n4c\t4e\t4g\n*clefF4\t*\t*\n4C\t4E\t4g\n*v\t*v\t*\n*-\t*-\n";
        let change = signatures(text)
            .into_iter()
            .filter(|signature| signature.kind == RenderSignatureKind::Clef)
            .find(|signature| signature.value == "*clefF4")
            .unwrap();
        assert_eq!(change.staff, 2);
        assert_eq!(change.rendered.field, 2);
        assert!(!change.hidden_in_every_layer);
        assert_eq!(change.source, Some(RenderAddress::field(5, 1)));
    }

    #[test]
    fn a_clef_hidden_in_every_layer_is_deliberate() {
        assert!(
            clefs("**kern\n*clefG2\n4c\n*^\n*clefF4yy\t*clefF4yy\n4C\t4E\n*v\t*v\n*-\n")
                .contains(&(1, "*clefF4".to_owned(), true))
        );
    }

    #[test]
    fn reports_key_signatures_per_staff() {
        let keys = signatures("**kern\t**kern\n*clefF4\t*clefG2\n*k[b-]\t*k[b-]\n4C\t4c\n*-\t*-\n")
            .into_iter()
            .filter(|signature| signature.kind == RenderSignatureKind::Key)
            .map(|signature| (signature.staff, signature.value))
            .collect::<Vec<_>>();
        assert_eq!(keys, [(2, "*k[b-]".to_owned()), (1, "*k[b-]".to_owned())]);
    }
}
