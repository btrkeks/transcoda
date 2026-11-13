//! Same-pitch slurs are valid and intentionally do not produce a warning.

use std::collections::BTreeMap;

use crate::{
    analysis::{
        resolve_slur_spans, LooseEnd, LooseEndSide, SlurKey, SlurLocation, SlurSide, Spanner,
        UnpairedSlur,
    },
    document::Framing,
    score::Score,
    validation::{SourceLocation, ValidationSeverity, ValidationWarning},
    KernDocument, KernLine,
};

pub(crate) fn check(score: &Score<'_>, framing: Framing) -> Vec<ValidationWarning> {
    let document = score.document();
    let analysis = resolve_slur_spans(score);

    let mut markers = BTreeMap::<(SlurLocation, SlurKey), MarkerCounts>::new();
    for unpaired in &analysis.unpaired {
        let counts = markers
            .entry((unpaired.location, unpaired.key))
            .or_default();
        counts.staff = unpaired.staff;
        match unpaired.side {
            SlurSide::Close => counts.closes += 1,
            SlurSide::Open => counts.opens += 1,
        }
    }
    for counts in markers.values_mut() {
        counts.same_token = counts.opens.min(counts.closes);
        counts.opens -= counts.same_token;
        counts.closes -= counts.same_token;
    }

    let barlines = Barlines::new(document);
    let loose = match framing {
        Framing::Complete => ValidationSeverity::Error,
        Framing::Excerpt => ValidationSeverity::Warning,
    };
    let mut warnings = Vec::new();
    for (open, close) in unlinked_cross_staff_pairs(&barlines, &markers) {
        markers.get_mut(&(open.location, open.key)).unwrap().opens -= 1;
        markers
            .get_mut(&(close.location, close.key))
            .unwrap()
            .closes -= 1;
        warnings.push(ValidationWarning {
            span: None,
            code: "slur.unlinked-cross-staff",
            severity: loose,
            loose_end: None,
            location: SourceLocation {
                line: open.location.line,
                column: Some(open.location.column),
            },
            message: unlinked_cross_staff_message(close.location),
        });
    }

    for ((location, key), counts) in markers {
        let (close_code, open_code) = if key.linked {
            ("slur.linked-unpaired", "slur.linked-unpaired")
        } else {
            ("slur.unmatched-close", "slur.unclosed-open")
        };
        let findings = [
            (
                1,
                "slur.same-token",
                ValidationSeverity::Warning,
                None,
                same_token_message(counts.same_token),
            ),
            (
                counts.closes,
                close_code,
                loose,
                Some(LooseEndSide::Close),
                unmatched_close_message(counts.closes, key.linked),
            ),
            (
                counts.opens,
                open_code,
                loose,
                Some(LooseEndSide::Open),
                unclosed_open_message(counts.opens, key.linked),
            ),
        ];
        warnings.extend(
            findings
                .into_iter()
                .flat_map(|(count, code, severity, side, message)| {
                    message.into_iter().flat_map(move |message| {
                        std::iter::repeat_n((code, severity, side, message), count)
                    })
                })
                .map(|(code, severity, side, message)| ValidationWarning {
                    span: None,
                    code,
                    severity,
                    loose_end: side.zip(counts.staff).map(|(side, staff)| LooseEnd {
                        spanner: Spanner::Slur,
                        side,
                        staff,
                        pitch: None,
                    }),
                    location: SourceLocation {
                        line: location.line,
                        column: Some(location.column),
                    },
                    message,
                }),
        );
    }
    warnings.extend(
        analysis
            .spans
            .iter()
            .filter(|span| span.key.linked && span.close.line < span.open.line)
            .map(|span| ValidationWarning {
                span: None,
                code: "slur.linked-misordered",
                severity: ValidationSeverity::Error,
                loose_end: None,
                location: SourceLocation {
                    line: span.open.line,
                    column: Some(span.open.column),
                },
                message: format!(
                    "linked slur pairs with the linked close at line {}, which comes before it",
                    span.close.line
                ),
            }),
    );
    warnings.extend(analysis.spans.iter().filter_map(|span| {
        let count = barlines.between(span.open, span.close);
        (count >= LONG_SPAN_MIN_BARLINES).then(|| ValidationWarning {
            span: None,
            code: "slur.long-span",
            severity: loose,
            loose_end: None,
            location: SourceLocation {
                line: span.open.line,
                column: Some(span.open.column),
            },
            message: format!(
                "slur opened here pairs with the close at line {}, field {}, {count} barlines later; a span this long is usually a stale open that a later close claimed",
                span.close.line, span.close.column
            ),
        })
    }));
    warnings.sort_by_key(|warning| (warning.location.line, warning.location.column));
    warnings
}

#[derive(Debug, Default)]
struct MarkerCounts {
    same_token: usize,
    opens: usize,
    closes: usize,
    staff: Option<u32>,
}

const UNLINKED_CROSS_STAFF_MAX_BARLINES: usize = 8;
const LONG_SPAN_MIN_BARLINES: usize = 40;

struct Barlines {
    before_line: Vec<usize>,
}

impl Barlines {
    fn new(document: &KernDocument) -> Self {
        let mut seen = 0;
        Self {
            before_line: std::iter::once(0)
                .chain(document.lines().iter().map(|line| {
                    seen += usize::from(matches!(line, KernLine::Barline(_)));
                    seen
                }))
                .collect(),
        }
    }

    fn between(&self, open: SlurLocation, close: SlurLocation) -> usize {
        self.before_line[close.line_index].saturating_sub(self.before_line[open.line_index])
    }
}

fn unlinked_cross_staff_pairs(
    barlines: &Barlines,
    markers: &BTreeMap<(SlurLocation, SlurKey), MarkerCounts>,
) -> Vec<(UnpairedSlur, UnpairedSlur)> {
    let loose: Vec<UnpairedSlur> = markers
        .iter()
        .flat_map(|(&(location, key), counts)| {
            let marker = |side| UnpairedSlur {
                key,
                side,
                location,
                staff: counts.staff,
            };
            std::iter::repeat_n(marker(SlurSide::Close), counts.closes)
                .chain(std::iter::repeat_n(marker(SlurSide::Open), counts.opens))
        })
        .collect();
    loose
        .windows(2)
        .filter_map(|window| match *window {
            [open, close] if is_unlinked_cross_staff(barlines, open, close) => Some((open, close)),
            _ => None,
        })
        .collect()
}

fn is_unlinked_cross_staff(barlines: &Barlines, open: UnpairedSlur, close: UnpairedSlur) -> bool {
    let other_staff = matches!(
        (open.staff, close.staff),
        (Some(open_staff), Some(close_staff)) if open_staff != close_staff
    );
    !open.key.linked
        && !close.key.linked
        && open.side == SlurSide::Open
        && close.side == SlurSide::Close
        && other_staff
        && close.location > open.location
        && barlines.between(open.location, close.location) <= UNLINKED_CROSS_STAFF_MAX_BARLINES
}

fn unlinked_cross_staff_message(close: SlurLocation) -> String {
    let close = format!("line {}, field {}", close.line, close.column);
    format!(
        "slur opened here never closes on this staff, and a slur closes on another staff at {close} with no open; if they are one cross-staff slur, link them with `N(` here and `N)` at {close}; otherwise close this slur on its own staff and find the open for {close}"
    )
}

fn same_token_message(count: usize) -> Option<String> {
    match count {
        0 => None,
        1 => Some("slur opens and closes on the same token, so no slur is drawn".to_owned()),
        _ => Some(format!(
            "{count} slurs open and close on the same token, so no slur is drawn"
        )),
    }
}

fn unmatched_close_message(count: usize, linked: bool) -> Option<String> {
    match (count, linked) {
        (0, _) => None,
        (1, false) => Some("closing slur marker has no matching open slur".to_owned()),
        (_, false) => Some(format!(
            "{count} closing slur markers have no matching open slurs"
        )),
        (1, true) => Some("linked slur close has no linked open left to pair with".to_owned()),
        (_, true) => Some(format!(
            "{count} linked slur closes have no linked opens left to pair with"
        )),
    }
}

fn unclosed_open_message(count: usize, linked: bool) -> Option<String> {
    match (count, linked) {
        (0, _) => None,
        (1, false) => Some("slur opened here is never closed".to_owned()),
        (_, false) => Some(format!("{count} slurs opened here are never closed")),
        (1, true) => {
            Some("linked slur opened here has no linked close left to pair with".to_owned())
        }
        (_, true) => Some(format!(
            "{count} linked slurs opened here have no linked closes left to pair with"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warnings(input: &str) -> Vec<(usize, Option<usize>, String)> {
        let document = KernDocument::parse(input).unwrap();
        check(
            &Score::resolve(&document, Framing::Excerpt).unwrap(),
            Framing::Excerpt,
        )
        .into_iter()
        .map(|warning| {
            (
                warning.location.line,
                warning.location.column,
                warning.message,
            )
        })
        .collect()
    }

    #[test]
    fn same_pitch_slur_is_allowed() {
        assert_eq!(warnings("**kern\n(4c\n4c)\n*-\n"), vec![]);
    }

    #[test]
    fn slur_closing_in_another_layer_of_the_staff_is_allowed() {
        assert_eq!(
            warnings("**kern\n*^\n(4c\t4e\n4d\t4f)\n*v\t*v\n*-\n"),
            vec![]
        );
    }

    #[test]
    fn warns_for_unmatched_close_at_marker_location() {
        assert_eq!(
            warnings("**kern\n4c)\n*-\n"),
            vec![(
                2,
                Some(1),
                "closing slur marker has no matching open slur".to_owned()
            )]
        );
    }

    #[test]
    fn warns_for_open_slur_that_is_never_closed() {
        assert_eq!(
            warnings("**kern\n(4c\n*-\n"),
            vec![(2, Some(1), "slur opened here is never closed".to_owned())]
        );
    }

    #[test]
    fn reports_each_unclosed_slur() {
        assert_eq!(
            warnings("**kern\n((4c\n*-\n"),
            vec![
                (
                    2,
                    Some(1),
                    "2 slurs opened here are never closed".to_owned()
                );
                2
            ]
        );
    }

    #[test]
    fn warns_once_for_a_slur_that_opens_and_closes_on_one_token() {
        assert_eq!(
            warnings("**kern\n4c\n(4d)\n*-\n"),
            vec![(
                3,
                Some(1),
                "slur opens and closes on the same token, so no slur is drawn".to_owned()
            )]
        );
    }

    #[test]
    fn warns_for_close_at_the_start_of_a_second_ending() {
        assert_eq!(
            warnings("**kern\n*>[A,A1,A,A2]\n*>A\n4f(\n*>A1\n4g)\n*>A2\n4a)\n*-\n"),
            vec![(
                8,
                Some(1),
                "closing slur marker has no matching open slur".to_owned()
            )]
        );
    }

    #[test]
    fn names_close_in_another_kern_spine_as_unlinked_cross_staff() {
        assert_eq!(
            warnings("**kern\t**kern\n(4C\t4c\n4D\t4d)\n*-\t*-\n"),
            vec![(
                2,
                Some(1),
                "slur opened here never closes on this staff, and a slur closes on another staff at line 3, field 2 with no open; if they are one cross-staff slur, link them with `N(` here and `N)` at line 3, field 2; otherwise close this slur on its own staff and find the open for line 3, field 2".to_owned()
            )]
        );
    }
}
