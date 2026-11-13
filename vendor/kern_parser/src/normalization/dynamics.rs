//! `**dynam` hairpin normalization.
//!
//! The parser already collapses repetition inside one token. This pass removes
//! the marks Verovio ignores across the tokens of one lane. A lane is a
//! `**dynam` spine, or one half of it inside a split, since Verovio pairs a
//! start only with an end in the same half:
//!
//! ```text
//! Before: <   <   [   [   ???
//! After:  <   <   [   .   .
//! ```
//!
//! A start without a usable end renders as text and must stay. An end or
//! continuation with nothing open goes when a
//! start of its direction came earlier in the lane; a leading end and a
//! trailing start stay, since a page split leaves those behind. A `Raw`
//! token's hairpin characters open and close hairpins like any other token's,
//! but the token itself is never edited. `???` parts go under
//! [`CanonicalizeDynamics::drop_placeholders`]. A token left with no marks
//! becomes `.`.

use std::collections::HashMap;

use crate::{
    normalization::NormalizationPass, DynamContent, DynamicHairpinDirection, KernDocument,
    KernLine, SpineContent, SpineManipulator, SpineType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalizeDynamics {
    pub drop_placeholders: bool,
}

impl Default for CanonicalizeDynamics {
    fn default() -> Self {
        Self {
            drop_placeholders: true,
        }
    }
}

/// Where Verovio pairs hairpin marks: one `**dynam` spine, and inside a split
/// one half of it, numbered from 1. Outside a split the spine is half 1, so
/// a start opened before `*^` closes in the first half.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Lane {
    spine: usize,
    half: usize,
}

/// A data field, by line index and column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Site {
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Default)]
struct Hairpin {
    open: Option<Site>,
    interrupted: Option<(Site, Site)>,
    started: bool,
}

/// What one token's hairpin marks did to its lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepEvent {
    /// An end with nothing open after an earlier start in the lane.
    StrayEnd(DynamicHairpinDirection),
    /// A continuation with nothing open after an earlier start in the lane.
    StrayContinue(DynamicHairpinDirection),
    /// An end with nothing open and no earlier start in the lane.
    LeadingEnd(DynamicHairpinDirection),
    /// An endpoint cannot reach its start because another token stopped lookup.
    IgnoredEnd(DynamicHairpinDirection, Site, Site),
}

/// Hairpin state per lane, following spine splits and merges.
#[derive(Debug, Default)]
pub(crate) struct HairpinLanes {
    columns: Vec<Option<Lane>>,
    spines: usize,
    hairpins: HashMap<(Lane, DynamicHairpinDirection), Hairpin>,
}

impl HairpinLanes {
    /// Follow `line` if it declares or manipulates spines.
    pub(crate) fn follow(&mut self, line: &KernLine) {
        let spines: Vec<Option<usize>> = match line {
            KernLine::SpineTypeLine(types) => types
                .iter()
                .map(|spine_type| {
                    (*spine_type == SpineType::Dynam).then(|| {
                        self.spines += 1;
                        self.spines
                    })
                })
                .collect(),
            KernLine::SpineManipulatorLine(line) => {
                let tokens = line.tokens();
                let mut next = Vec::new();
                let mut index = 0;
                while index < tokens.len() {
                    let spine = self
                        .columns
                        .get(index)
                        .copied()
                        .flatten()
                        .map(|lane| lane.spine);
                    match tokens[index] {
                        SpineManipulator::Split => next.extend([spine, spine]),
                        SpineManipulator::Merge => {
                            next.push(spine);
                            index += tokens[index..]
                                .iter()
                                .take_while(|token| **token == SpineManipulator::Merge)
                                .count();
                            continue;
                        }
                        SpineManipulator::Terminate => {}
                        SpineManipulator::Null => next.push(spine),
                    }
                    index += 1;
                }
                next
            }
            KernLine::TandemInterpretationLine(_)
            | KernLine::LocalCommentLine(_)
            | KernLine::Barline(_)
            | KernLine::Data(_) => return,
        };
        let mut halves: HashMap<usize, usize> = HashMap::new();
        for spine in spines.iter().flatten() {
            *halves.entry(*spine).or_default() += 1;
        }
        let mut seen: HashMap<usize, usize> = HashMap::new();
        self.columns = spines
            .iter()
            .map(|spine| {
                spine.map(|spine| {
                    let half = seen.entry(spine).or_default();
                    *half += 1;
                    Lane { spine, half: *half }
                })
            })
            .collect();
        // A half that merged away is a new lane if the spine splits again.
        self.hairpins.retain(|(lane, _), _| {
            lane.half <= halves.get(&lane.spine).copied().unwrap_or(0).max(1)
        });
    }

    pub(crate) fn lane(&self, column: usize) -> Option<Lane> {
        self.columns.get(column).copied().flatten()
    }

    /// Ends first, then the level, then starts, the way Verovio reads a token.
    pub(crate) fn step(
        &mut self,
        lane: Lane,
        site: Site,
        content: &DynamContent,
    ) -> Vec<StepEvent> {
        let (marks, has_letters) = match content {
            DynamContent::Null => return Vec::new(),
            DynamContent::Marks(marks) => (marks, marks.level.is_some() || marks.silence),
            DynamContent::Raw { text, marks } => {
                (marks, text.0.chars().any(|ch| ch.is_ascii_alphabetic()))
            }
        };
        // Verovio looks for `<` before `>`, so in a token with both starts
        // (only a Raw token has them) the diminuendo start is dead. An end
        // glued after it still ends a diminuendo.
        let dead = |direction| {
            direction == DynamicHairpinDirection::Diminuendo && marks.crescendo.begin.is_some()
        };
        let mut events = Vec::new();
        for direction in DynamicHairpinDirection::ALL {
            let hairpin = self.hairpins.entry((lane, direction)).or_default();
            let slots = marks.hairpin(direction);
            let single_note = slots.begin.is_some_and(|begin| begin.single_note);
            let ends = slots.end.is_some() || single_note;
            if ends && hairpin.open.take().is_none() {
                if let Some((start, interrupted)) = hairpin.interrupted.take() {
                    if !single_note {
                        events.push(StepEvent::IgnoredEnd(direction, start, interrupted));
                    }
                } else if !single_note {
                    events.push(if hairpin.started {
                        StepEvent::StrayEnd(direction)
                    } else {
                        StepEvent::LeadingEnd(direction)
                    });
                }
            }
            if slots.continues.is_some() && hairpin.open.is_none() && hairpin.started {
                events.push(StepEvent::StrayContinue(direction));
            }
        }
        if has_letters || marks.crescendo.begin.is_some() || marks.diminuendo.begin.is_some() {
            for direction in DynamicHairpinDirection::ALL {
                if let Some(hairpin) = self.hairpins.get_mut(&(lane, direction)) {
                    if let Some(start) = hairpin.open.take() {
                        hairpin.interrupted = Some((start, site));
                    }
                }
            }
        }
        for direction in DynamicHairpinDirection::ALL {
            let opens = marks
                .hairpin(direction)
                .begin
                .is_some_and(|begin| !begin.single_note);
            if opens && !dead(direction) {
                let hairpin = self.hairpins.entry((lane, direction)).or_default();
                hairpin.open = Some(site);
                hairpin.interrupted = None;
                hairpin.started = true;
            }
        }
        events
    }
}

impl NormalizationPass for CanonicalizeDynamics {
    fn normalize(&self, document: &mut KernDocument) {
        let mut lanes = HairpinLanes::default();
        let lines = document.lines_mut();
        for (line, record) in lines.iter_mut().enumerate() {
            lanes.follow(record);
            let KernLine::Data(contents) = record else {
                continue;
            };
            for (column, content) in contents.iter_mut().enumerate() {
                let (Some(lane), SpineContent::Dynam(content)) = (lanes.lane(column), content)
                else {
                    continue;
                };
                if let (true, DynamContent::Marks(marks)) = (self.drop_placeholders, &mut *content)
                {
                    marks.placeholder = false;
                }
                for event in lanes.step(lane, Site { line, column }, content) {
                    match (event, &mut *content) {
                        (StepEvent::StrayEnd(direction), DynamContent::Marks(marks)) => {
                            marks.hairpin_mut(direction).end = None;
                        }
                        (StepEvent::StrayContinue(direction), DynamContent::Marks(marks)) => {
                            marks.hairpin_mut(direction).continues = None;
                        }
                        _ => {}
                    }
                }
                null_if_empty(content);
            }
        }
    }
}

fn null_if_empty(content: &mut DynamContent) {
    if matches!(content, DynamContent::Marks(marks) if marks.is_empty()) {
        *content = DynamContent::Null;
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        normalization::{dynamics::CanonicalizeDynamics, NormalizationPass},
        KernDocument,
    };

    /// The rendered text without the clef, key and RDF records the renderer adds.
    fn rendered(document: &KernDocument) -> String {
        crate::serialize::render::to_string(document)
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with("*clef") && !line.starts_with("!!!"))
            .filter(|line| !line.starts_with("*k["))
            .map(|line| format!("{line}\n"))
            .collect()
    }

    fn normalized(input: &str) -> String {
        let mut document = KernDocument::parse(input).unwrap();
        CanonicalizeDynamics::default().normalize(&mut document);
        rendered(&document)
    }

    fn dynam_column(input: &str) -> Vec<String> {
        normalized(input)
            .lines()
            .filter(|line| !line.starts_with(['*', '=', '!']))
            .map(|line| line.split('\t').skip(1).collect::<Vec<_>>().join("\t"))
            .collect()
    }

    #[test]
    fn a_start_before_another_start_stays_as_text() {
        assert_eq!(
            dynam_column("**kern\t**dynam\n4c\t<\n4d\t<\n4e\t[\n*-\t*-\n"),
            ["<", "<", "["]
        );
    }

    #[cfg(feature = "engraving")]
    #[test]
    fn normalization_preserves_text_and_wedges_in_verovio() {
        fn appearance(source: &str) -> (Vec<String>, usize) {
            let mut toolkit = verovioxide::Toolkit::new().unwrap();
            toolkit.render_data(source, None).unwrap();
            let mei = toolkit.get_mei().unwrap();
            let xml = roxmltree::Document::parse(&mei).unwrap();
            let directions = xml
                .descendants()
                .filter(|node| node.has_tag_name("dir"))
                .map(|node| {
                    node.descendants()
                        .filter(|child| child.is_text())
                        .filter_map(|child| child.text())
                        .collect::<String>()
                })
                .collect();
            let hairpins = xml
                .descendants()
                .filter(|node| node.has_tag_name("hairpin"))
                .count();
            (directions, hairpins)
        }
        for (tokens, text, wedges) in [
            (["<", "<", "["], "cresc.", 1),
            ([">", ">", "]"], "decresc.", 1),
            (["<", "ff", "."], "cresc.", 0),
            (["<", ">", "]"], "cresc.", 1),
        ] {
            let input = format!(
                "**kern\t**dynam\n*clefG2\t*\n4c\t{}\n4d\t{}\n4e\t{}\n*-\t*-\n",
                tokens[0], tokens[1], tokens[2]
            );
            let expected = (vec![text.to_owned()], wedges);
            assert_eq!(appearance(&input), expected, "{input}");
            let mut document = KernDocument::parse(&input).unwrap();
            CanonicalizeDynamics::default().normalize(&mut document);
            assert_eq!(
                appearance(&crate::serialize::render::to_string(&document).unwrap()),
                expected,
                "{input}"
            );
        }
    }

    #[test]
    fn an_interrupted_endpoint_is_preserved_for_review() {
        let input = "**kern\t**dynam\n4c\t<\n4d\tff\n4e\t[\n*-\t*-\n";
        assert_normalizes(input, input);
    }

    #[test]
    fn a_second_end_after_the_hairpin_closed_goes() {
        assert_eq!(
            dynam_column("**kern\t**dynam\n4c\t<\n4d\t[\n4e\t[\n*-\t*-\n"),
            ["<", "[", "."]
        );
    }

    #[test]
    fn a_leading_end_stays() {
        assert_eq!(
            dynam_column("**kern\t**dynam\n4c\t[\n4d\t<\n4e\t[\n*-\t*-\n"),
            ["[", "<", "["]
        );
    }

    #[test]
    fn a_trailing_start_stays() {
        assert_eq!(
            dynam_column("**kern\t**dynam\n4c\t<\n4d\t[\n4e\t>\n*-\t*-\n"),
            ["<", "[", ">"]
        );
    }

    #[test]
    fn a_placeholder_goes_and_leaves_a_null() {
        assert_eq!(
            dynam_column("**kern\t**dynam\n4c\t???\n4d\t??? p\n*-\t*-\n"),
            [".", "p"]
        );
    }

    #[test]
    fn placeholders_stay_when_asked() {
        let mut document = KernDocument::parse("**kern\t**dynam\n4c\t???\n*-\t*-\n").unwrap();
        CanonicalizeDynamics {
            drop_placeholders: false,
        }
        .normalize(&mut document);
        assert_eq!(rendered(&document), "**kern\t**dynam\n4c\t???\n*-\t*-\n");
    }

    #[test]
    fn the_spine_is_followed_through_a_kern_split_and_merge() {
        let input = "**kern\t**dynam\n4c\t<\n*^\t*\n4d\t4f\t<\n*v\t*v\t*\n4e\t[\n4g\t[\n*-\t*-\n";
        assert_eq!(
            normalized(input),
            "**kern\t**dynam\n4c\t<\n*^\t*\n4d\t4f\t<\n*v\t*v\t*\n4e\t[\n4g\t.\n*-\t*-\n"
        );
    }

    #[test]
    fn an_end_in_the_second_half_leaves_a_start_from_before_the_split_open() {
        let input = "**kern\t**dynam\n4c\t<\n*\t*^\n4d\t.\t[\n*\t*v\t*v\n4e\t[\n*-\t*-\n";
        assert_normalizes(input, input);
    }

    #[test]
    fn each_dynam_spine_pairs_on_its_own() {
        let input = "**kern\t**dynam\t**kern\t**dynam\n4c\t<\t4e\t.\n4d\t.\t4f\t<\n4e\t[\t4g\t[\n*-\t*-\t*-\t*-\n";
        assert_eq!(normalized(input), input);
    }

    #[test]
    fn normalizing_twice_equals_normalizing_once() {
        let input =
            "**kern\t**dynam\n4c\t< <\n4d\t??? <\n4e\t( (\n4f\t[ [[\n4g\t[\n4a\t>\n*-\t*-\n";
        let mut once = KernDocument::parse(input).unwrap();
        CanonicalizeDynamics::default().normalize(&mut once);
        let mut twice = once.clone();
        CanonicalizeDynamics::default().normalize(&mut twice);
        assert_eq!(twice, once);
        assert_eq!(
            rendered(&once),
            "**kern\t**dynam\n4c\t<\n4d\t<\n4e\t(\n4f\t[[\n4g\t.\n4a\t>\n*-\t*-\n"
        );
    }

    #[test]
    fn the_omr_preset_canonicalizes_dynamics() {
        let mut document =
            KernDocument::parse("**kern\t**dynam\n4c\t<\n4d\t<\n4e\t[\n4f\t???\n*-\t*-\n").unwrap();
        document.normalize_for_omr();
        assert_eq!(
            rendered(&document),
            "**kern\t**dynam\n4c\t<\n4d\t<\n4e\t[\n4f\t.\n*-\t*-\n"
        );
    }

    fn assert_normalizes(input: &str, expected: &str) {
        let mut once = KernDocument::parse(input).unwrap();
        CanonicalizeDynamics::default().normalize(&mut once);
        assert_eq!(rendered(&once), expected);
        let mut twice = once.clone();
        CanonicalizeDynamics::default().normalize(&mut twice);
        assert_eq!(twice, once);
    }

    #[test]
    fn each_half_of_a_dynam_split_pairs_on_its_own() {
        let input = "**kern\t**dynam\n4c\t.\n*\t*^\n4d\t<\t<\n4e\t[\t.\n4f\t.\t[\n*\t*v\t*v\n4g\t.\n*-\t*-\n";
        assert_normalizes(input, input);
    }

    #[test]
    fn a_start_before_a_split_closes_in_the_first_half() {
        let input = "**kern\t**dynam\n4c\t<\n*\t*^\n4d\t[\t<\n4e\t.\t[\n*\t*v\t*v\n4g\t.\n*-\t*-\n";
        assert_normalizes(input, input);
    }

    #[test]
    fn a_half_that_merged_away_starts_fresh_on_the_next_split() {
        assert_normalizes(
            "**kern\t**dynam\n*\t*^\n4c\t.\t<\n*\t*v\t*v\n*\t*^\n4d\t.\t<\n4e\t.\t[\n*\t*v\t*v\n*-\t*-\n",
            "**kern\t**dynam\n*\t*^\n4c\t.\t<\n*\t*v\t*v\n*\t*^\n4d\t.\t<\n4e\t.\t[\n*\t*v\t*v\n*-\t*-\n",
        );
    }

    #[test]
    fn a_raw_token_closes_the_open_start() {
        let input = "**kern\t**dynam\n4c\t<\n4d\t[ [ f mf\n4e\t<\n4f\t[\n*-\t*-\n";
        assert_normalizes(input, input);
    }

    #[test]
    fn a_raw_start_preserves_the_earlier_text_dynamic() {
        assert_normalizes(
            "**kern\t**dynam\n4c\t<\n4d\t< f mf\n4e\t[\n*-\t*-\n",
            "**kern\t**dynam\n4c\t<\n4d\t< f mf\n4e\t[\n*-\t*-\n",
        );
        let input = "**kern\t**dynam\n4c\t< f mf\n4d\t<\n4e\t[\n*-\t*-\n";
        assert_normalizes(input, input);
    }

    #[test]
    fn a_raw_token_with_both_starts_opens_the_crescendo() {
        let input = "**kern\t**dynam\n4c\t<\n4d\t[\n4e\t> <\n4f\t[\n*-\t*-\n";
        assert_normalizes(input, input);
        assert_normalizes(
            "**kern\t**dynam\n4c\t<\n4d\tff > f <\n4e\t[\n*-\t*-\n",
            "**kern\t**dynam\n4c\t<\n4d\tff > f <\n4e\t[\n*-\t*-\n",
        );
    }

    #[test]
    fn an_end_glued_to_a_dead_diminuendo_start_still_ends_the_diminuendo() {
        assert_normalizes(
            "**kern\t**dynam\n4c\t>\n4d\t< >] f mf\n4e\t]\n4f\t[\n*-\t*-\n",
            "**kern\t**dynam\n4c\t>\n4d\t< >] f mf\n4e\t.\n4f\t[\n*-\t*-\n",
        );
    }
}
