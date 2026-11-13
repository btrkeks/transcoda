//! Preserve layout after render lowering has corrected blocked staff signifiers.
pub(super) struct LayoutRules {
    signs: Vec<String>,
    retain_all: bool,
    dotted_cross_staff: bool,
}
impl LayoutRules {
    pub(super) fn from_text(text: &str) -> Self {
        let mut rules = Self {
            signs: Vec::new(),
            retain_all: false,
            dotted_cross_staff: false,
        };
        for line in text.lines() {
            let Some((sign, definition)) = line
                .strip_prefix("!!!RDF**kern")
                .and_then(|s| s.trim_start().strip_prefix(':'))
                .and_then(|s| s.split_once('='))
            else {
                continue;
            };
            // An RDF alias can give an accidental display character another
            // visual role, such as note color. Refuse marker edits in that score.
            rules.retain_all |= sign.contains(['X', 'n', 'y']);
            if !definition.contains("above") && !definition.contains("below") {
                continue;
            }
            let sign = sign.trim();
            // Verovio interpolates RDF signs into regexes without escaping.
            // Refuse edits for regex syntax rather than interpreting a different dialect.
            if sign.is_empty()
                || sign.contains([
                    '.', '*', '+', '?', '[', ']', '(', ')', '{', '}', '\\', '^', '$', '|',
                ])
            {
                rules.retain_all = true;
            } else {
                rules.signs.push(sign.into());
            }
        }
        // MapOfDotLocs is keyed by Staff pointers. Marker edits anywhere in a
        // score can perturb allocation and reorder the SVG ellipses of a dotted
        // chord spanning staves. Preserve source bytes for this layout because
        // allocator ordering cannot be modeled from musical state.
        rules.dotted_cross_staff = text
            .lines()
            .filter(|line| !line.starts_with(['*', '!', '=']))
            .flat_map(|line| line.split('\t'))
            .any(|field| {
                let heads: Vec<_> = field.split_whitespace().collect();
                heads.len() > 1
                    && heads.iter().any(|head| {
                        head.find(pitch)
                            .is_some_and(|end| head[..end].contains('.'))
                    })
                    && rules.signs.iter().any(|sign| {
                        let moved = heads.iter().filter(|head| signature(head, sign)[4]).count();
                        moved > 0 && moved < heads.len()
                    })
            });
        rules
    }
    pub(super) fn same(&self, candidate: &str, original: &str) -> bool {
        if self.retain_all {
            return candidate == original;
        }
        if self.dotted_cross_staff && !self.is_cross_staff(original) {
            return candidate == original;
        }
        let candidate = crate::serialize::render::lower_cross_staff_head(candidate);
        let original = crate::serialize::render::lower_cross_staff_head(original);
        self.signs
            .iter()
            .all(|sign| signature(&candidate, sign) == signature(&original, sign))
    }
    fn is_cross_staff(&self, token: &str) -> bool {
        self.signs.iter().any(|sign| signature(token, sign)[4])
    }
}
fn pitch(c: char) -> bool {
    matches!(c, 'a'..='g' | 'A'..='G')
}
fn accidental(c: char) -> bool {
    matches!(c, '#' | 'n' | '-')
}
fn marker(c: char) -> bool {
    matches!(c, 'x' | 'X' | 'y' | 'Y')
}
fn signature(token: &str, sign: &str) -> [bool; 6] {
    let mut result = [false; 6];
    for (index, _) in token.match_indices(sign) {
        let prefix = &token[..index];
        let direct = prefix.trim_end_matches(accidental).chars().next_back();
        // humlib checkDataForCrossStaffStems: [A-Ga-g]+[#n-]*SIGN
        result[0] |= direct.is_some_and(pitch);
        // iohumdrum getNoteStaff, used for ornaments: [a-g]+[-#n]*SIGN
        result[1] |= direct.is_some_and(|c| matches!(c, 'a'..='g'));
        // iohumdrum checkBeamWith: [A-Ga-gn#-][XxYy]?SIGN
        let beam = if prefix.chars().next_back().is_some_and(marker) {
            &prefix[..prefix.len() - 1]
        } else {
            prefix
        };
        result[2] |= beam
            .chars()
            .next_back()
            .is_some_and(|c| pitch(c) || accidental(c));
        // iohumdrum getStaffAdjustment: [A-Ga-gr][#n-]*[xXyY]*SIGN
        result[3] |= prefix
            .trim_end_matches(marker)
            .trim_end_matches(accidental)
            .chars()
            .next_back()
            .is_some_and(|c| pitch(c) || c == 'r');
        // iohumdrum convertNote: [ra-gA-G]+[-#nxXyY]*SIGN
        result[4] |= prefix
            .trim_end_matches(|c| accidental(c) || marker(c))
            .chars()
            .next_back()
            .is_some_and(|c| pitch(c) || c == 'r');
        // iohumdrum processSlurs: [a-g]+[-n#]*[xy]*SIGN
        result[5] |= prefix
            .trim_end_matches(['x', 'y'])
            .trim_end_matches(accidental)
            .chars()
            .next_back()
            .is_some_and(|c| matches!(c, 'a'..='g'));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::LayoutRules;

    #[test]
    fn regex_signifiers_refuse_marker_edits_conservatively() {
        let rules = LayoutRules::from_text("!!!RDF**kern: + = above\n");
        assert!(!rules.same("4f#", "4f#X"));
        assert!(rules.same("4f#X", "4f#X"));
    }
}
