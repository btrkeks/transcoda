//! Canonical display overrides, patched at source notehead addresses.
use super::{analyze, AccidentalDisplay, AccidentalDisplayError, DisplayCause, NoteAddress};
use crate::{
    document::{Document, DocumentError, Framing, InputOptions},
    score::Score,
    serialize::render,
    ParseDialect,
};
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]

pub enum DisplayChangeKind {
    StripX,
    StripY,
    StripN,
    AddX,
    AddY,
    CrossStaffFix,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct DisplayChange {
    pub address: NoteAddress,
    pub source_address: NoteAddress,
    pub before: String,
    pub after: String,
    /// A replacement can strip more than one override. Each removed/added
    /// marker has a separate entry sharing the same before/after token.
    pub kind: DisplayChangeKind,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct DisplaySkip {
    pub address: NoteAddress,
    pub source_address: NoteAddress,
    pub token: String,
    pub reason: String,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]

pub struct GraceFollowers {
    pub total: usize,
    pub shown: usize,
    pub hidden_by_y: usize,
    pub other_hidden: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]

pub struct CanonicalAccidentalDisplay {
    pub text: String,
    pub changes: Vec<DisplayChange>,
    pub skipped: Vec<DisplaySkip>,
    pub grace_followers: GraceFollowers,
}

/// Lower a dialect for analysis and retain the address of every source head.
pub fn analyze_source_accidental_display(
    document: &Document,
) -> Result<Vec<AccidentalDisplay>, DocumentError> {
    let lowered = (document.options().dialect == ParseDialect::TranscodaV2)
        .then(|| {
            render::to_render_output(document.score(), render::RenderSerializeOptions::default())
        })
        .transpose()?;
    let text = lowered
        .as_ref()
        .map(|r| r.canonical_text())
        .unwrap_or_else(|| document.source().to_owned());
    let render_doc = Document::parse(&text, InputOptions::default())?;
    let analysis_error = |e: AccidentalDisplayError| {
        DocumentError::new("accidental.analysis", e.line, e.to_string())
    };
    let render_score = Score::resolve(render_doc.score(), Framing::Excerpt)
        .map_err(|e| analysis_error(e.into()))?;
    let mut notes = analyze(&render_doc, &render_score).map_err(analysis_error)?;
    if !document
        .score()
        .lines()
        .iter()
        .any(|line| matches!(line, crate::KernLine::Data(_)))
    {
        return Err(DocumentError::new(
            "accidental.empty-score",
            None,
            "document has no data records",
        ));
    }
    let provenance: BTreeMap<_, _> = lowered
        .as_ref()
        .map(|r| {
            r.note_provenance()
                .iter()
                .map(|p| (address(p.rendered), address(p.source)))
                .collect()
        })
        .unwrap_or_default();
    for note in &mut notes {
        note.source_address = Some(if lowered.is_some() {
            *provenance.get(&note.address).ok_or_else(|| {
                DocumentError::internal(
                    "accidental.provenance",
                    Some(note.address.line),
                    "missing source notehead",
                )
            })?
        } else {
            note.address
        });
    }
    if lowered.is_some() {
        let source = Score::resolve(document.score(), Framing::Excerpt)
            .map_err(|e| analysis_error(e.into()))?;
        let topology = source.topology();
        let mut source_parts = BTreeMap::new();
        for li in 0..document.score().lines().len() {
            for (fi, &layer) in topology.layers_at(li).iter().enumerate() {
                source_parts.insert(
                    (document.score().source(li).unwrap_or(li + 1), fi + 1),
                    topology.track(topology.track_of(layer)).part,
                );
            }
        }
        // Render lowering omits part labels. Restore their ownership through
        // note provenance while retaining the rendered staff and glyph state.
        let parts = notes
            .iter()
            .map(|note| {
                let source = note.source_address.unwrap();
                (note.address, source_parts[&(source.line, source.field)])
            })
            .collect();
        super::hidden::classify(&render_doc, &render_score, &mut notes, Some(&parts));
    }
    Ok(notes)
}
fn address(a: render::RenderNoteAddress) -> NoteAddress {
    NoteAddress {
        line: a.line_number,
        field: a.spine_index + 1,
        subtoken: a.chord_note_index.unwrap_or(0) + 1,
    }
}

struct Token {
    address: NoteAddress,
    source: NoteAddress,
    range: std::ops::Range<usize>,
    original: String,
    bare: String,
    pitch_end: usize,
    accidental: String,
    skipped: bool,
    kept_for_layout: bool,
}

/// Preserve the input's shown/glyph table, reaching a fixed point because
/// hidden notes do not seed the concurrent accidental state. Every byte outside
/// changed notehead tokens, including line endings and chord spacing, is kept.
pub fn canonicalize_accidental_display(
    document: &Document,
) -> Result<CanonicalAccidentalDisplay, DocumentError> {
    let target = analyze_source_accidental_display(document)?;
    canonicalize_to_display(document, &target, false)
}

/// Restore a page-approved table after assembly. Matching tokens stay untouched.
pub(crate) fn canonicalize_to_display(
    document: &Document,
    target: &[AccidentalDisplay],
    preserve_matching: bool,
) -> Result<CanonicalAccidentalDisplay, DocumentError> {
    let multi_spine = document.score().lines().iter().any(|line| {
        matches!(line, crate::KernLine::Data(contents)
            if contents.iter().filter(|content| matches!(content, crate::SpineContent::Kern(_))).count() > 1)
    });
    let editorial: Vec<_> = document
        .source()
        .lines()
        .filter_map(|line| {
            let (sign, value) = line.strip_prefix("!!!RDF**kern:")?.split_once('=')?;
            value.contains("editorial accidental").then(|| sign.trim())
        })
        .collect();
    let render_text = if document.options().dialect == ParseDialect::TranscodaV2 {
        render::to_render_output(document.score(), render::RenderSerializeOptions::default())?
            .canonical_text()
    } else {
        document.source().to_owned()
    };
    let layout = super::layout::LayoutRules::from_text(&render_text);
    let ranges = source_ranges(document.source());
    let mut tokens = Vec::new();
    let mut skipped = Vec::new();
    let mut grace_followers = GraceFollowers::default();
    for note in target {
        let source = note.source_address.unwrap();
        let range = ranges.get(&source).cloned().ok_or_else(|| {
            DocumentError::internal(
                "accidental.source-address",
                Some(source.line),
                "source notehead address is missing",
            )
        })?;
        let original = &document.source()[range.clone()];
        let protected = original.contains(['Z', 'i', 'I', 'j'])
            || editorial.iter().any(|sign| original.contains(sign))
            || note.accidental_layout
            || matches!(
                note.cause,
                DisplayCause::EditorialAccidental | DisplayCause::LayoutOverride
            );
        if protected {
            skipped.push(DisplaySkip {
                address: note.address,
                source_address: source,
                token: original.into(),
                reason: "editorial marker or accidental layout parameter".into(),
            });
        }
        if note.grace_predecessor_same_accidental {
            grace_followers.total += 1;
            if note.shown {
                grace_followers.shown += 1;
            } else if has_hide(original) {
                grace_followers.hidden_by_y += 1;
            } else {
                grace_followers.other_hidden += 1;
            }
        }
        let (bare, pitch_end, accidental) = bare_token(original)?;
        tokens.push(Token {
            address: note.address,
            source,
            range,
            original: original.into(),
            bare,
            pitch_end,
            accidental,
            skipped: protected,
            kept_for_layout: false,
        });
    }
    let mut replacements: Vec<_> = tokens
        .iter()
        .map(|t| {
            if preserve_matching {
                t.original.clone()
            } else if t.skipped {
                protected_natural(&t.original, false)
            } else {
                t.bare.clone()
            }
        })
        .collect();
    let mut seen = HashSet::new();
    loop {
        let text = patch(document.source(), &tokens, &replacements);
        use sha2::Digest;
        if !seen.insert(sha2::Sha256::digest(text.as_bytes())) {
            return Err(DocumentError::internal(
                "accidental.fixed-point",
                None,
                "canonical display overrides did not converge",
            ));
        }
        let current = Document::parse(&text, document.options())?;
        let actual = analyze_source_accidental_display(&current)?;
        if actual.len() != target.len()
            || actual
                .iter()
                .zip(target)
                .any(|(a, b)| a.address != b.address || a.source_address != b.source_address)
        {
            return Err(DocumentError::internal(
                "accidental.topology",
                None,
                "canonicalization changed notehead addresses",
            ));
        }
        let mut next = replacements.clone();
        for (i, ((note, desired), token)) in actual.iter().zip(target).zip(&mut tokens).enumerate()
        {
            if !layout.same(&note.token, &desired.token) {
                token.kept_for_layout = true;
            }
            if token.kept_for_layout {
                next[i] = token.original.clone();
                continue;
            }
            if token.skipped {
                if note.shown != desired.shown || note.glyph != desired.glyph {
                    next[i] = protected_natural(&token.original, true);
                }
                continue;
            }
            if preserve_matching && note.shown == desired.shown && note.glyph == desired.glyph {
                continue;
            }
            next[i] = if note.bare_glyph == desired.glyph {
                token.bare.clone()
            } else {
                let marker = if desired.shown {
                    if token.accidental.is_empty() {
                        "nX"
                    } else {
                        "X"
                    }
                } else if token.accidental.is_empty() {
                    "ny"
                } else {
                    "y"
                };
                let mut value = token.bare.clone();
                value.insert_str(token.pitch_end, marker);
                value
            };
        }
        if next == replacements {
            if let Some((a, b)) = actual
                .iter()
                .zip(target)
                .find(|(a, b)| a.shown != b.shown || a.glyph != b.glyph)
            {
                return Err(DocumentError::new(
                    "accidental.unrepresentable-display",
                    a.source_address.map(|s| s.line),
                    format!(
                        "smallest overrides cannot restore {:?}: input {} {:?}, candidate {} {:?}",
                        a.address, b.token, b.glyph, a.token, a.glyph
                    ),
                ));
            }
            for token in tokens.iter().filter(|t| t.kept_for_layout) {
                skipped.push(DisplaySkip {
                    address: token.address,
                    source_address: token.source,
                    token: token.original.clone(),
                    reason: "kept_for_layout".into(),
                });
            }
            let mut changes = Vec::new();
            for ((t, after), desired) in tokens.iter().zip(&replacements).zip(target) {
                if t.original == *after {
                    continue;
                }
                // Rhythm-derived beam spans can draw a natural on a head this
                // model classifies as suppressed. Do not erase that evidence.
                if multi_spine
                    && desired.cause == DisplayCause::TremoloSuppression
                    && t.original.contains('n')
                {
                    return Err(DocumentError::new(
                        "accidental.ambiguous-tremolo-natural",
                        Some(t.source.line),
                        "cannot prove natural display at a suppressed multi-spine tremolo head; rhythm-derived beam spans can draw it",
                    ));
                }
                for kind in change_kinds(&t.original, after) {
                    let kind = if render::lower_cross_staff_head(&t.original) != t.original
                        && matches!(kind, DisplayChangeKind::StripX | DisplayChangeKind::StripY)
                    {
                        DisplayChangeKind::CrossStaffFix
                    } else {
                        kind
                    };
                    changes.push(DisplayChange {
                        address: t.address,
                        source_address: t.source,
                        before: t.original.clone(),
                        after: after.clone(),
                        kind,
                    });
                }
            }
            return Ok(CanonicalAccidentalDisplay {
                text,
                changes,
                skipped,
                grace_followers,
            });
        }
        replacements = next;
    }
}

pub(crate) fn bare_token(token: &str) -> Result<(String, usize, String), DocumentError> {
    let start = token
        .find([
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        ])
        .ok_or_else(|| {
            DocumentError::internal("accidental.token", None, "missing pitch spelling")
        })?;
    let bytes = token.as_bytes();
    let mut pitch_end = start + 1;
    while pitch_end < bytes.len() && bytes[pitch_end] == bytes[start] {
        pitch_end += 1;
    }
    let mut bare = String::new();
    let mut accidental = String::new();
    let mut insertion = pitch_end;
    for (i, c) in token.char_indices() {
        let after_accidental = i > 0 && matches!(bytes[i - 1], b'#' | b'-' | b'n');
        if c == 'n'
            || (c == 'X' && after_accidental)
            || (c == 'y' && after_accidental && bytes.get(i + 1) != Some(&b'y'))
        {
            continue;
        }
        bare.push(c);
        if c == '#' || c == '-' {
            accidental.push(c);
            insertion = bare.len();
        }
    }
    Ok((bare, insertion, accidental))
}

fn protected_natural(token: &str, force: bool) -> String {
    if token.contains(['Z', 'j']) {
        return token.to_owned();
    }
    let bytes = token.as_bytes();
    let mut output = String::new();
    for (i, c) in token.char_indices() {
        if c == 'n'
            && bytes.get(i + 1) != Some(&b'X')
            && !(bytes.get(i + 1) == Some(&b'y') && bytes.get(i + 2) != Some(&b'y'))
        {
            if force {
                output.push_str("nX");
            }
        } else {
            output.push(c);
        }
    }
    output
}
fn has_hide(token: &str) -> bool {
    !token.contains("yy") && ["#y", "-y", "ny"].iter().any(|s| token.contains(s))
}
fn change_kinds(before: &str, after: &str) -> Vec<DisplayChangeKind> {
    let mut kinds = Vec::new();
    for (marker, strip, add) in [
        ('X', DisplayChangeKind::StripX, DisplayChangeKind::AddX),
        ('y', DisplayChangeKind::StripY, DisplayChangeKind::AddY),
    ] {
        let old = before.matches(marker).count();
        let new = after.matches(marker).count();
        kinds.extend(std::iter::repeat_n(
            if old > new { strip } else { add },
            old.abs_diff(new),
        ));
    }
    kinds.extend(std::iter::repeat_n(
        DisplayChangeKind::StripN,
        before
            .matches('n')
            .count()
            .saturating_sub(after.matches('n').count()),
    ));
    kinds
}
pub(crate) fn source_ranges(text: &str) -> BTreeMap<NoteAddress, std::ops::Range<usize>> {
    let mut ranges = BTreeMap::new();
    let mut line_offset = 0;
    for (li, line) in text.split_inclusive('\n').enumerate() {
        let mut field_offset = line_offset;
        for (fi, field) in line.trim_end_matches(['\r', '\n']).split('\t').enumerate() {
            let mut spans = Vec::new();
            let mut start = None;
            for (i, c) in field.char_indices() {
                if c.is_whitespace() {
                    if let Some(s) = start.take() {
                        spans.push(s..i);
                    }
                } else {
                    start.get_or_insert(i);
                }
            }
            if let Some(s) = start {
                spans.push(s..field.len());
            }
            for (si, span) in spans.into_iter().enumerate() {
                ranges.insert(
                    NoteAddress {
                        line: li + 1,
                        field: fi + 1,
                        subtoken: si + 1,
                    },
                    field_offset + span.start..field_offset + span.end,
                );
            }
            field_offset += field.len() + 1;
        }
        line_offset += line.len();
    }
    ranges
}
fn patch(text: &str, tokens: &[Token], replacements: &[String]) -> String {
    let mut edits: Vec<_> = tokens
        .iter()
        .zip(replacements)
        .filter(|(t, v)| t.original != **v)
        .collect();
    edits.sort_by_key(|(t, _)| t.range.start);
    let mut output = String::with_capacity(text.len());
    let mut offset = 0;
    for (token, value) in edits {
        output.push_str(&text[offset..token.range.start]);
        output.push_str(value);
        offset = token.range.end;
    }
    output.push_str(&text[offset..]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn change_list_classifies_each_marker_operation() {
        for (before, after, kind) in [
            ("4f#X", "4f#", DisplayChangeKind::StripX),
            ("4f#y", "4f#", DisplayChangeKind::StripY),
            ("4fn", "4f", DisplayChangeKind::StripN),
            ("4f#", "4f#X", DisplayChangeKind::AddX),
            ("4f#", "4f#y", DisplayChangeKind::AddY),
            ("4f", "4fnX", DisplayChangeKind::AddX),
        ] {
            assert_eq!(change_kinds(before, after), vec![kind]);
        }
    }
}
