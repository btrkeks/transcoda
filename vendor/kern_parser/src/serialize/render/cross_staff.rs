use crate::analysis::accidental_display::NoteAddress;
use std::collections::BTreeSet;

fn reorder_head(head: &str, lower: bool) -> String {
    let bytes = head.as_bytes();
    let Some(start) = bytes
        .iter()
        .position(|c| c.is_ascii_alphabetic() && matches!(c, b'A'..=b'G' | b'a'..=b'g'))
    else {
        return head.into();
    };
    let mut pitch_end = start + 1;
    while bytes.get(pitch_end) == bytes.get(start) {
        pitch_end += 1;
    }
    let accidental_start = pitch_end + usize::from(!lower);
    if !lower && !matches!(bytes.get(pitch_end), Some(b'<' | b'>')) {
        return head.into();
    }
    let mut end = accidental_start;
    while matches!(bytes.get(end), Some(b'#' | b'-' | b'n')) {
        end += 1;
    }
    if end == accidental_start || !matches!(bytes.get(end), Some(b'X' | b'y')) {
        return head.into();
    }
    let sign = if lower { end + 1 } else { pitch_end };
    if !matches!(bytes.get(sign), Some(b'<' | b'>')) || bytes.get(end + 1) == Some(&b'y') {
        return head.into();
    }
    let tail = if lower { sign + 1 } else { end + 1 };
    let accidental = &head[accidental_start..end + 1];
    if lower {
        format!(
            "{}{}{}{}",
            &head[..pitch_end],
            bytes[sign] as char,
            accidental,
            &head[tail..]
        )
    } else {
        format!(
            "{}{}{}{}",
            &head[..pitch_end],
            accidental,
            bytes[sign] as char,
            &head[tail..]
        )
    }
}

fn map_heads(text: &str, lower: bool, selected: Option<&BTreeSet<NoteAddress>>) -> String {
    let mut result = text
        .lines()
        .enumerate()
        .map(|(li, line)| {
            if line.starts_with(['*', '!', '=']) {
                return line.to_owned();
            }
            line.split('\t')
                .enumerate()
                .map(|(fi, field)| {
                    field
                        .split(' ')
                        .enumerate()
                        .map(|(si, head)| {
                            if selected.is_none_or(|s| {
                                s.contains(&NoteAddress {
                                    line: li + 1,
                                    field: fi + 1,
                                    subtoken: si + 1,
                                })
                            }) {
                                reorder_head(head, lower)
                            } else {
                                head.into()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect::<Vec<_>>()
                .join("\t")
        })
        .collect::<Vec<_>>()
        .join("\n");
    if text.ends_with('\n') {
        result.push('\n');
    }
    result
}

pub(super) fn lower_notes(text: &str, notes: impl Iterator<Item = NoteAddress>) -> String {
    map_heads(text, true, Some(&notes.collect()))
}

/// Decode only the render serializer's accidental/staff permutation.
/// This is a render-input boundary, not an extension of the source grammar.
pub(super) fn canonical_data(text: &str) -> Result<String, crate::document::DocumentError> {
    let candidate = map_heads(text, false, None);
    let document = crate::document::Document::parse(&candidate, Default::default())?;
    Ok(map_heads(text, false, Some(&note_addresses(&document))))
}

pub(super) fn lower_prepared(text: &str) -> Result<String, crate::document::DocumentError> {
    let document = crate::document::Document::parse(text, Default::default())?;
    Ok(map_heads(text, true, Some(&note_addresses(&document))))
}

fn note_addresses(document: &crate::document::Document) -> BTreeSet<NoteAddress> {
    let mut notes = BTreeSet::new();
    for (li, line) in document.score().lines().iter().enumerate() {
        let Some(line_number) = document.score().source(li) else {
            continue;
        };
        if let crate::KernLine::Data(fields) = line {
            for (fi, field) in fields.iter().enumerate() {
                if let crate::SpineContent::Kern(content) = field {
                    for head in content.noteheads() {
                        notes.insert(NoteAddress {
                            line: line_number,
                            field: fi + 1,
                            subtoken: head.chord_note_index.unwrap_or(0) + 1,
                        });
                    }
                }
            }
        }
    }
    notes
}

pub(super) fn lower_head(token: &str) -> String {
    reorder_head(token, true)
}
