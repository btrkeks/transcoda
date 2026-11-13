//! The per-notehead rule behind explicit-accidental display.
//!
//! Scope limit: this forces printed glyphs for written accidentals and for
//! plain naturals on steps altered by the active key signature. It does not
//! force repeated naturals on key-unaltered steps after measure-local
//! accidental memory, such as the second plain G after a G-sharp in C major.

use crate::types::{Accidental, DiatonicStep, KeySignature, NoteModifiers};

/// The display change explicit accidentals make to one notehead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExplicitAccidental {
    /// Force the written accidental: `X` after it.
    ForceWritten,
    /// Force a natural on a plain step the key signature alters: `nX`.
    ForceNatural,
}

/// Tie continuations and ends are skipped: the tie start carries the glyph.
/// Notes drawn invisible are skipped: Verovio still draws a forced accidental on them.
pub(crate) fn explicit_accidental(
    step: DiatonicStep,
    accidental: Option<&Accidental>,
    modifiers: &NoteModifiers,
    drawn_invisible: bool,
    key_signature: Option<&KeySignature>,
) -> Option<ExplicitAccidental> {
    let continues_tie = modifiers.ties.iter().any(|tie| !tie.is_start());
    if drawn_invisible || continues_tie || modifiers.forced_accidental.is_some() {
        return None;
    }
    if accidental.is_some() {
        return (!modifiers.invisible_accidental).then_some(ExplicitAccidental::ForceWritten);
    }
    key_signature
        .and_then(|signature| signature.accidental_for_step(step))
        .map(|_| ExplicitAccidental::ForceNatural)
}
