"""Articulation augmentations: staccatos, sforzandos, fermatas."""

from __future__ import annotations

from .kern_utils import (
    apply_suffix_to_notes,
    find_note_tokens,
    sample_positions,
)

__all__ = [
    "apply_staccatos",
    "apply_sforzandos",
    "apply_accents",
    "apply_fermatas",
]

# Kern articulation symbols
STACCATO = "'"
SFORZANDO = "z"
ACCENT = "^"
FERMATA = ";"


def apply_staccatos(
    krn: str,
    per_note_probability: float = 0.08,
) -> str:
    """Add staccato markings to random notes.

    Staccato (') indicates short, detached notes.

    Args:
        krn: The kern string to modify.
        per_note_probability: Probability of adding staccato to each note.

    Returns:
        Modified kern string with staccato markings.
    """
    positions = find_note_tokens(krn)
    selected = sample_positions(positions, per_note_probability)
    return apply_suffix_to_notes(krn, selected, STACCATO)


def apply_sforzandos(
    krn: str,
    per_note_probability: float = 0.05,
) -> str:
    """Add sforzando accent markings to random notes.

    Sforzando (z) indicates a sudden strong accent.

    Args:
        krn: The kern string to modify.
        per_note_probability: Probability of adding sforzando to each note.

    Returns:
        Modified kern string with sforzando markings.
    """
    positions = find_note_tokens(krn)
    selected = sample_positions(positions, per_note_probability)
    return apply_suffix_to_notes(krn, selected, SFORZANDO)


def apply_accents(
    krn: str,
    per_note_probability: float = 0.03,
) -> str:
    """Add accent markings to random notes.

    Accent (^) indicates emphasized note attack.

    Args:
        krn: The kern string to modify.
        per_note_probability: Probability of adding accent to each note.

    Returns:
        Modified kern string with accent markings.
    """
    positions = find_note_tokens(krn)
    selected = sample_positions(positions, per_note_probability)
    return apply_suffix_to_notes(krn, selected, ACCENT)


def apply_fermatas(
    krn: str,
    per_note_probability: float = 0.02,
) -> str:
    """Add fermata markings to random notes.

    Fermata (;) indicates a held note, longer than its written duration.
    Applied sparingly as fermatas are relatively rare in music.

    Args:
        krn: The kern string to modify.
        per_note_probability: Probability of adding fermata to each note.

    Returns:
        Modified kern string with fermata markings.
    """
    positions = find_note_tokens(krn)
    selected = sample_positions(positions, per_note_probability)
    return apply_suffix_to_notes(krn, selected, FERMATA)
