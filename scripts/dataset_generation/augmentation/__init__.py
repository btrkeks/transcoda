"""Kern augmentation module for adding musical expression to transcriptions.

This module provides functions to add musical expression markings, dynamics,
articulations, and other annotative symbols to **kern transcriptions.

Example usage::

    from scripts.dataset_generation.augmentation import augment_transcription, AugmentationConfig

    # Use default probabilities
    augmented = augment_transcription(kern_string)

    # Customize probabilities
    config = AugmentationConfig(staccatos=0.8, tempo=0.0)
    augmented = augment_transcription(kern_string, config=config)
"""

from __future__ import annotations

from .articulations import apply_accents, apply_fermatas, apply_sforzandos, apply_staccatos
from .config import AugmentationConfig
from .courtesy_naturals import apply_courtesy_naturals
from .expression_markings import apply_expression_markings
from .hairpins import apply_render_hairpins
from .invisible_rests import apply_invisible_rests
from .instrument_label import apply_instrument_label_piano
from .pedaling import apply_pedaling
from .pipeline import augment_transcription
from .render_dynamic_marks import apply_render_dynamic_marks
from .tempo_markings import apply_tempo_markings, has_tempo_markings
from .xtuplet import apply_xtuplet

__all__ = [
    # Main entry point
    "augment_transcription",
    # Configuration
    "AugmentationConfig",
    # Individual augmentation functions
    "apply_staccatos",
    "apply_sforzandos",
    "apply_accents",
    "apply_fermatas",
    "apply_tempo_markings",
    "has_tempo_markings",
    "apply_expression_markings",
    "apply_render_hairpins",
    "apply_render_dynamic_marks",
    "apply_pedaling",
    "apply_instrument_label_piano",
    "apply_xtuplet",
    "apply_invisible_rests",
    "apply_courtesy_naturals",
]
